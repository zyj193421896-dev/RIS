"""256QAM 3+8+128, 64-chip 25% header at 120 kHz. Never open a GUI."""
from pathlib import Path
import contextlib,io,sys,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_pwm import validate_pwm_header,acquire_pwm_header,pwm_header_layout
from rx706_pwm import pwm2_layout
from rx706_tdcm import recover_tdcm,validate_tdcm
from rx706_gui import parse_launch_args
from rx706_spectrum_gui import QAM256_120K_LEGACY_MODE
from test_qam64_controls import controls
from test_signal_analysis_view import Value
from qam64_fixtures import reference_points

FS=15360000.;RATE=120000.;SPS=128;TRAIN=(0,8,136,128)*2
PWM=dict(symbol_hz=RATE,total_chips=64,prefix_chips=16,guard_chips=48,symbols='0 0 0',qam_search_symbols=0)


def capture(frames=1,cfo=173.4):
    """Aggregate 64-slot harmonic model, independent Gray labels, not RF proof."""
    labels=np.random.default_rng(120064).integers(0,256,128)
    points=reference_points(np.r_[TRAIN,labels],256)
    # Each period has exactly 64 piecewise-constant slots, 2 ADC samples/slot.
    slots=2+.5j+points[:,None]*np.exp(2j*np.pi*(np.arange(64)+.5)/64)
    data=np.repeat(slots.reshape(-1),2)
    header=np.tile(np.r_[np.full(32,3+.5j),np.full(96,1+.5j)],3)
    x=np.r_[np.zeros(512),np.tile(np.r_[header,data],frames),np.zeros(64)]*700
    x*=np.exp(2j*np.pi*(800000+cfo)*np.arange(len(x))/FS)
    spec=SimpleNamespace(sample_rate_hz=FS,first_tuple=0,total_tuples=len(x),rate_source="synthetic")
    options=dict(symbol_hz=RATE,center_hint_hz=800000,order=256,
        tdcm_options=dict(base_hz=RATE,payload_symbols=136,payload_data_symbols=128),
        pwm_preamble_options=dict(PWM),preamble_options=dict(symbols='00 08 88 80 00 08 88 80',
            representation='hex',max_evm_percent=10,max_frequency_hz=1000))
    return spec,x,options,labels


class QAM256120kLegacyTests(unittest.TestCase):
    def setUp(self):
        self.no_tk=patch('tkinter.Tk',side_effect=AssertionError('No GUI allowed'));self.no_tk.start()
        self.no_window=patch('tkinter.Toplevel',side_effect=AssertionError('No GUI allowed'));self.no_window.start()
    def tearDown(self):self.no_window.stop();self.no_tk.stop()

    def test_fixed_known_header_and_two_samples_per_slot(self):
        self.assertEqual(pwm_header_layout(64,16,48),0)
        o=validate_pwm_header(PWM,FS);self.assertEqual(o['symbols'],[0,0,0])
        self.assertEqual(validate_tdcm(FS,RATE,dict(base_hz=RATE))['cycles_per_symbol'],1)
        # Standalone PWM v2 still needs an adjustable data region.
        with self.assertRaises(ValueError):pwm2_layout(64,16,48)
        for change in (dict(symbols='0 1 0'),dict(symbol_hz=120001),dict(prefix_chips=17)):
            with self.assertRaises(ValueError):validate_pwm_header(dict(PWM,**change),FS)

    def test_header_detection_signed_cfo_and_exact_boundary(self):
        for cfo in (-317.,173.4):
            spec,x,o,_=capture(cfo=cfo)
            base=x*np.exp(-2j*np.pi*800000*np.arange(len(x))/FS)
            sync=acquire_pwm_header(base,FS,PWM)
            self.assertAlmostEqual(sync['index'],512,delta=.5)
            self.assertAlmostEqual(sync['end_index'],512+384,delta=.5)
            self.assertAlmostEqual(sync['frequency_hz'],cfo,delta=1)
            self.assertLess(sync['error_percent'],.1)
        narrow=dict(PWM,prefix_chips=1,guard_chips=63)
        with self.assertRaisesRegex(ValueError,'高/低持续时间过短'):
            acquire_pwm_header(base,FS,narrow)

    def test_64_slot_qam_single_period_and_eight_training_removed(self):
        for cfo in (-317.,173.4):
            s,x,o,labels=capture(cfo=cfo)
            with patch('rx706_tdcm._read_span',return_value=(x.copy(),len(x),0)):
                r=recover_tdcm(s,**o)
            np.testing.assert_array_equal(r.labels,labels)
            self.assertEqual(r.bits.shape,(128,8));self.assertEqual(r.preamble['length'],8)
            self.assertEqual(r.tdcm['cycles_per_symbol'],1)
            self.assertEqual(r.preamble['expected_hex'],['00','08','88','80']*2)
            self.assertAlmostEqual(np.diff(r.time_s).mean(),1/RATE,places=12)
            self.assertLess(r.evm_percent,.1)

    def test_multi_frame_compensation_preserves_139_cycles_and_128_payload(self):
        s,x,o,labels=capture(frames=20)
        o['tdcm_options'].update(payload_symbols=0,compensation=dict(enabled=True,frame_mode='all',
            frame_symbols=136,clock_mode='repeat',clock_search_ppm=1000,phase_mode='square',fit_timing=False))
        with patch('rx706_tdcm_debug._read_span',return_value=(x.copy(),len(x),0)):
            r=recover_tdcm(s,**o)
        valid=sum(row['valid'] for row in r.frames)
        self.assertGreaterEqual(valid,18);self.assertEqual(len(r.labels),128*valid)
        self.assertAlmostEqual(r.diagnostics['frame_period_samples']/SPS,139,delta=.01)
        self.assertEqual(len(r.training_records),8*len(r.frames))
        for frame in np.unique(r.frame_ids)[1:-1]:np.testing.assert_array_equal(r.labels[r.frame_ids==frame],labels)
        self.assertIsNone(r.diagnostics['ber'])

    def test_preset_and_analysis_routing_without_tk(self):
        v=controls();v.pwm2_layout_summary=Value('');v.tdcm_carrier_mhz.set('0.42');v.rate.set('15.36')
        self.assertTrue(v.apply_qam256_120k_legacy_frame_profile())
        self.assertEqual(v.constellation_mode.get(),QAM256_120K_LEGACY_MODE)
        self.assertEqual(v.qam_preamble_symbols.get(),'00 08 88 80 00 08 88 80')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'136')
        self.assertEqual(v.pwm2_guard_chips.get(),'48');self.assertIn('每片2.000',v.pwm2_layout_summary.get())
        self.assertNotIn('不足',v.pwm2_layout_summary.get());self.assertEqual(v.tdcm_carrier_mhz.get(),'0.42')
        with patch('rx706_spectrum_gui.threading.Thread') as worker:
            v.analyze();self.assertTrue(worker.called,v.status.get());o=worker.call_args.kwargs['args'][2]
        self.assertEqual(o['order'],256);self.assertEqual(o['symbol_hz'],RATE)
        self.assertEqual(o['tdcm_options']['base_hz'],RATE)
        self.assertEqual(o['tdcm_options']['compensation']['frame_symbols'],136)
        self.assertEqual(o['tdcm_options']['payload_repeat'],1)
        for k,val in PWM.items():self.assertEqual(o['pwm_preamble_options'][k],val)
        v.apply_qam256_120k_legacy_frame_profile();self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'136')

    def test_selecting_named_mode_does_not_restore_old_25_chip_header(self):
        v=controls();v.pwm2_layout_summary=Value('');v.filename.set('')
        v.constellation_mode.set(QAM256_120K_LEGACY_MODE)
        with patch.object(v,'analyze') as analyze:
            v.on_mode_changed();analyze.assert_not_called()
            v.filename.set('/tmp/example-iq.bin');v.on_mode_changed();analyze.assert_called_once()
        self.assertEqual(v.pwm2_total_chips.get(),'64');self.assertEqual(v.pwm2_prefix_chips.get(),'16')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'136')

    def test_new_cli_is_explicit_and_old_defaults_are_unchanged(self):
        a=parse_launch_args(['--qam256-120k-legacy-frame']);self.assertEqual(a.qam_order,256);self.assertEqual(a.payload_symbols,128)
        a=parse_launch_args([]);self.assertFalse(a.qam256_120k_legacy_frame);self.assertEqual(a.payload_symbols,256)
        for extras in (['--qam-order','64'],['--payload-symbols','256'],['--tail-symbols','1'],['--paired-64-frame']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_launch_args(['--qam256-120k-legacy-frame']+extras)

if __name__=='__main__':unittest.main()

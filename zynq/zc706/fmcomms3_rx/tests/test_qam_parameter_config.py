"""Portable parameter imports preserve capture choices; no Tk or hardware."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import patch,Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_config import METHOD,read_parameter_config,validate_parameter_config
from rx706_qam import recover_qam
from test_qam64_controls import controls
from test_signal_analysis_view import Value
from test_qam256_120k import capture,FS
from qam64_fixtures import save_capture

CONFIG=Path(__file__).resolve().parents[5]/'数据/rx_256qam_120k_64slice_3_4_16_no_crc_config.json'


class ParameterConfigTests(unittest.TestCase):
    def setUp(self):
        self.no_tk=patch('tkinter.Tk',side_effect=AssertionError('No GUI'));self.no_tk.start()
        self.no_window=patch('tkinter.Toplevel',side_effect=AssertionError('No GUI'));self.no_window.start()
    def tearDown(self):self.no_window.stop();self.no_tk.stop()
    def view(self):
        v=controls();v.pwm2_layout_summary=Value('');return v

    def test_delivered_json_is_portable_valid_and_idempotent(self):
        raw=json.loads(CONFIG.read_text());cfg=read_parameter_config(CONFIG)
        self.assertEqual(cfg,validate_parameter_config(cfg));self.assertEqual(cfg['method'],METHOD)
        for k in ('input_fingerprint','input_path','sample_rate_hz','center_hint_hz','channel','first_tuple'):
            self.assertNotIn(k,raw)
        self.assertEqual(cfg['symbol_hz'],120000);self.assertEqual(cfg['tdcm_options']['compensation']['frame_symbols'],20)
        self.assertEqual(cfg['tdcm_options']['payload_data_symbols'],16)
        self.assertFalse(cfg['tdcm_options']['crc_debug']['enabled']);self.assertEqual(cfg['tdcm_options']['payload_tail_symbols'],0)

    def test_apply_preserves_capture_choices_and_clears_old_replay(self):
        v=self.view();v.filename.set('/tmp/unrelated-capture.bin');v.rate.set('30.72')
        v.tdcm_carrier_mhz.set('-0.73');v.channel.set('RX2');v.first_tuple.set('321')
        v.recovery_ms.set('250');v._next_qam_recipe={'old':'binding'}
        before=[x.get() for x in (v.filename,v.rate,v.tdcm_carrier_mhz,v.channel,v.first_tuple,v.recovery_ms)]
        self.assertTrue(v.apply_qam_parameter_config(read_parameter_config(CONFIG)))
        self.assertEqual(before,[x.get() for x in (v.filename,v.rate,v.tdcm_carrier_mhz,v.channel,v.first_tuple,v.recovery_ms)])
        self.assertIsNone(v._next_qam_recipe);self.assertEqual(v.pwm2_total_chips.get(),'64')
        self.assertEqual(v.qam_preamble_symbols.get(),'00 08 88 80');self.assertEqual(v.four_corner_payload.get(),'16')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'20')
        self.assertEqual(v.qam_compensation.clock_mode.get(),'关闭（配置周期）')

    def test_existing_dialog_loads_parameters_without_analysis(self):
        v=self.view();v.analyze=Mock();before=v.filename.get()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(CONFIG)),patch('rx706_spectrum_gui.messagebox.showerror') as error:
            v.load_qam_recipe();error.assert_not_called()
        v.analyze.assert_not_called();self.assertEqual(v.filename.get(),before)
        self.assertIn('没有自动分析',v.status.get());self.assertEqual(v.qam_order.get(),'256')

    def test_invalid_parameters_fail_before_controls_change(self):
        for edit in (lambda c:c.update(input_fingerprint={}),lambda c:c.update(center_hint_hz=800000),
                     lambda c:c.update(symbol_hz=float('nan')),
                     lambda c:c['tdcm_options']['compensation'].update(frame_symbols=136),
                     lambda c:c['tdcm_options'].update(replay_recipe={})):
            cfg=json.loads(CONFIG.read_text());edit(cfg);v=self.view();before=v.qam_order.get(),v.pwm2_total_chips.get()
            with self.assertRaises(ValueError):v.apply_qam_parameter_config(cfg)
            self.assertEqual(before,(v.qam_order.get(),v.pwm2_total_chips.get()))

    def test_same_json_runs_on_two_different_synthetic_iq_files(self):
        cfg=read_parameter_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            for index,cfo in enumerate((173.4,-317.)):
                _,x,_,expected=capture(frames=4,cfo=cfo)
                spectrum=save_capture(directory,f'capture_{index}.bin',x*.5,FS)
                v=self.view();v.filename.set(str(spectrum.path));v.tdcm_carrier_mhz.set('0.8')
                v.apply_qam_parameter_config(cfg)
                with patch('rx706_spectrum_gui.threading.Thread') as worker:
                    v.analyze();self.assertTrue(worker.called,v.status.get());o=dict(worker.call_args.kwargs['args'][2]);o.pop('kind')
                with patch('rx706_tdcm_debug._clock_repeat',side_effect=AssertionError('No repeat assumption')):
                    r=recover_qam(spectrum,**o)
                self.assertEqual(len(r.labels),64)
                for frame in np.unique(r.frame_ids):np.testing.assert_array_equal(r.labels[r.frame_ids==frame],expected[int(frame)])
                self.assertEqual(len(r.training_records),16);self.assertFalse(hasattr(r,'payload_debug'))

    def test_legacy_frozen_dialog_still_schedules_replay(self):
        v=self.view();v.analyze=Mock()
        old=dict(version=1,method='bounded_repeated_tdcm',order=256,symbol_hz=19531.25,
            input_fingerprint=dict(sample_rate_hz=15360000,channel=1,first_tuple=0))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.json';path.write_text(json.dumps(old))
            with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(path)),patch('rx706_spectrum_gui.messagebox.showerror') as error:
                v.load_qam_recipe();error.assert_not_called()
        v.analyze.assert_called_once();self.assertEqual(v._next_qam_recipe,old)

if __name__=='__main__':unittest.main()

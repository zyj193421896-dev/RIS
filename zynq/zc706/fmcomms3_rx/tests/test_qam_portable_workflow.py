"""Portable workflow tests with Tcl variables, no display or radio required."""
import copy
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_workflows import ReceiverWorkflows, AREAS
from rx706_qam_config import validate_parameter_config


def portable_config():
    return dict(version=1, method='tdcm_parameter_config', order=256, symbol_hz=20000,
        tdcm_options=dict(acquisition_mode='pwm', base_hz=20000, harmonic=1,
            normalization='blind', payload_symbols=0, payload_data_symbols=256,
            payload_repeat=1, payload_tail_symbols=0, crc_debug=dict(enabled=False),
            compensation=dict(enabled=True, clock_mode='off', phase_mode='off',
                frame_mode='all', frame_symbols=260)),
        pwm_preamble_options=dict(symbols='0 1 2 3 4 5 6 7 8 9', total_chips=20, prefix_chips=1,
            guard_chips=19, symbol_hz=20000,
            pulse_chips=[[10,5],[0,13],[6,8],[0,7],[4,9],[0,15],[2,5],[0,11],[8,5],[3,10]]),
        preamble_options=dict(symbols='00 08 88 80', representation='hex',
            orientation='auto', fit_dc_offset=True),
        transmitter=dict(qam_slices=256, qam_cycles_per_symbol=1,
            preamble_timing='independent_exact_percent',
            preamble_mt_percent=[[25,50],[65,0],[40,30],[35,0],[45,20],[75,0],[25,10],[55,0],[25,40],[50,15]],
            training_hex=['00','08','88','80'], payload_symbols=256, frame_symbols=270))


class PortableWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'config.json'
        self.path.write_text(json.dumps(portable_config()))
        self.bin = Path(self.temp.name)/'capture.bin'
        self.bin.write_bytes(b'\0'*4096)
        self.tcl = tk.Tcl()
        v = SimpleNamespace(busy=False, closed=False, window=None,
            pwm_frame_users=[], pwm_cycle_calibration=[], pwm_nearest_decision=False,
            _fixed_source_sha256=None, _receiver_changed=Mock(), _invalidate_analysis=Mock(),
            refresh_parameter_visibility=Mock(), _buttons=Mock(), _analyze_worker=Mock(),
            analyze=Mock(), cancel=threading.Event(), events=queue.Queue(), _autotune_token=0)
        for key, value in dict(filename=str(self.bin), channel='RX2', rate='30.72',
            first_tuple='12', nfft='1024', averages='2', recovery_ms='1000',
            tdcm_carrier_mhz='0.8', tdcm_symbol_khz='120', qam_order='256',
            pwm2_input_mode='时间片协议', constellation_mode='TDCM 谐波QAM', status='').items():
            setattr(v, key, tk.StringVar(self.tcl, value=value))
        v.qam_auto_acquisition = tk.BooleanVar(self.tcl, value=True)
        v.remove_dc = tk.BooleanVar(self.tcl, value=True)
        w = ReceiverWorkflows.__new__(ReceiverWorkflows)
        w.view=v; w.area=AREAS[1]; w.states={}; w.replay=None; w.portable=None
        w.confirmed=None; w.loading=False; w.refresh=Mock()
        w.area_var=tk.StringVar(self.tcl, value=w.area)
        w.defaults=w.snapshot()
        self.w=w; self.v=v

    def test_load_preserves_capture_and_never_rewrites_file(self):
        before=self.w.snapshot()['variables']; raw=self.path.read_bytes()
        self.assertTrue(self.w.load(self.path))
        before['tdcm_symbol_khz']='20'  # Loaded protocol rates now mirror the authoritative JSON.
        self.assertEqual(before, self.w.snapshot()['variables'])
        self.assertEqual(raw, self.path.read_bytes())
        self.assertEqual(self.w.portable, validate_parameter_config(portable_config()))
        self.assertEqual(self.w.confirmed,self.w.signature())
        self.v.analyze.assert_not_called()

    def test_delivered_main_json_loads_complete(self):
        path=Path(__file__).resolve().parents[1]/'host/frame_profiles/acceptance/rx_256qam_256slice_20khz_10_4_256_config.json'
        raw=path.read_bytes(); before=self.w.snapshot()['variables']
        self.assertTrue(self.w.load(path))
        before['tdcm_symbol_khz']='20'
        self.assertEqual(before,self.w.snapshot()['variables'])
        self.assertEqual(raw,path.read_bytes())
        self.assertEqual(self.w.portable['transmitter']['qam_slices'],256)
        self.assertEqual(len(self.w.portable['pwm_preamble_options']['pulse_chips']),10)
        self.assertEqual(self.w.portable['tdcm_options']['compensation']['frame_symbols'],260)

    def test_run_routes_full_options_with_current_capture(self):
        self.w.load(self.path)
        config=copy.deepcopy(self.w.portable)
        self.assertTrue(self.w.run())
        self.v.thread.join(5)
        name, options, recovery=self.v._analyze_worker.call_args.args
        self.assertEqual(name,str(self.bin)); self.assertEqual(options['channel'],2)
        self.assertEqual(options['sample_rate_hz'],30720000)
        self.assertEqual(options['first_tuple'],12)
        self.assertEqual(recovery['center_hint_hz'],800000)
        self.assertTrue(recovery['auto_acquisition'])
        for key in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options'):
            self.assertEqual(recovery[key],config[key])
        self.assertEqual(self.w.portable,config)
        self.v.analyze.assert_not_called()

    def test_invalid_load_keeps_valid_config_and_capture(self):
        self.w.load(self.path); before=self.w.snapshot()
        self.path.write_text(json.dumps(dict(portable_config(),symbol_hz=-1)))
        with patch('rx706_workflows.messagebox.showerror') as error:
            self.assertFalse(self.w.load(self.path)); error.assert_called_once()
        self.assertEqual(before,self.w.snapshot())

    def test_existing_worker_calls_recover_qam_without_downcasting(self):
        self.v.qam_auto_acquisition.set(False)
        from rx706_spectrum_gui import SpectrumWindow
        self.w.load(self.path)
        self.v._analyze_worker=SpectrumWindow._analyze_worker.__get__(self.v)
        spectrum, recovered=object(), object()
        with patch('rx706_spectrum_gui.analyze_bin',return_value=spectrum), \
                patch('rx706_spectrum_gui.display_trace',return_value='trace'), \
                patch('rx706_qam.recover_qam',return_value=recovered) as recover:
            self.assertTrue(self.w.run()); self.v.thread.join(5)
        self.assertIs(recover.call_args.args[0],spectrum)
        self.assertEqual(recover.call_args.kwargs['pwm_preamble_options'],self.w.portable['pwm_preamble_options'])
        self.assertEqual(recover.call_args.kwargs['tdcm_options'],self.w.portable['tdcm_options'])
        self.assertEqual(self.v.events.get_nowait(),('result',(spectrum,'trace',recovered,None)))

    def test_auto_worker_routes_hint_and_cancellation_without_payload_reference(self):
        from rx706_spectrum_gui import SpectrumWindow
        self.w.load(self.path)
        self.v._analyze_worker=SpectrumWindow._analyze_worker.__get__(self.v)
        spectrum, recovered=object(), object()
        automatic=Mock(return_value=recovered)
        module=SimpleNamespace(recover_auto_qam=automatic)
        with patch.dict(sys.modules, {'rx706_qam_acquisition': module}), \
                patch('rx706_spectrum_gui.analyze_bin',return_value=spectrum), \
                patch('rx706_spectrum_gui.display_trace',return_value='trace'), \
                patch('rx706_qam.recover_qam') as manual:
            self.assertTrue(self.w.run()); self.v.thread.join(5)
        manual.assert_not_called()
        self.assertIs(automatic.call_args.args[0], spectrum)
        kwargs=automatic.call_args.kwargs
        self.assertIs(kwargs['cancel_event'], self.v.cancel)
        self.assertEqual(kwargs['center_hint_hz'],800000)
        self.assertNotIn('auto_acquisition',kwargs)
        self.assertNotIn('reference_options',kwargs)
        self.assertEqual(kwargs['tdcm_options'],self.w.portable['tdcm_options'])
        self.assertEqual(self.v.events.get_nowait(),('result',(spectrum,'trace',recovered,None)))

    def test_uncertain_phase_reaches_gui_as_error_without_fake_result(self):
        from rx706_spectrum_gui import SpectrumWindow
        self.w.load(self.path)
        self.v._analyze_worker=SpectrumWindow._analyze_worker.__get__(self.v)
        spectrum=object();reason='已知前导频偏不确定性无法排除另一相位周数'
        with patch('rx706_spectrum_gui.analyze_bin',return_value=spectrum), \
             patch('rx706_spectrum_gui.display_trace',return_value='trace'), \
             patch('rx706_qam_acquisition.recover_auto_qam',side_effect=ValueError(reason)):
            self.assertTrue(self.w.run());self.v.thread.join(5)
        self.assertEqual(self.v.events.get_nowait(),('result',(spectrum,'trace',None,reason)))

    def test_auto_accepts_missing_bad_and_nonfinite_carrier_hints(self):
        self.w.load(self.path)
        for hint in ('', 'not-a-frequency', 'nan', 'inf'):
            with self.subTest(hint=hint):
                self.v.busy=False
                self.v.tdcm_carrier_mhz.set(hint)
                self.assertTrue(self.w.confirm())
                self.assertTrue(self.w.run()); self.v.thread.join(5)
                self.assertIsNone(self.v._analyze_worker.call_args.args[2]['center_hint_hz'])
        self.v.busy=False
        self.v.qam_auto_acquisition.set(False)
        self.assertTrue(self.w.confirm())
        self.assertFalse(self.w.run())

    def test_auto_choice_invalidates_confirmation_and_is_area_local(self):
        self.w.load(self.path)
        self.v.qam_auto_acquisition.set(False)
        self.assertFalse(self.w.run())
        self.assertTrue(self.w.confirm())
        before=self.w.snapshot()
        self.assertTrue(self.w.switch(AREAS[0]))
        self.assertTrue(self.v.qam_auto_acquisition.get())
        self.assertTrue(self.w.switch(AREAS[1]))
        self.assertEqual(self.w.snapshot(),before)
        self.assertFalse(self.v.qam_auto_acquisition.get())

    def test_measured_auto_result_visible_without_ber_claim(self):
        self.w.metrics=tk.StringVar(self.tcl)
        self.v.recovery=SimpleNamespace(kind='qam', order=256, scatter=[0], evm_percent=2,
            diagnostics={'auto_acquisition': {'carrier_hz':797021,'period_ppm':45.573}})
        self.w.results()
        self.assertIn('+797.021000 kHz',self.w.metrics.get())
        self.assertIn('+45.573 ppm',self.w.metrics.get())
        self.assertIn('BER 未测',self.w.metrics.get())

    def test_save_roundtrip_and_edit_is_read_only(self):
        self.w.load(self.path); output=self.path.with_name('saved.json')
        with patch('rx706_workflows.filedialog.asksaveasfilename',return_value=str(output)):
            self.w.save()
        self.assertEqual(json.loads(output.read_text()),self.w.portable)
        with patch.object(self.w,'show_replay') as show:
            self.w.edit(); show.assert_called_once()

    def test_switch_isolates_full_config(self):
        self.w.load(self.path); before=self.w.snapshot()
        self.assertTrue(self.w.switch(AREAS[0])); self.assertIsNone(self.w.portable)
        self.v.filename.set('different.bin')
        self.assertTrue(self.w.switch(AREAS[1]))
        self.assertEqual(self.w.snapshot(),before)

    def test_advanced_parameter_cannot_replace_json(self):
        self.w.load(self.path); before=copy.deepcopy(self.w.portable)
        self.v.tdcm_symbol_khz.set('500')
        self.assertFalse(self.w.run()); self.assertFalse(self.w.expert_run())
        self.assertTrue(self.w.confirm()); self.assertEqual(self.w.portable,before)
        self.assertTrue(self.w.run()); self.v.thread.join(5)
        self.assertEqual(self.v._analyze_worker.call_args.args[2]['symbol_hz'],20000)


class PortableGuiIntegrationTests(unittest.TestCase):
    def test_auto_control_visibility_busy_and_stale_result_guard(self):
        from rx706_spectrum_gui import SpectrumWindow
        root=tk.Tk(); root.withdraw()
        with patch.object(SpectrumWindow, 'maximize'):
            view=SpectrumWindow(root)
        view.window.withdraw()
        try:
            flow=view.workflows
            toggle=flow.auto_acquisition_toggle
            self.assertTrue(view.qam_auto_acquisition.get())
            self.assertFalse(toggle.winfo_manager())
            flow.switch(AREAS[1])
            self.assertEqual(toggle.winfo_manager(),'pack')  # New 10 kHz default is portable/automatic.
            self.assertEqual(flow.portable['symbol_hz'],10000)
            flow.portable=validate_parameter_config(portable_config())
            flow.refresh()
            self.assertEqual(toggle.winfo_manager(),'pack')
            view.busy=True; view._buttons()
            self.assertIn('disabled',toggle.state())
            view.busy=False; view._buttons()
            token=view._autotune_token
            view.result=object(); view.recovery=object()
            view.qam_auto_acquisition.set(False)
            self.assertGreater(view._autotune_token,token)
            self.assertIsNone(view.result)
            self.assertIsNone(view.recovery)
            flow.portable=None; flow.replay={'method':'iq_bound_tdcm_replay'}
            flow.refresh()
            self.assertFalse(toggle.winfo_manager())
        finally:
            view.close(); root.destroy()

    def test_synthetic_qam10_full_gui_receive(self):
        self._synthetic_qam10_full_gui_receive(automatic=False)

    def test_synthetic_qam10_auto_gui_receive_with_wrong_hint(self):
        self._synthetic_qam10_full_gui_receive(automatic=True)

    def test_synthetic_weak_qam_uses_default_auto_phase_fallback(self):
        self._synthetic_qam10_full_gui_receive(automatic=True,weak=True)

    def _synthetic_qam10_full_gui_receive(self, automatic,weak=False):
        import numpy as np
        from rx706_spectrum_gui import SpectrumWindow
        from test_qam10_frame import capture
        root=tk.Tk(); root.withdraw()
        with patch.object(SpectrumWindow, 'maximize'):
            view=SpectrumWindow(root)
        view.window.withdraw()
        try:
            with tempfile.TemporaryDirectory() as directory:
                frames = 40 if weak else (8 if automatic else 5)
                _, config, truth, _=capture(directory,frames=frames)
                if weak:
                    rawpath=Path(directory)/'qam10_synthetic.bin'
                    raw=np.fromfile(rawpath,dtype='<i2').reshape(-1,4)
                    t=np.arange(len(raw))/15360000.;rng=np.random.default_rng(972)
                    signal=.25*(raw[:,0]+1j*raw[:,1])*np.exp(2j*np.pi*(3*t+8*t*t))
                    signal+=112.5*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
                    raw[:,0]=np.rint(signal.real);raw[:,1]=np.rint(signal.imag);raw.tofile(rawpath)
                filename=Path(directory)/'config.json'
                filename.write_text(json.dumps(config))
                flow=view.workflows
                self.assertTrue(flow.switch(AREAS[1]))
                view.filename.set(str(Path(directory)/'qam10_synthetic.bin'))
                view.rate.set('15.36'); view.channel.set('RX1'); view.first_tuple.set('0')
                hint = '-0.8' if automatic else '0.12'
                view.tdcm_carrier_mhz.set(hint); view.recovery_ms.set('1000')
                view.nfft.set('1024'); view.averages.set('1')
                self.assertTrue(flow.load(filename))
                self.assertTrue(view.qam_auto_acquisition.get())
                self.assertEqual(flow.auto_acquisition_toggle.winfo_manager(), 'pack')
                view.qam_auto_acquisition.set(automatic)
                self.assertTrue(flow.confirm())
                view.show_advanced.set(True); view.refresh_parameter_visibility()
                carrier_controls=[widget for box, _ in view._parameter_groups
                    for widget in box.winfo_children()
                    if 'textvariable' in widget.keys()
                    if str(widget.cget('textvariable')) == str(view.tdcm_carrier_mhz)]
                self.assertTrue(carrier_controls)
                self.assertTrue(carrier_controls[0].winfo_manager())
                self.assertTrue(carrier_controls[0].master.winfo_manager())
                self.assertFalse(flow.expert_button.winfo_manager())
                self.assertTrue(flow.run())
                view.thread.join(90)
                self.assertFalse(view.thread.is_alive())
                view.poll()
                self.assertIsNotNone(view.recovery,view.status.get())
                self.assertFalse(view.busy)
                if weak:
                    self.assertGreater(len(view.recovery.labels),30*256)
                    self.assertEqual(view.recovery.diagnostics['phase_tracking']['source'],
                        'known_pwm_joint_fit_with_independent_qam_check')
                else:
                    self.assertEqual(len(view.recovery.labels),frames*256)
                    self.assertEqual(len(view.recovery.training_records),frames*4)
                    np.testing.assert_array_equal(view.recovery.labels,truth.ravel())
                self.assertIsNone(view.recovery.diagnostics['ber'])
                self.assertEqual(view.tdcm_carrier_mhz.get(),hint)
                if automatic:
                    self.assertIn('auto_acquisition', view.recovery.diagnostics)
                    self.assertIn('前导实测周期偏差', view.summary.get())
                    self.assertIn('原载波', flow.metrics.get())
                if not weak:self.assertLess(view.recovery.evm_percent,3)
        finally:
            view.close(); root.destroy()


if __name__ == '__main__':
    unittest.main()

"""Real control methods with Value/Agg; creating a Tk window is forbidden."""
from pathlib import Path
import sys,ast,inspect,textwrap,unittest,io,contextlib,json
from unittest.mock import Mock,patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_signal_analysis_view import headless_window,Value
from rx706_spectrum_gui import SpectrumWindow,QAM_MODE,QAM_PWM_MODE,TDCM_MODE,FOUR_CORNER_MODE
from rx706_qam_debug_gui import CompensationControls,CLOCK_MODES,PHASE_MODES,FRAME_MODES,ACQUISITION_MODES
from rx706_tdcm_debug import validate_compensation
from rx706_tdcm_auto import validate_search
from rx706_gui import parse_launch_args


def controls():
    v=headless_window();v.busy=False
    tree=ast.parse(textwrap.dedent(inspect.getsource(SpectrumWindow.__init__)))
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call):continue
        fun=node.value.func
        if not isinstance(fun,ast.Attribute) or fun.attr not in ('StringVar','BooleanVar'):continue
        name=node.targets[0].attr
        try:value=ast.literal_eval(next(k.value for k in node.value.keywords if k.arg=='value'))
        except (ValueError,StopIteration):continue
        setattr(v,name,Value(value))
    v.filename=Value('/tmp/example-iq.bin');v._next_qam_recipe=None;v._last_qam_order=256
    p=CompensationControls.__new__(CompensationControls);p.widgets=[]
    p.enabled=Value(False);p.search_enabled=Value(False);p.auto_open=Value(False)
    p.acquisition_mode=Value('有PWM前导（原流程）');p.clock_mode=Value(next(iter(CLOCK_MODES)))
    p.phase_mode=Value(next(iter(PHASE_MODES)));p.frame_mode=Value(next(iter(FRAME_MODES)));p.fit_timing=Value(False)
    p.values={k:Value(str(x)) for k,x in validate_compensation({}).items() if k not in ('enabled','clock_mode','phase_mode','frame_mode','fit_timing')}
    defaults=validate_search({});p.search_values={k:Value(str(defaults[k])) for k in ('clock_ppm','max_carriers','max_trials','max_seconds','minimum_coverage','minimum_geometry')}
    p.search_values.update(carrier_span_khz=Value('500'),frame_candidates=Value('64 128 256 512'))
    p.calibration_variables=dict(normalization=v.tdcm_normalization);p.search_callback=v.activate_qam_auto_search
    v.qam_compensation=p;v._buttons=Mock()
    return v


class QAM64ControlTests(unittest.TestCase):
    def setUp(self):
        self.no_tk=patch('tkinter.Tk',side_effect=AssertionError('No GUI allowed'));self.no_tk.start()
        self.no_window=patch('tkinter.Toplevel',side_effect=AssertionError('No GUI allowed'));self.no_window.start()
    def tearDown(self):self.no_window.stop();self.no_tk.stop()

    def test_four_corner_profiles_keep_order_and_use_explicit_payload(self):
        v=controls();v.qam_order.set('64')
        self.assertFalse(v.apply_four_corner_training_profile());self.assertIn('实际每帧净荷数',v.status.get())
        v.four_corner_payload.set('12');self.assertTrue(v.apply_four_corner_training_profile())
        self.assertEqual(v.qam_order.get(),'64');self.assertEqual(v.qam_preamble_symbols.get(),'00 04 24 20')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'16')
        self.assertEqual(v.qam_compensation.search_values['frame_candidates'].get(),'16')
        v.apply_four_corner_training_profile();self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'16')
        # Legacy 256 profile still infers the already configured net window count.
        v=controls();v.qam_compensation.values['frame_symbols'].set('12');v.apply_four_corner_training_profile()
        self.assertEqual(v.qam_preamble_symbols.get(),'00 08 88 80');self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'16')

    def test_mode_matrix_delivers_64_and_does_not_require_pwm_when_headerless(self):
        for mode in (QAM_MODE,QAM_PWM_MODE,TDCM_MODE,FOUR_CORNER_MODE):
            with self.subTest(mode=mode):
                v=controls();v.qam_order.set('64');v.constellation_mode.set(mode)
                if mode==FOUR_CORNER_MODE:v.four_corner_payload.set('12');v.apply_four_corner_training_profile()
                with patch('rx706_spectrum_gui.threading.Thread') as worker:
                    v.analyze();self.assertTrue(worker.called,v.status.get());o=worker.call_args.kwargs['args'][2]
                self.assertEqual(o['order'],64)
                if mode in (TDCM_MODE,FOUR_CORNER_MODE):self.assertIn('tdcm_options',o)
                if mode==FOUR_CORNER_MODE:
                    self.assertEqual(o['preamble_options']['symbols'],'00 04 24 20')
                    self.assertEqual(o['tdcm_options']['compensation']['frame_symbols'],16)
        for mode in ('无前导：重复图样','无前导：已知固定时钟/随机载荷'):
            v=controls();v.qam_order.set('64');v.qam_compensation.acquisition_mode.set(mode);v.qam_compensation._acquisition_changed()
            v.qam_pwm_symbols.set('not a header')
            with patch('rx706_spectrum_gui.threading.Thread') as worker:
                v.analyze();o=worker.call_args.kwargs['args'][2]
            self.assertEqual(o['order'],64);self.assertNotIn('pwm_preamble_options',o)

    def test_repeated_profile_preserves_64_rates_carrier_and_real_window_size(self):
        v=controls();v.qam_order.set('64');v.tdcm_symbol_khz.set('160');v.tdcm_base_khz.set('480')
        v.tdcm_carrier_mhz.set('1.2');v.qam_compensation.values['frame_symbols'].set('76')
        v.apply_qam_debug_profile(v.qam_compensation)
        self.assertEqual(v.qam_order.get(),'64');self.assertEqual(v.tdcm_base_khz.get(),'480')
        self.assertEqual(v.tdcm_symbol_khz.get(),'160');self.assertEqual(v.tdcm_carrier_mhz.get(),'1.2')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'76')

    def test_order_switch_only_translates_generated_training_and_clears_stale_result(self):
        v=controls();v.qam_compensation.values['frame_symbols'].set('12');v.apply_four_corner_training_profile()
        v.qam_order.set('64');v.recovery=SimpleNamespace(order=256);v.draw_constellation=Mock();v.canvas=Mock()
        v.on_qam_order_changed();self.assertIsNone(v.recovery);self.assertEqual(v.qam_preamble_symbols.get(),'00 04 24 20')
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'16')
        v.qam_preamble_symbols.set('00 01 02');v.qam_order.set('256');v.on_qam_order_changed()
        self.assertEqual(v.qam_preamble_symbols.get(),'00 01 02')

    def test_recipe_display_recognizes_64_corners_and_preserves_frame(self):
        v=controls();v.constellation_mode.set(TDCM_MODE)
        recipe=dict(method='bounded_repeated_tdcm',order=64,symbol_hz=160000,tdcm_options=dict(base_hz=160000,
            compensation=dict(frame_symbols=68)),preamble_options=dict(symbols='00 04 24 20',representation='hex'))
        v.apply_qam_recipe_parameters(recipe)
        self.assertEqual(v.qam_order.get(),'64');self.assertEqual(v.constellation_mode.get(),FOUR_CORNER_MODE)
        self.assertEqual(v.four_corner_payload.get(),'64');self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'68')

    def test_cli_is_testable_without_tk_and_64_requires_actual_payload(self):
        self.assertEqual(parse_launch_args([]).qam_order,256)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_launch_args(['--four-corner-training','--qam-order','64'])
        a=parse_launch_args(['--four-corner-training','--qam-order','64','--payload-symbols','12'])
        self.assertEqual(a.qam_order,64);self.assertEqual(a.payload_symbols,12)

if __name__=='__main__':unittest.main()

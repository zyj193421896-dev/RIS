"""Hidden Tk tests for headerless controls, worker options, report and failures."""
from pathlib import Path
import sys,tempfile,tkinter as tk,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_tdcm_headerless import setup_capture
from test_qam_debug_gui import HiddenToplevel
from rx706_spectrum_gui import SpectrumWindow,TDCM_MODE
from rx706_tdcm import recover_tdcm
from rx706_qam_debug_gui import export_recipe_json

class HeaderlessGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.spec,cls.options,_,_=setup_capture(cls.temp.name,auto=False,frames=16)
        cls.result=recover_tdcm(cls.spec,**cls.options)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        self.hidden=patch('tkinter.Toplevel',HiddenToplevel);self.hidden.start()
        self.maximize=patch.object(SpectrumWindow,'maximize',lambda self:None);self.maximize.start()
        self.root=tk.Tk();self.root.withdraw();self.errors=[]
        self.root.report_callback_exception=lambda *a:self.errors.append(str(a))
        self.view=SpectrumWindow(self.root);self.view.qam_compensation.auto_open.set(False)
        self.root.update_idletasks()
    def tearDown(self):
        self.view.busy=False;self.view.close();self.root.update_idletasks();self.root.destroy()
        self.maximize.stop();self.hidden.stop();self.assertFalse(self.errors,self.errors)
    def test_default_legacy_and_headerless_options_skip_invalid_pwm(self):
        v=self.view;p=v.qam_compensation;self.assertEqual(p.acquisition_options(),'pwm')
        p.acquisition_mode.set('无前导：重复图样');v.filename.set(str(self.spec.path));v.qam_pwm_symbols.set('')
        self.assertTrue(p.enabled.get());self.assertEqual(v.constellation_mode.get(),TDCM_MODE)
        self.assertEqual(v.tdcm_normalization.get(),'盲归一化')
        with patch('rx706_spectrum_gui.threading.Thread') as worker:
            v.analyze();o=worker.call_args.kwargs['args'][2]
            self.assertNotIn('pwm_preamble_options',o);self.assertEqual(o['tdcm_options']['acquisition_mode'],'no_header_repeat')
    def test_report_and_real_poll_dont_dereference_missing_pwm(self):
        v=self.view;v.constellation_mode.set(TDCM_MODE);v.qam_compensation.acquisition_mode.set('无前导：重复图样')
        from rx706_spectrum import display_trace
        v.events.put(('result',(self.spec,display_trace(self.spec),self.result,None)));v.poll()
        self.assertIn('无PWM获取',v.summary.get());self.assertIsNone(v.recovery.pwm_preamble)
        report=v.show_qam_report();self.root.update_idletasks();self.assertEqual(len(report.notebook.tabs()),6)
        self.assertEqual(len(report.frame_tree.get_children()),len(self.result.frames))
    def test_failure_and_config_load_are_supported(self):
        v=self.view;v.result=self.spec;v.recovery=self.result
        path=Path(self.temp.name)/'config.json';export_recipe_json(self.result,path)
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(path)),patch.object(v,'analyze') as analyse:
            v.load_qam_recipe();analyse.assert_called_once()
        self.assertEqual(v._next_qam_recipe['method'],'headerless_tdcm_config')
        v.recovery=None;v.result.auto_search_failure=dict(mode='no_header_repeat',status='failed',candidates=[dict(reason='独立验证失败')])
        report=v.show_qam_report();self.assertIsNone(report.r)
    def test_fixed_mode_does_not_claim_repeat_clock(self):
        p=self.view.qam_compensation;p.acquisition_mode.set('无前导：已知固定时钟/随机载荷')
        self.assertEqual(p.acquisition_options(),'no_header_fixed');self.assertEqual(p.options()['clock_mode'],'manual')

if __name__=='__main__':unittest.main()

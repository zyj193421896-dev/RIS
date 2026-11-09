"""Bounded auto-search UI tests in isolated hidden Tk instances."""
from pathlib import Path
import sys,tempfile,tkinter as tk,unittest,json,time
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_tdcm_auto import capture
from test_qam_debug_gui import HiddenToplevel
from rx706_spectrum_gui import SpectrumWindow,TDCM_MODE
from rx706_tdcm import recover_tdcm
from rx706_qam_debug_gui import export_recipe_json,export_report_html


class AutoGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:p=tk.Tk();p.withdraw();p.destroy()
        except tk.TclError as exc:raise unittest.SkipTest(str(exc))
        cls.temp=tempfile.TemporaryDirectory();cls.spec,cls.options,_,_=capture(cls.temp.name,frames=16)
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
        self.view.close();self.root.update_idletasks();self.root.destroy()
        self.maximize.stop();self.hidden.stop();self.assertFalse(self.errors,self.errors)

    def test_auto_controls_are_optional_and_delivered(self):
        v=self.view;p=v.qam_compensation
        self.assertFalse(p.search_enabled.get());p.search_enabled.set(True);self.assertTrue(p.enabled.get())
        self.assertEqual(v.constellation_mode.get(),TDCM_MODE)
        p.search_values['carrier_span_khz'].set('750');p.search_values['frame_candidates'].set('128 256')
        v.constellation_mode.set(TDCM_MODE);v.filename.set(str(self.spec.path));v.tdcm_count.set('256')
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            v.analyze();opts=thread.call_args.kwargs['args'][2]
            self.assertEqual(opts['tdcm_options']['auto_search']['carrier_span_hz'],750000)
            self.assertEqual(opts['tdcm_options']['payload_symbols'],0)
            self.assertEqual(opts['tdcm_options']['auto_search']['frame_candidates'],[128,256])
        v.busy=False

    def test_selected_scheme_table_json_export_and_replay_load(self):
        v=self.view;v.result=self.spec;v.recovery=self.result;v.busy=False
        report=v.show_qam_report();self.root.update_idletasks()
        self.assertEqual(len(report.notebook.tabs()),6)
        self.assertEqual(len(report.search_tree.get_children()),len(self.result.diagnostics['auto_search']['candidates']))
        self.assertTrue(report.search_tree.selection())
        output=Path(self.temp.name)/'scheme.json';export_recipe_json(self.result,output)
        self.assertEqual(json.loads(output.read_text())['search_outcome']['status'],'selected')
        html=Path(self.temp.name)/'auto.html';export_report_html(self.result,self.spec,html)
        page=html.read_text();self.assertIn('自动采用方案',page);self.assertIn('input_fingerprint',page)
        v.filename.set(str(self.spec.path))
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(output)),patch.object(v,'analyze') as analyse:
            v.load_qam_recipe();analyse.assert_called_once()
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            v.analyze();opts=thread.call_args.kwargs['args'][2]
            self.assertIn('replay_recipe',opts['tdcm_options'])
            self.assertIsNone(v._next_qam_recipe)
        v.busy=False
        # The replay is consumed once, but the visible profile must still
        # describe its carrier, clock and payload for the next analysis.
        recipe=json.loads(output.read_text())
        self.assertAlmostEqual(float(v.tdcm_carrier_mhz.get())*1e6,recipe['center_hint_hz'],places=5)
        self.assertEqual(v.qam_compensation.acquisition_options(),'pwm')
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            v.analyze();opts=thread.call_args.kwargs['args'][2]
            self.assertNotIn('replay_recipe',opts['tdcm_options'])
            self.assertFalse(v.qam_compensation.search_enabled.get())
            self.assertAlmostEqual(opts['tdcm_options']['compensation']['period_ppm'],recipe['tdcm_options']['compensation']['period_ppm'],places=7)
        v.busy=False

    def test_progress_does_not_finish_worker_and_failures_are_inspectable(self):
        v=self.view;v.busy=True;v.events.put(('progress','自动补偿：组合 2/8'));v.poll()
        self.assertTrue(v.busy);self.assertIn('2/8',v.status.get())
        failure=dict(status='failed',ranges=dict(carrier_span_hz=1000),candidates=[
            dict(id='T1',stage='精评',carrier_hz=-200000,frame_symbols=256,phase_mode='square',clock_mode='repeat',gain_mode='blind',status='拒绝',reason='独立验证失败')])
        v.busy=False;v.recovery=None;v.result=SimpleNamespace(auto_search_failure=failure)
        report=v.show_qam_report();self.root.update_idletasks();self.assertIsNone(report.r)
        self.assertIs(v.show_qam_report(),report)

    def test_invalid_recipe_is_reported_without_starting_analysis(self):
        file=Path(self.temp.name)/'invalid.json';file.write_text('[]')
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(file)),patch('rx706_spectrum_gui.messagebox.showerror') as error,patch.object(self.view,'analyze') as analyse:
            self.view.load_qam_recipe();error.assert_called_once();analyse.assert_not_called()

if __name__=='__main__':unittest.main()

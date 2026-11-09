"""Real Tk in withdrawn, isolated windows; never touches an existing GUI."""
from pathlib import Path
import sys,tempfile,time,tkinter as tk,unittest
from unittest.mock import patch
import numpy as np
from matplotlib.backend_bases import MouseEvent
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import SpectrumWindow,TDCM_MODE
from rx706_tdcm import recover_tdcm
from test_tdcm_debug import synthetic

OriginalToplevel=tk.Toplevel
class HiddenToplevel(OriginalToplevel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.withdraw()

class DebugGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe=tk.Tk();probe.withdraw();probe.destroy()
        except tk.TclError as exc:raise unittest.SkipTest(str(exc))
        cls.temp=tempfile.TemporaryDirectory()
        cls.spec,cls.options,_,_=synthetic(cls.temp.name,frames=14)
        cls.result=recover_tdcm(cls.spec,**cls.options)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.hidden=patch('tkinter.Toplevel',HiddenToplevel);self.hidden.start()
        self.maximize=patch.object(SpectrumWindow,'maximize',lambda self:None);self.maximize.start()
        self.root=tk.Tk();self.root.withdraw();self.errors=[]
        self.root.report_callback_exception=lambda *args:self.errors.append(args)
        self.view=SpectrumWindow(self.root)
        self.view.qam_compensation.auto_open.set(False)
        self.root.update_idletasks()

    def tearDown(self):
        self.view.close();self.root.update_idletasks();self.root.destroy()
        self.maximize.stop();self.hidden.stop();self.assertFalse(self.errors,self.errors)

    def test_separate_controls_profile_and_worker_options(self):
        view=self.view;panel=view.qam_compensation
        self.assertFalse(panel.enabled.get())
        view.qam_pwm_width_offset.set('1');view.apply_qam_debug_profile(panel)
        self.assertEqual(view.constellation_mode.get(),TDCM_MODE)
        self.assertEqual(view.qam_pwm_width_offset.get(),'0')
        self.assertEqual(view.tdcm_normalization.get(),'盲归一化')
        panel.values['phase_update_hz'].set('4000')
        self.assertIs(panel.calibration_variables['amplitude'],view.tdcm_amplitude)
        self.assertIs(panel.calibration_variables['training_symbols'],view.qam_preamble_symbols)
        view.filename.set(str(self.spec.path))
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            view.analyze();opts=thread.call_args.kwargs['args'][2]
            c=opts['tdcm_options']['compensation']
            self.assertTrue(c['enabled']);self.assertEqual(c['clock_mode'],'repeat');self.assertEqual(c['phase_mode'],'square')
            self.assertEqual(c['phase_update_hz'],4000);self.assertEqual(c['frame_symbols'],256)
            self.assertEqual(len(view.figure.axes),3)
        view.busy=False

    def test_report_tables_navigation_and_bounded_paging(self):
        view=self.view;view.busy=False;view.recovery=self.result;view.result=self.spec
        report=view.show_qam_report()
        self.root.update_idletasks()
        self.assertEqual(len(report.notebook.tabs()),5)
        self.assertLessEqual(len(report.symbol_tree.get_children()),500)
        self.assertEqual(len(report.frame_tree.get_children()),len(self.result.frames))
        report.set_page(1);self.assertEqual(report.page,1)
        fid=int(np.unique(self.result.frame_ids)[-1]);report.frame_filter.set(str(fid));report.set_page(0)
        self.assertEqual(len(report.symbol_tree.get_children()),256)
        canvas=report.canvases[0];ax=report.figures[0].axes[2];canvas.draw()
        old=ax.get_xlim();px,py=ax.transData.transform((0,0))
        report.navigation[0].scroll(MouseEvent('scroll_event',canvas,px,py,button='up',step=1))
        self.assertLess(np.diff(ax.get_xlim())[0],np.diff(old)[0]);report.navigation[0].reset(ax)
        self.assertIs(view.show_qam_report(),report)
        target=Path(self.temp.name)/'gui_export.csv'
        with patch('rx706_qam_debug_gui.filedialog.asksaveasfilename',return_value=str(target)):
            report.export('csv')
        deadline=time.monotonic()+5
        while report.export_thread.is_alive() and time.monotonic()<deadline:
            self.root.update_idletasks();time.sleep(.01)
        report.poll();self.assertTrue(target.is_file());self.assertIn('已导出',report.status.get())
        report.close();self.root.update_idletasks()

    def test_worker_result_reaches_summary_and_plot_without_auto_open(self):
        view=self.view;view.constellation_mode.set(TDCM_MODE)
        view._analyze_worker(self.spec.path,dict(nfft=1024,averages=1,sample_rate_hz=self.spec.sample_rate_hz),dict(kind='qam',**self.options))
        view.poll();self.assertIsNotNone(view.recovery)
        self.assertIn('后段重复模板EVM',view.summary.get());self.assertIn('未测BER',view.summary.get())
        self.assertIsNone(view._qam_report)
        self.assertGreater(len(view.iq_axes.collections[0].get_offsets()),1000)

if __name__=='__main__':unittest.main()

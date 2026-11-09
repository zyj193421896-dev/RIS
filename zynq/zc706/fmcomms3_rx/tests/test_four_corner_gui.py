"""Own hidden Tk views: four-corner preset, separate training, payload HEX."""
from pathlib import Path
import sys,tkinter as tk,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_qam_debug_gui import HiddenToplevel
from test_four_corner_training import capture
from rx706_tdcm import recover_tdcm
from rx706_spectrum_gui import SpectrumWindow,FOUR_CORNER_MODE


class FourCornerGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.spec,cls.options=capture(cls.temp.name)
        cls.result=recover_tdcm(cls.spec,**cls.options)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        self.hide=patch('tkinter.Toplevel',HiddenToplevel);self.hide.start()
        self.maximize=patch.object(SpectrumWindow,'maximize',lambda self:None);self.maximize.start()
        self.root=tk.Tk();self.root.withdraw();self.errors=[]
        self.root.report_callback_exception=lambda *e:self.errors.append(str(e))
        self.view=SpectrumWindow(self.root);self.view.qam_compensation.auto_open.set(False)
    def tearDown(self):
        self.view.busy=False;self.view.close();self.root.update_idletasks();self.root.destroy()
        self.maximize.stop();self.hide.stop();self.assertFalse(self.errors,self.errors)
    def test_preset_keeps_payload_count_and_delivers_explicit_hex_training(self):
        v=self.view;p=v.qam_compensation;p.values['frame_symbols'].set('12')
        v.apply_four_corner_training_profile();self.assertEqual(p.values['frame_symbols'].get(),'16')
        v.apply_four_corner_training_profile();self.assertEqual(p.values['frame_symbols'].get(),'16')
        self.assertEqual(v.qam_preamble_symbols.get(),'00 08 88 80')
        self.assertFalse(p.search_enabled.get());self.assertFalse(p.fit_timing.get())
        self.assertEqual(v.constellation_mode.get(),FOUR_CORNER_MODE)
        v.filename.set(str(self.spec.path))
        with patch('rx706_spectrum_gui.threading.Thread') as worker:
            v.analyze();o=worker.call_args.kwargs['args'][2]
        self.assertEqual(o['preamble_options']['representation'],'hex')
        self.assertEqual(o['preamble_options']['orientation'],'auto')
        self.assertTrue(o['preamble_options']['fit_dc_offset'])
        self.assertEqual(o['tdcm_options']['compensation']['frame_symbols'],16)
        self.assertEqual(o['pwm_preamble_options']['qam_search_symbols'],0)
    def test_training_tab_and_payload_hex_are_separate(self):
        v=self.view;v.result=self.spec;v.recovery=self.result
        report=v.show_qam_report();self.root.update_idletasks()
        names=[report.notebook.tab(tab,'text') for tab in report.notebook.tabs()]
        self.assertIn('已知训练 / 映射',names)
        values=report.symbol_tree.item(report.symbol_tree.get_children()[0],'values')
        self.assertRegex(values[5],r'^[0-9A-F]{2}$')
        first=report.training_tree.item(report.training_tree.get_children()[0],'values')
        self.assertEqual(first[2],'00');self.assertEqual(first[3],'00')
        self.assertEqual(len(self.result.scatter),12*sum(v['valid'] for v in self.result.frames))
    def test_named_mode_waits_for_input_and_preserves_net_payload_count(self):
        v=self.view;v.qam_compensation.values['frame_symbols'].set('12');v.filename.set('')
        v.constellation_mode.set(FOUR_CORNER_MODE)
        with patch.object(v,'analyze') as analyse:
            v.on_mode_changed();analyse.assert_not_called()
            v.filename.set(str(self.spec.path));v.on_mode_changed();analyse.assert_called_once()
        self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'16')

if __name__=='__main__':unittest.main()

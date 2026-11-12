"""Independent real-Tk regression for the explicit frame wizard."""
from pathlib import Path
import copy,json,sys,unittest,tempfile,tkinter as tk
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow
ORIGINAL=json.loads((ROOT/'host/gui_data/current_pwm/frame.json').read_text())
class GUIIndependentTests(unittest.TestCase):
 def setUp(self):
  self.root=tk.Tk(); self.root.withdraw(); self.view=SpectrumWindow(self.root); self.view.window.withdraw(); self.assertTrue(self.view.load_frame_config(config=copy.deepcopy(ORIGINAL)))
 def tearDown(self): self.view.close(); self.root.destroy()
 def open(self):
  self.view.generate_frame_config(); d=self.view._frame_builder; self.root.update_idletasks(); return d
 def snapshot(self): return copy.deepcopy(self.view._pwm_profile_snapshot())
 def test_cancel_no_mutation(self):
  before=self.snapshot(); self.view.recovery=object(); old=self.view.recovery; d=self.open(); d.vars['identity'].set('0 65 8 54 2 42 10 21 6 60 4 35'); d.window.destroy()
  self.assertEqual(before,self.snapshot()); self.assertIs(old,self.view.recovery); self.view.recovery=None
 def test_save_cancel_and_existing(self):
  before=self.snapshot(); d=self.open()
  with patch('tkinter.filedialog.asksaveasfilename',return_value=''): d.save()
  self.assertEqual(before,self.snapshot())
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'keep.json'; path.write_text('KEEP')
   with patch('tkinter.filedialog.asksaveasfilename',return_value=str(path)),patch('tkinter.messagebox.showerror') as error:
    d.save(); error.assert_called_once()
   self.assertEqual(path.read_text(),'KEEP'); self.assertEqual(before,self.snapshot())
 def test_new_header_clears_cal_and_apply_retires(self):
  d=self.open(); self.assertFalse(d.vars['reference_enabled'].get()); self.assertFalse(d.vars['calibration_enabled'].get())
  d.vars['reference_enabled'].set(True); d.vars['calibration_enabled'].set(True); d.vars['identity'].set('0 65 8 54 2 42 10 21 6 60 4 35')
  self.assertFalse(d.vars['reference_enabled'].get()); self.assertFalse(d.vars['calibration_enabled'].get())
  self.view.result=object(); self.view.recovery=object(); d.apply()
  self.assertIsNone(self.view.result); self.assertIsNone(self.view.recovery); self.assertFalse(self.view.pwm_cycle_calibration)
  self.assertFalse(self.view.pwm_frame_reference_enabled.get()); self.assertEqual(len(self.view.pulse_config()['frame_options']['users'][0]['identity']),12)
 def test_multiuser_gui_retains(self):
  self.view.pwm_frame_users.append(dict(name='U2',identity='65 65 65 65',payload_symbols=66,reference='')); self.view.pwm_frame_user_count.set('2'); before=self.snapshot()
  with patch('rx706_spectrum_gui.messagebox.showerror') as error: self.view.generate_frame_config(); error.assert_called_once()
  self.assertEqual(before,self.snapshot())
 def test_top_entry_at_minimum_width(self):
  self.view.window.geometry('1000x650'); self.view.window.deiconify()
  self.root.after(50,self.root.quit); self.root.mainloop()
  button=self.view.frame_builder_button
  self.assertTrue(button.winfo_ismapped())
  self.assertGreaterEqual(button.winfo_width(),button.winfo_reqwidth())
  self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),self.view.window.winfo_rootx()+self.view.window.winfo_width())
 def test_codebook_edit_disables_old_reference_and_calibration(self):
  d=self.open(); d.vars['reference_enabled'].set(True); d.vars['calibration_enabled'].set(True)
  d.texts['codebook'].insert('end','\n'); d.codebook_changed()
  self.assertFalse(d.vars['reference_enabled'].get()); self.assertFalse(d.vars['calibration_enabled'].get())

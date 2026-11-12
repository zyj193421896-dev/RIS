"""Withdrawn real-Tk exercise of 78-symbol draft/apply/save and legacy entry."""
import copy,json,sys,tempfile,tkinter as tk,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow
from rx706_pwm_symbols import PLUS_SOURCE,LEGACY_PROFILE
from rx706_workflows import AREAS
class Symbols78GUITests(unittest.TestCase):
 def setUp(self):
  self.root=tk.Tk();self.root.withdraw()
  with patch.object(SpectrumWindow,'maximize'):self.view=SpectrumWindow(self.root)
  self.view.window.withdraw();self.flow=self.view.workflows
  self.view.load_frame_config(filename=LEGACY_PROFILE)
 def tearDown(self):self.view.close();self.root.destroy()
 def open(self):
  self.assertTrue(self.flow.configure_plus_symbols());self.view._frame_builder.window.withdraw()
  self.root.update_idletasks();return self.view._frame_builder
 def test_open_cancel_and_missing_payload_do_not_mutate(self):
  self.view.filename.set('old.bin');before=self.flow.snapshot();sentinel=object();self.view.result=sentinel
  d=self.open();self.assertEqual(d.vars['payload_symbols'].get(),'')
  self.assertEqual(d.vars['identity'].get(),'75 8 60 2 46 10 23 6 68 38')
  with patch('tkinter.messagebox.showerror') as error:d.apply();error.assert_called_once()
  self.assertEqual(self.flow.snapshot(),before);self.assertIs(self.view.result,sentinel)
  d.window.destroy();self.assertEqual(self.flow.snapshot(),before)
 def test_save_apply_reload_and_legacy_switch(self):
  self.view.filename.set('old.bin');d=self.open();d.vars['payload_symbols'].set('78')
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'frame.json'
   with patch('tkinter.filedialog.asksaveasfilename',return_value=str(path)):d.save()
   self.assertEqual(self.view.pulse_config()['symbol_count'],66)
   d.apply();self.assertEqual(self.view.filename.get(),'');self.assertEqual(self.view.recovery_ms.get(),'2000')
   self.assertIsNone(self.flow.confirmed);self.assertFalse(self.view.pwm_frame_reference_enabled.get())
   self.assertEqual(self.view.pulse_config()['symbol_count'],78)
   self.assertEqual(self.view.pulse_config()['frame_options']['users'][0]['payload_symbols'],78)
   d.window.destroy();self.assertTrue(self.view.load_frame_config(filename=path))
   self.assertEqual(self.view.pulse_config()['symbol_count'],78)
   self.assertTrue(self.flow.load_new_pwm_protocol());self.assertEqual(self.view.pulse_config()['symbol_count'],66)
 def test_busy_duplicate_open_and_wrong_area_guard(self):
  self.view.busy=True;self.assertFalse(self.flow.configure_plus_symbols());self.view.busy=False
  d=self.open();self.assertFalse(self.flow.configure_plus_symbols());d.window.destroy()
  for area in AREAS[1:]:
   self.flow.switch(area);self.assertFalse(self.flow.configure_plus_symbols());self.assertEqual(self.flow.plus_symbols_button.winfo_manager(),'')
  self.flow.switch(AREAS[0]);self.assertEqual(self.flow.plus_symbols_button.winfo_manager(),'pack')
 def test_generic_import_cancel_bad_file_atomic_and_real_file(self):
  self.view.generate_frame_config();d=self.view._frame_builder;d.window.withdraw();before=d.fields()
  with patch('tkinter.filedialog.askopenfilename',return_value=''):self.assertFalse(d.import_mt())
  self.assertEqual(before,d.fields())
  with tempfile.TemporaryDirectory() as directory:
   bad=Path(directory)/'bad.txt';bad.write_text('(25,0);\n(30,0);')
   with patch('tkinter.messagebox.showerror') as error:self.assertFalse(d.import_mt(bad));error.assert_called_once()
  self.assertEqual(before,d.fields());self.assertTrue(d.import_mt(PLUS_SOURCE))
  self.assertEqual(d.vars['payload_symbols'].get(),'');self.assertEqual(d.vars['low_guard_chips'].get(),'4')
  self.assertEqual(len(d.texts['codebook'].get('1.0','end').strip().splitlines()),78)
if __name__=='__main__':unittest.main()

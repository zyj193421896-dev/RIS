"""The direct receiver entry owns and closes its hidden root."""
import sys
from pathlib import Path
import tkinter as tk
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_receiver_gui import create_app
from rx706_spectrum_gui import SpectrumWindow

class ReceiverLauncherTests(unittest.TestCase):
    def test_close_receiver_also_closes_capture_worker(self):
        with patch.object(SpectrumWindow, 'maximize'):
            root, view = create_app()
        view.window.withdraw()
        with patch.object(tk.Toplevel, 'deiconify'), patch.object(tk.Toplevel, 'lift'):
            app = view.board_capture.show()
        self.assertFalse(app.connected)
        view.close()
        app.controller.thread.join(2)
        self.assertFalse(app.controller.thread.is_alive())
        self.assertTrue(view.board_capture.closed)
        with self.assertRaises(tk.TclError):
            root.winfo_exists()

    def test_launch_no_guess_and_close_hidden_root(self):
        with patch.object(SpectrumWindow,'maximize'), patch.object(SpectrumWindow,'analyze') as analyze:
            root,view=create_app('example.bin')
        view.window.withdraw()
        self.assertEqual(root.state(),'withdrawn')
        self.assertEqual(view.filename.get(),'example.bin')
        self.assertIsNone(view.workflows.confirmed)
        self.assertIsNone(view.board_capture.app)
        self.assertEqual(view.board_capture.open_button.cget('text'), '连接板卡 / 采集…')
        analyze.assert_not_called()
        view.close(); view.close()
        self.assertTrue(view.closed); self.assertTrue(view.cancel.is_set())
        with self.assertRaises(tk.TclError): root.winfo_exists()

if __name__=='__main__': unittest.main()

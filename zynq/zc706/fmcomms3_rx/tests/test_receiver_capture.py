"""Receiver acquisition UI tests. All board traffic is an in-process fake."""
import copy
import _tkinter
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
import rx706 as protocol
from rx706_gui import Controller
from rx706_receiver_capture import ReceiverCaptureBridge
from rx706_spectrum_gui import SpectrumWindow


class FakeSocket:
    def settimeout(self, value):
        pass
    def shutdown(self, how):
        pass


class FakeClient:
    commands = []
    fail_connect = False
    read_error = False

    def __init__(self, *args):
        if self.fail_connect:
            raise OSError('simulated connection refused')
        self.socket = FakeSocket()
        self.closed = False
        self.info = self.ready_info(32768)

    @staticmethod
    def ready_info(size):
        return dict(ok=True, protocol=1, state='ready', result=0, reset_verified=1,
                    bytes=size, received_bytes=size, completed_packets=size // 32768,
                    packets=size // 32768, tuples=size // 8, dtype='int16', byte_order='little',
                    order=['I1', 'Q1', 'I2', 'Q2'], channels=2, valid_bits=12,
                    sample_rate_hz=15360000, capture_id='0123456789abcdef')

    def request(self, command):
        self.commands.append(command)
        if command.startswith('CAPTURE '):
            self.info = self.ready_info(int(command.split()[1]))
        elif command.startswith('RELEASE '):
            self.info['state'] = 'empty'
        return copy.deepcopy(self.info)

    def read_chunk(self, capture_id, offset, length):
        if self.read_error:
            raise protocol.ProtocolError('simulated CRC failure')
        data = bytes((i % 256 for i in range(offset, offset + length)))
        return data, f'{zlib.crc32(data) & 0xffffffff:08x}'

    def close(self):
        self.closed = True


class ReceiverCaptureTests(unittest.TestCase):
    def setUp(self):
        FakeClient.commands = []
        FakeClient.fail_connect = FakeClient.read_error = False
        self.client_patch = patch('rx706_gui.protocol.Client', FakeClient)
        self.console_patch = patch.object(Controller, '_console_reader')
        self.client_patch.start()
        self.console_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        with patch.object(SpectrumWindow, 'maximize'):
            self.view = SpectrumWindow(self.root)
        self.view.window.withdraw()
        self.bridge = ReceiverCaptureBridge(self.view)
        self.app = None

    def tearDown(self):
        self.bridge.close()
        self.view.close()
        self.root.destroy()
        if self.app:
            self.app.controller.thread.join(2)
            self.assertFalse(self.app.controller.thread.is_alive())
        self.console_patch.stop()
        self.client_patch.stop()
        self.temp.cleanup()
        self.assertFalse(self.errors, self.errors)

    def show(self):
        with patch.object(tk.Toplevel, 'deiconify'), patch.object(tk.Toplevel, 'lift'):
            self.app = self.bridge.show()
        self.app.root.withdraw()
        self.app.output.set(str(Path(self.temp.name) / 'capture.bin'))
        return self.app

    def until(self, predicate, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.root.tk.dooneevent(_tkinter.ALL_EVENTS | _tkinter.DONT_WAIT)
            if predicate():
                return
            time.sleep(.01)
        self.fail('GUI condition timed out: ' + self.app.notice.get())

    def connect(self):
        self.app.connect_button.invoke()
        self.until(lambda: self.app.connected and not self.app.busy)

    def download(self):
        self.show()
        self.connect()
        self.app.download_button.invoke()
        self.until(lambda: self.bridge.latest is not None and not self.app.busy)

    def test_launch_and_open_do_not_connect_or_capture(self):
        self.assertIsNone(self.bridge.app)
        self.assertFalse(FakeClient.commands)
        self.assertTrue(self.bridge.load_button.instate(['disabled']))
        self.show()
        self.root.tk.dooneevent(_tkinter.ALL_EVENTS | _tkinter.DONT_WAIT)
        self.assertFalse(FakeClient.commands)
        self.assertFalse(self.app.connected)
        self.assertTrue(self.app.capture_button.instate(['disabled']))
        with patch.object(tk.Toplevel, 'deiconify'), patch.object(tk.Toplevel, 'lift'):
            self.assertIs(self.bridge.show(), self.app)
        self.app.root.withdraw()

    def test_download_requires_explicit_load_and_reconfirmation(self):
        self.view.filename.set('existing.bin')
        self.view.auto.set(True)
        self.view.workflows.confirmed = 'previous confirmation'
        users = copy.deepcopy(self.view.pwm_frame_users)
        with patch.object(self.view, 'run_workflow') as run:
            self.download()
            self.assertEqual(self.view.filename.get(), 'existing.bin')
            self.assertTrue(self.bridge.load_latest())
            self.assertEqual(self.view.filename.get(), self.app.output.get())
            self.assertIsNone(self.view.workflows.confirmed)
            self.assertEqual(self.view.pwm_frame_users, users)
            run.assert_not_called()
        sidecar = json.loads(Path(self.app.output.get() + '.json').read_text())
        self.assertTrue(sidecar['complete'])
        self.assertEqual(sidecar['file_bytes'], 32768)
        self.assertEqual(len(sidecar['sha256']), 64)
        self.assertFalse(any(c.startswith('CAPTURE ') for c in FakeClient.commands))

    def test_busy_and_editor_block_loading_without_overwrite(self):
        self.download()
        self.view.filename.set('existing.bin')
        self.view.busy = True
        self.assertFalse(self.bridge.load_latest())
        self.assertEqual(self.view.filename.get(), 'existing.bin')
        self.view.busy = False
        with patch.object(self.view.workflows, 'editor_open', return_value=True):
            self.assertFalse(self.bridge.load_latest())
        self.assertEqual(self.view.filename.get(), 'existing.bin')
        self.assertTrue(self.bridge.load_latest())

    def test_load_targets_current_area_preserving_other_area(self):
        from rx706_workflows import AREAS
        self.download()
        self.view.filename.set('pwm.bin')
        self.assertTrue(self.view.workflows.switch(AREAS[1]))
        self.assertTrue(self.bridge.load_latest())
        self.assertEqual(self.view.filename.get(), self.app.output.get())
        self.assertTrue(self.view.workflows.switch(AREAS[0]))
        self.assertEqual(self.view.filename.get(), 'pwm.bin')

    def test_missing_or_partial_metadata_is_not_loaded(self):
        self.download()
        self.view.filename.set('existing.bin')
        Path(self.app.output.get() + '.json').unlink()
        self.assertFalse(self.bridge.load_latest())
        self.assertEqual(self.view.filename.get(), 'existing.bin')

    def test_changed_sidecar_is_not_loaded(self):
        self.download()
        self.view.filename.set('existing.bin')
        path = Path(self.app.output.get() + '.json')
        saved = json.loads(path.read_text())
        saved['capture']['sample_rate_hz'] = 1000000
        path.write_text(json.dumps(saved))
        self.assertFalse(self.bridge.load_latest())
        self.assertEqual(self.view.filename.get(), 'existing.bin')

    def test_hide_reopen_and_close_keep_owner_alive_no_release(self):
        self.download()
        self.app.hide()
        self.assertTrue(self.app.connected)
        with patch.object(tk.Toplevel, 'deiconify'), patch.object(tk.Toplevel, 'lift'):
            self.assertIs(self.bridge.show(), self.app)
        self.bridge.close()
        self.bridge.close()
        self.assertTrue(self.view.window.winfo_exists())
        self.assertFalse(self.view.closed)
        self.assertFalse(any(c.startswith(('ABORT', 'RELEASE', 'CAPTURE')) for c in FakeClient.commands))

    def test_capture_explicit_only_and_original_packet_alignment(self):
        self.show()
        self.connect()
        self.app.release_button.invoke()
        self.until(lambda: not self.app.busy and self.app.info['state'] == 'empty')
        self.app.preset('.001', '秒')
        self.app.capture_button.invoke()
        self.until(lambda: self.bridge.latest is not None and not self.app.busy)
        self.assertIn('CAPTURE 131072', FakeClient.commands)
        self.assertEqual(Path(self.app.output.get()).stat().st_size, 131072)
        self.assertEqual(self.view.filename.get(), '')
        self.assertTrue(self.bridge.load_latest())

    def test_connect_failure_then_retry(self):
        self.show()
        FakeClient.fail_connect = True
        self.app.connect_button.invoke()
        self.until(lambda: not self.app.busy)
        self.assertFalse(self.app.connected)
        self.assertIn('simulated connection refused', self.app.notice.get())
        self.assertTrue(self.app.connect_button.instate(['!disabled']))
        FakeClient.fail_connect = False
        self.connect()
        self.assertTrue(self.app.connected)

    def test_failed_download_never_offered_then_resume(self):
        self.show()
        self.connect()
        FakeClient.read_error = True
        self.app.download_button.invoke()
        self.until(lambda: not self.app.busy)
        self.assertIsNone(self.bridge.latest)
        self.assertTrue(self.bridge.load_button.instate(['disabled']))
        self.assertTrue(Path(self.app.output.get() + '.part').is_file())
        FakeClient.read_error = False
        self.connect()
        self.app.resume_button.invoke()
        self.until(lambda: self.bridge.latest is not None and not self.app.busy)
        self.assertTrue(self.bridge.load_latest())


if __name__ == '__main__':
    unittest.main(verbosity=2)

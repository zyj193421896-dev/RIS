"""Standalone package isolation, loopback transport/resume and copy integrity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import tkinter as tk
import unittest
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load('portable_builder', ROOT/'scripts/build_capture_portable.py')


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            line = self.rfile.readline()
            if not line:
                return
            parts = line.decode('ascii').strip().split()
            command = parts[0]
            data = b''
            info = self.server.info
            if command == 'CAPTURE':
                size = int(parts[1])
                self.server.data = struct.pack('<hhhh', 1, -2, 3, -4)*(size//8)
                info.update(state='ready', bytes=size, received_bytes=size, tuples=size//8,
                            completed_packets=size//32768, packets=size//32768)
                reply = dict(info, ok=True)
            elif command in ('STATUS', 'HELLO'):
                reply = dict(info, ok=True)
            elif command == 'READ':
                offset, length = int(parts[2]), int(parts[3])
                data = self.server.data[offset:offset+length]
                reply = dict(ok=True, type='data', capture_id=info['capture_id'], offset=offset,
                             length=length, crc32='%08x' % (zlib.crc32(data) & 0xffffffff))
                self.server.reads += 1
                if self.server.corrupt:
                    data = data[:-1]+bytes([data[-1] ^ 1])
                    self.server.corrupt = False
                if self.server.drop_once and offset:
                    self.server.drop_once = False
                    self.wfile.write((json.dumps(reply)+'\n').encode('ascii')+data[:1000])
                    self.wfile.flush()
                    self.request.shutdown(socket.SHUT_RDWR)
                    return
            else:
                reply = dict(ok=False, error='unsupported test command')
            header = (json.dumps(reply)+'\n').encode('ascii')
            # Force fragmented headers as on a real stream, never a packet API.
            for at in range(0, len(header), 7):
                self.wfile.write(header[at:at+7])
            self.wfile.write(data)
            self.wfile.flush()


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.stage = builder.build(Path(cls.temp.name), 'capture_test')
        cls.menu = load('portable_menu', cls.stage/'capture_menu.py')
        cls.verify = load('portable_verify', cls.stage/'verify_capture.py')
        cls.rx = cls.menu.protocol

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.server = Server(('127.0.0.1', 0), Handler)
        self.server.info = dict(protocol=1, state='empty', result=0, reset_verified=1,
            capture_id='0123456789abcdef', dtype='int16', byte_order='little',
            order=['I1', 'Q1', 'I2', 'Q2'], channels=2, valid_bits=12, sample_rate_hz=15360000)
        self.server.data = b''
        self.server.reads = 0
        self.server.drop_once = self.server.corrupt = False
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True).start()
        self.data_temp = tempfile.TemporaryDirectory()
        self.output = Path(self.data_temp.name)/'capture.bin'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.data_temp.cleanup()

    def client(self):
        return self.rx.Client(*self.server.server_address, timeout=3)

    def prime(self):
        with self.client() as client:
            client.request('CAPTURE %d' % (self.rx.CHUNK_BYTES+self.rx.PACKET_BYTES))

    def test_capture_and_download_complete_pair_without_scientific_packages(self):
        with self.client() as client:
            saved = self.menu.capture_and_download(client, '.003', self.output)
        self.assertEqual(self.output.read_bytes(), self.server.data)
        self.assertTrue(self.output.with_suffix('.bin.json').is_file())
        self.assertEqual(self.verify.verify(self.output)['sha256'], saved['sha256'])
        self.assertEqual(self.server.info['state'], 'ready')
        result = subprocess.run([sys.executable, '-I', '-X', 'utf8', str(self.stage/'verify_capture.py'), str(self.output)],
                                capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_disconnect_resume_retains_exact_bytes_and_rejects_copy_damage(self):
        self.prime()
        self.server.drop_once = True
        with self.client() as client, self.assertRaises(self.rx.ProtocolError):
            self.rx.download(client, self.output, progress=False)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.with_name(self.output.name+'.part').exists())
        with self.client() as client:
            saved = self.rx.download(client, self.output, resume=True, progress=False)
        self.assertEqual(self.output.read_bytes(), self.server.data)
        self.assertEqual(self.verify.verify(self.output)['sha256'], saved['sha256'])
        damaged = bytearray(self.output.read_bytes())
        damaged[20] ^= 1
        self.output.write_bytes(damaged)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.verify.verify(self.output)

    def test_bad_transport_crc_is_rejected_without_publishing_bin(self):
        self.prime()
        self.server.corrupt = True
        with self.client() as client, self.assertRaises(self.rx.ProtocolError):
            self.rx.download(client, self.output, progress=False)
        self.assertFalse(self.output.exists())

    def test_existing_capture_and_output_are_not_discarded(self):
        self.prime()
        with self.client() as client, self.assertRaises(ValueError):
            self.menu.capture_and_download(client, '1', self.output)
        self.assertEqual(self.server.info['state'], 'ready')
        self.assertEqual(self.server.reads, 0)
        self.assertEqual(self.menu.capture_bytes('1'), 122880000)
        for value in ('0', '-1', 'nan', 'inf', '18'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.menu.capture_bytes(value)

    def test_archives_allowlist_checksums_modes_and_original_protocol(self):
        source = ROOT/'host/rx706.py'
        self.assertEqual((self.stage/'app/rx706.py').read_bytes(), source.read_bytes())
        expected = set()
        for line in (self.stage/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(hashlib.sha256((self.stage/name).read_bytes()).hexdigest(), digest)
            expected.add('capture_test/'+name)
        expected.add('capture_test/MANIFEST.sha256')
        with tarfile.open(self.stage.with_suffix('.tar.gz')) as archive:
            self.assertEqual(set(archive.getnames()), expected)
            self.assertEqual(archive.getmember('capture_test/start.sh').mode, 0o755)
        with zipfile.ZipFile(self.stage.with_suffix('.zip')) as archive:
            self.assertEqual(set(archive.namelist()), expected)
        self.assertFalse(any('numpy' in name or 'spectrum' in name or 'frame_profiles' in name for name in expected))

    def test_capture_gui_has_no_analysis_entry(self):
        # Discovery imports the analysis suites in this process. Verify the
        # capture-only distribution in a fresh interpreter, as users launch it.
        code = r"""
from pathlib import Path
import sys
import tkinter as tk
sys.path.insert(0, sys.argv[1])
from capture_gui import CaptureApp
try:
    root = tk.Tk()
except tk.TclError as exc:
    print(str(exc)); raise SystemExit(77)
root.withdraw()
app = CaptureApp(root)
try:
    assert not app.spectrum_button.winfo_exists()
    assert str(app.output.get()).startswith(str(Path(sys.argv[1])/'captures'))
    assert 'rx706_spectrum_gui' not in sys.modules
    assert 'numpy' not in sys.modules
finally:
    app.controller.close()
    app.controller.thread.join(timeout=2)
    app.destroy()
"""
        result = subprocess.run([sys.executable, '-I', '-c', code, str(self.stage)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 77:
            self.skipTest(result.stdout.strip())
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()

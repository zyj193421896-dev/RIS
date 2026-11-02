#!/usr/bin/env python3
"""Real firmware dispatcher -> fragmented localhost TCP -> production client."""
import ctypes
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rx706', ROOT / 'host/rx706.py')
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)
core = ctypes.CDLL(sys.argv.pop(1))
core.test_rpc.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
core.test_rpc.restype = ctypes.c_uint
SIZE = 2 * rx.CHUNK_BYTES + rx.PACKET_BYTES
EXPECTED = b''.join(struct.pack('<4096h', *[((i + block) % 4096) - 2048 for i in range(4096)])
                    for block in range(SIZE // 8192))


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            line = self.rfile.readline(1024)
            if not line:
                return
            pointer = ctypes.c_void_p()
            count = core.test_rpc(line.strip(), ctypes.byref(pointer))
            answer = ctypes.string_at(pointer, count)
            if line.startswith(b'READ'):
                time.sleep(getattr(self.server, 'read_delay', 0))
                self.server.reads += 1
                if getattr(self.server, 'stall_once', False):
                    self.server.stall_once = False
                    self.connection.sendall(answer[:2000])
                    time.sleep(0.3)
                    return
                if self.server.corrupt:
                    answer = answer[:-1] + bytes([answer[-1] ^ 1])
                    self.server.corrupt = False
                if self.server.drop and self.server.reads == 2:
                    self.connection.sendall(answer[:2000])
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
            # Fragment JSON and the beginning of the binary payload arbitrarily.
            for pos in range(0, min(200, len(answer)), 7):
                self.connection.sendall(answer[pos:min(pos + 7, 200)])
            if len(answer) > 200:
                self.connection.sendall(answer[200:])


class Board(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TransferTests(unittest.TestCase):
    def setUp(self):
        core.test_reset()
        core.test_auto_complete()
        self.board = Board(('127.0.0.1', 0), Handler)
        self.board.corrupt = self.board.drop = False
        self.board.reads = 0
        self.thread = threading.Thread(target=self.board.serve_forever,
                                       kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name) / 'capture.bin'
        self.capture()

    def tearDown(self):
        self.board.shutdown()
        self.board.server_close()
        self.directory.cleanup()

    def client(self):
        return rx.Client(*self.board.server_address, timeout=3)

    def capture(self):
        with self.client() as client:
            client.request(f'CAPTURE {SIZE}')
            for _ in range(10):
                self.info = client.request('STATUS')
                if self.info['state'] == 'ready':
                    return
        self.fail('mock capture did not complete')

    def fetch(self, resume=False):
        with self.client() as client:
            return rx.download(client, self.output, resume=resume, progress=False)

    def test_exact_bin_crc_sha_and_metadata(self):
        saved = self.fetch()
        self.assertEqual(self.output.read_bytes(), EXPECTED)
        self.assertEqual(saved['sha256'], hashlib.sha256(EXPECTED).hexdigest())
        self.assertEqual(len(saved['chunk_crc32']), 3)
        self.assertTrue(saved['complete'])
        self.assertEqual(struct.unpack('<4h', self.output.read_bytes()[:8]),
                         (-2048, -2047, -2046, -2045))
        self.assertFalse(Path(str(self.output) + '.part').exists())
        self.assertEqual(json.loads(Path(str(self.output) + '.json').read_text())['file_bytes'], SIZE)
        self.assertEqual(saved['capture_final']['debug_read_ops'], 3)
        self.assertEqual(saved['capture_final']['debug_read_bad'], [0, 0, 0, 0])
        self.assertEqual(saved['debug']['capture_id'], self.info['capture_id'])
        self.assertTrue(saved['debug']['sampled_only'])
        self.assertEqual(saved['debug']['phases'], 7)
        self.assertGreater(len(saved['debug']['probes']), 0)
        for probe in saved['debug']['probes']:
            self.assertLessEqual(probe['offset'] + probe['length'], SIZE)
            for measurement in probe['measurements']:
                self.assertEqual(len(measurement), 9)
                self.assertEqual(measurement[0], measurement[1])
                self.assertEqual(measurement[0], measurement[2])
                self.assertEqual(measurement[3:7], [0, 0, 0, 0])

    def test_profile_metadata_survives_bin_export(self):
        profile = int(os.environ.get('RX706_TEST_PROFILE', '5800'))
        expected = profile * 1_000_000
        self.assertEqual(self.info['rx_lo_config_hz'], expected)
        self.assertEqual(self.info['rf_profile'], {2400:'CONFIG2P4',5800:'CONFIG5P8',915:'CONFIG915'}[profile])
        saved = self.fetch()
        self.assertEqual(saved['capture']['rx_lo_config_hz'], expected)

    def test_periodic_reconnect_keeps_frozen_identity_and_exact_bytes(self):
        with patch.object(rx, 'RECONNECT_BYTES', rx.CHUNK_BYTES):
            saved = self.fetch()
        self.assertEqual(saved['transport']['connection_rotations'], 2)
        self.assertEqual(saved['transport']['read_timeouts'], 0)
        self.assertEqual(saved['capture']['capture_id'], self.info['capture_id'])
        self.assertEqual(saved['capture_final']['capture_id'], self.info['capture_id'])
        self.assertEqual(self.output.read_bytes(), EXPECTED)

    def test_mid_chunk_timeout_retries_only_the_frozen_read(self):
        self.board.stall_once = True
        with rx.Client(*self.board.server_address, timeout=0.1) as client:
            saved = rx.download(client, self.output, progress=False)
        self.assertEqual(saved['transport']['read_timeouts'], 1)
        self.assertEqual(saved['capture']['capture_id'], self.info['capture_id'])
        self.assertEqual(saved['capture_final']['capture_id'], self.info['capture_id'])
        self.assertEqual(self.output.read_bytes(), EXPECTED)
        self.assertEqual(saved['sha256'], hashlib.sha256(EXPECTED).hexdigest())

    def test_mid_chunk_disconnect_then_resume(self):
        self.board.drop = True
        with self.assertRaises(rx.ProtocolError):
            self.fetch()
        saved = json.loads(Path(str(self.output) + '.part.json').read_text())
        self.assertEqual(saved['received_bytes'], rx.CHUNK_BYTES)
        # An uncommitted tail must be truncated, not appended to valid samples.
        with Path(str(self.output) + '.part').open('ab') as f:
            f.write(b'uncommitted tail')
        self.board.drop = False
        self.fetch(resume=True)
        self.assertEqual(self.output.read_bytes(), EXPECTED)

    def test_crc_error_never_publishes_bin(self):
        self.board.corrupt = True
        with self.assertRaisesRegex(rx.ProtocolError, 'CRC32'):
            self.fetch()
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads(Path(str(self.output) + '.part.json').read_text())['received_bytes'], 0)
        self.fetch(resume=True)
        self.assertEqual(self.output.read_bytes(), EXPECTED)

    def test_stale_capture_cannot_resume(self):
        self.board.drop = True
        with self.assertRaises(rx.ProtocolError):
            self.fetch()
        self.board.drop = False
        with self.client() as client:
            client.request(f"RELEASE {self.info['capture_id']}")
        self.capture()
        with self.assertRaisesRegex(rx.ProtocolError, 'changed'):
            self.fetch(resume=True)
        self.assertFalse(self.output.exists())

    def test_lo_change_cannot_resume_even_with_same_capture_id(self):
        self.board.drop = True
        with self.assertRaises(rx.ProtocolError):
            self.fetch()
        self.board.drop = False
        with self.client() as client:
            original_request = client.request

            def different_lo(command):
                answer = original_request(command)
                if command == 'STATUS':
                    answer['rx_lo_config_hz'] = 1_800_000_000
                return answer

            client.request = different_lo
            with self.assertRaisesRegex(rx.ProtocolError, 'changed'):
                rx.download(client, self.output, resume=True, progress=False)

    def test_corrupted_partial_file_rejected(self):
        self.board.drop = True
        with self.assertRaises(rx.ProtocolError):
            self.fetch()
        self.board.drop = False
        with Path(str(self.output) + '.part').open('r+b') as f:
            f.write(b'bad!')
        with self.assertRaisesRegex(rx.ProtocolError, 'local CRC'):
            self.fetch(resume=True)

    def test_frozen_capture_is_not_overwritten(self):
        with self.client() as client:
            with self.assertRaises(rx.ProtocolError):
                client.request(f'CAPTURE {SIZE}')
            self.assertEqual(client.request('STATUS')['capture_id'], self.info['capture_id'])
            with self.assertRaises(rx.ProtocolError):
                client.request('READ wrong 0 8')
            with self.assertRaises(rx.ProtocolError):
                client.request(f"READ {self.info['capture_id']} 4294967296 8")

    def test_existing_files_are_preserved(self):
        self.output.write_bytes(b'user data')
        with self.assertRaises(FileExistsError):
            self.fetch()
        self.assertEqual(self.output.read_bytes(), b'user data')

    def test_bad_capture_metadata_rejected(self):
        for key, value in [('received_bytes', SIZE-8), ('state','failed'),
                           ('reset_verified',0), ('dtype','uint16'), ('bytes',True)]:
            info = dict(self.info)
            info[key] = value
            with self.assertRaises(rx.ProtocolError):
                rx.validate_capture(info)

    def test_cli_default_capacity(self):
        with self.client() as client:
            client.request(f"RELEASE {self.info['capture_id']}")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = rx.main(['--host', self.board.server_address[0], '--port',
                              str(self.board.server_address[1]), 'capture', '--no-download'])
        self.assertEqual(result, 0)
        with self.client() as client:
            info = client.request('STATUS')
        self.assertEqual(info['bytes'], rx.MAX_BYTES)
        self.assertEqual(info['tuples'], 267911168)

    def test_download_cancel_then_resume(self):
        stopped = threading.Event()
        updates = []

        def progress(done, total, rate, phase):
            updates.append((done, total, phase))
            if phase == 'download' and done >= rx.CHUNK_BYTES:
                stopped.set()

        with self.client() as client:
            with self.assertRaises(rx.TransferCancelled):
                rx.download(client, self.output, progress=False,
                            progress_callback=progress, cancel_event=stopped)
            self.assertFalse(self.output.exists())
            self.assertEqual(client.request('STATUS')['state'], 'ready')
            rx.download(client, self.output, progress=False, resume=True)
        self.assertEqual(self.output.read_bytes(), EXPECTED)
        self.assertTrue(updates)


if __name__ == '__main__':
    unittest.main(verbosity=2)

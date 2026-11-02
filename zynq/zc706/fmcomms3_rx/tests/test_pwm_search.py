"""Leading receiver noise/CW and bounded PWM onset search in file coordinates."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin, AnalysisCancelled
from rx706_pwm import recover_pwm


class PWMSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'delayed.bin'

    def tearDown(self):
        self.tmp.cleanup()

    def capture(self, fs=1000000, end=1., prefix='noise', onset=.500023):
        self.fs, self.onset, self.rate = fs, onset, 10000
        t = np.arange(round(fs*end))/fs
        index = np.floor((t-onset)*self.rate).astype(int)
        within = ((t-onset)*self.rate-index)*10
        widths = np.array([8, 7, 5, 3, 1, 2, 4, 6])
        high = (index >= 0) & (within < widths[index % len(widths)])
        carrier = np.exp(2j*np.pi*100000*t+.43j)
        x = np.where(high, -300, 300)*carrier
        rng = np.random.default_rng(4)
        if prefix == 'noise':
            x[t < onset] = 0
        elif prefix == 'only_noise':
            x[:] = 0
        elif prefix == 'only_cw':
            x = 300*carrier
        x += rng.normal(0, 2, len(t))+1j*rng.normal(0, 2, len(t))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:,0], raw[:,1] = np.rint(x.real), np.rint(x.imag)
        raw.tofile(self.path)
        return analyze_bin(self.path, nfft=1024, averages=1, sample_rate_hz=fs)

    def check(self, r):
        self.assertTrue(r.decode_locked)
        rows = r.frames[r.frames['valid']]
        self.assertGreater(len(rows), 50)
        indices = np.rint((rows['start_s']-self.onset)*self.rate).astype(int)
        self.assertGreaterEqual(indices.min(), 0)
        self.assertLessEqual(indices.min(), 1, 'first complete data symbols lost in search')
        expected = np.array([8, 7, 5, 3, 1, 2, 4, 6])[indices % 8]-1
        np.testing.assert_array_equal(rows['symbol'], expected)
        self.assertGreater(r.first_tuple/self.fs, .49)
        self.assertIn('搜索范围', r.search_note)

    def test_half_second_noise_then_payload(self):
        spectrum = self.capture()
        original = self.path.read_bytes()
        r = recover_pwm(spectrum, symbol_hz=self.rate, auto_search=True)
        self.check(r)
        self.assertAlmostEqual((r.first_tuple+r.input_samples)/self.fs, 1., places=6)
        self.assertEqual(spectrum.first_tuple, 0)
        self.assertEqual(self.path.read_bytes(), original)

    def test_gui_delivers_search_result_and_original_timestamps(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import PWM_MODE
        self.capture(end=.53)
        view = headless_window()
        view.constellation_mode.set(PWM_MODE)
        view._analyze_worker(self.path,
            dict(nfft=1024, averages=1, sample_rate_hz=self.fs),
            dict(kind='pwm', duration_s=1., symbol_hz=self.rate, auto_search=True))
        view.poll()
        self.assertIsNotNone(view.recovery, view.status.get())
        self.check(view.recovery)
        self.assertIn('搜索范围', view.summary.get())

    def test_nonzero_requested_start_does_not_change_file_coordinates(self):
        spectrum = self.capture(end=.53)
        spectrum.first_tuple = round(.25*self.fs)
        r = recover_pwm(spectrum, symbol_hz=self.rate, auto_search=True, duration_s=.28)
        self.check(r)
        self.assertEqual(spectrum.first_tuple, round(.25*self.fs))

    def test_half_second_fixed_state_then_payload(self):
        r = recover_pwm(self.capture(prefix='cw'), symbol_hz=10000, auto_search=True)
        self.check(r)

    def test_onsets_inside_search_windows(self):
        for prefix in ('noise', 'cw'):
            for onset in (.50237, .50493):
                with self.subTest(prefix=prefix, onset=onset):
                    s = self.capture(prefix=prefix, onset=onset, end=.53)
                    r = recover_pwm(s, symbol_hz=10000, auto_search=True)
                    self.check(r)

    def test_search_reaches_past_old_sample_limit(self):
        r = recover_pwm(self.capture(fs=10000000, end=.515), symbol_hz=10000, auto_search=True)
        self.check(r)
        self.assertGreater(r.first_tuple, 1 << 22)

    def test_full_second_processing_exceeds_old_read_cap(self):
        s = self.capture(fs=5000000, end=1., onset=.000023)
        r = recover_pwm(s, symbol_hz=10000, auto_search=True)
        self.assertEqual(r.input_samples, 5000000)
        self.assertEqual(r.first_tuple, 0)
        rows = r.frames[r.frames['valid']]
        indices = np.rint((rows['start_s']-self.onset)*self.rate).astype(int)
        self.assertEqual(indices.min(), 0)
        self.assertGreater(indices.max(), 9900)
        expected = np.array([8, 7, 5, 3, 1, 2, 4, 6])[indices % 8]-1
        np.testing.assert_array_equal(rows['symbol'], expected)

    def test_no_payload_and_limited_search_and_cancel(self):
        for prefix in ('only_noise', 'only_cw'):
            s = self.capture(end=.025, prefix=prefix)
            with self.assertRaisesRegex(ValueError, '未找到有效PWM'):
                recover_pwm(s, symbol_hz=10000, auto_search=True)
        s = self.capture()
        with self.assertRaisesRegex(ValueError, '未找到有效PWM'):
            recover_pwm(s, symbol_hz=10000, auto_search=True, duration_s=.2)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AnalysisCancelled):
            recover_pwm(s, symbol_hz=10000, auto_search=True, cancel_event=cancel)


if __name__ == '__main__':
    unittest.main()

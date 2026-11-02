"""Mixed PWM header/QAM captures; short equal-energy headers and real boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam import recover_qam
from rx706_qam_pwm import acquire_pwm_header, validate_pwm_header
from rx706_qam_preamble import points_from_labels
from rx706_spectrum import analyze_bin, AnalysisCancelled
import threading


class MixedQAMTests(unittest.TestCase):
    def capture(self, directory, *, silence=5000, cfo=173, gap=100,
                sequence=(0, 0, 0), width_offset=0, pilots=(0, 0, 0), ook=False):
        fs, rate, period = 10000000, 100000, 100
        rng = np.random.default_rng(111)
        labels = rng.integers(0, 256, 600)
        pwm = []
        for k in sequence:
            high = round((5+k+width_offset)*4)
            pwm.extend([1]*high + [0 if ook else -1]*(period-high))
        qam = np.repeat(points_from_labels(np.r_[pilots, labels], 256), period)
        x = 700*np.r_[np.zeros(silence), pwm, np.zeros(gap), qam]
        x = x.astype(complex)*np.exp(1j*(2.3+2*np.pi*cfo*np.arange(len(x))/fs))
        raw = np.zeros((len(x), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        path = Path(directory)/'mixed.bin'
        raw.tofile(path)
        spectrum = analyze_bin(path, nfft=1024, averages=1, sample_rate_hz=fs)
        options = dict(duration_s=1, symbol_hz=rate, pulse='Rectangular',
            pwm_preamble_options=dict(symbols=' '.join(map(str, sequence)), gap_s=gap/fs,
                                      width_offset_chips=width_offset),
            preamble_options=dict(symbols=' '.join(map(str, pilots)), max_evm_percent=5))
        return spectrum, labels, options, (silence+len(pwm)+gap)/fs

    def test_short_antipodal_header_signed_cfo_and_qam_pilots(self):
        with tempfile.TemporaryDirectory() as directory:
            for cfo in (173, -317):
                with self.subTest(cfo=cfo):
                    spectrum, labels, options, boundary = self.capture(directory, cfo=cfo)
                    r = recover_qam(spectrum, **options)
                    self.assertAlmostEqual(r.pwm_preamble['qam_time_s'], boundary, delta=2/spectrum.sample_rate_hz)
                    self.assertEqual(r.pwm_preamble['symbols'], [0, 0, 0])
                    self.assertLess(r.pwm_preamble['error_percent'], 2)
                    self.assertAlmostEqual(r.preamble['total_frequency_hz'], cfo, delta=4)
                    np.testing.assert_array_equal(r.labels[:250], labels[:250])
                    self.assertGreater(r.time_s[0], boundary)

    def test_distinct_ook_header_width_offset_and_file_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum, labels, options, boundary = self.capture(directory, sequence=(0, 4, 8),
                                                              width_offset=-1, ook=True)
            spectrum.first_tuple = 2000
            r = recover_qam(spectrum, **options)
            self.assertAlmostEqual(r.pwm_preamble['time_s'], .0005, delta=2/spectrum.sample_rate_hz)
            self.assertAlmostEqual(r.pwm_preamble['qam_time_s'], boundary, delta=2/spectrum.sample_rate_hz)
            np.testing.assert_array_equal(r.labels[:250], labels[:250])

    def test_without_qam_pilot_keeps_first_data_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum, labels, options, boundary = self.capture(directory, cfo=0, pilots=())
            options['preamble_options'] = None
            r = recover_qam(spectrum, **options)
            self.assertIsNone(r.preamble)
            self.assertIsNotNone(r.pwm_preamble)
            self.assertGreaterEqual(r.time_s[0], boundary-1/spectrum.sample_rate_hz)
            self.assertLess(r.time_s[0], boundary+1/100000)
            self.assertLess(r.evm_percent, 3)
            # Blind phase has quadrant ambiguity, so compare with allowed rotations.
            expected = points_from_labels(labels[:250], 256)
            self.assertLess(min(np.mean(abs(r.scatter[:250]-expected*p)**2) for p in (1,1j,-1,-1j)), .001)

    def test_missing_pwm_and_qam_pilot_restricted_to_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            # Move a valid header outside the configured window.
            spectrum, _, options, _ = self.capture(directory, silence=20000)
            options['pwm_preamble_options']['search_s'] = .001
            with self.assertRaisesRegex(ValueError, '未找到匹配的PWM'):
                recover_qam(spectrum, **options)
            spectrum, _, options, _ = self.capture(directory, gap=20000)
            options['pwm_preamble_options']['gap_s'] = 0
            with self.assertRaisesRegex(ValueError, '未找到匹配的QAM'):
                recover_qam(spectrum, **options)

    def test_half_second_blank_and_fractional_samples_per_symbol(self):
        with tempfile.TemporaryDirectory() as directory:
            fs, pwm_rate, qam_rate = 3072000, 20000, 30000
            silence = int(.5*fs)
            pwm_size = int(np.ceil(3*fs/pwm_rate))
            qam_start = silence+pwm_size+100
            rng = np.random.default_rng(49)
            labels = rng.integers(0, 256, 400)
            qam_points = points_from_labels(np.r_[0, 0, 0, labels], 256)
            qam_indices = np.floor(np.arange(int(len(qam_points)*fs/qam_rate))*qam_rate/fs).astype(int)
            pwm_phase = (np.arange(pwm_size)*pwm_rate/fs) % 1
            pwm = np.where(pwm_phase < .2, 1., -1.)
            x = 700*np.r_[np.zeros(silence), pwm, np.zeros(100), qam_points[qam_indices]]
            x *= np.exp(1j*(2.3+2*np.pi*73*np.arange(len(x))/fs))
            raw = np.zeros((len(x), 4), dtype='<i2')
            raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
            path = Path(directory)/'late_mixed.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=1024, averages=1, sample_rate_hz=fs)
            r = recover_qam(spectrum, duration_s=1, symbol_hz=qam_rate, pulse='Rectangular',
                pwm_preamble_options=dict(symbol_hz=pwm_rate, gap_s=(qam_start-silence-3*fs/pwm_rate)/fs,
                                           max_frequency_hz=500),
                preamble_options=dict(symbols='0 0 0', max_evm_percent=5))
            self.assertAlmostEqual(r.pwm_preamble['time_s'], .5, delta=2/fs)
            self.assertAlmostEqual(r.pwm_preamble['qam_time_s'], qam_start/fs, delta=2/fs)
            np.testing.assert_array_equal(r.labels[:200], labels[:200])

    def test_validation_silence_and_cancellation(self):
        for change in [dict(symbols='0 0'), dict(symbols='0 13 0'),
                       dict(symbol_hz=1000000), dict(gap_s=-1),
                       dict(width_offset_chips=-5), dict(qam_search_symbols=0)]:
            with self.assertRaises(ValueError):
                validate_pwm_header(change, 10000000)
        with self.assertRaisesRegex(ValueError, '未找到'):
            acquire_pwm_header(np.zeros(10000, complex), 10000000, {})
        event = threading.Event()
        event.set()
        with self.assertRaises(AnalysisCancelled):
            acquire_pwm_header(np.zeros(10000, complex), 10000000, {}, event)

    def test_long_header_is_verified_after_frequency_correction(self):
        fs = 10000000
        sequence = [(i*5)%13 for i in range(96)]
        header = np.concatenate([np.r_[np.ones((5+k)*4), -np.ones(100-(5+k)*4)] for k in sequence])
        x = 700*np.r_[np.zeros(300), header, np.zeros(1000)].astype(complex)
        x *= np.exp(1j*(.7+2*np.pi*500*np.arange(len(x))/fs))
        result = acquire_pwm_header(x, fs, dict(symbols=' '.join(map(str, sequence))))
        self.assertAlmostEqual(result['index'], 300, delta=1)
        self.assertAlmostEqual(result['frequency_hz'], 500, delta=1)
        self.assertLess(result['error_percent'], 1)

    def test_mixed_gui_worker_delivery(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import QAM_PWM_MODE
        with tempfile.TemporaryDirectory() as directory:
            spectrum, _, options, _ = self.capture(directory)
            view = headless_window()
            view.constellation_mode.set(QAM_PWM_MODE)
            view._analyze_worker(spectrum.path, dict(nfft=1024, averages=1, sample_rate_hz=10000000),
                                 dict(kind='qam', **options))
            view.poll()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertIn('PWM前导候选', view.status.get())
            self.assertIn('PWM粗频偏', view.summary.get())

    def test_switching_between_qam_header_modes_reanalyzes(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import QAM_MODE, QAM_PWM_MODE
        view = headless_window()
        view.busy = False
        view.analyze = Mock()
        view.draw_constellation = Mock()
        view.recovery = SimpleNamespace(kind='qam', pwm_preamble=None)
        view.constellation_mode.set(QAM_PWM_MODE)
        view.on_mode_changed()
        view.analyze.assert_called_once()
        view.analyze.reset_mock()
        view.recovery.pwm_preamble = {}
        view.constellation_mode.set(QAM_MODE)
        view.on_mode_changed()
        view.analyze.assert_called_once()
        view.draw_constellation.assert_not_called()


if __name__ == '__main__':
    unittest.main()

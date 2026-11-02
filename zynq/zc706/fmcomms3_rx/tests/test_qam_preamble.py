"""Known pilots: mapping, delayed start, CFO, phase quadrant and GUI delivery."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam import recover_qam, decide
from rx706_qam_preamble import parse_preamble, points_from_labels, validate_preamble
from rx706_spectrum import analyze_bin


class PreambleTests(unittest.TestCase):
    def test_mapping_and_validation(self):
        for order in (16, 64, 256):
            labels = np.arange(order)
            points = points_from_labels(labels, order)
            np.testing.assert_array_equal(decide(points, order)[1], labels)
        np.testing.assert_allclose(parse_preamble('0 0b00000000 0x00', 'labels', 256),
                                   parse_preamble('-15,-15 -15,-15 -15,-15', 'iq', 256))
        for text, mode in [('1 2', 'labels'), ('0 256 0', 'labels'),
                           ('0 0 0', 'iq'), ('0,0 1,1 1,1', 'iq')]:
            with self.assertRaises(ValueError):
                parse_preamble(text, mode, 256)
        for change in [dict(max_frequency_hz=50000), dict(search_s=0),
                       dict(max_evm_percent=float('nan'))]:
            with self.assertRaises(ValueError):
                validate_preamble(dict(symbols='0 0 0', **change), 256, 100000)

    def capture(self, directory, pilot_labels, cfo=173, pulse='Rectangular', silence=4000):
        rng = np.random.default_rng(2026)
        labels = rng.integers(0, 256, 600)
        points = points_from_labels(np.r_[pilot_labels, labels], 256)
        fs, rate = 800000, 100000
        if pulse == 'Rectangular':
            signal = np.repeat(points, 8)
        else:
            n = 1 << (8*len(points)+255).bit_length()
            impulses = np.zeros(n, complex)
            impulses[128:128+8*len(points):8] = points
            f = abs(np.fft.fftfreq(n, 1/fs))
            raised = np.zeros(n)
            raised[f <= rate*.65/2] = 1
            transition = (f > rate*.65/2) & (f < rate*1.35/2)
            raised[transition] = .5*(1+np.cos(np.pi*(f[transition]-rate*.65/2)/(rate*.35)))
            signal = np.fft.ifft(np.fft.fft(impulses)*np.sqrt(raised))*8
        x = np.r_[np.zeros(silence), signal, np.zeros(200)]*700
        x *= np.exp(1j*(2.3+2*np.pi*cfo*np.arange(len(x))/fs))
        raw = np.zeros((len(x), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        path = Path(directory)/'pilot.bin'
        raw.tofile(path)
        spectrum = analyze_bin(path, nfft=1024, averages=1, sample_rate_hz=fs)
        return spectrum, labels, dict(duration_s=.1, order=256, symbol_hz=rate, pulse=pulse,
            preamble_options=dict(symbols=' '.join(map(str, pilot_labels)), max_frequency_hz=1000,
                                  max_evm_percent=5))

    def test_three_equal_symbols_delayed_signed_cfo(self):
        with tempfile.TemporaryDirectory() as directory:
            for cfo in (-317, 173):
                with self.subTest(cfo=cfo):
                    spectrum, labels, options = self.capture(directory, [0, 0, 0], cfo)
                    r = recover_qam(spectrum, **options)
                    self.assertAlmostEqual(r.preamble['frequency_hz'], cfo, delta=3)
                    self.assertLess(abs(r.preamble['time_s']-.005005), 2e-6)
                    np.testing.assert_array_equal(r.labels[:300], labels[:300])
                    self.assertEqual(r.preamble['length'], 3)
                    self.assertAlmostEqual(r.preamble['payload_time_s']-r.preamble['time_s'], 3/100000)

    def test_iq_pilots_at_capture_start_and_absolute_times(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum, labels, options = self.capture(directory, [0, 0, 0], 173, silence=0)
            options['preamble_options'].update(symbols='-15,-15 -15,-15 -15,-15', representation='iq')
            r = recover_qam(spectrum, **options)
            self.assertLess(r.preamble['time_s'], 1/100000)
            np.testing.assert_array_equal(r.labels[:300], labels[:300])
            # Reading from a different original tuple must preserve file timestamps.
            spectrum, labels, options = self.capture(directory, [0, 0, 0], 173)
            spectrum.first_tuple = 1600
            r = recover_qam(spectrum, **options)
            self.assertLess(abs(r.preamble['time_s']-.005005), 2e-6)
            np.testing.assert_array_equal(r.labels[:300], labels[:300])

    def test_rrc_distinct_pilots_and_gui(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import QAM_MODE
        with tempfile.TemporaryDirectory() as directory:
            pilots = [0, 17, 85, 255, 68, 204, 51, 170]*2
            spectrum, labels, options = self.capture(directory, pilots, -197, 'RRC')
            r = recover_qam(spectrum, **options)
            self.assertAlmostEqual(r.preamble['frequency_hz'], -197, delta=4)
            np.testing.assert_array_equal(r.labels[:250], labels[:250])
            view = headless_window()
            view.constellation_mode.set(QAM_MODE)
            view._analyze_worker(spectrum.path, dict(nfft=1024, averages=1, sample_rate_hz=800000),
                                 dict(kind='qam', **options))
            view.poll()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertIn('前导', view.status.get())
            self.assertIn('总校正频偏', view.summary.get())
            self.assertIn('Gray', view.summary.get())

    def test_missing_preamble_and_search_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum, _, options = self.capture(directory, [0, 0, 0])
            options['preamble_options']['search_s'] = .002
            with self.assertRaisesRegex(ValueError, '未找到'):
                recover_qam(spectrum, **options)
            # An all-zero input is not a perfectly fitting zero-gain preamble.
            np.zeros((10000, 4), dtype='<i2').tofile(spectrum.path)
            spectrum = analyze_bin(spectrum.path, nfft=1024, averages=1, sample_rate_hz=800000)
            with self.assertRaisesRegex(ValueError, '未找到'):
                recover_qam(spectrum, **options)


if __name__ == '__main__':
    unittest.main()

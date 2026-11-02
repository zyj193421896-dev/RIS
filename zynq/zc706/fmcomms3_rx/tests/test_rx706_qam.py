"""Synthetic QAM file recovery and unchanged GUI delivery contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam import recover_qam, decide


class QAMTests(unittest.TestCase):
    def test_rrc_capture(self):
        # Synthesize TX RRC using a square-root raised-cosine frequency response.
        from rx706_spectrum import analyze_bin
        rng = np.random.default_rng(19)
        n = 32768
        fs, rate, beta = 800000, 100000, .35
        impulses = np.zeros(n, complex)
        impulses[::8] = ((2*rng.integers(0,16,n//8)-15)
                        +1j*(2*rng.integers(0,16,n//8)-15))
        f = abs(np.fft.fftfreq(n, 1/fs))
        raised = np.zeros(n)
        raised[f <= rate*(1-beta)/2] = 1
        transition = (f > rate*(1-beta)/2) & (f < rate*(1+beta)/2)
        raised[transition] = .5*(1+np.cos(np.pi*(f[transition]-rate*(1-beta)/2)/(rate*beta)))
        x = np.fft.ifft(np.fft.fft(impulses)*np.sqrt(raised))*350
        x *= np.exp(.3j)
        raw = np.zeros((n,4), dtype='<i2')
        raw[:,0], raw[:,1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rrc.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=1024, averages=1, sample_rate_hz=fs)
            r = recover_qam(spectrum, duration_s=.05, symbol_hz=rate, order=256)
            self.assertLess(r.evm_percent, 3)

    def test_all_gray_labels(self):
        for order in (16, 64, 256):
            side = int(np.sqrt(order))
            levels = np.arange(-side+1, side, 2)/np.sqrt(2*(order-1)/3)
            points = (levels[:, None]+1j*levels).ravel()
            ideal, labels, bits = decide(points, order)
            np.testing.assert_allclose(ideal, points)
            self.assertEqual(len(np.unique(labels)), order)
            self.assertEqual(bits.shape, (order, int(np.log2(order))))

    def test_rectangular_capture_and_gui(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import QAM_MODE
        from rx706_spectrum import analyze_bin
        rng = np.random.default_rng(7)
        symbols = (2*rng.integers(0,16,1200)-15)+1j*(2*rng.integers(0,16,1200)-15)
        fs, rate, carrier = 800000, 100000, 40000
        x = 60*np.repeat(symbols, 8)
        x = x*np.exp(1j*(.2+2*np.pi*carrier*np.arange(len(x))/fs))
        raw = np.zeros((len(x),4), dtype='<i2')
        raw[:,0] = np.rint(x.real)
        raw[:,1] = np.rint(x.imag)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'qam.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=1024, averages=1, sample_rate_hz=fs)
            options = dict(duration_s=.02, symbol_hz=rate, center_hint_hz=carrier,
                           order=256, pulse='Rectangular')
            r = recover_qam(spectrum, **options)
            self.assertLess(r.evm_percent, 2)
            view = headless_window()
            view.constellation_mode.set(QAM_MODE)
            view._analyze_worker(path, dict(nfft=1024, averages=1, sample_rate_hz=fs), dict(kind='qam', **options))
            view.poll()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertEqual(view.recovery.kind, 'qam')
            self.assertIn('Gray', view.summary.get())
            with self.assertRaises(ValueError):
                recover_qam(spectrum, **dict(options, center_hint_hz=None))


if __name__ == '__main__':
    unittest.main()

"""TDCM pulse synthesis, not direct-IQ QAM, exercises harmonic extraction."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_tdcm import harmonic_coefficients, recover_tdcm, validate_tdcm
from rx706_qam_preamble import points_from_labels
from rx706_spectrum import analyze_bin, AnalysisCancelled


class TDCMTests(unittest.TestCase):
    def test_noninteger_windows_reject_large_dc_and_select_signed_harmonic(self):
        fs,rate = 15360000,19531.25
        n = int(20*fs/rate+100)
        # DC leakage exactly cancels even when a boundary lies inside a sample.
        z,centres = harmonic_coefficients(np.full(n,1000+300j),fs,13.75,rate,rate,count=16)
        self.assertLess(float(np.max(abs(z))),1e-9)
        self.assertAlmostEqual(centres[1]-centres[0],fs/rate)
        x = (2+3j)*np.exp(2j*np.pi*rate*np.arange(n)/fs)+900
        positive,_ = harmonic_coefficients(x,fs,0,rate,rate,count=16)
        negative,_ = harmonic_coefficients(x,fs,0,rate,-rate,count=16)
        np.testing.assert_allclose(positive,2+3j,atol=.02)
        self.assertLess(float(np.max(abs(negative))),.01)
        with self.assertRaises(ValueError):
            harmonic_coefficients(x,fs,n-5,rate,rate,count=2)

    def capture(self,directory,*,harmonic=1,pilots=(),cfo=173.4):
        fs,rate = 15360000,19531.25
        sps = fs/rate
        rng = np.random.default_rng(85)
        labels = rng.permutation(256)
        labels = np.r_[labels,labels]
        points = points_from_labels(np.r_[pilots,labels],256)
        radius = np.max(abs(points_from_labels(np.arange(256),256)))
        duty = np.arcsin(abs(points)/radius)/np.pi/abs(harmonic)
        delay = (-np.angle(points)/(2*np.pi*harmonic)-duty/2)%1
        silence = 3000
        qam_start = silence+int(np.ceil(3*sps))+300
        size = int(np.ceil(qam_start+len(points)*sps))
        x = np.zeros(size,complex)
        t = np.arange(int(np.ceil(3*sps)))/sps
        header_high = t%1 < .2
        delta = 600*np.exp(.4j)
        low = 200+150j
        x[silence:silence+len(t)] = low+delta*header_high
        phase = np.arange(size-qam_start)/sps
        symbol = np.minimum(phase.astype(int),len(points)-1)
        high = ((phase-symbol-delay[symbol])%1) < duty[symbol]
        x[qam_start:] = low+delta*high
        # Small deterministic receiver noise plus a nonzero RF carrier and CFO.
        x *= np.exp(2j*np.pi*(-200000+cfo)*np.arange(size)/fs)
        x += .15*(rng.normal(size=size)+1j*rng.normal(size=size))
        raw = np.zeros((size,4),dtype='<i2')
        raw[:,0],raw[:,1] = np.rint(x.real),np.rint(x.imag)
        path = Path(directory)/'tdcm.bin'
        raw.tofile(path)
        spectrum = analyze_bin(path,nfft=1024,averages=1,sample_rate_hz=fs)
        options = dict(duration_s=.1,symbol_hz=rate,center_hint_hz=-200000,
            tdcm_options=dict(base_hz=rate,harmonic=harmonic,payload_symbols=len(points)),
            pwm_preamble_options=dict(symbol_hz=rate,gap_s=(qam_start-silence-3*sps)/fs))
        return spectrum,labels,options,qam_start/fs

    def test_duty_delay_256qam_with_pwm_reference_and_cfo(self):
        with tempfile.TemporaryDirectory() as directory:
            for harmonic in (1,-1):
                with self.subTest(harmonic=harmonic):
                    spectrum,labels,options,start = self.capture(directory,harmonic=harmonic)
                    # Leave a fraction of a cell unused at EOF for boundary fitting.
                    options['tdcm_options']['payload_symbols'] = len(labels)-1
                    r = recover_tdcm(spectrum,**options)
                    np.testing.assert_array_equal(r.labels[:400],labels[:400])
                    self.assertLess(r.evm_percent,3)
                    self.assertAlmostEqual(r.pwm_preamble['frequency_hz'],173.4,delta=2)
                    self.assertAlmostEqual(r.pwm_preamble['qam_time_s'],start,delta=2/spectrum.sample_rate_hz)
                    self.assertAlmostEqual(r.time_s[0],start+.5/19531.25,delta=2/spectrum.sample_rate_hz)
                    self.assertAlmostEqual(r.tdcm['target_hz'],-200000+173.4+harmonic*19531.25,delta=2)

    def test_qam_training_works_on_coefficients_and_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            pilots = (0,17,85,255,68,204,51,170)*2
            spectrum,labels,options,start = self.capture(directory,pilots=pilots,cfo=-247.8)
            options['tdcm_options']['payload_symbols'] -= 1
            options['preamble_options'] = dict(symbols=' '.join(map(str,pilots)),max_evm_percent=5)
            r = recover_tdcm(spectrum,**options)
            self.assertEqual(r.preamble['length'],len(pilots))
            self.assertEqual(len(r.labels),len(labels)-1)
            np.testing.assert_array_equal(r.labels[:400],labels[:400])
            self.assertAlmostEqual(r.time_s[0],start+(len(pilots)+.5)/19531.25,delta=2/spectrum.sample_rate_hz)

    def test_validation_and_cancel(self):
        for settings in [dict(harmonic=0),dict(harmonic=1.5),dict(base_hz=18000),
                         dict(amplitude_scale=0),dict(payload_symbols=-1),dict(phase_deg=float('nan'))]:
            with self.assertRaises(ValueError):
                validate_tdcm(15360000,19531.25,settings)
        event = threading.Event();event.set()
        with self.assertRaises(AnalysisCancelled):
            harmonic_coefficients(np.zeros(10000),15360000,0,19531.25,19531.25,cancel_event=event)

    def test_worker_and_gui_mode_switch(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import TDCM_MODE,QAM_PWM_MODE
        with tempfile.TemporaryDirectory() as directory:
            spectrum,_,options,_ = self.capture(directory)
            options['tdcm_options']['payload_symbols'] -= 1
            view = headless_window()
            view.constellation_mode.set(TDCM_MODE)
            view._analyze_worker(spectrum.path,dict(nfft=1024,averages=1,sample_rate_hz=15360000),
                                 dict(kind='qam',**options))
            view.poll()
            self.assertIsNotNone(view.recovery,view.status.get())
            self.assertIn('TDCM +1',view.summary.get())
            self.assertIn('51.2',view.summary.get())
            self.assertIn('TDCM +1',view.iq_axes.get_title())
            self.assertIn('TDCM +1 harmonic',[line.get_label() for line in view.axes.lines])
            view.busy = False
            view.analyze = Mock()
            view.constellation_mode.set(QAM_PWM_MODE)
            view.on_mode_changed()
            view.analyze.assert_called_once()


if __name__=='__main__':unittest.main()

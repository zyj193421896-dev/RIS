"""Known phase/amplitude steps, CFO and negative controls for periodic recovery."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin, AnalysisCancelled
from rx706_recovery import recover_periodic,recover_tone

FS=15360000
CARRIER=1812345.67
CYCLE=100003.


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'capture.bin'

    def tearDown(self): self.tmp.cleanup()

    def write_signal(self, *, square=True, overload=False):
        count=round(FS*.08); t=np.arange(count)/FS
        state=(t*CYCLE+.17)%1>=.5
        a=np.where(state,100*np.exp(1j*np.deg2rad(177)),90) if square else np.full(count,100)
        rng=np.random.default_rng(9361)
        z=a*np.exp(2j*np.pi*CARRIER*t)+rng.normal(0,2,count)+1j*rng.normal(0,2,count)
        if overload: z*=40
        iq=np.rint(np.column_stack([z.real,z.imag,z.real,z.imag]))
        np.clip(iq,-2048,2047).astype('<i2').tofile(self.path)

    def test_phase_steps_survive_frequency_recovery_and_filtering(self):
        self.write_signal()
        s=analyze_bin(self.path,first_tuple=round(FS*.01),nfft=65536,averages=1)
        r=recover_periodic(s,cycle_hz=100000,duration_s=.05)
        self.assertAlmostEqual(r.carrier_hz,CARRIER,delta=1)
        self.assertAlmostEqual(r.cycle_hz,CYCLE,delta=1)
        self.assertAlmostEqual(abs(r.phase_difference_deg),177,delta=1)
        self.assertAlmostEqual(abs(r.amplitude_difference_db),20*np.log10(100/90),delta=.2)
        self.assertGreater(r.separation,10)
        self.assertGreater(r.heldout_clock_agreement,.99)
        self.assertLessEqual(len(r.scatter),8192)
        self.assertGreaterEqual(r.waveform_time_s[0],.01)
        self.assertEqual(r.output_rate_hz,FS/r.decimation)
        self.assertEqual(r.clipped,0)

    def test_single_tone_is_not_forced_into_two_states(self):
        self.write_signal(square=False)
        with self.assertRaises(ValueError):
            recover_periodic(analyze_bin(self.path),cycle_hz=100000,duration_s=.05)

    def test_fixed_tone_recovers_frequency_without_erasing_phase_changes(self):
        self.write_signal(square=False)
        s=analyze_bin(self.path,nfft=65536,averages=1)
        r=recover_tone(s,duration_s=.05)
        self.assertEqual(r.kind,'tone')
        self.assertAlmostEqual(r.carrier_hz,CARRIER,delta=.2)
        self.assertAlmostEqual(abs(r.centers[0]),100,delta=1)
        self.assertLess(r.phase_rms_deg,1)
        self.assertLessEqual(len(r.scatter),8192)
        # A phase step confined to the held-out half must remain after recovery.
        raw=np.fromfile(self.path,dtype='<i2').reshape(-1,4)
        first=round(FS*.03)
        signal=raw[first:,:2].astype(float)
        z=(signal[:,0]+1j*signal[:,1])*np.exp(1j*np.pi/4)
        raw[first:,0]=np.rint(z.real); raw[first:,1]=np.rint(z.imag)
        raw.tofile(self.path)
        changed=recover_tone(s,duration_s=.05,track_phase=False)
        self.assertGreater(changed.phase_rms_deg,20)

    def test_tone_slow_tracking_reduces_drift_but_retains_scatter(self):
        count=round(FS*.08); t=np.arange(count)/FS
        rng=np.random.default_rng(12)
        z=100*np.exp(2j*np.pi*(CARRIER*t+400*t*t))+rng.normal(0,2,count)+1j*rng.normal(0,2,count)
        np.rint(np.column_stack([z.real,z.imag,z.real,z.imag])).astype('<i2').tofile(self.path)
        s=analyze_bin(self.path,nfft=65536,averages=1)
        plain=recover_tone(s,duration_s=.06,track_phase=False)
        tracked=recover_tone(s,duration_s=.06,track_phase=True)
        self.assertGreater(plain.phase_rms_deg,20)
        self.assertLess(tracked.phase_rms_deg,plain.phase_rms_deg/5)
        self.assertGreater(np.std(tracked.scatter.imag),.01)

    def test_tone_mode_rejects_strong_modulation_and_noise(self):
        self.write_signal()
        with self.assertRaisesRegex(ValueError,'多个谱峰'):
            recover_tone(analyze_bin(self.path),duration_s=.05)
        count=round(FS*.04)
        noise=np.random.default_rng(9).normal(0,20,(count,4))
        np.rint(noise).astype('<i2').tofile(self.path)
        with self.assertRaises(ValueError):
            recover_tone(analyze_bin(self.path),duration_s=.04)

    def test_overload_bad_extra_samples_and_cancellation(self):
        self.write_signal(overload=True)
        with self.assertRaisesRegex(ValueError,'削顶'):
            recover_periodic(analyze_bin(self.path),duration_s=.05)
        self.write_signal()
        s=analyze_bin(self.path,nfft=65536,averages=1)
        with self.path.open('r+b') as stream:
            stream.seek(200000*8)
            stream.write(np.array([3000],dtype='<i2').tobytes())
        with self.assertRaisesRegex(ValueError,'12 位'):
            recover_periodic(s,duration_s=.05)
        event=threading.Event(); event.set()
        with self.assertRaises(AnalysisCancelled):
            recover_periodic(s,cancel_event=event)


if __name__=='__main__': unittest.main()

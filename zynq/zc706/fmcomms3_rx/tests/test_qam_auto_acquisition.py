"""Automatic carrier/clock uses known headers, never the random payload truth."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_qam10_frame import capture
from rx706_qam_acquisition import recover_auto_qam, _carrier_candidates
from rx706_spectrum import AnalysisCancelled


class AutoAcquisitionTests(unittest.TestCase):
    def run_case(self, spectrum, config, **kw):
        return recover_auto_qam(spectrum,duration_s=1,center_hint_hz=-800000,
            **{k:copy.deepcopy(config[k]) for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')},**kw)

    def test_wrong_hint_and_changing_payload_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,truth,lead=capture(d,frames=10);original=copy.deepcopy(c)
            result=self.run_case(s,c);a=result.diagnostics['auto_acquisition']
            self.assertAlmostEqual(a['carrier_hz'],120173.4,delta=4)
            self.assertAlmostEqual(a['period_ppm'],0,delta=2)
            self.assertFalse(a['payload_used']);self.assertEqual(c,original)
            self.assertEqual(len(result.labels),10*256)
            for frame in np.unique(result.frame_ids):
                k=int(round((result.frames[int(frame)]['header_time_s']*s.sample_rate_hz-lead)/(270*768)))
                np.testing.assert_array_equal(result.labels[result.frame_ids==frame],truth[k])
            self.assertIsNone(result.diagnostics['ber'])

    def test_negative_carrier_not_positive_harmonic(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,_,_=capture(d,frames=10)
            raw=np.fromfile(s.path,dtype='<i2').reshape(-1,4);raw[:,1]*=-1;raw.tofile(s.path)
            result=self.run_case(s,c);a=result.diagnostics['auto_acquisition']
            self.assertAlmostEqual(a['carrier_hz'],-120173.4,delta=4)
            self.assertEqual(len(result.labels),10*256)
            self.assertTrue(any(v['proposed_carrier_hz']>0 for v in a['candidate_trials']))

    def test_clock_from_headers_not_repeated_payload(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,_,_=capture(d,frames=30);raw=np.fromfile(s.path,dtype='<i2').reshape(-1,4)
            scale=1.00008;pos=np.arange(int(len(raw)*scale))/scale;warped=np.zeros((len(pos),4),dtype='<i2')
            for j in range(2):warped[:,j]=np.rint(np.interp(pos,np.arange(len(raw)),raw[:,j]))
            warped.tofile(s.path);s.total_tuples=len(warped)
            result=self.run_case(s,c);a=result.diagnostics['auto_acquisition']
            self.assertAlmostEqual(a['period_ppm'],80,delta=2)
            self.assertEqual(a['clock']['source'],'known_pwm_headers_only')
            self.assertEqual(len(result.labels),30*256)
            self.assertLess(a['clock']['calibration_headers'],30)

    def test_wrong_training_never_accepts_pwm_alone(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,_,_=capture(d,frames=10,wrong=True)
            with self.assertRaisesRegex(ValueError,'未找到同时匹配PWM与QAM训练'):self.run_case(s,c)

    def test_noise_rejected_and_cancelled(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,_,_=capture(d,frames=10);rng=np.random.default_rng(53)
            raw=np.rint(rng.normal(0,15,(s.total_tuples,4))).astype('<i2');raw.tofile(s.path)
            with self.assertRaisesRegex(ValueError,'未找到同时匹配PWM与QAM训练'):self.run_case(s,c)
            event=threading.Event();event.set()
            with self.assertRaises(AnalysisCancelled):self.run_case(s,c,cancel_event=event)

    def test_short_capture_does_not_invent_clock(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,_,_=capture(d,frames=5)
            with self.assertRaisesRegex(ValueError,'至少需要6个'):self.run_case(s,c)


    def test_time_varying_phase_from_known_headers(self):
        with tempfile.TemporaryDirectory() as d:
            s,c,truth,lead=capture(d,frames=30)
            raw=np.fromfile(s.path,dtype='<i2').reshape(-1,4);t=np.arange(len(raw))/s.sample_rate_hz
            signal=(raw[:,0]+1j*raw[:,1])*np.exp(2j*np.pi*(3*t+12*t*t))
            raw[:,0]=np.rint(signal.real);raw[:,1]=np.rint(signal.imag);raw.tofile(s.path)
            result=self.run_case(s,c);phase=result.diagnostics['phase_tracking']
            self.assertEqual(phase['source'],'known_pwm_state_contrast_only')
            self.assertFalse(phase['payload_used']);self.assertGreaterEqual(phase['header_count'],30)
            self.assertLess(phase['heldout_phase_error_p95_deg'],2)
            self.assertLess(result.evm_percent,3)
            for frame in np.unique(result.frame_ids):
                k=int(round((result.frames[int(frame)]['header_time_s']*s.sample_rate_hz-lead)/(270*768)))
                np.testing.assert_array_equal(result.labels[result.frame_ids==frame],truth[k])

    def test_inconsistent_header_phase_rejected(self):
        from rx706_tdcm_debug import _phase_track_headers
        from rx706_qam_pwm import validate_pwm_header,header_edges
        from rx706_qam10_profile import build_qam10_config
        fs=15360000.;pwm=validate_pwm_header(build_qam10_config(symbol_hz=20000)['pwm_preamble_options'],fs)
        period=fs/20000;frame=270*period;n=int(10*period)
        signal=np.zeros(int(8*frame+4),complex);h=np.zeros(n,bool)
        for a,b in zip(*header_edges(pwm,period)):h[a:b]=True
        for k in range(8):signal[int(k*frame):int(k*frame)+n]=(h-.5)*900*np.exp(1j*(2.5 if k%2 else 0))
        with self.assertRaisesRegex(ValueError,'相位矛盾'):
            _phase_track_headers(signal,fs,pwm,0,frame)


if __name__=='__main__':unittest.main()

"""Independent known-header phase tests; no payload truth enters the tracker."""
from pathlib import Path
import sys,unittest,tempfile,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_phase import pchip,pchip_slopes,joint_header_fit,resolve_header_phase

class KnownPhaseTests(unittest.TestCase):
    def test_pchip_linear_monotone_and_scipy_reference(self):
        from scipy.interpolate import PchipInterpolator
        t=np.array([0,.1,.23,.5,.9,1.]);y=np.array([1,1.2,.9,.9,1.5,2.])
        q=np.linspace(0,1,1001)
        np.testing.assert_allclose(pchip(t,y,q),PchipInterpolator(t,y)(q),atol=1e-12)
        np.testing.assert_allclose(pchip(t,3*t+2,q),3*q+2,atol=1e-12)
    def test_common_alias_and_curved_phase(self):
        dt=.0135;t=np.arange(80)*dt;truth=2*np.pi*(80*t+2*t*t)
        rng=np.random.default_rng(400);angles=np.angle(np.exp(1j*(truth+rng.normal(0,.008,len(t)))))
        phase,e=resolve_header_phase(t,angles,80+4*t+rng.normal(0,1,len(t)),np.full(len(t),1.5),dt)
        self.assertEqual(e['alias_branch'],1)
        self.assertLess(np.max(abs((phase-phase[0])-(truth-truth[0]))),.05)
        self.assertLess(e['heldout_phase_error_p95_deg'],3)
    def test_uncertainty_and_wrong_phase_rejected(self):
        dt=.0135;t=np.arange(60)*dt;phase=2*np.pi*10*t
        with self.assertRaisesRegex(ValueError,'不确定性'):
            resolve_header_phase(t,np.angle(np.exp(1j*phase)),np.full(len(t),10.),np.full(len(t),100.),dt)
        wrong=np.where(np.arange(len(t))%2,2.4,0.)
        with self.assertRaises(ValueError):resolve_header_phase(t,wrong,np.zeros(len(t)),np.ones(len(t)),dt)
        with self.assertRaisesRegex(ValueError,'歧义边界'):
            resolve_header_phase(t,np.angle(np.exp(1j*2*np.pi*36*t)),np.full(len(t),36.),np.ones(len(t)),dt)
    def test_short_and_missing_headers_rejected(self):
        t=np.arange(10)*.0135
        with self.assertRaises(ValueError):resolve_header_phase(t,t,t,t,.0135)
        t=np.r_[np.arange(20),np.arange(25,50)]*.0135
        with self.assertRaisesRegex(ValueError,'缺口'):resolve_header_phase(t,t,t,t,.0135)
    def test_missing_two_headers_cannot_hide_full_turn_slip(self):
        t=np.delete(np.arange(150)*.0135,[50,51]);phase=2*np.pi*20*t
        with self.assertRaisesRegex(ValueError,'缺口'):
            resolve_header_phase(t,np.angle(np.exp(1j*phase)),np.full(len(t),20.),np.ones(len(t)),.0135)

    def test_precise_local_alias_cannot_be_hidden_by_average(self):
        t=np.arange(150)*.0135;phase=2*np.pi*20*t;frequency=np.full(len(t),20.)
        frequency[50:52]+=1/.0135
        with self.assertRaisesRegex(ValueError,'局部频偏'):
            resolve_header_phase(t,np.angle(np.exp(1j*phase)),frequency,np.ones(len(t)),.0135)

    def test_bad_early_headers_are_not_hidden_by_later_holdout(self):
        t=np.arange(100)*.0135;phase=2*np.pi*10*t
        phase[:20]+=np.tile([-.3,.3],10)
        frequency=pchip_slopes(t,phase)/(2*np.pi)
        with self.assertRaisesRegex(ValueError,'留一验证'):
            resolve_header_phase(t,np.angle(np.exp(1j*phase)),frequency,np.ones(len(t)),.0135)

    def test_joint_fit_uses_known_samples_and_noise_uncertainty(self):
        from rx706_qam10_profile import build_qam10_config
        from rx706_qam_pwm import validate_pwm_header,header_edges
        fs=15360000.;period=768;o=validate_pwm_header(build_qam10_config(symbol_hz=20000)['pwm_preamble_options'],fs)
        high=np.zeros(7680,bool)
        for a,b in zip(*header_edges(o,period)):high[a:b]=True
        good=np.ones(len(high),bool);t=(np.arange(len(high))-len(high)/2)/fs
        z=(np.where(high,2+.3j,-.5+.1j))*np.exp(2j*np.pi*31.5*t)
        f,phase,error,mag,se=joint_header_fit(z,good,high,fs,period)
        self.assertAlmostEqual(f,31.5,places=5);self.assertLess(error,1e-7);self.assertLess(se,1e-5)
        rng=np.random.default_rng(471);noisy=z+.8*(rng.normal(size=len(z))+1j*rng.normal(size=len(z)))
        f,phase,error,mag,se=joint_header_fit(noisy,good,high,fs,period)
        self.assertGreater(se,0);self.assertLess(abs(f-31.5),5*se)
    def test_training_validation_cannot_cherry_pick_successes(self):
        from rx706_qam_phase import validate_training_phase
        eligible=np.arange(100);ids=np.r_[np.arange(20),np.arange(90,100)]
        with self.assertRaisesRegex(ValueError,'覆盖'):
            validate_training_phase(ids,eligible,np.zeros(len(ids)),20)
        ids=np.delete(eligible,[50,51,52])
        with self.assertRaisesRegex(ValueError,'缺口'):
            validate_training_phase(ids,eligible,np.zeros(len(ids)),20)
        bad=np.zeros(100);bad[:20]=np.tile([-.5,.5],10)
        with self.assertRaisesRegex(ValueError,'校准区'):
            validate_training_phase(eligible,eligible,bad,20)
        offset,p95,e=validate_training_phase(eligible,eligible,np.full(100,.7),20)
        self.assertAlmostEqual(offset,.7);self.assertLess(p95,1e-10)

    def test_low_snr_random_payload_has_independent_checks(self):
        from test_qam10_frame import capture
        from rx706_qam_acquisition import recover_auto_qam
        with tempfile.TemporaryDirectory() as d:
            s,c,truth,lead=capture(d,frames=40);before=copy.deepcopy(c)
            raw=np.fromfile(s.path,dtype='<i2').reshape(-1,4);t=np.arange(len(raw))/s.sample_rate_hz
            rng=np.random.default_rng(972)
            signal=(raw[:,0]+1j*raw[:,1])*np.exp(2j*np.pi*(3*t+8*t*t))
            signal=.25*signal+112.5*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
            raw[:,0]=np.rint(signal.real);raw[:,1]=np.rint(signal.imag);raw.tofile(s.path)
            r=recover_auto_qam(s,duration_s=1,**{k:c[k] for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')})
            p=r.diagnostics['phase_tracking']
            self.assertEqual(p['source'],'known_pwm_joint_fit_with_independent_qam_check')
            self.assertLess(p['heldout_phase_error_p95_deg'],np.degrees(.2))
            self.assertLess(p['training_phase_holdout_p95_deg'],np.degrees(.2))
            self.assertFalse(p['payload_used']);self.assertEqual(c,before)
            self.assertGreater(sum(f['valid'] for f in r.frames),30)
            self.assertIsNone(r.diagnostics['ber'])

if __name__=='__main__':unittest.main()

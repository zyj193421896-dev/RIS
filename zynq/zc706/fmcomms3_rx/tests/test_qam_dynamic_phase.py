"""10 kHz dynamic frequency: independent analytic truth and negative controls."""
from pathlib import Path
import sys,unittest,tempfile,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_phase import cubic_phase,cubic_phase_slopes,resolve_dynamic_header_phase,resolve_header_phase

class DynamicPhaseTests(unittest.TestCase):
    def observations(self):
        dt=.027;t=np.arange(74)*dt
        # Frequency crosses both +/- half-cycle boundaries, with no payload.
        truth=2*np.pi*(20*t+5*t*t+2.0*np.sin(2*np.pi*.6*t))
        freq=20+10*t+2.0*2*np.pi*.6*np.cos(2*np.pi*.6*t)
        return dt,t,truth,freq
    def test_spline_matches_independent_scipy_and_linear_reference(self):
        from scipy.interpolate import CubicSpline
        t=np.array([0,.1,.23,.5,.9,1.]);y=np.array([1,1.2,.9,.9,1.5,2.]);q=np.linspace(0,1,1001)
        np.testing.assert_allclose(cubic_phase(t,y,q),CubicSpline(t,y,bc_type='not-a-knot')(q),atol=1e-12)
        np.testing.assert_allclose(cubic_phase_slopes(t,y),CubicSpline(t,y,bc_type='not-a-knot')(t,1),atol=1e-12)
        np.testing.assert_allclose(cubic_phase(t,y+43*t+2,q),cubic_phase(t,y,q)+43*q+2,atol=1e-12)
        np.testing.assert_allclose(cubic_phase(t,3*t+2,[-.2,1.2]),[1.4,5.6],atol=1e-12)
    def test_high_precision_quadratic_endpoint_frequency(self):
        t=np.arange(36)*.027;y=2*np.pi*(15*t+50*t*t)
        np.testing.assert_allclose(cubic_phase_slopes(t,y)/(2*np.pi),15+100*t,atol=1e-10)
        for n in (2,3,4):
            t=np.arange(n)*.027;y=3*t+2
            np.testing.assert_allclose(cubic_phase_slopes(t,y),3,atol=1e-10)

    def test_chirp_crosses_wrapping_boundary_without_payload(self):
        dt,t,truth,f=self.observations();rng=np.random.default_rng(1101)
        angles=np.angle(np.exp(1j*(truth+rng.normal(0,.004,len(t)))))
        with self.assertRaises(ValueError):resolve_header_phase(t,angles,f,np.full(len(t),1.),dt)
        p,e=resolve_dynamic_header_phase(t,angles,f+rng.normal(0,.1,len(t)),np.full(len(t),1.),dt)
        np.testing.assert_allclose(p-p[0],truth-truth[0],atol=.025)
        self.assertEqual(e['tracking_strategy'],'known_header_dynamic_frequency')
        self.assertLess(e['max_interpolation_error_deg'],np.degrees(.2))
        self.assertFalse(e['payload_used'])
    def test_modest_local_frequency_tolerance_keeps_strict_alias_checks(self):
        dt=.027;t=np.arange(74)*dt;phase=np.zeros(len(t));f=np.zeros(len(t));se=np.ones(len(t))
        f[30]=5.49
        with self.assertRaisesRegex(ValueError,"局部频偏"):
            resolve_header_phase(t,phase,f,se,dt)
        p,e=resolve_dynamic_header_phase(t,phase,f,se,dt)
        np.testing.assert_allclose(p,0,atol=1e-12)
        self.assertEqual(e["local_frequency_sigma"],5.5)
        self.assertEqual(e["alias_uncertainty_sigma"],5.)
        self.assertEqual(len(e["locally_relaxed_headers"]),1)
        for offset in (5.51,1/dt):
            f[30]=offset
            with self.assertRaisesRegex(ValueError,"局部频偏"):
                resolve_dynamic_header_phase(t,phase,f,se,dt)
        with self.assertRaisesRegex(ValueError,"不确定性"):
            resolve_dynamic_header_phase(t,phase,np.zeros(len(t)),np.full(len(t),100.),dt)

    def test_frequency_reference_invariance(self):
        dt,t,p,f=self.observations()
        baseline,_=resolve_dynamic_header_phase(t,np.angle(np.exp(1j*p)),f,np.ones(len(t)),dt)
        for offset in (-200.,-17.,12.,100.):
            shifted,_=resolve_dynamic_header_phase(t,np.angle(np.exp(1j*(p-2*np.pi*offset*t))),f-offset,np.ones(len(t)),dt)
            np.testing.assert_allclose(shifted-shifted[0]+2*np.pi*offset*t,baseline-baseline[0],atol=1e-10)
    def test_missing_header_and_uncertain_branch_rejected(self):
        dt,t,p,f=self.observations();a=np.angle(np.exp(1j*p));keep=np.arange(len(t))!=30
        with self.assertRaisesRegex(ValueError,'缺口'):resolve_dynamic_header_phase(t[keep],a[keep],f[keep],np.ones(sum(keep)),dt)
        with self.assertRaisesRegex(ValueError,'不确定性'):resolve_dynamic_header_phase(t,a,f,np.full(len(t),100.),dt)
    def test_isolated_phase_jump_cannot_hide_under_percentile(self):
        dt=.027;t=np.arange(74)*dt;p=2*np.pi*(20*t+2*t*t);f=20+4*t
        p[37:]+=1.2
        with self.assertRaises(ValueError):resolve_dynamic_header_phase(t,np.angle(np.exp(1j*p)),f,np.full(len(t),10.),dt)
    def test_frequency_slip_and_false_local_measurement_rejected(self):
        dt,t,p,f=self.observations()
        f=f.copy();f[30:32]+=1/dt
        with self.assertRaisesRegex(ValueError,'局部频偏'):resolve_dynamic_header_phase(t,np.angle(np.exp(1j*p)),f,np.ones(len(t)),dt)
        t=np.arange(74)*dt;p=2*np.pi*(20*t+40*np.maximum(t-t[37],0));f=np.where(t<t[37],20.,60.)
        with self.assertRaises(ValueError):resolve_dynamic_header_phase(t,np.angle(np.exp(1j*p)),f,np.ones(len(t)),dt)
    def test_full_10khz_waveform_random_payload_and_wrong_training(self):
        from test_qam10_frame import capture
        from rx706_qam_acquisition import recover_auto_qam
        for wrong in (False,True):
            with self.subTest(wrong_training=wrong),tempfile.TemporaryDirectory() as directory:
                spectrum,c,truth,lead=capture(directory,frames=36,rate=10000,wrong=wrong)
                original=copy.deepcopy(c)
                raw=np.fromfile(spectrum.path,dtype='<i2').reshape(-1,4)
                t=np.arange(len(raw))/spectrum.sample_rate_hz
                # Independently imposed chirp; unknown payload only in assert.
                z=(raw[:,0]+1j*raw[:,1])*np.exp(2j*np.pi*(15*t+50*t*t))
                raw[:,0]=np.rint(z.real);raw[:,1]=np.rint(z.imag);raw.tofile(spectrum.path)
                kwargs={k:c[k] for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')}
                if wrong:
                    with self.assertRaises(ValueError):recover_auto_qam(spectrum,duration_s=1,**kwargs)
                    continue
                r=recover_auto_qam(spectrum,duration_s=1,**kwargs)
                e=r.diagnostics['phase_tracking']
                self.assertEqual(e.get('tracking_strategy'),'known_header_dynamic_frequency')
                self.assertFalse(e['payload_used']);self.assertEqual(c,original)
                self.assertLess(e['max_interpolation_error_deg'],np.degrees(.2))
                self.assertLess(e['training_phase_holdout_p95_deg'],np.degrees(.2))
                self.assertEqual(len(r.labels),35*256)
                self.assertEqual(e['boundary_erased_frames'],[35])
                self.assertFalse(r.frames[35]['valid'])
                self.assertFalse(e['payload_phase_extrapolation_allowed'])
                self.assertIsNone(r.diagnostics['ber'])
                for frame in np.unique(r.frame_ids):
                    k=int(round((r.frames[int(frame)]['header_time_s']*spectrum.sample_rate_hz-lead)/(270*spectrum.sample_rate_hz/10000)))
                    np.testing.assert_array_equal(r.labels[r.frame_ids==frame],truth[k])

    def test_dynamic_fallback_is_10khz_only_and_cancellation_propagates(self):
        from unittest.mock import patch
        import threading
        from rx706_qam_phase import track_known_headers
        from rx706_qam10_profile import build_qam10_config
        from rx706_qam_pwm import validate_pwm_header
        from rx706_spectrum import AnalysisCancelled
        fs=100000.;options=build_qam10_config()['pwm_preamble_options']
        options['symbol_hz']=1000.;options['observation_average_chips']=1.;options['max_frequency_hz']=100.
        pwm=validate_pwm_header(options,fs);frame=27000.;x=np.zeros(int(14*frame+2000),complex)
        for rate,expected in ((10000,True),(20000,False)):
            context=(np.array([1+1j]),1000,10,rate,rate,'auto',True)
            with patch('rx706_qam_phase.joint_header_fit',return_value=(0.,0.,1.,1.,1.)), \
                 patch('rx706_qam_phase.resolve_header_phase',side_effect=ValueError('static rejection')), \
                 patch('rx706_qam_phase.resolve_dynamic_header_phase',side_effect=ValueError('dynamic rejection')) as dynamic:
                with self.assertRaises(ValueError):track_known_headers(x,fs,pwm,0,frame,context)
                self.assertEqual(dynamic.called,expected)
        event=threading.Event();event.set()
        with self.assertRaises(AnalysisCancelled):track_known_headers(x,fs,pwm,0,frame,context,event)

    def test_random_phase_and_invalid_inputs_rejected(self):
        dt,t,p,f=self.observations();rng=np.random.default_rng(1102)
        with self.assertRaises(ValueError):resolve_dynamic_header_phase(t,rng.uniform(-np.pi,np.pi,len(t)),f,np.ones(len(t)),dt)
        with self.assertRaises(ValueError):resolve_dynamic_header_phase(t,p[:-1],f,np.ones(len(t)),dt)
        with self.assertRaises(ValueError):resolve_dynamic_header_phase(t,p,f,np.full(len(t),-1),dt)

if __name__=='__main__':unittest.main()

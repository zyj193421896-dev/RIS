"""Known PWM acquisition under coloured noise and sparse payload diagnostics."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_pwm import acquire_pwm_header
from rx706_qam_preamble import points_from_labels
from rx706_spectrum import analyze_bin
from rx706_tdcm import recover_tdcm


class PWMCandidateRecoveryTests(unittest.TestCase):
    def pwm(self, *, cfo=173.4, sequence=(0,0,0), interference=70):
        fs=3072000.;period=fs/19531.25;lead=500;length=round(len(sequence)*period)
        n=np.arange(lead+length+1000);cycle=(n-lead)/period
        index=np.clip(np.floor(cycle).astype(int),0,len(sequence)-1)
        high=(cycle-np.floor(cycle))<(5+np.array(sequence)[index])/25
        signal=np.where((n>=lead)&(n<lead+length),350*np.where(high,1.,-1.),0).astype(complex)
        signal*=np.exp(1j*(.7+2*np.pi*cfo*n/fs))
        return signal+interference*np.exp(-2j*np.pi*.09*n+1.1j),fs,lead

    def test_coloured_noise_does_not_become_a_false_large_cfo(self):
        x,fs,lead=self.pwm()
        r=acquire_pwm_header(x,fs,dict(symbol_hz=19531.25,search_s=.001))
        self.assertEqual(r['frequency_method'],'bounded_known_pwm_fit')
        self.assertAlmostEqual(r['index'],lead,delta=1)
        self.assertAlmostEqual(r['frequency_hz'],173.4,delta=10)
        self.assertLess(r['error_percent'],25)

    def test_wrong_header_and_out_of_range_frequency_remain_rejected(self):
        for parameters in (dict(sequence=(0,4,9)),dict(cfo=4500)):
            with self.subTest(parameters=parameters):
                x,fs,_=self.pwm(**parameters)
                with self.assertRaises(ValueError):
                    acquire_pwm_header(x,fs,dict(symbol_hz=19531.25,search_s=.001,max_frequency_hz=1000))
        rng=np.random.default_rng(144)
        with self.assertRaises(ValueError):
            acquire_pwm_header(rng.normal(size=5000)+1j*rng.normal(size=5000),3072000,
                               dict(symbol_hz=19531.25,search_s=.001))

    def capture(self,directory):
        fs=3072000.;nominal=19531.25;ppm=45.;sps=fs/nominal*(1+ppm*1e-6)
        labels=np.array([0,17,34,51,68,85,102,119,136,153,170,255]);points=points_from_labels(labels,256)
        radius=np.sqrt(2)*15/np.sqrt(170);duty=np.arcsin(abs(points)/radius)/np.pi
        delay=(-np.angle(points)/(2*np.pi)-duty/2)%1
        lead=1000;n=np.arange(int(lead+40*15*sps+30));cycle=(n-lead)/sps
        frame=np.floor(cycle/15).astype(int);slot=np.floor(cycle).astype(int)%15;fraction=cycle-np.floor(cycle)
        active=(frame>=0)&(frame<40);high=np.zeros(len(n),bool)
        hdr=active&(slot<3);high[hdr]=fraction[hdr]<.2
        data=active&(slot>=3);j=slot[data]-3;high[data]=((fraction[data]-delay[j])%1)<duty[j]
        x=np.where(active,700*np.exp(.4j)*(high-.5),0)*np.exp(2j*np.pi*(-200000+173.4)*n/fs)
        raw=np.zeros((len(x),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
        p=Path(directory)/'sparse.bin';raw.tofile(p)
        return analyze_bin(p,nfft=1024,averages=1,sample_rate_hz=fs),nominal

    def test_sparse_pwm_payload_returns_coefficients_without_claiming_dense_lock(self):
        with tempfile.TemporaryDirectory() as d:
            s,rate=self.capture(d)
            r=recover_tdcm(s,symbol_hz=rate,center_hint_hz=-200000,
                tdcm_options=dict(base_hz=rate,normalization='pwm',compensation=dict(enabled=True,
                    frame_symbols=12,frame_mode='all',clock_mode='repeat',phase_mode='square',calibration_frames=8)),
                pwm_preamble_options=dict(symbol_hz=rate))
            self.assertGreater(len(r.scatter),12*25)
            self.assertTrue(all(v['symbols']==12 for v in r.frames if v['valid']))
            self.assertFalse(r.diagnostics['structural_validation'])
            self.assertEqual(r.diagnostics['processing_stages']['quality_validation'],'sparse_repeat_verified_mapping_unverified')
            self.assertTrue(r.diagnostics['repeat_validation_passed'])
            self.assertTrue(r.diagnostics['processing_stages']['sparse_payload'])
            self.assertIsNone(r.diagnostics['ber'])
            self.assertLess(r.evm_percent,8)

    def test_single_short_frame_can_be_inspected_without_symbol_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            s,rate=self.capture(d)
            r=recover_tdcm(s,symbol_hz=rate,center_hint_hz=-200000,
                tdcm_options=dict(base_hz=rate,normalization='pwm',compensation=dict(enabled=True,
                    frame_symbols=12,frame_mode='single',clock_mode='manual',period_ppm=45,phase_mode='off')),
                pwm_preamble_options=dict(symbol_hz=rate))
            self.assertEqual(len(r.scatter),12)
            self.assertFalse(r.diagnostics['structural_validation'])
            self.assertTrue(any('不足16' in v for v in r.diagnostics['warnings']))


if __name__=='__main__':unittest.main()

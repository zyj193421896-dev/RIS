"""Independent waveform synthesis for exact PWM10 + corners + unknown QAM data."""
from pathlib import Path
import copy, json, math, sys, tempfile, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam10_profile import build_qam10_config,MT_PERCENT
from rx706_qam_config import validate_parameter_config
from rx706_qam_pwm import validate_pwm_header,acquire_pwm_header,header_pulse_fractions
from rx706_spectrum import analyze_bin
from rx706_qam import recover_qam


def capture(directory, *, quantized=False, mirrored=False, phase=.7, wrong=False, frames=5, rate=20000.):
    # Independent TX optimizer, not receiver decisions. No payload truth is passed
    # into the decoder. Each frame has a different deterministic random ordering.
    from tx_qam_fixture import Config,quantize_pulse,qam_symbol
    cfg=build_qam10_config(symbol_hz=rate, quantized_preamble=quantized)
    txcfg=Config(rows=1,cols=1,slots=256,repeats=1,pwm_mhz=rate/1e6,
                 gamma0_amplitude=0,gamma1_amplitude=1,gamma1_phase_deg=0)
    pulse=[]
    for label in range(256):
        z=qam_symbol(label)[2]
        if mirrored:z=z.conjugate()
        p,r,_,_=quantize_pulse(abs(z),np.angle(z),txcfg);pulse.append((r,p))
    pulse=np.array(pulse)
    fs=15360000.;sps=round(fs/rate);lead=1301
    rng=np.random.default_rng(294);payload=np.array([rng.permutation(256) for _ in range(frames)])
    train=np.array([0,8,0x88,0x80])
    if wrong: train=train[[0,2,1,3]]
    sent=np.column_stack((np.tile(train,(frames,1)),payload))
    n=np.arange(lead+frames*270*sps+77);local=(n-lead)/sps
    active=(local>=0)&(local<frames*270);slot=np.floor(local).astype(int)%270
    fi=np.floor(local/270).astype(int);phase_s=local-np.floor(local)
    high=np.zeros(len(n),bool)
    header=active&(slot<10)
    p=np.array(MT_PERCENT,float)[:,::-1]/100
    if quantized:p=np.floor(p*256+.5)/256
    high[header]=(phase_s[header]>=p[slot[header],0])&(phase_s[header]<p[slot[header]].sum(axis=1))
    data=active&(slot>=10);labels=sent[fi[data],slot[data]-10]
    q=np.floor(phase_s[data]*256+1e-7).astype(int)
    high[data]=((q-pulse[labels,0])%256)<pulse[labels,1]
    common=np.exp(1j*(2*np.pi*120173.4*n/fs+phase))
    x=np.where(active,900*(high-.5),0)*common
    x+=.2*(rng.normal(size=len(n))+1j*rng.normal(size=len(n)))
    raw=np.zeros((len(n),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
    path=Path(directory)/'qam10_synthetic.bin';raw.tofile(path)
    spectrum=analyze_bin(path,nfft=1024,averages=1,sample_rate_hz=fs)
    return spectrum,cfg,payload,lead


class QAM10Tests(unittest.TestCase):
    def test_geometry_and_metadata_roundtrip(self):
        c=build_qam10_config(symbol_hz=20000);self.assertEqual(c,validate_parameter_config(c))
        self.assertEqual(c['transmitter']['frame_symbols'],270)
        self.assertEqual(c['tdcm_options']['compensation']['frame_symbols'],260)
        p=validate_pwm_header(c['pwm_preamble_options'],15360000)
        np.testing.assert_allclose(header_pulse_fractions(p),np.array(MT_PERCENT)[:,::-1]/100)
        q=build_qam10_config(symbol_hz=20000, quantized_preamble=True)
        self.assertEqual(q['pwm_preamble_options']['pulse_chips'],
            [[128,64],[0,166],[77,102],[0,90],[51,115],[0,192],[26,64],[0,141],[102,64],[38,128]])
        for key in ('sample_rate_hz','input_path','input_fingerprint','center_hint_hz'):
            self.assertNotIn(key,c)

    def test_conflicting_metadata_and_invalid_pulse_rejected(self):
        for edit in (lambda c:c['transmitter'].update(frame_symbols=260),
                     lambda c:c['transmitter'].update(qam_cycles_per_symbol=3),
                     lambda c:c['pwm_preamble_options']['pulse_chips'].__setitem__(0,[21,5]),
                     lambda c:c['pwm_preamble_options'].update(symbols='65 8 54 2 42 10 21 6 60 35'),
                     lambda c:c['pwm_preamble_options'].update(unknown_pulse=3),
                     lambda c:c['pwm_preamble_options'].update(width_offset_chips=1),
                     lambda c:c['pwm_preamble_options'].update(total_chips=20.9)):
            c=build_qam10_config(symbol_hz=20000);edit(c)
            with self.assertRaises(ValueError):validate_parameter_config(c)

    def test_independent_data_and_iq_orientation(self):
        for case in (dict(),dict(mirrored=True,phase=-1.3),dict(quantized=True,phase=2.1)):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                s,c,truth,lead=capture(directory,**case)
                kwargs={k:copy.deepcopy(c[k]) for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')}
                r=recover_qam(s,duration_s=1,center_hint_hz=120000,**kwargs)
                self.assertEqual(r.preamble['expected_hex'],['00','08','88','80'])
                self.assertEqual(len(r.labels),5*256)
                self.assertEqual(len(r.training_records),5*4)
                self.assertLess(r.evm_percent,3)
                self.assertAlmostEqual(r.pwm_preamble['time_s'],lead/s.sample_rate_hz,delta=2/s.sample_rate_hz)
                self.assertEqual(r.preamble['conjugated'],case.get('mirrored',False))
                for frame in np.unique(r.frame_ids):
                    at=int(round((r.frames[int(frame)]['header_time_s']*s.sample_rate_hz-lead)/(270*768)))
                    np.testing.assert_array_equal(r.labels[r.frame_ids==frame],truth[at])
                self.assertIsNone(r.diagnostics['ber'])
                self.assertEqual(r.tdcm['payload_repeat'],1)
                self.assertEqual(r.preamble['frequency_hz'],0)
                self.assertEqual(r.pwm_preamble['header_kind'],'independent_pwm_sequence')

    def test_wrong_corner_order_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            s,c,_,_=capture(directory,wrong=True,frames=2)
            kwargs={k:c[k] for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')}
            with self.assertRaisesRegex(ValueError,'训练'):recover_qam(s,duration_s=1,center_hint_hz=120000,**kwargs)

if __name__=='__main__':unittest.main()

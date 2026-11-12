"""Independent PWM/TDCM synthesis: known corner calibration and payload HEX."""
from pathlib import Path
import sys,tempfile,unittest,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_qam_preamble import points_from_labels,parse_preamble
from rx706_qam_training import fit_training,normalize_training
from rx706_tdcm import recover_tdcm
from rx706_qam import decision_margins
from rx706_qam_debug_gui import export_symbols_csv,export_training_csv,export_report_html

CORNERS=np.array([0,8,0x88,0x80])
PAYLOAD=np.array([0x00,0x27,0x88,0xFA,0x34,0x91,0x56,0x80,0x08,0xCF,0x12,0xDD])


def capture(directory,*,mirrored=False,phase=.7,scale=1.,noise=.4,wrong=False,damaged=False,offset=False):
    fs=3072000.;nominal=19531.25;rate=nominal/(1+45e-6);sps=fs/rate;frames=32;lead=1000
    sent=np.r_[CORNERS,PAYLOAD]
    if wrong:sent[:4]=[0,0x88,8,0x80]
    tx=np.tile(points_from_labels(sent,256),(frames,1))
    if damaged:tx[:2,:4]=np.array([.11+.2j,.3-.12j,-.22+.03j,.04+.04j])
    if mirrored:tx=tx.conj()
    radius=np.sqrt(2)*15/np.sqrt(170);duty=np.arcsin(abs(tx)/radius)/np.pi
    delay=(-np.angle(tx)/(2*np.pi)-duty/2)%1
    n=np.arange(int(np.ceil(lead+frames*19*sps+20)));cycles=(n-lead)/sps
    frame=np.floor(cycles/19).astype(int);slot=np.floor(cycles).astype(int)%19;fraction=cycles-np.floor(cycles)
    active=(frame>=0)&(frame<frames);high=np.zeros(len(n),bool)
    hdr=active&(slot<3);high[hdr]=fraction[hdr]<.2
    data=active&(slot>=3);fi=frame[data];j=slot[data]-3
    high[data]=((fraction[data]-delay[fi,j])%1)<duty[fi,j]
    t=n/fs;common=2*np.pi*120173.4*t+phase+7*np.sin(2*np.pi*3*t)
    x=np.where(active,700*scale*(high-.5),0)*np.exp(1j*common)
    if offset:x+=np.where(active,9*scale*np.exp(1j*(common+2*np.pi*rate*t+.2)),0)
    rng=np.random.default_rng(174);x+=noise*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    raw=np.zeros((len(x),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
    p=Path(directory)/'corners.bin';raw.tofile(p);s=analyze_bin(p,nfft=1024,averages=1,sample_rate_hz=fs)
    options=dict(duration_s=1,center_hint_hz=120000,symbol_hz=nominal,order=256,
        tdcm_options=dict(base_hz=nominal,normalization='blind',compensation=dict(enabled=True,
            clock_mode='repeat',clock_search_ppm=1000,phase_mode='square',frame_mode='all',frame_symbols=16,max_frames=64,fit_timing=False)),
        pwm_preamble_options=dict(symbol_hz=nominal,qam_search_symbols=0),
        preamble_options=dict(symbols='00 08 88 80',representation='hex',orientation='auto',fit_dc_offset=True,max_evm_percent=8))
    return s,options


class FourCornerTrainingTests(unittest.TestCase):
    def test_actual_gray_corners_and_hex_parser(self):
        z=parse_preamble('00 08 88 80','hex',256)*np.sqrt(170)
        np.testing.assert_allclose(z,[-15-15j,-15+15j,15+15j,15-15j])
        self.assertEqual(points_from_labels([255],256)[0]*np.sqrt(170),5+5j)
        spacing=2/np.sqrt(170)
        self.assertAlmostEqual(decision_margins(points_from_labels([0x88],256),256)[0],spacing/2)
        self.assertAlmostEqual(decision_margins(np.array([14/np.sqrt(170)+15j/np.sqrt(170)]),256)[0],0)

    def test_complex_gain_offset_and_all_iq_symmetries(self):
        points=points_from_labels(CORNERS,256)
        for mirror in [False,True]:
            for phase in [0,np.pi/2,np.pi,-1.3]:
                with self.subTest(mirror=mirror,phase=phase):
                    gain=1.7*np.exp(1j*phase);bias=.2-.15j
                    samples=gain*(points.conj() if mirror else points)+bias
                    fit=fit_training(samples,points,19531.25,1000,8,orientation='auto',fit_dc_offset=True)
                    self.assertEqual(fit['conjugated'],mirror)
                    np.testing.assert_allclose(normalize_training(samples,fit),points,atol=1e-12)

    def test_nonidentifying_and_wrong_training_are_rejected(self):
        p=points_from_labels(CORNERS,256)
        for observed,expected in [(np.ones(4),np.ones(4)),(p[[0,2,1,3]],p)]:
            with self.assertRaises(ValueError):fit_training(observed,expected,19531.25,1000,8,orientation='auto',fit_dc_offset=True)
        with tempfile.TemporaryDirectory() as d:
            s,o=capture(d,wrong=True)
            with self.assertRaisesRegex(ValueError,'训练'):recover_tdcm(s,**o)

    def test_excess_training_noise_does_not_validate_a_mapping(self):
        rng=np.random.default_rng(631);points=points_from_labels(CORNERS,256)
        observed=points+2*(rng.normal(size=4)+1j*rng.normal(size=4))
        with self.assertRaises(ValueError):fit_training(observed,points,19531.25,1000,8,orientation='auto',fit_dc_offset=True)

    def test_per_frame_training_corrects_rotation_gain_mirror_and_slow_drift(self):
        for case in [dict(phase=2.4,scale=.7),dict(phase=-1.7,scale=1.3,mirrored=True),dict(phase=.3,offset=True)]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                s,o=capture(d,**case);r=recover_tdcm(s,**o)
                self.assertEqual(r.preamble['conjugated'],case.get('mirrored',False))
                self.assertEqual(r.preamble['expected_hex'],['00','08','88','80'])
                self.assertGreaterEqual(sum(v['valid'] for v in r.frames),28)
                for frame in np.unique(r.frame_ids)[2:-2]:
                    labels=r.labels[r.frame_ids==frame];np.testing.assert_array_equal(labels,PAYLOAD)
                self.assertEqual(len(r.scatter),12*sum(v['valid'] for v in r.frames))
                self.assertTrue(r.training_records);self.assertIsNone(r.diagnostics['ber'])

    def test_bad_early_training_is_not_used_to_calibrate_later_payload(self):
        with tempfile.TemporaryDirectory() as d:
            s,o=capture(d,damaged=True);r=recover_tdcm(s,**o)
            self.assertGreaterEqual(r.diagnostics['training_summary']['failed_frames'],2)
            self.assertGreater(len(r.scatter),12*20)
            self.assertTrue(any(not row['training_passed'] for row in r.training_records))

    def test_training_and_payload_exports_are_separate_uppercase_hex(self):
        with tempfile.TemporaryDirectory() as d:
            s,o=capture(d);r=recover_tdcm(s,**o);out=Path(d)
            export_symbols_csv(r,out/'payload.csv');export_training_csv(r,out/'training.csv');export_report_html(r,s,out/'report.html')
            with (out/'payload.csv').open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
            self.assertEqual(rows[1][-1],'candidate_hex');self.assertEqual(len(rows),len(r.scatter)+2)
            self.assertTrue(all(len(row[-1])==2 and row[-1]==row[-1].upper() for row in rows[2:]))
            with (out/'training.csv').open(encoding='utf-8-sig') as f:train=list(csv.reader(f))
            self.assertEqual(len(train),len(r.training_records)+2)
            page=(out/'report.html').read_text();self.assertIn('已知训练（与净荷分离）',page);self.assertEqual(page.count('data:image/png;base64,'),2)


if __name__=='__main__':unittest.main()

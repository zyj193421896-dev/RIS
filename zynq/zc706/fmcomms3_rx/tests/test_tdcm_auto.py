"""Independent transmitted pulses exercise bounded search and false acceptance."""
from pathlib import Path
import sys,tempfile,unittest,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin,AnalysisCancelled
from rx706_tdcm import recover_tdcm
from rx706_tdcm_auto import validate_search,AutoSearchError,replay_recipe
from rx706_qam_preamble import points_from_labels


def capture(directory,*,carrier=120173.4,ppm=350,order=256,ndata=256,frames=24,
            header=(0,0,0),rate=19531.25,ripple=8,repeat=True,tail_change=False,continuous=False,noise=False):
    fs=3072000.;period=fs/rate*(1+ppm*1e-6);frame=ndata+len(header);lead=3000
    rng=np.random.default_rng(321);all_labels=np.arange(order)
    all_points=points_from_labels(all_labels,order)
    if ndata<order:
        scale=np.sqrt(2*(order-1)/3);axis=(all_points.real*scale+np.sqrt(order)-1)/2
        other=(all_points.imag*scale+np.sqrt(order)-1)/2
        choose=(np.rint(axis+other).astype(int)%2)==0
        all_labels=all_labels[choose]
    labels=rng.permutation(np.resize(all_labels,ndata));points=points_from_labels(labels,order)
    if continuous:
        points=np.sqrt(rng.random(ndata))*np.exp(2j*np.pi*rng.random(ndata));points/=np.sqrt(np.mean(abs(points)**2))
    tx=np.tile(points,(frames,1))
    if not repeat:tx=points_from_labels(rng.integers(0,order,frames*ndata),order).reshape(frames,ndata)
    if tail_change:
        tx[int(frames*.65):]=points_from_labels(rng.integers(0,order,(frames-int(frames*.65))*ndata),order).reshape(-1,ndata)
    radius=np.sqrt(2)*(np.sqrt(order)-1)/np.sqrt(2*(order-1)/3)
    duty=np.arcsin(np.clip(abs(tx)/radius,0,1))/np.pi
    delay=(-np.angle(tx)/(2*np.pi)-duty/2)%1
    size=int(np.ceil(lead+frames*frame*period+30));n=np.arange(size);c=(n-lead)/period
    fi=np.floor(c/frame).astype(int);slot=np.floor(c).astype(int)%frame;within=c-np.floor(c)
    active=(fi>=0)&(fi<frames);indices=np.flatnonzero(active);fr=fi[indices];sl=slot[indices];p=within[indices]
    high=np.zeros(size,bool);hp=sl<len(header)
    widths=(5+np.array(header))/25
    high[indices[hp]]=p[hp]<widths[sl[hp]]
    data_slot=sl[~hp]-len(header);high[indices[~hp]]=((p[~hp]-delay[fr[~hp],data_slot])%1)<duty[fr[~hp],data_slot]
    t=n/fs;x=np.where(active,700*np.exp(.4j)*(high-.5),0)*np.exp(2j*np.pi*carrier*t+1j*ripple*np.sin(2*np.pi*2*t))
    x+=.7*(rng.normal(size=size)+1j*rng.normal(size=size))
    if noise:x=20*(rng.normal(size=size)+1j*rng.normal(size=size))
    raw=np.zeros((size,4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
    path=Path(directory)/'auto.bin';raw.tofile(path);spec=analyze_bin(path,nfft=1024,averages=1,sample_rate_hz=fs)
    options=dict(duration_s=1,center_hint_hz=-200000,symbol_hz=rate,order=order,
        tdcm_options=dict(base_hz=rate,normalization='blind',auto_search=dict(enabled=True,max_trials=6,max_carriers=2,
            carrier_span_hz=1000000 if carrier>300000 else 500000,frame_candidates='64 128 256 512'),
            compensation=dict(enabled=True,calibration_frames=8)),
        pwm_preamble_options=dict(symbol_hz=rate,symbols=' '.join(map(str,header))))
    return spec,options,labels,period


class AutoTests(unittest.TestCase):
    def test_varied_carrier_clock_and_frame_models(self):
        cases=[dict(carrier=120173.4,ppm=350),
               dict(carrier=-530247.8,ppm=-420,order=64,ndata=64,header=(0,4,9),rate=20000,ripple=0),
               dict(carrier=700005,ppm=80,order=256,ndata=128,header=(0,4,9,4,0),ripple=14)]
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                s,o,labels,period=capture(directory,**case);before=hashlib.sha256(s.path.read_bytes()).hexdigest()
                r=recover_tdcm(s,**o);a=r.diagnostics['auto_search']
                self.assertEqual(a['status'],'selected',[(v['status'],v.get('reason')) for v in a['candidates']])
                self.assertEqual(a['adopted']['frame_symbols'],case.get('ndata',256))
                self.assertAlmostEqual(a['adopted']['period_samples'],period,delta=.02)
                self.assertLess(a['final_test_metrics']['repeat_evm_percent'],8)
                self.assertLess(abs(a['adopted']['carrier_seed_hz']-case['carrier']),2000)
                self.assertIsNone(r.diagnostics['ber']);self.assertTrue(r.diagnostics['structural_validation'])
                self.assertEqual(before,hashlib.sha256(s.path.read_bytes()).hexdigest())
                expected=points_from_labels(labels,o['order']);ids=np.unique(r.frame_ids);z=r.scatter[r.frame_ids==ids[-3]]
                error=min(np.sqrt(np.mean(abs(z*np.exp(1j*k*np.pi/2)-expected)**2)) for k in range(4))
                self.assertLess(error,.12)

    def test_frozen_recipe_roundtrip_and_input_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            s,o,_,_=capture(directory,frames=16)
            r=recover_tdcm(s,**o);recipe=json.loads(json.dumps(r.diagnostics['auto_search']['recipe']))
            replay=replay_recipe(s,recipe)
            np.testing.assert_allclose(replay.scatter,r.scatter,atol=1e-10)
            self.assertIn('replay_recipe',replay.diagnostics)
            with s.path.open('r+b') as stream:stream.write(b'\x01\x00')
            with self.assertRaisesRegex(ValueError,'指纹'):replay_recipe(s,recipe)

    def test_nonrepeat_noise_and_nongrid_are_rejected(self):
        for case in [dict(repeat=False),dict(noise=True),dict(continuous=True)]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                s,o,_,_=capture(directory,frames=16,**case)
                with self.assertRaises(AutoSearchError):recover_tdcm(s,**o)

    def test_wrong_bounds_or_frame_models_fail(self):
        cases=[('carrier',dict(carrier=700000)),('clock',dict(ppm=2000)),('frame',{})]
        for kind,params in cases:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                s,o,_,_=capture(directory,frames=16,**params);search=o['tdcm_options']['auto_search']
                if kind=='carrier':search['carrier_span_hz']=100000
                if kind=='clock':search['clock_ppm']=100
                if kind=='frame':search['frame_candidates']='64 128'
                with self.assertRaises(AutoSearchError):recover_tdcm(s,**o)

    def test_final_holdout_failure_does_not_trigger_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            s,o,_,_=capture(directory,frames=24,tail_change=True)
            r=recover_tdcm(s,**o);a=r.diagnostics['auto_search']
            self.assertEqual(a['status'],'test_failed');self.assertFalse(r.diagnostics['structural_validation'])
            self.assertGreater(a['final_test_metrics']['repeat_evm_percent'],8)
            self.assertLess(a['selection_metrics']['repeat_evm_percent'],8)
            self.assertIn('没有使用该后段',r.diagnostics['warnings'][-1])

    def test_known_training_can_be_searched_without_claiming_payload_ber(self):
        with tempfile.TemporaryDirectory() as directory:
            s,o,labels,_=capture(directory,frames=20)
            o['preamble_options']=dict(symbols=' '.join(map(str,labels[:16])),max_evm_percent=8)
            r=recover_tdcm(s,**o)
            self.assertEqual(r.diagnostics['auto_search']['status'],'selected')
            self.assertEqual(r.diagnostics['auto_search']['adopted']['gain_mode'],'pilot')
            self.assertIsNone(r.diagnostics['ber'])
            last=r.labels[r.frame_ids==np.unique(r.frame_ids)[-3]]
            self.assertGreater(np.mean(last==labels[16:]),.98)

    def test_alias_frame_models_are_reported_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            s,o,_,_=capture(directory,frames=40)
            o['tdcm_options']['auto_search']['frame_candidates']='256 515'
            o['tdcm_options']['auto_search']['max_trials']=8
            r=recover_tdcm(s,**o)
            self.assertEqual(r.diagnostics['auto_search']['status'],'ambiguous')
            self.assertFalse(r.diagnostics['structural_validation'])

    def test_one_carrier_and_trial_budget_still_searches_and_declares_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            s,o,_,_=capture(directory,frames=16)
            o['tdcm_options']['auto_search'].update(max_carriers=1,max_trials=1,frame_candidates='256')
            r=recover_tdcm(s,**o);a=r.diagnostics['auto_search']
            self.assertEqual(a['status'],'selected');self.assertEqual(a['evaluated_trials'],1)
            self.assertTrue(a['budget_stopped']);self.assertLess(abs(a['adopted']['carrier_seed_hz']-120173.4),2000)

    def test_parsing_and_cancel(self):
        import threading
        for o in [dict(enabled='yes'),dict(frame_candidates='2 3'),dict(max_trials=0),dict(carrier_span_hz=float('inf'))]:
            with self.assertRaises(ValueError):validate_search(o)
        with tempfile.TemporaryDirectory() as directory:
            s,o,_,_=capture(directory,frames=12);event=threading.Event();event.set()
            with self.assertRaises(AnalysisCancelled):recover_tdcm(s,cancel_event=event,**o)

if __name__=='__main__':unittest.main()

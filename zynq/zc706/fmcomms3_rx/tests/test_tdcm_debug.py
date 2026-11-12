"""Known TDCM pulses: independent sample clock, changing CFO and heldout frames."""
from pathlib import Path
import sys,tempfile,threading,unittest,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_tdcm import recover_tdcm
from rx706_tdcm_debug import validate_compensation
from rx706_qam_preamble import points_from_labels
from rx706_spectrum import analyze_bin,AnalysisCancelled
from rx706_qam_debug_gui import export_symbols_csv,export_report_html,report_figures,scatter_indices


def synthetic(directory,*,repeat=True,frames=32,pilots=()):
    fs=3072000.;nominal=19531.25;rate=nominal/(1+45e-6);sps=fs/rate
    rng=np.random.default_rng(31);labels=rng.permutation(256)
    data=np.r_[np.asarray(pilots,dtype=int),labels];ndata=len(data);frame=ndata+3
    lead=3000;size=int(np.ceil(lead+frames*frame*sps+10))
    n=np.arange(size);cycle=(n-lead)/sps;fi=np.floor(cycle/frame).astype(int)
    slot=np.floor(cycle).astype(int)%frame;fraction=cycle-np.floor(cycle)
    all_labels=np.tile(data,(frames,1)) if repeat else rng.integers(0,256,(frames,ndata))
    points=points_from_labels(all_labels.ravel(),256).reshape(frames,ndata)
    radius=np.sqrt(2)*15/np.sqrt(170)
    duty=np.arcsin(abs(points)/radius)/np.pi;delay=(-np.angle(points)/(2*np.pi)-duty/2)%1
    active=(fi>=0)&(fi<frames);indices=np.flatnonzero(active);f=fi[indices];s=slot[indices];p=fraction[indices]
    high=np.zeros(size,bool);header=s<3;high[indices[header]]=p[header]<.2
    di=s[~header]-3;high[indices[~header]]=((p[~header]-delay[f[~header],di])%1)<duty[f[~header],di]
    t=n/fs;delta=700*np.exp(.4j)
    x=np.where(active,delta*(high-.5),0)*np.exp(2j*np.pi*(-200000+173.4)*t+12j*np.sin(2*np.pi*2*t))
    x+=.6*(rng.normal(size=size)+1j*rng.normal(size=size))
    raw=np.zeros((size,4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
    path=Path(directory)/'sro.bin';raw.tofile(path)
    spec=analyze_bin(path,nfft=1024,averages=1,sample_rate_hz=fs)
    debug=dict(enabled=True,clock_mode='repeat',phase_mode='square',frame_mode='all',fit_timing=not bool(pilots),frame_symbols=ndata,calibration_frames=10)
    options=dict(duration_s=1,symbol_hz=nominal,center_hint_hz=-200000,order=256,tdcm_options=dict(normalization='blind',compensation=debug),pwm_preamble_options=dict(symbol_hz=nominal,search_s=.05))
    if pilots:options['preamble_options']=dict(symbols=' '.join(map(str,pilots)),max_evm_percent=8)
    return spec,options,labels,rate


class CompensationTests(unittest.TestCase):
    def test_sro_and_varying_phase_recover_individual_payload_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,options,labels,rate=synthetic(directory)
            r=recover_tdcm(spec,**options);d=r.diagnostics
            self.assertAlmostEqual(d['period_ppm'],45,delta=.3)
            self.assertLess(d['heldout_template_evm_percent'],6)
            self.assertGreaterEqual(d['occupied_labels'],250)
            self.assertTrue(d['structural_validation']);self.assertIsNone(d['ber'])
            self.assertGreater(len(r.frames),25);self.assertEqual(len(r.scatter),256*sum(v['valid'] for v in r.frames))
            actual=r.scatter[r.frame_ids==np.unique(r.frame_ids)[-3]];expected=points_from_labels(labels,256)
            error=min(np.sqrt(np.mean(abs(actual*np.exp(1j*k*np.pi/2)-expected)**2)) for k in range(4))
            self.assertLess(error,.08);self.assertLess(abs(r.symbol_hz-rate),.01)
            self.assertTrue(np.all(np.diff(r.time_s)>0));self.assertEqual(spec.path.stat().st_size,spec.total_tuples*8)

    def test_real_training_removes_ambiguity_and_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            pilots=(0,17,85,255,68,204,51,170)*2
            spec,options,labels,_=synthetic(directory,pilots=pilots)
            r=recover_tdcm(spec,**options)
            self.assertIsNotNone(r.preamble);self.assertLess(r.preamble['evm_percent'],8)
            self.assertEqual(r.preamble['length'],len(pilots));self.assertEqual(len(r.scatter),256*sum(v['valid'] for v in r.frames))
            last=r.labels[r.frame_ids==np.unique(r.frame_ids)[-3]]
            self.assertGreater(np.mean(last==labels),.98)
            self.assertIn('训练',r.diagnostics['quality']);self.assertIsNone(r.diagnostics['ber'])

    def test_random_new_payload_is_not_accepted_as_repeated_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,options,_,_=synthetic(directory,repeat=False,frames=12)
            with self.assertRaisesRegex(ValueError,'重复|前导'):recover_tdcm(spec,**options)

    def test_plot_sampling_keeps_all_grid_labels_in_repeated_sweeps(self):
        labels=np.tile(np.arange(256),100)
        selected=scatter_indices(len(labels),12000,labels)
        self.assertEqual(len(selected),12000)
        self.assertEqual(len(np.unique(labels[selected])),256)
        self.assertEqual(len(np.unique(selected)),len(selected))

    def test_validation_and_cancel(self):
        for bad in [dict(period_ppm=float('nan')),dict(frame_symbols=0),dict(calibration_frames=0),dict(phase_mode='per_symbol'),dict(phase_update_hz=500,phase_bandwidth_hz=300),dict(timing_offset_symbols=1),dict(max_input_samples=1),dict(enabled='false')]:
            with self.assertRaises(ValueError):validate_compensation(bad)
        with tempfile.TemporaryDirectory() as directory:
            spec,options,_,_=synthetic(directory,frames=6);event=threading.Event();event.set()
            with self.assertRaises(AnalysisCancelled):recover_tdcm(spec,cancel_event=event,**options)

    def test_exports_are_candidates_and_html_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            spec,options,_,_=synthetic(directory,frames=12);r=recover_tdcm(spec,**options);out=Path(directory)
            export_symbols_csv(r,out/'symbols.csv');export_report_html(r,spec,out/'report.html')
            with (out/'symbols.csv').open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
            self.assertEqual(len(rows),len(r.scatter)+2);self.assertIn('未测BER',rows[0][0]);self.assertEqual(rows[1][7],'candidate_gray_label')
            page=(out/'report.html').read_text();self.assertEqual(page.count('data:image/png;base64,'),2);self.assertNotIn('src="http',page);self.assertIn('候选',page)
            self.assertEqual(len(report_figures(r,spec)),2)

if __name__=='__main__':unittest.main()

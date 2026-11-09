"""64QAM parity: all RX routes, controls, pilots, clock/phase and exports."""
from pathlib import Path
import sys,tempfile,unittest,json,csv,threading,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from qam64_fixtures import reference_points,direct_capture,tdcm_capture,compensation,save_capture
from rx706_qam import recover_qam,decide,decision_margins
from rx706_tdcm import recover_tdcm
from rx706_tdcm_auto import AutoSearchError,replay_recipe
from rx706_qam_profiles import four_corner_labels,four_corner_hex,mapping_note
from rx706_qam_preamble import parse_preamble
from rx706_qam_debug_gui import export_symbols_csv,export_training_csv,export_report_html,export_recipe_json,summary_text
from rx706_spectrum import AnalysisCancelled

CORNERS=(0,4,36,32)

class QAM64ModeTests(unittest.TestCase):
    def check_width(self,r):
        self.assertEqual(r.order,64);self.assertEqual(r.bits.shape,(len(r.labels),6))
        self.assertTrue(np.all((r.labels>=0)&(r.labels<64)))
        np.testing.assert_array_equal(np.sum(r.bits*2**np.arange(5,-1,-1),axis=1),r.labels)

    def test_mapping_corner_range_and_decision_boundaries(self):
        expected=reference_points(np.arange(64));np.testing.assert_array_equal(decide(expected,64)[1],np.arange(64))
        self.assertEqual(four_corner_labels(64),list(CORNERS));self.assertEqual(four_corner_hex(64),'00 04 24 20')
        self.assertEqual(four_corner_hex(256),'00 08 88 80')
        np.testing.assert_allclose(parse_preamble('00 04 24 20','hex',64),expected[list(CORNERS)])
        for invalid in ('00 08 88 80','00 04 40 20','00 04 FF 20'):
            with self.assertRaises(ValueError):parse_preamble(invalid,'hex',64)
        scale=np.sqrt(42);self.assertEqual(decision_margins(np.array([6/scale+7j/scale]),64)[0],0)
        self.assertAlmostEqual(decision_margins(expected[[36]],64)[0],1/scale)

    def test_direct_pulse_and_known_pwm_matrix(self):
        for pulse in ('Rectangular','RRC'):
            for pwm in (False,True):
                for known in (False,True):
                    with self.subTest(pulse=pulse,pwm=pwm,known=known),tempfile.TemporaryDirectory() as d:
                        s,o,labels=direct_capture(d,pulse=pulse,pwm=pwm,known=known,cfo=-173 if known else 0)
                        r=recover_qam(s,**o);self.check_width(r);self.assertLess(r.evm_percent,6)
                        if known:np.testing.assert_array_equal(r.labels[:300],labels[:300])
                        self.assertEqual(r.pwm_preamble is not None,pwm);self.assertEqual(r.preamble is not None,known)

    def test_tdcm_legacy_signed_harmonic_and_both_normalizations(self):
        for harmonic in (-1,1):
            for norm in ('blind','pwm'):
                with self.subTest(harmonic=harmonic,norm=norm),tempfile.TemporaryDirectory() as d:
                    s,o,payload,_=tdcm_capture(d,frames=2,ppm=0,ripple=0,harmonic=harmonic)
                    o['tdcm_options'].update(normalization=norm,payload_symbols=64)
                    r=recover_tdcm(s,**o);self.check_width(r);self.assertLess(r.evm_percent,5)
                    expected=reference_points(payload[0]);error=min(np.sqrt(np.mean(abs(r.scatter*1j**k-expected)**2)) for k in range(4))
                    self.assertLess(error,.1)

    def test_manual_repeat_and_continuous_phase_compensations(self):
        for clock,phase,fit in [('off','off',False),('manual','square',False),('repeat','mean',True),('repeat','square',True)]:
            with self.subTest(clock=clock,phase=phase),tempfile.TemporaryDirectory() as d:
                s,o,_,period=tdcm_capture(d,ppm=0 if clock=='off' else 45,ripple=0 if phase=='off' else 5)
                compensation(o,clock=clock,phase=phase,fit=fit)
                r=recover_tdcm(s,**o);self.check_width(r);self.assertIsNone(r.diagnostics['ber'])
                if clock=='repeat':self.assertLess(r.diagnostics['heldout_template_evm_percent'],8)
                else:self.assertLess(r.evm_percent,8)
                self.assertAlmostEqual(r.diagnostics['period_samples'],period,delta=.03)

    def test_four_corner_training_per_frame_mirror_offset_and_payload_split(self):
        for mirror in (False,True):
            with self.subTest(mirror=mirror),tempfile.TemporaryDirectory() as d:
                s,o,payload,_=tdcm_capture(d,ndata=12,pilots=CORNERS,mirror=mirror,dc_offset=True)
                compensation(o,ndata=12,pilots=4)
                r=recover_tdcm(s,**o);self.check_width(r)
                self.assertEqual(r.preamble['expected_hex'],['00','04','24','20']);self.assertEqual(r.preamble['conjugated'],mirror)
                for k in np.unique(r.frame_ids)[2:-2]:np.testing.assert_array_equal(r.labels[r.frame_ids==k],payload[0])
                self.assertEqual(len(r.scatter),12*sum(v['valid'] for v in r.frames))
                out=Path(d);export_symbols_csv(r,out/'payload.csv');export_training_csv(r,out/'training.csv');export_report_html(r,s,out/'report.html')
                with (out/'payload.csv').open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
                self.assertIn('高2位必须为0',rows[0][0]);self.assertTrue(all(len(v[8])==6 and int(v[-1],16)<64 for v in rows[2:]))
                self.assertIn('IIIQQQ',summary_text(r,s));page=(out/'report.html').read_text()
                self.assertEqual(page.count('data:image/png;base64,'),2);self.assertIn('已知训练（与净荷分离）',page)

    def test_auto_pwm_with_blind_or_training_and_frozen_recipe(self):
        for pilots,ndata in [((),64),(CORNERS,12)]:
            with self.subTest(pilots=pilots),tempfile.TemporaryDirectory() as d:
                s,o,payload,period=tdcm_capture(d,ndata=ndata,pilots=pilots,frames=40)
                compensation(o,ndata=ndata,pilots=len(pilots))
                o['tdcm_options']['auto_search']=dict(enabled=True,frame_candidates=str(ndata+len(pilots)),max_trials=6,max_carriers=1)
                r=recover_tdcm(s,**o);self.check_width(r);a=r.diagnostics['auto_search']
                self.assertEqual(a['status'],'selected');self.assertEqual(a['adopted']['bits_per_symbol'],6)
                self.assertAlmostEqual(a['adopted']['period_samples'],period,delta=.03)
                replay=replay_recipe(s,json.loads(json.dumps(a['recipe'])))
                np.testing.assert_allclose(replay.scatter,r.scatter,atol=1e-10)
                if pilots:
                    for k in np.unique(r.frame_ids)[2:-2]:np.testing.assert_array_equal(r.labels[r.frame_ids==k],payload[0])

    def test_headerless_repeat_auto_and_known_clock_random(self):
        for repeat in (True,False):
            with self.subTest(repeat=repeat),tempfile.TemporaryDirectory() as d:
                s,o,payload,period=tdcm_capture(d,header=False,repeat=repeat,frames=32)
                o.pop('pwm_preamble_options');compensation(o,clock='repeat' if repeat else 'manual')
                o['tdcm_options']['acquisition_mode']='no_header_repeat' if repeat else 'no_header_fixed'
                o['center_hint_hz']=120173.4
                o['tdcm_options']['auto_search']=dict(enabled=repeat,frame_candidates='32 64 128',max_carriers=1,max_trials=8)
                r=recover_tdcm(s,**o);self.check_width(r);a=r.diagnostics['acquisition']
                self.assertEqual(a['status'],'candidate_validated',a['final_test_metrics']);self.assertIsNone(r.pwm_preamble)
                self.assertEqual(a['adopted']['bits_per_symbol'],6);self.assertIsNone(r.diagnostics['ber'])
                out=Path(d)/'config.json';export_recipe_json(r,out)
                replay=recover_tdcm(s,tdcm_options=dict(replay_recipe=json.loads(out.read_text())))
                np.testing.assert_allclose(replay.scatter,r.scatter,atol=1e-10)
                if not repeat:
                    idx=np.floor((r.time_s*s.sample_rate_hz-1000)/period).astype(int)
                    expected=reference_points(payload.ravel()[idx])
                    self.assertLess(min(np.sqrt(np.mean(abs(r.scatter*1j**k-expected)**2)) for k in range(4)),.1)

    def test_legacy_three_pwm_single_training_payload_boundaries(self):
        # Legacy single-pass mode remains available alongside current double-pass mode.
        # N is independent of modulation order; retain a 256QAM counterpart.
        cases=[(64,(0,4,36,32),12),(64,(0,4,36,32),64),(256,(0,8,136,128),12)]
        for order,pilots,ndata in cases:
            with self.subTest(order=order,ndata=ndata),tempfile.TemporaryDirectory() as d:
                s,o,payload,_=tdcm_capture(d,order=order,ndata=ndata,pilots=pilots,frames=32)
                compensation(o,ndata=ndata,pilots=4)
                r=recover_tdcm(s,**o)
                self.assertEqual(r.pwm_preamble['symbols'],[0,0,0])
                self.assertEqual(r.preamble['expected_hex'],[f'{v:02X}' for v in pilots])
                self.assertEqual(r.bits.shape[1],6 if order==64 else 8)
                self.assertEqual(len(r.scatter),ndata*sum(v['valid'] for v in r.frames))
                for frame in np.unique(r.frame_ids)[2:-2]:
                    ids=np.flatnonzero(r.frame_ids==frame);row=next(v for v in r.frames if v['index']==frame)
                    np.testing.assert_array_equal(r.labels[ids],payload[0])
                    self.assertEqual(len(ids),ndata)
                    self.assertAlmostEqual((row['data_time_s']-row['header_time_s'])*r.symbol_hz,3,delta=.01)
                    self.assertAlmostEqual((r.time_s[ids[0]]-row['header_time_s'])*r.symbol_hz,7.5,delta=.02)
                    self.assertEqual(sum(t['frame']==frame for t in r.training_records),4)
                self.assertIsNone(r.diagnostics['ber'])

    def test_sparse_pwm_and_single_short_frame_remain_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=tdcm_capture(d,ndata=12,ppm=45,ripple=0)
            compensation(o,ndata=12,clock='repeat',phase='square')
            o['tdcm_options']['normalization']='pwm'
            r=recover_tdcm(s,**o);self.check_width(r)
            self.assertGreater(len(r.scatter),12*20);self.assertFalse(r.diagnostics['structural_validation'])
            self.assertTrue(r.diagnostics['repeat_validation_passed'])
            self.assertTrue(r.diagnostics['processing_stages']['sparse_payload'])
            self.assertLess(r.evm_percent,8)
            o['tdcm_options']['compensation'].update(frame_mode='single',clock_mode='manual',period_ppm=45,phase_mode='off')
            r=recover_tdcm(s,**o);self.assertEqual(len(r.scatter),12)
            self.assertFalse(r.diagnostics['structural_validation']);self.assertIsNone(r.diagnostics['ber'])

    def test_multiple_pwm_periods_per_symbol_and_training_noise_boundary(self):
        from rx706_qam_training import fit_training,normalize_training
        with tempfile.TemporaryDirectory() as d:
            s,o,payload,period=tdcm_capture(d,ndata=12,pilots=CORNERS,cycles=3,pwm_rate=58593.75)
            compensation(o,ndata=12,pilots=4)
            r=recover_tdcm(s,**o);self.check_width(r);self.assertEqual(r.tdcm['cycles_per_symbol'],3)
            for k in np.unique(r.frame_ids)[2:-2]:np.testing.assert_array_equal(r.labels[r.frame_ids==k],payload[0])
        points=reference_points(CORNERS)
        for mirror in (False,True):
            for phase in (0,np.pi/2,np.pi,-np.pi/2):
                samples=1.7*np.exp(1j*phase)*(points.conj() if mirror else points)+.1-.2j
                fit=fit_training(samples,points,100000,1000,8,orientation='auto',fit_dc_offset=True)
                self.assertEqual(fit['conjugated'],mirror)
                np.testing.assert_allclose(normalize_training(samples,fit),points,atol=1e-12)
        rng=np.random.default_rng(741)
        with self.assertRaises(ValueError):fit_training(points+2*(rng.normal(size=4)+1j*rng.normal(size=4)),points,100000,1000,8,orientation='auto',fit_dc_offset=True)

    def test_wrong_training_noise_nonrepeat_and_cancellation(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=tdcm_capture(d,ndata=12,pilots=CORNERS,bad_training=True)
            compensation(o,ndata=12,pilots=4)
            with self.assertRaises(ValueError):recover_tdcm(s,**o)
        for repeat in (False,True):
            with self.subTest(repeat=repeat),tempfile.TemporaryDirectory() as d:
                s,o,_,_=tdcm_capture(d,header=False,repeat=repeat,frames=16)
                o.pop('pwm_preamble_options');compensation(o)
                o['tdcm_options']['acquisition_mode']='no_header_repeat'
                if repeat:
                    rng=np.random.default_rng(111);save_capture(d,'tdcm.bin',30*(rng.normal(size=s.total_tuples)+1j*rng.normal(size=s.total_tuples)),s.sample_rate_hz)
                with self.assertRaises(ValueError):recover_tdcm(s,**o)
                e=threading.Event();e.set()
                with self.assertRaises(AnalysisCancelled):recover_tdcm(s,**o,cancel_event=e)

if __name__=='__main__':unittest.main()

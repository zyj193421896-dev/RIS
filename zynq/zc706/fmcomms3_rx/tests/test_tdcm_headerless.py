"""Independent two-state pulse generation, including random payload and controls."""
from pathlib import Path
import sys,tempfile,unittest,json,threading
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from test_tdcm_auto import capture
from rx706_tdcm import recover_tdcm
from rx706_tdcm_auto import AutoSearchError
from rx706_qam_preamble import points_from_labels
from rx706_tdcm_headerless import validate_acquisition
from rx706_spectrum import AnalysisCancelled
from rx706_qam_debug_gui import export_report_html,export_recipe_json


def setup_capture(directory,*,mode='no_header_repeat',auto=True,**parameters):
    s,o,labels,period=capture(directory,header=(),**parameters)
    td=o['tdcm_options'];td['acquisition_mode']=mode
    td['compensation'].update(phase_mode='square',clock_mode='repeat' if mode=='no_header_repeat' else 'manual',
        clock_search_ppm=1000,frame_symbols=parameters.get('ndata',256),period_ppm=parameters.get('ppm',350))
    td['auto_search']['enabled']=auto
    if auto:td['auto_search'].update(max_trials=12,max_carriers=1)
    o.pop('pwm_preamble_options',None)
    if not auto:o['center_hint_hz']=parameters.get('carrier',120173.4)
    return s,o,labels,period


class HeaderlessTests(unittest.TestCase):
    def test_repeated_no_header_varied_carrier_clock_and_size(self):
        for case in [dict(carrier=120173.4,ppm=350),dict(carrier=-530247.8,ppm=-420,ndata=64,order=64,rate=20000,ripple=0),dict(carrier=700005,ppm=80,ndata=128,ripple=14)]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                s,o,labels,period=setup_capture(d,**case);r=recover_tdcm(s,**o);a=r.diagnostics['acquisition']
                self.assertEqual(a['status'],'candidate_validated',a['final_test_metrics'])
                self.assertAlmostEqual(r.diagnostics['period_samples'],period,delta=.02)
                self.assertEqual(a['adopted']['block_symbols'],case.get('ndata',256))
                self.assertIsNone(r.pwm_preamble);self.assertIsNone(r.preamble);self.assertIsNone(r.diagnostics['ber'])
                expected=points_from_labels(labels,o['order']);z=r.scatter[r.frame_ids==np.unique(r.frame_ids)[-3]]
                error=min(np.sqrt(np.mean(abs(np.roll(z,j)*np.exp(1j*k*np.pi/2)-expected)**2)) for k in range(4) for j in range(len(expected)))
                self.assertLess(error,.12)

    def test_random_payload_supported_with_known_clock_not_as_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,period=setup_capture(d,mode='no_header_fixed',auto=False,repeat=False,ripple=5)
            r=recover_tdcm(s,**o)
            self.assertEqual(r.diagnostics['acquisition']['status'],'candidate_validated',r.diagnostics['acquisition']['final_test_metrics'])
            self.assertIsNone(r.diagnostics['heldout_template_evm_percent'])
            rng=np.random.default_rng(321);rng.permutation(np.arange(256));truth=rng.integers(0,256,(24,256)).ravel()
            index=np.floor((r.time_s*s.sample_rate_hz-3000)/period).astype(int)
            expected=points_from_labels(truth[index],256)
            error=min(np.sqrt(np.mean(abs(r.scatter*np.exp(1j*k*np.pi/2)-expected)**2)) for k in range(4))
            self.assertLess(error,.12)
            o['tdcm_options']['acquisition_mode']='no_header_repeat';o['tdcm_options']['compensation']['clock_mode']='repeat'
            with self.assertRaises(AutoSearchError):recover_tdcm(s,**o)

    def test_noise_and_continuous_nongrid_rejected(self):
        for case in [dict(noise=True),dict(continuous=True)]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                s,o,_,_=setup_capture(d,frames=16,**case)
                with self.assertRaises(AutoSearchError):recover_tdcm(s,**o)

    def test_final_payload_change_fails_repeat_without_reselection(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=setup_capture(d,tail_change=True)
            r=recover_tdcm(s,**o)
            self.assertEqual(r.diagnostics['acquisition']['status'],'test_failed')
            self.assertFalse(r.diagnostics['structural_validation'])

    def test_wrong_clock_for_random_data_fails_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=setup_capture(d,mode='no_header_fixed',auto=False,repeat=False,ppm=700)
            o['tdcm_options']['compensation']['period_ppm']=0
            try:r=recover_tdcm(s,**o)
            except AutoSearchError:return
            self.assertFalse(r.diagnostics['structural_validation'])

    def test_zero_and_single_tone_are_not_false_locks(self):
        for kind in ('zero','tone'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                s,o,_,_=setup_capture(d,auto=False,frames=12)
                raw=np.memmap(s.path,mode='r+',dtype='<i2').reshape(-1,4)
                raw[:]=0
                if kind=='tone':
                    v=500*np.exp(2j*np.pi*o['center_hint_hz']*np.arange(len(raw))/s.sample_rate_hz)
                    raw[:,0]=np.rint(v.real);raw[:,1]=np.rint(v.imag)
                raw.flush();del raw
                with self.assertRaises((AutoSearchError,ValueError)):recover_tdcm(s,**o)

    def test_config_roundtrip_guard_and_self_contained_report(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=setup_capture(d,auto=False,frames=20);r=recover_tdcm(s,**o)
            out=Path(d);export_recipe_json(r,out/'config.json');recipe=json.loads((out/'config.json').read_text())
            replay=recover_tdcm(s,tdcm_options=dict(replay_recipe=recipe))
            np.testing.assert_allclose(r.scatter,replay.scatter,atol=1e-10)
            export_report_html(r,s,out/'report.html');text=(out/'report.html').read_text()
            self.assertEqual(text.count('data:image/png;base64,'),2);self.assertIn('无PWM获取',text)
            with s.path.open('r+b') as stream:stream.write(b'\x01\x00')
            with self.assertRaisesRegex(ValueError,'指纹'):recover_tdcm(s,tdcm_options=dict(replay_recipe=recipe))

    def test_validation_cancel_and_no_fake_pilot(self):
        with self.assertRaises(ValueError):validate_acquisition('anything')
        with tempfile.TemporaryDirectory() as d:
            s,o,_,_=setup_capture(d,auto=False,frames=12)
            with self.assertRaisesRegex(ValueError,'训练'):recover_tdcm(s,**dict(o,preamble_options=dict(symbols='0 0 0')))
            event=threading.Event();event.set()
            with self.assertRaises(AnalysisCancelled):recover_tdcm(s,**o,cancel_event=event)

if __name__=='__main__':unittest.main()

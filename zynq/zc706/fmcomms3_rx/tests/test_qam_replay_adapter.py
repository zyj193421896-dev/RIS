"""Strict bound-replay contract: no silent settings loss, no GUI/hardware."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys,tempfile,threading,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_replay_adapter import validate_replay,load_replay,run_replay
from rx706_spectrum import AnalysisCancelled
RECIPE=Path(__file__).resolve().parent/'fixtures/qam_replay_synthetic.json'

class ReplayAdapterTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        iq=Path(self.temp.name)/'synthetic_qam_replay.bin'
        content=b'\0'*(65536*8);iq.write_bytes(content)
        self.recipe=json.loads(RECIPE.read_text())
        self.recipe['input_path']=str(iq)
        self.recipe['input_fingerprint']=dict(file_bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    def test_idempotent_and_unknown_fields_rejected_at_every_level(self):
        valid=validate_replay(self.recipe);self.assertEqual(valid,validate_replay(valid))
        for path in [(),('spectrum',),('recovery',),('input_fingerprint',),('recovery','tdcm_options'),
                     ('recovery','tdcm_options','compensation'),('recovery','tdcm_options','crc_debug'),
                     ('recovery','pwm_preamble_options'),('recovery','preamble_options')]:
            with self.subTest(path=path):
                r=copy.deepcopy(self.recipe);v=r
                for part in path:v=v[part]
                v['unsupported_option']=123
                with self.assertRaisesRegex(ValueError,'未知字段'):validate_replay(r)
    def test_loader_checks_raw_fields_before_normalization(self):
        r=copy.deepcopy(self.recipe);r['recovery']['tdcm_options']['crc_debug']['secret_algorithm']='unknown'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps(r))
            with patch('rx706_qam_replay_adapter.verify_input') as verify:
                with self.assertRaises(ValueError):load_replay(p)
                verify.assert_not_called()
    def test_mismatched_path_sha_order_range_and_method_fail(self):
        with self.assertRaises(ValueError):validate_replay(self.recipe,'/tmp/other.bin')
        for section,key,value in [(None,'method','bounded_repeated_tdcm'),('recovery','order',64),('spectrum','first_tuple',999999999)]:
            r=copy.deepcopy(self.recipe);(r if section is None else r[section])[key]=value
            with self.assertRaises(ValueError):validate_replay(r)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';r=copy.deepcopy(self.recipe);r['input_fingerprint']['sha256']='0'*64;p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'SHA-256'):load_replay(p)
    def test_exact_options_forwarded_and_sha_checked_before_after(self):
        r=validate_replay(self.recipe);s=SimpleNamespace(**r['spectrum']);q=object()
        with patch('rx706_qam_replay_adapter.verify_input',return_value=Path(r['input_path'])) as verify,patch('rx706_qam_replay_adapter.analyze_bin',return_value=s) as analyze,patch('rx706_qam_replay_adapter.recover_qam',return_value=q) as recover:
            self.assertEqual(run_replay(r),(s,q));self.assertEqual(verify.call_count,2)
            analyze.assert_called_once_with(Path(r['input_path']),**r['spectrum'],cancel_event=None)
            recover.assert_called_once_with(s,**r['recovery'],cancel_event=None)
    def test_metadata_rate_and_clamped_fft_fail_before_recovery(self):
        r=validate_replay(self.recipe)
        for key in ('sample_rate_hz','nfft','averages'):
            s=SimpleNamespace(**r['spectrum']);setattr(s,key,getattr(s,key)*2)
            with patch('rx706_qam_replay_adapter.verify_input',return_value=Path(r['input_path'])),patch('rx706_qam_replay_adapter.analyze_bin',return_value=s),patch('rx706_qam_replay_adapter.recover_qam') as recover:
                with self.assertRaises(ValueError):run_replay(r)
                recover.assert_not_called()
    def test_cancel_before_hash(self):
        event=threading.Event();event.set()
        with patch('rx706_qam_replay_adapter.verify_input') as verify:
            with self.assertRaises(AnalysisCancelled):run_replay(self.recipe,event)
            verify.assert_not_called()

if __name__=='__main__':unittest.main()

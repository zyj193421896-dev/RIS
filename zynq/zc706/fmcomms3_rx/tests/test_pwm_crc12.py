import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm_crc12 import crc12, test_frames, score_slots, append_crc

class CRC12Tests(unittest.TestCase):
    def records(self):
        return [dict(user_id=i+1, received_symbols=list(f), identity_ok=True,
                     calibration_ok=True, complete=True) for i,f in enumerate(test_frames())]
    def test_vectors(self):
        self.assertEqual([crc12(f[:16]) for f in test_frames()], [0x04e,0x64b,0x7c7])
        self.assertEqual([crc12(f) for f in test_frames()], [0,0,0])
    def test_ideal(self):
        s=score_slots(self.records(),True)
        self.assertEqual(s['overall']['continuous_window_bler'],0)
        self.assertIsNone(s['overall']['transmission_bler'])
    def test_each_single_bit(self):
        for j in range(20):
            for bit in (1,2,4):
                r=self.records();r[0]['received_symbols'][j]^=bit
                self.assertEqual(score_slots(r,True)['overall']['failed_slots'],1)
    def test_missing_round(self):
        r=self.records()
        for x in r:x['received_symbols']=None;x['detected']=False
        s=score_slots(r,True)['overall']
        self.assertEqual(s['failed_slots'],3);self.assertEqual(s['detected_complete_slots'],0)
    def test_unknown_gap(self):
        s=score_slots(self.records())
        self.assertIsNone(s['overall']['continuous_window_bler'])
        self.assertFalse(s['missing_whole_cycles_identifiable'])
    def test_partial(self):
        r=self.records();r[-1]['complete']=False
        self.assertEqual(score_slots(r)['overall']['expected_complete_slots'],2)
    def test_bad_calibration(self):
        r=self.records();r[0]['calibration_ok']=False
        self.assertEqual(score_slots(r)['overall']['failed_slots'],1)
    def test_no_reference_dependence(self):
        a=self.records();b=self.records()
        for x in b:x['reference']=[7]*20
        self.assertEqual(score_slots(a),score_slots(b))
    def test_recomputed_crc_does_not_override_identity_or_calibration(self):
        for position in (0,12):
            r=self.records();covered=r[0]['received_symbols'][:16]
            covered[position]^=1;r[0]['received_symbols']=append_crc(covered)
            result=score_slots(r)['slots'][0]
            self.assertTrue(result['crc_ok']);self.assertTrue(result['failed'])
    def test_valid_changed_payload_is_not_ground_truth(self):
        r=self.records();covered=r[0]['received_symbols'][:16]
        covered[4]^=1;r[0]['received_symbols']=append_crc(covered)
        result=score_slots(r)
        self.assertFalse(result['slots'][0]['failed'])
        self.assertIn('ground truth',result['overall']['metric_caveat'])
        self.assertIn('independent',result['overall']['confidence_caveat'])
    def test_uid_float_rejected(self):
        r=self.records();r[0]['user_id']=1.0
        with self.assertRaises(ValueError):score_slots(r)
    def test_numpy_integrals(self):
        try:import numpy as np
        except ImportError:self.skipTest('NumPy unavailable')
        r=self.records()
        for x in r:
            x['user_id']=np.int64(x['user_id'])
            x['received_symbols']=np.asarray(x['received_symbols'],dtype=np.int64)
        result=score_slots(r)
        self.assertEqual(result['overall']['failed_slots'],0)
        self.assertIs(type(result['slots'][0]['user_id']),int)
    def test_received_symbols_json_roundtrip(self):
        import json
        r=self.records();r[0]['received_symbols'][5]=None
        report=score_slots(r)
        restored=json.loads(json.dumps(report))
        self.assertEqual(restored['slots'][0]['received_symbols'],r[0]['received_symbols'])
        self.assertEqual(len(restored['slots'][0]['received_symbols']),20)
    def test_pass_fail_partition(self):
        records=self.records()
        records[0]['received_symbols'][4]^=1
        report=score_slots(records,True)
        self.assertEqual(report['overall']['passed_slots'],2)
        self.assertEqual(report['overall']['failed_slots'],1)
        for summary in [report['overall']]+list(report['users'].values()):
            self.assertEqual(summary['expected_complete_slots'],summary['passed_slots']+summary['failed_slots'])
    def test_bad_input(self):
        with self.assertRaises(ValueError):crc12([8])
        r=self.records();r[0]['received_symbols'].pop()
        with self.assertRaises(ValueError):score_slots(r)

if __name__=='__main__':unittest.main()

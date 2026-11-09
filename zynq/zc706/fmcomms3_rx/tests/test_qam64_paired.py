"""User-confirmed 3+4+4+128 frame; physical data is never merged or guessed."""
from pathlib import Path
import sys,tempfile,unittest,json,csv,copy
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from qam64_fixtures import tdcm_capture,compensation
from rx706_tdcm import recover_tdcm
from rx706_qam_payload import attach_payload_debug,validate_payload_options
from rx706_qam_debug_gui import export_symbols_csv,export_training_csv,export_payload_pairs_csv,export_tail_csv,export_report_html
from rx706_gui import parse_launch_args
from rx706_spectrum_gui import PAIRED_64_MODE
from test_qam64_controls import controls

TRAIN=(0,4,36,32)*2


def capture(directory,*,tail=(),crc=False,**kwargs):
    s,o,payload,period=tdcm_capture(directory,ndata=128,pilots=TRAIN,paired_payload=True,tail_labels=tail,**kwargs)
    compensation(o,ndata=128+len(tail),pilots=8)
    o['tdcm_options'].update(payload_repeat=2,payload_data_symbols=128,payload_tail_symbols=len(tail),crc_debug=dict(enabled=crc))
    return s,o,payload,period


class Paired64Tests(unittest.TestCase):
    def test_multi_loop_two_training_passes_phase_clock_and_mirror(self):
        for case in [dict(ppm=45,carrier=120173.4),dict(ppm=-420,carrier=-530247.8,mirror=True,ripple=8)]:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                s,o,payload,period=capture(d,**case);r=recover_tdcm(s,**o)
                self.assertEqual(r.preamble['expected_hex'],['00','04','24','20']*2)
                self.assertEqual(r.pwm_preamble['symbols'],[0,0,0]);self.assertEqual(r.bits.shape[1],6)
                self.assertAlmostEqual(r.diagnostics['frame_period_samples']/(s.sample_rate_hz/r.symbol_hz),139,delta=.001)
                valid=sum(v['valid'] for v in r.frames);self.assertGreaterEqual(valid,28)
                self.assertEqual(len(r.scatter),128*valid);self.assertEqual(len(r.payload_pairs),64*valid)
                self.assertEqual(len(r.training_records),8*len(r.frames));self.assertEqual(r.payload_debug['crc']['status'],'disabled')
                for k in np.unique(r.frame_ids)[2:-2]:
                    ids=np.flatnonzero(r.frame_ids==k);row=next(v for v in r.frames if v['index']==k)
                    np.testing.assert_array_equal(r.labels[ids],payload[0]);self.assertEqual(len(ids),128)
                    self.assertAlmostEqual((r.time_s[ids[0]]-row['header_time_s'])*r.symbol_hz,11.5,delta=.02)
                    pairs=[v for v in r.payload_pairs if v['frame']==k]
                    self.assertEqual([v['logical_label'] for v in pairs],payload[0][::2].tolist())
                self.assertIsNone(r.diagnostics['ber'])

    def test_disagreement_keeps_both_physical_values_and_no_logical_vote(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,payload,_=capture(d,pair_disagreement=True);r=recover_tdcm(s,**o)
            for k in np.unique(r.frame_ids)[2:-2]:
                np.testing.assert_array_equal(r.labels[r.frame_ids==k],payload[0])
                row=next(v for v in r.payload_pairs if v['frame']==k and v['logical_index']==4)
                self.assertFalse(row['agree']);self.assertIsNone(row['logical_label']);self.assertIsNone(row['logical_hex'])
                self.assertNotEqual(row['left_hex'],row['right_hex'])
            before=r.scatter.copy();labels=r.labels.copy();attach_payload_debug(r)
            np.testing.assert_array_equal(r.scatter,before);np.testing.assert_array_equal(r.labels,labels)

    def test_extra_tail_is_raw_separate_and_crc_unknown_never_passes_or_fails(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,payload,_=capture(d,tail=(9,10,11),crc=True);r=recover_tdcm(s,**o)
            valid=sum(v['valid'] for v in r.frames)
            self.assertEqual(len(r.scatter),131*valid);self.assertEqual(len(r.payload_pairs),64*valid)
            self.assertEqual(len(r.tail_records),3*valid);self.assertEqual(r.payload_debug['crc']['status'],'unverified')
            self.assertNotIn('pass',r.payload_debug['crc']);self.assertNotIn('poly',r.payload_debug['crc'])
            for k in np.unique(r.frame_ids)[2:-2]:
                np.testing.assert_array_equal(r.labels[r.frame_ids==k][:128],payload[0])
                self.assertEqual([v['candidate_hex'] for v in r.tail_records if v['frame']==k],['09','0A','0B'])
            out=Path(d);export_symbols_csv(r,out/'physical.csv');export_training_csv(r,out/'training.csv')
            export_payload_pairs_csv(r,out/'pairs.csv');export_tail_csv(r,out/'tail.csv');export_report_html(r,s,out/'report.html')
            for name,count in [('physical',len(r.scatter)),('pairs',len(r.payload_pairs)),('tail',len(r.tail_records))]:
                with (out/(name+'.csv')).open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
                self.assertEqual(len(rows),count+2)
            text=(out/'report.html').read_text();self.assertIn('CRC',text);self.assertIn('unverified',text)
            self.assertIn('重复净荷 / 额外帧尾 / CRC',text)

    def test_auto_search_and_recipe_keep_eight_training_and_pair_policy(self):
        with tempfile.TemporaryDirectory() as d:
            s,o,payload,_=capture(d,frames=40)
            o['tdcm_options']['auto_search']=dict(enabled=True,frame_candidates='136',max_carriers=1,max_trials=6)
            r=recover_tdcm(s,**o);self.assertEqual(r.diagnostics['auto_search']['status'],'selected')
            recipe=json.loads(json.dumps(r.diagnostics['auto_search']['recipe']))
            replay=recover_tdcm(s,tdcm_options=dict(replay_recipe=recipe))
            np.testing.assert_array_equal(replay.scatter,r.scatter);self.assertEqual(replay.payload_pairs,r.payload_pairs)
            v=controls();v.apply_qam_recipe_parameters(recipe)
            self.assertEqual(v.constellation_mode.get(),PAIRED_64_MODE);self.assertEqual(v.qam_training_repeats.get(),'2')
            self.assertEqual(v.four_corner_payload.get(),'128')

    def test_headless_preset_cli_and_extra_window_parameters(self):
        with patch('tkinter.Tk',side_effect=AssertionError('No GUI')),patch('tkinter.Toplevel',side_effect=AssertionError('No GUI')):
            v=controls();self.assertTrue(v.apply_paired_64_frame_profile())
            self.assertEqual(v.qam_order.get(),'64');self.assertEqual(v.qam_preamble_symbols.get(),'00 04 24 20 00 04 24 20')
            self.assertEqual(v.qam_compensation.values['frame_symbols'].get(),'136');self.assertEqual(v.qam_payload_repeat.get(),'2')
            v.qam_extra_tail_symbols.set('3');v.qam_crc_enabled.set(True)
            with patch('rx706_spectrum_gui.threading.Thread') as worker:
                v.analyze();o=worker.call_args.kwargs['args'][2]
            td=o['tdcm_options'];self.assertEqual(td['compensation']['frame_symbols'],139)
            self.assertEqual(td['payload_tail_symbols'],3);self.assertTrue(td['crc_debug']['enabled'])
            a=parse_launch_args(['--paired-64-frame','--crc-debug','--tail-symbols','3'])
            self.assertEqual(a.qam_order,64);self.assertEqual(a.payload_symbols,128);self.assertTrue(a.crc_debug)
            a=parse_launch_args([]);self.assertEqual(a.qam_order,256);self.assertFalse(a.crc_debug)

    def test_no_known_boundary_does_not_invent_pairing_or_crc(self):
        r=SimpleNamespace(labels=np.array([1,1]),scatter=np.array([1+1j,1+1j]),tdcm=dict(payload_repeat=2,crc_debug=dict(enabled=True)))
        attach_payload_debug(r);self.assertEqual(r.payload_debug['status'],'alignment_unverified')
        self.assertEqual(r.payload_pairs,[]);self.assertEqual(r.payload_debug['crc']['status'],'unverified')
        for bad in [dict(payload_repeat=3),dict(payload_repeat=True),dict(payload_tail_symbols=4),dict(crc_debug=dict(enabled=1))]:
            with self.assertRaises(ValueError):validate_payload_options(bad)

if __name__=='__main__':unittest.main()

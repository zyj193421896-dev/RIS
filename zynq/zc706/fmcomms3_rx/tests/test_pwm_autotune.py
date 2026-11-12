"""Reference independence and strict capture/profile contract for offline search."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
import rx706_pwm_autotune as tune
from rx706_spectrum import AnalysisCancelled
from rx706_pwm_pulse import decode_pulses
from test_pwm_pulse import config, synthesize


class AutotuneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name)/'test.bin'
        self.source.write_bytes(b'\0'*8000)
        self.meta = dict(complete=True, file_bytes=8000, capture=dict(bytes=8000,
            order=['I1','Q1','I2','Q2'], dtype='int16', byte_order='little', valid_bits=12,
            channels=2, sample_rate_hz=15360000))
        self.metadata = Path(str(self.source)+'.json')
        self.metadata.write_text(json.dumps(self.meta))
        self.profile = json.loads((Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json').read_text())
        self.path = Path(self.tmp.name)/'frame.json'
        self.path.write_text(json.dumps(self.profile))

    def test_metadata_and_profile_required(self):
        with self.assertRaisesRegex(ValueError, 'Frame profile required'):
            tune.load_inputs(self.source)
        self.metadata.unlink()
        with self.assertRaisesRegex(ValueError, 'metadata'):
            tune.load_inputs(self.source, self.path)

    def test_format_and_mismatch(self):
        self.meta['capture']['byte_order'] = 'big'
        self.metadata.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            tune.load_inputs(self.source, self.path)
        self.meta['capture']['byte_order'] = 'little'
        self.metadata.write_text(json.dumps(self.meta))
        self.profile['source_sha256'] = 'wrong'
        self.path.write_text(json.dumps(self.profile))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            tune.load_inputs(self.source, self.path)

    def test_verified_bound_profile(self):
        self.meta['pwm_search_profile'] = dict(path=self.path.name,
            source_sha256=tune.sha256(self.source), profile_sha256=tune.sha256(self.path))
        self.metadata.write_text(json.dumps(self.meta))
        self.assertEqual(tune.load_inputs(self.source)[1], self.path.resolve())
        self.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            tune.load_inputs(self.source)

    def test_references_removed_before_validation(self):
        changed = copy.deepcopy(self.profile)
        changed['users'][0]['reference'] = 'not even a valid reference'
        changed['reference_enabled'] = True
        changed['reference_format'] = 'invalid'
        self.assertEqual(tune.search_options(self.profile), tune.search_options(changed))
        self.assertEqual(tune.candidates(self.profile), tune.candidates(changed))

    def test_new_preamble_and_reservations_preserved(self):
        self.profile['users'][0]['identity'] = '1 2 3 4 5 6 7 8'
        opts = tune.search_options(self.profile)
        self.assertEqual(opts['frame_options']['users'][0]['identity'], '1 2 3 4 5 6 7 8')
        self.assertEqual(opts['cycle_calibration'], self.profile['cycle_calibration'])

    def test_deterministic_rank_and_no_all_erasure_win(self):
        a, b = tune.candidates(self.profile)[:2]
        good = dict(candidate=b, metrics=dict(complete_blocks=10, usable_fraction=.8,
            identity_fraction=1, competing_headers=0, median_distance_chips=.2))
        erased = dict(candidate=a, metrics=dict(complete_blocks=10, usable_fraction=0,
            identity_fraction=1, competing_headers=0, median_distance_chips=0))
        self.assertEqual(sorted([good, erased], key=tune.rank_key), sorted([erased, good], key=tune.rank_key))
        self.assertIs(sorted([erased, good], key=tune.rank_key)[0], good)

    def test_cancellation_and_candidate_limit(self):
        event = threading.Event(); event.set()
        with self.assertRaises(AnalysisCancelled):
            tune.search(self.source, self.path, cancel_event=event)
        with self.assertRaisesRegex(ValueError, 'Candidate limit'):
            tune.search(self.source, self.path, max_candidates=1)

    def test_corruption_and_truncation(self):
        self.meta['chunk_crc32'] = ['00000000']
        self.meta['capture']['chunk_bytes'] = 8000
        self.metadata.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, 'CRC mismatch'):
            tune.load_inputs(self.source, self.path)
        self.source.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            tune.load_inputs(self.source, self.path)

    def test_ambiguous_and_failed_candidates_not_exported(self):
        with patch.object(tune, 'analyze_bin', return_value=None), patch.object(tune, 'recover_pwm2', return_value=None), patch.object(tune, 'metrics') as metric:
            metric.return_value = dict(complete_blocks=5, usable_fraction=.9,
                identity_fraction=1., competing_headers=0, median_distance_chips=.1)
            report = tune.search(self.source, self.path)
            self.assertEqual(report['status'], 'ambiguous')
            self.assertIsNone(report['selected_profile'])
            metric.return_value = dict(complete_blocks=5, usable_fraction=0,
                identity_fraction=0., competing_headers=0, median_distance_chips=0)
            report = tune.search(self.source, self.path)
            self.assertIsNone(report['selected_profile'])
            self.assertIsNone(report['selection'])
        with patch.object(tune, 'analyze_bin', return_value=None), patch.object(tune, 'recover_pwm2', side_effect=ValueError('wrong identity')):
            with self.assertRaisesRegex(ValueError, 'No candidate matched'):
                tune.search(self.source, self.path)

    def test_reference_perturbation_actual_decoder(self):
        options, sent, rises, falls, times, states, origin = synthesize(config(payload=32), rounds=4)
        profile = dict(options, users=options['frame_options']['users'], symbol_khz=169.971671)
        edges = np.sort(np.r_[rises, falls]); after = np.isin(edges, rises)
        results = []
        for reference in ('0 1 2', '99999 malformed', ''):
            profile['users'][0]['reference'] = reference
            clean = tune.search_options(profile)
            rows, rate, calibration, normalized = decode_pulses(states, edges, after, times,
                15.36e6, 169971.671, clean)
            result = SimpleNamespace(frames=rows, pulse_options=normalized, measured_symbol_rate_hz=rate)
            results.append(tune.metrics(result, times[-1], normalized, 169971.671))
        self.assertEqual(results[0], results[1]); self.assertEqual(results[0], results[2])

    def test_search_reference_independent_ranking(self):
        def fake_recover(spectrum, **kwargs):
            self.assertFalse(kwargs['pulse_options']['frame_options']['reference_enabled'])
            self.assertEqual(kwargs['pulse_options']['frame_options']['users'][0]['reference'], '')
            return kwargs
        def fake_metrics(result, *args):
            return dict(complete_blocks=5, usable_fraction=.9 if result['state_one']=='A' else .1,
                identity_fraction=1., competing_headers=0, median_distance_chips=.1,
                decisions_sha256='identical', timing_note='')
        reports = []
        with patch.object(tune, 'analyze_bin', return_value=None), patch.object(tune, 'recover_pwm2', side_effect=fake_recover), patch.object(tune, 'metrics', side_effect=fake_metrics), patch.object(tune, 'file_evaluation', return_value={}):
            for ref in ('0 1 2', 'invalid held-out data'):
                self.profile['users'][0]['reference'] = ref
                self.path.write_text(json.dumps(self.profile))
                reports.append(tune.search(self.source, self.path))
        for key in ('scan', 'full_record', 'selection', 'candidate_set'):
            self.assertEqual(reports[0][key], reports[1][key])

    def _mock_duration_search(self, duration, *, ambiguous=False, failed=False, cancel_full=False, unusable=False):
        loaded = tune.load_inputs(self.source, self.path)
        provenance = dict(loaded[3], duration_s=duration)
        calls = []
        def recover(spectrum, **kwargs):
            calls.append(kwargs)
            self.assertFalse(kwargs['pulse_options']['frame_options']['reference_enabled'])
            self.assertEqual(kwargs['pulse_options']['frame_options']['users'][0]['reference'], '')
            if kwargs['duration_s'] > 1 and cancel_full:
                raise AnalysisCancelled()
            if kwargs['duration_s'] > 1 and failed:
                raise ValueError('full capture cannot decode')
            return kwargs
        def metric(result, *args):
            return dict(complete_blocks=5, usable_fraction=0 if unusable else .9 if ambiguous or result['state_one']=='A' else .1,
                identity_fraction=1., competing_headers=0, median_distance_chips=.1)
        with patch.object(tune, 'load_inputs', return_value=(*loaded[:3], provenance)), \
                patch.object(tune, 'analyze_bin', return_value=None), \
                patch.object(tune, 'recover_pwm2', side_effect=recover), \
                patch.object(tune, 'metrics', side_effect=metric), \
                patch.object(tune, 'file_evaluation', side_effect=lambda result, *args: dict(analyzed_duration_s=result['duration_s'])):
            report = tune.search(self.source, self.path)
        return report, calls

    def test_two_seconds_freezes_first_second_winner_then_full_record(self):
        report, calls = self._mock_duration_search(2.)
        self.assertEqual([x['duration_s'] for x in calls], [.03]*12+[1.]*3+[2.])
        self.assertEqual(report['selection_interval']['duration_s'], 1.)
        self.assertEqual(report['selection']['duration_s'], 1.)
        self.assertEqual(len(report['selection_record']), 3)
        self.assertEqual(len(report['full_record']), 1)
        self.assertEqual(report['full_record'][0]['duration_s'], 2.)
        self.assertEqual(report['full_record'][0]['candidate'], report['selection']['candidate'])
        self.assertEqual(report['post_selection_evaluation']['analyzed_duration_s'], 2.)
        self.assertTrue(report['analysis_interval']['completed'])

    def test_one_second_keeps_original_replay_path(self):
        report, calls = self._mock_duration_search(1.)
        self.assertEqual([x['duration_s'] for x in calls], [.03]*12+[1.]*3)
        self.assertEqual(report['full_record'], report['selection_record'])
        self.assertEqual(report['post_selection_evaluation']['analyzed_duration_s'], 1.)

    def test_two_second_ambiguity_does_not_publish_or_replay(self):
        report, calls = self._mock_duration_search(2., ambiguous=True)
        self.assertEqual(report['status'], 'ambiguous')
        self.assertIsNone(report['selected_profile'])
        self.assertEqual(report['full_record'], [])
        self.assertFalse(report['analysis_interval']['completed'])
        self.assertTrue(all(x['duration_s'] <= 1 for x in calls))

    def test_two_second_unusable_candidate_never_published(self):
        report, calls = self._mock_duration_search(2., unusable=True)
        self.assertIsNone(report['selection'])
        self.assertIsNone(report['selected_profile'])
        self.assertIsNone(report['post_selection_evaluation'])
        self.assertEqual(report['full_record'], [])
        self.assertFalse(report['analysis_interval']['completed'])
        self.assertTrue(all(x['duration_s'] <= 1 for x in calls))

    def test_two_second_full_failure_and_cancel_publish_nothing(self):
        with self.assertRaisesRegex(ValueError, 'full-record replay failed'):
            self._mock_duration_search(2., failed=True)
        with self.assertRaises(AnalysisCancelled):
            self._mock_duration_search(2., cancel_full=True)
        with self.assertRaisesRegex(ValueError, 'up to two seconds'):
            self._mock_duration_search(2.0001)

    def test_two_second_reference_does_not_change_selection(self):
        first, _ = self._mock_duration_search(2.)
        self.profile['users'][0]['reference'] = 'invalid changed held-out reference'
        self.path.write_text(json.dumps(self.profile))
        second, _ = self._mock_duration_search(2.)
        for key in ('selection', 'selection_record', 'full_record', 'scan'):
            self.assertEqual(first[key], second[key])

    def test_no_reference_still_reports_physical_erasures(self):
        from rx706_pwm_frames import check_frame_reference
        import math
        options, sent, rises, falls, times, states, origin = synthesize(config(payload=32), rounds=4)
        edges = np.sort(np.r_[rises, falls]); after = np.isin(edges, rises)
        rows, rate, calibration, normalized = decode_pulses(states, edges, after, times,
            15.36e6, 169971.671, options)
        payload = rows['block_complete'] & (rows['frame_part'] == '净荷')
        rows['valid'][np.flatnonzero(payload)[:5]] = False
        widths = tuple(w for d, w in normalized['codebook'])
        bits = math.ceil(math.log2(len(widths)))
        bitwords = [f'{i:0{bits}b}' for i in range(len(widths))]
        check = check_frame_reference(rows, normalized['frame_options'], widths, bitwords)
        result = SimpleNamespace(frames=rows, pulse_options=normalized, measured_symbol_rate_hz=rate,
            frame_check=check, symbol_widths=widths, symbol_bitwords=bitwords, bits_per_symbol=bits)
        evaluation = tune.file_evaluation(result)
        self.assertIsNone(evaluation['ber'])
        self.assertEqual(evaluation['reference_overall']['erased_bits'], 0)
        self.assertGreaterEqual(evaluation['erased_payload_symbols'], 5)
        self.assertGreater(evaluation['physical_erasure_fraction'], 0)
        before = rows.copy()
        profile = dict(users=normalized['frame_options']['users'], reference_enabled=True)
        profile['users'][0]['reference'] = '0 1 2 3'
        self.assertIsNotNone(tune.file_evaluation(result, profile)['ber'])
        for field in ('symbol', 'valid', 'identity_ok'):
            np.testing.assert_array_equal(before[field], rows[field])


if __name__ == '__main__':
    unittest.main()

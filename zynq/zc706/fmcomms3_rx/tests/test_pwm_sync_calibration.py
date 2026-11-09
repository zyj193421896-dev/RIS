"""Independent regression cases for preamble-only PWM synchronization/calibration.

The synthesizer uses transmit d/w directly. References are never used to build
receiver timing or calibrations; changing them must not change a decision.
"""
from pathlib import Path
import copy
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
import rx706_pwm_pulse as pulse
from rx706_pwm_pulse import validate_pulse_options, decode_pulses, _header, _competing_marker

FS = 15.36e6
RATE = 169971.671


def options(identity=(0, 10, 45, 5), payload=66, users=None, **extra):
    if users is None:
        users = [dict(name='U1', identity=' '.join(map(str, identity)),
                      payload_symbols=payload, reference=' '.join(map(str, range(payload))))]
    return dict(step_percent=5, min_high_chips=5, low_guard_chips=5,
                symbol_count=66, joint_delay=True,
                frame_options=dict(user_count=len(users), users=users,
                                   identity_errors=0, reference_enabled=True,
                                   reference_format='symbols'), **extra)


def synthesize(config, rounds=4, gain=1., offset_chips=.2,
               rise_jitter=None, missing=(), actual_rate=RATE, payload_start=0):
    normalized = validate_pulse_options(config, RATE, FS)
    book = np.asarray(normalized['codebook'])
    sent, kinds, block_ids = [], [], []
    block = 0
    for cycle in range(rounds):
        for user in normalized['frame_options']['users']:
            block += 1
            header = list(user['identity'])
            payload = ((np.arange(user['payload_symbols']) + payload_start) % len(book)).tolist()
            sent.extend(header + payload)
            kinds.extend([False] * len(header) + [True] * len(payload))
            block_ids.extend([block] * (len(header) + len(payload)))
    sent = np.asarray(sent)
    origin = .37 / actual_rate
    chip = 1 / actual_rate / 20
    rises = origin + np.arange(len(sent)) / actual_rate + book[sent, 0] * chip
    falls = rises + (gain * book[sent, 1] + offset_chips) * chip
    for index, displacement_chips in (rise_jitter or {}).items():
        rises[index] += displacement_chips * chip
        falls[index] += displacement_chips * chip
    keep = np.ones(len(sent), bool)
    keep[list(missing)] = False
    rises, falls = rises[keep], falls[keep]
    times = np.arange(int((origin + (len(sent) + .8) / actual_rate) * FS)) / FS
    i = np.searchsorted(rises, times, side='right') - 1
    states = (i >= 0) & (times < falls[np.maximum(i, 0)])
    edges = np.r_[rises, falls]
    after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
    order = np.argsort(edges)
    return states, edges[order], after[order], times, sent, np.asarray(kinds), origin


def decode(config, **kwargs):
    states, edges, after, times, sent, payload, origin = synthesize(config, **kwargs)
    result = decode_pulses(states, edges, after, times, FS, RATE, config)
    return result, sent, payload, origin


class SyncCalibrationTests(unittest.TestCase):
    def test_limit15_codebook_includes_equal_boundary(self):
        config = validate_pulse_options(options(), RATE, FS)
        self.assertEqual(len(config['codebook']), 66)
        for index, pair in ((0, (0., 5.)), (5, (0., 10.)), (10, (0., 15.)),
                            (11, (1., 5.)), (45, (5., 5.)), (65, (10., 5.))):
            self.assertEqual(config['codebook'][index], pair)

    def test_reference_changes_do_not_change_receiver_decisions(self):
        config = options()
        (first, _, _, _), _, _, _ = decode(config)
        changed = copy.deepcopy(config)
        changed['frame_options']['users'][0]['reference'] = '65 64 63'
        (second, _, _, _), _, _, _ = decode(changed)
        for field in ('symbol', 'valid', 'bits', 'start_s', 'delay_chips',
                      'corrected_high_chips', 'identity_ok'):
            np.testing.assert_array_equal(first[field], second[field], err_msg=field)

    def test_single_missing_payload_keeps_erasure_slot_and_later_position(self):
        config = options()
        (rows, _, _, _), sent, _, origin = decode(config, missing=(12,))
        missing = np.argmin(abs(rows['start_s'] - (origin + 12 / RATE)))
        self.assertFalse(rows['valid'][missing])
        self.assertEqual(rows['symbol'][missing], -1)
        later = np.argmin(abs(rows['start_s'] - (origin + 13 / RATE)))
        self.assertTrue(rows['valid'][later])
        self.assertEqual(rows['symbol'][later], sent[13])


    def test_five_identical_pulses_report_boundary_ambiguity(self):
        config = options(identity=(0, 0, 0, 0))
        (rows, _, _, learned), _, _, _ = decode(config)
        self.assertGreater(learned['sync_ambiguous_blocks'], 0)
        ambiguous = rows['sync_status'] == 'ambiguous'
        self.assertTrue(ambiguous.any())
        self.assertFalse(rows['valid'][ambiguous].any())

    def test_first_preamble_edge_outlier_does_not_shift_payload_phase(self):
        config = options(identity=(0, 5, 45, 2))
        (rows, _, _, _), sent, _, origin = decode(config, rise_jitter={0: .9})
        payload = (rows['frame_part'] == '净荷') & (rows['block_id'] == 1)
        self.assertTrue(rows['valid'][payload].any())
        indices = np.rint((rows['start_s'][payload] - origin) * RATE).astype(int)
        np.testing.assert_allclose(rows['start_s'][payload], origin + indices / RATE,
                                   atol=.15 / RATE / 20)
        good = rows['valid'][payload]
        np.testing.assert_array_equal(rows['symbol'][payload][good], sent[indices][good])

    def test_final_period_and_phase_use_same_clock(self):
        actual_rate = RATE * 1.0003
        config = options(identity=(0, 5, 45, 2))
        (rows, measured, _, _), sent, _, origin = decode(config, actual_rate=actual_rate,
            rise_jitter={0: .3, 1: .1, 2: .1, 3: .1})
        # Allow 50 ppm from finite noisy header timing; phase stays below 0.15 chip.
        self.assertAlmostEqual(measured / actual_rate, 1., delta=5e-5)
        good = rows['valid'] & (rows['frame_part'] == '净荷') & (rows['block_id'] > 1)
        self.assertGreater(good.sum(), 100)
        indices = np.rint((rows['start_s'][good] - origin) * actual_rate).astype(int)
        np.testing.assert_allclose(rows['start_s'][good], origin + indices / actual_rate,
                                   atol=.15 / actual_rate / 20)
        np.testing.assert_array_equal(rows['symbol'][good], sent[indices])

    def test_missing_header_does_not_destroy_later_reacquisition(self):
        config = options(identity=(0, 5, 45, 2))
        # Lose two header pulses in the second 70-symbol frame.
        (rows, _, _, _), sent, _, origin = decode(config, rounds=5, missing=(70, 71))
        damaged = rows['block_id'] == 2
        self.assertTrue(damaged.any())
        self.assertFalse(rows['valid'][damaged].any())
        later = (rows['block_id'] >= 3) & (rows['frame_part'] == '净荷') & rows['valid']
        self.assertGreater(later.sum(), 100)
        indices = np.rint((rows['start_s'][later] - origin) * RATE).astype(int)
        np.testing.assert_array_equal(rows['symbol'][later], sent[indices])

    def test_multiple_preamble_widths_identify_affine_width_response(self):
        config = options(identity=(0, 5, 45, 2))
        (rows, _, calibrations, _), sent, _, origin = decode(config, gain=1.1057,
                                                            offset_chips=-.43891)
        affine = [item for item in calibrations if item['model_identifiable']]
        self.assertTrue(affine)
        for item in affine:
            self.assertAlmostEqual(item['width_scale'], 1.1057, places=6)
            self.assertAlmostEqual(item['width_offset_chips'], -.43891, places=6)
        good = rows['valid'] & (rows['frame_part'] == '净荷')
        self.assertGreater(good.sum(), 200)
        indices = np.rint((rows['start_s'][good] - origin) * RATE).astype(int)
        np.testing.assert_array_equal(rows['symbol'][good], sent[indices])

    def test_single_width_cannot_claim_gain_and_offset_identified(self):
        config = options(identity=(0, 0, 0, 0))
        (rows, _, calibrations, _), _, _, _ = decode(config, payload_start=1)
        self.assertTrue(calibrations)
        self.assertTrue(all(not item['model_identifiable'] for item in calibrations))
        model = rows['calibration_model'][rows['identity_ok']]
        self.assertTrue(np.all(model == 'offset_only'))


    def test_experimental_profile_is_separate_and_tx_order_is_width_delay(self):
        base = Path(__file__).resolve().parents[1] / 'host/frame_profiles/limit15_step5'
        old = json.loads((base / 'single_user_mt_test_169971.json').read_text())
        new = json.loads((base / 'experimental_mixed_preamble_169971.json').read_text())
        self.assertEqual(old['users'][0]['identity'], '0 0 0 0')
        self.assertEqual(new['users'][0]['identity'], '0 5 45 2')
        self.assertEqual(new['users'][0]['reference'], old['users'][0]['reference'])
        profile = dict(new, step_percent=new['duty_step'],
            frame_options=dict(new, reference_format='symbols'))
        validated = validate_pulse_options(profile, RATE, FS)
        self.assertEqual(len(validated['codebook']), 66)
        lines = (base / 'experimental_mixed_preamble_169971_tx_percent.txt').read_text().splitlines()
        self.assertEqual(len(lines), 70)
        self.assertEqual(lines[:4], ['(25, 0)', '(50, 0)', '(25, 25)', '(35, 0)'])
        sent = [0, 5, 45, 2] + list(range(66))
        book = validated['codebook']
        for line, symbol in zip(lines, sent):
            width, delay = map(float, line.strip('()').split(','))
            self.assertEqual((delay / 5, width / 5), book[symbol])

    def test_two_users_keep_their_slots_and_independent_calibrations(self):
        users = [dict(name='U1', identity='0 5 45 2', payload_symbols=33, reference=''),
                 dict(name='U2', identity='10 0 5 45', payload_symbols=21, reference='')]
        config = options(users=users)
        (rows, _, calibrations, _), sent, _, origin = decode(config, rounds=3)
        self.assertEqual(set(rows['user_id']), {1, 2})
        self.assertEqual(set(item['user_id'] for item in calibrations), {1, 2})
        for user_id, count in ((1, 33), (2, 21)):
            good = (rows['user_id'] == user_id) & rows['valid'] & (rows['frame_part'] == '净荷')
            self.assertEqual(good.sum(), count * 3)
            indices = np.rint((rows['start_s'][good] - origin) * RATE).astype(int)
            np.testing.assert_array_equal(rows['symbol'][good], sent[indices])


    def test_competing_markers_are_symmetric_with_one_width_outlier(self):
        config = validate_pulse_options(options(identity=(0, 0, 0, 0)), RATE, FS)
        user = config['frame_options']['users'][0]
        book = np.asarray(config['codebook'])
        period = 1 / RATE
        rises = np.arange(6) * period
        falls = rises + (np.array([5., 5., 5., 5., 5., 6.]) + .2) * period / 20
        falls[0] += .9 * period / 20
        first = _header(rises, falls, 0, user, book, period, config)
        shifted = _header(rises, falls, 1, user, book, period, config)
        self.assertIsNotNone(first)
        self.assertIsNotNone(shifted)
        self.assertEqual(first['width_rejected'], 1)
        self.assertEqual(shifted['width_rejected'], 0)
        self.assertTrue(_competing_marker(rises, falls, first, user, book, period, config))
        self.assertTrue(_competing_marker(rises, falls, shifted, user, book, period, config))


    def test_ambiguous_anchor_does_not_bias_long_baseline_clock(self):
        config = options(identity=(0, 5, 45, 2))
        original_header = pulse._header
        original_competing = pulse._competing_marker

        def displaced_header(*args, **kwargs):
            match = original_header(*args, **kwargs)
            if match is not None and match['pulse_at'] == 70:
                match = dict(match, anchor=match['anchor'] + .7 / RATE)
            return match

        def competing(*args, **kwargs):
            return args[2]['pulse_at'] == 70 or original_competing(*args, **kwargs)

        with patch.object(pulse, '_header', side_effect=displaced_header), \
                patch.object(pulse, '_competing_marker', side_effect=competing):
            (rows, measured, _, config_out), _, _, _ = decode(config, rounds=5)
        self.assertAlmostEqual(measured, RATE, places=5)
        second = rows['block_id'] == 2
        self.assertTrue(np.all(rows['sync_status'][second] == 'ambiguous'))
        self.assertFalse(rows['valid'][second].any())
        self.assertGreater(config_out['sync_ambiguous_blocks'], 0)

    def test_failed_final_rephase_invalidates_inherited_tracking(self):
        config = options(identity=(0, 5, 45, 2))
        original_header = pulse._header
        original_calibration = pulse._marker_calibration
        in_header = [False]

        def acquisition_header(*args, **kwargs):
            in_header[0] = True
            try:
                return original_header(*args, **kwargs)
            finally:
                in_header[0] = False

        def calibration(*args, **kwargs):
            # Model a marker that was acceptable at acquisition but fails when
            # recomputed with the final clock. Do not affect acquisition calls.
            if not in_header[0] and args[2] == 0:
                return None
            return original_calibration(*args, **kwargs)

        def competing(*args, **kwargs):
            return args[2]['pulse_at'] >= 70

        with patch.object(pulse, '_header', side_effect=acquisition_header), \
                patch.object(pulse, '_marker_calibration', side_effect=calibration), \
                patch.object(pulse, '_competing_marker', side_effect=competing):
            (rows, _, _, _), _, _, _ = decode(config)
        first = rows['block_id'] == 1
        inherited = rows['block_id'] == 2
        self.assertFalse(rows['valid'][first].any())
        self.assertFalse(rows['valid'][inherited].any())
        self.assertFalse(np.any(rows['sync_status'][inherited] == 'tracking'))
        self.assertTrue(np.all(rows['sync_status'][inherited] == 'ambiguous'))


if __name__ == '__main__':
    unittest.main()

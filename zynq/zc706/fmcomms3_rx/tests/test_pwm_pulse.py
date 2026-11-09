"""Independent falling-edge-hold synthesis and pulse-position decoding."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm_pulse import validate_pulse_options, decode_pulses, _nearest_codeword, _header
from rx706_pwm_frames import default_users, check_frame_reference
from rx706_pwm import recover_pwm2
from rx706_spectrum import analyze_bin


def config(users=1, payload=128, **extra):
    user_rows = default_users(users)
    for user in user_rows:
        user['payload_symbols'] = payload
        user['reference'] = ''
    return dict(step_percent=5, min_high_chips=6, low_guard_chips=7, joint_delay=True,
                frame_options=dict(user_count=users, users=user_rows), **extra)


def synthesize(options, rounds=3, fs=15.36e6, rate=169971.671, tail_us=1.72, crop=0):
    options = validate_pulse_options(options, rate, fs)
    book, n = np.asarray(options['codebook']), options['total_chips']
    rng = np.random.default_rng(806)
    sent, payloads = [], []
    for _ in range(rounds):
        for user in options['frame_options']['users']:
            payload = rng.integers(len(book), size=user['payload_symbols'])
            payloads.append(payload)
            sent.extend(user['identity'])
            sent.extend(payload)
    sent = np.asarray(sent, int)
    origin = .37/rate
    rises = origin+(np.arange(len(sent))+book[sent, 0]/n)/rate
    falls = rises+book[sent, 1]/n/rate+tail_us*1e-6
    t = np.arange(int((origin+(len(sent)+.8)/rate)*fs))/fs
    index = np.searchsorted(rises, t, side='right')-1
    high = (index >= 0) & (t < falls[np.maximum(index, 0)])
    first = int(crop*fs/rate)
    return options, sent, rises, falls, t[first:]-first/fs, high[first:], origin-first/fs


def screenshot_case():
    """Replay measured payload pairs behind an unambiguous synthetic marker.

    This tests nearest-codeword handling, not the historical capture framing.
    The original on-disk profile and its ambiguous marker remain unchanged.
    """
    path = Path(__file__).resolve().parents[1]/'host/frame_profiles/limit14_step5/single_user_image23.json'
    profile = json.loads(path.read_text(encoding='utf-8'))
    profile['users'][0]['identity'] = '0 5 2 7'
    c = validate_pulse_options(dict(profile, step_percent=profile['duty_step'],
        symbol_count=profile['duty_count'], frame_options=dict(profile, reference_format='symbols')), 169971.671)
    sent = np.tile(np.r_[[0, 5, 2, 7], np.arange(1, 24)], 3)
    fs, rate, total = 15.36e6, 169971.671, 20
    period, origin, tail = 1/rate, .37/rate, .1760764430549e-6
    book = np.asarray(c['codebook'])
    starts = origin+np.arange(len(sent))*period
    rises = starts+book[sent, 0]/total*period
    falls = rises+book[sent, 1]/total*period+tail
    measurements = [(3, .0066841476, 9.4527), (4, .0804981680, 10.4442),
                    (8, .0445346699, 14.4477), (14, 1.050288565, 11.4709),
                    (15, 1.068759672, 12.4729)]
    affected = []
    for symbol, delay, width in measurements:
        at = 4+symbol-1
        affected.append(at)
        rises[at] = starts[at]+delay/total*period
        falls[at] = rises[at]+width/total*period+tail
    t = np.arange(int((origin+(len(sent)+.8)*period)*fs))/fs
    index = np.searchsorted(rises, t, side='right')-1
    high = (index >= 0) & (t < falls[np.maximum(index, 0)])
    return c, sent, rises, falls, t, high, np.asarray(affected)


class PulseTests(unittest.TestCase):
    def test_short_preamble_slope_cannot_drift_payload_or_later_header_search(self):
        fs, rate = 15.36e6, 213980.029
        chosen = config(payload=1024)
        # Mixed widths avoid repeated-marker ambiguity; zero delays preserve
        # the deliberately linear local edge-slope disturbance.
        chosen['frame_options']['users'][0]['identity'] = '0 5 2 7'
        for rounds in (1, 3):
            with self.subTest(rounds=rounds):
                c, sent, rises, falls, t, high, origin = synthesize(chosen, rounds=rounds, rate=rate, tail_us=.2)
                # Four edges imply a 1.3% local slope error, still inside the
                # acquisition search range but far too noisy for 1024 symbols.
                jitter = np.arange(4)*.013/rate
                rises[:4] += jitter
                falls[:4] += jitter
                indices = np.searchsorted(rises, t, side='right')-1
                high = (indices >= 0) & (t < falls[np.maximum(indices, 0)])
                edges = np.r_[rises, falls]
                after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
                order = np.argsort(edges)
                rows, measured, calibrations, learned = decode_pulses(high, edges[order], after[order], t, fs, rate, c)
                self.assertEqual(len(calibrations), rounds)
                self.assertAlmostEqual(measured, rate, places=4)
                payload = rows['frame_part'] == '净荷'
                self.assertTrue(np.all(rows['valid'][payload]))
                ids = np.rint((rows['start_s'][payload]-origin)*rate).astype(int)
                np.testing.assert_array_equal(rows['symbol'][payload], sent[ids])
                self.assertIn('配置符号率' if rounds == 1 else '长间隔', learned['clock_source'])

    def test_preamble_width_margin_accounts_for_processing_sample_resolution(self):
        fs, rate = 15.36e6, 213980.029
        chosen = config()
        chosen['frame_options']['users'][0]['identity'] = '0 0 0 0'
        c = validate_pulse_options(chosen, rate, fs)
        rises = np.arange(4)/rate
        chip = 1/rate/c['total_chips']
        falls = rises+6*chip+.2e-6+np.array([-.55, .55, -.55, .55])*chip
        user = c['frame_options']['users'][0]
        pairs = np.asarray(c['codebook'])
        self.assertIsNone(_header(rises, falls, 0, user, pairs, 1/rate,
                                  dict(c, preamble_sample_margin_us=0)))
        match = _header(rises, falls, 0, user, pairs, 1/rate, c)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match['tail']*1e6, .2)

    def test_screenshot_outliers_are_shown_and_scored_as_nearest_decisions(self):
        c, sent, rises, falls, t, high, affected = screenshot_case()
        edges = np.r_[rises, falls]
        after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
        order = np.argsort(edges)
        rows, _, _, _ = decode_pulses(high, edges[order], after[order], t, 15.36e6, 169971.671, c)
        np.testing.assert_array_equal(rows['symbol'][affected], [3, 4, 8, 14, 15])
        self.assertTrue(np.all(rows['valid'][affected]))
        self.assertTrue(np.all(rows['decision_estimated'][affected]))
        self.assertTrue(all('超容差' in reason for reason in rows['reason'][affected]))
        self.assertEqual(list(rows['bits'][affected]), ['000011', '000100', '001000', '001110', '001111'])
        words = tuple(format(k, '06b') for k in range(45))
        check = check_frame_reference(rows, c['frame_options'], tuple(w for _, w in c['codebook']), words)
        self.assertEqual(check.overall['estimated_symbols'], 5)
        self.assertEqual(check.overall['compared_bits'], 3*23*6)
        self.assertEqual(check.overall['erased_bits'], 0)
        self.assertEqual(check.overall['bad_blocks'], 0)
        self.assertEqual(check.overall['bit_errors'], 0)
        # Wrong approximate decisions are errors, not exempted from BER/BLER.
        wrong = rows.copy()
        wrong['symbol'][affected[0]], wrong['bits'][affected[0]] = 4, '000100'
        checked = check_frame_reference(wrong, c['frame_options'], tuple(w for _, w in c['codebook']), words)
        self.assertEqual(checked.overall['bit_errors'], 3)
        self.assertEqual(checked.overall['bad_blocks'], 1)
        strict, _, _, _ = decode_pulses(high, edges[order], after[order], t, 15.36e6, 169971.671,
                                        dict(c, nearest_decision=False))
        self.assertFalse(np.any(strict['valid'][affected]))
        np.testing.assert_array_equal(strict['candidate_symbol'][affected], [3, 4, 8, 14, 15])

    def test_missing_pulse_stays_erased_and_reference_does_not_change_estimates(self):
        c, sent, rises, falls, t, high, affected = screenshot_case()
        keep = np.arange(len(rises)) != 4
        edges = np.r_[rises[keep], falls[keep]]
        after = np.r_[np.ones(int(keep.sum()), bool), np.zeros(int(keep.sum()), bool)]
        order = np.argsort(edges)
        rows, _, _, _ = decode_pulses(high, edges[order], after[order], t, 15.36e6, 169971.671, c)
        self.assertFalse(rows['valid'][4])
        self.assertEqual(rows['symbol'][4], -1)
        self.assertEqual(rows['candidate_symbol'][4], -1)
        changed = dict(c, frame_options=dict(c['frame_options'], users=[dict(c['frame_options']['users'][0], reference='44 43 42')]))
        other, _, _, _ = decode_pulses(high, edges[order], after[order], t, 15.36e6, 169971.671, changed)
        np.testing.assert_array_equal(rows['symbol'], other['symbol'])
        np.testing.assert_array_equal(rows['bits'], other['bits'])
        np.testing.assert_array_equal(rows['valid'], other['valid'])

    def test_nearest_joint_codeword_uses_both_dimensions_and_reports_ties(self):
        pairs = np.asarray([(0, 6), (0, 7), (0, 8)])
        k, within, tied, distance = _nearest_codeword(1.8, 7.1, pairs, True, .4)
        self.assertEqual(k, 1)
        self.assertFalse(within)
        self.assertFalse(tied)
        self.assertAlmostEqual(distance, math.hypot(1.8, .1))
        self.assertEqual(_nearest_codeword(0., 6.5, pairs, True, .4)[:3], (0, False, True))
        self.assertIsNone(_nearest_codeword(float('nan'), 7., pairs, True, .4))

    def test_steps_derive_chips_and_count_joint_combinations(self):
        c = validate_pulse_options(config(), 169971.671)
        self.assertEqual(c['total_chips'], 20)
        self.assertEqual(c['symbol_count'], 36)
        self.assertAlmostEqual(c['information_bits'], math.log2(36))
        self.assertEqual(c['integer_payload_bits'], 5)
        self.assertEqual(c['codebook'][:2], ((0., 6.), (0., 7.)))
        self.assertTrue(all(d+w+7 <= 20 for d, w in c['codebook']))
        c = validate_pulse_options(dict(config(), step_percent=4), 169971.671)
        self.assertEqual(c['total_chips'], 25)
        self.assertEqual(c['symbol_count'], 91)
        self.assertNotIn('tail_us', c)
        for invalid in (dict(step_percent=7), dict(min_high_chips=0),
                        dict(codebook='7 13; 0 6'), dict(low_guard_chips=0)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_pulse_options(dict(config(), **invalid), 169971.671)

    def test_rate_includes_preamble_and_per_user_slot_lengths(self):
        c = validate_pulse_options(config(users=3, payload=1024), 169971.671)
        expected = 169971.671*math.log2(36)*1024/(3*1040)
        np.testing.assert_allclose(c['user_information_rates'], [expected]*3)
        one = validate_pulse_options(config(payload=1024), 169971.671)
        self.assertAlmostEqual(one['user_information_rates'][0], 3*expected)

    def test_identity_only_acquires_at_arbitrary_capture_offset_and_estimates_tail(self):
        fs, rate = 15.36e6, 169971.671
        c, sent, rises, falls, t, high, origin = synthesize(config(users=3), tail_us=1.79, crop=71)
        delta = .37/rate-origin
        edge_t = np.r_[rises-delta, falls-delta]
        after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
        order = np.argsort(edge_t)
        keep = (edge_t[order] >= t[0]) & (edge_t[order] <= t[-1])
        rows, measured, calibrations, _ = decode_pulses(high, edge_t[order][keep], after[order][keep], t, fs, rate, c)
        payload = rows['frame_part'] == '净荷'
        indices = np.rint((rows['start_s'][payload]-origin)*rate).astype(int)
        self.assertTrue(np.all(rows['valid'][payload]))
        np.testing.assert_array_equal(rows['symbol'][payload], sent[indices])
        self.assertAlmostEqual(measured, rate, places=4)
        self.assertTrue(all(abs(row['tail_us']-1.79) < 1e-7 for row in calibrations))
        self.assertGreater(np.count_nonzero(rows['delay_chips'][payload] > .5), 30)
        self.assertFalse(np.any(rows['frame_part'] == '公共训练'))

    def test_rf_pipeline_decodes_delayed_pulses_with_tail(self):
        fs, rate = 15.36e6, 169971.671
        c, sent, _, _, t, high, origin = synthesize(config(users=3), tail_us=1.79)
        rng = np.random.default_rng(808)
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        x += rng.normal(0, .3, len(t))+1j*rng.normal(0, .3, len(t))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tail.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=4096, averages=1, sample_rate_hz=fs)
            result = recover_pwm2(spectrum, symbol_hz=rate, pulse_options=c)
        self.assertTrue(result.decode_locked, result.decode_note)
        self.assertEqual(result.frame_check.overall['complete_blocks'], 9)
        rows = result.frames
        payload = rows['frame_part'] == '净荷'
        self.assertTrue(np.all(rows['valid'][payload]), str(rows['reason'][payload][~rows['valid'][payload]][:8]))
        indices = np.rint((rows['start_s'][payload]-origin)*rate).astype(int)
        np.testing.assert_array_equal(rows['symbol'][payload], sent[indices])
        self.assertEqual(result.total_chips, 20)
        self.assertTrue(all(abs(value['tail_us']-1.79) < .04 for value in result.pulse_calibrations))

    def test_bad_user_header_retains_slot_and_does_not_shift_later_users(self):
        fs, rate = 15.36e6, 169971.671
        c, sent, rises, falls, t, high, origin = synthesize(config(users=3))
        # Damage three known widths in U2's first header, beyond its allowed
        # one-symbol mismatch, while leaving the scheduled slot present.
        falls = falls.copy()
        falls[np.array([144, 146, 149])] += 1/rate/20
        edge_t = np.r_[rises, falls]
        after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
        order = np.argsort(edge_t)
        rows, _, calibrations, _ = decode_pulses(high, edge_t[order], after[order], t, fs, rate, c)
        self.assertFalse(np.any(rows['identity_ok'][rows['block_id'] == 2]))
        failed = (rows['block_id'] == 2) & (rows['frame_part'] == '净荷')
        self.assertFalse(np.any(rows['valid'][failed]))
        good = (rows['block_id'] >= 3) & (rows['frame_part'] == '净荷')
        self.assertTrue(np.all(rows['valid'][good]))
        indices = np.rint((rows['start_s'][good]-origin)*rate).astype(int)
        np.testing.assert_array_equal(rows['symbol'][good], sent[indices])

    def test_preamble_clock_fit_handles_frequency_error_without_data_rise_lock(self):
        fs, configured, actual = 15.36e6, 169971.671, 170300.
        c, sent, rises, falls, t, high, origin = synthesize(config(), rate=actual)
        edge_t = np.r_[rises, falls]
        after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
        order = np.argsort(edge_t)
        rows, measured, _, _ = decode_pulses(high, edge_t[order], after[order], t, fs, configured, c)
        self.assertAlmostEqual(measured, actual, places=4)
        payload = rows['frame_part'] == '净荷'
        self.assertTrue(np.all(rows['valid'][payload]))
        indices = np.rint((rows['start_s'][payload]-origin)*actual).astype(int)
        np.testing.assert_array_equal(rows['symbol'][payload], sent[indices])

    def test_tail_is_learned_without_initial_value_or_search_bound(self):
        fs, rate = 15.36e6, 169971.671
        for tail in (0., .12, .65, 1.05, 1.79, 2.9):
            with self.subTest(tail=tail):
                chosen = config()
                if tail > 2:
                    chosen.update(step_percent=4, low_guard_chips=14)
                c, sent, rises, falls, t, high, origin = synthesize(chosen, tail_us=tail)
                edge_t = np.r_[rises, falls]
                after = np.r_[np.ones(len(rises), bool), np.zeros(len(falls), bool)]
                order = np.argsort(edge_t)
                rows, _, calibrated, learned = decode_pulses(high, edge_t[order], after[order], t, fs, rate, c)
                self.assertTrue(all(abs(item['tail_us']-tail) < 1e-6 for item in calibrated))
                self.assertNotIn('tail_us', learned)
                payload = rows['frame_part'] == '净荷'
                self.assertTrue(np.all(rows['valid'][payload]))


if __name__ == '__main__':
    unittest.main()

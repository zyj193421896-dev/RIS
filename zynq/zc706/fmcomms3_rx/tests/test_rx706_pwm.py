"""Protocol and timing checks with known, independently generated RF IQ."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin, AnalysisCancelled
from rx706_pwm import recover_pwm, _fit_clock, _deglitch_runs, _candidate_decisions, FRAME_DTYPE


class PWMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'pwm.bin'

    def tearDown(self):
        self.tmp.cleanup()

    def capture(self, *, fs=30e6, rate=1e6/3, guard=2, widths=None,
                noise=2., phase=177., first=0, constant=False):
        if widths is None:
            widths = np.tile([1, 8, 2, 7, 3, 6, 4, 5, 8, 1, 3, 7], 24)
        self.widths = np.asarray(widths)
        self.rate, self.fs = rate, fs
        self.origin = .23/rate
        count = int(np.ceil((len(widths)+.8)*fs/rate))
        t = np.arange(count)/fs
        index = np.floor((t-self.origin)*rate).astype(int)
        within = ((t-self.origin)*rate-index)*(8+guard)
        high = (index >= 0) & (index < len(widths))
        high &= within < self.widths[np.clip(index, 0, len(widths)-1)]
        carrier = 1812345.67
        state = np.where(high, 300*np.exp(1j*np.deg2rad(phase)), 300)
        if constant:
            state[:] = 300
        rng = np.random.default_rng(42)
        x = state*np.exp(1j*(2*np.pi*carrier*t+.43+150*t*t))
        x += noise*(rng.normal(size=count)+1j*rng.normal(size=count))
        np.rint(np.column_stack((x.real, x.imag, x.real, x.imag))).astype('<i2').tofile(self.path)
        return analyze_bin(self.path, nfft=4096, averages=1, sample_rate_hz=fs, first_tuple=first)

    def check_decode(self, recovery):
        rows = recovery.frames
        active = (rows['start_s'] >= self.origin-.1/self.rate) & (
            rows['start_s'] < self.origin+(len(self.widths)-.2)/self.rate)
        indices = np.rint((rows['start_s'][active]-self.origin)*self.rate).astype(int)
        selected = rows[active]
        expected = self.widths[indices]
        data = (expected >= 1) & (expected <= 8)
        self.assertGreater(len(selected), 50)
        self.assertTrue(np.all(selected['valid'][data]), selected[~selected['valid']][:5])
        np.testing.assert_array_equal(selected['symbol'][data], expected[data]-1)
        # Crossing-time quantization is bounded by one input sample per width.
        tolerance = (8+recovery.guard_chips)*self.rate/recovery.output_rate_hz+.02
        np.testing.assert_allclose(selected['high_chips'][data], expected[data], atol=tolerance)
        np.testing.assert_allclose(selected['one_us']+selected['zero_us'],
                                   1e6/self.rate, atol=1e6/self.fs*1.5)

    def test_equal_energy_phase_states_all_eight_symbols(self):
        r = recover_pwm(self.capture(), duration_s=.01)
        self.check_decode(r)
        self.assertEqual(r.guard_chips, 2)
        self.assertLess(abs(r.amplitude_difference_db), .1)
        self.assertGreater(abs(r.phase_difference_deg), 170)
        self.assertEqual(r.decimation, 1)
        self.assertEqual(r.output_rate_hz, self.fs)

    def test_fractional_samples_per_slot_and_mid_symbol_capture(self):
        s = self.capture(fs=15.36e6, first=37)
        r = recover_pwm(s, duration_s=.01)
        self.check_decode(r)

    def test_400k_with_3p84_samples_per_slot_and_interpolated_edges(self):
        s = self.capture(fs=15.36e6, rate=400000)
        r = recover_pwm(s, symbol_hz=400000, duration_s=.01)
        self.check_decode(r)
        self.assertTrue(r.decode_locked)
        self.assertIn('3.840', r.sampling_note)
        with self.assertRaisesRegex(ValueError, '每片只有1.920'):
            recover_pwm(s, symbol_hz=400000, decimation=2)

    def test_startup_burst_uses_later_seed_without_removing_samples(self):
        s = self.capture(fs=15.36e6, rate=400000, phase=180,
                         widths=np.tile([1, 8, 2, 7, 3, 6, 4, 5], 300))
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1, 4)
        end = round(.0004*self.fs)
        burst = np.random.default_rng(29).normal(0, 700, (end, 2))
        raw[:end, :2] = np.clip(np.rint(burst), -2000, 2000).astype('<i2')
        raw.tofile(self.path)
        original = self.path.read_bytes()
        r = recover_pwm(s, symbol_hz=400000, duration_s=.01)
        self.assertGreaterEqual(r.calibration_start_s, .0004)
        self.assertEqual(r.first_tuple, 0)
        self.assertEqual(r.input_samples, len(raw))
        self.assertLess(r.frames['start_s'][0], .0001)
        selected = r.frames[(r.frames['start_s'] > .0011) & r.frames['valid']]
        indices = np.rint((selected['start_s']-self.origin)*self.rate).astype(int)
        self.assertGreater(len(selected), 1500)
        np.testing.assert_array_equal(selected['symbol'], self.widths[indices]-1)
        self.assertEqual(self.path.read_bytes(), original)

    def test_partial_state_recovery_publishes_no_bits_when_code_is_unlocked(self):
        # Physical two-state PWM and a good symbol clock, but half-slot widths
        # cannot carry the integer-width alphabet used by this decoder.
        s = self.capture(phase=180, widths=np.tile(np.arange(1.5, 9), 36))
        with self.assertRaisesRegex(ValueError, '脉宽未通过|映射有歧义'):
            recover_pwm(s, duration_s=.01)
        r = recover_pwm(s, duration_s=.01, allow_unlocked=True)
        self.assertFalse(r.decode_locked)
        self.assertTrue(r.decode_note)
        self.assertFalse(np.any(r.frames['valid']))
        self.assertTrue(np.all(r.frames['symbol'] == -1))
        self.assertTrue(np.all(r.frames['bits'] == '???'))
        self.assertGreater(np.count_nonzero(r.frames['candidate_symbol'] >= 0), 200)
        self.assertTrue(np.any(r.frames['candidate_bits'] == '111'))
        self.assertEqual(r.decoded_bit_rate, 0)
        self.assertGreater(r.separation, 3.5)
        self.assertEqual(set(r.levels), {False, True})

    def test_candidates_do_not_replace_invalid_decisions_or_idle(self):
        frames = np.zeros(4, dtype=FRAME_DTYPE)
        frames['symbol'] = [-1, -1, -2, 5]
        frames['bits'] = ['???', '???', '-', '101']
        frames['valid'] = [False, False, False, True]
        frames['high_chips'] = [9.2, np.nan, 0, 6.01]
        frames['pattern'] = ['1111111110', '1010100000', '0000000000', '1111110000']
        _candidate_decisions(frames, 8)
        np.testing.assert_array_equal(frames['candidate_symbol'], [7, 2, -2, 5])
        np.testing.assert_array_equal(frames['candidate_bits'], ['111', '010', '-', '101'])
        self.assertIn('限幅', frames['candidate_source'][0])
        self.assertIn('缺主脉宽', frames['candidate_source'][1])
        np.testing.assert_array_equal(frames['bits'], ['???', '???', '-', '101'])
        np.testing.assert_array_equal(frames['valid'], [False, False, False, True])

    def test_antipodal_states_do_not_lock_to_half_sample_rate_alias(self):
        r = recover_pwm(self.capture(phase=180), duration_s=.01)
        self.check_decode(r)
        self.assertAlmostEqual(r.carrier_hz, 1812345.67, delta=20)

    def test_one_guard_slot_is_still_configurable(self):
        r = recover_pwm(self.capture(guard=1), guard_chips=1, duration_s=.01)
        self.check_decode(r)

    def test_nonopposite_phase_states_and_antialias_decimation(self):
        s = self.capture(rate=10000, fs=1.536e6, phase=90)
        r = recover_pwm(s, symbol_hz=10000, duration_s=.025, decimation=2)
        self.check_decode(r)
        self.assertEqual(r.output_rate_hz, self.fs/2)
        self.assertAlmostEqual(abs(r.phase_difference_deg), 90, delta=3)

    def test_idle_and_bad_guard_are_not_valid_payload(self):
        widths = np.tile([1, 3, 0, 0, 7, 8, 2, 9, 5, 6], 24)
        s = self.capture(widths=widths)
        r = recover_pwm(s, duration_s=.01)
        self.assertGreater(np.count_nonzero(r.frames['symbol'] == -2), 5)
        self.assertGreater(np.count_nonzero(r.frames['bits'] == '???'), 5)
        self.assertTrue(np.all(~r.frames['valid'][r.frames['symbol'] < 0]))

    def test_constant_width_requires_explicit_polarity(self):
        s = self.capture(widths=np.full(250, 4))
        with self.assertRaisesRegex(ValueError, '映射有歧义'):
            recover_pwm(s, duration_s=.01)
        r = recover_pwm(s, duration_s=.01, state_one='B')
        # User may select either label; mapping is disclosed, never guessed.
        self.assertIn('用户指定', r.mapping_note)
        self.assertIn('B=1', r.mapping_note)

    def test_equal_duty_polarity_is_unlabelled_but_same_payload(self):
        r = recover_pwm(self.capture(widths=np.full(250, 5)), duration_s=.01)
        self.assertIn('真实控制 0/1 未标定', r.mapping_note)
        self.assertTrue(np.all(r.frames['symbol'][r.frames['valid']] == 4))
        self.assertTrue(np.all(r.frames['bits'][r.frames['valid']] == '100'))

    def test_long_clock_fit_does_not_count_spurious_edges_as_symbols(self):
        rng = np.random.default_rng(19)
        true_rate = 99995.53
        rise = 1.3e-6+np.arange(20000)/true_rate+rng.normal(0, 5e-9, 20000)
        extra = rise[rng.choice(len(rise)-1, 320, replace=False)]+rng.uniform(.05, .95, 320)/true_rate
        period, anchor = _fit_clock(np.sort(np.r_[rise, extra]), 100000, 10, None)
        self.assertAlmostEqual(1/period, true_rate, delta=.1)
        self.assertAlmostEqual(anchor, rise[0], delta=2e-8)

    def test_glitch_keeps_bad_frame_and_does_not_shift_following_payload(self):
        s = self.capture()
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1, 4)
        for symbol in range(15, 240, 4):
            at = int(round((self.origin+(symbol+.72)/self.rate)*self.fs))
            raw[at] = -raw[at]
        raw.tofile(self.path)
        r = recover_pwm(s, duration_s=.01, state_one='A', glitch_tolerance_chips=0)
        good = r.frames[r.frames['valid']]
        indexes = np.rint((good['start_s']-self.origin)*self.rate).astype(int)
        expected = self.widths[indexes]-1
        # Explicit A is a relative label; select its complement if necessary.
        observed = good['symbol']
        if not np.array_equal(observed, expected):
            r = recover_pwm(s, duration_s=.01, state_one='B', glitch_tolerance_chips=0)
            good = r.frames[r.frames['valid']]
            indexes = np.rint((good['start_s']-self.origin)*self.rate).astype(int)
            np.testing.assert_array_equal(good['symbol'], self.widths[indexes]-1)
        self.assertGreater(np.count_nonzero(~r.frames['valid']), 10)
        self.assertGreater(len(r.frames), 280)

    def test_closed_short_excursion_is_removed_but_real_slot_is_not(self):
        # Units are slots; a 0.05-slot high excursion inside the low guard.
        edges = np.array([0., 8., 8.35, 8.40, 10.])
        result = _deglitch_runs(edges, [True, False, True, False], .15, .30)
        np.testing.assert_allclose(result[0], [0, 8, 10])
        np.testing.assert_array_equal(result[1], [True, False])
        self.assertEqual(result[2], 1)
        self.assertAlmostEqual(result[3], .05)
        # A whole-slot excursion and a legitimate 1-slot PWM pulse survive.
        wide = _deglitch_runs([0, 2, 3, 8, 10], [True, False, True, False], .15, .30)
        self.assertEqual(len(wide[1]), 4)
        pulse = _deglitch_runs([0, 1, 10], [True, False], .15, .30)
        np.testing.assert_allclose(pulse[0], [0, 1, 10])

    def test_many_short_excursions_exceed_the_per_symbol_budget(self):
        result = _deglitch_runs([0, 2, 2.12, 4, 4.12, 6, 6.12, 8, 10],
                               [True, False, True, False, True, False, True, False], .15, .30)
        self.assertTrue(result[-1])
        self.assertEqual(result[2], 2)

    def test_short_glitches_keep_payload_with_an_independent_warning(self):
        s = self.capture()
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1, 4)
        for symbol in range(15, 240, 4):
            at = int(round((self.origin+(symbol+.72)/self.rate)*self.fs))
            raw[at] = -raw[at]
        raw.tofile(self.path)
        strict = recover_pwm(s, duration_s=.01, glitch_tolerance_chips=0)
        one = 'AB'[strict.state_one]
        recovered = recover_pwm(s, duration_s=.01, state_one=one)
        warning = recovered.frames['valid'] & (recovered.frames['glitch_count'] > 0)
        self.assertGreater(np.count_nonzero(warning), 10)
        self.assertTrue(np.all(recovered.frames['warning'][warning] != ''))
        self.assertEqual(len(strict.frames), len(recovered.frames))
        self.assertTrue(np.any(~strict.frames['valid'][warning]))
        good = recovered.frames[recovered.frames['valid']]
        indexes = np.rint((good['start_s']-self.origin)*self.rate).astype(int)
        np.testing.assert_array_equal(good['symbol'], self.widths[indexes]-1)

    def test_wrong_symbol_rate_has_measured_clock_hint(self):
        s = self.capture(rate=100000, fs=15.36e6)
        with self.assertRaisesRegex(ValueError, '实测稳定同向边沿'):
            recover_pwm(s, duration_s=.01, symbol_hz=1e6/3)

    def test_tone_noise_bad_parameters_and_cancellation(self):
        with self.assertRaises(ValueError):
            recover_pwm(self.capture(constant=True), duration_s=.01)
        s = self.capture()
        for options in [dict(symbol_hz=0), dict(guard_chips=0), dict(decimation=16),
                        dict(width_tolerance=.5), dict(state_one='wrong'),
                        dict(glitch_tolerance_chips=-.1), dict(glitch_tolerance_chips=.5)]:
            with self.assertRaises(ValueError):
                recover_pwm(s, **options)
        event = threading.Event(); event.set()
        with self.assertRaises(AnalysisCancelled):
            recover_pwm(s, cancel_event=event)


if __name__ == '__main__':
    unittest.main()

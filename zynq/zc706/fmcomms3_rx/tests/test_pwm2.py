"""Independent TX synthesis for legacy 20-slot and configurable 50-slot PWM."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2, pwm2_layout, _decode, _candidate_decisions, FRAME_DTYPE
from rx706_pwm_reference import check_reference, parse_reference


class PWM2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'pwm2.bin'

    def tearDown(self):
        self.tmp.cleanup()

    def capture(self, *, fs=15.36e6, rate=100000, delay=0., end=None, ks=None,
                total=20, prefix=8, guard=2):
        code = total-prefix-guard
        self.ks = np.tile(np.arange(code+1), 30) if ks is None else np.asarray(ks)
        self.fs, self.rate = fs, rate
        self.origin = delay+.23/rate
        length = self.origin+(len(self.ks)+.8)/rate if end is None else end
        t = np.arange(int(np.ceil(length*fs)))/fs
        idx = np.floor((t-self.origin)*rate).astype(int)
        within = ((t-self.origin)*rate-idx)*total
        high = (idx >= 0) & (idx < len(self.ks))
        high &= within < prefix+self.ks[np.clip(idx, 0, len(self.ks)-1)]
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        if delay:
            x[t < delay] = 0
        rng = np.random.default_rng(26)
        x += 2*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
        raw = np.zeros((len(t),4), dtype='<i2')
        raw[:,0], raw[:,1] = np.rint(x.real), np.rint(x.imag)
        raw.tofile(self.path)
        return analyze_bin(self.path, nfft=4096, averages=1, sample_rate_hz=fs)

    def check_payload(self, r):
        rows = r.frames
        mask = ((rows['start_s'] >= self.origin-.1/self.rate) &
                (rows['start_s'] < self.origin+(len(self.ks)-.2)/self.rate))
        rows = rows[mask]
        ids = np.rint((rows['start_s']-self.origin)*self.rate).astype(int)
        self.assertGreater(len(rows), 50)
        self.assertTrue(np.all(rows['valid']), rows[~rows['valid']][:5])
        np.testing.assert_array_equal(rows['symbol'], self.ks[ids])
        np.testing.assert_array_equal(rows['bits'], ['1'*k+'0'*(r.code_chips-k) for k in self.ks[ids]])
        self.assertEqual(r.kind, 'pwm2')
        self.assertEqual(r.total_chips, r.prefix_chips+r.code_chips+r.guard_chips)
        self.assertEqual(r.bits_per_symbol, r.code_chips)
        np.testing.assert_allclose(rows['high_chips'], r.prefix_chips+self.ks[ids], atol=.3)

    def test_all_eleven_states_and_fractional_timing(self):
        r = recover_pwm2(self.capture(), symbol_hz=100000, duration_s=.01)
        self.check_payload(r)
        self.assertEqual(set(r.frames['symbol'][r.frames['valid']]), set(range(11)))

    def test_late_payload_search_and_gui_reference(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import PWM2_MODE
        ks = np.tile(np.arange(11), 40)
        self.capture(fs=8e6, delay=.50237, ks=ks)
        view = headless_window()
        view.constellation_mode.set(PWM2_MODE)
        view._analyze_worker(self.path, dict(nfft=4096, averages=1, sample_rate_hz=self.fs),
            dict(kind='pwm2', symbol_hz=self.rate, duration_s=1., auto_search=True,
                 reference_options=dict(sequence=tuple(range(11)), sync_symbols=64)))
        view.poll()
        self.assertIsNotNone(view.recovery, view.status.get())
        self.check_payload(view.recovery)
        self.assertTrue(view.recovery.reference_check.locked, view.summary.get())
        self.assertEqual(view.recovery.reference_check.bit_errors, 0)
        self.assertIn('log2(11)', view.summary.get())
        self.assertIn('version2', view.wave_axes.get_title())
        self.assertEqual(view.recovery.prefix_chips, 8)
        self.assertEqual(len(view.figure.axes), 3)

    def test_fixed_protection_and_idle(self):
        fs, rate = 400000, 1000
        widths = np.tile([8, 9, 10, 12, 15, 18, 7, 19, 0], 20)
        t = np.arange(len(widths)*400)/fs
        slot = np.arange(len(t))/20
        frame = np.arange(len(t))//400
        states = (slot % 20) < widths[frame]
        changes = np.flatnonzero(states[1:] != states[:-1])+1
        rows, _ = _decode(states, t[changes], states[changes], t, fs, rate,
                          10, 2, .3, None, prefix_chips=8)
        k = np.rint(rows['start_s']*rate).astype(int)
        w = widths[k]
        # The capture starts already high, so its very first rise is absent.
        self.assertTrue(np.all(rows['valid'][(w >= 8) & (w <= 18) & (k > 0)]))
        self.assertTrue(np.all(~rows['valid'][(w == 7) | (w == 19)]))
        self.assertTrue(np.all(rows['symbol'][w == 0] == -2))
        self.assertTrue(np.all(rows['bits'][(w == 8) & (k > 0)] == '0000000000'))
        self.assertTrue(np.all(rows['bits'][w == 18] == '1111111111'))

    def test_reference_uses_ten_slot_words_not_binary_k(self):
        frames = np.zeros(128, dtype=FRAME_DTYPE)
        frames['index'] = np.arange(128)
        frames['valid'] = True
        frames['symbol'] = np.tile(np.arange(11), 12)[:128]
        frames['symbol'][80] = (frames['symbol'][80]+1) % 11
        words = tuple('1'*k+'0'*(10-k) for k in range(11))
        r = check_reference(frames, tuple(range(11)), bits_per_symbol=10, symbol_bitwords=words)
        self.assertTrue(r.locked)
        self.assertEqual(r.symbol_errors, 1)
        self.assertEqual(r.bit_errors, 1)
        self.assertEqual(r.compared_bits, 640)
        self.assertEqual(parse_reference('0 5 10', 11), (0, 5, 10))
        with self.assertRaises(ValueError):
            parse_reference('11', 11)

    def test_candidates_keep_ten_bits_and_do_not_validate_bad_guards(self):
        frames = np.zeros(3, dtype=FRAME_DTYPE)
        frames['symbol'] = -1
        frames['high_chips'] = [8.1, 18., 19.]
        _candidate_decisions(frames, 10, 8)
        np.testing.assert_array_equal(frames['candidate_symbol'], [0, 10, 10])
        np.testing.assert_array_equal(frames['candidate_bits'], ['0000000000', '1111111111', '1111111111'])
        self.assertFalse(np.any(frames['valid']))

    def test_ambiguous_polarity_does_not_publish_confirmed_words(self):
        spectrum = self.capture(ks=np.zeros(300, dtype=int))
        with self.assertRaisesRegex(ValueError, '映射有歧义'):
            recover_pwm2(spectrum, symbol_hz=self.rate)
        result = recover_pwm2(spectrum, symbol_hz=self.rate, allow_unlocked=True)
        self.assertFalse(result.decode_locked)
        self.assertFalse(np.any(result.frames['valid']))
        self.assertTrue(np.all(result.frames['bits'] == '??????????'))
        self.assertTrue(np.all(np.char.str_len(result.frames['candidate_bits']) == 10))

    def test_fifty_slots_variable_guards_and_long_codewords(self):
        for prefix, guard in ((20, 5), (8, 2), (5, 25)):
            with self.subTest(prefix=prefix, guard=guard):
                s = self.capture(fs=20e6, total=50, prefix=prefix, guard=guard)
                r = recover_pwm2(s, total_chips=50, prefix_chips=prefix,
                                 guard_chips=guard, symbol_hz=self.rate)
                self.check_payload(r)
                self.assertEqual(r.total_chips, 50)
                self.assertEqual(r.prefix_chips, prefix)
                code = 50-prefix-guard
                self.assertEqual(r.output_rate_hz/(self.rate*50), 4)
                valid = r.frames[r.frames['valid']]
                self.assertEqual(set(valid['symbol']), set(range(code+1)))
                self.assertTrue(np.all(np.char.str_len(valid['pattern']) == 50))
                self.assertTrue(np.all(np.char.str_len(valid['bits']) == code))
                self.assertIn('1'*code, valid['bits'])
                self.assertIn('0'*code, valid['bits'])
                check = check_reference(r.frames, tuple(range(code+1)), bits_per_symbol=code,
                    symbol_bitwords=r.symbol_bitwords, sync_symbols=128)
                self.assertTrue(check.locked, check.summary())
                self.assertEqual(check.bit_errors, 0)

    def test_configurable_gui_summary_and_search(self):
        from test_signal_analysis_view import headless_window, Value
        from rx706_spectrum_gui import PWM2_MODE
        self.capture(fs=15.36e6, delay=.02237, total=50, prefix=20, guard=5,
                     ks=np.tile(np.arange(26), 15))
        view = headless_window()
        view.constellation_mode.set(PWM2_MODE)
        view._analyze_worker(self.path, dict(nfft=4096, averages=1, sample_rate_hz=self.fs),
            dict(kind='pwm2', symbol_hz=self.rate, duration_s=.1, auto_search=True,
                 total_chips=50, prefix_chips=20, guard_chips=5,
                 reference_options=dict(sequence=tuple(range(26)), sync_symbols=64)))
        view.poll()
        self.assertIsNotNone(view.recovery, view.status.get())
        self.check_payload(view.recovery)
        self.assertTrue(view.recovery.reference_check.locked)
        self.assertEqual(view.recovery.reference_check.bit_errors, 0)
        for text in ('25片可调', '步进2%', 'log2(26)', '中间25片码型'):
            self.assertIn(text, view.summary.get())
        self.assertEqual(view.recovery.prefix_chips, 20)
        self.assertEqual(len(view.figure.axes), 3)
        view.pwm2_total_chips, view.pwm2_prefix_chips, view.pwm2_guard_chips = Value('50'), Value('20'), Value('5')
        view.pwm2_layout_summary, view.pwm2_preset = Value(''), Value('自定义')
        view.pwm_symbol_khz, view.pwm_decimation = Value('100'), Value('1')
        view.rate.set('15.36')
        view.update_pwm2_layout()
        for text in ('26态', '步进2%', '3.072'):
            self.assertIn(text, view.pwm2_layout_summary.get())
        view.pwm2_preset.set('20片：8+10+2')
        view.apply_pwm2_preset()
        self.assertEqual(view.pwm2_total_chips.get(), '20')
        self.assertIn('11态', view.pwm2_layout_summary.get())
        view.pwm2_total_chips.set('50')
        view.update_pwm2_layout()
        self.assertEqual(view.pwm2_preset.get(), '自定义')
        self.assertIn('41态', view.pwm2_layout_summary.get())

    def test_invalid_layout_and_insufficient_sampling_are_rejected(self):
        for config in ((50, 0, 2), (50, 49, 2), (257, 8, 2), (50, 8, -1), (50., 8, 2)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                pwm2_layout(*config)
        s = self.capture(total=50, prefix=20, guard=5)
        with self.assertRaisesRegex(ValueError, '每片只有1.536'):
            recover_pwm2(s, total_chips=50, prefix_chips=20, guard_chips=5,
                         symbol_hz=self.rate, decimation=2)

    def test_clipped_search_probe_does_not_reject_acceptable_full_capture(self):
        s = self.capture(fs=1.536e6, rate=10000, total=25, prefix=5, guard=8)
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1,4)
        # Startup 10ms reaches 0.156%; full capture remains below 0.1%.
        raw[np.arange(0,15360,640),0] = 2047
        raw.tofile(self.path)
        before = self.path.read_bytes()
        r = recover_pwm2(s, total_chips=25, prefix_chips=5, guard_chips=8,
                         symbol_hz=self.rate, auto_search=True)
        self.assertTrue(r.decode_locked)
        self.assertEqual(r.first_tuple, 0)
        self.assertEqual(r.input_samples, len(raw))
        self.assertEqual(r.clipped, 24)
        self.assertLess(r.clipped/r.input_samples, .001)
        self.assertIn('0.156%', r.search_note)
        self.assertEqual(self.path.read_bytes(), before)

    def test_full_span_clipping_still_blocks_final_decoding(self):
        s = self.capture(fs=1.536e6, rate=10000, total=25, prefix=5, guard=8)
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1,4)
        raw[::200,0] = 2047
        raw.tofile(self.path)
        with self.assertRaisesRegex(ValueError, '削顶'):
            recover_pwm2(s, total_chips=25, prefix_chips=5, guard_chips=8,
                         symbol_hz=self.rate, auto_search=True)

    def test_search_probe_never_bypasses_invalid_twelve_bit_data(self):
        s = self.capture(fs=1.536e6, rate=10000, total=25, prefix=5, guard=8)
        raw = np.fromfile(self.path, dtype='<i2').reshape(-1,4)
        raw[0,0] = 4096
        raw.tofile(self.path)
        with self.assertRaisesRegex(ValueError, '12 位范围'):
            recover_pwm2(s, total_chips=25, prefix_chips=5, guard_chips=8,
                         symbol_hz=self.rate, auto_search=True)


if __name__ == '__main__':
    unittest.main()

"""Causal 20% pilot calibration with independent distorted RF synthesis."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm import _frame_dtype, recover_pwm2, _candidate_decisions, pattern_symbol, decision_width_chips
from rx706_pwm_training import apply_training, validate_training
from rx706_spectrum import analyze_bin
from rx706_pwm_reference import check_reference


def measurements(ks, bias=.6):
    ks = np.asarray(ks)
    rows = np.zeros(len(ks), dtype=_frame_dtype(12, 25, True))
    rows['index'] = np.arange(len(ks))+1
    rows['start_s'] = np.arange(len(ks))*1e-5
    rows['high_chips'] = 5+ks+bias
    rows['one_us'] = rows['high_chips']*.4
    rows['zero_us'] = 10-rows['one_us']
    rows['structure_ok'] = True
    rows['symbol'] = -1
    rows['reason'] = 'uncalibrated width'
    return rows


class TrainingTests(unittest.TestCase):
    def test_raw_pattern_six_can_disagree_with_width_candidate_seven(self):
        rows = np.zeros(3, dtype=_frame_dtype(12,25,True))
        rows['symbol'] = -1
        rows['high_chips'] = [11.7380,12.6608,13.6743]
        rows['calibration_id'] = 1
        rows['width_correction_chips'] = -.2
        rows['corrected_high_chips'] = rows['high_chips']-.2
        rows['pattern'] = ['1'*n+'0'*(25-n) for n in (11,12,13)]
        _candidate_decisions(rows,12,5)
        # Width estimates straddle rounding boundaries; raw slot voting is
        # a separate diagnostic and must never silently overwrite the result.
        np.testing.assert_array_equal(rows['candidate_symbol'], [7,7,8])
        self.assertEqual([pattern_symbol(p,25,12,5) for p in rows['pattern']], [6,7,8])
        np.testing.assert_allclose([decision_width_chips(row) for row in rows], [11.538,12.4608,13.4743])
        self.assertIn('11.5380', rows['candidate_source'][0])
        self.assertIn('码型推断6', rows['candidate_source'][0])
        self.assertFalse(np.any(rows['valid']))
        self.assertEqual(pattern_symbol('101'+'0'*22,25,12,5), -1)
        self.assertEqual(pattern_symbol('1'*11,25,12,5), -1)

    def test_offset_is_learned_only_from_preceding_pilots(self):
        ks = [2, 4, 6, 0, 0, 0, 1, 3, 7, 12, 5, 2]
        for bias in (.6, -1.2):
            with self.subTest(bias=bias):
                rows = measurements(ks, bias)
                original = rows[['one_us', 'zero_us', 'high_chips']].copy()
                pilots = apply_training(rows, 25, 5, 12, 8, .3, {})
                self.assertEqual(len(pilots), 1)
                np.testing.assert_array_equal(rows['training'], [False]*3+[True]*3+[False]*6)
                self.assertFalse(np.any(rows['valid'][:6]))
                self.assertTrue(np.all(rows['valid'][6:]))
                np.testing.assert_array_equal(rows['symbol'][6:], ks[6:])
                np.testing.assert_allclose(rows['width_correction_chips'][3:], -bias)
                for key in ('one_us', 'zero_us', 'high_chips'):
                    np.testing.assert_array_equal(rows[key], original[key])
                altered = measurements(ks, bias)
                altered['high_chips'][-1] += .4
                apply_training(altered, 25, 5, 12, 8, .3, {})
                np.testing.assert_allclose(altered['width_correction_chips'], rows['width_correction_chips'], equal_nan=True)
                self.assertFalse(altered['valid'][-1])

    def test_new_pilot_updates_offset_without_future_leakage(self):
        ks = [0, 0, 0, 1, 5, 8, 2, 0, 0, 0, 12, 6, 4]
        bias = np.r_[np.full(7,.6), np.full(6,-.8)]
        rows = measurements(ks, bias)
        pilots = apply_training(rows, 25, 5, 12, 8, .3, {})
        self.assertEqual(len(pilots), 2)
        np.testing.assert_allclose(rows['width_correction_chips'][:7], -.6)
        np.testing.assert_allclose(rows['width_correction_chips'][7:], .8)
        self.assertTrue(np.all(rows['valid'][[3,4,5,6,10,11,12]]))
        np.testing.assert_array_equal(rows['symbol'][[3,4,5,6,10,11,12]], np.array(ks)[[3,4,5,6,10,11,12]])

    def test_no_pilot_and_broken_edges_are_not_repaired(self):
        rows = measurements([0, 0, 0, 1, 4, 9])
        rows['structure_ok'][1] = False
        self.assertEqual(apply_training(rows,25,5,12,8,.3,{}), [])
        self.assertFalse(np.any(rows['valid']))
        rows = measurements([0,0,0,1,5,8])
        rows['structure_ok'][4] = False
        apply_training(rows,25,5,12,8,.3,{})
        self.assertFalse(rows['valid'][4])
        self.assertTrue(rows['valid'][3] and rows['valid'][5])
        with self.assertRaises(ValueError):
            validate_training(20,8,10,{})

    def test_any_equal_triple_is_a_marker_but_long_run_does_not_retrigger(self):
        rows = measurements([7,7,7,7,7,7,2,4])
        pilots = apply_training(rows,25,5,12,8,.3,{})
        self.assertEqual(len(pilots), 1)
        np.testing.assert_array_equal(rows['training'], [True,True,True,False,False,False,False,False])
        self.assertAlmostEqual(pilots[0]['correction_chips'], -7.6)

    def make_capture(self, path, bias=.6, delay=0., groups=40):
        fs, rate, total = 5e6, 10000, 25
        payload = [1,3,7,12,4,9]
        ks = np.tile([0,0,0]+payload, groups)
        origin = delay+.23/rate
        t = np.arange(round((origin+(len(ks)+.8)/rate)*fs))/fs
        index = np.floor((t-origin)*rate).astype(int)
        within = ((t-origin)*rate-index)*total
        group_bias = bias+.08*np.sin(np.arange(groups)*.2)
        widths = 5+ks+np.repeat(group_bias,9)
        high = (index >= 0) & (index < len(ks))
        high &= within < widths[np.clip(index,0,len(ks)-1)]
        x = np.where(high,-300,300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        if delay:
            x[t < delay] = 0
        rng = np.random.default_rng(72)
        x += rng.normal(0,1,len(t))+1j*rng.normal(0,1,len(t))
        raw = np.zeros((len(t),4), dtype='<i2')
        raw[:,0], raw[:,1] = np.rint(x.real), np.rint(x.imag)
        raw.tofile(path)
        return analyze_bin(path,nfft=4096,averages=1,sample_rate_hz=fs), origin, ks, payload

    def test_distorted_rf_and_gui_reference_exclude_pilots(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import PWM2_MODE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pilots.bin'
            s, origin, ks, payload = self.make_capture(path)
            before = path.read_bytes()
            r = recover_pwm2(s,total_chips=25,prefix_chips=5,guard_chips=8,
                             symbol_hz=10000,training_options={})
            self.assertTrue(r.decode_locked)
            self.assertEqual(r.training_calibration['updates'],40)
            self.assertEqual(int(r.frames['training'].sum()),120)
            active = r.frames['valid']
            ids = np.rint((r.frames['start_s'][active]-origin)*10000).astype(int)
            np.testing.assert_array_equal(r.frames['symbol'][active],ks[ids])
            self.assertEqual(int(active.sum()),240)
            check = check_reference(r.frames,payload,bits_per_symbol=12,symbol_bitwords=r.symbol_bitwords)
            self.assertTrue(check.locked,check.summary())
            self.assertEqual(check.bit_errors,0)
            self.assertEqual(check.erasures,0)
            self.assertEqual(check.evaluation_symbols,240-64)
            self.assertTrue(np.all(check.expected[r.frames['training']] == -1))
            self.assertTrue(np.all(check.row_bit_errors[r.frames['training']] == -1))
            view = headless_window()
            view.constellation_mode.set(PWM2_MODE)
            view._analyze_worker(path,dict(nfft=4096,averages=1,sample_rate_hz=5e6),
                dict(kind='pwm2',symbol_hz=10000,total_chips=25,prefix_chips=5,guard_chips=8,
                     training_options={},reference_options=dict(sequence=payload,sync_symbols=64)))
            view.poll()
            self.assertIsNotNone(view.recovery,view.status.get())
            self.assertIn('20%三连训练补偿',view.summary.get())
            self.assertIn('P = pilot',view.wave_axes.get_title())
            self.assertEqual(path.read_bytes(),before)

    def test_search_keeps_all_three_pilots_and_supports_shrinkage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'delayed.bin'
            s, origin, ks, _ = self.make_capture(path,bias=-1.2,delay=.02237)
            r = recover_pwm2(s,total_chips=25,prefix_chips=5,guard_chips=8,
                             symbol_hz=10000,training_options={},auto_search=True)
            self.assertTrue(r.decode_locked)
            self.assertGreater(r.training_calibration['updates'],30)
            self.assertTrue(np.all(np.array(r.training_calibration['corrections_chips']) > 0))
            data = r.frames[r.frames['valid']]
            ids = np.rint((data['start_s']-origin)*10000).astype(int)
            np.testing.assert_array_equal(data['symbol'],ks[ids])
            self.assertLessEqual(ids.min(),3)


if __name__ == '__main__':
    unittest.main()

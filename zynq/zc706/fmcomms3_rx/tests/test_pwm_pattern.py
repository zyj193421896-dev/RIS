"""Pattern decisions must follow the raw legal word, not width rounding."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm import (_frame_dtype, apply_pattern_decisions, _candidate_decisions,
                      recover_pwm, recover_pwm2)
import test_pwm2 as capture_fixtures
import test_pwm_training as pilot_fixtures


def screenshot_rows():
    rows = np.zeros(3, dtype=_frame_dtype(12,25,True))
    rows['index'] = [32,33,34]
    rows['symbol'] = -1
    rows['structure_ok'] = True
    rows['high_chips'] = [11.7380,12.6608,13.6743]
    rows['calibration_id'] = 1
    rows['width_correction_chips'] = -.2282
    rows['corrected_high_chips'] = rows['high_chips']-.2282
    rows['pattern'] = ['1'*n+'0'*(25-n) for n in (11,12,13)]
    return rows


class PatternTests(unittest.TestCase):
    def test_screenshot_six_seven_eight_instead_of_seven_seven_eight(self):
        width = screenshot_rows()
        _candidate_decisions(width,12,5)
        np.testing.assert_array_equal(width['candidate_symbol'], [7,7,8])
        pattern = screenshot_rows()
        before = pattern['high_chips'].copy()
        apply_pattern_decisions(pattern,25,12,5,8,require_training=True)
        _candidate_decisions(pattern,12,5,decision_mode='pattern')
        self.assertTrue(np.all(pattern['valid']))
        np.testing.assert_array_equal(pattern['symbol'], [6,7,8])
        np.testing.assert_array_equal(pattern['candidate_symbol'], [6,7,8])
        np.testing.assert_array_equal(pattern['bits'], ['1'*k+'0'*(12-k) for k in (6,7,8)])
        np.testing.assert_array_equal(pattern['high_chips'], before)
        self.assertIn('不使用脉宽取整', pattern['candidate_source'][0])

    def test_invalid_shape_missing_edges_and_unlocked_pilots_stay_invalid(self):
        rows = screenshot_rows()
        rows['pattern'][0] = '111110111111'+'0'*13
        rows['structure_ok'][1] = False
        rows['calibration_id'][2] = 0
        apply_pattern_decisions(rows,25,12,5,8,require_training=True)
        self.assertFalse(np.any(rows['valid']))
        _candidate_decisions(rows,12,5,decision_mode='pattern')
        self.assertEqual(rows['candidate_symbol'][0], -1)
        self.assertIn('训练', rows['reason'][2])

    def test_configurations_and_gui_reference(self):
        from test_signal_analysis_view import headless_window
        from rx706_spectrum_gui import PWM2_MODE
        fixture = capture_fixtures.PWM2Tests()
        fixture.setUp()
        try:
            for total,prefix,guard,fs in ((25,5,8,15.36e6),(50,20,5,20e6),(20,8,2,15.36e6)):
                with self.subTest(total=total):
                    spectrum = fixture.capture(total=total,prefix=prefix,guard=guard,fs=fs)
                    r = recover_pwm2(spectrum,total_chips=total,prefix_chips=prefix,guard_chips=guard,
                                     symbol_hz=100000,decision_mode='pattern')
                    fixture.check_payload(r)
                    self.assertEqual(r.decision_mode,'pattern')
            fixture.capture(total=25,prefix=5,guard=8)
            view = headless_window()
            view.constellation_mode.set(PWM2_MODE)
            view._analyze_worker(fixture.path,dict(nfft=4096,averages=1,sample_rate_hz=fixture.fs),
                dict(kind='pwm2',total_chips=25,prefix_chips=5,guard_chips=8,symbol_hz=100000,
                     decision_mode='pattern',reference_options=dict(sequence=tuple(range(13)),sync_symbols=64)))
            view.poll()
            self.assertIsNotNone(view.recovery,view.status.get())
            self.assertTrue(view.recovery.reference_check.locked)
            self.assertEqual(view.recovery.reference_check.bit_errors,0)
            self.assertIn('时间片码型（不使用连续脉宽取整）',view.summary.get())
            self.assertEqual(view.recovery.decision_mode, 'pattern')
            with self.assertRaises(ValueError):
                recover_pwm(spectrum,decision_mode='pattern')
        finally:
            fixture.tearDown()

    def test_pilot_offsets_do_not_change_raw_pattern_decisions(self):
        rows = screenshot_rows()
        apply_pattern_decisions(rows,25,12,5,8,require_training=True)
        expected = rows['symbol'].copy()
        rows['corrected_high_chips'] += 3
        rows['width_correction_chips'] += 3
        apply_pattern_decisions(rows,25,12,5,8,require_training=True)
        np.testing.assert_array_equal(rows['symbol'],expected)

    def test_real_pipeline_keeps_pilot_exclusion_and_search(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            s,origin,ks,_ = pilot_fixtures.TrainingTests().make_capture(Path(directory)/'pilot.bin',bias=0.,delay=.02237)
            r = recover_pwm2(s,total_chips=25,prefix_chips=5,guard_chips=8,symbol_hz=10000,
                             decision_mode='pattern',training_options={},auto_search=True)
            self.assertTrue(r.decode_locked)
            self.assertIsNotNone(r.training_calibration)
            self.assertGreater(r.training_calibration['training_symbols'],90)
            valid = r.frames[r.frames['valid']]
            ids = np.rint((valid['start_s']-origin)*10000).astype(int)
            np.testing.assert_array_equal(valid['symbol'],ks[ids])
            self.assertFalse(np.any(r.frames['valid'][r.frames['training']]))


if __name__ == '__main__':
    unittest.main()

"""Duty UI semantics, independent RF synthesis, and causal calibration checks."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm import _frame_dtype, recover_pwm2
from rx706_pwm_protocol import duty_protocol, apply_protocol_training, _alphabet_scale
from rx706_pwm_reference import check_reference
from rx706_spectrum import analyze_bin


class ProtocolTests(unittest.TestCase):
    def test_twenty_step_seven_eight_symbols_ends_at_sixty_nine(self):
        options, values = duty_protocol('20', '7', '8')
        self.assertEqual(values, (20, 27, 34, 41, 48, 55, 62, 69))
        self.assertEqual(options['symbol_widths'], values)
        for args in (('20', '0', '8'), ('20', '12', '8'), ('0', '7', '8'),
                     ('20', '7', '8.5'), ('nan', '7', '8')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                duty_protocol(*args)

    def test_independent_preamble_ranges_and_fractional_widths(self):
        from rx706_pwm_protocol import preamble_duties, validate_protocol_training
        self.assertEqual(preamble_duties('24', '0', '3'), (24, 24, 24))
        self.assertEqual(preamble_duties('20', '7', '3'), (20, 27, 34))
        for args in (('24', '-1', '3'), ('90', '7', '3'), ('24', '0', '3.5'), ('nan', '0', '3')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                preamble_duties(*args)
        # Preamble lies below, and is independent of, this payload alphabet.
        options = dict(preamble_widths=(5, 6.75, 8.5))
        preamble, _, _, _ = validate_protocol_training(25, 10, 8, options)
        np.testing.assert_array_equal(preamble, [5, 6.75, 8.5])
        rows = np.zeros(6, dtype=_frame_dtype(8, 25, True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        truth = np.array([5, 6.75, 8.5, 10, 13, 18])
        rows['high_chips'] = truth*1.1+.3
        rows['clock_high_chips'] = rows['high_chips']
        rows['one_us'], rows['zero_us'] = 4, 6
        apply_protocol_training(rows, 25, 10, 8, 7, .3, options)
        self.assertTrue(np.all(rows['training'][:3]))
        self.assertTrue(np.all(rows['valid'][3:]))
        np.testing.assert_allclose(rows['corrected_high_chips'], truth)
        # No matching pilot must leave every payload unconfirmed.
        options['preamble_widths'] = (2, 2, 2)
        apply_protocol_training(rows, 25, 10, 8, 7, .3, options)
        self.assertFalse(np.any(rows['valid']))
        self.assertFalse(np.any(rows['calibration_id']))

    def test_nonlinear_centers_validate_pilot_in_same_model(self):
        from rx706_pwm import PWMRecovery
        widths = np.arange(20, 70, 7)
        centers = np.array([19.8,29.4,38.0,46.0,53.3,60.7,67.9,75.0])
        rng = np.random.default_rng(407)
        sent = np.concatenate([np.r_[[0,0,0], rng.permutation(8)] for _ in range(50)])
        rows = np.zeros(len(sent), dtype=_frame_dtype(49,100,True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        rows['high_chips'] = centers[sent]
        rows['clock_high_chips'] = rows['high_chips']
        rows['one_us'], rows['zero_us'] = 3, 3
        apply_protocol_training(rows,100,20,49,31,.3,
            dict(preamble_widths=(20,20,20), alphabet_widths=widths,
                 payload_symbols=8, clock_width=True, auto_scale=True))
        calibrated = np.arange(len(rows)) > 150
        data = calibrated & ~rows['training']
        np.testing.assert_allclose(rows['corrected_high_chips'][data], widths[sent[data]], atol=.05)
        self.assertTrue(np.any(rows['calibration_source'] == '前导＋档位标定'))
        display = PWMRecovery.__new__(PWMRecovery)
        display.output_binary, display.symbol_widths = True, tuple(widths)
        for k in range(8):
            self.assertEqual(display.duty_thermometer(dict(symbol=k, valid=True)), '1'*(k+1)+'0'*(7-k))
        self.assertEqual(display.duty_thermometer(dict(symbol=4,valid=False)), '?'*8)
        self.assertEqual(display.duty_thermometer(dict(symbol=-3,valid=False)), '-')

    def test_nearest_level_keeps_outliers_but_requires_lock_and_edges(self):
        from rx706_pwm_protocol import apply_alphabet_decisions
        rows = np.zeros(8, dtype=_frame_dtype(49,100,True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        rows['calibration_id'] = 1
        rows['corrected_high_chips'] = [30.522,38.367,75.291,16.,27.,34.,float('nan'),20.]
        rows['structure_ok'][4] = False
        rows['calibration_id'][5] = 0
        rows['symbol'][7] = -3
        apply_alphabet_decisions(rows,np.arange(20,70,7),20,49,.3,True,nearest=True)
        # Internal decisions are width minus prefix; recovery then maps labels.
        np.testing.assert_array_equal(rows['symbol'][:4], [14,21,49,0])
        self.assertTrue(np.all(rows['valid'][:4]))
        self.assertTrue(all('超容差' in text for text in rows['reason'][:4]))
        self.assertFalse(np.any(rows['valid'][4:]))
        self.assertEqual(rows['symbol'][7], -3)
        strict = rows.copy()
        apply_alphabet_decisions(strict,np.arange(20,70,7),20,49,.3,True)
        self.assertFalse(np.any(strict['valid']))

    def test_equal_preamble_uses_per_symbol_residual_not_peak_to_peak(self):
        # rx_160k.bin at 213906--213918 us: clock fit jitters slightly more
        # than .5 peak-to-peak although every symbol fits within .5 of median.
        rows = np.zeros(9, dtype=_frame_dtype(49,100,True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        rows['one_us'], rows['zero_us'] = 2, 4
        rows['high_chips'] = [76.39,18.52,20.428,20.496,20.843,28.781,37.42,46.467,54.058]
        rows['clock_high_chips'] = [75.03,18.52,21.1769502,20.6763833,20.9773100,29.38,38.,46.,54.]
        rows['structure_ok'][:2] = False
        original = rows[['high_chips','clock_high_chips']].copy()
        pilots = apply_protocol_training(rows,100,20,49,31,.3,
            dict(preamble_widths=(20,20,20), alphabet_widths=tuple(range(20,70,7)),
                 payload_symbols=8, clock_width=True, same_tolerance_chips=.5))
        self.assertEqual([(p['first_row'],p['last_row']) for p in pilots], [(2,4)])
        np.testing.assert_array_equal(rows['training'], [False,False,True,True,True,False,False,False,False])
        self.assertTrue(np.all(rows['calibration_id'][2:] == 1))
        np.testing.assert_allclose(rows['corrected_high_chips'][2:5],
                                   rows['clock_high_chips'][2:5]-(20.9773100-20))
        for key in original.dtype.names:
            np.testing.assert_array_equal(rows[key], original[key])

    def test_equal_preamble_rejects_bad_fit_offset_and_incomplete_edges(self):
        for values, broken in (((20.,20.6,21.2),False),
                               ((24.,24.1,24.2),False),
                               ((20.1,20.2,20.3),True),
                               ((20.1,float('nan'),20.3),False)):
            with self.subTest(values=values, broken=broken):
                rows = np.zeros(4, dtype=_frame_dtype(49,100,True))
                rows['symbol'] = -1
                rows['structure_ok'] = True
                rows['high_chips'] = rows['clock_high_chips'] = (*values,41.)
                rows['one_us'], rows['zero_us'] = 2,4
                if broken:
                    rows['structure_ok'][1] = False
                pilots = apply_protocol_training(rows,100,20,49,31,.3,
                    dict(preamble_widths=(20,20,20),clock_width=True,same_tolerance_chips=.5))
                self.assertEqual(pilots, [])
                self.assertFalse(np.any(rows['training']))
                self.assertFalse(np.any(rows['valid']))
                self.assertFalse(np.any(rows['calibration_id']))

    def test_first_three_pilots_allow_sampling_jitter_and_keep_fourth_as_payload(self):
        # First four 20% pulses in rx_160k.bin: three pilots then payload 0.
        rows = np.zeros(11, dtype=_frame_dtype(49,100,True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        rows['one_us'], rows['zero_us'] = 2,4
        rows['high_chips'] = [21.3195,19.8200,20.3763,19.1913,29.113,38.579,45.209,53.325,60.357,67.425,74.967]
        rows['clock_high_chips'] = [19.42075,20.23483,20.74484,19.29493,29.479,37.805,45.971,53.912,60.738,67.840,74.535]
        options = dict(preamble_widths=(20,20,20),alphabet_widths=tuple(range(20,70,7)),
                       payload_symbols=8,clock_width=True,same_tolerance_chips=.5)
        uncertainty = .5*100*169971.671/15.36e6
        original = rows.copy()
        pilots = apply_protocol_training(rows,100,20,49,31,.3,options,
                                         measurement_uncertainty_chips=uncertainty)
        self.assertEqual([(p['first_row'],p['last_row']) for p in pilots], [(0,2)])
        np.testing.assert_array_equal(rows['training'], [True]*3+[False]*8)
        self.assertEqual(rows['calibration_id'][3], 1)
        self.assertFalse(rows['training'][3])
        np.testing.assert_array_equal(rows['high_chips'],original['high_chips'])
        np.testing.assert_array_equal(rows['clock_high_chips'],original['clock_high_chips'])
        # Higher sample rates imply a smaller allowance, not a hardcoded loosening.
        narrow = original.copy()
        self.assertEqual(apply_protocol_training(narrow,100,20,49,31,.3,options,
            measurement_uncertainty_chips=uncertainty/10), [])
        # The same sampling margin cannot repair a missing preamble edge.
        broken = original.copy()
        broken['structure_ok'][1] = False
        broken['clock_high_chips'][3] = 30
        self.assertEqual(apply_protocol_training(broken,100,20,49,31,.3,options,
            measurement_uncertainty_chips=uncertainty), [])
        for margin in (-1,float('nan'),float('inf'),50):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                apply_protocol_training(original.copy(),100,20,49,31,.3,options,
                                        measurement_uncertainty_chips=margin)

    def test_sampling_allowance_does_not_lock_only_two_pilots_or_wrong_width(self):
        options = dict(preamble_widths=(20,20,20),payload_symbols=8,clock_width=True)
        for high in ((20.2,19.8,30.,40.), (27.,27.2,26.9,40.), (20.,21.8,23.6,40.)):
            with self.subTest(high=high):
                rows = np.zeros(4,dtype=_frame_dtype(49,100,True))
                rows['symbol'] = -1
                rows['structure_ok'] = True
                rows['high_chips'] = rows['clock_high_chips'] = high
                rows['one_us'], rows['zero_us'] = 2,4
                pilots = apply_protocol_training(rows,100,20,49,31,.3,options,
                                                 measurement_uncertainty_chips=.554)
                self.assertEqual(pilots, [])
                self.assertFalse(np.any(rows['training']))
                self.assertFalse(np.any(rows['valid']))

    def make_rows(self):
        rng = np.random.default_rng(912)
        payload = [rng.permutation(np.arange(6, 19)) for _ in range(70)]
        widths = np.concatenate([np.r_[[6, 6, 6], p] for p in payload])
        rows = np.zeros(len(widths), dtype=_frame_dtype(12, 25, True))
        rows['symbol'] = -1
        rows['structure_ok'] = True
        rows['index'] = np.arange(len(rows))+1
        rows['start_s'] = np.arange(len(rows))/150000
        rows['high_chips'] = 1.1*widths-1.3
        rows['clock_high_chips'] = rows['high_chips']
        rows['one_us'] = rows['high_chips']/25/150000*1e6
        rows['zero_us'] = 1e6/150000-rows['one_us']
        return rows, widths

    def test_single_width_preamble_and_unordered_alphabet_calibrate_causally(self):
        rows, truth = self.make_rows()
        options = dict(preamble_widths='6 6 6', payload_symbols=13, auto_scale=True)
        apply_protocol_training(rows, 25, 6, 12, 7, .3, options)
        data = ~rows['training'] & (rows['index'] > 256)
        self.assertTrue(np.all(rows['valid'][data]))
        np.testing.assert_array_equal(rows['symbol'][data], truth[data]-6)
        self.assertAlmostEqual(rows['width_scale'][-1], 1.1, places=5)
        altered, _ = self.make_rows()
        altered['high_chips'][-100:] = 12.4
        apply_protocol_training(altered, 25, 6, 12, 7, .3, options)
        for key in ('symbol', 'valid', 'corrected_high_chips', 'width_scale'):
            np.testing.assert_array_equal(rows[key][:-103], altered[key][:-103])
        self.assertIsNone(_alphabet_scale(np.tile([5.3, 6.4, 7.5], 50), np.arange(6, 19)))

    def test_broken_scheduled_preamble_does_not_eat_first_payload_symbol(self):
        rows, truth = self.make_rows()
        rows['structure_ok'][16] = False
        apply_protocol_training(rows, 25, 6, 12, 7, .3,
                                dict(preamble_widths='6 6 6', payload_symbols=13))
        np.testing.assert_array_equal(np.flatnonzero(rows['training'])[:6], [0, 1, 2, 16, 17, 18])
        self.assertFalse(rows['training'][19])
        self.assertEqual(rows['calibration_id'][19], rows['calibration_id'][3])
        self.assertEqual(rows['calibration_source'][19], '前导偏移')

    def test_30_percent_pilots_after_long_high_pulse_keep_frame_alignment(self):
        # Independent TX: the rising edge advances after a long high pulse.
        # The first preamble follows level 7, with a 5.6%-period advance.
        fs, rate = 15.36e6, 169971.671
        options, _ = duty_protocol(30,7,8)
        sent = np.r_[np.arange(8), np.tile(np.r_[[0,0,0],np.arange(8)],40)]
        origin = .23/rate
        base = origin+np.arange(len(sent))/rate
        prior = np.r_[-1,sent[:-1]]
        shift = np.where(prior==7,-.056,np.where(prior==6,-.027,0.))
        rises = base+shift/rate
        falls = base+(1.04*(30+7*sent)+.5)/100/rate
        t = np.arange(int((origin+(len(sent)+.8)/rate)*fs))/fs
        index = np.searchsorted(rises,t,side='right')-1
        high = (index>=0)&(index<len(sent))&(t<falls[np.clip(index,0,len(sent)-1)])
        rng = np.random.default_rng(310)
        x = np.where(high,-300,300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        x += rng.normal(0,.5,len(t))+1j*rng.normal(0,.5,len(t))
        raw = np.zeros((len(t),4),dtype='<i2')
        raw[:,0],raw[:,1] = np.rint(x.real),np.rint(x.imag)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pilot30.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path,nfft=4096,averages=1,sample_rate_hz=fs)
            result = recover_pwm2(spectrum,symbol_hz=rate,**options,
                training_options=dict(preamble_widths=(30,30,30),payload_symbols=8,
                                      clock_width=True,auto_scale=True))
            rows = result.frames
            ids = np.rint((rows['start_s']-origin)*rate).astype(int)
            within = (ids>=8)&(ids<len(sent))
            expected_training = (ids-8)%11<3
            np.testing.assert_array_equal(rows['training'][within],expected_training[within])
            self.assertTrue(np.all(rows['structure_ok'][within]))
            self.assertEqual(ids[np.flatnonzero(rows['training'])[0]],8)
            stable = within & ~expected_training & (ids>256)
            self.assertTrue(np.all(rows['valid'][stable]))
            np.testing.assert_array_equal(rows['symbol'][stable],sent[ids[stable]])
            self.assertGreater(result.training_calibration['spacing_symbols'],100)

    def test_result_reports_actual_calibration_configuration_and_startup_stage(self):
        from rx706_pwm import PWMRecovery
        result = PWMRecovery.__new__(PWMRecovery)
        result.prefix_chips, result.code_chips, result.guard_chips = 20,49,31
        result.training_calibration = dict(preamble_widths=(20,20,20),payload_symbols=8,
            auto_scale=True,clock_width=True,spacing_symbols=150)
        description = result.calibration_description
        self.assertIn('本次前导 20%、20%、20%', description)
        self.assertIn('每帧净荷 8', description)
        self.assertIn('150行已标定', description)
        self.assertIn('初期仅前导偏移', description)
        result.training_calibration = dict(preamble_widths='24 24 24',payload_symbols=0,
            auto_scale=False,clock_width=True,spacing_symbols=0)
        self.assertIn('本次前导 24%、24%、24%', result.calibration_description)
        self.assertIn('未固定帧长', result.calibration_description)
        self.assertIn('只校准偏移', result.calibration_description)

    def test_rf_capture_percent_alphabet_and_reversed_symbol_mapping(self):
        # At 150 kHz the nominal 1%-unit has only 1.024 samples. The relevant
        # 7%-symbol spacing has 7.168 samples and must remain decodable.
        options, _ = duty_protocol('20', '7', '8')
        options['symbol_widths'] = tuple(reversed(options['symbol_widths']))
        fs, rate = 15.36e6, 150000
        codebook = np.asarray(options['symbol_widths'])
        payload = np.arange(8)
        sent = np.tile(np.r_[[0, 7, 0], payload], 40)
        origin = .23/rate
        t = np.arange(int((origin+(len(sent)+.8)/rate)*fs))/fs
        index = np.floor((t-origin)*rate).astype(int)
        within = ((t-origin)*rate-index)*100
        widths = 1.07*codebook[sent]-2
        high = (index >= 0) & (index < len(sent)) & (within < widths[np.clip(index, 0, len(sent)-1)])
        rng = np.random.default_rng(31)
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        x += rng.normal(0, 1, len(t))+1j*rng.normal(0, 1, len(t))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'percent.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=4096, averages=1, sample_rate_hz=fs)
            r = recover_pwm2(spectrum, symbol_hz=rate, auto_search=True, **options,
                training_options=dict(preamble_widths='69 20 69', payload_symbols=8,
                                      clock_width=True, same_tolerance_chips=.8))
            self.assertTrue(r.decode_locked)
            self.assertEqual(r.bits_per_symbol, 3)
            valid = r.frames['valid']
            ids = np.rint((r.frames['start_s'][valid]-origin)*rate).astype(int)
            np.testing.assert_array_equal(r.frames['symbol'][valid], sent[ids])
            self.assertGreater(valid.sum(), 300)
            check = check_reference(r.frames, payload, bits_per_symbol=3,
                                    symbol_bitwords=r.symbol_bitwords)
            self.assertTrue(check.locked)
            self.assertEqual((check.symbol_errors, check.erasures), (0, 0))


if __name__ == '__main__':
    unittest.main()

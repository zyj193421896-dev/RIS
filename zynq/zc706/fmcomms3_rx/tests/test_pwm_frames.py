"""Known training, configurable TDMA boundaries and unbiased error accounting."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm import _frame_dtype, recover_pwm2
from rx706_pwm_protocol import duty_protocol, apply_alphabet_decisions
from rx706_pwm_frames import (default_users, validate_frame_options,
                             apply_frame_training, check_frame_reference)
from rx706_spectrum import analyze_bin

WIDTHS = tuple(range(20, 70, 7))
WORDS = tuple(format(k, '03b') for k in range(8))


def configuration(count=1, length=24):
    users = default_users(count)
    for user in users:
        user.update(payload_symbols=length, reference='0 1 2 3 4 5 6 7')
    return dict(user_count=count, users=users, reference_enabled=True)


def sent_symbols(config, rounds=2):
    c = validate_frame_options(config, WIDTHS)
    pattern = c['startup_pattern']
    prefix = [pattern[i % len(pattern)] for i in range(c['startup_symbols'])]
    symbols = prefix+list(c['training_sequence'])
    for _ in range(rounds):
        for user in c['users']:
            symbols.extend(user['identity'])
            symbols.extend(i % 8 for i in range(user['payload_symbols']))
    return np.array(symbols)


def measured_rows(sent, nonlinear=True):
    rows = np.zeros(len(sent), dtype=_frame_dtype(49, 100, True))
    rows['index'] = np.arange(len(sent))+1
    rows['structure_ok'] = True
    centers = np.array([21., 29., 38., 46., 53., 61., 69., 77.]) if nonlinear else np.array(WIDTHS)
    rows['high_chips'] = rows['clock_high_chips'] = centers[sent]
    rows['one_us'], rows['zero_us'] = 3., 3.
    return rows


def decode_rows(rows, config):
    apply_frame_training(rows, 100, config, WIDTHS)
    apply_alphabet_decisions(rows, WIDTHS, 20, 49, .3, True, nearest=True)
    active = rows['valid']
    rows['symbol'][active] //= 7
    rows['bits'][active] = [WORDS[k] for k in rows['symbol'][active]]
    failed = (rows['frame_part'] == '净荷') & ~rows['identity_ok']
    rows['valid'][failed], rows['symbol'][failed], rows['bits'][failed] = False, -1, '???'
    return check_frame_reference(rows, config, WIDTHS, WORDS)


class FrameTests(unittest.TestCase):
    def test_single_user_uses_full_known_training_before_first_payload(self):
        config = configuration()
        sent = sent_symbols(config)
        rows = measured_rows(sent)
        # Startup can be completely unusable without contaminating the fit.
        rows['structure_ok'][:32] = False
        result = decode_rows(rows, config)
        self.assertTrue(result.locked)
        self.assertEqual(result.overall['evaluated_blocks'], 2)
        self.assertEqual(result.overall['bad_blocks'], 0)
        self.assertEqual(result.overall['bit_errors'], 0)
        payload = rows['frame_part'] == '净荷'
        self.assertTrue(np.all(rows['valid'][payload]))
        np.testing.assert_array_equal(rows['symbol'][payload], sent[payload])
        self.assertEqual(int(payload.sum()), 48)
        self.assertEqual(int((rows['frame_part'] == '公共训练').sum()), 64)

    def test_configurable_sequences_lengths_and_variable_length_users(self):
        config = configuration(3)
        config.update(startup_symbols=11, startup_pattern='7 0', training_symbols=96,
                      training_pattern='7 6 5 4 3 2 1 0')
        for i, user in enumerate(config['users']):
            user['payload_symbols'] = 16+8*i
        config['users'][1]['identity'] += ' 0 7 0 7'
        sent = sent_symbols(config)
        result = decode_rows(measured_rows(sent), config)
        self.assertEqual(result.overall['evaluated_blocks'], 6)
        self.assertEqual(result.overall['bad_blocks'], 0)
        self.assertEqual([u['compared_bits'] for u in result.per_user], [96, 144, 192])

    def test_bit_errors_erasures_and_bad_identity_count_without_sliding(self):
        config = configuration(3, 16)
        sent = sent_symbols(config)
        # First block starts at 96; ID16 then 16 payload. A 0->7 is 3 bit errors.
        sent[112] = 7
        rows = measured_rows(sent)
        rows['structure_ok'][113] = False
        # Third ID fails completely; fourth ID/data must stay at original indices.
        rows['structure_ok'][160:176] = False
        result = decode_rows(rows, config)
        self.assertEqual(result.overall['evaluated_blocks'], 6)
        self.assertEqual(result.overall['bad_blocks'], 2)
        self.assertEqual(result.overall['bit_errors'], 3)
        self.assertEqual(result.overall['erased_bits'], 51)
        self.assertEqual(result.overall['compared_bits'], 237)
        self.assertAlmostEqual(result.overall['ber'], 3/237)
        self.assertAlmostEqual(result.overall['bler'], 2/6)
        self.assertEqual(result.overall['identity_failures'], 1)
        self.assertFalse(result.blocks[3]['bad'])
        self.assertEqual(result.per_user[2]['bad_blocks'], 1)

    def test_tail_block_and_unreferenced_users_do_not_fake_zero_ber(self):
        config = configuration(2, 16)
        config['users'][1]['reference'] = ''
        sent = sent_symbols(config)[:-3]
        result = decode_rows(measured_rows(sent), config)
        self.assertEqual(result.overall['complete_blocks'], 3)
        self.assertEqual(result.overall['partial_blocks'], 1)
        self.assertEqual(result.overall['evaluated_blocks'], 2)
        self.assertEqual(result.overall['unreferenced_blocks'], 1)
        self.assertIsNone(result.per_user[1]['ber'])
        config['reference_enabled'] = False
        result = decode_rows(measured_rows(sent), config)
        self.assertIsNone(result.overall['ber'])
        self.assertIsNone(result.overall['bler'])

    def test_reference_is_not_an_acquisition_or_calibration_input(self):
        config = configuration()
        sent = sent_symbols(config)
        rows1 = measured_rows(sent)
        before = decode_rows(rows1, config)
        changed = copy.deepcopy(config)
        changed['users'][0]['reference'] = '7 6 5 4 3 2 1 0'
        rows2 = measured_rows(sent)
        after = decode_rows(rows2, changed)
        np.testing.assert_array_equal(rows1, rows2)
        self.assertEqual(before.overall['bit_errors'], 0)
        self.assertGreater(after.overall['bit_errors'], 0)
        self.assertEqual(after.overall['bad_blocks'], 2)

    def test_bits_and_finite_continuous_references(self):
        config = configuration(length=16)
        config.update(reference_format='bits', reference_repeat=False, reference_reset=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'known.txt'
            path.write_text(' '.join(WORDS)*2, encoding='utf-8')
            config['users'][0]['reference'] = '@'+str(path)
            sent = sent_symbols(config, rounds=3)
            result = decode_rows(measured_rows(sent), config)
        self.assertEqual(result.overall['evaluated_blocks'], 1)
        self.assertEqual(result.overall['unreferenced_blocks'], 2)
        self.assertEqual(result.overall['bit_errors'], 0)

    def test_no_match_and_invalid_config_fail_closed(self):
        config = configuration()
        rows = measured_rows(np.zeros(250, dtype=int))
        result = decode_rows(rows, config)
        self.assertFalse(result.locked)
        self.assertFalse(np.any(rows['valid']))
        for edit in (dict(user_count=0), dict(training_symbols=7), dict(training_pattern='0 0 0'),
                     dict(startup_symbols=-1), dict(identity_errors=8)):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_frame_options(dict(config, **edit), WIDTHS)
        bad = configuration(2)
        bad['users'][1]['identity'] = bad['users'][0]['identity']
        with self.assertRaises(ValueError):
            validate_frame_options(bad, WIDTHS)

    def test_zero_training_uses_nominal_widths(self):
        config = configuration()
        config.update(training_symbols=0, training_pattern='', startup_symbols=0, startup_pattern='')
        sent = sent_symbols(config)
        result = decode_rows(measured_rows(sent, nonlinear=False), config)
        self.assertTrue(result.locked)
        self.assertEqual(result.overall['bad_blocks'], 0)

    def test_all_payload_erasures_keep_bler_denominator_and_undefined_ber(self):
        config = configuration(length=16)
        sent = sent_symbols(config)
        rows = measured_rows(sent)
        rows['structure_ok'][112:128] = False
        rows['structure_ok'][144:160] = False
        result = decode_rows(rows, config)
        self.assertTrue(result.locked)
        self.assertEqual(result.overall['evaluated_blocks'], 2)
        self.assertEqual(result.overall['bad_blocks'], 2)
        self.assertEqual(result.overall['erased_bits'], 96)
        self.assertEqual(result.overall['bler'], 1.)
        self.assertIsNone(result.overall['ber'])

    def test_continuous_reference_consumes_damaged_slots_per_user(self):
        config = configuration(2, 16)
        config.update(reference_reset=False, reference_repeat=False)
        reference = list(range(8))*4
        for user in config['users']:
            user['reference'] = ' '.join(map(str, reference))
        sent = sent_symbols(config)
        rows = measured_rows(sent)
        # Second user loses its first marker; next-round reference still starts
        # at symbol16 for each user rather than restarting after an error.
        rows['structure_ok'][128:144] = False
        result = decode_rows(rows, config)
        self.assertEqual(result.overall['evaluated_blocks'], 4)
        self.assertEqual(result.overall['bad_blocks'], 1)
        self.assertEqual(result.per_user[1]['compared_bits'], 48)
        self.assertEqual(result.per_user[1]['erased_bits'], 48)
        self.assertFalse(result.blocks[-1]['bad'])

    def test_independent_rf_synthesis_acquires_training_and_three_users(self):
        config = configuration(3, length=64)
        sent = sent_symbols(config, rounds=2)
        fs, rate = 15.36e6, 169971.671
        origin = .23/rate
        t = np.arange(int((origin+(len(sent)+.8)/rate)*fs))/fs
        index = np.floor((t-origin)*rate).astype(int)
        phase = (t-origin)*rate-index
        physical = np.array([21.,29.,38.,46.,53.,61.,69.,77.])[sent]/100
        high = (index >= 0) & (index < len(sent)) & (phase < physical[np.clip(index, 0, len(sent)-1)])
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        rng = np.random.default_rng(732)
        x += rng.normal(0, .5, len(t))+1j*rng.normal(0, .5, len(t))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        compiled, _ = duty_protocol('20', '7', '8')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'minimal.bin'
            raw.tofile(path)
            spectrum = analyze_bin(path, nfft=4096, averages=1, sample_rate_hz=fs)
            result = recover_pwm2(spectrum, symbol_hz=rate, auto_search=True, **compiled,
                training_options=dict(frame_options=config, alphabet_widths=WIDTHS))
        self.assertTrue(result.decode_locked, result.decode_note)
        self.assertTrue(result.frame_check.locked)
        self.assertEqual(result.frame_check.overall['evaluated_blocks'], 6)
        self.assertEqual(result.frame_check.overall['bad_blocks'], 0)
        self.assertEqual(len(result.bitstream), 6*64*3)
        payload = result.frames['frame_part'] == '净荷'
        indices = np.rint((result.frames['start_s'][payload]-origin)*rate).astype(int)
        np.testing.assert_array_equal(result.frames['symbol'][payload], sent[indices])


if __name__ == '__main__':
    unittest.main()

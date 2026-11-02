"""Known-cycle audit: BER, held-out synchronization, erasures and no correction."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm import FRAME_DTYPE
from rx706_pwm_reference import check_reference, parse_reference


def capture(n=128, phase=1):
    frames = np.zeros(n, dtype=FRAME_DTYPE)
    frames['index'] = np.arange(n)+100
    frames['symbol'] = np.array([7, 6, 4, 2])[(np.arange(n)+phase) % 4]
    frames['bits'] = [format(int(v), '03b') for v in frames['symbol']]
    frames['valid'] = True
    return frames


class ReferenceTests(unittest.TestCase):
    def test_rotated_cycle_and_warned_rows_are_correct_and_unchanged(self):
        frames = capture()
        frames['glitch_count'][[20, 80]] = 1
        original = frames.copy()
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertTrue(r.locked)
        self.assertEqual(r.offset, 1)
        self.assertEqual(r.compared_symbols, 64)
        self.assertEqual(r.compared_bits, 192)
        self.assertEqual(r.ber, 0.)
        self.assertEqual(r.warned_symbols, 2)
        self.assertEqual(r.warned_mismatches, 0)
        self.assertEqual(r.row_status[20], '同步/匹配')
        self.assertEqual(r.row_status[80], '匹配')
        np.testing.assert_array_equal(frames, original)

    def test_counts_xor_bits_not_number_of_errors_or_warnings(self):
        frames = capture()
        frames['symbol'][70] ^= 7  # 3 wrong bits
        frames['symbol'][71] ^= 1  # 1 wrong bit
        frames['glitch_count'][[70, 81]] = 1
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertEqual((r.symbol_errors, r.bit_errors, r.warned_mismatches), (2, 4, 1))
        self.assertAlmostEqual(r.ber, 4/192)
        self.assertEqual(r.row_status[70], '不匹配')

    def test_display_candidates_never_enter_ber_or_fill_erasures(self):
        frames = capture()
        frames['candidate_symbol'] = frames['symbol'] ^ 7
        frames['valid'][70] = False
        frames['symbol'][70] = -1
        frames['candidate_symbol'][70] = 2
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertEqual(r.bit_errors, 0)
        self.assertEqual(r.erasures, 1)
        self.assertEqual(r.compared_bits, 63*3)
        self.assertEqual(r.row_bit_errors[70], -1)

    def test_erasures_keep_positions_in_sync_and_evaluation(self):
        frames = capture()
        frames['valid'][[0, 1, 65, 66]] = False
        frames['symbol'][[0, 1, 65]] = -1
        frames['symbol'][66] = -2  # idle is an erasure when data is expected
        frames['bits'][[0, 1, 65]] = '???'
        frames['bits'][66] = '---'
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertTrue(r.locked)
        self.assertEqual((r.sync_valid, r.offset), (62, 1))
        self.assertEqual((r.erasures, r.compared_symbols, r.compared_bits), (2, 62, 186))
        self.assertEqual(r.bit_errors, 0)
        self.assertEqual(r.row_status[65], '擦除')
        self.assertEqual(r.row_bit_errors[65], -1)
        self.assertEqual(frames['bits'][65], '???')

    def test_sync_errors_excluded_and_later_cycle_slip_never_realigned(self):
        frames = capture()
        frames['symbol'][10:14] ^= 1
        frames['symbol'][80:] = np.roll(frames['symbol'][80:], 1)
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertTrue(r.locked)
        self.assertEqual(r.sync_matches, 60)
        self.assertEqual(r.offset, 1)
        self.assertEqual(r.symbol_errors, 48)
        self.assertGreater(r.bit_errors, 48)
        self.assertEqual(r.row_status[10], '同步/不匹配')

    def test_unrelated_ambiguous_and_short_data_produce_no_ber(self):
        cases = [(capture(64), (7, 6, 4, 2)),
                 (capture(), (0, 1, 3, 5)),
                 (capture(), (7, 7, 7, 6))]
        ambiguous = capture()
        ambiguous['symbol'] = 7
        cases[-1] = (ambiguous, cases[-1][1])
        no_sync = capture()
        no_sync['valid'][:64] = False
        cases.append((no_sync, (7, 6, 4, 2)))
        for frames, sequence in cases:
            with self.subTest(sequence=sequence, n=len(frames)):
                r = check_reference(frames, sequence)
                self.assertFalse(r.locked)
                self.assertIsNone(r.ber)
                self.assertIn('不报告 BER', r.summary())
                self.assertTrue(np.all(r.expected == -1))

    def test_no_decisions_after_sync_means_erasures_and_no_ber(self):
        frames = capture()
        frames['valid'][64:] = False
        r = check_reference(frames, (7, 6, 4, 2))
        self.assertTrue(r.locked)
        self.assertIsNone(r.ber)
        self.assertEqual(r.erasures, 64)
        self.assertEqual(r.compared_bits, 0)

    def test_parse_validate_and_reduce_repeated_periods(self):
        for text in ('7642', '7 6 4 2', '7,6,4,2', '7，6；4;2'):
            self.assertEqual(parse_reference(text), (7, 6, 4, 2))
        self.assertEqual(parse_reference('7676'), (7, 6))
        self.assertEqual(parse_reference('10 9 2', 16), (10, 9, 2))
        self.assertEqual(parse_reference('10', 16), (10,))
        for text in ('', '8', '7 -1', '7x6', '7.6', ',', '7'*257):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_reference(text)
        frames = capture()
        frames['index'][80:] += 1
        with self.assertRaises(ValueError):
            check_reference(frames, (7, 6, 4, 2))


if __name__ == '__main__':
    unittest.main()

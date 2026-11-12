"""Portable transmitter-frame builder: no implicit reference or captured state."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm_frame_builder import initial_fields, build_profile, preview_profile, save_new_profile
from rx706_pwm_pulse import validate_pulse_options


class FrameBuilderTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json').read_text())
        self.fields = initial_fields(self.original)

    def test_default_draft_does_not_mutate_or_enable_old_claims(self):
        before = copy.deepcopy(self.original)
        self.original.update(source_sha256='old', state_one='B', receiver_settings={'channel': 2})
        draft = initial_fields(self.original)
        built = build_profile(draft)
        self.assertFalse(built['reference_enabled'])
        self.assertEqual(built['users'][0]['reference'], '')
        self.assertEqual(built['cycle_calibration'], {})
        self.assertEqual(built['users'][0]['identity'], before['users'][0]['identity'] if isinstance(before['users'][0]['identity'], list) else [int(v) for v in before['users'][0]['identity'].split()])
        for key in ('source_sha256', 'state_one', 'receiver_settings', 'nearest_decision', 'tolerance_chips'):
            self.assertNotIn(key, built)
        self.assertEqual(self.original['users'], before['users'])
        self.assertEqual(self.original['cycle_calibration'], before['cycle_calibration'])

    def test_explicit_reference_and_calibration_roundtrip(self):
        self.fields.update(reference_enabled=True, calibration_enabled=True)
        built = build_profile(self.fields)
        self.assertEqual(built['cycle_calibration']['payload_indices'], [3, 4, 5, 6])
        self.assertEqual(len(built['codebook']), 66)
        self.assertEqual(len(built['users'][0]['reference']), 66)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_new_profile(Path(tmp)/'new.json', built)
            self.assertEqual(json.loads(path.read_text()), built)
            with self.assertRaises(FileExistsError):
                save_new_profile(path, built)
            self.assertEqual(json.loads(path.read_text()), built)

    def test_blank_requires_explicit_preamble_and_codebook(self):
        blank = initial_fields()
        self.assertEqual(blank['identity'], '')
        self.assertEqual(blank['codebook'], '')
        with self.assertRaises(ValueError):
            build_profile(blank)
        self.fields['identity'] = ''
        with self.assertRaises(ValueError):
            build_profile(self.fields)

    def test_multiuser_refused_not_sliced(self):
        for count, users in ((2, [self.original['users'][0]]*2), (1, [self.original['users'][0]]*2)):
            p = dict(self.original, user_count=count, users=users)
            before = copy.deepcopy(p)
            with self.assertRaisesRegex(ValueError, '单用户'):
                initial_fields(p)
            self.assertEqual(p, before)

    def test_enabled_reference_must_be_real_inline_input(self):
        self.fields['reference_enabled'] = True
        for value in ('', '@missing.txt', '999999', '-1', 'x'):
            with self.subTest(value=value):
                self.fields['reference'] = value
                with self.assertRaises(ValueError):
                    build_profile(self.fields)
        self.fields['reference_enabled'] = False
        self.assertFalse(build_profile(self.fields)['reference_enabled'])

    def test_bit_reference_preserved(self):
        self.fields.update(reference_enabled=True, reference_format='bits', reference='1010 0011', reference_repeat=False, reference_reset=False)
        p = build_profile(self.fields)
        self.assertEqual(p['users'][0]['reference'], '10100011')
        self.assertFalse(p['reference_repeat'])
        self.assertFalse(p['reference_reset'])

    def test_bad_calibration_positions(self):
        self.fields['calibration_enabled'] = True
        for value in ('0 4 5 6', '4 4 5 6', '4 5 6 67', '4 5 6', '4 5 6 7 8', '4 5 6 7.5'):
            with self.subTest(value=value):
                self.fields['calibration_positions'] = value
                with self.assertRaises(ValueError):
                    build_profile(self.fields)

    def test_bad_calibration_symbols_and_averaging(self):
        self.fields['calibration_enabled'] = True
        for value in ('3 3 3 3', '3 4 5', '3 4 5 66', '3 4 5 -1'):
            with self.subTest(value=value):
                self.fields['calibration_symbols'] = value
                with self.assertRaises(ValueError):
                    build_profile(self.fields)
        self.fields['calibration_symbols'] = '3 4 5 6'
        for value in ('0', '4097', '1.5'):
            self.fields['average_blocks'] = value
            with self.assertRaises(ValueError):
                build_profile(self.fields)

    def test_invalid_timing_lengths_and_preamble(self):
        for key, values in {'symbol_khz': ['0', 'nan', 'inf', '-1'], 'duty_step': ['0', '7'],
                            'payload_symbols': ['0', '1.5'], 'identity': ['0 0 0', '0 0 0 66'],
                            'min_high_chips': ['0', '5.5'], 'low_guard_chips': ['0', '5.5']}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    build_profile(dict(self.fields, **{key: value}))

    def test_invalid_codebook_and_width_only(self):
        for value in ('', '0 5\n0 5', '0 5\n11 5', '0 5\n0 5.5', '0 4\n0 5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_profile(dict(self.fields, codebook=value))
        p = build_profile(dict(self.fields, codebook='0 5\n0 6', joint_delay=False))
        self.assertEqual(p['duty_count'], 2)

    def test_preview_distinguishes_scored_and_reserved(self):
        p = build_profile(self.fields)
        text = preview_profile(p)
        for phrase in ('BER未测', '不是实测信号认证', '延迟0片', '高5片', '66 个', '任意净荷'):
            self.assertIn(phrase, text)
        self.fields['calibration_enabled'] = True
        text = preview_profile(build_profile(self.fields))
        self.assertIn('第4个：保留校准', text)
        self.assertIn('评分的位置 62 个', text)
        self.assertIn('第8～66个：数据评分区', text)

    def test_implicit_current_codebook_materialized(self):
        p = dict(self.original, codebook='')
        fields = initial_fields(p)
        self.assertEqual(len(build_profile(fields)['codebook']), 66)

    def test_saved_profile_accepted_by_existing_validator(self):
        p = build_profile(self.fields)
        parsed = validate_pulse_options(dict(p, step_percent=p['duty_step'], symbol_count=p['duty_count'], frame_options=p), p['symbol_khz']*1000)
        self.assertEqual(parsed['frame_cycle_symbols'], 70)
        self.assertEqual(parsed['cycle_calibration'], {})


if __name__ == '__main__':
    unittest.main()

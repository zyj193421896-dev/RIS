"""Explicit pulse profiles may analyze two seconds without truncating legacy modes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
from rx706_recovery import _read_span
from rx706_pwm import recover_pwm2
from rx706_pwm_autotune import search_options


class TwoSecondSpanTests(unittest.TestCase):
    def test_span_reads_all_two_seconds_and_preserves_legacy_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'iq.bin'
            np.zeros((2000, 4), dtype='<i2').tofile(path)
            spectrum = SimpleNamespace(path=path, sample_rate_hz=1000,
                total_tuples=2000, first_tuple=0, channel=1)
            with self.assertRaisesRegex(ValueError, '1000 ms'):
                _read_span(spectrum, 2, None, None, max_samples=2000)
            x, count, clipped = _read_span(spectrum, 2, None, None,
                max_samples=2000, max_duration_s=2)
            self.assertEqual((len(x), count, clipped), (2000, 2000, 0))
            spectrum.first_tuple = 250
            x, count, clipped = _read_span(spectrum, 2, None, None,
                max_samples=2000, max_duration_s=2)
            self.assertEqual((len(x), count), (1750, 1750))
            with self.assertRaisesRegex(ValueError, '2000 ms'):
                _read_span(spectrum, 2.001, None, None, max_samples=2000, max_duration_s=2)

    def test_only_explicit_pulse_profile_raises_read_bound(self):
        profile = json.loads((Path(__file__).resolve().parents[1] /
            'host/frame_profiles/fixed_preamble10/frame.json').read_text())
        spectrum = SimpleNamespace(sample_rate_hz=15360000)
        for options, duration_limit in ((dict(pulse_options=search_options(profile)), 2), ({}, 1)):
            with patch('rx706_pwm._read_span', side_effect=RuntimeError('read boundary')) as read:
                with self.assertRaisesRegex(RuntimeError, 'read boundary'):
                    recover_pwm2(spectrum, symbol_hz=169971.671, duration_s=2, **options)
                self.assertEqual(read.call_args.kwargs['max_duration_s'], duration_limit)
                self.assertEqual(read.call_args.kwargs['max_samples'], duration_limit*15360000)

if __name__ == '__main__':
    unittest.main()

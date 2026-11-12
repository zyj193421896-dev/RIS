"""NEW capture profile: independent physical-edge synthesis, no payload-aided fitting."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'host'))
from rx706_pwm_pulse import validate_pulse_options, decode_pulses, _header

PROFILE = ROOT / 'host/frame_profiles/new_capture_preamble12_20261110/frame_new_capture_preamble12.json'
RATE, FS = 169971.671, 15.36e6


def load_profile():
    p = json.loads(PROFILE.read_text())
    # Exactly the schema conversion used by rx706_spectrum_gui.load_frame_config.
    return validate_pulse_options(dict(p, step_percent=p['duty_step'],
        symbol_count=p['duty_count'], frame_options=dict(p, reference_format='symbols')), RATE, FS)


def edges(c, rounds=3, gain=1.1, offset_us=.04, noise_chips=0., phase=.37, seed=0, random_payload=False):
    rng = np.random.default_rng(seed)
    identity = c['frame_options']['users'][0]['identity']
    sent = np.concatenate([np.r_[identity, rng.integers(66, size=66) if random_payload else np.arange(66)] for _ in range(rounds)]).astype(int)
    book, chip = np.asarray(c['codebook']), 1/RATE/20
    rises = (phase+np.arange(len(sent))+book[sent, 0]/20)/RATE
    falls = rises+gain*book[sent, 1]*chip+offset_us*1e-6
    # Independent jitter on both edges, specified in chips; gain and offset act on widths.
    rises += rng.normal(0, noise_chips*chip, len(sent))
    falls += rng.normal(0, noise_chips*chip, len(sent))
    return sent, rises, falls


class Preamble12Tests(unittest.TestCase):
    def test_schema_no_payload_calibration_or_reference(self):
        c = load_profile()
        self.assertEqual(c['frame_cycle_symbols'], 78)
        self.assertEqual(c['symbol_count'], 66)
        self.assertFalse(c['cycle_calibration'])
        self.assertFalse(c['frame_options']['reference_enabled'])
        self.assertFalse(c['frame_options']['users'][0]['reference'])
        self.assertGreater(len(set(np.asarray(c['codebook'])[c['frame_options']['users'][0]['identity'], 1])), 2)

    def test_only_correct_training_cycle_start_matches(self):
        c = load_profile()
        book, user = np.asarray(c['codebook']), c['frame_options']['users'][0]
        for noise in (0., .03, .06):
            for seed in range(10):
                _, rises, falls = edges(c, noise_chips=noise, seed=seed)
                matched = [i for i in range(78) if _header(rises, falls, i, user, book, 1/RATE, c) is not None]
                self.assertEqual(matched, [0], (noise, seed, matched))

    def test_independent_gain_offset_phase_and_noise_decode(self):
        c = load_profile()
        for gain, offset, phase, noise in [(1., 0., .13, 0.), (.9, .12, .81, .02), (1.1, .04, .37, .04), (1.2, .08, .62, .04)]:
            for random_payload in (False, True):
                sent, rises, falls = edges(c, gain=gain, offset_us=offset, phase=phase, noise_chips=noise, random_payload=random_payload)
                t = np.arange(int((len(sent)+2)/RATE*FS))/FS
                idx = np.searchsorted(rises, t, side='right')-1
                high = (idx >= 0) & (t < falls[np.maximum(idx, 0)])
                all_edges = np.r_[rises, falls]
                after = np.r_[np.ones(len(sent), bool), np.zeros(len(sent), bool)]
                order = np.argsort(all_edges)
                rows, _, calibration, _ = decode_pulses(high, all_edges[order], after[order], t, FS, RATE, c)
                data = rows[rows['frame_part'] == '净荷']
                self.assertEqual(len(data), 198)
                self.assertTrue(np.all(data['valid']))
                np.testing.assert_array_equal(data['symbol'], sent.reshape(3,78)[:,12:].reshape(-1))
                self.assertTrue(all(x['width_gain_identified'] for x in calibration))


if __name__ == '__main__':
    unittest.main()

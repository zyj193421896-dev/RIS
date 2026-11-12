"""Reproducible non-gating jitter sweep; print actual failure boundaries, not a guarantee."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests'))
from test_pwm_preamble12 import load_profile, edges, RATE, FS, _header, decode_pulses

c = load_profile()
book, user = np.asarray(c['codebook']), c['frame_options']['users'][0]
report = []
for sigma_us in (0., .02, .05, .10):
    acquired = false_starts = correct = compared = trials = 0
    for gain in (1.05, 1.08, 1.11):
        for offset in (-.018, .069, .156):
            for seed in range(10):
                trials += 1
                sent, rises, falls = edges(c, gain=gain, offset_us=offset,
                    noise_chips=sigma_us*1e-6*RATE*20, seed=seed)
                matches = [i for i in range(78) if _header(rises, falls, i, user, book, 1/RATE, c) is not None]
                acquired += 0 in matches
                false_starts += sum(i != 0 for i in matches)
                t = np.arange(int((len(sent)+2)/RATE*FS))/FS
                idx = np.searchsorted(rises, t, side='right')-1
                high = (idx >= 0) & (t < falls[np.maximum(idx, 0)])
                ee = np.r_[rises, falls]
                aa = np.r_[np.ones(len(sent),bool), np.zeros(len(sent),bool)]
                order = np.argsort(ee)
                try:
                    rows, _, _, _ = decode_pulses(high, ee[order], aa[order], t, FS, RATE, c)
                except ValueError:
                    continue
                data = rows[rows['frame_part'] == '净荷']
                compared += len(data)
                correct += np.count_nonzero(data['valid'] & (data['symbol'] == data['payload_index']))
    report.append(dict(independent_edge_gaussian_sigma_us=sigma_us, trials=trials,
        first_header_acquired=int(acquired), false_header_windows=int(false_starts),
        decoded_payload_rows=int(compared), correct_payload_rows=int(correct),
        correct_fraction_of_decoded_rows=correct/compared if compared else None))
print(json.dumps(report, indent=2))

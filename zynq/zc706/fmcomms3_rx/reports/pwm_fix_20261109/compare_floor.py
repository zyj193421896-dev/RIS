"""Evaluate quantizers after identical acquisition/calibration; no reference fitting."""
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2
from rx706_pwm_frames import check_frame_reference


def main():
    profile = json.loads((PROJECT/'host/gui_data/current_pwm/frame.json').read_text(encoding='utf-8'))
    options = dict(profile, step_percent=profile['duty_step'], symbol_count=profile['duty_count'],
                   frame_options=dict(profile, reference_format='symbols'))
    spectrum = analyze_bin(PROJECT/'host/gui_data/current_pwm/rx_test.bin', channel=1, nfft=4096, averages=4)
    result = recover_pwm2(spectrum, pulse_options=options, symbol_hz=float(profile['symbol_khz'])*1000,
                          duration_s=1., state_one='B', allow_unlocked=True)
    original = result.frames
    payload = ~original['training']
    valid = original['valid'] & payload
    book = np.asarray(result.pulse_options['codebook'])
    widths = original['corrected_high_chips'][valid]
    delays = original['delay_chips'][valid]
    report = dict(source=str(spectrum.path), reference_used_for_quantizer=False, variants={})
    for mode in ('nearest', 'floor_width_then_nearest_legal', 'floor_both_then_nearest_legal', 'literal_floor_width_invalid_erased'):
        rows = original.copy()
        if mode != 'nearest':
            ww = np.floor(widths)
            dd = np.floor(delays) if mode == 'floor_both_then_nearest_legal' else delays
            if mode == 'literal_floor_width_invalid_erased':
                dd = np.rint(delays)
                lookup = {tuple(pair): i for i, pair in enumerate(book)}
                symbols = np.array([lookup.get((d,w), -1) for d,w in zip(dd,ww)])
            else:
                distance = (dd[:, None]-book[:, 0])**2+(ww[:, None]-book[:, 1])**2
                symbols = np.argmin(distance, axis=1)
            rows['symbol'][valid] = symbols
            rows['valid'][valid] = symbols >= 0
            rows['bits'][valid] = [format(int(k), '07b') if k >= 0 else '?'*7 for k in symbols]
        check = check_frame_reference(rows, result.frame_options, result.symbol_widths, result.symbol_bitwords)
        decided = rows['valid'] & payload
        item = dict(check.overall, decided_symbols=int(decided.sum()),
            symbol_errors=int(np.count_nonzero(rows['symbol'][decided] != rows['payload_index'][decided] % 66)),
            first_payload_symbols=rows['symbol'][payload][:16].tolist())
        report['variants'][mode] = item
        print(mode, json.dumps(item, ensure_ascii=False), flush=True)
    (Path(__file__).parent/'floor_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

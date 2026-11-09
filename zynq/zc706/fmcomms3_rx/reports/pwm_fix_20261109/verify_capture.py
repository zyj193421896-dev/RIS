"""Replay the fixed production decoder; never fit on the expected payload."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--identity-errors', type=int)
    args = parser.parse_args()
    profile = json.loads((PROJECT/'host/frame_profiles/limit15_step5/single_user_mt_test_169971.json').read_text(encoding='utf-8'))
    if args.identity_errors is not None:
        profile['identity_errors'] = args.identity_errors
    options = dict(profile, step_percent=profile['duty_step'], symbol_count=profile['duty_count'],
                   frame_options=dict(profile, reference_format='symbols'))
    spectrum = analyze_bin(PROJECT/'captures/rx_test.bin', channel=1, nfft=4096, averages=4)
    result = recover_pwm2(spectrum, pulse_options=options, symbol_hz=float(profile['symbol_khz'])*1000,
                          duration_s=args.duration, state_one='B', allow_unlocked=True)
    rows = result.frames
    payload = rows['frame_part'] == '净荷'
    valid = rows['valid'] & payload
    report = dict(duration=args.duration, identity_errors=profile['identity_errors'], locked=result.decode_locked,
        measured_hz=result.measured_symbol_rate_hz, payload=int(payload.sum()), valid=int(valid.sum()),
        symbol_errors=int(np.count_nonzero(rows['symbol'][valid] != rows['payload_index'][valid] % 66)),
        reference=result.frame_check.overall, timing_note=result.pulse_options['timing_note'],
        calibration_note=result.pulse_options['calibration_note'],
        first_rows=[{key: row[key].item() for key in ('index', 'frame_part', 'payload_index',
                    'delay_chips', 'corrected_high_chips', 'tail_us', 'symbol', 'valid', 'reason')}
                    for row in rows[:20]])
    (Path(__file__).parent/f'after_{args.duration:g}s_errors{profile["identity_errors"]}.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'first_rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

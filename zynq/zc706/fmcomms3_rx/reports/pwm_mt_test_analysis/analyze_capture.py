"""Replay actual IQ; compare current timing with a preamble-only phase refit.

Payload references are used only for diagnostic scoring after decoding.
This script does not modify production code or capture/profile files.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'host'))
import rx706_pwm_pulse as pulse
from rx706_pwm import recover_pwm2
from rx706_spectrum import analyze_bin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=.02)
    parser.add_argument('--channel', type=int, default=1)
    parser.add_argument('--mode', choices=('current', 'preamble_phase_refit', 'phase_refit_and_narrow_header_search'))
    args = parser.parse_args()
    profile = json.loads((PROJECT / 'host/frame_profiles/limit15_step5/single_user_mt_test_169971.json').read_text(encoding='utf-8'))
    options = dict(profile, step_percent=profile['duty_step'], symbol_count=profile['duty_count'],
                   frame_options=dict(profile, reference_format='symbols'))
    source = PROJECT / 'captures/rx_test.bin'
    spectrum = analyze_bin(source, channel=args.channel, nfft=4096, averages=4)
    original = pulse._header

    def phase_refit(rises, falls, at, user, pairs, period, config):
        match = original(rises, falls, at, user, pairs, period, config)
        if match is not None:
            code = pairs[np.asarray(user['identity'], int)]
            relative = np.arange(len(code)) + code[:, 0] / config['total_chips']
            match['anchor'] = float(np.median(rises[at:at+len(code)] - relative*period))
        return match

    original_decode = pulse.decode_pulses
    experiment_source = Path(pulse.__file__).read_text(encoding='utf-8')
    before = "abs(value['anchor']-anchor) < 2*period"
    assert experiment_source.count(before) == 1
    experiment_source = experiment_source.replace(before, "abs(value['anchor']-anchor) < .45*period")
    namespace = dict(pulse.__dict__)
    exec(compile(experiment_source, '<diagnostic_nearest_header>', 'exec'), namespace)
    namespace['_header'] = phase_refit
    report = dict(source=str(source), duration=args.duration, channel=args.channel,
                  reference_used_to_decode=False, hypotheses={})
    destination = Path(__file__).parent / f'replay_ch{args.channel}_{args.duration:g}s.json'
    if args.mode and destination.exists():
        report = json.loads(destination.read_text(encoding='utf-8'))
    for mode, header, decoder in [('current', original, original_decode),
                                 ('preamble_phase_refit', phase_refit, original_decode),
                                 ('phase_refit_and_narrow_header_search', phase_refit, namespace['decode_pulses'])]:
        if args.mode and args.mode != mode:
            continue
        pulse._header = header
        pulse.decode_pulses = decoder
        start = time.monotonic()
        result = recover_pwm2(spectrum, pulse_options=options, symbol_hz=float(profile['symbol_khz'])*1000,
                              duration_s=args.duration, state_one='B', allow_unlocked=True)
        rows = result.frames
        payload = rows['frame_part'] == '净荷'
        valid = payload & rows['valid']
        book = np.asarray(result.pulse_options['codebook'])
        expected = rows['payload_index'][payload] % 66
        selected = rows[payload]
        measured = np.isfinite(selected['delay_chips']) & np.isfinite(selected['corrected_high_chips'])
        ideal = book[expected[measured]]
        observed = selected[measured]
        width_fit = np.polyfit(ideal[:, 1], observed['corrected_high_chips'], 1)
        by_symbol = []
        for k in range(66):
            subset = selected[(expected == k) & measured]
            if len(subset):
                by_symbol.append(dict(expected_symbol=k, expected_pair=book[k].tolist(), count=len(subset),
                    median_delay=float(np.median(subset['delay_chips'])),
                    median_width=float(np.median(subset['corrected_high_chips'])),
                    symbol_error_fraction=float(np.mean(subset['symbol'] != k))))
        item = dict(locked=result.decode_locked, measured_hz=result.measured_symbol_rate_hz,
                    rows=len(rows), payload=int(payload.sum()), valid=int(valid.sum()),
                    symbol_errors_in_decided=int(np.count_nonzero(rows['symbol'][valid] != rows['payload_index'][valid] % 66)),
                    reference=result.frame_check.overall,
                    median_delay_error=float(np.median(observed['delay_chips'] - ideal[:, 0])),
                    width_fit_gain=float(width_fit[0]), width_fit_offset=float(width_fit[1]),
                    median_tail_us=float(np.median([c['tail_us'] for c in result.pulse_calibrations])),
                    by_symbol=by_symbol, calibrations=result.pulse_calibrations, runtime_s=time.monotonic()-start,
                    first_rows=[{key: row[key].item() for key in ('index', 'frame_part', 'payload_index',
                        'delay_chips', 'corrected_high_chips', 'tail_us', 'symbol', 'valid', 'reason')}
                        for row in rows[:20]])
        report['hypotheses'][mode] = item
        print(mode, json.dumps({k:v for k,v in item.items() if k not in ('by_symbol', 'first_rows', 'calibrations')}, ensure_ascii=False), flush=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pulse._header = original
    pulse.decode_pulses = original_decode


if __name__ == '__main__':
    main()

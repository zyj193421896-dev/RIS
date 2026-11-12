"""Bounded, reference-free PWM parameter search. Never certifies payload BER."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import zlib

import numpy as np
from rx706_pwm import recover_pwm2
from rx706_pwm_pulse import validate_pulse_options
from rx706_spectrum import analyze_bin, _cancel


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(source, profile_path=None):
    source = Path(source).resolve()
    metadata_path = Path(str(source)+'.json')
    if not metadata_path.is_file():
        raise ValueError('Capture metadata .bin.json required; cannot guess sample rate or format')
    metadata = json.loads(metadata_path.read_text())
    capture = metadata.get('capture', {})
    if (capture.get('order') != ['I1', 'Q1', 'I2', 'Q2'] or
            capture.get('dtype') != 'int16' or capture.get('byte_order') != 'little' or
            capture.get('valid_bits') != 12 or capture.get('channels') != 2):
        raise ValueError('Unsupported capture format: requires little-endian I1/Q1/I2/Q2 int16, 12 valid bits')
    fs = float(capture.get('sample_rate_hz', 0))
    size = source.stat().st_size
    if not math.isfinite(fs) or fs <= 0 or not size or size % 8:
        raise ValueError('Invalid sample rate or truncated capture')
    if metadata.get('complete') is not True or any(int(x) != size for x in
            (capture.get('bytes', -1), metadata.get('file_bytes', -1))):
        raise ValueError('Capture is incomplete or metadata size does not match')
    digest = sha256(source)
    if 'chunk_crc32' in metadata:
        chunk_size = int(capture.get('chunk_bytes', 0))
        if chunk_size <= 0:
            raise ValueError('Invalid CRC chunk size')
        with source.open('rb') as stream:
            crc = [f'{zlib.crc32(b):08x}' for b in iter(lambda: stream.read(chunk_size), b'')]
        if crc != metadata['chunk_crc32']:
            raise ValueError('Capture CRC mismatch')
    binding = metadata.get('pwm_search_profile')
    bound = profile_path is None
    if profile_path is None:
        if not binding:
            raise ValueError('Frame profile required: supply --profile with actual identity, layout and codebook; bin alone cannot identify them')
        if binding.get('source_sha256') != digest:
            raise ValueError('Bound profile source hash mismatch')
        profile_path = metadata_path.parent/binding['path']
        if sha256(profile_path) != binding.get('profile_sha256'):
            raise ValueError('Bound profile hash mismatch')
    profile_path = Path(profile_path).resolve()
    profile = json.loads(profile_path.read_text())
    if profile.get('source_sha256', digest) != digest:
        raise ValueError('Profile source hash mismatch')
    return source, profile_path, profile, dict(source_sha256=digest,
        metadata_sha256=sha256(metadata_path), profile_sha256=sha256(profile_path),
        bytes=size, sample_rate_hz=fs, duration_s=size/8/fs,
        profile_provenance='verified capture profile binding' if bound else 'explicit user profile (not independently transmitter-certified)')


def search_options(profile):
    """Remove held-out references BEFORE validation, decoding and candidate ranking."""
    p = copy.deepcopy(profile)
    for key in list(p):
        if key.startswith('reference'):
            p.pop(key)
    if not p.get('users') or not p.get('codebook'):
        raise ValueError('Explicit users/identity/layout and codebook required')
    for user in p['users']:
        user['reference'] = ''
        if not user.get('identity') or 'payload_symbols' not in user:
            raise ValueError('Each user requires actual identity and payload_symbols')
    p['reference_enabled'] = False
    p['reference_format'] = 'symbols'
    p.pop('frame_options', None)
    p.update(step_percent=p.get('step_percent', p.get('duty_step')),
             symbol_count=p.get('symbol_count', p.get('duty_count')))
    p['frame_options'] = copy.deepcopy(p)
    return p


def candidates(profile):
    nominal = float(profile['symbol_khz'])*1000
    base = float(profile.get('tolerance_chips', .4))
    return [dict(symbol_hz=nominal*(1+ppm/1e6), state_one=polarity, offset_ppm=ppm,
                 tolerance_chips=tolerance)
            for ppm in (0, -100, 100) for polarity in ('A', 'B')
            for tolerance in sorted({base, .49})]


def metrics(result, duration, options, nominal):
    rows = result.frames
    complete = rows['block_complete']
    payload = complete & (rows['frame_part'] == '净荷')
    good = payload & rows['valid'] & rows['identity_ok']
    blocks = np.unique(rows['block_id'][complete])
    good_blocks = np.unique(rows['block_id'][complete & rows['identity_ok']])
    cycle = sum(len(u['identity'])+u['payload_symbols'] for u in options['frame_options']['users'])
    payload_per_cycle = sum(u['payload_symbols'] for u in options['frame_options']['users'])
    payload_per_cycle -= len(options.get('cycle_calibration', {}).get('payload_indices', []))
    if options.get('air_crc12'):
        payload_per_cycle = 8*len(options['frame_options']['users'])
    # Duration-based denominator also penalizes late acquisitions and dropped blocks.
    expected = max(int(payload.sum()), int(duration*nominal/cycle*payload_per_cycle))
    residual = rows['decision_distance_chips'][good]
    residual = residual[np.isfinite(residual)]
    return dict(complete_payload=int(payload.sum()), valid_payload=int(good.sum()),
        expected_payload=expected, usable_fraction=float(good.sum()/max(1, expected)),
        complete_blocks=len(blocks), identity_blocks=len(good_blocks),
        identity_fraction=len(good_blocks)/max(1, len(blocks)),
        median_distance_chips=float(np.median(residual)) if len(residual) else 1e9,
        competing_headers=int(result.pulse_options.get('competing_headers', 0)),
        measured_symbol_hz=float(result.measured_symbol_rate_hz),
        timing_note=result.pulse_options.get('timing_note', ''),
        decisions_sha256=hashlib.sha256(np.column_stack([rows[k] for k in
            ('symbol', 'valid', 'identity_ok')]).astype('<i8').tobytes()).hexdigest())


def rank_key(item):
    m = item.get('metrics', {})
    eligible = m.get('complete_blocks', 0) >= 2 and m.get('usable_fraction', 0) >= .5 and m.get('identity_fraction', 0) >= .5
    c = item['candidate']
    return (-int(eligible), -m.get('usable_fraction', 0), -m.get('identity_fraction', 0),
            m.get('competing_headers', 10**9), m.get('median_distance_chips', 1e9),
            c['tolerance_chips'], abs(c.get('offset_ppm', 0)), c['symbol_hz'], c['state_one'])


def physical_rates(result):
    """Report clock-fit capacity separately from configured clocks and bit goodput."""
    pulse = result.pulse_options
    source = pulse.get('clock_source', '')
    raw = float(result.measured_symbol_rate_hz)
    measured = (source == '多个身份前导的长间隔拟合' and
                math.isfinite(raw) and raw > 0 and getattr(result, 'decode_locked', True))
    rate = raw if measured else None
    size = len(pulse['codebook'])
    return dict(physical_rate_measured=measured, physical_rate_source=source or '未提供实测时钟证据',
        physical_symbol_rate_hz=rate, codebook_size=size,
        physical_equivalent_ceiling_bps=rate * math.log2(size) if measured else None,
        physical_chip_rate_hz=rate * pulse['total_chips'] if measured else None,
        actual_transport_bps=None,
        physical_rate_definition='Received identity-marker clock fit Rs × log2(M). Includes preamble/training occupancy; no payload, erasure or BER scaling. A codebook-equivalent theoretical upper bound, not verified encoded bit throughput or independent entropy of fixed training. Binary mapping/FEC unspecified.')


def symbol_rate_description(result):
    """Keep legacy PWM wording, qualify pulse clock fallback in status and details."""
    if not getattr(result, 'pulse_options', None):
        return f'符号率 {result.measured_symbol_rate_hz/1000:.6f} kHz'
    e = physical_rates(result)
    if e['physical_rate_measured']:
        return f"实测码元率 {e['physical_symbol_rate_hz']/1000:.6f} kBaud"
    return f'配置码元率 {result.measured_symbol_rate_hz/1000:.6f} kBaud（实测未确认）'


def payload_rates(result, decided, complete):
    """Rates over the captured analysis interval, never over selected frame rows.

    input_samples is the actual raw read count (after first_tuple and EOF
    clipping); output_rate_hz * decimation reconstructs the source sample rate.
    No assumptions about transport mapping, CRC, or FEC are made.
    """
    try:
        fs = float(result.output_rate_hz) * int(result.decimation)
        samples = int(result.input_samples)
        duration = samples / fs
        if not math.isfinite(duration) or duration <= 0 or fs <= 0:
            raise ValueError('No positive captured interval')
    except (AttributeError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        fs = duration = None
    information = math.log2(len(result.pulse_options['codebook']))
    rate = decided / duration if duration is not None else None
    return dict(analyzed_duration_s=duration, analyzed_sample_rate_hz=fs,
        analyzed_first_tuple=getattr(result, 'first_tuple', None),
        analyzed_input_samples=getattr(result, 'input_samples', None),
        decided_payload_symbol_rate_hz=rate,
        complete_payload_symbol_rate_hz=complete / duration if duration is not None else None,
        decided_payload_information_ceiling_bps=rate * information if rate is not None else None,
        decided_payload_equivalent_ceiling_bps=rate * information if rate is not None else None,
        **physical_rates(result),
        actual_goodput_bps=None,
        rate_definition='Decided complete payload symbols / actual analyzed input duration. Excludes identity, calibration, erasures and partial blocks from numerator only. Decided does not imply error-free. log2(M) is a capacity upper bound, not transport goodput; mapping/CRC/FEC unspecified.')


def file_evaluation(result, profile=None, cancel_event=None):
    """Independent physical coverage and optional scoring of already fixed decisions."""
    rows = result.frames
    payload = rows['block_complete'] & (rows['frame_part'] == '净荷')
    valid = payload & rows['valid'] & rows['identity_ok']
    total, decided = int(payload.sum()), int(valid.sum())
    pulse = result.pulse_options
    frame = pulse['frame_options']
    cycle = sum(len(u['identity'])+u['payload_symbols'] for u in frame['users'])
    useful = sum(u['payload_symbols'] for u in frame['users'])
    useful -= len(pulse.get('cycle_calibration', {}).get('payload_indices', []))
    measured = float(result.measured_symbol_rate_hz)
    information = math.log2(len(pulse['codebook']))
    check = result.frame_check
    if profile is not None:
        from rx706_pwm_frames import check_frame_reference
        ref = dict(frame, users=copy.deepcopy(profile['users']),
            reference_enabled=bool(profile.get('reference_enabled', False)),
            reference_format='bits' if profile.get('reference_format') in ('bits', '二进制比特') else 'symbols',
            reference_repeat=profile.get('reference_repeat', True), reference_reset=profile.get('reference_reset', True))
        check = check_frame_reference(rows, ref, result.symbol_widths, result.symbol_bitwords, cancel_event)
    stats = check.overall
    return dict(**payload_rates(result, decided, total), measured_symbol_rate_hz=measured,
        gross_codebook_information_bps=measured*information,
        payload_information_ceiling_bps=measured*information*useful/cycle,
        frame_cycle_symbols=cycle, scored_payload_per_cycle=useful,
        label_bits_per_symbol=int(result.bits_per_symbol), information_bits_per_symbol=information,
        complete_payload_symbols=total, decided_payload_symbols=decided,
        erased_payload_symbols=total-decided, physical_erasure_fraction=(total-decided)/total if total else None,
        complete_blocks=int(len(np.unique(rows['block_id'][rows['block_complete']]))),
        identity_failed_blocks=int(len(np.unique(rows['block_id'][rows['block_complete'] & ~rows['identity_ok']]))),
        reference_overall=stats, ber=stats['ber'], bler=stats['bler'],
        reference_status='按用户JSON参考对固定判决事后比对；参考未独立认证，未参与参数/相位/标签选择。无参考覆盖时BER未测。',
        exclusions='Only complete payload, excluding identity and explicit calibration slots; BER denominator is compared reference label bits, excluding erasures. Label bits are not independent information bits.')


def search(source, profile_path=None, *, scan_seconds=.03, finalists=2,
           max_candidates=12, max_seconds=600, channel=1, cancel_event=None, progress=None):
    started = time.monotonic()
    if not 0 < scan_seconds <= 1 or not 1 <= finalists <= 12 or not 1 <= max_candidates <= 12 or not 0 < max_seconds <= 86400:
        raise ValueError('Invalid search limits')
    _cancel(cancel_event)
    source, profile_path, profile, provenance = load_inputs(source, profile_path)
    before = source.stat()
    options = search_options(profile)
    nominal = float(profile['symbol_khz'])*1000
    options = validate_pulse_options(options, nominal, provenance['sample_rate_hz'])
    duration = provenance['duration_s']
    if duration > 2:
        raise ValueError('Full-record validation supports captures up to two seconds; longer files require explicit segmentation, never silent truncation')
    selection_duration = min(duration, 1.0)
    grid = candidates(profile)
    if len(grid) > max_candidates:
        raise ValueError(f'Candidate limit {max_candidates} smaller than required set {len(grid)}')
    spectrum = analyze_bin(source, channel=channel, nfft=4096, averages=4, cancel_event=cancel_event)
    full_results = {}
    def evaluate(candidate, seconds, stage):
        _cancel(cancel_event)
        if time.monotonic()-started >= max_seconds:
            raise TimeoutError('Search time budget exhausted between candidates; no final configuration published')
        item = dict(candidate=candidate, duration_s=seconds, first_tuple=0, stage=stage)
        try:
            opt = dict(options, tolerance_chips=candidate['tolerance_chips'])
            result = recover_pwm2(spectrum, pulse_options=opt, symbol_hz=candidate['symbol_hz'],
                state_one=candidate['state_one'], duration_s=seconds, decimation=1,
                allow_unlocked=True, cancel_event=cancel_event)
            item['metrics'] = metrics(result, seconds, options, nominal)
            if stage in ('selection', 'full_record'):
                full_results[json.dumps(candidate, sort_keys=True)] = result
        except ValueError as exc:
            item['error'] = str(exc)
        if progress:
            progress(item)
        return item
    scan = sorted([evaluate(c, min(scan_seconds, duration), 'scan') for c in grid], key=rank_key)
    viable = [item for item in scan if 'metrics' in item]
    if not viable:
        raise ValueError('No candidate matched actual frame profile')
    chosen = viable[:finalists]
    # Always validate the best opposing polarity, if acquired, to expose ambiguity.
    for item in viable:
        if item['candidate']['state_one'] != chosen[0]['candidate']['state_one']:
            if item not in chosen:
                chosen.append(item)
            break
    selection_record = sorted([evaluate(item['candidate'], selection_duration, 'selection') for item in chosen], key=rank_key)
    best = selection_record[0]
    eligible = rank_key(best)[0] == -1
    ambiguous = any('metrics' in item and item['candidate']['state_one'] != best['candidate']['state_one']
        and abs(item['metrics']['usable_fraction']-best.get('metrics', {}).get('usable_fraction', 0)) < .02 for item in selection_record[1:])
    evaluation = None
    full = selection_record if duration <= 1 else []
    if eligible and not ambiguous:
        if duration > selection_duration:
            # Freeze the winner before seeing the rest of the capture. Do not
            # rank by held-out payload or by the full-record replay outcome.
            full_results.clear()
            whole = evaluate(best['candidate'], duration, 'full_record')
            if 'metrics' not in whole:
                raise ValueError('Selected candidate full-record replay failed: '+whole.get('error', 'unknown error'))
            full = [whole]
        fixed = full_results[json.dumps(best['candidate'], sort_keys=True)]
        try:
            evaluation = file_evaluation(fixed, profile, cancel_event)
        except ValueError as exc:
            evaluation = dict(ber=None, error='Post-selection reference evaluation failed: '+str(exc))
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(source) != provenance['source_sha256']:
        raise ValueError('Source changed during search')
    host = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=host, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    selected = copy.deepcopy(profile)
    selected.pop('frame_options', None)
    for u in selected['users']:
        u['reference'] = ''
    selected['reference_enabled'] = False
    if eligible:
        selected.update(symbol_khz=str(best['candidate']['symbol_hz']/1000),
                        tolerance_chips=best['candidate']['tolerance_chips'],
                        state_one=best['candidate']['state_one'],
                        source_sha256=provenance['source_sha256'])
    return dict(schema_version=1, source=str(source), profile=str(profile_path), provenance=provenance,
        status='ambiguous' if ambiguous else 'candidate_selected' if eligible else 'no_usable_candidate',
        selection=best if eligible else None, selected_profile=selected if eligible and not ambiguous else None,
        post_selection_evaluation=evaluation,
        scan=scan, selection_record=selection_record, full_record=full, candidate_set=grid,
        selection_interval=dict(first_tuple=0, duration_s=selection_duration),
        analysis_interval=dict(first_tuple=0, duration_s=duration, full_file=True,
                               completed=bool(full) and eligible and not ambiguous),
        score_definition='Lexicographic: eligibility, duration-normalized valid payload, identity coverage, fewer competing headers, lower median codeword distance, deterministic parameter ties. No payload reference.',
        exclusions='Identity and explicit cycle calibration slots excluded; partial blocks excluded; expected payload denominator includes unacquired time.',
        validation='Same-file tuning and full-record replay, NOT independent validation or certified BER; best only among listed candidates. Repeated identities can leave initial frame phase ambiguous.',
        limits=dict(scan_seconds=scan_seconds, selection_seconds=selection_duration, max_capture_seconds=2, finalists=finalists, max_candidates=max_candidates, max_seconds=max_seconds,
                    deadline='checked between candidates; decoder cancellation supported'),
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=__import__('scipy').__version__, platform=platform.platform(), commit=commit,
                     decoder_sha256={p.name:sha256(p) for p in sorted(host.glob('*.py'))}),
        command=sys.argv, channel=channel, elapsed_s=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory (never overwrites)')
    parser.add_argument('--scan-seconds', type=float, default=.03)
    parser.add_argument('--finalists', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=12)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--channel', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    try:
        report = search(args.source, args.profile, scan_seconds=args.scan_seconds, finalists=args.finalists,
            max_candidates=args.max_candidates, max_seconds=args.max_seconds, channel=args.channel,
            progress=lambda item: print(json.dumps(item, ensure_ascii=False), flush=True))
    except (ValueError, TimeoutError, OSError, KeyError) as exc:
        parser.exit(2, f'PWM search failed: {exc}\n')
    args.output.mkdir(parents=True)
    (args.output/'search.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if report['selected_profile'] is not None:
        (args.output/'selected_frame.json').write_text(json.dumps(report['selected_profile'], ensure_ascii=False, indent=2)+'\n')
    print(report['status'])


if __name__ == '__main__':
    main()

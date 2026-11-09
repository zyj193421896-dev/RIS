"""Checksum-bound, hardware-free PWM replay shared by the GUI and CLI.

Payload references are removed before recovery and used only by the final scorer.
Fingerprinting, metadata checks and recovery belong on a worker, never Tk's thread.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from rx706_spectrum import _cancel, _metadata, analyze_bin

METHOD = 'iq_bound_pwm_replay'
INPUT_FORMAT = dict(dtype='int16', byte_order='little', order=['I1', 'Q1', 'I2', 'Q2'])
CAVEAT = ('前导、帧长及码表按配置假设恢复，须与发端核对；完成分析不等于成功解码。'
          '未核验净荷参考时不测BER；净荷参考不参与同步、时钟、校准、极性或判决。')


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def fingerprint(path, cancel_event=None):
    path = Path(path)
    before = _stamp(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            _cancel(cancel_event)
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    if before != _stamp(path):
        raise ValueError('IQ文件在计算指纹期间发生变化')
    return dict(file_bytes=before[2], sha256=digest.hexdigest())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_recipe(value):
    """Validate an untrusted JSON value without reading IQ or reference files."""
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1 or value.get('method') != METHOD:
        raise ValueError('不支持的PWM IQ绑定方案格式')
    result = copy.deepcopy(value)
    sp, rec, fp = (result.get(k) for k in ('spectrum', 'recovery', 'input_fingerprint'))
    if not isinstance(result.get('input_path'), str) or not result['input_path'].strip() or not all(isinstance(v, dict) for v in (sp, rec, fp)):
        raise ValueError('PWM绑定方案缺少文件、采样或恢复参数')
    if not Path(result['input_path']).is_absolute():
        raise ValueError('绑定IQ文件须为绝对路径')
    if result.get('input_format') != INPUT_FORMAT:
        raise ValueError('PWM复放仅支持小端int16的I1/Q1/I2/Q2双通道格式')
    digest = fp.get('sha256')
    if type(fp.get('file_bytes')) is not int or fp['file_bytes'] <= 0 or fp['file_bytes'] % 8 or not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('IQ文件大小或SHA-256指纹无效')
    if set(sp) != {'sample_rate_hz', 'channel', 'first_tuple', 'nfft', 'averages', 'remove_dc'}:
        raise ValueError('采样和频谱参数不完整或包含未知字段')
    if not _number(sp['sample_rate_hz']) or sp['sample_rate_hz'] <= 0 or type(sp['channel']) is not int or sp['channel'] not in (1, 2) or type(sp['first_tuple']) is not int or sp['first_tuple'] < 0:
        raise ValueError('采样率、通道或起点无效')
    if type(sp['nfft']) is not int or not 256 <= sp['nfft'] <= 1048576 or sp['nfft'] & (sp['nfft'] - 1) or type(sp['averages']) is not int or not 1 <= sp['averages'] <= 64 or type(sp['remove_dc']) is not bool:
        raise ValueError('频谱参数无效')
    required = {'duration_s', 'symbol_hz', 'center_hint_hz', 'decimation', 'state_one',
                'width_tolerance', 'hysteresis', 'glitch_tolerance_chips', 'pulse_options'}
    if set(rec) != required:
        raise ValueError('PWM恢复参数不完整或包含未知字段')
    if not _number(rec['duration_s']) or not .001 <= rec['duration_s'] <= 1 or not _number(rec['symbol_hz']) or rec['symbol_hz'] <= 0:
        raise ValueError('恢复时长须为1～1000毫秒，符号率须为有限正数')
    samples = math.ceil(rec['duration_s'] * sp['sample_rate_hz'])
    if samples < 256 or sp['first_tuple'] + samples > fp['file_bytes'] // 8:
        raise ValueError('恢复范围不足256点或超出IQ文件')
    if rec['center_hint_hz'] is not None and (not _number(rec['center_hint_hz']) or abs(rec['center_hint_hz']) >= sp['sample_rate_hz'] / 2):
        raise ValueError('载波提示须为自动(null)或奈奎斯特范围内的Hz数值')
    if type(rec['decimation']) is not int or rec['decimation'] not in (1, 2, 4, 8, 16) or rec['state_one'] not in ('auto', 'A', 'B'):
        raise ValueError('抽取或两态极性无效')
    for key, lower, upper, inclusive in (('width_tolerance', 0, .5, False), ('hysteresis', 0, .4, True), ('glitch_tolerance_chips', 0, .5, True)):
        x = rec[key]
        if not _number(x) or not (lower <= x < upper if inclusive else lower < x < upper):
            raise ValueError('PWM恢复参数无效：' + key)
    assumptions = result.get('assumptions')
    if not isinstance(assumptions, dict) or assumptions.get('preamble_status') not in ('hypothesis', 'confirmed') or assumptions.get('reference_status') not in ('unverified', 'confirmed') or not isinstance(assumptions.get('notes'), list) or not all(isinstance(note, str) for note in assumptions['notes']):
        raise ValueError('须明确标注前导/帧/码表假设与净荷参考核验状态')
    pulse = rec['pulse_options']
    if not isinstance(pulse, dict) or not isinstance(pulse.get('frame_options'), dict):
        raise ValueError('缺少PWM脉冲或帧配置')
    frames = pulse['frame_options']
    frames.setdefault('reference_enabled', False)
    if type(frames['reference_enabled']) is not bool:
        raise ValueError('reference_enabled须为布尔值')
    if frames['reference_enabled'] and assumptions['reference_status'] != 'confirmed':
        raise ValueError('未核验净荷参考时reference_enabled必须为false')
    users = frames.get('users')
    if not isinstance(users, (list, tuple)) or not users or not all(isinstance(user, dict) for user in users):
        raise ValueError('须完整保存每个用户的身份前导和净荷长度')
    for user in users:
        if isinstance(user.get('reference'), str) and user['reference'].lstrip().startswith('@'):
            raise ValueError('IQ绑定方案须内嵌参考数据，不允许外部@参考文件')
    from rx706_pwm_pulse import validate_pulse_options
    try:
        canonical = validate_pulse_options(pulse, rec['symbol_hz'], sp['sample_rate_hz'] / rec['decimation'])
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError('PWM脉冲或帧配置无效：' + str(exc)) from exc
    # Store only inputs, never fitted clock/calibration diagnostics from a previous run.
    pulse_keys = ('step_percent', 'min_high_chips', 'low_guard_chips', 'symbol_count',
                  'joint_delay', 'codebook', 'tolerance_chips', 'nearest_decision')
    frame_keys = ('user_count', 'users', 'search_symbols', 'identity_errors', 'training_tolerance',
                  'reference_enabled', 'reference_format', 'reference_repeat', 'reference_reset', 'clock_width',
                  'startup_symbols', 'startup_pattern', 'training_symbols', 'training_pattern')
    rec['pulse_options'] = {k: canonical[k] for k in pulse_keys}
    rec['pulse_options']['frame_options'] = {k: canonical['frame_options'][k] for k in frame_keys if k in canonical['frame_options']}
    rec['pulse_options']['frame_options'].setdefault('clock_width', True)
    # Disabled reference text remains available for review, but is not interpreted.
    if not frames['reference_enabled']:
        for target, source in zip(rec['pulse_options']['frame_options']['users'], users):
            target['reference'] = source.get('reference', '')
    result['caveat'] = str(result.get('caveat') or CAVEAT)
    return result


def make_recipe(path, pulse_options, *, sample_rate_hz, symbol_hz, channel=1, first_tuple=0,
                duration_s=1., center_hint_hz=None, decimation=1, state_one='auto',
                width_tolerance=.3, hysteresis=.10, glitch_tolerance_chips=.15,
                nfft=65536, averages=16, remove_dc=True, assumptions=None, cancel_event=None):
    path = Path(path).expanduser().resolve(strict=True)
    recipe = dict(version=1, method=METHOD, input_path=str(path),
                  input_fingerprint=fingerprint(path, cancel_event), input_format=copy.deepcopy(INPUT_FORMAT),
                  spectrum=dict(sample_rate_hz=sample_rate_hz, channel=channel, first_tuple=first_tuple,
                                nfft=nfft, averages=averages, remove_dc=remove_dc),
                  recovery=dict(duration_s=duration_s, symbol_hz=symbol_hz, center_hint_hz=center_hint_hz,
                                decimation=decimation, state_one=state_one, width_tolerance=width_tolerance,
                                hysteresis=hysteresis, glitch_tolerance_chips=glitch_tolerance_chips,
                                pulse_options=copy.deepcopy(pulse_options)),
                  assumptions=copy.deepcopy(assumptions) if assumptions is not None else dict(
                      preamble_status='hypothesis', reference_status='unverified', notes=[]), caveat=CAVEAT)
    return validate_recipe(recipe)


def read_recipe(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024:
        raise ValueError('方案JSON超过512KiB')
    return validate_recipe(json.loads(path.read_text(encoding='utf-8')))


def _sidecar_stamp(recipe):
    path = Path(recipe['input_path'])
    sidecar = Path(str(path) + '.json')
    if not sidecar.exists():
        return None
    if sidecar.stat().st_size > 1024 * 1024:
        raise ValueError('同名采集JSON超过1MiB')
    before = _stamp(sidecar)
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    rate, _, _ = _metadata(path, recipe['input_fingerprint']['file_bytes'], recipe['spectrum']['sample_rate_hz'])
    if rate != recipe['spectrum']['sample_rate_hz']:
        raise ValueError('同名采集元数据的采样率与绑定方案不一致')
    if before != _stamp(sidecar):
        raise ValueError('同名采集元数据在校验期间发生变化')
    return before, digest


def verify_input(recipe, cancel_event=None):
    recipe = validate_recipe(recipe)
    path = Path(recipe['input_path'])
    if not path.is_file():
        raise ValueError('绑定的IQ文件不存在：' + str(path))
    if fingerprint(path, cancel_event) != recipe['input_fingerprint']:
        raise ValueError('IQ文件大小或SHA-256不符；已停止复算')
    _sidecar_stamp(recipe)
    return path


def assumption_warning(recipe):
    status = recipe['assumptions']
    preamble = '前导/帧长/码表未经发端确认，本结果依赖配置假设。' if status['preamble_status'] == 'hypothesis' else '前导/帧长/码表按已确认配置。'
    reference = '未启用已核验净荷参考，BER未测。' if not recipe['recovery']['pulse_options']['frame_options']['reference_enabled'] else '参考仅在恢复完成后独立计分，未参与同步或校准。'
    return preamble + reference + ' ' + recipe['caveat']


def run_recipe(recipe_or_path, cancel_event=None, progress=None):
    """Return (SpectrumResult, PWMRecovery); unlocked reception is diagnostic data.

    Any file/metadata change, cancellation or demodulator exception discards the run.
    This function never opens hardware and never updates GUI controls.
    """
    recipe = validate_recipe(recipe_or_path) if isinstance(recipe_or_path, dict) else read_recipe(recipe_or_path)
    if progress:
        progress('正在校验PWM绑定IQ的完整SHA-256…')
    path = verify_input(recipe, cancel_event)
    before, metadata_before = _stamp(path), _sidecar_stamp(recipe)
    if progress:
        progress('IQ绑定校验通过，正在离线恢复PWM…')
    spectrum = analyze_bin(path, cancel_event=cancel_event, **recipe['spectrum'])
    if spectrum.sample_rate_hz != recipe['spectrum']['sample_rate_hz']:
        raise ValueError('实际采样率与PWM绑定方案不一致')
    recovery_options = copy.deepcopy(recipe['recovery'])
    score_options = copy.deepcopy(recovery_options['pulse_options']['frame_options'])
    blind_frames = recovery_options['pulse_options']['frame_options']
    blind_frames['reference_enabled'] = False
    for user in blind_frames['users']:
        user['reference'] = ''
    from rx706_pwm import recover_pwm
    recovery = recover_pwm(spectrum, allow_unlocked=True, cancel_event=cancel_event, **recovery_options)
    if score_options['reference_enabled']:
        from rx706_pwm_frames import check_frame_reference
        recovery.frame_options = score_options
        recovery.pulse_options['frame_options'] = score_options
        recovery.frame_check = check_frame_reference(recovery.frames, score_options, recovery.symbol_widths,
                                                     recovery.symbol_bitwords, cancel_event)
        recovery.reference_check = recovery.frame_check
        if not recovery.decode_locked:
            recovery.frame_check.locked = False
            recovery.frame_check.reason = recovery.decode_note
    _cancel(cancel_event)
    if before != _stamp(path) or metadata_before != _sidecar_stamp(recipe):
        raise ValueError('IQ文件或同名采集元数据在分析期间发生变化，结果已丢弃')
    if progress:
        progress('正在复核PWM绑定IQ，确认分析期间数据未变…')
    verify_input(recipe, cancel_event)
    if before != _stamp(path) or metadata_before != _sidecar_stamp(recipe):
        raise ValueError('IQ文件或同名采集元数据在复核期间发生变化，结果已丢弃')
    stat = path.stat()
    spectrum._source_stat = (stat.st_size, stat.st_mtime_ns)
    spectrum.pwm_replay_recipe = copy.deepcopy(recipe)
    counts = payload_diagnostics(recovery)
    recovery.replay_warning = (f"净荷擦除{counts['erased_payload_symbols']}/{counts['payload_symbols']}符号；"
        '参考统计的擦除位数只覆盖有参考的完整块，不能代替总接收擦除。' + assumption_warning(recipe))
    return spectrum, recovery


def payload_diagnostics(recovery):
    """Reception/erasure counts remain meaningful when no reference is available."""
    import numpy as np
    frames = recovery.frames
    payload = frames['frame_part'] == '净荷'
    count = int(np.count_nonzero(payload))
    decided = int(np.count_nonzero(payload & frames['valid']))
    sync_counts = {}
    if 'sync_status' in frames.dtype.names:
        labels, counts = np.unique(frames['sync_status'][payload], return_counts=True)
        sync_counts = {str(label): int(n) for label, n in zip(labels, counts)}
    return dict(payload_symbols=count, decided_payload_symbols=decided,
                erased_payload_symbols=count - decided, sync_counts=sync_counts)


def result_summary(spectrum, recovery):
    """Portable JSON diagnostics; never call decision agreement a measured BER."""
    recipe = spectrum.pwm_replay_recipe
    check = recovery.frame_check
    return _json_value(dict(method=METHOD, input_path=recipe['input_path'], input_fingerprint=recipe['input_fingerprint'],
                spectrum=recipe['spectrum'], recovery=recipe['recovery'], assumptions=recipe['assumptions'],
                warning=recovery.replay_warning, decode_locked=bool(recovery.decode_locked),
                decode_note=recovery.decode_note, carrier_hz=float(recovery.carrier_hz),
                measured_symbol_rate_hz=float(recovery.measured_symbol_rate_hz),
                symbols=len(recovery.frames), valid_symbols=int(recovery.frames['valid'].sum()),
                **payload_diagnostics(recovery),
                frame_check=dict(locked=check.locked, reason=check.reason, overall=check.overall,
                                 per_user=check.per_user, blocks=check.blocks) if check is not None else None,
                pulse_calibrations=recovery.pulse_calibrations))


def _json_value(value):
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='校验绑定IQ并离线复放PWM，不操作硬件')
    parser.add_argument('recipe', help='iq_bound_pwm_replay JSON')
    parser.add_argument('--output', help='另存诊断JSON（不覆盖已有文件）')
    args = parser.parse_args(argv)
    try:
        spectrum, recovery = run_recipe(args.recipe)
        summary = json.dumps(result_summary(spectrum, recovery), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            with Path(args.output).open('x', encoding='utf-8') as stream:
                stream.write(summary + '\n')
        else:
            print(summary)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, 'PWM复放失败：' + str(exc) + '\n')
    return 0


if __name__ == '__main__':
    main()

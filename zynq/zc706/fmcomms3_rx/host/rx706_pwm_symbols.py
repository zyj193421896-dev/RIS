"""Import MT (high %, delay %) codebooks without inferring a sender payload."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

PLUS_SOURCE = Path(__file__).resolve().parent/'frame_profiles/symbols_plus_78/mt_test_plus.txt'
PLUS_SHA256 = '7ed5818a505b4fb052dca969cdd461c0bbfd3338c13ffa2e4440544c261b02e7'
LEGACY_PROFILE = Path(__file__).resolve().parent/'frame_profiles/fixed_preamble10/frame.json'
LEGACY_SHA256 = 'afa8d1b0f87b964c63daee8cc410f6447f794c293cba1cefae811fe1ce4ca2a0'


def parse_mt_percent(text, step_percent=5):
    """Strict external format, preserved row order; result is (delay, width) chips."""
    step = float(step_percent)
    if not math.isfinite(step) or step <= 0:
        raise ValueError('片宽百分比须为有限正数')
    total = round(100/step)
    if not 3 <= total <= 256 or not math.isclose(total*step, 100, abs_tol=1e-7):
        raise ValueError('片宽须等分100%为3～256片')
    rows = []
    for line_number, line in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)\s*;?\s*', line)
        if not match:
            raise ValueError(f'第{line_number}行格式错误：须为(高电平百分比,时延百分比);')
        try:
            high, delay = map(float, match.groups())
        except ValueError:
            raise ValueError(f'第{line_number}行必须为数值') from None
        values = [delay/step, high/step]
        if any(not math.isfinite(v) or not math.isclose(v, round(v), abs_tol=1e-8) for v in values):
            raise ValueError(f'第{line_number}行必须为有限数且对齐{step:g}%片宽')
        d, w = map(round, values)
        if d < 0 or w < 1 or d+w >= total:
            raise ValueError(f'第{line_number}行须满足时延≥0、高电平>0，并至少保留一片低电平')
        rows.append((d, w))
    if not 2 <= len(rows) <= 32768 or len(set(rows)) != len(rows):
        raise ValueError('码表须有2～32768个互异码字，不可重复')
    return tuple(rows)


def chip_table(text):
    """Internal explicit (delay chips, high chips), never parse as external MT."""
    if isinstance(text, str):
        table = [tuple(map(float, row.replace(',', ' ').split()))
                 for row in text.replace(';', '\n').splitlines() if row.strip()]
    else:
        table = [tuple(map(float, row)) for row in text]
    if not table or any(len(row) != 2 or any(not math.isfinite(x) for x in row) for row in table):
        raise ValueError('原码表缺失或无效；不能按旧编号猜测新前导')
    return tuple(table)


def remap_symbols(symbols, old_table, new_table):
    from rx706_pwm_frames import parse_symbols
    old = chip_table(old_table)
    new = {tuple(row): i for i, row in enumerate(new_table)}
    ids = parse_symbols(symbols, len(old))
    try:
        return [new[old[i]] for i in ids]
    except KeyError as exc:
        raise ValueError(f'新码表缺少原前导物理波形{exc.args[0]}；请明确设置实际前导') from None


def import_mt_fields(fields, text):
    """Atomic draft conversion; preserve physical header, clear stale scored data."""
    result = copy.deepcopy(fields)
    step = float(result['duty_step'])
    table = parse_mt_percent(text, step)
    identity = str(result.get('identity', '')).strip()
    mapped = remap_symbols(identity, result['codebook'], table) if identity else []
    result.update(codebook='\n'.join(f'{d} {w}' for d, w in table),
        min_high_chips=str(min(w for d, w in table)),
        low_guard_chips=str(round(100/step)-max(d+w for d, w in table)),
        joint_delay=True, identity=' '.join(map(str, mapped)),
        payload_symbols='', reference_enabled=False, reference='', calibration_enabled=False,
        calibration_positions='', calibration_symbols='')
    return result


def plus_fields():
    from rx706_pwm_frame_builder import initial_fields
    source, legacy = PLUS_SOURCE.read_bytes(), LEGACY_PROFILE.read_bytes()
    if hashlib.sha256(source).hexdigest() != PLUS_SHA256 or hashlib.sha256(legacy).hexdigest() != LEGACY_SHA256:
        raise ValueError('78符号源文件或原10前导模板已改变；请通过帧向导明确导入实际码表')
    fields = import_mt_fields(initial_fields(json.loads(legacy.decode('utf-8-sig'))), source.decode('utf-8-sig'))
    fields['payload_symbols'] = ''  # Alphabet size is not frame payload length.
    return fields


def main():
    from rx706_pwm_frame_builder import build_profile, save_new_profile
    parser = argparse.ArgumentParser(description='78-symbol MT profile; payload length and actual symbol rate are explicit inputs.')
    parser.add_argument('--payload-symbols', type=int, required=True)
    parser.add_argument('--symbol-khz', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fields = plus_fields()
    fields.update(payload_symbols=str(args.payload_symbols), symbol_khz=str(args.symbol_khz))
    print(save_new_profile(args.output, build_profile(fields)))


if __name__ == '__main__':
    main()

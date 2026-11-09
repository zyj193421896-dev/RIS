"""Offline PWM bursts: shared known training, user marker, fixed payload.

All lengths are PWM symbols. Training/markers alone acquire the frame; known
payloads only score frozen decisions. Missing decisions retain their positions.
"""
from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import numpy as np
from rx706_spectrum import _cancel


FRAME_FIELDS = [('frame_part', 'U16'), ('user_id', 'i4'), ('block_id', 'i4'),
                ('payload_index', 'i4'), ('identity_ok', '?'), ('block_complete', '?')]
DEFAULT_TRAINING = '0 7 1 6 2 5 3 4'


def default_users(count=3, alphabet=8):
    """Deterministic marker suggestions; saved configuration defines the wire format."""
    rng = np.random.default_rng(7062026)
    codes = []
    while len(codes) < count:
        code = rng.permutation([0]*8+[alphabet-1]*8).tolist()
        # Avoid cyclic aliases as well as nearby words at the known boundary.
        if all(min(np.count_nonzero(np.roll(code, shift) != old)
                   for shift in range(16)) >= 4 for old in codes):
            codes.append(code)
    return [dict(name=f'U{i+1}', identity=' '.join(map(str, code)),
                 payload_symbols=1024, reference='') for i, code in enumerate(codes)]


def parse_symbols(value, alphabet, *, empty=False):
    if isinstance(value, str):
        text = value.strip()
        tokens = re.split(r'[\s,，;；]+', text) if text else []
    else:
        tokens = list(value)
    try:
        values = tuple(int(v) for v in tokens)
        if any(str(v).strip() != str(n) for v, n in zip(tokens, values)):
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise ValueError('帧序列须为空格分隔的整数符号值') from None
    if (not values and not empty) or any(not 0 <= v < alphabet for v in values):
        raise ValueError(f'帧序列须为0～{alphabet-1}的符号值；每个数表示一个完整PWM周期')
    return values


def _integer(value, name, lower, upper):
    try:
        result = int(value)
        if float(value) != result or not lower <= result <= upper:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name}须为{lower}～{upper}的整数') from None
    return result


def _reference(value, representation, alphabet):
    if representation == 'symbols' and not isinstance(value, str):
        return parse_symbols(value, alphabet, empty=True)
    text = str(value).strip()
    if text.startswith('@'):
        try:
            text = Path(text[1:]).read_text(encoding='utf-8-sig').strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f'无法读取已知净荷文件：{exc}') from exc
    if not text:
        return () if representation == 'symbols' else ''
    if representation == 'symbols':
        return parse_symbols(text, alphabet)
    bits = re.sub(r'[\s,_|]', '', text)
    if set(bits)-{'0', '1'}:
        raise ValueError('已知比特只能包含0、1和空白分隔符')
    return bits


def validate_frame_options(options, widths):
    """Normalize UI/API options without consulting received or expected payloads."""
    result = dict(options)
    alphabet = len(widths)
    if alphabet < 2:
        raise ValueError('最简帧至少需要两个占空比档位')
    for key, default, maximum in (('startup_symbols', 32, 65536),
                                   ('training_symbols', 64, 4096),
                                   ('search_symbols', 4096, 1000000)):
        result[key] = _integer(options.get(key, default), key, 0, maximum)
    result['startup_pattern'] = parse_symbols(options.get('startup_pattern', DEFAULT_TRAINING),
                                               alphabet, empty=not result['startup_symbols'])
    result['training_pattern'] = parse_symbols(options.get('training_pattern', DEFAULT_TRAINING),
                                                alphabet, empty=not result['training_symbols'])
    pattern = result['training_pattern']
    train = tuple(pattern[i % len(pattern)] for i in range(result['training_symbols']))
    result['training_sequence'] = train
    if train and any(train.count(k) < 2 for k in range(alphabet)):
        raise ValueError('公共训练须覆盖每个档位至少两次；请修改序列或增加训练长度')
    users = options['users'] if 'users' in options else default_users(alphabet=alphabet)
    count = _integer(options.get('user_count', len(users)), '用户数', 1, 16)
    if len(users) < count:
        raise ValueError('用户配置行数不足，请打开“用户身份与已知净荷”补充')
    representation = options.get('reference_format', 'symbols')
    if representation not in ('symbols', 'bits'):
        raise ValueError('参考格式须为symbols或bits')
    configured = []
    for i, user in enumerate(users[:count]):
        code = parse_symbols(user.get('identity', ''), alphabet)
        if not 4 <= len(code) <= 256:
            raise ValueError('每个用户身份码长度须为4～256个PWM符号')
        length = _integer(user.get('payload_symbols', 1024), '净荷长度', 1, 1000000)
        reference = _reference(user.get('reference', ''), representation, alphabet) if options.get('reference_enabled', False) else ''
        configured.append(dict(name=str(user.get('name', f'U{i+1}')), identity=code,
                               payload_symbols=length, reference=reference))
    errors = _integer(options.get('identity_errors', 1), '身份码最大错误数', 0, 255)
    for i, user in enumerate(configured):
        if 2*errors >= len(user['identity']):
            raise ValueError('身份码容错数须小于身份码长度的一半')
        for other in configured[:i]:
            a, b = user['identity'], other['identity']
            # Prefix ambiguity is invalid even when code lengths differ.
            if sum(x != y for x, y in zip(a, b)) <= 2*errors:
                raise ValueError('用户身份码或其前缀距离不足，无法按当前容错数唯一识别')
    result.update(user_count=count, users=configured, identity_errors=errors,
                  reference_format=representation,
                  reference_repeat=bool(options.get('reference_repeat', True)),
                  reference_reset=bool(options.get('reference_reset', True)))
    tolerance = float(options.get('training_tolerance', .3))
    if not math.isfinite(tolerance) or not 0 < tolerance < .5:
        raise ValueError('训练残差上限须大于0且小于0.5个档位间隔')
    result['training_tolerance'] = tolerance
    return result


def _calibrate(raw, usable, sequence, widths, tolerance):
    if not sequence:
        return np.asarray(widths, float)
    labels = np.asarray(sequence)
    if np.count_nonzero(usable) < .75*len(labels):
        return None
    groups = [raw[usable & (labels == k)] for k in range(len(widths))]
    if any(len(group) < 2 for group in groups):
        return None
    centers = np.array([np.median(group) for group in groups])
    if np.any(np.diff(centers) <= 0):
        return None
    slope, offset = np.polyfit(widths, centers, 1)
    spacing = np.min(np.diff(widths))
    if not .7 <= slope <= 1.4 or abs(offset) > .15*(max(widths)-min(widths)):
        return None
    if np.min(np.diff(centers)) < .4*spacing:
        return None
    residual = abs(raw[usable]-centers[labels[usable]])
    if np.quantile(residual, .9) > tolerance*np.min(np.diff(centers)):
        return None
    return centers


def _correct(raw, centers, widths):
    corrected = np.interp(raw, centers, widths)
    for left, right in ((0, 1), (-1, -2)):
        mask = raw < centers[0] if left == 0 else raw > centers[-1]
        corrected[mask] = widths[left]+(raw[mask]-centers[left])*(widths[right]-widths[left])/(centers[right]-centers[left])
    return corrected


def _identity(raw, usable, start, code, centers):
    stop = start+len(code)
    if stop > len(raw):
        return len(code)
    observed = raw[start:stop]
    distances = abs(observed[:, None]-centers)
    nearest = distances.argmin(axis=1)
    gap = np.min(np.diff(centers))
    good = usable[start:stop] & (distances.min(axis=1) <= .4*gap)
    return int(np.count_nonzero(~good | (nearest != code)))


def apply_frame_training(frames, total, options, widths, cancel_event=None):
    """Jointly acquire shared training and first identity; never search payload labels.

    Subsequent slot boundaries follow the configured round-robin schedule. A
    damaged identity retains its slot and becomes an erased payload block.
    """
    config = validate_frame_options(options, widths)
    widths = np.asarray(widths, float)
    if np.any(np.diff(widths) <= 0):
        raise ValueError('最简帧要求符号档位按占空比递增排列')
    n, train = len(frames), config['training_sequence']
    raw = frames['clock_high_chips' if config.get('clock_width', True) else 'high_chips'].copy()
    usable = frames['structure_ok'] & np.isfinite(raw) & (raw > 0) & (raw < total)
    frames['raw_reason'] = frames['reason']
    frames['frame_part'], frames['payload_index'] = '未定位', -1
    frames['user_id'], frames['block_id'] = 0, 0
    frames['identity_ok'], frames['block_complete'] = False, False
    frames['training'], frames['valid'] = True, False
    frames['symbol'], frames['bits'] = -3, '-'
    frames['calibration_id'] = 0
    frames['reason'] = '最简帧尚未定位：不输出净荷'
    for key in ('corrected_high_chips', 'corrected_one_us', 'corrected_zero_us',
                'width_correction_chips', 'width_scale'):
        frames[key] = np.nan
    found = None
    # First exact marker wins; tolerant matches require a unique best user.
    # Training is fitted separately for each candidate, so startup transients
    # never enter the calibration merely because they precede the first edge.
    stop = min(n, config['search_symbols'] or n)
    for at in range(len(train), stop):
        if at % 256 == 0:
            _cancel(cancel_event)
        centers = _calibrate(raw[at-len(train):at], usable[at-len(train):at],
                             train, widths, config['training_tolerance'])
        if centers is None:
            continue
        scores = [_identity(raw, usable, at, user['identity'], centers) for user in config['users']]
        best = min(scores)
        if best <= config['identity_errors'] and scores.count(best) == 1:
            found = at, scores.index(best), centers
            break
    if found is None:
        return
    at, first_user, centers = found
    train_start = at-len(train)
    buffer_start = max(0, train_start-config['startup_symbols'])
    frames['frame_part'][:buffer_start] = '前置采样'
    frames['frame_part'][buffer_start:train_start] = '启动缓冲'
    frames['frame_part'][train_start:at] = '公共训练'
    frames['reason'][:at] = '公共前缀：排除净荷和误码统计'
    high = _correct(raw, centers, widths)
    period_us = frames['one_us']+frames['zero_us']
    frames['calibration_id'][train_start:] = 1
    frames['calibration_source'][train_start:] = '公共已知训练档位表' if train else '标称档位（无训练）'
    frames['corrected_high_chips'] = high
    frames['width_correction_chips'] = high-frames['high_chips']
    frames['width_scale'] = np.polyfit(widths, centers, 1)[0]
    frames['corrected_one_us'] = high/total*period_us
    frames['corrected_zero_us'] = period_us-frames['corrected_one_us']
    block, user_at = 0, first_user
    while at < n:
        _cancel(cancel_event)
        block += 1
        user = config['users'][user_at]
        code = user['identity']
        payload = at+len(code)
        end = payload+user['payload_symbols']
        scores = [_identity(raw, usable, at, u['identity'], centers) for u in config['users']]
        ok = (scores[user_at] <= config['identity_errors'] and
              scores.count(min(scores)) == 1 and scores[user_at] == min(scores))
        frames['user_id'][at:end], frames['block_id'][at:end] = user_at+1, block
        frames['identity_ok'][at:end], frames['block_complete'][at:end] = ok, end <= n
        frames['frame_part'][at:payload] = '用户身份'
        frames['reason'][at:payload] = '用户身份码：不输出净荷' if ok else '用户身份码失败：保留块位置'
        frames['frame_part'][payload:end] = '净荷'
        frames['training'][payload:end] = False
        frames['symbol'][payload:end], frames['bits'][payload:end] = -1, '?'
        frames['payload_index'][payload:end] = np.arange(max(0, min(n, end)-payload))
        frames['reason'][payload:end] = '净荷等待档位判决' if ok else '身份码失败：净荷按擦除保位'
        at, user_at = end, (user_at+1) % config['user_count']


@dataclass
class FrameCheck:
    locked: bool
    expected: object
    row_expected_bits: object
    row_status: object
    row_bit_errors: object
    bitwords: object
    blocks: list = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    per_user: list = field(default_factory=list)
    reason: str = ''

    def summary(self):
        if not self.locked:
            return '最简帧未定位；不报告BER或误块率。'+self.reason
        rows = []
        for name, stats in [('整体', self.overall)]+[(u['name'], u) for u in self.per_user]:
            ber = f"{stats['bit_errors']}/{stats['compared_bits']}={stats['ber']:.6g}" if stats['ber'] is not None else '未测'
            bler = f"{stats['bad_blocks']}/{stats['evaluated_blocks']}={stats['bler']:.3%}" if stats['bler'] is not None else '未测'
            rows.append(f"{name}：完整块{stats['complete_blocks']}，尾部不完整块{stats['partial_blocks']}，"
                        f"身份失败{stats['identity_failures']}；BER（已判决位）{ber}；"
                        f"擦除{stats['erased_bits']}位；未覆盖参考{stats['unreferenced_blocks']}块；误块率{bler}。")
            if stats['estimated_symbols']:
                rows[-1] += f" 完整块中{stats['estimated_symbols']}个最近码字估计，按实际判决参与校验。"
        return ('\n'.join(rows)+'\n误块=完整净荷块有错比特、擦除或身份失败；仅参考覆盖整块时计误块率。'
                '尾块单列；参考从配置起点固定比对，不搜索最佳相位、不纠错。')


def _stats():
    return dict(complete_blocks=0, partial_blocks=0, identity_failures=0,
                evaluated_blocks=0, bad_blocks=0, bit_errors=0, compared_bits=0,
                erased_bits=0, unreferenced_blocks=0, estimated_symbols=0, ber=None, bler=None)


def check_frame_reference(frames, options, widths, bitwords, cancel_event=None):
    """Score complete scheduled blocks, including erased/identity-failed slots."""
    config = validate_frame_options(options, widths)
    n, word_size = len(frames), len(bitwords[0])
    result = FrameCheck(bool(np.any(frames['block_id'] > 0)),
        np.full(n, -1, dtype=np.int16), np.full(n, '', dtype=f'U{word_size}'),
        np.full(n, '非净荷/不统计', dtype='U24'), np.full(n, -1, dtype=np.int16), bitwords)
    result.overall = _stats()
    result.per_user = [dict(_stats(), name=u['name'], user_id=i+1) for i, u in enumerate(config['users'])]
    if not n:
        return result
    consumed = [0]*config['user_count']
    # Contiguous slices keep scoring linear even for hundreds of thousands of blocks.
    changes = np.flatnonzero(np.r_[True, np.diff(frames['block_id']) != 0, True])
    for begin, end in zip(changes[:-1], changes[1:]):
        if not frames['block_id'][begin]:
            continue
        _cancel(cancel_event)
        rows = np.arange(begin, end)
        user_at = int(frames['user_id'][begin])-1
        user, stats = config['users'][user_at], result.per_user[user_at]
        positions = rows[frames['frame_part'][rows] == '净荷']
        complete = bool(frames['block_complete'][begin])
        identity_ok = bool(frames['identity_ok'][begin])
        estimated = (int(np.count_nonzero(frames['decision_estimated'][positions] & frames['valid'][positions]))
                     if 'decision_estimated' in frames.dtype.names else 0)
        block = dict(block_id=int(frames['block_id'][begin]), user_id=user_at+1,
                     user=user['name'], first_row=int(begin), payload_first_row=int(positions[0]) if len(positions) else int(end),
                     payload_symbols=len(positions), complete=complete, identity_ok=identity_ok,
                     evaluated=False, bit_errors=0, compared_bits=0, erased_bits=0, estimated_symbols=estimated, bad=None)
        result.blocks.append(block)
        stats['complete_blocks' if complete else 'partial_blocks'] += 1
        stats['identity_failures'] += int(complete and not identity_ok)
        if not complete:
            result.row_status[positions] = '尾部不完整块/不统计'
            continue
        stats['estimated_symbols'] += estimated
        units = user['payload_symbols']*(word_size if config['reference_format'] == 'bits' else 1)
        start = 0 if config['reference_reset'] else consumed[user_at]
        consumed[user_at] += units
        reference = user['reference']
        if not reference or (not config['reference_repeat'] and start+units > len(reference)):
            stats['unreferenced_blocks'] += 1
            result.row_status[positions] = '参考未覆盖/不统计'
            continue
        if config['reference_repeat']:
            expected = [reference[i % len(reference)] for i in range(start, start+units)]
        else:
            expected = reference[start:start+units]
        if config['reference_format'] == 'symbols':
            expected_symbols = list(expected)
            expected_bits = [bitwords[k] for k in expected_symbols]
        else:
            bit_string = ''.join(expected)
            expected_bits = [bit_string[i:i+word_size] for i in range(0, len(bit_string), word_size)]
            lookup = {word: i for i, word in enumerate(bitwords)}
            expected_symbols = [lookup.get(word, -1) for word in expected_bits]
        result.expected[positions] = expected_symbols
        result.row_expected_bits[positions] = expected_bits
        valid = frames['valid'][positions] & identity_ok
        errors = np.array([sum(a != b for a, b in zip(str(frames['bits'][row]), word))
                           if good else -1 for row, word, good in zip(positions, expected_bits, valid)], dtype=np.int16)
        result.row_bit_errors[positions] = errors
        result.row_status[positions] = np.where(valid, np.where(errors > 0, '不匹配', '匹配'), '擦除')
        bit_errors, compared = int(errors[valid].sum()), int(valid.sum())*word_size
        erased = int((~valid).sum())*word_size
        bad = bool(bit_errors or erased or not identity_ok)
        block.update(evaluated=True, bit_errors=bit_errors, compared_bits=compared, erased_bits=erased, bad=bad)
        for key, increment in dict(evaluated_blocks=1, bad_blocks=int(bad), bit_errors=bit_errors,
                                   compared_bits=compared, erased_bits=erased).items():
            stats[key] += increment
    for stats in result.per_user:
        for key, value in stats.items():
            if key in result.overall and key not in ('ber', 'bler'):
                result.overall[key] += value
    for stats in [result.overall]+result.per_user:
        stats['ber'] = stats['bit_errors']/stats['compared_bits'] if stats['compared_bits'] else None
        stats['bler'] = stats['bad_blocks']/stats['evaluated_blocks'] if stats['evaluated_blocks'] else None
    return result

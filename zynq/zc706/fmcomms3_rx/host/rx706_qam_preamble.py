"""Known-symbol QAM acquisition. No PWM width calibration or board access."""
import math
import re
import numpy as np
from rx706_spectrum import _cancel


def points_from_labels(labels, order):
    side = math.isqrt(order)
    labels = np.asarray(labels, dtype=int)
    axes = []
    for gray in (labels // side, labels % side):
        binary = gray.copy()
        shift = 1
        while shift < side:
            binary ^= binary >> shift
            shift *= 2
        axes.append(2 * binary - side + 1)
    return (axes[0] + 1j * axes[1]) / math.sqrt(2 * (order - 1) / 3)


def parse_preamble(text, representation, order):
    """Labels use receiver Gray mapping; IQ input uses unnormalized odd levels."""
    if order not in (16, 64, 256):
        raise ValueError('QAM阶数须为16、64或256')
    tokens = re.split(r'[;；\s]+', str(text).strip())
    if not 3 <= len(tokens) <= 256:
        raise ValueError('QAM前导须填写3～256个已知符号')
    if representation == 'labels':
        try:
            labels = [int(t, 16) if t.lower().startswith('0x') else
                      int(t, 2) if t.lower().startswith('0b') else int(t) for t in tokens]
        except ValueError:
            raise ValueError('前导符号值用空格分隔；二进制加0b，十六进制加0x') from None
        if any(v < 0 or v >= order for v in labels):
            raise ValueError(f'前导符号值须在0～{order - 1}内')
        return points_from_labels(labels, order)
    if representation != 'iq':
        raise ValueError('未知QAM前导表示方式')
    try:
        pairs = [tuple(float(v) for v in t.replace('，', ',').split(',')) for t in tokens]
        if any(len(p) != 2 for p in pairs):
            raise ValueError()
        points = np.array([complex(*p) for p in pairs])
    except (ValueError, TypeError):
        raise ValueError('I/Q前导格式为 I,Q I,Q I,Q，例如 -15,-15 -15,-15 -15,-15') from None
    side = math.isqrt(order)
    levels = np.arange(1 - side, side, 2)
    if not np.all(np.isin(points.real, levels) & np.isin(points.imag, levels)):
        raise ValueError(f'I/Q须为未归一化星座电平：{1 - side}、{3 - side}…{side - 1}')
    return points / math.sqrt(2 * (order - 1) / 3)


def validate_preamble(options, order, symbol_hz):
    points = parse_preamble(options.get('symbols', ''), options.get('representation', 'labels'), order)
    bound = float(options.get('max_frequency_hz', 1000))
    tolerance = float(options.get('max_evm_percent', 10))
    search_s = float(options.get('search_s', 1))
    if not math.isfinite(bound) or not 0 < bound < symbol_hz / 2:
        raise ValueError('残余频偏范围须大于0且小于符号率的一半')
    if not math.isfinite(tolerance) or not 0 < tolerance <= 50:
        raise ValueError('前导匹配EVM上限须在0～50%内且不为0')
    if not math.isfinite(search_s) or not .001 <= search_s <= 1:
        raise ValueError('前导搜索时长须在1～1000 ms内')
    return points, bound, tolerance, search_s


def _sample(x, indices):
    lo = indices.astype(np.int64)
    fraction = indices - lo
    return x[lo] * (1 - fraction) + x[lo + 1] * fraction


def acquire_preamble(x, fs, symbol_hz, margin, points, bound, tolerance,
                     search_s, decide, order, cancel_event=None, *,
                     start_index=0., start_window_symbols=None):
    """Search fractional timing and start; fit phase slope after pilot removal.

    A short following data window ranks candidates but cannot prove frame lock.
    Pilots are never inferred from three arbitrary equal received decisions.
    """
    sps, length = fs / symbol_hz, len(points)
    candidates = []
    search_end = min(len(x) - margin - 1, start_index + search_s * fs)
    if start_window_symbols is not None:
        search_end = min(search_end, start_index + start_window_symbols*sps)
    # Include the first symbol: filtering keeps its centre even at file start.
    for offset in np.linspace(0, sps, 32, endpoint=False):
        _cancel(cancel_event)
        first_index = start_index + offset
        indices = np.arange(first_index, len(x) - margin - 1, sps)
        z = _sample(x, indices)
        count = min(len(z) - length - 16, int(np.floor((search_end - first_index) / sps)) + 1)
        if count <= 0:
            continue
        # Blocked search limits temporary storage for long captures/preambles.
        for start in range(0, count, 4096):
            _cancel(cancel_event)
            stop = min(count, start + 4096)
            windows = np.lib.stride_tricks.sliding_window_view(z[start:stop + length - 1], length)
            ratios = windows / points
            adjacent = np.sum(ratios[:, 1:] * ratios[:, :-1].conj(), axis=1)
            frequency = np.angle(adjacent) * symbol_hz / (2 * np.pi)
            # Least squares on unwrapped relative phases, at the coarse slope.
            j = np.arange(length)
            residual = np.unwrap(np.angle(ratios * np.exp(-2j * np.pi * frequency[:, None] * j / symbol_hz)), axis=1)
            centered = j - j.mean()
            frequency += (residual @ centered) / np.sum(centered ** 2) * symbol_hz / (2 * np.pi)
            rotated = windows * np.exp(-2j * np.pi * frequency[:, None] * j / symbol_hz)
            gain = (rotated @ points.conj()) / np.sum(abs(points) ** 2)
            power = abs(gain) ** 2 * np.mean(abs(points) ** 2)
            error = np.mean(abs(rotated - gain[:, None] * points) ** 2, axis=1) / np.maximum(power, 1e-30)
            valid = (abs(frequency) <= bound) & (power > 1e-12) & (error <= (tolerance / 100) ** 2)
            choices = np.flatnonzero(valid)
            if not len(choices):
                continue
            choices = choices[np.argsort(error[choices])[:8]]
            for k in choices:
                position = start + int(k)
                candidates.append((float(error[k]), indices[position], float(frequency[k]), gain[k], offset))
    if not candidates:
        raise ValueError('未找到匹配的QAM前导；请核对符号、映射、符号率、频偏范围和搜索时长')
    # Keep search cost bounded; independent starts remain eligible, not just one run.
    candidates.sort(key=lambda c: c[0])
    best = None
    for pilot_error, index, frequency, gain, offset in candidates[:512]:
        _cancel(cancel_event)
        indices = index + np.arange(length, length + 128) * sps
        indices = indices[indices < len(x) - margin - 1]
        if len(indices) < 16:
            continue
        data = _sample(x, indices) / gain * np.exp(-2j * np.pi * frequency * (indices - index) / fs)
        ideal, _, _ = decide(data, order)
        data_error = np.mean(abs(data - ideal) ** 2)
        score = pilot_error + data_error
        if best is None or score < best[0]:
            best = (score, index, frequency, gain, pilot_error, data_error)
    if best is None:
        raise ValueError('前导后至少需要16个数据符号用于候选核对')
    _, index, frequency, gain, pilot_error, data_error = best
    return dict(index=index, frequency_hz=frequency, gain=gain,
                evm_percent=100 * math.sqrt(pilot_error),
                check_evm_percent=100 * math.sqrt(data_error), length=length)

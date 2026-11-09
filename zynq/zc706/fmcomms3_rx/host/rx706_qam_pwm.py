"""Acquire a short known PWM v2 header before QAM, without a long PWM clock fit."""
import math
import re
import numpy as np
from rx706_spectrum import _cancel
from rx706_pwm import pwm2_layout
from rx706_qam_preamble import _sample


def _bounded_header_frequency(z, high, fs, bound, cancel_event=None):
    """Fit the two known plateaus inside the configured CFO range.

    A one-sample phase correlation can be dominated by coloured receiver noise.
    This model uses only the known PWM states, never unknown QAM decisions.
    """
    times = np.arange(len(z))/fs
    good = np.ones(len(z), dtype=bool)
    reach = 3
    for edge in np.r_[0, np.flatnonzero(high[1:] != high[:-1])+1, len(z)]:
        good[max(0, edge-reach):min(len(z), edge+reach+1)] = False
    if min(np.count_nonzero(good & high), np.count_nonzero(good & ~high)) < 3:
        raise ValueError('PWM稳定平台不足以估计频偏')
    sample, t, state = z[good], times[good], high[good]

    def fit(frequency):
        _cancel(cancel_event)
        v = sample*np.exp(-2j*np.pi*frequency*t)
        low, top = v[~state].mean(), v[state].mean()
        return float(np.mean(abs(v-np.where(state, top, low))**2))

    # Resolve the main lobe over the actual known header duration. The caller
    # initially fits only three symbols; the full sequence is still verified.
    cells = min(513, max(33, int(math.ceil(8*bound*len(z)/fs))+1))
    grid = np.linspace(-bound, bound, cells)
    scores = [fit(v) for v in grid]
    at = int(np.argmin(scores))
    if at in (0, len(grid)-1):
        raise ValueError('PWM频偏最优值位于搜索边界，不能确认在配置范围内')
    left, right = grid[max(0, at-1)], grid[min(len(grid)-1, at+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(24):
        a, b = right-ratio*(right-left), left+ratio*(right-left)
        if fit(a) < fit(b): right = b
        else: left = a
    return float((left+right)/2)


def _bounded_header_fallback(x, fs, candidates, high, probe_size, step, limit,
                             bound, cancel_event):
    """Bounded recovery when the adjacent-cell CFO proposal was rejected."""
    size = len(high)
    positions = np.arange(size)
    times = positions/fs
    ranked = []
    for _, coarse in candidates[:32]:
        _cancel(cancel_event)
        probe = _sample(x, coarse+np.arange(probe_size))
        try:
            frequency = _bounded_header_frequency(probe, high[:probe_size], fs, bound, cancel_event)
        except ValueError:
            continue
        z = _sample(x, coarse+positions)*np.exp(-2j*np.pi*frequency*times)
        low, top = z[~high].mean(), z[high].mean()
        error = float(np.sqrt(np.mean(abs(z-np.where(high, top, low))**2))/max(abs(top-low), 1e-15))
        ranked.append((error, coarse, frequency))
    ranked.sort()
    best = None
    for _, coarse, frequency in ranked[:8]:
        _cancel(cancel_event)
        rotation = np.exp(-2j*np.pi*frequency*times)
        for index in np.arange(max(0, coarse-step), min(limit, coarse+step)+.01, .25):
            z = _sample(x, index+positions)*rotation
            low, top = z[~high].mean(), z[high].mean()
            error = float(np.sqrt(np.mean(abs(z-np.where(high, top, low))**2))/max(abs(top-low), 1e-15))
            if best is None or error < best[0]: best = error, float(index), frequency
    if best is None: return None
    # Refine CFO with all known symbols, then check the original complete-header
    # residual. Keep exactly the original state-distance-normalized tolerance.
    _, index, _ = best
    z = _sample(x, index+positions)
    try:
        frequency = _bounded_header_frequency(z, high, fs, bound, cancel_event)
    except ValueError:
        return None
    z *= np.exp(-2j*np.pi*frequency*times)
    low, top = z[~high].mean(), z[high].mean()
    error = float(np.sqrt(np.mean(abs(z-np.where(high, top, low))**2))/max(abs(top-low), 1e-15))
    return error, index, frequency


def pwm_header_layout(total, prefix, guard):
    """Known headers may be fixed-width; standalone PWM v2 remains unchanged."""
    if (all(type(v) is int for v in (total, prefix, guard)) and
            3 <= total <= 256 and prefix >= 1 and guard >= 1 and prefix+guard == total):
        return 0
    return pwm2_layout(total, prefix, guard)


def validate_pwm_header(options, fs):
    total = int(options.get('total_chips', 25))
    prefix = int(options.get('prefix_chips', 5))
    guard = int(options.get('guard_chips', 8))
    code = pwm_header_layout(total, prefix, guard)
    try:
        sequence = [int(v) for v in re.split(r'[\s,，;；]+', str(options.get('symbols', '0 0 0')).strip())]
    except ValueError:
        raise ValueError('PWM前导填写version2符号值，用空格分隔，例如0 0 0') from None
    if not 3 <= len(sequence) <= 256 or any(v < 0 or v > code for v in sequence):
        raise ValueError(f'PWM前导须为3～256个符号，每个在0～{code}内')
    rate = float(options.get('symbol_hz', 100000))
    gap = float(options.get('gap_s', 0))
    search = float(options.get('search_s', 1))
    mismatch = float(options.get('max_error_percent', 25))
    width_offset = float(options.get('width_offset_chips', 0))
    bound = float(options.get('max_frequency_hz', 1000))
    window = float(options.get('qam_search_symbols', 8))
    # Known high/low runs are fitted jointly; individual chips are not decoded.
    # Retain the >=3-sample plateau and full-fit checks in acquire_pwm_header.
    if not math.isfinite(rate) or rate <= 0 or fs / rate / total < 2:
        raise ValueError('PWM前导符号率须大于0，且每时间片至少2个原始采样点')
    if not math.isfinite(gap) or not 0 <= gap <= 1:
        raise ValueError('PWM到QAM的切换间隔须在0～1000000 µs内')
    if not math.isfinite(search) or not .001 <= search <= 1:
        raise ValueError('PWM前导搜索时长须在1～1000 ms内')
    if not math.isfinite(mismatch) or not 0 < mismatch <= 50:
        raise ValueError('PWM前导拟合误差上限须大于0且不超过50%')
    if not math.isfinite(width_offset) or any(not 0 < prefix+v+width_offset < total for v in sequence):
        raise ValueError('PWM脉宽偏移后，每个前导符号的高片数须大于0且小于总片数')
    if not math.isfinite(bound) or not 0 < bound < rate/2:
        raise ValueError('PWM残余频偏范围须大于0且小于PWM符号率的一半')
    if not math.isfinite(window) or not 0 <= window <= 256:
        raise ValueError('QAM起始搜索窗须为0～256个QAM符号；0为紧接PWM的首窗')
    return dict(total_chips=total, prefix_chips=prefix, guard_chips=guard, symbols=sequence,
                symbol_hz=rate, gap_s=gap, search_s=search, max_error_percent=mismatch,
                width_offset_chips=width_offset, max_frequency_hz=bound, qam_search_symbols=window)


def acquire_pwm_header(x, fs, options, cancel_event=None):
    """Fit two complex levels to known high/low runs, allowing either polarity.

    Error is RMS residual divided by state distance, not PWM bit error rate.
    The fitted state phase is not treated as QAM's absolute phase reference.
    """
    o = validate_pwm_header(options, fs)
    period = fs/o['symbol_hz']
    length = len(o['symbols'])*period
    size = int(round(length))
    limit = min(len(x)-size-2, int(o['search_s']*fs))
    if limit < 0:
        raise ValueError('恢复片段不足以容纳完整PWM前导')
    starts = np.rint(np.arange(len(o['symbols']))*period).astype(int)
    falls = np.rint((np.arange(len(o['symbols']))+
                    (o['prefix_chips']+np.array(o['symbols'])+o['width_offset_chips'])/o['total_chips'])*period).astype(int)
    if np.any(falls-starts < 3) or np.any(np.r_[starts[1:],size]-falls < 3):
        raise ValueError('PWM前导实测高/低持续时间过短，至少需要3个采样点')
    # A long header can rotate significantly before CFO correction. Search its
    # first three symbols, then verify the COMPLETE sequence after fitting CFO.
    probe_size = int(round(3*period))
    high_count = int(np.sum(falls[:3]-starts[:3]))
    low_count = probe_size-high_count
    sums = np.r_[0j, np.cumsum(x)]
    powers = np.r_[0., np.cumsum(abs(x)**2)]
    step = max(1, int(period/o['total_chips']/2))
    candidates = []
    for first in range(0, limit+1, step*4096):
        _cancel(cancel_event)
        at = np.arange(first, min(limit+1, first+step*4096), step)
        high = np.zeros(len(at), complex)
        for a, b in zip(starts[:3], falls[:3]):
            high += sums[at+b]-sums[at+a]
        whole = sums[at+probe_size]-sums[at]
        low = whole-high
        a, b = high/high_count, low/low_count
        energy = powers[at+probe_size]-powers[at]
        variance = np.maximum(0, energy-abs(high)**2/high_count-abs(low)**2/low_count)/probe_size
        distance = abs(a-b)**2
        error = np.sqrt(variance/np.maximum(distance, 1e-30))
        good = np.flatnonzero((distance > 1e-8) & (error <= o['max_error_percent']/100))
        for k in good[np.argsort(error[good])[:4]]:
            candidates.append((float(error[k]), int(at[k])))
    if not candidates:
        raise ValueError('未找到匹配的PWM前导；请核对符号序列、片数、PWM符号率、脉宽偏移和搜索范围')
    candidates.sort()
    # Refit nearby sample positions and estimate common CFO only INSIDE plateaus.
    high = np.zeros(size, dtype=bool)
    for a, b in zip(starts, falls):
        high[a:b] = True
    interior = high[1:] == high[:-1]
    # Avoid interpolation across edges and symbol boundaries in the CFO fit.
    interior &= np.r_[False, interior[:-1]] & np.r_[interior[1:], False]
    best = None
    for _, coarse in candidates[:32]:
        _cancel(cancel_event)
        for index in np.arange(max(0, coarse-step), min(limit, coarse+step)+.01, .25):
            z = _sample(x, index+np.arange(size))
            correlation = np.vdot(z[:-1][interior], z[1:][interior])
            frequency = float(np.angle(correlation)*fs/(2*np.pi))
            if abs(frequency) > o['max_frequency_hz']:
                continue
            z = z*np.exp(-2j*np.pi*frequency*np.arange(size)/fs)
            centers = np.array([np.mean(z[~high]), np.mean(z[high])])
            distance = abs(centers[1]-centers[0])
            error = float(np.sqrt(np.mean(abs(z-centers[high.astype(int)])**2))/max(distance, 1e-15))
            if best is None or error < best[0]:
                best = error, index, frequency
    frequency_method = 'adjacent_plateau_correlation'
    if best is None or best[0] > o['max_error_percent']/100:
        bounded = _bounded_header_fallback(x, fs, candidates, high, probe_size,
            step, limit, o['max_frequency_hz'], cancel_event)
        if bounded is not None and (best is None or bounded[0] < best[0]):
            best = bounded
            frequency_method = 'bounded_known_pwm_fit'
    if best is None or best[0] > o['max_error_percent']/100:
        raise ValueError('PWM前导未通过两态拟合或残余频偏范围检查')
    error, index, frequency = best
    return dict(index=float(index), end_index=float(index+length),
                qam_index=float(index+length+o['gap_s']*fs), frequency_hz=frequency,
                error_percent=100*error, frequency_method=frequency_method,
                symbols=o['symbols'], symbol_hz=o['symbol_hz'],
                gap_s=o['gap_s'], qam_search_symbols=o['qam_search_symbols'],
                total_chips=o['total_chips'], prefix_chips=o['prefix_chips'], guard_chips=o['guard_chips'])

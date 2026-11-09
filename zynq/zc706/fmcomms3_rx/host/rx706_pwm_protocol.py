"""Explicit PWM width alphabet and causal, configurable preamble calibration.

The payload reference/order is deliberately not an input to this module.
"""
import math
from fractions import Fraction
import numpy as np
from rx706_spectrum import _cancel


def duty_protocol(start_percent, step_percent, symbols):
    """Compile the four-field duty UI to an exact width alphabet."""
    try:
        start, step = Fraction(str(start_percent))/100, Fraction(str(step_percent))/100
        count = int(symbols)
    except (ValueError, ZeroDivisionError):
        raise ValueError('起始占空比、步长和符号数量须为有效数字') from None
    if not 2 <= count <= 256 or step <= 0 or start <= 0 or start+step*(count-1) >= 1:
        raise ValueError('符号数量须为2～256；步长须为正，起始和结束占空比须在0%与100%之间')
    total = math.lcm(start.denominator, step.denominator)
    if total > 256:
        raise ValueError('当前占空比精度需要超过256个时间单位；请简化小数步长')
    widths = tuple(int((start+k*step)*total) for k in range(count))
    return dict(total_chips=total, prefix_chips=widths[0], guard_chips=total-widths[-1],
                symbol_widths=widths, output_binary=True, decision_mode='width'), tuple(
                    float((start+k*step)*100) for k in range(count))


def preamble_duties(start_percent, step_percent, symbols):
    """Generate an independent preamble range, including repeated widths."""
    try:
        start, step = Fraction(str(start_percent)), Fraction(str(step_percent))
        number = Fraction(str(symbols))
        count = int(number)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        raise ValueError('前导参数须为有效数字') from None
    if number != count or not 3 <= count <= 256 or step < 0 or not 0 < start <= start+step*(count-1) < 100:
        raise ValueError('前导数量须为3～256的整数；步长可为0；占空比须在0%与100%之间')
    return tuple(float(start+i*step) for i in range(count))


def parse_preamble_widths(value, total):
    try:
        tokens = value.replace(',', ' ').split() if isinstance(value, str) else value
        widths = tuple(float(v) for v in tokens)
    except (TypeError, ValueError):
        raise ValueError('前导宽度须为有效数字') from None
    if not 3 <= len(widths) <= 256 or any(not math.isfinite(v) or not 0 < v < total for v in widths):
        raise ValueError('前导须含3～256个符号，宽度须在0与周期之间')
    return widths


def apply_alphabet_decisions(frames, widths, prefix, code, tolerance, require_training=False,
                             nearest=False):
    """Decide configured levels; optionally keep nearest out-of-tolerance data."""
    widths = np.sort(np.asarray(widths))
    gaps = np.diff(widths)
    for row in frames:
        if row['symbol'] in (-2, -3):
            continue
        row['valid'], row['symbol'], row['bits'] = False, -1, '?'*code
        if require_training and row['calibration_id'] <= 0:
            row['reason'] = '尚未检测到已配置前导'
            continue
        if not row['structure_ok']:
            row['reason'] = '原始边沿/跳变不完整，不能确认符号'
            continue
        high = (float(row['corrected_high_chips']) if require_training else float(row['high_chips']))
        if not math.isfinite(high):
            row['reason'] = '没有可用脉宽'
            continue
        at = int(np.argmin(abs(widths-high)))
        spacing = min(gaps[max(0, at-1)], gaps[min(len(gaps)-1, at)]) if len(gaps) else 1
        outside = abs(high-widths[at]) > tolerance*spacing
        if outside and not nearest:
            row['reason'] = '脉宽偏离已配置占空比档位'
            continue
        k = int(widths[at])-prefix
        row['valid'], row['symbol'], row['bits'] = True, k, '1'*k+'0'*(code-k)
        row['reason'] = ('有效（最近档位恢复；超容差）' if outside else '有效（占空比档位判决）')


def parse_widths(value, total, *, minimum=1, maximum=None):
    maximum = total-1 if maximum is None else maximum
    try:
        tokens = value.replace(',', ' ').split() if isinstance(value, str) else value
        widths = tuple(float(v) for v in tokens)
    except (TypeError, ValueError):
        raise ValueError('高片数序列须为空格分隔的整数') from None
    if not widths or any(not math.isfinite(v) or v != round(v) or
                         not minimum <= v <= maximum for v in widths):
        raise ValueError(f'高片数须为{minimum}～{maximum}的整数，序列不能为空')
    return tuple(int(v) for v in widths)


def validate_protocol_training(total, prefix, code, options):
    # A preamble may be outside the payload alphabet, but must fit the period.
    preamble = parse_preamble_widths(options['preamble_widths'], total)
    if not 3 <= len(preamble) <= 256:
        raise ValueError('前导序列长度须为3～256个符号')
    tolerance = float(options.get('same_tolerance_chips', .5))
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('前导匹配误差上限须大于0且小于1片')
    max_offset = float(options.get('max_offset_chips', 3.))
    if not math.isfinite(max_offset) or not 0 < max_offset < total/2:
        raise ValueError('前导最大宽度偏移须大于0且小于半个符号周期')
    widths = parse_widths(options.get('alphabet_widths', range(prefix, prefix+code+1)),
                          total, minimum=prefix, maximum=prefix+code)
    if len(set(widths)) != len(widths):
        raise ValueError('符号映射表中的高片数不能重复')
    payload = options.get('payload_symbols', 0)
    if type(payload) is not int or not 0 <= payload <= 1000000:
        raise ValueError('每帧净荷符号数须为0～1000000的整数；0表示不固定帧长')
    return np.asarray(preamble, dtype=float), tolerance, max_offset, np.sort(widths)


def _alphabet_scale(samples, widths):
    """Estimate spacing only after every configured level is well populated.

    No payload ordering/labels or BER reference is consulted. Incomplete or
    non-affine alphabets fail closed; caller keeps the pilot-only correction.
    """
    count = len(widths)
    if not 3 <= count <= 64 or len(samples) < max(48, 4*count):
        return None
    data = np.asarray(samples, dtype=float)
    low, high = np.quantile(data, [.02, .98])
    centers = low+(widths-widths[0])/(widths[-1]-widths[0])*(high-low)
    # Endpoint seeds avoid splitting the frequent preamble-width level. The
    # fit below rejects collapsed/missing levels, without inventing endpoints.
    for _ in range(24):
        labels = abs(data[:, None]-centers).argmin(axis=1)
        groups = [data[labels == j] for j in range(count)]
        if any(len(g) < 3 for g in groups):
            return None
        updated = np.array([np.median(g) for g in groups])
        if np.max(abs(updated-centers)) < 1e-5:
            centers = updated
            break
        centers = updated
    slope, intercept = np.polyfit(widths, centers, 1)
    gap = float(np.min(np.diff(widths)))
    if not .8 <= slope <= 1.25:
        return None
    if np.max(abs(centers-(slope*widths+intercept)))/slope > .35*gap:
        return None
    if np.min(np.diff(centers)) < .55*slope*gap:
        return None
    if any(np.quantile(abs(g-c), .9)/slope > .35*gap for g, c in zip(groups, centers)):
        return None
    return float(slope), float(intercept), centers


def apply_protocol_training(frames, total, prefix, code, guard, width_tolerance,
                            options, cancel_event=None, *, measurement_uncertainty_chips=0.):
    preamble, match_error, max_offset, alphabet = validate_protocol_training(total, prefix, code, options)
    if (not math.isfinite(measurement_uncertainty_chips) or
            not 0 <= measurement_uncertainty_chips < total/2):
        raise ValueError('测宽采样余量须为有限非负数且小于半个周期')
    length = len(preamble)
    multi = np.ptp(preamble) > 1e-9
    auto_scale = bool(options.get('auto_scale', False)) and not multi
    raw = frames['clock_high_chips' if options.get('clock_width', False) else 'high_chips'].copy()
    idle = frames['symbol'] == -2
    usable = frames['structure_ok'] & np.isfinite(raw) & (raw > 0) & (raw < total)
    frames['raw_reason'] = frames['reason']
    frames['valid'], frames['symbol'], frames['bits'] = False, -1, '?'*code
    frames['training'], frames['calibration_id'], frames['calibration_source'] = False, 0, ''
    for key in ('corrected_high_chips', 'corrected_one_us', 'corrected_zero_us',
                'width_correction_chips', 'width_scale'):
        frames[key] = np.nan
    identifier, marker_end = 0, -1
    scale, observed_pilot, pilot_width = 1., None, float(preamble[-1])
    intercept, method = 0., '前导偏移'
    samples, pilots = [], []
    pilot_history, learned_centers = [], None
    constant_run = False
    payload_count, frame_origin = options.get('payload_symbols', 0), None
    last_fit_index = -32

    def corrected(at):
        row = frames[at]
        high = (raw[at]-intercept)/scale
        if learned_centers is not None:
            # Interpolate calibrated width levels, allowing mild repeatable
            # nonlinearity. Extrapolation still faces the normal range checks.
            drift = float(np.median(pilot_history))-float(np.interp(pilot_width, alphabet, learned_centers))
            value = raw[at]-drift
            high = float(np.interp(value, learned_centers, alphabet))
            if value < learned_centers[0]:
                high = alphabet[0]+(value-learned_centers[0])/scale
            elif value > learned_centers[-1]:
                high = alphabet[-1]+(value-learned_centers[-1])/scale
        period_us = float(row['one_us']+row['zero_us'])
        row['calibration_id'], row['width_scale'] = identifier, scale
        row['calibration_source'] = method
        row['width_correction_chips'] = high-row['high_chips']
        row['corrected_high_chips'] = high
        row['corrected_one_us'] = high/total*period_us
        row['corrected_zero_us'] = period_us-row['corrected_one_us']
        return high

    for i, row in enumerate(frames):
        if i % 2048 == 0:
            _cancel(cancel_event)
        if idle[i]:
            row['symbol'], row['bits'], row['reason'] = -2, '-', '空闲：不输出数据比特'
            continue
        if constant_run and (not usable[i] or abs(raw[i]-observed_pilot) >
                             match_error*scale+measurement_uncertainty_chips):
            constant_run = False
        scheduled = (payload_count and frame_origin is not None and
                     (i-frame_origin) % (length+payload_count) < length)
        if scheduled and (i-frame_origin) % (length+payload_count) < length-1:
            corrected(i)
            row['training'], row['symbol'], row['bits'] = True, -3, '-'
            row['reason'] = '按已配置帧长排除前导；等待完整校准'
            continue
        match = False
        first = i-length+1
        permitted = (not payload_count or frame_origin is None or
                     (first-frame_origin) % (length+payload_count) == 0)
        if (permitted and first > marker_end and (not constant_run or scheduled) and
                first >= 0 and np.all(usable[first:i+1])):
            observed = raw[first:i+1]
            if multi:
                trial_scale, trial_intercept = np.polyfit(preamble, observed, 1)
            else:
                trial_scale = scale
                trial_intercept = float(np.median(observed))-trial_scale*pilot_width
            residual = abs(observed-(trial_scale*preamble+trial_intercept))
            # Tolerance bounds each symbol's error about the fitted preamble.
            # Using peak-to-peak for equal-width pilots incorrectly applies
            # twice the restriction used for a multi-width preamble: two
            # acceptable errors on opposite sides can reject the whole run.
            match = (.8 <= trial_scale <= 1.25 and abs(trial_intercept) <= max_offset and
                     np.max(residual) <= match_error*trial_scale+measurement_uncertainty_chips)
        if match:
            if frame_origin is None:
                frame_origin = first
            scale, intercept = float(trial_scale), float(trial_intercept)
            observed_pilot = float(np.median(raw[first:i+1]))
            if not multi:
                pilot_history.append(observed_pilot)
                pilot_history = pilot_history[-32:]
                intercept = float(np.median(pilot_history))-scale*pilot_width
            identifier += 1
            marker_end, constant_run = i, not multi
            method = '多宽前导比例＋偏移' if multi else ('前导＋档位标定' if learned_centers is not None else '前导偏移')
            # A candidate may previously have entered the unlabeled history.
            samples = [(j, v) for j, v in samples if j < first]
            for j in range(first, i+1):
                corrected(j)
                frames['training'][j] = True
                frames['valid'][j], frames['symbol'][j], frames['bits'][j] = False, -3, '-'
                frames['reason'][j] = '已配置前导：仅校准，不输出净荷'
            pilots.append(dict(id=identifier, first_row=first, last_row=i,
                               correction_chips=float(frames['width_correction_chips'][i]),
                               scale=scale))
            continue
        if scheduled:
            corrected(i)
            row['training'], row['symbol'], row['bits'] = True, -3, '-'
            row['reason'] = '帧内前导受损：排除净荷，沿用先前校准'
            continue
        if not identifier:
            row['reason'] = '尚未检测到已配置前导'
            continue
        # A spacing update uses only earlier observations. It never revises
        # earlier decisions and never sees the reference cycle or future IQ.
        if auto_scale and len(samples) >= 48 and i-last_fit_index >= 32:
            last_fit_index = i
            fitted = _alphabet_scale([v for _, v in samples], alphabet)
            if fitted is not None:
                fitted_scale, fitted_intercept, fitted_centers = fitted
                # The fitted centers also describe mild nonlinearity. Check
                # the pilot against that same model, rather than rejecting it
                # against the global affine approximation used for spacing.
                pilot_center = float(np.interp(pilot_width, alphabet, fitted_centers))
                if (alphabet[0] <= pilot_width <= alphabet[-1] and
                        abs(observed_pilot-pilot_center)/fitted_scale <= match_error):
                    scale = fitted_scale
                    learned_centers = fitted_centers
                    intercept = float(np.median(pilot_history))-scale*pilot_width
                    method = '前导＋档位标定'
        high = corrected(i)
        reason = ''
        if not usable[i]:
            reason = '原始边沿/跳变不完整，不能用补偿修复'
        elif not 0 < high < total:
            reason = '补偿后脉宽超出符号周期'
        else:
            number = int(math.floor(high+.5))
            if not prefix <= number <= prefix+code:
                reason = '补偿后高片数超出编码范围'
            elif abs(high-number) > width_tolerance:
                reason = '补偿后脉宽仍偏离整数时间片'
            elif total-high < guard-width_tolerance:
                reason = '补偿后末尾低电平保护不足'
            if not reason:
                k = number-prefix
                row['valid'], row['symbol'] = True, k
                row['bits'] = '1'*k+'0'*(code-k)
        row['reason'] = reason or '有效（可配置前导补偿）'
        if usable[i]:
            samples.append((i, float(raw[i])))
            samples = samples[-512:]
    return pilots

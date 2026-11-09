"""Delayed PWM pulses with falling-edge hold time, acquired from user preambles.

The symbol clock is fitted to known preambles, never to data-dependent rises.
Payload references are only passed to the independent block scorer.
"""
import math
import numpy as np
from rx706_spectrum import _cancel
from rx706_pwm_frames import validate_frame_options


PULSE_FIELDS = [('rise_s', 'f8'), ('fall_s', 'f8'), ('delay_chips', 'f8'),
                ('delay_us', 'f8'), ('raw_pulse_us', 'f8'), ('tail_us', 'f8'),
                ('calibration_error_us', 'f8'), ('decision_estimated', '?'),
                ('decision_distance_chips', 'f8')]


def validate_pulse_options(options, symbol_hz, fs=None):
    config = dict(options)
    config.pop('tail_us', None)
    config.pop('tail_search_us', None)
    step = float(options.get('step_percent', 5))
    if not math.isfinite(step) or step <= 0:
        raise ValueError('占空比步长须为有限正数')
    total = round(100/step)
    if not 3 <= total <= 256 or abs(total*step-100) > 1e-7:
        raise ValueError('占空比步长须将100%等分成3～256片，例如5→20片、4→25片')
    minimum = options.get('min_high_chips', 6)
    count = options.get('symbol_count', 8)
    if int(minimum) != float(minimum) or not 1 <= int(minimum) < total:
        raise ValueError('最小高电平片数须为正整数且小于总片数')
    if int(count) != float(count) or not 2 <= int(count) <= 32768:
        raise ValueError('符号数量须为2～32768的整数')
    minimum, count = int(minimum), int(count)
    joint = options.get('joint_delay', True)
    if not math.isfinite(float(symbol_hz)) or symbol_hz <= 0:
        raise ValueError('PWM符号率须为有限正数')
    period = 1/float(symbol_hz)
    low = int(options.get('low_guard_chips', 1))
    if float(options.get('low_guard_chips', 1)) != low or not 1 <= low < total:
        raise ValueError('末尾低电平预留须为正整数片且小于总片数')
    usable = total-low
    supplied = options.get('codebook', '')
    if isinstance(supplied, str):
        pairs = []
        for line in supplied.replace(';', '\n').splitlines():
            if not line.strip():
                continue
            values = line.replace(',', ' ').split()
            if len(values) != 2:
                raise ValueError('码表每行填写“延迟片数 高电平片数”，按行对应符号0、1…')
            pairs.append(tuple(float(v) for v in values))
    else:
        pairs = [tuple(float(v) for v in row) for row in supplied]
    if not pairs:
        if joint:
            pairs = [(float(delay), float(width)) for delay in range(max(0, usable-minimum+1))
                     for width in range(minimum, usable-delay+1)]
            count = len(pairs)
        else:
            pairs = [(0., float(minimum+k)) for k in range(count)]
    elif joint:
        count = len(pairs)
    if not 2 <= count <= 32768:
        raise ValueError('当前合法码字须为2～32768种；请调整步长、最小宽度或周期')
    if len(pairs) != count or len(set(pairs)) != count:
        raise ValueError('码表须恰好包含配置数量的互异码字')
    for pair in pairs:
        if len(pair) != 2 or any(not math.isfinite(v) or v != round(v) for v in pair):
            raise ValueError('延迟和高电平宽度须为整数片')
        delay, width = pair
        if delay < 0 or width < minimum or delay+width > usable:
            raise ValueError('码字须满足延迟≥0、高电平≥最小片数，且延迟＋宽度＋末尾低电平预留≤总片数')
    if not joint and len(set(w for _, w in pairs)) != count:
        raise ValueError('仅脉宽映射不能含同宽码字；同宽不同时延请启用联合码表')
    tolerance = float(options.get('tolerance_chips', .4))
    if not math.isfinite(tolerance) or not 0 < tolerance < .5:
        raise ValueError('边沿判决容差须大于0且小于0.5片')
    nearest = options.get('nearest_decision', True)
    if type(nearest) is not bool:
        raise ValueError('nearest_decision须为布尔值')
    if fs is not None and fs*period/total < 3:
        raise ValueError('每片至少需要3个处理采样点；请提高采样率或降低符号率/抽取倍数')
    frames = dict(options.get('frame_options', {}), startup_symbols=0, training_symbols=0,
                  startup_pattern='', training_pattern='')
    frames = validate_frame_options(frames, tuple(w for _, w in pairs))
    for user in frames['users']:
        if len(user['identity']) < 4:
            raise ValueError('身份前导至少需要4个已知符号')
    config.update(step_percent=step, total_chips=total, min_high_chips=minimum,
                  symbol_count=count, codebook=tuple(pairs), joint_delay=bool(joint),
                  low_guard_chips=low,
                  available_combinations=max(0, usable-minimum+1)*max(0, usable-minimum+2)//2,
                  tolerance_chips=tolerance, nearest_decision=nearest, frame_options=frames)
    # Rising/falling interpolation each carries up to half a processing sample
    # of timing uncertainty. A four-symbol preamble does not average that away.
    config['preamble_sample_margin_us'] = 1e6/fs if fs is not None else 0.
    config['estimated_min_period_us'] = None  # Only known after receiving a preamble.
    config.update(information_bits=math.log2(count), integer_payload_bits=math.floor(math.log2(count)),
                  frame_cycle_symbols=sum(len(u['identity'])+u['payload_symbols'] for u in frames['users']))
    cycle = sum(len(u['identity'])+u['payload_symbols'] for u in frames['users'])
    config['user_information_rates'] = [symbol_hz*u['payload_symbols']/cycle*math.log2(count)
                                        for u in frames['users']]
    return config


def pulses_from_edges(times, after):
    """Return only complete high runs; never invent a fall for a clipped run."""
    starts = np.flatnonzero(after)
    complete = starts[starts+1 < len(times)]
    complete = complete[~after[complete+1]]
    return np.asarray(times)[complete], np.asarray(times)[complete+1]


def _nearest_codeword(delay, width, pairs, joint, tolerance):
    """Choose a real codeword by geometric distance, independent of references.

    Tolerance labels confidence; it does not suppress a usable nearest decision.
    Exact ties use codebook order and are explicitly marked as estimates.
    """
    if not math.isfinite(delay) or not math.isfinite(width):
        return None
    width_error = abs(pairs[:, 1]-width)
    delay_error = abs(pairs[:, 0]-delay)
    distance = np.hypot(width_error, delay_error) if joint else width_error
    k = int(distance.argmin())
    within = width_error[k] <= tolerance and (not joint or delay_error[k] <= tolerance)
    tied = np.count_nonzero(np.isclose(distance, distance[k], rtol=0, atol=1e-9)) > 1
    return k, within, tied, float(distance[k])


def _header(rises, falls, pulse_at, user, pairs, period, config):
    code = np.asarray(user['identity'], dtype=int)
    length = len(code)
    if pulse_at < 0 or pulse_at+length > len(rises):
        return None
    selected = pairs[code]
    relative = np.arange(length)+selected[:, 0]/config['total_chips']
    observed = rises[pulse_at:pulse_at+length]
    # Nominal period seeds phase. Robust fitting excludes at most the explicitly
    # configured number of marker errors; the known payload is never consulted.
    allowance = (config['tolerance_chips']*period/config['total_chips']+
                 config.get('preamble_sample_margin_us', 0.)*1e-6)
    trial, anchor = np.polyfit(relative, observed, 1)
    if abs(trial/period-1) > .02:
        return None
    good = abs(observed-(anchor+relative*trial)) <= allowance
    permitted = config['frame_options']['identity_errors']
    if good.sum() < max(4, length-permitted):
        return None
    fitted, anchor = np.polyfit(relative[good], observed[good], 1)
    if abs(fitted/period-1) > .02:
        return None
    rise_error = observed-(anchor+relative*fitted)
    widths = falls[pulse_at:pulse_at+length]-observed
    tails = widths-selected[:, 1]/config['total_chips']*fitted
    tail = float(np.median(tails[good]))
    if tail < -allowance:
        return None
    tail = max(0., tail)
    residual = tails-tail
    good &= (abs(rise_error) <= allowance) & (abs(residual) <= allowance)
    if good.sum() < max(4, length-permitted):
        return None
    error = float(np.sqrt(np.mean(np.r_[rise_error[good], residual[good]]**2)))
    return dict(anchor=float(anchor), period=float(fitted), tail=float(tail),
                error=error, pulse_at=int(pulse_at), errors=int(length-good.sum()))


def decode_pulses(states, transition_times, after, times, fs, symbol_hz, options, cancel_event=None):
    """Acquire user preambles anywhere in a finite file, then decode fixed slots."""
    from rx706_pwm import _frame_dtype
    config = validate_pulse_options(options, symbol_hz, fs)
    total, pairs = config['total_chips'], np.asarray(config['codebook'])
    frame_config, nominal = config['frame_options'], 1/symbol_hz
    rises, falls = pulses_from_edges(transition_times, after)
    if len(rises) < 4:
        raise ValueError('没有足够完整高电平脉冲用于身份前导同步')
    found = None
    search = frame_config['search_symbols']
    search_end = times[0]+search*nominal if search else times[-1]
    for at in range(len(rises)):
        if at % 128 == 0:
            _cancel(cancel_event)
        if rises[at] > search_end:
            break
        candidates = []
        for user_at, user in enumerate(frame_config['users']):
            match = _header(rises, falls, at, user, pairs, nominal, config)
            if match is not None:
                candidates.append((match['errors'], match['error'], user_at, match))
        if candidates:
            candidates.sort(key=lambda item: item[:2])
            if len(candidates) == 1 or candidates[0][0] < candidates[1][0]:
                found = candidates[0][2], candidates[0][3]
                break
    if found is None:
        raise ValueError('未匹配用户身份前导；请核对身份码、时延码表、符号率及拖尾值')
    first_user, initial = found
    schedule, symbol_at, user_at = [], 0, first_user
    # A four-symbol header can fit its own edge jitter with an inaccurate
    # slope. Use the configured clock to reach the next known header; only a
    # long baseline between headers may change the payload/schedule period.
    anchor, period = initial['anchor'], nominal
    prior = None
    while anchor <= times[-1]:
        _cancel(cancel_event)
        user = frame_config['users'][user_at]
        length = len(user['identity'])+user['payload_symbols']
        match = initial if not schedule else None
        if schedule:
            begin = np.searchsorted(rises, anchor-2*period)
            end = np.searchsorted(rises, anchor+(2+max(pairs[:, 0])/total)*period)
            candidates = []
            for at in range(begin, min(end, len(rises))):
                value = _header(rises, falls, at, user, pairs, nominal, config)
                if value is not None and abs(value['anchor']-anchor) < 2*period:
                    candidates.append(value)
            if candidates:
                match = min(candidates, key=lambda value: (value['errors'], value['error'], abs(value['anchor']-anchor)))
        if match is not None:
            anchor = match['anchor']
            if prior is not None:
                refined = (anchor-prior[1])/(symbol_at-prior[0])
                if abs(refined/nominal-1) <= .02:
                    period = refined
            prior = symbol_at, anchor
        schedule.append(dict(symbol_at=symbol_at, anchor=anchor, user_at=user_at,
                             length=length, match=match))
        symbol_at += length
        anchor += period*length
        user_at = (user_at+1) % frame_config['user_count']
    # Long-baseline timing from known markers avoids clock drift accumulating
    # over a 1024-symbol payload when preamble edge estimates have sample jitter.
    known = [item for item in schedule if item['match'] is not None]
    if len(known) >= 2:
        period, _ = np.polyfit([item['symbol_at'] for item in known],
                               [item['anchor'] for item in known], 1)
        config['clock_source'] = '多个身份前导的长间隔拟合'
    else:
        period = nominal
        config['clock_source'] = '配置符号率（仅一个完整身份前导）'
    count = max(0, int(math.floor((times[-1]+1/fs-initial['anchor'])/period+1e-6)))
    if count < len(frame_config['users'][first_user]['identity']):
        raise ValueError('完整身份前导后没有足够符号')
    bits = max(1, math.ceil(math.log2(len(pairs))))
    rows = np.zeros(count, dtype=_frame_dtype(bits, total, True)+PULSE_FIELDS)
    rows['index'] = np.arange(count)+1
    rows['start_s'] = initial['anchor']+np.arange(count)*period
    rows['payload_index'] = -1
    rows['symbol'], rows['candidate_symbol'] = -1, -1
    rows['bits'], rows['candidate_bits'] = '?'*bits, '?'*bits
    rows['width_scale'] = 1.
    for key in ('rise_s', 'fall_s', 'delay_chips', 'delay_us', 'raw_pulse_us', 'tail_us',
                'calibration_error_us', 'decision_distance_chips', 'one_us', 'zero_us', 'high_chips', 'duty_percent',
                'clock_high_chips', 'width_correction_chips', 'corrected_high_chips',
                'corrected_one_us', 'corrected_zero_us'):
        rows[key] = np.nan
    integral = np.r_[0., np.cumsum(states, dtype=float)]/fs
    edges = np.r_[times, times[-1]+1/fs]
    calibrations, last_by_user = [], {}
    for block_at, item in enumerate(schedule):
        begin = item['symbol_at']
        end = min(count, begin+item['length'])
        if begin >= end:
            continue
        user_at = item['user_at']
        user = frame_config['users'][user_at]
        header_end = min(end, begin+len(user['identity']))
        match = item['match']
        if match is not None:
            selected = pairs[np.asarray(user['identity'], int)]
            at = match['pulse_at']
            tail = max(0., float(np.median(falls[at:at+len(selected)]-rises[at:at+len(selected)]-
                                          selected[:, 1]/total*period)))
            match['tail'] = tail
            last_by_user[user_at] = match
            calibrations.append(dict(block_id=block_at+1, user_id=user_at+1,
                start_s=float(item['anchor']), tail_us=tail*1e6,
                residual_us=match['error']*1e6, identity_errors=match['errors']))
        calibration = match or last_by_user.get(user_at)
        tail = calibration['tail'] if calibration else np.nan
        valid_header = match is not None
        # Use local marker phase with the globally measured period. Bad headers
        # keep their scheduled slots and are erased, including their payload.
        start = item['anchor']
        rows['start_s'][begin:end] = start+np.arange(end-begin)*period
        rows['user_id'][begin:end], rows['block_id'][begin:end] = user_at+1, block_at+1
        rows['identity_ok'][begin:end] = valid_header
        rows['block_complete'][begin:end] = begin+item['length'] <= count
        rows['frame_part'][begin:header_end], rows['training'][begin:header_end] = '用户身份', True
        rows['frame_part'][header_end:end] = '净荷'
        rows['payload_index'][header_end:end] = np.arange(end-header_end)
        rows['tail_us'][begin:end] = tail*1e6
        rows['calibration_id'][begin:end] = block_at+1 if valid_header else 0
        rows['calibration_source'][begin:end] = '身份前导：下降沿拖尾校正' if valid_header else '身份失败：保持时隙'
        rows['calibration_error_us'][begin:end] = calibration['error']*1e6 if calibration else np.nan
        for i in range(begin, end):
            if i % 1024 == 0:
                _cancel(cancel_event)
            row = rows[i]
            slot = row['start_s']
            allowance = config['tolerance_chips']*period/total
            left = np.searchsorted(rises, slot-allowance)
            right = np.searchsorted(rises, slot+period-allowance)
            pattern_edges = slot+np.arange(total+1)*period/total
            fractions = np.diff(np.interp(pattern_edges, edges, integral))/(period/total)
            row['pattern'] = ''.join('1' if value >= .5 else '0' for value in fractions)
            row['reason'] = '缺少唯一完整脉冲（可能拖尾跨周期或脉冲粘连）'
            if right-left == 1:
                rise, fall = rises[left], falls[left]
                delay, raw_width = (rise-slot)/period*total, (fall-rise)/period*total
                width = raw_width-tail/period*total
                row['rise_s'], row['fall_s'] = rise, fall
                row['delay_chips'], row['delay_us'] = delay, (rise-slot)*1e6
                row['raw_pulse_us'] = (fall-rise)*1e6
                row['one_us'], row['zero_us'] = (fall-rise)*1e6, (period-fall+rise)*1e6
                row['high_chips'], row['clock_high_chips'] = raw_width, raw_width
                row['duty_percent'] = raw_width/total*100
                row['corrected_high_chips'], row['width_correction_chips'] = width, width-raw_width
                row['corrected_one_us'], row['corrected_zero_us'] = width/total*period*1e6, (1-width/total)*period*1e6
                decision = _nearest_codeword(delay, width, pairs, config['joint_delay'], config['tolerance_chips'])
                if decision is not None:
                    k, within, tied, distance = decision
                    row['candidate_symbol'], row['candidate_bits'] = k, format(k, f'0{bits}b')
                    row['decision_distance_chips'] = distance
                    row['candidate_source'] = '最近脉宽＋时延码字' if config['joint_delay'] else '最近补偿脉宽码字；时延单列'
                physical = (np.isfinite(tail) and delay >= -config['tolerance_chips'] and
                            width >= config['min_high_chips']-config['tolerance_chips'] and
                            delay+width <= total+config['tolerance_chips'] and width > 0)
                row['structure_ok'] = physical
                good = (physical and decision is not None and
                        (config['nearest_decision'] or (within and not tied)))
                row['raw_reason'] = '已测得高脉冲；尚未扣除下降沿拖尾'
                if not physical:
                    row['reason'] = '脉冲过短或拖尾超过符号周期，无法确认'
                elif not good:
                    row['reason'] = '补偿后脉宽/时延偏离码表或码字不唯一'
                elif not valid_header:
                    row['reason'] = '身份前导失败：净荷按擦除保位'
                else:
                    row['valid'], row['symbol'], row['bits'] = True, k, format(k, f'0{bits}b')
                    row['decision_estimated'] = not within or tied
                    row['reason'] = ('有效（最近码字估计；等距，按码表顺序）' if tied else
                                     '有效（最近码字估计；超容差）' if not within else
                                     '有效（身份前导拖尾校正）')
            if i < header_end:
                row['valid'], row['symbol'], row['bits'] = False, -3, '-'
                row['decision_estimated'] = False
                row['reason'] = '身份前导：仅定位与校准，不输出净荷' if valid_header else '身份前导失败：保持块位置'
    for calibration in calibrations:
        tail_seconds = calibration['tail_us']*1e-6
        guard = int(math.ceil(tail_seconds/period*total-1e-9))
        calibration['measured_tail_chips'] = tail_seconds/period*total
        calibration['conservative_codewords'] = int(np.count_nonzero(pairs.sum(axis=1)+guard+config['low_guard_chips'] <= total))
        calibration['high_stability_period_us'] = tail_seconds*total/config['min_high_chips']*1e6
        calibration['no_spill_period_us'] = (tail_seconds+3/fs)/(1-float(pairs.sum(axis=1).max())/total)*1e6
    config['estimated_min_period_us'] = max(c['high_stability_period_us'] for c in calibrations)
    return rows, float(1/period), calibrations, config

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
    quantization = options.get('width_quantization', 'nearest')
    if quantization not in ('nearest', 'floor_width'):
        raise ValueError('脉宽判决须为nearest或floor_width')
    if fs is not None and fs*period/total < 3:
        raise ValueError('每片至少需要3个处理采样点；请提高采样率或降低符号率/抽取倍数')
    frames = dict(options.get('frame_options', {}), startup_symbols=0, training_symbols=0,
                  startup_pattern='', training_pattern='')
    frames = validate_frame_options(frames, tuple(w for _, w in pairs))
    for user in frames['users']:
        if len(user['identity']) < 4:
            raise ValueError('身份前导至少需要4个已知符号')
    cycle_cal = dict(options.get('cycle_calibration') or {})
    if cycle_cal:
        def integer(value, low, high):
            if int(value) != float(value) or not low <= int(value) <= high:
                raise ValueError('循环校准用户、位置及平均块数须为合法整数')
            return int(value)
        uid = integer(cycle_cal.get('user_id', 1), 1, frames['user_count'])
        positions, symbols = cycle_cal.get('payload_indices', []), cycle_cal.get('symbols', [])
        if len(positions) != 4 or len(symbols) != 4:
            raise ValueError('循环已知码校准须明确指定4个位置及各自发送符号')
        positions = [integer(v, 0, frames['users'][uid-1]['payload_symbols']-1) for v in positions]
        symbols = [integer(v, 0, count-1) for v in symbols]
        if len(set(positions)) != 4 or len(set(pairs[k][1] for k in symbols)) < 2:
            raise ValueError('循环校准位置须互异且覆盖至少两种脉宽')
        cycle_cal.update(user_id=uid, payload_indices=positions, symbols=symbols,
                         average_blocks=integer(cycle_cal.get('average_blocks', 64), 1, 4096))
    config.update(step_percent=step, total_chips=total, min_high_chips=minimum,
                  symbol_count=count, codebook=tuple(pairs), joint_delay=bool(joint),
                  low_guard_chips=low,
                  available_combinations=max(0, usable-minimum+1)*max(0, usable-minimum+2)//2,
                  tolerance_chips=tolerance, nearest_decision=nearest, frame_options=frames)
    config.update(width_quantization=quantization, cycle_calibration=cycle_cal)
    # Rising/falling interpolation each carries up to half a processing sample
    # of timing uncertainty. A four-symbol preamble does not average that away.
    config['preamble_sample_margin_us'] = 1e6/fs if fs is not None else 0.
    config['estimated_min_period_us'] = None  # Only known after receiving a preamble.
    config.update(information_bits=math.log2(count), integer_payload_bits=math.floor(math.log2(count)),
                  frame_cycle_symbols=sum(len(u['identity'])+u['payload_symbols'] for u in frames['users']))
    cycle = config['frame_cycle_symbols']
    config['coded_information_rate'] = symbol_hz*math.log2(count)
    config['user_information_rates'] = [symbol_hz*(len(u['identity'])+u['payload_symbols'])/cycle*math.log2(count)
                                        for u in frames['users']]
    return config


def pulses_from_edges(times, after):
    """Return only complete high runs; never invent a fall for a clipped run."""
    starts = np.flatnonzero(after)
    complete = starts[starts+1 < len(times)]
    complete = complete[~after[complete+1]]
    return np.asarray(times)[complete], np.asarray(times)[complete+1]


def _nearest_codeword(delay, width, pairs, joint, tolerance, quantization='nearest'):
    """Choose a real codeword by geometric distance, independent of references.

    Tolerance labels confidence; it does not suppress a usable nearest decision.
    Exact ties use codebook order and are explicitly marked as estimates.
    """
    if not math.isfinite(delay) or not math.isfinite(width):
        return None
    width_error = abs(pairs[:, 1]-width)
    delay_error = abs(pairs[:, 0]-delay)
    used_error = abs(pairs[:, 1]-math.floor(width)) if quantization == 'floor_width' else width_error
    distance = np.hypot(used_error, delay_error) if joint else used_error
    k = int(distance.argmin())
    within = width_error[k] <= tolerance and (not joint or delay_error[k] <= tolerance)
    tied = np.count_nonzero(np.isclose(distance, distance[k], rtol=0, atol=1e-9)) > 1
    return k, within, tied, float(distance[k])


def _width_model(expected, observed, resolution=0.):
    """Fit only known preamble widths; a single level identifies offset only."""
    levels, counts = np.unique(expected, return_counts=True)
    identifiable = bool(len(levels) >= 2 and len(expected) >= 4)
    gain = 1.
    if identifiable:
        # Pairwise slopes between distinct known widths are robust to one
        # noisy edge, without using payloads or an assumed device response.
        dx = expected[:, None]-expected[None, :]
        dy = observed[:, None]-observed[None, :]
        mask = abs(dx) > np.finfo(float).eps
        gain = float(np.median(dy[mask]/dx[mask]))
        if not math.isfinite(gain) or gain <= 0:
            return None
        if abs(gain-1)*np.ptp(expected) <= 2*resolution:
            gain = 1.
    offset = float(np.median(observed-gain*expected))
    return gain, offset, identifiable


def _phase(rises, at, selected, period, total):
    relative = np.arange(len(selected))+selected[:, 0]/total
    return float(np.median(rises[at:at+len(selected)]-relative*period))


def _cycle_models(schedule, config, pairs, rises, falls, period, fs):
    """Offline median calibration from FOUR explicitly reserved known slots.

    The rest of the payload reference is never consulted. Original headers,
    frame length and all physical slot positions remain unchanged.
    """
    spec = config['cycle_calibration']
    if not spec:
        return {}
    total = config['total_chips']
    user_at = spec['user_id']-1
    user = config['frame_options']['users'][user_at]
    selected = pairs[np.asarray(spec['symbols'], int)]
    observations, blocks = [], []
    allowance = config['tolerance_chips']*period/total+1/fs
    for block, item in enumerate(schedule):
        if item['user_at'] != user_at:
            continue
        observed = np.full(4, np.nan)
        if item['match'] is not None:
            for j, position in enumerate(spec['payload_indices']):
                target = item['anchor']+(len(user['identity'])+position+selected[j, 0]/total)*period
                left, right = np.searchsorted(rises, [target-allowance, target+allowance])
                if right-left == 1:
                    observed[j] = falls[left]-rises[left]
        blocks.append(block)
        observations.append(observed)
    observations = np.asarray(observations)
    models = {}
    window = spec['average_blocks']
    for i, block in enumerate(blocks):
        begin = max(0, min(i-window//2, len(blocks)-window))
        sample = observations[begin:min(len(blocks), begin+window)]
        centers = np.array([np.median(column[np.isfinite(column)]) if np.any(np.isfinite(column)) else np.nan
                            for column in sample.T])
        if not np.all(np.isfinite(centers)):
            continue
        model = _width_model(selected[:, 1]*period/total, centers, 0.)
        if model is not None:
            models[block] = dict(gain=model[0], offset=model[1], blocks=len(sample))
    return models


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
    required = max(3, length-permitted)
    if good.sum() < required:
        return None
    fitted, anchor = np.polyfit(relative[good], observed[good], 1)
    if abs(fitted/period-1) > .02:
        return None
    rise_error = observed-(anchor+relative*fitted)
    widths = falls[pulse_at:pulse_at+length]-observed
    expected = selected[:, 1]/config['total_chips']*fitted
    model = _width_model(expected[good], widths[good], config['preamble_sample_margin_us']*1e-6)
    if model is None:
        return None
    gain, offset, identifiable = model
    if not identifiable and offset < -allowance:
        return None
    if not identifiable:
        offset = max(0., offset)
    residual = widths-(gain*expected+offset)
    good &= (abs(rise_error) <= allowance) & (abs(residual) <= allowance)
    if good.sum() < required:
        return None
    error = float(np.sqrt(np.mean(np.r_[rise_error[good], residual[good]]**2)))
    return dict(anchor=_phase(rises, pulse_at, selected, period, config['total_chips']),
                period=float(fitted), tail=max(0., offset), width_gain=gain,
                width_offset=offset, width_gain_identified=identifiable,
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
    period = initial['period'] if len(frame_config['users'][first_user]['identity']) >= 8 else nominal
    first_selected = pairs[np.asarray(frame_config['users'][first_user]['identity'], int)]
    anchor = initial['anchor'] = _phase(rises, initial['pulse_at'], first_selected, period, total)
    clock_markers = []
    competing_headers = 0
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
                value = _header(rises, falls, at, user, pairs, period, config)
                if value is not None and abs(value['anchor']-anchor) < .45*period:
                    candidates.append(value)
            if candidates:
                competing_headers += len(candidates) > 1
                match = min(candidates, key=lambda value: (abs(value['anchor']-anchor), value['errors'], value['error']))
        if match is not None:
            anchor = match['anchor']
            clock_markers.append((symbol_at, anchor))
            if len(clock_markers) >= 2:
                recent = np.asarray(clock_markers[-64:])
                refined, _ = np.polyfit(recent[:, 0], recent[:, 1], 1)
                if abs(refined/nominal-1) <= .02:
                    period = refined
        schedule.append(dict(symbol_at=symbol_at, anchor=anchor, user_at=user_at,
                             length=length, match=match))
        symbol_at += length
        anchor += period*length
        user_at = (user_at+1) % frame_config['user_count']
    # Long-baseline timing from known markers avoids clock drift accumulating
    # over a 1024-symbol payload when preamble edge estimates have sample jitter.
    known = [item for item in schedule if item['match'] is not None]
    if len(known) >= 2:
        local = np.asarray([item['match']['period'] for item in known])
        stable = [item for item, value in zip(known, local) if abs(value/np.median(local)-1) <= .01]
        clock_known = stable if len(stable) >= 2 else known
        period, _ = np.polyfit([item['symbol_at'] for item in clock_known],
                               [item['anchor'] for item in clock_known], 1)
        config['clock_source'] = '多个身份前导的长间隔拟合'
    else:
        period = nominal
        config['clock_source'] = '配置符号率（仅一个完整身份前导）'
    # Refit every marker's intercept at the FINAL period. Never combine the
    # intercept of a noisy short-period fit with a different symbol period.
    for item in known:
        user = frame_config['users'][item['user_at']]
        selected = pairs[np.asarray(user['identity'], int)]
        item['anchor'] = item['match']['anchor'] = _phase(
            rises, item['match']['pulse_at'], selected, period, total)
    # Failed headers retain a clock-predicted slot, not an accumulated sequence
    # of stale local slopes. Their payload remains erased.
    if len(known) >= 2:
        _, origin = np.polyfit([item['symbol_at'] for item in known],
                               [item['anchor'] for item in known], 1)
        for item in schedule:
            if item['match'] is None:
                item['anchor'] = float(origin+item['symbol_at']*period)
        # A phase estimate depends slightly on the period used to compute it.
        # Iterate the known-marker clock and phase together to consistency.
        for _ in range(3):
            period, _ = np.polyfit([item['symbol_at'] for item in clock_known],
                                   [item['anchor'] for item in clock_known], 1)
            for item in known:
                selected = pairs[np.asarray(frame_config['users'][item['user_at']]['identity'], int)]
                item['anchor'] = item['match']['anchor'] = _phase(
                    rises, item['match']['pulse_at'], selected, period, total)
        _, origin = np.polyfit([item['symbol_at'] for item in clock_known],
                               [item['anchor'] for item in clock_known], 1)
        for item in schedule:
            if item['match'] is None:
                item['anchor'] = float(origin+item['symbol_at']*period)
    repeated = any(len(set(u['identity'])) == 1 for u in frame_config['users'])
    config['timing_note'] = ('前导重复：按时钟连续性跟踪；首次帧位置仍需唯一前导确认。' if repeated else '')
    config['competing_headers'] = competing_headers
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
    cycle_models = _cycle_models(schedule, config, pairs, rises, falls, period, fs)
    config['cycle_calibrated_blocks'] = len(cycle_models)
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
            model = _width_model(selected[:, 1]/total*period,
                                 falls[at:at+len(selected)]-rises[at:at+len(selected)], 1/fs)
            gain, offset, identifiable = model
            if block_at in cycle_models:
                model = cycle_models[block_at]
                gain, offset, identifiable = model['gain'], model['offset'], True
            if not identifiable:
                offset = max(0., offset)
            tail = max(0., offset)
            match['tail'] = tail
            match.update(width_gain=gain, width_offset=offset, width_gain_identified=identifiable)
            last_by_user[user_at] = match
            calibrations.append(dict(block_id=block_at+1, user_id=user_at+1,
                start_s=float(item['anchor']), tail_us=tail*1e6,
                residual_us=match['error']*1e6, identity_errors=match['errors'],
                width_gain=gain, width_offset_us=offset*1e6, width_gain_identified=identifiable))
        calibration = match or last_by_user.get(user_at)
        tail = calibration['tail'] if calibration else np.nan
        gain = calibration['width_gain'] if calibration else 1.
        offset = calibration['width_offset'] if calibration else np.nan
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
        pilot_rows = set()
        spec = config['cycle_calibration']
        if spec and user_at+1 == spec['user_id']:
            pilot_rows = {begin+len(user['identity'])+position for position in spec['payload_indices']}
            for position in pilot_rows:
                if position < end:
                    rows['training'][position] = True
                    rows['frame_part'][position] = '校准前导'
        rows['tail_us'][begin:end] = tail*1e6
        rows['width_scale'][begin:end] = gain
        rows['calibration_id'][begin:end] = block_at+1 if valid_header else 0
        rows['calibration_source'][begin:end] = '身份前导：下降沿拖尾校正' if valid_header else '身份失败：保持时隙'
        if valid_header and block_at in cycle_models:
            rows['calibration_source'][begin:end] = '循环已知码4点：跨循环中位数标定'
        rows['calibration_error_us'][begin:end] = calibration['error']*1e6 if calibration else np.nan
        for i in range(begin, end):
            if i % 1024 == 0:
                _cancel(cancel_event)
            row = rows[i]
            slot = row['start_s']
            allowance = config['tolerance_chips']*period/total
            left = np.searchsorted(rises, slot-allowance)
            right = np.searchsorted(rises, slot+period-allowance)
            # A jittered marker edge may lie just inside the preceding data
            # slot. Its known assignment must not erase that data pulse.
            next_match = schedule[block_at+1]['match'] if block_at+1 < len(schedule) else None
            if next_match is not None:
                right = min(right, next_match['pulse_at'])
            if valid_header:
                if i < header_end:
                    left = match['pulse_at']+i-begin
                    right = left+1
                else:
                    left = max(left, match['pulse_at']+len(user['identity']))
            pattern_edges = slot+np.arange(total+1)*period/total
            fractions = np.diff(np.interp(pattern_edges, edges, integral))/(period/total)
            row['pattern'] = ''.join('1' if value >= .5 else '0' for value in fractions)
            row['reason'] = '缺少唯一完整脉冲（可能拖尾跨周期或脉冲粘连）'
            if right-left == 1:
                rise, fall = rises[left], falls[left]
                delay, raw_width = (rise-slot)/period*total, (fall-rise)/period*total
                width = (raw_width-offset/period*total)/gain
                row['rise_s'], row['fall_s'] = rise, fall
                row['delay_chips'], row['delay_us'] = delay, (rise-slot)*1e6
                row['raw_pulse_us'] = (fall-rise)*1e6
                row['one_us'], row['zero_us'] = (fall-rise)*1e6, (period-fall+rise)*1e6
                row['high_chips'], row['clock_high_chips'] = raw_width, raw_width
                row['duty_percent'] = raw_width/total*100
                row['corrected_high_chips'], row['width_correction_chips'] = width, width-raw_width
                row['corrected_one_us'], row['corrected_zero_us'] = width/total*period*1e6, (1-width/total)*period*1e6
                decision = _nearest_codeword(delay, width, pairs, config['joint_delay'], config['tolerance_chips'],
                                             config['width_quantization'])
                if decision is not None:
                    k, within, tied, distance = decision
                    row['candidate_symbol'], row['candidate_bits'] = k, format(k, f'0{bits}b')
                    row['decision_distance_chips'] = distance
                    row['candidate_source'] = '最近脉宽＋时延码字' if config['joint_delay'] else '最近补偿脉宽码字；时延单列'
                    if config['width_quantization'] == 'floor_width':
                        row['candidate_source'] = '补偿脉宽向下取整后匹配合法码字；时延不取整'
                # Allow one chip of downward measurement error at the lowest
                # legal width: with minimum 5, measured 4..5 maps to width 5.
                # Confidence remains separate; strict mode retains its own
                # tolerance. Never manufacture a missing/merged pulse.
                physical = (np.isfinite(tail) and delay >= -config['tolerance_chips'] and
                            width+1e-9 >= config['min_high_chips']-1. and
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
                                     '有效（循环已知码增益/偏置校正）' if block_at in cycle_models else
                                     '有效（身份前导拖尾校正）')
            if i < header_end or i in pilot_rows:
                row['valid'], row['symbol'], row['bits'] = False, -3, '-'
                row['decision_estimated'] = False
                row['reason'] = '身份前导：仅定位与校准，不输出净荷' if valid_header else '身份前导失败：保持块位置'
                if i in pilot_rows:
                    row['reason'] = '循环已知码校准前导：不参与净荷BER/BLER'
    for calibration in calibrations:
        tail_seconds = calibration['tail_us']*1e-6
        guard = int(math.ceil(tail_seconds/period*total-1e-9))
        calibration['measured_tail_chips'] = tail_seconds/period*total
        calibration['conservative_codewords'] = int(np.count_nonzero(pairs.sum(axis=1)+guard+config['low_guard_chips'] <= total))
        calibration['high_stability_period_us'] = tail_seconds*total/config['min_high_chips']*1e6
        calibration['no_spill_period_us'] = (tail_seconds+3/fs)/(1-float(pairs.sum(axis=1).max())/total)*1e6
    config['estimated_min_period_us'] = max(c['high_stability_period_us'] for c in calibrations)
    config['calibration_note'] = ('前导含多种重复脉宽：估计脉宽增益与偏置。' if
        all(c['width_gain_identified'] for c in calibrations) else
        '部分或全部前导仅一个可靠脉宽标定点：只能估计宽度偏置，无法识别随脉宽变化的误差；建议更换混合前导。')
    if config['cycle_calibration']:
        config['calibration_note'] = ('循环已知码校准实验：指定4个原净荷位置为校准前导，跨循环取中位数估计增益与偏置；'
                                    '这些位置不参与净荷BER/BLER，不代表独立身份识别测试。')
    return rows, float(1/period), calibrations, config

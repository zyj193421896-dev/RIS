"""Offline PWM demodulation from two complex IQ states, with bounded reads.

Default protocol: 8 data slots + 2 low guard slots, high widths 1..8.
Zero duty is idle. Display subsampling never participates in decoding.
"""
from dataclasses import dataclass
from copy import copy
import math
import numpy as np
from rx706_spectrum import _cancel
from rx706_recovery import _read_span, _lowpass


FRAME_DTYPE = [('index', 'i4'), ('start_s', 'f8'), ('one_us', 'f8'),
               ('zero_us', 'f8'), ('high_chips', 'f8'), ('duty_percent', 'f8'),
               ('symbol', 'i2'), ('bits', 'U16'), ('pattern', 'U40'),
               ('valid', '?'), ('reason', 'U64'), ('glitch_count', 'i4'),
               ('glitch_us', 'f8'), ('glitch_max_us', 'f8'), ('warning', 'U96'),
               ('candidate_symbol', 'i2'), ('candidate_bits', 'U16'), ('candidate_source', 'U64')]


def pwm2_layout(total_chips=20, prefix_chips=8, guard_chips=2):
    """Validate explicit protocol timing, independently of received pulse widths."""
    if any(type(n) is not int for n in (total_chips, prefix_chips, guard_chips)):
        raise ValueError('version2总片数、前高和后低片数必须为整数')
    if not 3 <= total_chips <= 256 or prefix_chips < 1 or guard_chips < 1:
        raise ValueError('version2总片数须为3～256；前高和后低保护至少各1片')
    code_chips = total_chips-prefix_chips-guard_chips
    if code_chips < 1:
        raise ValueError('前高+后低必须小于总片数，至少保留1片可调区')
    return code_chips


def _frame_dtype(bit_width, total_chips, training=False, pattern=False):
    # Size strings to the actual protocol; never truncate 25/40-slot words.
    fields = [(name, f'U{bit_width}' if name in ('bits', 'candidate_bits') else
             f'U{total_chips}' if name == 'pattern' else dtype)
            for name, dtype in FRAME_DTYPE]
    if training:
        from rx706_pwm_training import TRAINING_FIELDS
        from rx706_pwm_frames import FRAME_FIELDS
        fields += TRAINING_FIELDS+FRAME_FIELDS
    elif pattern:
        fields += [('raw_reason', 'U64'), ('structure_ok', '?')]
    return fields


@dataclass
class PWMRecovery:
    carrier_hz: float
    output_rate_hz: float
    cutoff_hz: float
    decimation: int
    first_tuple: int
    input_samples: int
    clipped: int
    centers: object
    phase_difference_deg: float
    amplitude_difference_db: float
    separation: float
    scatter: object
    scatter_states: object
    waveform: object
    waveform_time_s: object
    power_time_s: object
    power_dbfs: object
    level_time_s: object
    levels: object
    decision_metric: object
    frames: object
    symbol_rate_hz: float
    measured_symbol_rate_hz: float
    code_chips: int
    guard_chips: int
    bits_per_symbol: int
    state_one: int
    mapping_note: str
    bitstream: str
    decoded_bit_rate: float
    glitch_tolerance_chips: float = .15
    reference_check: object = None
    sampling_note: str = ''
    calibration_start_s: float = 0.
    decode_locked: bool = True
    decode_note: str = ''
    state_fit_note: str = ''
    search_note: str = ''
    version: int = 1
    prefix_chips: int = 0
    training_calibration: object = None
    decision_mode: str = 'width'
    symbol_widths: object = None
    output_binary: bool = False
    frame_check: object = None
    frame_options: object = None
    pulse_options: object = None
    pulse_calibrations: object = None

    @property
    def kind(self):
        return 'pwm2' if self.version == 2 else 'pwm'

    @property
    def total_chips(self):
        return self.prefix_chips+self.code_chips+self.guard_chips

    @property
    def protocol_label(self):
        if self.output_binary:
            return f'PWM占空比（{len(self.symbol_widths)}态）'
        return (f'PWM version2 ({self.prefix_chips}+{self.code_chips}+{self.guard_chips})'
                if self.version == 2 else f'PWM ({self.code_chips}+{self.guard_chips})')

    @property
    def symbol_bitwords(self):
        if self.output_binary:
            return tuple(format(k, f'0{self.bits_per_symbol}b') for k in range(len(self.symbol_widths)))
        if self.symbol_widths is not None:
            return tuple('1'*(width-self.prefix_chips)+'0'*(self.total_chips-self.guard_chips-width)
                         for width in self.symbol_widths)
        return tuple('1'*k+'0'*(self.code_chips-k) for k in range(self.code_chips+1)) if self.version == 2 else None

    @property
    def calibration_description(self):
        """Describe the settings actually used by this result, not live GUI edits."""
        if self.pulse_options is not None:
            return (f"用户身份前导＋净荷；{self.frame_options['user_count']}用户；"
                    f"步长{self.pulse_options['step_percent']:g}%→{self.total_chips}片；"
                    f"最小高电平{self.pulse_options['min_high_chips']}片。"
                    '每个身份前导独立估计下降沿拖尾和符号边界，无公共训练。')
        if self.frame_options is not None:
            config = self.frame_options
            return (f"最简帧：启动缓冲{config['startup_symbols']}，公共训练{config['training_symbols']}；"
                    f"{config['user_count']}用户，按配置顺序轮询。公共训练直接标定档位，身份码兼作块定位。")
        cal = self.training_calibration
        if cal is None:
            return '本次前导校准未启用。'
        widths = cal.get('preamble_widths')
        if widths is None:
            return '本次使用旧接口20%三连训练。'
        from rx706_pwm_protocol import parse_preamble_widths
        duties = [100*w/self.total_chips for w in parse_preamble_widths(widths, self.total_chips)]
        preamble = '、'.join(f'{d:g}%' for d in duties[:16])+('…' if len(duties)>16 else '')
        count = cal.get('payload_symbols', 0)
        framing = f'每帧净荷 {count}' if count else '未固定帧长（连续同宽前导与净荷可能有歧义）'
        source = '拟合时钟测宽' if cal.get('clock_width', False) else '边沿测宽'
        if np.ptp(duties) > 1e-9:
            spacing = '多宽前导校准比例与偏移'
        elif cal.get('auto_scale', False):
            spacing = f"自动档位标定已启用；{cal['spacing_symbols']}行已标定，初期仅前导偏移"
        else:
            spacing = '自动档位标定已关闭；同宽前导只校准偏移'
        uncertainty = cal.get('measurement_uncertainty_chips', 0.)
        return (f'本次前导 {preamble}；{framing}；{source}；{spacing}。'
                f'前导匹配另计采样测宽余量 {uncertainty:.3f}片（原始宽度）。')

    def duty_thermometer(self, row):
        """Map a confirmed duty level to unary display, separate from bits.

        This is a level mapping, not a measurement of physical time slots.
        Candidates never turn erasures into confirmed display words.
        """
        if not self.output_binary or self.symbol_widths is None:
            return ''
        if row['symbol'] in (-2, -3):
            return '-'
        count = len(self.symbol_widths)
        if not row['valid'] or not 0 <= row['symbol'] < count:
            return '?'*count
        high = int(row['symbol'])+1
        return '1'*high+'0'*(count-high)

    def width_symbol(self, high):
        if self.symbol_widths is None:
            return high-self.prefix_chips
        return self.symbol_widths.index(high) if high in self.symbol_widths else -1


def _two_states(z, cancel_event=None):
    """Learn a separating line; no time/clock labels are imposed on samples."""
    a = z[::max(1, len(z)//30000)]
    xy = np.column_stack((a.real, a.imag))
    _, vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
    axis = vectors[:, -1]
    if axis[np.argmax(abs(axis))] < 0:
        axis = -axis
    projection = xy @ axis
    lo, hi = np.percentile(projection, [15, 85])
    centers = np.array([np.mean(a[projection <= lo]), np.mean(a[projection >= hi])])
    for _ in range(20):
        _cancel(cancel_event)
        labels = abs(a-centers[1]) < abs(a-centers[0])
        if min(np.mean(labels), np.mean(~labels)) < .005:
            raise ValueError('只检测到一个稳定状态，不能解调 PWM')
        updated = np.array([np.mean(a[~labels]), np.mean(a[labels])])
        if np.max(abs(updated-centers)) < 1e-7:
            break
        centers = updated
    labels = abs(a-centers[1]) < abs(a-centers[0])
    spread = math.sqrt(float(np.mean(abs(a-centers[labels.astype(int)])**2)))
    separation = float(abs(centers[1]-centers[0])/max(spread, 1e-12))
    return centers, separation


def _carrier_seed(x, fs, seed_size, hint, chip_rate, symbol_hz, cancel_event):
    """Try raw/squared spectral lines; choose measured two-state separation.

    Squaring supplies the carrier for nearly antipodal/equal-energy states even
    when the ordinary carrier is suppressed. Both square-root aliases are tried.
    Raw lines cover non-antipodal states. No periodic sideband spacing is assumed.
    """
    nf = min(262144, 1 << (len(x).bit_length()-1))
    frequency = np.fft.fftfreq(nf, 1/fs)
    window = np.hanning(nf)
    candidates = []
    for order in (1, 2):
        _cancel(cancel_event)
        spectrum = abs(np.fft.fft(x[:nf]**order*window))**2
        peaks = np.flatnonzero((spectrum > np.roll(spectrum, 1)) & (spectrum >= np.roll(spectrum, -1)))
        peaks = peaks[(peaks > 0) & (peaks < nf-1)]
        for at in peaks[np.argsort(spectrum[peaks])[-10:]]:
            log = np.log(np.maximum(spectrum[at-1:at+2], 1e-30))
            denominator = log[0]-2*log[1]+log[2]
            correction = .5*(log[0]-log[2])/denominator if abs(denominator) > 1e-12 else 0.
            f = (frequency[at]+np.clip(correction, -.5, .5)*fs/nf)/order
            candidates.append(f)
            if order == 2:
                candidates.append(f+(-fs/2 if f >= 0 else fs/2))
    correlation = np.vdot(x[:-1], x[1:])
    candidates.append(float(np.angle(correlation)*fs/(2*np.pi)))
    if hint is not None:
        candidates.append(float(hint))
    else:
        candidates.append(0.)
    tried = []
    for f in candidates:
        if abs(f) >= fs/2 or any(abs(f-previous) < fs/nf*.05 for previous in tried):
            continue
        if hint is not None and abs(f-hint) > max(chip_rate*.4, fs/nf*20):
            continue
        tried.append(f)
    best = None
    # A capture can start with a short settling burst. If the first seed fails,
    # try one later contiguous window; retain every original sample for decoding.
    later = min(max(seed_size, round(.001*fs)), len(x)-seed_size)
    for first in dict.fromkeys((0, max(0, later))):
        seed = x[first:first+seed_size]
        time = (first+np.arange(len(seed)))/fs
        for f in tried:
            _cancel(cancel_event)
            try:
                mixed = seed*np.exp(-2j*np.pi*f*time)
                centers, separation = _two_states(mixed, cancel_event)
            except ValueError:
                continue
            labels = abs(mixed-centers[1]) < abs(mixed-centers[0])
            # An Fs/2 alias flips almost every sample, unlike this PWM protocol.
            if np.mean(labels[1:] != labels[:-1]) > min(.25, 6*symbol_hz/fs):
                continue
            if best is None or separation > best[0]:
                best = (separation, float(f), centers, first)
        if best is not None and best[0] >= 3.5:
            break
    if best is None or best[0] < 3.5:
        raise ValueError('载波补偿后没有足够分开的两态，请核对频点提示/信号状态')
    return best[1], best[2], best[3]


def _schmitt(metric, hysteresis):
    initial = bool(metric[0] >= .5)
    events = np.flatnonzero((metric >= .5+hysteresis) | (metric <= .5-hysteresis))
    if not len(events):
        return np.full(len(metric), initial, dtype=bool)
    positions = np.r_[0, events]
    values = np.r_[initial, metric[events] >= .5+hysteresis]
    return np.repeat(values, np.diff(np.r_[positions, len(metric)]))


def _crossings(metric, states, times, samples_per_chip, cancel_event):
    changes = np.flatnonzero(np.diff(states.astype(np.int8)))+1
    result = np.empty(len(changes))
    reach = max(2, int(math.ceil(.4*samples_per_chip)))
    for i, at in enumerate(changes):
        if i % 4096 == 0:
            _cancel(cancel_event)
        lo = max(0, at-reach)
        q = metric[lo:at+1]-.5
        crossings = np.flatnonzero((q[:-1] < 0) & (q[1:] >= 0) if states[at]
                                  else (q[:-1] >= 0) & (q[1:] < 0))
        left = lo+int(crossings[-1]) if len(crossings) else at-1
        delta = metric[left+1]-metric[left]
        weight = np.clip((.5-metric[left])/delta, 0, 1) if abs(delta) > 1e-12 else .5
        result[i] = times[left]+weight*(times[left+1]-times[left])
    return result, states[changes]


def _fit_clock(rise, symbol_hz, total_chips, cancel_event):
    """Fit a common edge grid without advancing the symbol counter on glitches."""
    relative = rise-rise[0]
    # A small event histogram locates the clock line. Precise refinement below
    # uses fractional crossing times, not histogram bins or downsampled IQ.
    histogram_rate = 8*symbol_hz
    bins = np.rint(relative*histogram_rate).astype(int)
    histogram = np.bincount(bins).astype(float)
    nfft = max(64, 1 << (len(histogram)-1).bit_length())
    power = abs(np.fft.rfft(histogram*np.hanning(len(histogram)), nfft))**2
    frequency = np.fft.rfftfreq(nfft, 1/histogram_rate)
    search = np.flatnonzero((frequency >= .98*symbol_hz) & (frequency <= 1.02*symbol_hz))
    if not len(search):
        raise ValueError('符号时钟搜索范围没有有效频点')
    peak = int(search[np.argmax(power[search])])
    df = histogram_rate/nfft
    lo, hi = max(.98*symbol_hz, frequency[peak]-df), min(1.02*symbol_hz, frequency[peak]+df)
    def objective(f):
        _cancel(cancel_event)
        return abs(np.sum(np.exp(2j*np.pi*f*relative)))**2
    ratio = (math.sqrt(5)-1)/2
    a, b = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    ya, yb = objective(a), objective(b)
    for _ in range(32):
        if ya > yb:
            hi, b, yb = b, a, ya
            a = hi-ratio*(hi-lo)
            ya = objective(a)
        else:
            lo, a, ya = a, b, yb
            b = lo+ratio*(hi-lo)
            yb = objective(b)
    clock = (lo+hi)/2
    phasor = np.mean(np.exp(2j*np.pi*clock*relative))
    if abs(phasor) < .45:
        raise ValueError('上升沿没有形成与设置相符的稳定符号时钟')
    period = 1/clock
    anchor = float(rise[0]+np.angle(phasor)/(2*np.pi*clock))
    for _ in range(4):
        _cancel(cancel_event)
        indices = np.rint((rise-anchor)/period).astype(int)
        residual = rise-(anchor+period*indices)
        valid = abs(residual) <= .35*period/total_chips
        if np.count_nonzero(valid) < 3 or len(np.unique(indices[valid])) < 3:
            raise ValueError('稳定符号边沿不足，不能恢复时钟')
        period, anchor = np.polyfit(indices[valid], rise[valid], 1)
    return float(period), float(anchor)


def _clock_hint(transition_times, transition_states):
    """Advisory only: do not change the configured symbol rate automatically."""
    best = None
    for state in (False, True):
        intervals = np.diff(transition_times[transition_states == state])
        if len(intervals) < 8:
            continue
        median = float(np.median(intervals))
        jitter = float(np.median(abs(intervals-median))/max(median, 1e-20))
        if median > 0 and jitter < .05 and (best is None or jitter < best[0]):
            best = jitter, median
    if best is None:
        return ''
    return (f' 实测稳定同向边沿间隔约 {best[1]*1e6:.4f} μs'
            f'（重复率约 {1/best[1]/1000:.5f} kHz），请据此核对 PWM 符号率。')


def _deglitch_runs(edges, levels, limit, budget):
    """Remove only short closed excursions bounded by longer matching runs.

    Symbol-boundary runs and valid >=1-slot pulses are never removed. A fixed
    per-symbol budget prevents a long corrupted interval becoming many repairs.
    Returns a local decision hypothesis; raw IQ and state samples stay intact.
    """
    edges, levels = np.asarray(edges, dtype=float), np.asarray(levels, dtype=bool)
    count, total, maximum, exceeded = 0, 0., 0., False
    while limit > 0 and len(levels) >= 3:
        durations = np.diff(edges)
        candidate = np.flatnonzero((durations[1:-1] <= limit) &
            (durations[:-2] >= limit) & (durations[2:] >= limit) &
            (levels[:-2] == levels[2:]))+1
        if not len(candidate):
            break
        at = int(candidate[np.argmin(durations[candidate])])
        width = float(durations[at])
        if total+width > budget+1e-15:
            exceeded = True
            break
        count += 1
        total += width
        maximum = max(maximum, width)
        edges = np.delete(edges, [at, at+1])
        levels = np.delete(levels, [at, at+1])
    return edges, levels, count, total, maximum, exceeded


def _decode(states, transition_times, transition_states, times, fs, symbol_hz,
            code_chips, guard_chips, tolerance, cancel_event, glitch_tolerance_chips=.15,
            prefix_chips=0, training_options=None, clock_start_s=0., decision_mode='width', width_alphabet=None,
            nearest_level=False):
    total_chips = prefix_chips+code_chips+guard_chips
    bit_width = code_chips if prefix_chips else int(math.log2(code_chips))
    minimum_high = prefix_chips or 1
    maximum_high = prefix_chips+code_chips
    nominal_period = 1/symbol_hz
    chip = nominal_period/total_chips
    rise = transition_times[transition_states]
    if len(rise) < 3:
        raise ValueError('完整上升沿不足 3 个，请增加片段长度或检查两态/符号率')
    # Fit all edges to one grid. Internal spurious rises do not accumulate
    # phantom symbols; missing/idle periods remain in their actual positions.
    fit_rise = rise[rise >= clock_start_s]
    if len(fit_rise) < 3:
        raise ValueError('校准窗后的稳定上升沿不足3个')
    timing_unit = (float(min(np.min(np.diff(sorted(width_alphabet))), min(width_alphabet),
                             total_chips-max(width_alphabet)))
                   if width_alphabet is not None and len(width_alphabet) > 1 else 1.)
    if training_options is not None and 'preamble_widths' in training_options:
        from rx706_pwm_protocol import parse_preamble_widths
        preamble = parse_preamble_widths(training_options['preamble_widths'], total_chips)
        timing_unit = min(timing_unit, min(preamble), total_chips-max(preamble))
    slope, anchor = _fit_clock(fit_rise, symbol_hz, total_chips/timing_unit, cancel_event)
    if abs(slope/nominal_period-1) > .02:
        raise ValueError('测得符号率偏离设置超过 2%，请核对符号率与时间片数')
    period, chip = float(slope), float(slope/total_chips)
    first = int(math.ceil((times[0]-anchor)/period-1e-8))
    last = int(math.floor((times[-1]+1/fs-anchor)/period+1e-8))
    indices = np.arange(first, last)
    if len(indices) < 2:
        raise ValueError('片段中没有足够完整的符号')
    if len(indices) > 1000000:
        raise ValueError('完整符号超过 1000000 个，请减小恢复时长')
    expected = anchor+indices*period
    nearest = np.searchsorted(rise, expected)
    nearest = np.minimum(nearest, len(rise)-1)
    prior = np.maximum(nearest-1, 0)
    nearest = np.where(abs(rise[prior]-expected) < abs(rise[nearest]-expected), prior, nearest)
    edge_margin = .35*chip*timing_unit
    rise_margin = edge_margin
    if width_alphabet is not None:
        # Alphabet spacing controls level resolution, not the timing shift of
        # a measured rising edge after a long high pulse. Bound association by
        # the shortest configured physical high/low pulse instead. Interior
        # transition checks retain their existing, narrower exclusion margin.
        shortest_pulse = min(min(width_alphabet), total_chips-max(width_alphabet))
        if training_options is not None and 'preamble_widths' in training_options:
            shortest_pulse = min(shortest_pulse, min(preamble), total_chips-max(preamble))
        rise_margin = .35*chip*shortest_pulse
    starts = np.where(abs(rise[nearest]-expected) <= rise_margin, rise[nearest], expected)
    ends = np.r_[starts[1:], anchor+last*period]
    frames = np.zeros(len(indices), dtype=_frame_dtype(bit_width, total_chips,
        training_options is not None, decision_mode == 'pattern' or width_alphabet is not None))
    frames['symbol'] = -1
    frames['bits'] = '?'*bit_width
    frames['one_us'] = frames['zero_us'] = frames['high_chips'] = np.nan
    if training_options is not None:
        frames['clock_high_chips'] = np.nan
    # Integrate the measured decisions at full processing rate, rather than
    # selecting one sample per slot or using a plotted/decimated view.
    integral = np.r_[0., np.cumsum(states, dtype=float)]/fs
    sample_edges = np.r_[times, times[-1]+1/fs]
    for i, (start, end) in enumerate(zip(starts, ends)):
        if i % 2048 == 0:
            _cancel(cancel_event)
        row = frames[i]
        row['index'], row['start_s'] = i+1, start
        points = start+np.arange(total_chips+1)*(end-start)/total_chips
        fractions = np.diff(np.interp(points, sample_edges, integral))/np.diff(points)
        pattern = fractions >= .5
        row['pattern'] = ''.join('1' if v else '0' for v in pattern)
        first_event = int(np.searchsorted(transition_times, start, side='right'))
        last_event = int(np.searchsorted(transition_times, end, side='left'))
        initial = transition_states[first_event-1] if first_event else states[0]
        raw_edges = np.r_[start, transition_times[first_event:last_event], end]
        raw_levels = np.r_[initial, transition_states[first_event:last_event]]
        clean_edges, clean_levels, count, width, maximum, exceeded = _deglitch_runs(
            raw_edges, raw_levels, glitch_tolerance_chips*chip, .30*chip)
        row['glitch_count'], row['glitch_us'], row['glitch_max_us'] = count, width*1e6, maximum*1e6
        if count:
            row['warning'] = f'忽略闭合短毛刺{count}次，累计{width*1e9:.2f}ns，非误码判定'
        if not np.any(pattern) and len(clean_levels) == 1 and not clean_levels[0] and not exceeded:
            row['one_us'], row['zero_us'], row['high_chips'], row['duty_percent'] = 0, (end-start)*1e6, 0, 0
            row['symbol'], row['bits'], row['reason'] = -2, '-', '空闲：不输出数据比特'
            continue
        clean_times = clean_edges[1:-1]
        clean_after = clean_levels[1:]
        # Preserve the existing symbol-edge uncertainty margins. A noisy
        # boundary must not be reinterpreted as a newly invalid data transition.
        interior = clean_after[(clean_times >= start+edge_margin) & (clean_times < end-edge_margin)]
        falling = np.flatnonzero(clean_levels[:-1] & ~clean_levels[1:] &
                                 (clean_times >= start+edge_margin))+1
        if not len(falling):
            row['reason'] = '没有完整的 1→0 边沿'
            continue
        main_fall = clean_edges[int(falling[0])]
        one, zero = main_fall-start, end-main_fall
        high = one/chip
        number = int(math.floor(high+.5))
        row['one_us'], row['zero_us'] = one*1e6, zero*1e6
        row['high_chips'], row['duty_percent'] = high, 100*one/(end-start)
        if training_options is not None:
            row['clock_high_chips'] = (main_fall-expected[i])/chip
        reason = ''
        if abs(rise[nearest[i]]-expected[i]) > rise_margin:
            reason = '符号起点缺少上升沿'
        elif exceeded:
            reason = '短毛刺累计超过0.30个时间片'
        elif len(interior) != 1 or bool(interior[0]):
            reason = '存在未能忽略的额外跳变'
        if training_options is not None or decision_mode == 'pattern' or width_alphabet is not None:
            row['structure_ok'] = not bool(reason)
        if not reason and not minimum_high <= number <= maximum_high:
            reason = '固定高电平保护不足或脉宽超范围' if prefix_chips else '高电平长度超出编码范围'
        elif not reason and abs(high-number) > tolerance:
            reason = '脉宽偏离整数时间片'
        elif not reason and (zero/chip < guard_chips-tolerance or np.any(pattern[-guard_chips:])):
            reason = '末尾低电平保护片不足'
        elif not reason and not np.array_equal(pattern, np.arange(total_chips) < number):
            reason = '时间片状态不符合连续 1 后接 0'
        row['reason'] = reason or ('有效（短毛刺告警）' if count else '有效')
        row['valid'] = not reason
        if not reason:
            row['symbol'] = number-minimum_high
            row['bits'] = ('1'*(number-prefix_chips)+'0'*(maximum_high-number)
                           if prefix_chips else format(number-1, f'0{bit_width}b'))
    if training_options is not None:
        from rx706_pwm_training import apply_training
        # Both the falling edge and the origin (measured or clock-fitted from
        # rising edges) have timing uncertainty. Budget half a processing
        # sample for each; fitting a clock does not make its origin exact.
        uncertainty = 1./(fs*chip)
        apply_training(frames, total_chips, prefix_chips, code_chips, guard_chips,
                       tolerance, training_options, cancel_event,
                       measurement_uncertainty_chips=uncertainty)
    if decision_mode == 'pattern':
        if training_options is None:
            frames['raw_reason'] = frames['reason']
        apply_pattern_decisions(frames, total_chips, code_chips, prefix_chips,
                                guard_chips, training_options is not None, cancel_event)
    if width_alphabet is not None and decision_mode == 'width':
        from rx706_pwm_protocol import apply_alphabet_decisions
        apply_alphabet_decisions(frames, width_alphabet, prefix_chips, code_chips, tolerance,
                                 training_options is not None, nearest=nearest_level)
        if (training_options is not None and training_options.get('auto_scale', False) and
                'preamble_widths' in training_options):
            from rx706_pwm_protocol import parse_preamble_widths
            preamble = parse_preamble_widths(training_options['preamble_widths'], total_chips)
            if np.ptp(preamble) <= 1e-9:
                warming = (frames['valid'] & (frames['calibration_id'] > 0) &
                           (frames['calibration_source'] != '前导＋档位标定'))
                frames['reason'][warming] = np.char.add(frames['reason'][warming],
                                                       '；档位标定预热，仅前导偏移')
    if training_options is not None and 'frame_options' in training_options:
        failed = (frames['frame_part'] == '净荷') & ~frames['identity_ok']
        frames['valid'][failed], frames['symbol'][failed], frames['bits'][failed] = False, -1, '?'*bit_width
        frames['reason'][failed] = '身份码失败：净荷按擦除保位'
    return frames, 1/period


def pattern_symbol(pattern, total_chips, code_chips, prefix_chips=0):
    """Independent raw slot-pattern interpretation; no width rounding/calibration."""
    pattern = str(pattern)
    if len(pattern) != total_chips or set(pattern)-{'0', '1'}:
        return -1
    ones = pattern.count('1')
    if not ones:
        return -2
    minimum = prefix_chips or 1
    maximum = prefix_chips+code_chips
    if not minimum <= ones <= maximum or pattern != '1'*ones+'0'*(total_chips-ones):
        return -1
    return ones-minimum


def decision_width_chips(row):
    """Return the same continuous width used for candidate decisions and UI."""
    calibrated = 'calibration_id' in row.dtype.names and row['calibration_id'] > 0
    return float(row['corrected_high_chips'] if calibrated else row['high_chips'])


def apply_pattern_decisions(frames, total_chips, code_chips, prefix_chips, guard_chips,
                            require_training=False, cancel_event=None):
    """Decide the raw slot word, independently of continuous width rounding."""
    for i, row in enumerate(frames):
        if i % 2048 == 0:
            _cancel(cancel_event)
        if row['symbol'] in (-2, -3):
            continue
        row['valid'], row['symbol'], row['bits'] = False, -1, '?'*code_chips
        if require_training and row['calibration_id'] <= 0:
            row['reason'] = '尚未检测到20%三连训练段'
            continue
        if not row['structure_ok']:
            row['reason'] = '原始边沿/跳变不完整，码型不确认为有效'
            continue
        k = pattern_symbol(row['pattern'], total_chips, code_chips, prefix_chips)
        if k < 0:
            row['reason'] = '码型不符合固定前高/后低或连续1后接0'
            continue
        row['valid'], row['symbol'] = True, k
        row['bits'] = str(row['pattern'])[prefix_chips:total_chips-guard_chips]
        row['reason'] = '有效（时间片码型，毛刺告警）' if row['glitch_count'] else '有效（时间片码型）'


def _candidate_decisions(frames, code_chips, prefix_chips=0, decision_mode='width'):
    """Nearest-width hypotheses for inspection, independent of validity/BER."""
    frames['candidate_symbol'] = -1
    frames['candidate_bits'] = '?'*(code_chips if prefix_chips else int(math.log2(code_chips)))
    for row in frames:
        if row['symbol'] == -3:
            row['candidate_symbol'], row['candidate_bits'] = -3, '-'
            row['candidate_source'] = '训练前导段，不输出净荷'
            continue
        if row['symbol'] == -2:
            row['candidate_symbol'], row['candidate_bits'] = -2, '-'
            row['candidate_source'] = '全0，按当前协议为空闲'
            continue
        if row['valid']:
            row['candidate_symbol'], row['candidate_bits'] = row['symbol'], row['bits']
            row['candidate_source'] = ('按原始时间片码型判决；不使用脉宽取整' if decision_mode == 'pattern' else
                                      '补偿后脉宽通过判决；原始码型未改写' if 'calibration_id' in frames.dtype.names and row['calibration_id'] > 0
                                       else '满足局部脉宽/保护检查')
            continue
        if decision_mode == 'pattern':
            raw_pattern = str(row['pattern'])
            k = pattern_symbol(raw_pattern, len(raw_pattern), code_chips, prefix_chips)
            if k >= 0:
                row['candidate_symbol'] = k
                row['candidate_bits'] = '1'*k+'0'*(code_chips-k)
                row['candidate_source'] = '码型推断；原始边沿或训练未通过，非确认信息'
            else:
                row['candidate_source'] = '原始码型不合法；未生成候选'
            continue
        calibrated = ('calibration_id' in frames.dtype.names and row['calibration_id'] > 0)
        width = decision_width_chips(row)
        if math.isfinite(width) and width > 0:
            rounded = int(math.floor(width+.5))
            source = f'{"补偿" if calibrated else "实测"}脉宽{width:.4f}片取整；未通过检查'
            raw_pattern = str(row['pattern'])
            raw_k = pattern_symbol(raw_pattern, len(raw_pattern), code_chips, prefix_chips)
            if raw_k >= 0:
                source += f'；原始码型推断{raw_k}'
        else:
            rounded = str(row['pattern']).count('1')
            if not rounded:
                row['candidate_source'] = '无可用高电平宽度'
                continue
            source = '缺主脉宽，按时间片中1的数量估计；未通过检查'
        minimum, maximum = prefix_chips or 1, prefix_chips+code_chips
        nearest = min(maximum, max(minimum, rounded))
        row['candidate_symbol'] = nearest-minimum
        row['candidate_bits'] = ('1'*(nearest-prefix_chips)+'0'*(maximum-nearest)
                                 if prefix_chips else format(nearest-1, f'0{int(math.log2(code_chips))}b'))
        row['candidate_source'] = source+('；超范围限幅' if nearest != rounded else '')


def _find_pwm_span(spectrum, duration_s, **options):
    """Find the first PWM run with overlapping bounded, protocol-checked probes.

    Search is restricted to the requested interval. Probes are disposable fits,
    not pilots: final decoding rereads the onset and reports original timestamps.
    """
    fs = spectrum.sample_rate_hz
    if not math.isfinite(float(duration_s)) or not .001 <= duration_s <= 1:
        raise ValueError('恢复时长必须为 1～1000 ms')
    hint = options['center_hint_hz']
    if hint is not None and (not math.isfinite(float(hint)) or abs(hint) >= fs/2):
        raise ValueError('载波中心提示必须在基带采样范围内')
    cancel = options['cancel_event']
    begin = spectrum.first_tuple
    end = min(spectrum.total_tuples, begin+round(duration_s*fs))
    # At least 32 symbols and enough samples for initialization; 10ms normally.
    width = max(round(.01*fs), math.ceil(32*fs/options['symbol_hz']), 8192)
    width = min(width, end-begin)
    if width < 512:
        raise ValueError('搜索范围太短，无法检测PWM')
    step = max(1, width//2)
    last_error = ''
    initial_diagnostic = False
    peak_probe_clipping = 0.

    def probe_note():
        return (f' 搜索小窗触限最高{peak_probe_clipping:.3%}，仅用于定位；'
                '完整恢复片段仍执行原0.1%削顶拦截。') if peak_probe_clipping > .001 else ''
    for position in range(begin, end-511, step):
        _cancel(cancel)
        span = min(width, end-position)
        if span < max(512, .001*fs, 8*fs/options['symbol_hz']):
            break
        probe = copy(spectrum)
        probe.first_tuple = position
        probe.total_tuples = spectrum.total_tuples
        seconds = max(.001, span/fs)
        # Reject receiver noise and fixed-state CW cheaply before fitting PWM.
        # A local startup spike must not reject a full capture whose overall
        # clipping fraction is acceptable. Only internal search probes bypass
        # this check; the delivered recovery is reread with the strict limit.
        x, read_count, probe_clipped = _read_span(probe, seconds, hint, cancel,
                                                 max_samples=span, allow_clipped=True)
        peak_probe_clipping = max(peak_probe_clipping, probe_clipped/read_count)
        energy = float(np.vdot(x, x).real)
        corr = np.vdot(x[:-1], x[1:])
        coherence = abs(corr)/max(energy, 1e-30)
        if coherence < .1:
            continue
        frequency = np.angle(corr)*fs/(2*np.pi)
        mixed = x*np.exp(-2j*np.pi*frequency*np.arange(len(x))/fs)
        if np.mean(abs(mixed-np.mean(mixed))**2) < .002*energy/len(x):
            continue
        try:
            result = recover_pwm(probe, duration_s=seconds, allow_unlocked=True,
                                 auto_search=False, _search_probe=True, **options)
        except ValueError as exc:
            last_error = str(exc)
            continue
        # Keep diagnostics for malformed PWM already present at the requested
        # start. This preserves the existing two-state/invalid-code view.
        valid = result.frames['valid']
        runs = np.convolve(valid.astype(int), np.ones(8, dtype=int), mode='valid') if len(valid) >= 8 else []
        good = np.flatnonzero(np.asarray(runs) == 8)
        if options.get('training_options') is not None:
            good = np.flatnonzero(valid)
        if position == begin:
            if result.decode_locked and len(good):
                return spectrum, duration_s, 'PWM搜索：首窗已检测到有效PWM；从原始起点重新解调，未预留导频。'+probe_note()
            initial_diagnostic = True
        if not result.decode_locked or not len(good):
            continue
        # Inspect one preceding overlap using the detected polarity. It can
        # contain noise; require eight consecutive complete valid symbols.
        earlier = copy(spectrum)
        earlier.first_tuple = max(begin, position-step)
        expanded = min(end-earlier.first_tuple, position+span-earlier.first_tuple)
        try:
            # A/B centroid ordering may rotate when the local time origin
            # changes. Re-evaluate auto polarity in the expanded window.
            local_options = dict(options)
            previous = recover_pwm(earlier, duration_s=max(.001, expanded/fs),
                allow_unlocked=True, auto_search=False,
                _seed_offset=position-earlier.first_tuple, _search_probe=True, **local_options)
            pv = previous.frames['valid'].astype(int)
            pr = np.flatnonzero(np.convolve(pv, np.ones(8, int), 'valid') == 8) if len(pv) >= 8 else []
            if options.get('training_options') is not None:
                pr = np.flatnonzero(pv)
            if len(pr):
                result, good = previous, pr
        except ValueError:
            pass
        onset = result.frames['start_s'][good[0]]
        if options.get('training_options') is not None:
            pilots_before = np.flatnonzero(result.frames['training'] & (np.arange(len(result.frames)) < good[0]))
            if len(pilots_before):
                onset = result.frames['start_s'][pilots_before[0]]
        # Two symbols of pretrigger protect the first complete data symbol.
        first = max(begin, int(math.floor((onset-2/options['symbol_hz'])*fs)))
        selected = copy(spectrum)
        selected.first_tuple = first
        remaining = (end-first)/fs
        if remaining < .001:
            raise ValueError('检测到PWM但剩余恢复片段不足1ms')
        note = (f'PWM搜索范围 {begin/fs:.6f}～{end/fs:.6f} s；首个连续有效符号段约 '
                f'{onset:.6f} s；从 {first/fs:.6f} s 重新解调（保留2符号前置余量）。'
                '搜索窗不作为导频丢弃；时间戳保持原文件坐标。')+probe_note()
        return selected, remaining, note
    if initial_diagnostic:
        return spectrum, duration_s, 'PWM搜索未找到连续8个有效符号；保留原始起点的两态/编码未锁定诊断。'+probe_note()
    raise ValueError('指定范围内未找到有效PWM片段（需连续8个有效符号）；请核对符号率、频点和信号电平。'
                     + (f' 最后诊断：{last_error}' if last_error else '')+probe_note())


def recover_pwm(spectrum, *, symbol_hz=1e6/3, code_chips=8, guard_chips=2,
                duration_s=1., center_hint_hz=None, decimation=1,
                state_one='auto', width_tolerance=.3, hysteresis=.10,
                glitch_tolerance_chips=.15, cancel_event=None, allow_unlocked=False,
                auto_search=False, _seed_offset=0, _version=1,
                _prefix_chips=8, _total_chips=20, _search_probe=False, training_options=None,
                decision_mode='width', symbol_widths=None, output_binary=False, pulse_options=None):
    """Measure 0/1 duration and decode PWM without assuming a 50% duty cycle.

    Carrier initialization compares raw/squared spectral-line candidates.
    Decision-directed, common block phase tracking preserves the two states;
    no per-sample or per-state phase resets are applied. Known preambles are
    still needed for absolute labels, packet framing and measured BER.
    """
    fs = float(spectrum.sample_rate_hz)
    pulse_calibrations, pulse_choices = None, {}
    if pulse_options is not None:
        from rx706_pwm_pulse import validate_pulse_options
        pulse_options = validate_pulse_options(pulse_options, symbol_hz, fs/decimation)
        _version, _total_chips, _prefix_chips = 2, pulse_options['total_chips'], 0
        code_chips, guard_chips = _total_chips, 0
        symbol_widths = tuple(w for _, w in pulse_options['codebook'])
        output_binary, training_options, auto_search = True, None, False
    if not math.isfinite(float(symbol_hz)) or symbol_hz <= 0:
        raise ValueError('PWM 符号率必须为有限正数')
    if decision_mode not in ('width', 'pattern') or (_version != 2 and decision_mode != 'width'):
        raise ValueError('时间片码型判决仅支持PWM version2；判决依据须为width或pattern')
    prefix_chips = _prefix_chips if _version == 2 else 0
    if _version == 2 and pulse_options is None:
        code_chips = pwm2_layout(_total_chips, prefix_chips, guard_chips)
    if symbol_widths is not None and pulse_options is None:
        if _version != 2:
            raise ValueError('自定义符号映射仅支持PWM version2')
        from rx706_pwm_protocol import parse_widths
        symbol_widths = parse_widths(symbol_widths, _total_chips, minimum=prefix_chips,
                                    maximum=_total_chips-guard_chips)
        if len(set(symbol_widths)) != len(symbol_widths):
            raise ValueError('符号映射表中的高片数不能重复')
        if training_options is not None:
            training_options = dict(training_options, alphabet_widths=symbol_widths)
    if _version != 2 and (type(code_chips) is not int or code_chips not in (2, 4, 8, 16)):
        raise ValueError('编码片数必须为 2、4、8 或 16')
    if training_options is not None:
        if _version != 2:
            raise ValueError('三连训练补偿仅支持PWM version2')
        from rx706_pwm_training import validate_training
        training_target, _ = validate_training(prefix_chips+code_chips+guard_chips, prefix_chips, code_chips, training_options)
    if _version != 2 and (type(guard_chips) is not int or not 1 <= guard_chips <= 16):
        raise ValueError('保护片数必须为 1～16 的整数')
    if decimation not in (1, 2, 4, 8, 16):
        raise ValueError('处理抽取倍数必须为 1、2、4、8 或 16')
    if state_one not in ('auto', 'A', 'B'):
        raise ValueError('状态 1 映射必须为 auto、A 或 B')
    if not math.isfinite(float(width_tolerance)) or not 0 < width_tolerance < .5:
        raise ValueError('脉宽容差必须大于 0 且小于 0.5 个时间片')
    if not math.isfinite(float(hysteresis)) or not 0 <= hysteresis < .4:
        raise ValueError('两态判决滞回半宽必须为 0～0.4（不含 0.4）')
    if not math.isfinite(float(glitch_tolerance_chips)) or not 0 <= glitch_tolerance_chips < .5:
        raise ValueError('单次毛刺上限必须大于等于0且小于0.5个时间片；0表示严格模式')
    chip_rate = symbol_hz*(prefix_chips+code_chips+guard_chips)
    spacing = (1 if pulse_options is not None else min(min(np.diff(sorted(symbol_widths))), min(symbol_widths),
                   _total_chips-max(symbol_widths))
               if symbol_widths is not None and len(symbol_widths) > 1 else 1)
    if training_options is not None:
        targets = np.atleast_1d(training_target)
        spacing = min(spacing, float(targets.min()), prefix_chips+code_chips+guard_chips-float(targets.max()))
    points_per_chip = fs/decimation/chip_rate*spacing
    if points_per_chip < 3:
        if symbol_widths is not None:
            raise ValueError(f'最小占空比间隔或最短高/低脉冲仅{points_per_chip:.3f}个IQ点，至少需要3点；'
                             '请提高实际采样率、降低方波频率，或增大占空比间隔。')
        raise ValueError(f'Fs={fs/1e6:g} MS/s，抽取{decimation}倍，PWM={symbol_hz/1000:g} kHz，'
                         f'{prefix_chips}+{code_chips}+{guard_chips}片：每片只有{points_per_chip:.3f}个IQ点。'
                         '当前插值判决至少需要3点/片，建议4点以上；请减小抽取倍数，'
                         '或提高实际采集采样率/降低符号率。修改文件的Fs标签不能增加采样点。')
    sampling_note = (f'每时间片{points_per_chip:.3f}个IQ点，低于建议的4点；使用原始点间边沿插值，'
                     '计时余量较小，请结合波形和擦除比例判断可靠性。' if points_per_chip < 4 else '')
    if symbol_widths is not None and points_per_chip < 4:
        sampling_note = f'最小占空比间隔/最短脉冲{points_per_chip:.3f}个IQ点，低于建议4点；请结合擦除比例判断。'
    search_note = ''
    if auto_search and not (training_options is not None and 'frame_options' in training_options):
        spectrum, duration_s, search_note = _find_pwm_span(spectrum, duration_s,
            symbol_hz=symbol_hz, code_chips=code_chips, guard_chips=guard_chips,
            center_hint_hz=center_hint_hz, decimation=decimation,
            state_one=state_one, width_tolerance=width_tolerance, hysteresis=hysteresis,
            glitch_tolerance_chips=glitch_tolerance_chips, cancel_event=cancel_event, _version=_version,
            _prefix_chips=prefix_chips, _total_chips=prefix_chips+code_chips+guard_chips,
            training_options=training_options, decision_mode=decision_mode, symbol_widths=symbol_widths,
            output_binary=output_binary)
    # PWM may process a complete one-second span; other recovery modes keep
    # their existing bounded read. Search probes always remain small.
    x, count, clipped = _read_span(spectrum, duration_s, center_hint_hz, cancel_event,
                                  max_samples=math.ceil(fs), allow_clipped=_search_probe)
    if count < max(512, 8*fs/symbol_hz):
        raise ValueError('PWM 恢复至少需要 8 个完整符号，请增加恢复时长')
    _cancel(cancel_event)
    correlation = np.vdot(x[:-1], x[1:])
    coherence = abs(correlation)/max(float(np.vdot(x, x).real), 1e-30)
    if coherence < .10:
        raise ValueError('载波相干性不足，不能可靠恢复两态 PWM')
    block = max(256, int(round(fs*max(4/symbol_hz, .0001))))
    seed_size = min(count, max(block, 4096))
    carrier, centers, calibration_start = _carrier_seed(x[_seed_offset:], fs,
        min(seed_size, count-_seed_offset), center_hint_hz, chip_rate, symbol_hz, cancel_event)
    if _seed_offset:
        centers *= np.exp(-2j*np.pi*carrier*_seed_offset/fs)
        calibration_start += _seed_offset
    times = np.arange(count)/fs
    mixed = x*np.exp(-2j*np.pi*carrier*times)
    track_times, track_phases = [], []
    phase = 0.
    for start in range(0, count, block):
        _cancel(cancel_event)
        end = min(start+block, count)
        z = mixed[start:end]
        if end-start < 16:
            break
        for _ in range(4):
            rotated = z*np.exp(-1j*phase)
            labels = abs(rotated-centers[1]) < abs(rotated-centers[0])
            expected = centers[labels.astype(int)]
            residual = rotated*np.conj(expected)
            # One phase correction for many symbols; label differences remain.
            correction = float(np.angle(np.sum(residual)))
            phase += correction
            if abs(correction) < 1e-5:
                break
        track_times.append((start+end-1)/(2*fs))
        track_phases.append(phase)
    tracked = np.interp(times, track_times, np.unwrap(track_phases))
    z = mixed*np.exp(-1j*tracked)
    if len(track_times) >= 2:
        carrier += float(np.polyfit(track_times, np.unwrap(track_phases), 1)[0]/(2*np.pi))
    rate, offset, remaining, cutoff = fs, 0., decimation, 0.
    while remaining > 1:
        cutoff = min(2*chip_rate, .35*(rate/2))
        z, take = _lowpass(z, rate, cutoff, 2, 128, cancel_event)
        if not len(take):
            raise ValueError('片段太短，无法进行抗混叠抽取')
        offset += take[0]/rate
        rate /= 2
        remaining //= 2
    times = offset+np.arange(len(z))/rate
    fit_start = max(0, int(math.ceil((calibration_start/fs-offset)*rate)))
    fit_end = min(len(z), max(len(z)//2, fit_start+int(math.ceil(seed_size*rate/fs))))
    centers, separation = _two_states(z[fit_start:fit_end], cancel_event)
    state_fit_note = ''
    if separation < 3.5:
        whole_separation = separation
        local_end = min(len(z), fit_start+int(math.ceil(seed_size*rate/fs)))
        centers, separation = _two_states(z[fit_start:local_end], cancel_event)
        if separation < 3.5:
            raise ValueError('处理后的两个 IQ 状态不够分离，请检查恢复参数')
        state_fit_note = (f'长窗两态分离度{whole_separation:.2f}不足；改用初始化校准窗拟合质心'
                          f'（分离度{separation:.2f}）。全段原始点仍保留；局部校准不代表全段质量一致。')
    # Rotation changes coordinates only; relative phase/amplitude is preserved.
    rotation = np.exp(-1j*np.angle(centers[1]-centers[0]))
    centers *= rotation
    z *= rotation
    delta = centers[1]-centers[0]
    metric = np.real((z-centers[0])*np.conj(delta))/abs(delta)**2
    states = _schmitt(metric, hysteresis)
    transitions, next_states = _crossings(metric, states, times, rate/chip_rate, cancel_event)
    choices = []
    for one in ([0, 1] if state_one == 'auto' else [0 if state_one == 'A' else 1]):
        level = states if one == 1 else ~states
        after = next_states if one == 1 else ~next_states
        try:
            if pulse_options is not None:
                from rx706_pwm_pulse import decode_pulses
                frames, measured, calibrations, normalized = decode_pulses(level, transitions, after,
                    times, rate, symbol_hz, pulse_options, cancel_event)
                pulse_choices[one] = calibrations, normalized
            else:
                frames, measured = _decode(level, transitions, after, times, rate, symbol_hz,
                                       code_chips, guard_chips, width_tolerance, cancel_event,
                                       glitch_tolerance_chips, prefix_chips, training_options,
                                       max(0., _seed_offset/fs-offset), decision_mode, symbol_widths,
                                       nearest_level=output_binary)
            active = frames['symbol'] != -2
            if training_options is not None:
                active &= ~frames['training'] & (frames['calibration_id'] > 0)
            score = float(np.mean(frames['valid'][active])) if np.any(active) else 0.
            if training_options is not None and 'frame_options' in training_options:
                # A successfully acquired known prefix establishes polarity even
                # when every subsequent payload decision is an erasure.
                score += 2*bool(np.any(frames['block_id'] > 0))
            if pulse_options is not None:
                score += 2*bool(np.any(frames['block_id'] > 0))
            choices.append((score, one, frames, measured))
        except ValueError:
            if state_one != 'auto':
                raise
    if not choices:
        raise ValueError(f'设置的 PWM 符号率为 {symbol_hz/1000:.6g} kHz，未匹配到符号时钟。'
                         +_clock_hint(transitions, next_states))
    choices.sort(key=lambda item: item[0], reverse=True)
    score, one, frames, measured = choices[0]
    if pulse_options is not None:
        pulse_calibrations, pulse_options = pulse_choices[one]
    ambiguous = len(choices) > 1 and score-choices[1][0] < .15
    fixed = [np.unique(item[2]['symbol'][item[2]['valid']]) for item in choices]
    equivalent = (len(fixed) == 2 and len(fixed[0]) == len(fixed[1]) == 1
                  and fixed[0][0] == fixed[1][0])
    decode_note = ''
    if training_options is not None and 'frame_options' in training_options and not np.any(frames['block_id'] > 0):
        decode_note = '最简帧未定位：公共训练或首个用户身份码未匹配；请核对帧配置和恢复起点。'
    elif training_options is not None and not np.any(frames['training']):
        decode_note = '未锁定：未检测到有效的已配置训练前导，未进行补偿或输出净荷。'
    elif training_options is not None and 'frame_options' not in training_options and not np.any(frames['valid']):
        decode_note = '检测到训练但没有通过校准后判决的数据符号。'
    elif state_one == 'auto' and score < .5:
        decode_note = (f'候选时钟{measured/1000:.6f} kHz；PWM编码符合比例仅{score:.1%}，'
                       + ('码型未通过结构检查。' if decision_mode == 'pattern' else '脉宽未通过编码检查。')
                       + '请核对实际占空比、末尾保护和切换响应。')
    elif state_one == 'auto' and ambiguous and not equivalent:
        decode_note = '自动0/1映射有歧义；请选择A=1 / B=1，并核对训练序列。'
    if decode_note and not allow_unlocked:
        raise ValueError(f'设置的 PWM 符号率为 {symbol_hz/1000:.6g} kHz，'+decode_note+
                         _clock_hint(transitions, next_states))
    bit_width = (max(1, math.ceil(math.log2(len(symbol_widths)))) if pulse_options is not None else
                 code_chips if prefix_chips else int(math.log2(code_chips)))
    if pulse_options is None:
        _candidate_decisions(frames, code_chips, prefix_chips, decision_mode)
    if symbol_widths is not None and pulse_options is None:
        if decision_mode == 'width':
            for row in frames:
                if row['symbol'] in (-2, -3) or row['valid']:
                    continue
                high = decision_width_chips(row)
                if math.isfinite(high) and high > 0:
                    nearest = min(symbol_widths, key=lambda w: abs(w-high))
                    k = nearest-prefix_chips
                    row['candidate_symbol'], row['candidate_bits'] = k, '1'*k+'0'*(code_chips-k)
                    row['candidate_source'] = '最邻近配置占空比；未确认，不计BER'
        # Keep physical thermometer words unchanged; only symbol labels follow
        # the user codebook. Unknown widths remain erasures, never clamped.
        for key in ('symbol', 'candidate_symbol'):
            active = frames[key] >= 0
            physical = frames[key][active].copy()+prefix_chips
            frames[key][active] = [symbol_widths.index(int(w)) if w in symbol_widths else -1 for w in physical]
        rejected = frames['valid'] & (frames['symbol'] < 0)
        frames['valid'][rejected] = False
        frames['bits'][rejected] = '?'*bit_width
        frames['reason'][rejected] = '高片数不在已配置符号映射表中'
        frames['candidate_bits'][frames['candidate_symbol'] == -1] = '?'*bit_width
    if output_binary and pulse_options is None:
        if symbol_widths is None:
            raise ValueError('二进制符号输出需要显式占空比映射表')
        bit_width = max(1, math.ceil(math.log2(len(symbol_widths))))
        for key, bit_key in (('symbol', 'bits'), ('candidate_symbol', 'candidate_bits')):
            active = frames[key] >= 0
            frames[bit_key][active] = [format(int(k), f'0{bit_width}b') for k in frames[key][active]]
            frames[bit_key][frames[key] == -1] = '?'*bit_width
    if decode_note:
        # Diagnostic state recovery is useful even if the PWM protocol fails.
        # Preserve explicit candidate columns; they are never treated as BER data.
        invalidated = ~frames['training'] if training_options is not None or pulse_options is not None else np.ones(len(frames), dtype=bool)
        frames['valid'][invalidated] = False
        frames['symbol'][invalidated] = -1
        frames['bits'][invalidated] = '?'*bit_width
        frames['reason'][invalidated] = np.char.add('PWM未锁定；', frames['reason'][invalidated])
    mapping = f'{"AB"[one]}=1；'+('按 PWM 约束推断，需训练序列确认' if state_one == 'auto' else '用户指定映射')
    if state_one == 'auto' and ambiguous and equivalent:
        mapping = (f'相对 {"AB"[one]}=1；真实控制 0/1 未标定；两种映射均为固定符号 '
                   f'{int(fixed[0][0])}，占空比测量可用')
    if decode_note:
        mapping = f'相对{"AB"[one]}电平记作1；真实0/1未确定；候选符号仅供观察'
    view = min(len(z), max(256, int(math.ceil(12*rate/symbol_hz))))
    # Keep a contiguous view, never a one-sample-per-slot trace.
    view = min(view, 20000)
    selected = np.sort(np.random.default_rng(706).choice(len(z), min(8192, len(z)), replace=False))
    scatter_labels = np.where((metric > .2) & (metric < .8), -1, states.astype(int))
    power_step = max(1, round(rate*.001))
    starts = np.arange(0, len(z), power_step)
    power = np.add.reduceat(abs(z)**2, starts)/np.diff(np.r_[starts, len(z)])
    origin = spectrum.first_tuple/fs
    frames['start_s'] += origin
    bitstream = ''.join(frames['bits'][~frames['training']] if training_options is not None or pulse_options is not None else frames['bits'])
    valid_count = int(np.count_nonzero(frames['valid']))
    measured_duration = (frames['start_s'][-1]-frames['start_s'][0])+1/measured
    training_calibration = None
    if training_options is not None:
        mask = frames['training']
        starts_of_training = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        training_calibration = dict(training_symbols=int(mask.sum()),
            updates=int(frames['calibration_id'].max(initial=0)),
            corrections_chips=frames['width_correction_chips'][starts_of_training].tolist(),
            starts_s=frames['start_s'][starts_of_training].tolist(),
            target_duty_percent=None if 'preamble_widths' in training_options else 20.,
            preamble_widths=training_options.get('preamble_widths'),
            payload_symbols=training_options.get('payload_symbols', 0),
            auto_scale=bool(training_options.get('auto_scale', False)),
            clock_width=bool(training_options.get('clock_width', False)),
            measurement_uncertainty_chips=(prefix_chips+code_chips+guard_chips)*measured/rate,
            scale_range=[float(np.nanmin(frames['width_scale'][frames['calibration_id'] > 0])),
                         float(np.nanmax(frames['width_scale'][frames['calibration_id'] > 0]))]
                        if np.any(frames['calibration_id'] > 0) else [],
            spacing_symbols=int(np.count_nonzero(frames['calibration_source'] == '前导＋档位标定')))
    result = PWMRecovery(carrier, rate, cutoff, decimation, spectrum.first_tuple, count, clipped,
        centers, float(np.angle(centers[1]*np.conj(centers[0]), deg=True)),
        float(20*np.log10(max(abs(centers[1]), 1e-20)/max(abs(centers[0]), 1e-20))),
        separation, z[selected], scatter_labels[selected], z[:view], times[:view]+origin,
        times[np.minimum(starts+power_step//2, len(z)-1)]+origin,
        10*np.log10(np.maximum(power/2048**2, 1e-18)), times[:view]+origin,
        (states[:view] if one == 1 else ~states[:view]),
        (metric[:view] if one == 1 else 1-metric[:view]), frames, float(symbol_hz), measured,
        code_chips, guard_chips, bit_width, one, mapping, bitstream,
        valid_count*bit_width/measured_duration, float(glitch_tolerance_chips),
        sampling_note=sampling_note, calibration_start_s=origin+calibration_start/fs,
        decode_locked=not bool(decode_note), decode_note=decode_note, state_fit_note=state_fit_note,
        search_note=search_note, version=_version, prefix_chips=prefix_chips,
        training_calibration=training_calibration, decision_mode=decision_mode, symbol_widths=symbol_widths,
        output_binary=output_binary)
    if pulse_options is not None:
        from rx706_pwm_frames import check_frame_reference
        result.pulse_options, result.pulse_calibrations = pulse_options, pulse_calibrations
        result.frame_options = pulse_options['frame_options']
        result.frame_check = check_frame_reference(frames, result.frame_options, symbol_widths,
                                                  result.symbol_bitwords, cancel_event)
        result.reference_check = result.frame_check
        result.decoded_bit_rate = valid_count*pulse_options['information_bits']/measured_duration
        result.training_calibration = dict(training_symbols=int(frames['training'].sum()),
            updates=len(pulse_calibrations), corrections_chips=[], starts_s=[], target_duty_percent=None,
            preamble_widths=None, payload_symbols=0, auto_scale=False, clock_width=True,
            scale_range=[1., 1.], spacing_symbols=0)
        for calibration in pulse_calibrations:
            calibration['start_s'] += origin
        for key in ('rise_s', 'fall_s'):
            frames[key] += origin
        result.sampling_note += (f" 依据前导实测拖尾及高电平稳定假设估计周期下限{pulse_options['estimated_min_period_us']:.3f}µs，"
                                f"当前{1e6/symbol_hz:.3f}µs；器件稳定条件仍需实测。")
        if not result.decode_locked:
            result.frame_check.locked = False
            result.frame_check.reason = result.decode_note
    if training_options is not None and 'frame_options' in training_options:
        from rx706_pwm_frames import validate_frame_options, check_frame_reference
        result.frame_options = validate_frame_options(training_options['frame_options'], symbol_widths)
        result.frame_check = check_frame_reference(frames, training_options['frame_options'],
                                                  symbol_widths, result.symbol_bitwords, cancel_event)
        result.reference_check = result.frame_check
        if not result.decode_locked:
            result.frame_check.locked = False
            result.frame_check.reason = result.decode_note
    return result


def recover_pwm2(spectrum, *, total_chips=20, prefix_chips=8, guard_chips=2, **options):
    """Configurable fixed-high + thermometer-coded + fixed-low PWM protocol."""
    if options.get('pulse_options') is None:
        pwm2_layout(total_chips, prefix_chips, guard_chips)
    return recover_pwm(spectrum, _version=2, _total_chips=total_chips,
                       _prefix_chips=prefix_chips, guard_chips=guard_chips, **options)

"""Bounded offline recovery of fixed-state CW and periodic two-state waveforms.

The clock is inferred from the recording. State A/B agreement is not measured BER.
No SciPy dependency, board access, per-state phase reset, or source-file writes.
"""
from dataclasses import dataclass
import math
import numpy as np
from rx706_spectrum import _cancel


@dataclass
class PeriodicRecovery:
    carrier_hz: float
    cycle_hz: float
    sidebands_hz: tuple
    sideband_snr_db: float
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
    heldout_clock_agreement: float
    scatter: object
    scatter_states: object
    waveform: object
    waveform_time_s: object
    power_time_s: object
    power_dbfs: object

    @property
    def kind(self):
        return 'periodic'


@dataclass
class ToneRecovery:
    carrier_hz: float
    output_rate_hz: float
    cutoff_hz: float
    decimation: int
    first_tuple: int
    input_samples: int
    clipped: int
    centers: object
    phase_rms_deg: float
    residual_frequency_hz: float
    spectral_peak_margin_db: float
    phase_tracking: bool
    phase_rms_before_tracking_deg: float
    scatter: object
    waveform: object
    waveform_time_s: object
    power_time_s: object
    power_dbfs: object

    @property
    def kind(self):
        return 'tone'


def _read_span(spectrum, duration_s, center_hint_hz, cancel_event, *, max_samples=1 << 22,
               allow_clipped=False):
    _cancel(cancel_event)
    fs = spectrum.sample_rate_hz
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or not .001 <= duration_s <= 1:
        raise ValueError('恢复时长必须为 1～1000 ms')
    if center_hint_hz is not None and (not math.isfinite(float(center_hint_hz)) or abs(float(center_hint_hz)) >= fs/2):
        raise ValueError('载波中心提示必须在基带采样范围内')
    size = spectrum.total_tuples*8
    count = min(round(duration_s*fs), spectrum.total_tuples-spectrum.first_tuple, max_samples)
    with spectrum.path.open('rb') as stream:
        stream.seek(0, 2)
        if stream.tell() != size:
            raise ValueError('恢复前文件长度发生变化')
        stream.seek(spectrum.first_tuple*8)
        raw = np.fromfile(stream, dtype='<i2', count=count*4)
        if raw.size != count*4:
            raise ValueError('恢复时文件被截短')
        stream.seek(0, 2)
        if stream.tell() != size:
            raise ValueError('恢复期间文件长度发生变化')
    iq = raw.reshape(-1, 4)[:, (spectrum.channel-1)*2:spectrum.channel*2]
    invalid = (iq < -2048) | (iq > 2047)
    if np.any(invalid):
        row, component = np.argwhere(invalid)[0]
        raise ValueError(f'恢复片段 RX{spectrum.channel} 采样点 {spectrum.first_tuple+row} 的 {"IQ"[component]} 超出 12 位范围')
    clipped = int(np.count_nonzero(np.any((iq <= -2048) | (iq >= 2047), axis=1)))
    if clipped/count > .001 and not allow_clipped:
        raise ValueError(f'恢复片段削顶 {clipped/count:.2%}，请先降低输入电平或接收增益')
    return iq[:, 0].astype(float)+1j*iq[:, 1].astype(float), count, clipped


def _lowpass(x, fs, cutoff, factor, margin, cancel_event, phase=None):
    grid = np.arange(129)-64
    h = 2*cutoff/fs*np.sinc(2*cutoff/fs*grid)*np.blackman(129)
    h /= h.sum()
    zs, indices = [], []
    for start in range(0, len(x), 262144):
        _cancel(cancel_event)
        end = min(start+262144, len(x))
        lo, hi = max(0, start-64), min(len(x), end+64)
        z = x[lo:hi]
        if phase is not None:
            z = z*np.exp(-1j*phase(np.arange(lo, hi)/fs))
        nconv = 1 << (len(z)+127).bit_length()
        filtered = np.fft.ifft(np.fft.fft(z, nconv)*np.fft.fft(h, nconv))[64:64+len(z)]
        first, last = max(start, margin), min(end, len(x)-margin)
        first += (-first) % factor
        take = np.arange(first, last, factor)
        zs.append(filtered[take-lo]); indices.append(take)
    return np.concatenate(zs), np.concatenate(indices)


def recover_tone(spectrum, *, duration_s=.2, center_hint_hz=None, cutoff_hz=10000, track_phase=True,
                 cancel_event=None):
    """Fit one constant CFO/initial phase to fixed-state CW, preserving residuals.

    Linear fit uses first-half 1ms phasors. Optional symmetric 11ms phase tracking
    removes slow drift under the fixed-state assumption, not individual sample
    angles. Frequency-only mode preserves slow physical phase changes as well.
    Neither mode establishes absolute phase across separate capture files.
    """
    fs = spectrum.sample_rate_hz
    cutoff_hz = float(cutoff_hz)
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < fs/8:
        raise ValueError('单音低通截止必须大于 0 且小于原始采样率的 1/8')
    x, count, clipped = _read_span(spectrum, duration_s, center_hint_hz, cancel_event)
    if count/fs < max(.008, 128/cutoff_hz):
        raise ValueError('所选单音低通带宽需要更长片段；请增加恢复时长或增大截止频率')
    block = max(256, round(fs*.001))
    if count < 8*block:
        raise ValueError('单音恢复至少需要约 8 ms 数据，请增加恢复时长')
    nf = min(262144, 1 << (count.bit_length()-1))
    w = np.hanning(nf); ps = np.zeros(nf)
    for start in np.linspace(0, count-nf, min(12, max(1, count//nf)), dtype=int):
        _cancel(cancel_event)
        ps += abs(np.fft.fft(x[start:start+nf]*w))**2
    freq = np.fft.fftshift(np.fft.fftfreq(nf, 1/fs)); ps = np.fft.fftshift(ps)
    df = fs/nf
    allowed = (abs(freq) > 2*df) & (abs(freq) < fs/2-2*df)
    if center_hint_hz is not None:
        allowed &= abs(freq-float(center_hint_hz)) < max(200000, 100*df)
    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        raise ValueError('单音载波搜索范围无有效频点')
    peak = int(candidates[np.argmax(ps[candidates])])
    neighborhood = (abs(freq-freq[peak]) < max(50000, 100*df)) & (abs(freq-freq[peak]) > 5*df)
    noise = max(float(np.median(ps[neighborhood])), 1e-30)
    if 10*np.log10(max(ps[peak], 1e-30)/noise) < 12:
        raise ValueError('没有足够明显的单音载波，请检查频率提示和输入信号')
    other = allowed & (abs(freq-freq[peak]) > 8*df)
    margin_db = float(10*np.log10(max(ps[peak], 1e-30)/max(float(ps[other].max()), 1e-30)))
    if margin_db < 6:
        raise ValueError('存在强度相近的多个谱峰；切换信号请使用周期方波恢复，避免把边带误当单音载波')
    p = np.log(np.maximum(ps[peak-1:peak+2], 1e-30))
    denom = p[0]-2*p[1]+p[2]
    delta = .5*(p[0]-p[2])/denom if abs(denom) > 1e-12 else 0.
    carrier = float(freq[peak]+np.clip(delta, -.5, .5)*df)
    times, phasors = [], []
    w = np.hanning(block); local = np.arange(block)/fs
    for start in range(0, count-block+1, block):
        _cancel(cancel_event)
        t = start/fs+local
        phasors.append(np.sum(x[start:start+block]*w*np.exp(-2j*np.pi*carrier*t))/w.sum())
        times.append((start+(block-1)/2)/fs)
    times = np.asarray(times); phasors = np.asarray(phasors)
    phi = np.unwrap(np.angle(phasors))
    train = times < count/fs/2
    valid = train & (abs(phasors) > np.median(abs(phasors))*.25)
    weights = np.minimum(abs(phasors), np.percentile(abs(phasors), 75))
    for iteration in range(3):
        if np.count_nonzero(valid) < 4:
            raise ValueError('单音相位拟合点不足，可能存在深衰落')
        fit = np.polyfit(times[valid], phi[valid], 1, w=weights[valid])
        residual = phi-np.polyval(fit, times)
        mad = np.median(abs(residual[valid]-np.median(residual[valid])))
        valid = train & (abs(residual) < max(.1, 5*mad)) & (abs(phasors) > np.median(abs(phasors))*.25)
    measured = carrier+fit[0]/(2*np.pi)
    test = ~train
    residual_frequency = float(np.polyfit(times[test], residual[test], 1)[0]/(2*np.pi))
    phase_before = float(np.rad2deg(np.sqrt(np.mean(np.angle(np.exp(1j*residual))**2))))
    if track_phase:
        length = min(11, (len(residual)//2)*2-1)
        weight = np.hanning(length); weight/=weight.sum()
        slow = np.convolve(np.pad(residual,(length//2,length//2),mode='edge'),weight,mode='valid')
        phase = lambda t: 2*np.pi*measured*t+fit[1]+np.interp(t,times,slow)
    else:
        phase = lambda t: 2*np.pi*measured*t+fit[1]
    # Multistage anti-alias filtering: never directly decimate RF-offset samples.
    stages=[]; output_rate=fs; total_factor=1
    while output_rate/2 >= 8*cutoff_hz:
        factor=2
        while factor < 8 and output_rate/(factor*2) >= 8*cutoff_hz:
            factor*=2
        stages.append(factor); output_rate/=factor; total_factor*=factor
    z=x; rate=fs; start_time=0.
    for index,factor in enumerate(stages):
        z, take = _lowpass(z, rate, .30*rate/factor, factor, 128, cancel_event,
                          phase=phase if index == 0 else None)
        start_time += take[0]/rate; rate/=factor
    z, take = _lowpass(z, rate, cutoff_hz, 1, 128, cancel_event,
                      phase=phase if not stages else None)
    start_time += take[0]/rate
    t = start_time+np.arange(len(z))/rate
    center = np.mean(z[t < count/fs/2])
    amplitude_gate = abs(z) > np.median(abs(z))*.25
    phase_error = np.angle(z[amplitude_gate]*center.conjugate())
    phase_rms = float(np.rad2deg(np.sqrt(np.mean(phase_error**2))))
    rng=np.random.default_rng(9361)
    selected=np.sort(rng.choice(len(z),min(8192,len(z)),replace=False))
    view_count=min(4096,round(rate*.01))
    power_step=max(1,round(rate*.001)); starts=np.arange(0,len(z),power_step)
    power=np.add.reduceat(abs(z)**2,starts)/np.diff(np.r_[starts,len(z)])
    power_times=t[np.minimum(starts+power_step//2,len(z)-1)]+spectrum.first_tuple/fs
    _cancel(cancel_event)
    return ToneRecovery(float(measured), rate, cutoff_hz, total_factor, spectrum.first_tuple,
        count, clipped, np.array([center]), phase_rms, residual_frequency, margin_db, bool(track_phase), phase_before,
        z[selected], z[:view_count], t[:view_count]+spectrum.first_tuple/fs,
        power_times, 10*np.log10(np.maximum(power/2048**2,1e-18)))


def recover_periodic(spectrum, *, cycle_hz=100000, duration_s=.2,
                     center_hint_hz=None, cancel_event=None):
    """Recover a balanced periodic square wave via its two first sidebands.

    The sideband pair and common phase are fitted over the selected recovery span.
    Clock phase and centroids use the first half, with agreement on the second.
    Bounded to 2^22 input tuples and <=8192 scatter points.
    """
    _cancel(cancel_event)
    fs = spectrum.sample_rate_hz
    cycle_hz, duration_s = float(cycle_hz), float(duration_s)
    if not math.isfinite(cycle_hz) or not 0 < cycle_hz < fs/8:
        raise ValueError('完整周期频率必须大于 0，且每周期至少需要 8 个原始采样点')
    if not math.isfinite(duration_s) or not .001 <= duration_s <= 1:
        raise ValueError('恢复时长必须为 1～1000 ms')
    if center_hint_hz is not None:
        center_hint_hz = float(center_hint_hz)
        if not math.isfinite(center_hint_hz) or abs(center_hint_hz) >= fs/2:
            raise ValueError('载波中心提示必须在基带采样范围内')
    size = spectrum.total_tuples*8
    count = min(round(duration_s*fs), spectrum.total_tuples-spectrum.first_tuple, 1 << 22)
    block = max(round(fs*.001), round(fs*16/cycle_hz))
    if count < 6*block:
        raise ValueError('恢复片段太短：至少需要 6 个同步窗（每窗至少 1 ms、16 个周期）；请增加恢复时长')
    with spectrum.path.open('rb') as stream:
        stream.seek(0, 2)
        if stream.tell() != size:
            raise ValueError('恢复前文件长度发生变化')
        stream.seek(spectrum.first_tuple*8)
        raw = np.fromfile(stream, dtype='<i2', count=count*4)
        if raw.size != count*4:
            raise ValueError('恢复时文件被截短')
        stream.seek(0, 2)
        if stream.tell() != size:
            raise ValueError('恢复期间文件长度发生变化')
    iq = raw.reshape(-1, 4)[:, (spectrum.channel-1)*2:spectrum.channel*2]
    invalid = (iq < -2048) | (iq > 2047)
    if np.any(invalid):
        row, component = np.argwhere(invalid)[0]
        raise ValueError(f'恢复片段 RX{spectrum.channel} 采样点 {spectrum.first_tuple+row} 的 {"IQ"[component]} 超出 12 位范围')
    clipped = int(np.count_nonzero(np.any((iq <= -2048) | (iq >= 2047), axis=1)))
    if clipped/count > .001:
        raise ValueError(f'恢复片段削顶 {clipped/count:.2%}，请先降低输入电平或接收增益')
    x = iq[:, 0].astype(float)+1j*iq[:, 1].astype(float)
    del raw, iq, invalid
    _cancel(cancel_event)
    nf = min(262144, 1 << (count.bit_length()-1))
    df = fs/nf
    if cycle_hz < 8*df:
        raise ValueError('当前模式的频谱分辨率不足以分开这两条边带；请使用更高周期频率或专门的低速分析')
    w = np.hanning(nf)
    ps = np.zeros(nf)
    for start in np.linspace(0, count-nf, min(12, max(1, count//nf)), dtype=int):
        _cancel(cancel_event)
        ps += abs(np.fft.fft(x[start:start+nf]*w))**2
    freq = np.fft.fftshift(np.fft.fftfreq(nf, 1/fs))
    ps = np.fft.fftshift(ps)
    candidates = np.flatnonzero((ps > np.roll(ps, 1)) & (ps >= np.roll(ps, -1)))
    candidates = candidates[(candidates > 1) & (candidates < nf-2) & (abs(freq[candidates]) > 2*df)]
    candidates = candidates[np.argsort(ps[candidates])[::-1]][:256]
    tolerance = max(3*df, .04*cycle_hz)
    best = None
    for low in candidates:
        upper = candidates[abs(freq[candidates]-freq[low]-2*cycle_hz) < tolerance]
        for high in upper:
            center = (freq[low]+freq[high])/2
            if center_hint_hz is not None and abs(center-center_hint_hz) > max(.3*cycle_hz, 8*df):
                continue
            score = min(ps[low], ps[high])
            if best is None or score > best[0]:
                best = (score, int(low), int(high))
    if best is None:
        raise ValueError('没有找到符合完整周期频率的边带对；请核对周期、载波中心提示或输入信号')
    _, low, high = best
    neighborhood = (freq > freq[low]-2*cycle_hz) & (freq < freq[high]+2*cycle_hz)
    noise = float(np.median(ps[neighborhood]))
    snr_db = float(10*np.log10(min(ps[low], ps[high])/max(noise, 1e-30)))
    if snr_db < 12:
        raise ValueError(f'候选边带相对局部谱底仅 {snr_db:.1f} dB，无法可靠锁定；保留原始视图')
    def refine(k):
        p = np.log(np.maximum(ps[k-1:k+2], 1e-30))
        denom = p[0]-2*p[1]+p[2]
        d = .5*(p[0]-p[2])/denom if abs(denom) > 1e-12 else 0.
        return float(freq[k]+np.clip(d, -.5, .5)*df)
    lower, upper = refine(low), refine(high)
    carrier, clock = (upper+lower)/2, (upper-lower)/2
    w = np.hanning(block); times=[]; positive=[]; negative=[]
    local = np.arange(block)/fs
    for start in range(0, count-block+1, block):
        _cancel(cancel_event)
        t = start/fs+local
        z = x[start:start+block]*w
        positive.append(np.sum(z*np.exp(-2j*np.pi*upper*t))/w.sum())
        negative.append(np.sum(z*np.exp(-2j*np.pi*lower*t))/w.sum())
        times.append((start+(block-1)/2)/fs)
    times = np.asarray(times)
    positive, negative = np.asarray(positive), np.asarray(negative)
    if min(np.percentile(abs(positive), 10), np.percentile(abs(negative), 10)) < 1e-8:
        raise ValueError('边带衰落，无法稳定恢复公共相位')
    pp, pn = np.unwrap(np.angle(positive)), np.unwrap(np.angle(negative))
    common, differential = (pp+pn)/2, (pp-pn)/2
    clock_fit = np.polyfit(times, differential, 1)
    carrier_fit = np.polyfit(times, common, 1)
    measured_clock = clock+clock_fit[0]/(2*np.pi)
    measured_carrier = carrier+carrier_fit[0]/(2*np.pi)
    cutoff = min(7*measured_clock, .35*fs)
    # Leave the fixed-length Blackman FIR a transition band before output Nyquist.
    target_rate = max(32*measured_clock, 2.6*cutoff, 2*(cutoff+3*fs/128))
    decimation = 1
    while fs/(2*decimation) >= target_rate and decimation < 64:
        decimation *= 2
    grid = np.arange(129)-64
    h = 2*cutoff/fs*np.sinc(2*cutoff/fs*grid)*np.blackman(129)
    h /= h.sum()
    margin = max(128, block//2)
    zs=[]; indices=[]
    # Overlapping chunks: symmetric FIR output with explicit edge removal.
    for start in range(0, count, 262144):
        _cancel(cancel_event)
        end = min(start+262144, count)
        lo, hi = max(0, start-64), min(count, end+64)
        t = np.arange(lo, hi)/fs
        z = x[lo:hi]*np.exp(-1j*(2*np.pi*carrier*t+np.interp(t, times, common)))
        nconv = 1 << (len(z)+127).bit_length()
        filtered = np.fft.ifft(np.fft.fft(z, nconv)*np.fft.fft(h, nconv))[64:64+len(z)]
        first, last = max(start, margin), min(end, count-margin)
        first += (-first) % decimation
        take = np.arange(first, last, decimation)
        zs.append(filtered[take-lo]); indices.append(take)
    z = np.concatenate(zs); index = np.concatenate(indices); t = index/fs
    theta = 2*np.pi*clock*t+np.polyval(clock_fit, t)
    train = t < count/fs/2
    best = None
    # Subsample training points only to bound phase-search work; metrics use all.
    stride = max(1, int(np.sum(train))//30000)
    zt, th = z[train][::stride], theta[train][::stride]
    for offset in np.linspace(0, np.pi, 121, endpoint=False):
        _cancel(cancel_event)
        phase = (th-offset) % (2*np.pi)
        state = phase >= np.pi
        interior = (phase % np.pi > .2*np.pi) & (phase % np.pi < .8*np.pi)
        a, b = zt[interior & ~state], zt[interior & state]
        if not len(a) or not len(b):
            continue
        ca, cb = a.mean(), b.mean()
        spread = np.sqrt((np.mean(abs(a-ca)**2)+np.mean(abs(b-cb)**2))/2)
        score = abs(cb-ca)/max(spread, 1e-12)
        if best is None or score > best[0]:
            best = (float(score), float(offset), ca, cb)
    if best is None or best[0] < 4:
        raise ValueError('移频后仍未形成足够分离的两态；请查看原始信号并核对周期')
    separation, offset, ca, cb = best
    phase = (theta-offset) % (2*np.pi)
    state = phase >= np.pi
    interior = (phase % np.pi > .2*np.pi) & (phase % np.pi < .8*np.pi)
    predicted = abs(z-cb) < abs(z-ca)
    test = ~train & interior
    agreement = float(np.mean(predicted[test] == state[test]))
    if agreement < .9:
        raise ValueError(f'后半段与恢复周期的符合率仅 {agreement:.1%}，两态锁定不稳定')
    # One common rotation aligns the separation axis. State-dependent phases remain.
    rotation = np.exp(-1j*np.angle(cb-ca))
    z *= rotation; ca *= rotation; cb *= rotation
    labels = np.where(interior, state.astype(int), -1)
    rng = np.random.default_rng(706)
    scatter_index = np.sort(rng.choice(len(z), min(8192, len(z)), replace=False))
    view_count = min(4096, int(np.ceil(10*fs/decimation/measured_clock)))
    power_step = max(1, round(fs/decimation*.001))
    starts = np.arange(0, len(z), power_step)
    power = np.add.reduceat(abs(z)**2, starts)/np.diff(np.r_[starts, len(z)])
    power_times = t[np.minimum(starts+power_step//2, len(t)-1)]
    _cancel(cancel_event)
    return PeriodicRecovery(float(measured_carrier), float(measured_clock), (lower, upper),
        snr_db, fs/decimation, cutoff, decimation, spectrum.first_tuple, count, clipped,
        np.array([ca, cb]), float(np.angle(cb*ca.conjugate(), deg=True)),
        float(20*np.log10(max(abs(cb), 1e-20)/max(abs(ca), 1e-20))), separation,
        agreement, z[scatter_index], labels[scatter_index], z[:view_count],
        (index[:view_count]+spectrum.first_tuple)/fs,
        power_times+spectrum.first_tuple/fs, 10*np.log10(np.maximum(power/2048**2, 1e-18)))

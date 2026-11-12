"""Offline square QAM decisions; blind phase is ambiguous modulo 90 degrees."""
from types import SimpleNamespace
import math
import numpy as np
from rx706_recovery import _read_span
from rx706_spectrum import _cancel


def decide(z, order):
    side = math.isqrt(order)
    scale = math.sqrt(2*(order-1)/3)
    i = np.clip(np.rint((z.real*scale+side-1)/2), 0, side-1).astype(int)
    q = np.clip(np.rint((z.imag*scale+side-1)/2), 0, side-1).astype(int)
    ideal = ((2*i-side+1)+1j*(2*q-side+1))/scale
    # Gray labels along increasing I, then increasing Q; MSB first.
    labels = ((i ^ (i >> 1)) << int(math.log2(side))) | (q ^ (q >> 1))
    bits = ((labels[:, None] >> np.arange(int(math.log2(order))-1, -1, -1)) & 1).astype(np.uint8)
    return ideal, labels, bits


def decision_margins(z,order):
    """Geometric distance to the nearest square-QAM decision boundary, not BER."""
    side=math.isqrt(order);scale=math.sqrt(2*(order-1)/3)
    boundaries=np.arange(2-side,side-1,2)/scale
    def distance(values):
        index=np.searchsorted(boundaries,values)
        lower=np.where(index>0,abs(values-boundaries[np.clip(index-1,0,len(boundaries)-1)]),np.inf)
        upper=np.where(index<len(boundaries),abs(values-boundaries[np.clip(index,0,len(boundaries)-1)]),np.inf)
        return np.minimum(lower,upper)
    values=np.asarray(z)
    return np.minimum(distance(values.real),distance(values.imag))


def recover_qam(spectrum, *, duration_s=.2, center_hint_hz=0, symbol_hz=100000,
                order=256, rolloff=.35, pulse='RRC', preamble_options=None,
                pwm_preamble_options=None, tdcm_options=None, cancel_event=None):
    if tdcm_options is not None:
        from rx706_tdcm import recover_tdcm
        return recover_tdcm(spectrum,duration_s=duration_s,center_hint_hz=center_hint_hz,
            symbol_hz=symbol_hz,order=order,tdcm_options=tdcm_options,
            pwm_preamble_options=pwm_preamble_options,preamble_options=preamble_options,
            cancel_event=cancel_event)
    if preamble_options is not None and (preamble_options.get('orientation','normal')!='normal' or preamble_options.get('fit_dc_offset',False)):
        raise ValueError('镜像/零点训练校准目前用于TDCM谐波QAM路径')
    fs = spectrum.sample_rate_hz
    if order not in (16, 64, 256):
        raise ValueError('QAM 阶数须为16、64或256')
    if not math.isfinite(symbol_hz) or not 0 < symbol_hz <= fs/2:
        raise ValueError('QAM符号率须大于0且每符号至少2个采样点')
    if not math.isfinite(rolloff) or not 0 < rolloff <= 1:
        raise ValueError('RRC滚降系数须在(0,1]内')
    if center_hint_hz is None:
        raise ValueError('QAM请填写基带载波频偏MHz（已居中填0）；不能用频谱主峰代替载波')
    if abs(center_hint_hz)+(1+rolloff)*symbol_hz/2 >= fs/2:
        raise ValueError('QAM带宽超出采样范围')
    from rx706_qam_preamble import validate_preamble, acquire_preamble, _sample
    settings = None if preamble_options is None else validate_preamble(preamble_options, order, symbol_hz)
    pwm_bound = 0.
    if pwm_preamble_options is not None:
        from rx706_qam_pwm import validate_pwm_header
        pwm_bound = validate_pwm_header(pwm_preamble_options, fs)['max_frequency_hz']
    if abs(center_hint_hz)+pwm_bound+(0 if settings is None else settings[1])+(1+rolloff)*symbol_hz/2 >= fs/2:
        raise ValueError('包含残余频偏搜索范围的QAM带宽超出采样范围')
    x, count, clipped = _read_span(spectrum, duration_s, center_hint_hz, cancel_event,
                                  max_samples=math.ceil(fs))
    sps = fs/symbol_hz
    if count/sps < 256:
        raise ValueError('QAM分析至少需要256个符号')
    baseband = x*np.exp(-2j*np.pi*center_hint_hz*np.arange(count)/fs)
    pwm_sync = None
    qam_start = 0.
    pwm_frequency = 0.
    if pwm_preamble_options is not None:
        from rx706_qam_pwm import acquire_pwm_header
        pwm_sync = acquire_pwm_header(baseband, fs, pwm_preamble_options, cancel_event)
        qam_start = pwm_sync['qam_index']
        pwm_frequency = pwm_sync['frequency_hz']
        if count-qam_start < 256*sps:
            raise ValueError('PWM前导和切换间隔后不足256个QAM符号，请增加恢复时长或检查切换间隔')
        baseband *= np.exp(-2j*np.pi*pwm_frequency*np.arange(count)/fs)
        # Do not convolve PWM levels into the first QAM training symbols.
        baseband[:int(math.ceil(qam_start))] = 0
        origin = spectrum.first_tuple/fs
        pwm_sync.update(time_s=origin+pwm_sync['index']/fs,
                        end_time_s=origin+pwm_sync['end_index']/fs,
                        qam_time_s=origin+qam_start/fs)
        del x
    # Preserve QAM's amplitude information; no CW phase cancellation or DC fit.
    if pulse == 'RRC':
        radius = int(math.ceil(8*sps))
        if radius > 65536:
            raise ValueError('每符号采样点过多，请提高符号率或先抽取')
        t = np.arange(-radius, radius+1)/sps
        h = np.empty_like(t)
        zero = abs(t) < 1e-10
        singular = abs(abs(4*rolloff*t)-1) < 1e-8
        regular = ~(zero | singular)
        u = t[regular]
        h[regular] = (np.sin(np.pi*u*(1-rolloff))+4*rolloff*u*np.cos(np.pi*u*(1+rolloff)))/(np.pi*u*(1-(4*rolloff*u)**2))
        h[zero] = 1+rolloff*(4/np.pi-1)
        h[singular] = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff))+(1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        h /= np.sqrt(np.sum(h*h))
        margin = radius
    elif pulse == 'Rectangular':
        width = max(1, round(sps))
        h = np.ones(width)/width
        margin = width
    else:
        raise ValueError('未知QAM脉冲形状')

    def matched(signal):
        _cancel(cancel_event)
        if pulse == 'Rectangular':
            return np.convolve(signal, h, mode='same')
        n = 1 << (len(signal)+len(h)-2).bit_length()
        return np.fft.ifft(np.fft.fft(signal,n)*np.fft.fft(h,n))[radius:radius+count]

    x = matched(baseband)
    if settings is not None:
        points, bound, tolerance, search_s = settings
        sync = acquire_preamble(x, fs, symbol_hz, margin, points, bound,
                                tolerance, search_s, decide, order, cancel_event,
                                start_index=qam_start,
                                start_window_symbols=None if pwm_sync is None else pwm_sync['qam_search_symbols'])
        frequency = sync['frequency_hz']
        # Residual CFO must be removed BEFORE matching to avoid filter distortion.
        x = matched(baseband * np.exp(-2j*np.pi*frequency*np.arange(count)/fs))
        pilot_indices = sync['index'] + np.arange(len(points))*sps
        pilots = _sample(x, pilot_indices)
        gain = np.vdot(points, pilots)/np.vdot(points, points)
        if abs(gain) < 1e-12:
            raise ValueError('QAM前导幅度不足')
        sync['evm_percent'] = float(100*np.sqrt(np.mean(abs(pilots/gain-points)**2)/np.mean(abs(points)**2)))
        if sync['evm_percent'] > tolerance:
            raise ValueError('频偏校正后的QAM前导未通过匹配容差')
        indices = np.arange(sync['index']+len(points)*sps, count-margin-1, sps)
        z = _sample(x, indices)/gain
        ideal, labels, bits = decide(z, order)
        sync.update(time_s=(sync['index']+spectrum.first_tuple)/fs,
                    payload_time_s=(indices[0]+spectrum.first_tuple)/fs,
                    phase_deg=float(np.angle(gain, deg=True)),
                    total_frequency_hz=center_hint_hz+pwm_frequency+frequency)
        return SimpleNamespace(kind='qam', carrier_hz=center_hint_hz, order=order,
            symbol_hz=symbol_hz, scatter=z, ideal=ideal, labels=labels, bits=bits,
            evm_percent=100*np.sqrt(np.mean(abs(z-ideal)**2)/np.mean(abs(ideal)**2)),
            time_s=(indices+spectrum.first_tuple)/fs, preamble=sync, pwm_preamble=pwm_sync,
            first_tuple=spectrum.first_tuple, input_samples=count, clipped=clipped,
            cutoff_hz=(1+rolloff)*symbol_hz/2, decimation=1, output_rate_hz=fs)
    best = None
    # Joint fractional timing and static phase search, bounded calibration span.
    for offset in np.linspace(0, sps, 32, endpoint=False):
        _cancel(cancel_event)
        indices = np.arange((margin if pwm_sync is None else qam_start)+offset, count-margin-1, sps)
        if len(indices) < 256:
            continue
        z = np.interp(indices, np.arange(count), x.real)+1j*np.interp(indices, np.arange(count), x.imag)
        sample = z[:2048]
        sample /= np.sqrt(np.mean(abs(sample)**2))
        for phase in np.linspace(0, np.pi/2, 181, endpoint=False):
            trial = sample*np.exp(-1j*phase)
            ideal, _, _ = decide(trial, order)
            error = np.mean(abs(trial-ideal)**2)
            if best is None or error < best[0]:
                best = (error, indices, phase)
    if best is None:
        raise ValueError('滤波后符号不足256个')
    _, indices, phase = best
    z = np.interp(indices, np.arange(count), x.real)+1j*np.interp(indices, np.arange(count), x.imag)
    z /= np.sqrt(np.mean(abs(z)**2))
    z *= np.exp(-1j*phase)
    ideal, labels, bits = decide(z, order)
    evm = 100*np.sqrt(np.mean(abs(z-ideal)**2)/np.mean(abs(ideal)**2))
    return SimpleNamespace(kind='qam', carrier_hz=center_hint_hz, order=order,
        symbol_hz=symbol_hz, scatter=z, ideal=ideal, labels=labels, bits=bits,
        evm_percent=evm, time_s=(indices+spectrum.first_tuple)/fs,
        first_tuple=spectrum.first_tuple, input_samples=count, clipped=clipped,
        preamble=None, pwm_preamble=pwm_sync,
        cutoff_hz=(1+rolloff)*symbol_hz/2, decimation=1, output_rate_hz=fs)

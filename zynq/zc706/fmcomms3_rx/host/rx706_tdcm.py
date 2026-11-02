"""PWM-framed TDCM QAM: integrate a selected harmonic over each symbol.

The raw RF carrier is removed before harmonic extraction. Fractional sample
boundaries use exact cell integrals, so a DC carrier cancels over full cycles.
"""
from types import SimpleNamespace
import math
import numpy as np
from rx706_recovery import _read_span
from rx706_spectrum import _cancel
from rx706_qam_pwm import acquire_pwm_header, validate_pwm_header
from rx706_qam_preamble import _sample, validate_preamble


def validate_tdcm(fs, symbol_hz, options):
    rate = float(symbol_hz)
    base = float(options.get('base_hz', 19531.25))
    harmonic = float(options.get('harmonic', 1))
    amplitude = float(options.get('amplitude_scale', 1))
    phase = float(options.get('phase_deg', 0))
    count = float(options.get('payload_symbols', 0))
    normalization = options.get('normalization', 'pwm')
    if not math.isfinite(rate) or not 0 < rate < fs/2:
        raise ValueError('TDCM符号率须为有限正数且小于采样率一半')
    if not math.isfinite(base) or not 0 < base < fs/2:
        raise ValueError('TDCM谐波基频须为有限正数且小于采样率一半')
    if not math.isfinite(harmonic) or not harmonic.is_integer() or not 1 <= abs(harmonic) <= 32:
        raise ValueError('TDCM谐波阶数须为±1～±32的整数，不能为0')
    cycles = base/rate
    if cycles < 1 or abs(cycles-round(cycles)) > 1e-8:
        raise ValueError('TDCM每个QAM符号须包含整数个调制周期；请核对符号率与谐波基频')
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('TDCM幅度系数须为有限正数')
    if not math.isfinite(phase):
        raise ValueError('TDCM相位修正须为有限角度')
    if not math.isfinite(count) or not count.is_integer() or not 0 <= count <= 1000000:
        raise ValueError('TDCM数据符号数须为0～1000000的整数；0表示处理到文件尾')
    if normalization not in ('pwm', 'blind'):
        raise ValueError('TDCM归一化须为PWM两态参考或盲归一化')
    return dict(base_hz=base, harmonic=int(harmonic), amplitude_scale=amplitude,
                phase_deg=phase, payload_symbols=int(count), normalization=normalization,
                cycles_per_symbol=int(round(cycles)))


def harmonic_coefficients(x, fs, start, symbol_hz, harmonic_hz, *, count=0, cancel_event=None):
    """Integrate zero-order-held input cells with a continuous complex kernel.

    Returns complex Fourier coefficients and original-file-relative sample
    centres. Complete windows only; no rounding of samples per symbol.
    """
    sps = fs/symbol_hz
    available = max(0, int(math.floor((len(x)-start)/sps+1e-10)))
    if count > available:
        raise ValueError(f'TDCM数据不足：配置{count}符号，片段内只有{available}个完整符号')
    count = count or available
    if count < 1:
        raise ValueError('PWM前导和切换间隔后没有完整TDCM符号')
    coefficients = np.empty(count, complex)
    f = harmonic_hz/fs
    block_symbols = max(1, min(256, int(262144/sps)))
    for first in range(0, count, block_symbols):
        _cancel(cancel_event)
        last = min(count, first+block_symbols)
        edges = start+np.arange(first,last+1)*sps
        lo, hi = int(math.floor(edges[0])), int(math.ceil(edges[-1]))
        hi = min(hi,len(x))
        z = x[lo:hi]
        anchor = edges[0]
        positions = lo+np.arange(len(z))
        kernel = np.sinc(f)*np.exp(-2j*np.pi*f*(positions+.5-anchor))
        integral = np.r_[0j,np.cumsum(z*kernel)]
        lower = np.floor(edges).astype(np.int64)
        fraction = edges-lower
        values = integral[np.minimum(lower-lo,len(z))].copy()
        partial = (fraction > 1e-12) & (lower < hi)
        a = fraction[partial]
        values[partial] += z[lower[partial]-lo]*a*np.sinc(f*a)*np.exp(
            -2j*np.pi*f*(lower[partial]+a/2-anchor))
        # Each coefficient uses its own symbol-start phase convention.
        coefficients[first:last] = np.diff(values)/sps*np.exp(2j*np.pi*f*(edges[:-1]-anchor))
    return coefficients, start+(np.arange(count)+.5)*sps


def _pwm_reference(x, fs, sync, options, cancel_event):
    """Fit a common CFO and two stationary complex states over all pilot cells."""
    period = fs/options['symbol_hz']
    count = int(round(len(options['symbols'])*period))
    indices = sync['index']+np.arange(count)
    z = _sample(x,indices)
    high = np.zeros(count,bool)
    for i,k in enumerate(options['symbols']):
        a = round(i*period)
        b = round((i+(options['prefix_chips']+k+options['width_offset_chips'])/options['total_chips'])*period)
        high[a:b] = True
    valid = np.ones(count,bool)
    reach = max(2,int(.2*period/options['total_chips']))
    for at in np.r_[0,np.flatnonzero(np.diff(high))+1,count]:
        valid[max(0,at-reach):min(count,at+reach+1)] = False
    times = np.arange(count)[valid]/fs
    sample, state = z[valid],high[valid]
    if min(np.count_nonzero(state),np.count_nonzero(~state)) < 3:
        raise ValueError('PWM稳定高/低片段不足，不能校准TDCM')

    def fit(frequency):
        _cancel(cancel_event)
        v = sample*np.exp(-2j*np.pi*frequency*times)
        low, high_level = np.mean(v[~state]),np.mean(v[state])
        error = float(np.mean(abs(v-np.where(state,high_level,low))**2))
        return error,low,high_level

    grid = np.linspace(-options['max_frequency_hz'],options['max_frequency_hz'],65)
    errors = [fit(f)[0] for f in grid]
    best = int(np.argmin(errors))
    left,right = grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(28):
        a,b = right-ratio*(right-left),left+ratio*(right-left)
        if fit(a)[0] < fit(b)[0]:
            right = b
        else:
            left = a
    frequency = (left+right)/2
    error,low,high_level = fit(frequency)
    reference = np.exp(-2j*np.pi*frequency*sync['index']/fs)
    low,high_level = low*reference,high_level*reference
    if abs(high_level-low) < 1e-9:
        raise ValueError('PWM两态差值不足，不能校准TDCM')
    return frequency,low,high_level,100*math.sqrt(error)/abs(high_level-low)


def _coefficient_pilot(coefficients, points, rate, bound, tolerance, window, decide, order, cancel_event):
    length = len(points)
    best = None
    j = np.arange(length)
    centered = j-j.mean()
    for start in range(min(int(window)+1,len(coefficients)-length-15)):
        _cancel(cancel_event)
        ratios = coefficients[start:start+length]/points
        phase = np.unwrap(np.angle(ratios))
        frequency = float(np.dot(phase,centered)/np.sum(centered**2)*rate/(2*np.pi))
        if abs(frequency) > bound:
            continue
        v = coefficients[start:start+length]*np.exp(-2j*np.pi*frequency*j/rate)
        gain = np.vdot(points,v)/np.vdot(points,points)
        if abs(gain) < 1e-12:
            continue
        error = float(np.mean(abs(v/gain-points)**2)/np.mean(abs(points)**2))
        if error > (tolerance/100)**2:
            continue
        data = coefficients[start+length:start+length+128]/gain*np.exp(-2j*np.pi*frequency*np.arange(length,min(length+128,len(coefficients)-start))/rate)
        ideal,_,_ = decide(data,order)
        score = error+float(np.mean(abs(data-ideal)**2))
        if best is None or score < best[0]:
            best = score,start,frequency,gain,error
    if best is None:
        raise ValueError('PWM后的TDCM谐波系数未匹配配置的QAM训练序列')
    return best[1:]


def recover_tdcm(spectrum, *, duration_s=1, center_hint_hz=-200000, symbol_hz=19531.25,
                 order=256, tdcm_options=None, pwm_preamble_options=None,
                 preamble_options=None, cancel_event=None):
    from rx706_qam import decide
    fs = spectrum.sample_rate_hz
    if order not in (16,64,256):
        raise ValueError('QAM阶数须为16、64或256')
    if center_hint_hz is None or not math.isfinite(center_hint_hz):
        raise ValueError('TDCM请填写原载波基带频率；不能填写+1阶谐波频率')
    options = validate_tdcm(fs,symbol_hz,tdcm_options or {})
    pwm = validate_pwm_header(pwm_preamble_options or dict(symbol_hz=19531.25),fs)
    selected = options['harmonic']*options['base_hz']
    if (abs(center_hint_hz)+pwm['max_frequency_hz'] >= fs/2 or
            abs(center_hint_hz+selected)+pwm['max_frequency_hz']+symbol_hz/2 >= fs/2):
        raise ValueError('TDCM目标谐波带宽超出采样范围')
    training = None if preamble_options is None else validate_preamble(preamble_options,order,symbol_hz)
    x,count,clipped = _read_span(spectrum,duration_s,center_hint_hz,cancel_event,max_samples=math.ceil(fs))
    x *= np.exp(-2j*np.pi*center_hint_hz*np.arange(count)/fs)
    sync = acquire_pwm_header(x,fs,dict(pwm,symbols=' '.join(map(str,pwm['symbols']))),cancel_event)
    frequency,low,high,error = _pwm_reference(x,fs,sync,pwm,cancel_event)
    pwm_frequency = frequency
    x *= np.exp(-2j*np.pi*frequency*np.arange(count)/fs)
    start = sync['qam_index']
    coefficients,centres = harmonic_coefficients(x,fs,start,symbol_hz,selected,
        count=options['payload_symbols'],cancel_event=cancel_event)
    if len(coefficients) < 16:
        raise ValueError('TDCM至少需要16个数据符号用于显示与判决')
    original_coefficients = coefficients.copy()
    pilot = None
    phase = math.radians(options['phase_deg'])
    if training is not None:
        points,bound,tolerance,_ = training
        at,residual,_,_ = _coefficient_pilot(coefficients,points,symbol_hz,bound,tolerance,
            pwm['qam_search_symbols'],decide,order,cancel_event)
        # Re-extract from original IQ after removing the remaining frequency.
        x *= np.exp(-2j*np.pi*residual*np.arange(count)/fs)
        frequency += residual
        coefficients,centres = harmonic_coefficients(x,fs,start,symbol_hz,selected,
            count=options['payload_symbols'],cancel_event=cancel_event)
        observed = coefficients[at:at+len(points)]
        gain = np.vdot(points,observed)/np.vdot(points,points)
        if abs(gain) < 1e-12:
            raise ValueError('TDCM的QAM训练幅度不足')
        pilot_error = float(100*np.sqrt(np.mean(abs(observed/gain-points)**2)/np.mean(abs(points)**2)))
        if pilot_error > tolerance:
            raise ValueError('重新提取的TDCM QAM训练未通过匹配容差')
        first = at+len(points)
        z = coefficients[first:]/gain
        pilot = dict(length=len(points),time_s=(centres[at]+spectrum.first_tuple)/fs,
            payload_time_s=(centres[first]+spectrum.first_tuple)/fs,frequency_hz=residual,
            total_frequency_hz=center_hint_hz+frequency,phase_deg=float(np.angle(gain,deg=True)),evm_percent=pilot_error)
        coefficients,centres = coefficients[first:],centres[first:]
    elif options['normalization'] == 'pwm':
        radius = math.sqrt(2)*(math.isqrt(order)-1)/math.sqrt(2*(order-1)/3)
        gain = (high-low)/(np.pi*abs(options['harmonic'])*radius)*options['amplitude_scale']
        z = coefficients/gain*np.exp(-1j*phase)
    else:
        rms = float(np.sqrt(np.mean(abs(coefficients)**2)))
        if rms < 1e-12:
            raise ValueError('TDCM目标谐波能量不足')
        sample = coefficients[:2048]/rms/options['amplitude_scale']
        best = None
        for p in np.linspace(0,np.pi/2,181,endpoint=False):
            _cancel(cancel_event)
            trial = sample*np.exp(-1j*(p+phase))
            ideal,_,_ = decide(trial,order)
            score = float(np.mean(abs(trial-ideal)**2))
            if best is None or score < best[0]:
                best = score,p
        z = coefficients/rms/options['amplitude_scale']*np.exp(-1j*(best[1]+phase))
    ideal,labels,bits = decide(z,order)
    evm = float(100*np.sqrt(np.mean(abs(z-ideal)**2)/np.mean(abs(ideal)**2)))
    origin = spectrum.first_tuple/fs
    sync.update(time_s=origin+sync['index']/fs,end_time_s=origin+sync['end_index']/fs,
                qam_time_s=origin+start/fs,frequency_hz=pwm_frequency,error_percent=error)
    details = dict(**options,original_carrier_hz=center_hint_hz,
        corrected_carrier_hz=center_hint_hz+frequency,target_hz=center_hint_hz+frequency+selected,
        pwm_low_iq=low,pwm_high_iq=high,coefficient_units='ADC counts',
        calibration_note='已知QAM训练' if pilot is not None else
                         'PWM两态模型：假定方形QAM最远角点对应所选谐波理论最大幅度' if options['normalization']=='pwm' else
                         '盲RMS/静态相位归一化，存在90°相位歧义')
    return SimpleNamespace(kind='qam',carrier_hz=center_hint_hz,order=order,symbol_hz=symbol_hz,
        scatter=z,ideal=ideal,labels=labels,bits=bits,evm_percent=evm,
        time_s=(centres+spectrum.first_tuple)/fs,first_tuple=spectrum.first_tuple,
        input_samples=count,clipped=clipped,cutoff_hz=symbol_hz/2,decimation=1,
        output_rate_hz=symbol_hz,preamble=pilot,pwm_preamble=sync,tdcm=details,
        coefficients=coefficients,raw_coefficients=original_coefficients)

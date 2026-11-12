"""Optional TDCM compensation and diagnostics. NumPy only; never writes input.

The repeat clock estimator requires a repeated waveform, not arbitrary payload.
Carrier tracking observes the RF carrier / squared two-state waveform, never
normalizes each QAM symbol's phase. Blind grid fits remain unverified candidates.
"""
import math
from types import SimpleNamespace
import numpy as np
from rx706_spectrum import _cancel
from rx706_recovery import _read_span, _lowpass
from rx706_qam import decide
from rx706_qam_pwm import acquire_pwm_header, validate_pwm_header, header_pulse_fractions, smooth_pwm_observation
from rx706_qam_preamble import validate_preamble, _sample
from rx706_tdcm import validate_tdcm, harmonic_coefficients, _pwm_reference, _coefficient_pilot


def validate_compensation(options):
    if options is not None and not isinstance(options, dict):
        raise ValueError('TDCM补偿须为配置对象')
    o = dict(options or {})
    for flag in ('enabled','fit_timing'):
        if flag in o and not isinstance(o[flag], bool):
            raise ValueError(f'TDCM补偿 {flag} 须为布尔值')
    def number(key, default, low, high, integer=False):
        try: value = float(o.get(key, default))
        except (TypeError, ValueError): raise ValueError(f'TDCM补偿 {key} 必须为数值') from None
        if not math.isfinite(value) or not low <= value <= high or (integer and not value.is_integer()):
            raise ValueError(f'TDCM补偿 {key} 须在 {low}～{high}' + (' 内取整数' if integer else ' 内'))
        return int(value) if integer else value
    for key, choices, default in [('clock_mode',('off','manual','repeat'),'off'),
                                 ('phase_mode',('off','square','mean','pwm_headers'),'off'),
                                 ('frame_mode',('single','all'),'single')]:
        o[key] = o.get(key, default)
        if o[key] not in choices: raise ValueError(f'无效的TDCM补偿 {key}')
    o.update(period_ppm=number('period_ppm',0,-5000,5000),
             clock_search_ppm=number('clock_search_ppm',200,1,5000),
             phase_update_hz=number('phase_update_hz',2000,500,20000),
             phase_bandwidth_hz=number('phase_bandwidth_hz',300,20,4000),
             frame_symbols=number('frame_symbols',256,1,65536,True),
             max_frames=number('max_frames',128,1,2048,True),
             max_input_samples=number('max_input_samples',16777216,4096,33554432,True),
             calibration_frames=number('calibration_frames',20,1,128,True),
             timing_offset_symbols=number('timing_offset_symbols',0,-.5,.5),
             timing_search_symbols=number('timing_search_symbols',.05,.001,.5),
             quality_evm_percent=number('quality_evm_percent',8,1,50),
             fit_timing=bool(o.get('fit_timing',False)), enabled=bool(o.get('enabled',False)))
    if o['phase_bandwidth_hz'] >= .45*o['phase_update_hz']:
        raise ValueError('公共相位带宽须小于更新率的45%；更新率不是环路带宽')
    return o


def _decimate(x, fs, target, cancel_event, cutoff=None):
    """Bounded FFT FIR stages, zero delay, preserve t=0 and sample-cell convention."""
    factor = 1
    while fs/2 >= target:
        _cancel(cancel_event)
        step = 4 if fs/4 >= target else 2
        x, _ = _lowpass(x,fs,.35*(fs/step),step,0,cancel_event)
        fs /= step; factor *= step
    if cutoff is not None:
        # At this low rate a longer FIR resolves the tracking band independently
        # of update rate. Pad edges; do not extrapolate an unobserved phase ramp.
        half = min(512, max(64, int(4*fs/cutoff)))
        grid = np.arange(-half,half+1)
        h = 2*cutoff/fs*np.sinc(2*cutoff/fs*grid)*np.blackman(len(grid)); h /= h.sum()
        x = np.convolve(np.pad(x,(half,half),mode='edge'),h,'valid')
    return x, fs, factor


def _phase_track(y, fs, o, cancel_event):
    if o['phase_mode'] == 'off':
        return np.array([0.,(len(y)-1)/fs]), np.zeros(2), 0., 1.
    square = o['phase_mode'] == 'square'
    z = y*y if square else y
    q, rate, _ = _decimate(z,fs,o['phase_update_hz'],cancel_event,
                            cutoff=o['phase_bandwidth_hz'])
    t = np.arange(len(q))/rate
    if len(q) < 8: raise ValueError('公共载波相位跟踪片段不足')
    strength = abs(q)
    coherence = float(np.median(strength)/max(np.sqrt(np.mean(abs(z)**2)),1e-15))
    valid = strength > max(float(np.median(strength))*.15, 1e-12)
    if np.count_nonzero(valid) < 8 or coherence < .025:
        raise ValueError('公共载波参考太弱；请改用真实训练，不能从数据符号逐点消相位')
    phase = np.unwrap(np.angle(q[valid]))/(2 if square else 1)
    phase = np.interp(t,t[valid],phase)
    frequency = np.gradient(phase,t)/(2*np.pi)
    if np.percentile(abs(frequency[2:-2]),99) > rate/(5 if square else 2.5):
        raise ValueError('公共相位轨迹接近混叠；增大更新率或先修正粗频偏')
    return t,phase,float(rate),coherence


def _phase_track_headers(x, fs, pwm, anchor, frame_period, cancel_event=None, *, training_context=None):
    try:
        return _phase_track_headers_legacy(x, fs, pwm, anchor, frame_period, cancel_event)
    except ValueError as legacy_error:
        if not pwm.get('observation_average_chips') or training_context is None:
            raise
        from rx706_qam_phase import track_known_headers
        result = track_known_headers(x, fs, pwm, anchor, frame_period, training_context, cancel_event)
        result[-1]['legacy_rejection_reason'] = str(legacy_error)
        return result


def _phase_track_headers_legacy(x, fs, pwm, anchor, frame_period, cancel_event=None):
    """Track only the known PWM state contrast; never use payload decisions.

    Short-header local slopes disambiguate inter-header 2pi wraps. Leave-one-
    header-out phase errors independently check the interpolation assumption.
    """
    period = fs/pwm['symbol_hz']; n = int(round(len(pwm['symbols'])*period))
    local = np.arange(n)/period; slot = np.minimum(local.astype(int),len(pwm['symbols'])-1)
    pulses = header_pulse_fractions(pwm)[slot]; within = local-slot
    high = (within>=pulses[:,0]) & (within<pulses.sum(axis=1))
    reach = max(.2, .5*pwm.get('observation_average_chips', 0.))/pwm['total_chips']
    good = ((within>reach)&(within<1-reach)&(abs(within-pulses[:,0])>reach)&
            (abs(within-pulses.sum(axis=1))>reach))
    first = math.ceil(-anchor/frame_period)
    last = math.floor((len(x)-n-2-anchor)/frame_period)
    times=[];angles=[];slopes=[];errors=[];magnitudes=[];rejected=[]
    for k in range(first,last+1):
        _cancel(cancel_event)
        at=anchor+k*frame_period;z=smooth_pwm_observation(_sample(x,at+np.arange(n)),fs,pwm)
        def contrast(mask):
            low=z[mask&~high].mean();top=z[mask&high].mean()
            return top-low,low,top
        delta,low,top=contrast(good)
        error=float(100*np.sqrt(np.mean(abs(z[good]-np.where(high[good],top,low))**2))/max(abs(delta),1e-15))
        if not math.isfinite(error) or error>pwm['max_error_percent']:
            rejected.append(float(at/fs));continue
        early=good&(np.arange(n)<n//2);late=good&~(np.arange(n)<n//2)
        if min(np.count_nonzero(mask&state) for mask in (early,late) for state in (high,~high))<3:
            raise ValueError('已知PWM前导两半的稳定两态不足，不能验证相位连续性')
        d1,_,_=contrast(early);d2,_,_=contrast(late)
        dt=(np.mean(np.flatnonzero(late))-np.mean(np.flatnonzero(early)))/fs
        slope=float(np.angle(d2*np.conj(d1))/(2*np.pi*dt))
        times.append((at+n/2)/fs);angles.append(float(np.angle(delta)))
        slopes.append(slope);errors.append(error);magnitudes.append(float(abs(delta)))
    if len(times)<6:raise ValueError('相位跟踪至少需要6个可信已知PWM前导')
    times=np.array(times);slopes=np.array(slopes);phase=np.array(angles)
    if np.max(np.diff(times))>3.1*frame_period/fs:
        raise ValueError('已知前导相位观测缺口超过3帧，不能可靠跟踪')
    innovations=[]
    for i in range(1,len(phase)):
        step=(angles[i]-angles[i-1]+np.pi)%(2*np.pi)-np.pi
        proposal=2*np.pi*.5*(slopes[i]+slopes[i-1])*(times[i]-times[i-1])
        step+=2*np.pi*round((proposal-step)/(2*np.pi))
        innovation=abs(step-proposal);innovations.append(float(innovation))
        if innovation>np.pi*.65:
            raise ValueError('已知前导的局部频偏与跨帧相位矛盾，拒绝相位假锁')
        phase[i]=phase[i-1]+step
    heldout=np.array([phase[i]-np.interp(times[i],times[[i-1,i+1]],phase[[i-1,i+1]])
                       for i in range(1,len(phase)-1)])
    if np.percentile(abs(heldout),95)>.20:
        raise ValueError('已知前导留一验证的相位误差过大；当前前导间隔不足以跟踪变化')
    evidence=dict(source='known_pwm_state_contrast_only',payload_used=False,
        header_count=len(times),rejected_header_times_s=rejected,
        header_times_s=times.tolist(),header_phase_rad=phase.tolist(),
        local_frequency_hz=slopes.tolist(),header_errors_percent=errors,
        unwrap_innovation_rad=innovations,
        heldout_phase_error_deg=(np.degrees(heldout)).tolist(),
        heldout_phase_error_p95_deg=float(np.degrees(np.percentile(abs(heldout),95))))
    # Bounded linear extension covers at most one edge frame. Interior payload
    # phase is interpolated between independent observed headers.
    end=(len(x)-1)/fs
    begin_phase=phase[0]+(0-times[0])*(phase[1]-phase[0])/(times[1]-times[0])
    end_phase=phase[-1]+(end-times[-1])*(phase[-1]-phase[-2])/(times[-1]-times[-2])
    out_t=np.r_[0.,times,end];out_p=np.r_[begin_phase,phase,end_phase]
    coherence=float(np.median(magnitudes)/max(np.sqrt(np.mean(abs(x[:min(len(x),n)])**2)),1e-15))
    return out_t,out_p,float(1/np.median(np.diff(times))),coherence,evidence


def _clock_repeat(y, fs, expected, ppm, cancel_event):
    _cancel(cancel_event)
    if len(y) < 4*expected: raise ValueError('重复时钟估计至少需要4个完整重复帧')
    v = y-y.mean(); energy = float(np.mean(abs(v)**2))
    if energy < 1e-12: raise ValueError('重复时钟估计没有可用交流波形')
    size = 1 << (2*len(v)-1).bit_length()
    spectrum = np.fft.fft(v,size)
    ac = np.fft.ifft(spectrum*spectrum.conj()).real[:len(v)]
    del spectrum
    ac /= np.arange(len(v),0,-1)*energy
    ks, lags, heights = [],[],[]
    for k in range(1,min(65,int(.8*len(y)/expected)+1)):
        _cancel(cancel_event)
        center = k*expected
        reach = max(4,int(math.ceil(center*ppm*1e-6)))
        left,right = max(1,int(center)-reach),min(len(ac)-2,int(center)+reach+1)
        if right <= left: continue
        j = left+int(np.argmax(ac[left:right]))
        # A boundary maximum means the actual clock may lie outside the range.
        if j in (left,right-1): continue
        denominator = ac[j-1]-2*ac[j]+ac[j+1]
        shift = .5*(ac[j-1]-ac[j+1])/denominator if denominator else 0.
        ks.append(k); lags.append(j+np.clip(shift,-1,1)); heights.append(ac[j])
    if len(ks) < 3 or np.median(heights) < .45:
        raise ValueError('未检出可信的整帧重复；自动重复模式不能用于随机新净荷，请用已知时钟/训练')
    k = np.array(ks); lags = np.array(lags)
    period = float(np.dot(k,lags)/np.dot(k,k))
    residual = lags-k*period
    if abs(period/expected-1)*1e6 > ppm:
        raise ValueError('重复时钟超出配置的ppm搜索范围')
    return period,dict(coherence=float(np.median(heights)),lags=k.tolist(),
                       lag_residual_samples=residual.tolist(),period_samples=period,
                       note='整帧重复波形估计；不是任意净荷的通用时钟锁定')


def _header_fit(x, fs, at, pwm):
    period = fs/pwm['symbol_hz']; n = int(len(pwm['symbols'])*period)
    pos = at+np.arange(n)
    if n < 8 or at < 0 or pos[-1] >= len(x)-1: return float('inf'),0j,0j
    z = smooth_pwm_observation(_sample(x,pos),fs,pwm)
    local = np.arange(n)/period
    symbols = np.minimum(local.astype(int),len(pwm['symbols'])-1)
    pulses = header_pulse_fractions(pwm)[symbols]
    rise, fall = pulses[:,0], pulses.sum(axis=1)
    phase = local-symbols; high = (phase >= rise) & (phase < fall)
    reach = max(.2, .5*pwm.get('observation_average_chips', 0.))/pwm['total_chips']
    good = (phase>reach)&(phase<1-reach)&(abs(phase-rise)>reach)&(abs(phase-fall)>reach)
    if min(np.count_nonzero(good&high),np.count_nonzero(good&~high)) < 3: return float('inf'),0j,0j
    low,top = z[good&~high].mean(),z[good&high].mean()
    error = 100*float(np.sqrt(np.mean(abs(z[good]-np.where(high[good],top,low))**2)))/max(abs(top-low),1e-15)
    return error,low,top


def _fit_grid(c, order, cancel_event):
    rms = float(np.sqrt(np.mean(abs(c)**2)))
    if rms < 1e-10: raise ValueError('目标谐波能量不足')
    best = None
    for phase in np.linspace(0,np.pi/2,181,endpoint=False):
        _cancel(cancel_event)
        gain = rms*np.exp(1j*phase)
        for _ in range(4):
            ideal,_,_ = decide(c/gain,order)
            gain = np.vdot(ideal,c)/np.vdot(ideal,ideal)
        z = c/gain; ideal,_,_ = decide(z,order)
        error = float(np.mean(abs(z-ideal)**2)/np.mean(abs(ideal)**2))
        if best is None or error < best[0]: best = error,gain,ideal
    return best


def _fit_timing(y, fs, starts, frame_period, rate, selected, ndata, o, order, cancel_event):
    """Three real parameters fit on calibration frames, never heldout frames."""
    length = max(1,int(round(frame_period)))
    positions = np.arange(length)*frame_period/length
    grid = np.arange(len(y)); mean = np.zeros(length,complex)
    for start in starts:
        _cancel(cancel_event)
        pos = start+positions
        mean += np.interp(pos,grid,y.real)+1j*np.interp(pos,grid,y.imag)
    mean /= len(starts)
    effective_fs = fs*length/frame_period
    repeated = np.tile(mean,3)
    best = None
    for offset in np.linspace(-o['timing_search_symbols'],o['timing_search_symbols'],51):
        _cancel(cancel_event)
        c,_ = harmonic_coefficients(repeated,effective_fs,length+offset*effective_fs/rate,
                                     rate,selected,count=ndata,cancel_event=cancel_event)
        error,gain,ideal = _fit_grid(c,order,cancel_event)
        if best is None or error < best[0]: best = error,float(offset),gain,ideal,c
    # Reconcile zero-order sample-cell phase/sinc conventions with original Fs.
    return best, effective_fs


def _rotate_inplace(x, fs, frequency=0., phase_t=None, phase=None, cancel_event=None):
    for start in range(0,len(x),262144):
        _cancel(cancel_event); end = min(start+262144,len(x)); t = np.arange(start,end)/fs
        angle = 2*np.pi*frequency*t
        if phase_t is not None: angle = angle+np.interp(t,phase_t,phase)
        x[start:end] *= np.exp(-1j*angle)


def _evm(z, ideal):
    return float(100*np.sqrt(np.mean(abs(z-ideal)**2)/max(np.mean(abs(ideal)**2),1e-30)))


def recover_compensated_tdcm(spectrum, *, duration_s=1, center_hint_hz=-200000,
        symbol_hz=19531.25, order=256, tdcm_options=None,
        pwm_preamble_options=None, preamble_options=None, cancel_event=None):
    fs = spectrum.sample_rate_hz
    cfg = validate_tdcm(fs,symbol_hz,tdcm_options or {})
    o = validate_compensation((tdcm_options or {}).get('compensation'))
    pwm = validate_pwm_header(pwm_preamble_options or dict(symbol_hz=19531.25),fs)
    if order not in (16,64,256): raise ValueError('QAM阶数须为16、64或256')
    if center_hint_hz is None or not math.isfinite(center_hint_hz): raise ValueError('请设置原载波基带频率')
    if abs(center_hint_hz)+pwm['max_frequency_hz']+abs(cfg['harmonic']*cfg['base_hz']) >= fs/2:
        raise ValueError('TDCM载波或目标谐波超出采样范围')
    ndata = cfg['payload_symbols'] or o['frame_symbols']
    if cfg['payload_symbols'] and cfg['payload_symbols'] != o['frame_symbols']:
        raise ValueError('增强模式下“提取符号数”须为0或等于“每帧QAM窗口数”')
    training = None if preamble_options is None else validate_preamble(preamble_options,order,symbol_hz)
    if training is not None and len(training[0])+1 > ndata:
        raise ValueError('每帧窗口数不足：须包含QAM训练及至少1个净荷符号')
    from rx706_qam_training import validate_orientation,fit_training,normalize_training
    training_orientation,training_offset_enabled=validate_orientation(preamble_options or {})
    x,count,clipped = _read_span(spectrum,duration_s,center_hint_hz,cancel_event,
        max_samples=min(math.ceil(2*fs),o['max_input_samples']),max_duration_s=2)
    _rotate_inplace(x,fs,center_hint_hz,cancel_event=cancel_event)
    original_pwm = dict(pwm)
    sync = acquire_pwm_header(x,fs,dict(pwm,symbols=' '.join(map(str,pwm['symbols']))),cancel_event)
    frequency,low,top,pwm_error = _pwm_reference(x,fs,sync,pwm,cancel_event)
    _rotate_inplace(x,fs,frequency,cancel_event=cancel_event)
    y,coarse_fs,factor = _decimate(x,fs,max(500000.,48*cfg['base_hz']),cancel_event)
    before = y.copy()
    phase_tracking_info=None
    phase_options=dict(o,phase_mode='off') if o['phase_mode']=='pwm_headers' else o
    phase_t,phase,actual_update,phase_coherence = _phase_track(y,coarse_fs,phase_options,cancel_event)
    if o['phase_mode'] not in ('off','pwm_headers'):
        _rotate_inplace(y,coarse_fs,phase_t=phase_t,phase=phase,cancel_event=cancel_event)
        _rotate_inplace(x,fs,phase_t=phase_t,phase=phase,cancel_event=cancel_event)
    scale = 1+o['period_ppm']*1e-6 if o['clock_mode']=='manual' else 1.
    nominal_cycles = len(pwm['symbols'])*fs/pwm['symbol_hz']+ndata*fs/symbol_hz
    nominal_frame = nominal_cycles+pwm['gap_s']*fs
    clock_info = dict(note='配置固定周期',coherence=None,lags=[],lag_residual_samples=[])
    if o['clock_mode']=='repeat':
        measured_frame,clock_info = _clock_repeat(y,coarse_fs,nominal_frame/factor,o['clock_search_ppm'],cancel_event)
        scale = (measured_frame*factor-pwm['gap_s']*fs)/nominal_cycles
    rate = symbol_hz/scale; base = cfg['base_hz']/scale
    pwm = dict(pwm,symbol_hz=pwm['symbol_hz']/scale)
    selected = cfg['harmonic']*base
    frame_period = nominal_cycles*scale+pwm['gap_s']*fs
    header_length = len(pwm['symbols'])*fs/pwm['symbol_hz']
    # Re-anchor the short PWM template with the corrected period / common phase.
    anchor = sync['index']
    best = None
    for delta in np.linspace(-fs/pwm['symbol_hz']/pwm['total_chips'],fs/pwm['symbol_hz']/pwm['total_chips'],33):
        candidate = anchor+delta; fit = _header_fit(x,fs,candidate,pwm)
        if best is None or fit[0] < best[0]: best = (*fit,candidate)
    pwm_error,low,top,anchor = best
    if pwm_error > pwm['max_error_percent']: raise ValueError('补偿后的PWM前导仍未通过匹配容差')
    if o['frame_mode']=='single': headers = [anchor]
    else:
        first = math.ceil(-anchor/frame_period)
        last = math.floor((count-header_length-pwm['gap_s']*fs-ndata*fs/rate-anchor)/frame_period)
        headers = [anchor+k*frame_period for k in range(first,last+1)][:o['max_frames']]
    if not headers: raise ValueError('配置帧长度超过当前片段，没有完整数据帧')
    if o['phase_mode']=='pwm_headers':
        phase_t,phase,actual_update,phase_coherence,phase_tracking_info = _phase_track_headers(
            x,fs,pwm,anchor,frame_period,cancel_event,
            training_context=None if training is None else (training[0],training[1],training[2],rate,selected,training_orientation,training_offset_enabled))
        _rotate_inplace(x,fs,phase_t=phase_t,phase=phase,cancel_event=cancel_event)
        _rotate_inplace(y,coarse_fs,phase_t=phase_t,phase=phase,cancel_event=cancel_event)
        pwm_error,low,top=_header_fit(x,fs,anchor,pwm)
    # Fixed global grid offset is calibrated only on the first N full frames.
    starts = np.array(headers)+header_length+pwm['gap_s']*fs
    fit_frames = min(o['calibration_frames'],len(starts))
    timing_offset = o['timing_offset_symbols']
    blind_fit = None; template = None; gain_source = ''
    frozen = (tdcm_options or {}).get('frozen_calibration')
    if frozen is not None:
        if not isinstance(frozen,dict) or training is not None:
            raise ValueError('冻结盲标定须为配置对象且不能覆盖真实QAM训练')
        try:
            gain=complex(float(frozen['gain_real']),float(frozen['gain_imag']))
            pairs=np.asarray(frozen['template'],dtype=float)
        except (KeyError,TypeError,ValueError):raise ValueError('冻结标定格式无效') from None
        if not np.isfinite(gain) or abs(gain)<1e-12 or pairs.shape!=(ndata,2) or not np.all(np.isfinite(pairs)):
            raise ValueError('冻结复增益或重复模板无效')
        template=pairs[:,0]+1j*pairs[:,1]
        blind_fit=dict(frozen=True,calibration_frames=fit_frames)
        gain_source='自动搜索前段冻结盲标定（最终后段未参与拟合/选择）'
    elif o['fit_timing']:
        if training is not None or cfg['normalization'] != 'blind':
            raise ValueError('盲网格定时拟合仅用于盲归一化且未启用真实QAM训练时')
        if o['clock_mode'] != 'repeat':
            raise ValueError('盲网格定时拟合需要已验证的整帧重复时钟模式')
        usable = starts[:fit_frames]/factor
        usable = usable[usable+frame_period/factor < len(y)-1]
        if not len(usable): raise ValueError('盲标定需要至少一个完整重复波形窗口')
        best,eff_fs = _fit_timing(y,coarse_fs,usable,frame_period/factor,rate,selected,ndata,o,order,cancel_event)
        fit_error,offset,gain,template,train_coeff = best
        timing_offset += offset
        correction = np.sinc(selected/fs)/np.sinc(selected/eff_fs)*np.exp(1j*np.pi*selected*(1/eff_fs-1/fs))
        gain *= correction
        blind_fit = dict(fitted_offset_symbols=offset,training_nn_evm_percent=100*math.sqrt(fit_error),
                         calibration_frames=len(usable),sample_cell_phase_correction_deg=float(np.angle(correction,deg=True)))
        gain_source = '重复图样前段盲网格拟合（非真实发端参考；90°歧义）'
    starts += timing_offset*fs/rate
    coefficients=[]; centres=[]; frame_ids=[]; rows=[]
    for index,(header,start) in enumerate(zip(headers,starts)):
        _cancel(cancel_event)
        error,lv,hv = _header_fit(x,fs,header,pwm)
        row=dict(index=index,header_time_s=(header+spectrum.first_tuple)/fs,
                 data_time_s=(start+spectrum.first_tuple)/fs,header_error_percent=error,
                 valid=error<=pwm['max_error_percent'],calibration=index<fit_frames,
                 symbols=0,nn_evm_percent=None,template_evm_percent=None,pilot_evm_percent=None)
        if start < 0 or start+ndata*fs/rate > len(x)+1e-6: row['valid']=False
        if phase_tracking_info and phase_tracking_info.get('tracking_strategy')=='known_header_dynamic_frequency':
            observed=phase_tracking_info['header_times_s']
            # Dynamic phase is validated between observed headers, not after
            # the final header. Do not turn a short-pilot extrapolation into
            # apparently validated high-order QAM decisions at the boundary.
            if start/fs < observed[0] or (start+ndata*fs/rate)/fs > observed[-1]:
                row['valid']=False
                row['phase_failure']='净荷缺少双侧已知前导相位观测，按边界擦除'
        if row['valid']:
            c,t = harmonic_coefficients(x,fs,start,rate,selected,count=ndata,cancel_event=cancel_event)
            coefficients.append(c); centres.append(t); frame_ids.append(np.full(len(c),index,int));row['symbols']=ndata
        rows.append(row)
    if not coefficients: raise ValueError('所有帧前导或完整窗口检查失败')
    coefficients=np.concatenate(coefficients);centres=np.concatenate(centres);frame_ids=np.concatenate(frame_ids)
    raw_coefficients=coefficients.copy()
    pilot=None; trim=0;training_records=[];training_fit=None
    if training is not None:
        points,bound,tolerance,_ = training
        seeds=[];seed_failures=[]
        for seed_id in np.unique(frame_ids)[:8]:
            first_c=coefficients[frame_ids==seed_id]
            try:
                seed_at,seed_fit = _coefficient_pilot(first_c,points,rate,bound,tolerance,pwm['qam_search_symbols'],decide,order,cancel_event,
                    orientation=training_orientation,fit_dc_offset=training_offset_enabled,
                    estimate_frequency=o['phase_mode']=='off' and preamble_options.get('estimate_frequency',True),return_details=True)
                seeds.append((seed_fit['evm_percent'],int(seed_id),seed_at,seed_fit))
            except ValueError as exc:seed_failures.append(str(exc))
        if not seeds:raise ValueError('前段已知QAM训练均未匹配：'+(seed_failures[0] if seed_failures else '无完整帧'))
        _,training_seed,at,training_fit=min(seeds,key=lambda v:v[0])
        residual=training_fit['frequency_hz']
        # A short pilot slope is noisy; after continuous RF phase tracking do
        # not extrapolate it across the full record. Each actual pilot below
        # supplies its frame's complex reference without touching data phases.
        if o['phase_mode'] != 'off': residual = 0.
        if abs(residual)>1e-6:
            _rotate_inplace(x,fs,residual,cancel_event=cancel_event)
            parts=[]
            for row,start in zip(rows,starts):
                if row['valid']: parts.append(harmonic_coefficients(x,fs,start,rate,selected,count=ndata,cancel_event=cancel_event)[0])
            coefficients=np.concatenate(parts);frequency+=residual
        gain=training_fit['gain']
        trim=at+len(points);gain_source='每帧真实QAM训练复增益（训练与净荷分离；未测净荷BER）'
        seed_indices=np.flatnonzero(frame_ids==training_seed)
        pilot=dict(length=len(points),time_s=(centres[seed_indices[at]]+spectrum.first_tuple)/fs,
                   payload_time_s=(centres[seed_indices[trim]]+spectrum.first_tuple)/fs,frequency_hz=residual,
                   total_frequency_hz=center_hint_hz+frequency,phase_deg=float(np.angle(gain,deg=True)),
                   evm_percent=training_fit['evm_percent'],conjugated=training_fit['conjugated'],
                   orientation_mode=training_orientation,fit_dc_offset=training_offset_enabled,
                   expected_hex=[f'{int(v):02X}' for v in decide(points,order)[1]])
        template=None
    elif blind_fit is None:
        if cfg['normalization']=='pwm':
            radius=math.sqrt(2)*(math.isqrt(order)-1)/math.sqrt(2*(order-1)/3)
            gain=(top-low)/(np.pi*abs(cfg['harmonic'])*radius)
            gain_source='PWM两态模型（须核对目标谐波幅相映射）'
        else:
            calibration=coefficients[np.isin(frame_ids,np.unique(frame_ids)[:fit_frames])]
            _,gain,_=_fit_grid(calibration,order,cancel_event)
            gain_source='前段盲幅相拟合（未验证发端映射；90°歧义）'
    if training is None: gain *= cfg['amplitude_scale']*np.exp(1j*np.deg2rad(cfg['phase_deg']))
    if abs(gain)<1e-12: raise ValueError('目标谐波校准增益不足')
    z=coefficients/gain
    if training is None and template is None and o['clock_mode']=='repeat' and o['frame_mode']=='all':
        # A short known-PWM payload can be repeated without occupying half the
        # QAM alphabet. Freeze a prefix waveform template for diagnostics;
        # sparse occupancy still cannot validate the unknown QAM mapping.
        learned=[z[frame_ids==row['index']] for row in rows if row['valid'] and row['calibration']]
        if len(learned)>=2: template=np.mean(learned,axis=0)
    if training is not None:
        for row in rows:
            ids=np.flatnonzero(frame_ids==row['index'])
            if not len(ids): continue
            observed=coefficients[ids[at:trim]]
            try:
                frame_fit=fit_training(observed,points,rate,bound,tolerance,
                    orientation='conjugate' if training_fit['conjugated'] else 'normal',
                    fit_dc_offset=training_offset_enabled,estimate_frequency=False,cancel_event=cancel_event)
            except ValueError as exc:
                row['valid']=False;row['pilot_evm_percent']=float('inf');row['training_failure']=str(exc);frame_fit=None
            if frame_fit is not None:
                z[ids]=normalize_training(coefficients[ids],frame_fit)
                row['pilot_gain_iq']=complex(frame_fit['gain']);row['pilot_offset_iq']=complex(frame_fit['offset'])
                row['pilot_evm_percent']=frame_fit['evm_percent'];row['pilot_conjugated']=frame_fit['conjugated']
            expected_labels=decide(points,order)[1]
            for j,(raw_point,expected) in enumerate(zip(observed,points)):
                value=z[ids[at+j]] if frame_fit is not None else None
                training_records.append(dict(frame=int(row['index']),training_index=j,
                    time_s=float((centres[ids[at+j]]+spectrum.first_tuple)/fs),expected_hex=f'{int(expected_labels[j]):02X}',
                    expected_i=float(expected.real),expected_q=float(expected.imag),
                    coefficient_i_counts=float(raw_point.real),coefficient_q_counts=float(raw_point.imag),
                    normalized_i=None if value is None else float(value.real),normalized_q=None if value is None else float(value.imag),
                    observed_hex=None if value is None else f'{int(decide(np.array([value]),order)[1][0]):02X}',
                    training_passed=bool(frame_fit is not None),conjugated=bool(training_fit['conjugated']),
                    evm_percent=None if frame_fit is None else frame_fit['evm_percent']))
    ideal,labels,bits=decide(z,order)
    keep=np.ones(len(z),bool)
    for row in rows:
        ids=np.flatnonzero(frame_ids==row['index'])
        if not len(ids): continue
        if training is not None:
            row['pilot_evm_percent']=_evm(z[ids[at:trim]],points)
            if row['pilot_evm_percent']>tolerance: row['valid']=False
            keep[ids[:trim]]=False
        used=ids[trim:]
        row['symbols']=len(used);row['nn_evm_percent']=_evm(z[used],ideal[used])
        if template is not None: row['template_evm_percent']=_evm(z[used],template[trim:])
        if not row['valid']: keep[ids]=False
    if np.count_nonzero(keep)<1: raise ValueError('训练剔除或逐帧校验后没有候选数据符号')
    z=z[keep];ideal=ideal[keep];labels=labels[keep];bits=bits[keep];centres=centres[keep]
    coefficients=coefficients[keep];frame_ids=frame_ids[keep]
    heldout=[r['template_evm_percent'] for r in rows if not r['calibration'] and r['valid'] and r['template_evm_percent'] is not None]
    occupied=len(np.unique(labels));nn=_evm(z,ideal)
    repeat_passed=len(heldout)>=3 and float(np.median(heldout))<=o['quality_evm_percent']
    sparse_repeat=repeat_passed and occupied<order//2
    structure=repeat_passed and occupied>=order//2
    quality=('训练已匹配；净荷仍为候选' if training is not None else
             '重复网格结构可恢复；盲估候选' if structure else
             'PWM已匹配；稀疏重复候选可提取；QAM映射未验证' if sparse_repeat else
             '候选，尚无充分独立验证')
    warnings=['未提供净荷比特参考，未测BER；最近邻EVM不证明正确解码。']
    if phase_tracking_info and phase_tracking_info.get('tracking_strategy')=='known_header_dynamic_frequency':
        erased=[row['index'] for row in rows if row.get('phase_failure')]
        phase_tracking_info['boundary_erased_frames']=erased
        phase_tracking_info['payload_phase_extrapolation_allowed']=False
        if erased:
            warnings.append(f'动态相位有{len(erased)}帧缺少双侧已知前导，已作为边界擦除；不外推未知净荷相位。')

    if occupied < order//2:
        warnings.append('载荷格点覆盖较少：保留已匹配PWM后的候选复系数；不能仅据此识别QAM阶数或确认映射。')
    if len(z) < 16:
        warnings.append('本次不足16个候选符号，只供窗口与复系数诊断；没有充分符号统计。')
    if training is None: warnings.append('盲标定有90°旋转/映射歧义；模板误差不是发端参考EVM。')
    if not heldout and training is None: warnings.append('没有独立的后段重复模板验证。')
    if count < min(round(duration_s*fs),spectrum.total_tuples-spectrum.first_tuple):
        warnings.append(f'本次受输入点数上限限制，仅处理{count}点；可移动起始采样点继续。')
    if spectrum.rate_source != '同名 JSON': warnings.append('Fs来自手动/默认设置；Hz与时间依赖该假设。')
    if len(headers)==o['max_frames']: warnings.append('显示结果可能达到最大处理帧数；可增加帧数或移动起始采样点。')
    # Before/after use one gain; applying a different cell rate has analytic phase.
    before_start=(headers[0]+len(original_pwm['symbols'])*fs/original_pwm['symbol_hz']+original_pwm['gap_s']*fs)/factor
    available=int((len(before)-before_start)/(coarse_fs/symbol_hz))
    bc,_=harmonic_coefficients(before,coarse_fs,before_start,symbol_hz,cfg['harmonic']*cfg['base_hz'],
                               count=min(8192,max(0,available)),cancel_event=cancel_event)
    coarse_gain=gain*np.sinc(selected/coarse_fs)/np.sinc(selected/fs)*np.exp(-1j*np.pi*selected*(1/coarse_fs-1/fs))
    before_scatter=bc/coarse_gain
    origin=spectrum.first_tuple/fs
    phase_frequency=np.gradient(phase,phase_t)/(2*np.pi)+frequency
    diagnostics=dict(options=o,quality=quality,structural_validation=structure,ber=None,
        processing_stages=dict(pwm_header='matched_candidate',symbol_extraction='candidate_coefficients',
            quality_validation='training_matched' if training is not None else 'repeat_structure_passed' if structure else
                'sparse_repeat_verified_mapping_unverified' if sparse_repeat else 'unvalidated',
            bit_validation='not_measured',sparse_payload=occupied<order//2),
        repeat_validation_passed=repeat_passed,
        training_summary=None if training is None else dict(expected_hex=pilot['expected_hex'],
            conjugated=pilot['conjugated'],orientation_mode=training_orientation,fit_dc_offset=training_offset_enabled,
            matched_frames=sum(bool(v['valid']) for v in rows),failed_frames=sum(not bool(v['valid']) for v in rows),
            note='仅使用各帧已知训练校准复增益/零点；训练从净荷剔除。训练匹配不是净荷BER。'),
        rate_source=spectrum.rate_source,warnings=warnings,clock=clock_info,
        nominal_symbol_hz=symbol_hz,measured_symbol_hz=rate,period_samples=fs/rate,
        period_ppm=(scale-1)*1e6,timing_offset_symbols=timing_offset,
        phase_time_s=phase_t+origin,phase_rad=phase,frequency_hz=phase_frequency,
        phase_update_hz=actual_update,phase_coherence=phase_coherence,phase_tracking=phase_tracking_info,
        before_scatter=before_scatter,gain_iq=gain,gain_source=gain_source,
        blind_fit=blind_fit,heldout_template_evm_percent=float(np.median(heldout)) if heldout else None,
        nearest_evm_percent=nn,occupied_labels=occupied,frames=rows,
        frame_period_samples=frame_period,parameter_source='配置、PWM拟合与重复波形估计；真实发端参数待核对')
    sync.update(index=float(headers[0]),qam_index=float(starts[0]),end_index=float(headers[0]+header_length),
        time_s=origin+headers[0]/fs,end_time_s=origin+(headers[0]+header_length)/fs,
        qam_time_s=origin+starts[0]/fs,frequency_hz=frequency,error_percent=pwm_error)
    details=dict(cfg,base_hz=base,normalization=cfg['normalization'],original_carrier_hz=center_hint_hz,
        corrected_carrier_hz=center_hint_hz+frequency,target_hz=center_hint_hz+frequency+selected,
        pwm_low_iq=low,pwm_high_iq=top,coefficient_units='ADC counts',calibration_note=gain_source)
    result=SimpleNamespace(kind='qam',carrier_hz=center_hint_hz,order=order,symbol_hz=rate,
        scatter=z,ideal=ideal,labels=labels,bits=bits,evm_percent=nn,time_s=(centres+spectrum.first_tuple)/fs,
        first_tuple=spectrum.first_tuple,input_samples=count,clipped=clipped,cutoff_hz=rate/2,
        decimation=1,output_rate_hz=rate,preamble=pilot,pwm_preamble=sync,tdcm=details,
        coefficients=coefficients,raw_coefficients=raw_coefficients,diagnostics=diagnostics,
        frame_ids=frame_ids,frames=rows,training_records=training_records)

    from rx706_qam_payload import attach_payload_debug
    return attach_payload_debug(result)

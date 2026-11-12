"""Low-SNR phase recovery from known PWM only, with independent QAM checks.

The model assumes smooth phase between headers. It cannot identify arbitrary
phase excursions inside an interval containing only unknown payload.
"""
import math
import numpy as np
from rx706_spectrum import _cancel
from rx706_qam_preamble import _sample
from rx706_qam_pwm import header_pulse_fractions, smooth_pwm_observation


def pchip_slopes(t, y):
    """Fritsch-Butland shape-preserving cubic slopes; no SciPy dependency."""
    t=np.asarray(t,float);y=np.asarray(y,float)
    h=np.diff(t);d=np.diff(y)/h
    if len(t)<2 or np.any(h<=0):raise ValueError('相位插值需要递增的已知前导时间')
    if len(t)==2:return np.array([d[0],d[0]])
    m=np.zeros(len(t));same=(d[:-1]*d[1:])>0
    w1=2*h[1:]+h[:-1];w2=h[1:]+2*h[:-1]
    m[1:-1][same]=(w1[same]+w2[same])/(w1[same]/d[:-1][same]+w2[same]/d[1:][same])
    for at,h0,h1,d0,d1 in ((0,h[0],h[1],d[0],d[1]),(-1,h[-1],h[-2],d[-1],d[-2])):
        v=((2*h0+h1)*d0-h0*d1)/(h0+h1)
        if np.sign(v)!=np.sign(d0):v=0.
        elif np.sign(d0)!=np.sign(d1) and abs(v)>3*abs(d0):v=3*d0
        m[at]=v
    return m


def pchip(t,y,q):
    t=np.asarray(t,float);y=np.asarray(y,float);q=np.asarray(q,float);m=pchip_slopes(t,y)
    j=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2)
    h=t[j+1]-t[j];u=(q-t[j])/h
    out=(2*u**3-3*u**2+1)*y[j]+(u**3-2*u**2+u)*h*m[j]+(-2*u**3+3*u**2)*y[j+1]+(u**3-u**2)*h*m[j+1]
    # Extrapolation is bounded by the caller and linear, never cubic runaway.
    out=np.where(q<t[0],y[0]+(q-t[0])*m[0],out)
    return np.where(q>t[-1],y[-1]+(q-t[-1])*m[-1],out)


def cubic_phase_slopes(t, y):
    """Not-a-knot cubic phase derivatives, invariant to a linear carrier trend.

    Solve the tridiagonal second-derivative system without SciPy. Unlike a
    monotonicity limiter, this does not clamp a valid frequency zero crossing.
    """
    t=np.asarray(t,float);y=np.asarray(y,float);h=np.diff(t)
    if len(t)<2 or len(t)!=len(y) or np.any(h<=0):
        raise ValueError('三次相位插值需要递增等长前导观测')
    d=np.diff(y)/h
    if len(t)==2:return np.array([d[0],d[0]])
    if len(t)==3:
        second=np.full(3,2*(d[1]-d[0])/(h[0]+h[1]))
        return np.r_[d-h*(2*second[:-1]+second[1:])/6,
                      d[-1]+h[-1]*(second[-2]+2*second[-1])/6]
    diagonal=2*(h[:-1]+h[1:]);rhs=6*np.diff(d)
    lower=h[1:-1].copy();upper=h[1:-1].copy()
    # Eliminate endpoint second derivatives using continuity of the third
    # derivative at the first/last interior knot. This reproduces a genuine
    # quadratic frequency sweep without imposing zero endpoint curvature.
    diagonal[0]=(h[0]+h[1])*(h[0]/h[1]+2)
    upper[0]=h[1]-h[0]*h[0]/h[1]
    diagonal[-1]=(h[-1]+h[-2])*(h[-1]/h[-2]+2)
    lower[-1]=h[-2]-h[-1]*h[-1]/h[-2]
    diagonal=diagonal.copy();rhs=rhs.copy()
    for k in range(1,len(diagonal)):
        ratio=lower[k-1]/diagonal[k-1]
        diagonal[k]-=ratio*upper[k-1];rhs[k]-=ratio*rhs[k-1]
    second=np.zeros(len(t));second[-2]=rhs[-1]/diagonal[-1]
    for k in range(len(diagonal)-2,-1,-1):
        second[k+1]=(rhs[k]-upper[k]*second[k+2])/diagonal[k]
    second[0]=((h[0]+h[1])*second[1]-h[0]*second[2])/h[1]
    second[-1]=((h[-1]+h[-2])*second[-2]-h[-1]*second[-3])/h[-2]
    return np.r_[d-h*(2*second[:-1]+second[1:])/6,
                  d[-1]+h[-1]*(second[-2]+2*second[-1])/6]


def cubic_phase(t,y,q):
    """Not-a-knot cubic within observed headers, linear outside their range."""
    t=np.asarray(t,float);y=np.asarray(y,float);q=np.asarray(q,float)
    slopes=cubic_phase_slopes(t,y)
    j=np.clip(np.searchsorted(t,q,side='right')-1,0,len(t)-2)
    h=t[j+1]-t[j];u=(q-t[j])/h
    out=(2*u**3-3*u**2+1)*y[j]+(u**3-2*u**2+u)*h*slopes[j]+(-2*u**3+3*u**2)*y[j+1]+(u**3-u**2)*h*slopes[j+1]
    out=np.where(q<t[0],y[0]+(q-t[0])*slopes[0],out)
    return np.where(q>t[-1],y[-1]+(q-t[-1])*slopes[-1],out)


def joint_header_fit(z, good, high, fs, period):
    """Joint profile fit with one-PWM-symbol delete-block jackknife SE."""
    state=high[good];positions=np.flatnonzero(good)
    t=(positions-len(z)/2)/fs;y=z[good];slot=(positions/period).astype(int)
    def fit(mask):
        s=state[mask];tt=t[mask];sample=y[mask];omega=0.
        if min(np.count_nonzero(s),np.count_nonzero(~s))<3:raise ValueError('稳定前导平台不足')
        for _ in range(5):
            v=sample*np.exp(-1j*omega*tt);low=v[~s].mean();top=v[s].mean();model=np.where(s,top,low)
            derivative=1j*model*tt
            derivative[s]-=derivative[s].mean();derivative[~s]-=derivative[~s].mean()
            energy=float(np.vdot(derivative,derivative).real)
            if energy<1e-20:raise ValueError('前导频偏不可观测')
            omega+=float(np.vdot(derivative,v-model).real)/energy
        v=sample*np.exp(-1j*omega*tt);low=v[~s].mean();top=v[s].mean()
        error=100*np.sqrt(np.mean(abs(v-np.where(s,top,low))**2))/max(abs(top-low),1e-15)
        return omega/(2*np.pi),float(np.angle(top-low)),float(error),float(abs(top-low))
    result=fit(np.ones(len(y),bool));blocks=np.unique(slot)
    if len(blocks)<6:raise ValueError('弱信号相位验证至少需要6个已知PWM符号')
    jack=np.array([fit(slot!=k)[0] for k in blocks])
    se=math.sqrt((len(jack)-1)/len(jack)*np.sum((jack-jack.mean())**2))
    return (*result,float(se))


def resolve_header_phase(times,angles,frequencies,se,frame_s):
    """Resolve a common alias under a smooth-phase model, then hold out knots.

    Five-scale margins include empirical scatter, symbol-block jackknife,
    short-lag covariance, and nonoverlapping four-header block means. Every overlapping window is checked, with at most 32 headers.
    This is a model-based uncertainty guard, not a distribution-free guarantee.
    """
    t=np.asarray(times);phase=np.unwrap(angles);f=np.asarray(frequencies);se=np.asarray(se)
    if len(t)<12:raise ValueError('弱信号相位回退至少需要12个可信前导用于独立验证')
    frame=np.rint((t-t[0])/frame_s).astype(int)
    if np.any(np.diff(frame)!=1):raise ValueError('弱信号相位回退存在前导缺口，不能可靠确定跨缺口周数')
    if np.max(abs((t-t[0])-frame*frame_s))>frame_s*.01:raise ValueError('相位前导时间不符合已验证帧时钟')
    # Large principal increments approach Nyquist ambiguity. Never invent a
    # branch at such crossings from a smooth interpolant alone.
    if np.max(abs(np.diff(phase)))>=.8*np.pi:raise ValueError('前导间相位接近周数歧义边界')
    return _validate_phase_candidate(t,phase,f,se,frame_s,frame)


def _validate_phase_candidate(t,phase,f,se,frame_s,frame, *, interpolate=pchip, slopes=pchip_slopes, local_frequency_sigma=5.):
    derivative=slopes(t,phase)/(2*np.pi)
    alias_hz=1/frame_s;branch=int(np.rint(np.median(f-derivative)/alias_hz))
    residual=f-derivative-branch*alias_hz
    if np.any(abs(residual)>local_frequency_sigma*np.maximum(se,1e-6)):
        raise ValueError('已知前导局部频偏显著矛盾，不能用跨帧平均掩盖周数变化')
    width=min(32,len(t));margins=[]
    for start in range(len(t)-width+1):
        r=residual[start:start+width];s=se[start:start+width]
        centered=r-np.mean(r);lag=min(4,width//4)
        long_run=float(np.mean(centered*centered))
        for j in range(1,lag+1):
            long_run+=2*(1-j/(lag+1))*float(np.mean(centered[j:]*centered[:-j]))
        blocks=np.array([b.mean() for b in np.array_split(r,max(2,width//4))])
        uncertainty=max(float(np.std(r,ddof=1))/math.sqrt(width),
            float(np.sqrt(np.mean(s*s)/width)),math.sqrt(max(0,long_run)/width),
            float(np.std(blocks,ddof=1))/math.sqrt(len(blocks)))
        margin=abs(float(np.mean(r)))+5*uncertainty
        margins.append(margin)
        if margin>=alias_hz/2:raise ValueError('已知前导频偏不确定性无法排除另一相位周数')
    phase=phase+2*np.pi*branch*frame
    errors=[];indices=[]
    for k in range(1,len(t)-1):
        keep=np.arange(len(t))!=k
        # The knot is omitted from interpolation, conditional on the full
        # series' prior unwrapping/alias decision. This is not reacquisition.
        errors.append(float(phase[k]-interpolate(t[keep],phase[keep],t[k])));indices.append(k)
    calibration=min(20,max(6,len(t)//3));heldout=np.asarray(errors)[np.asarray(indices)>=calibration]
    if len(heldout)<4:raise ValueError('弱信号相位独立留出前导不足')
    p95=float(np.percentile(abs(heldout),95))
    all_p95=float(np.percentile(abs(np.asarray(errors)),95))
    if max(p95,all_p95)>.20:raise ValueError('已知前导留一验证的相位误差过大；当前前导间隔不足以跟踪变化')
    return phase,dict(phase_model='not_a_knot_cubic' if interpolate is cubic_phase else 'shape_preserving_cubic',calibration_headers=calibration,
        all_header_interpolation_p95_deg=float(np.degrees(all_p95)),
        alias_branch=branch,alias_frequency_hz=alias_hz,alias_margin_hz=margins,
        alias_uncertainty_method='five_scale_max_jackknife_empirical_hac_and_header_blocks',
        alias_uncertainty_caveat='Model-based guard, not a calibrated confidence probability; long-range common bias is not ruled out.',
        heldout_phase_error_deg=np.degrees(heldout).tolist(),heldout_phase_error_p95_deg=float(np.degrees(p95)),
        heldout_scope='Conditional interpolation holdout after known-header alias selection, not independent reacquisition.',
        local_frequency_sigma=float(local_frequency_sigma),
        locally_relaxed_headers=[dict(index=int(i), residual_hz=float(residual[i]), standard_error_hz=float(se[i])) for i in np.flatnonzero(abs(residual)>5*np.maximum(se,1e-6))],
        alias_uncertainty_sigma=5.,
        training_check_scope='Independent QAM observations check relative phase consistency; a constant alias is not ruled out by offset-calibrated QAM alone.')


def resolve_dynamic_header_phase(times, angles, frequencies, se, frame_s):
    """Unwrap frequency increments before phase; use known headers only.

    This assumes smooth inter-header frequency, NOT a fixed carrier offset.
    The remaining common integer branch still requires the existing local
    frequency/uncertainty checks and held-out phase interpolation validation.
    No missing header is bridged. It cannot prove absence of payload-only
    excursions and does not use QAM decisions or nearest-neighbour EVM.
    """
    t=np.asarray(times,float);a=np.asarray(angles,float)
    f=np.asarray(frequencies,float);s=np.asarray(se,float)
    if not all(v.ndim==1 and len(v)==len(t) for v in (a,f,s)) or not all(np.all(np.isfinite(v)) for v in (t,a,f,s)) or np.any(s<0):
        raise ValueError('动态相位观测须为等长有限数组及非负不确定性')
    if not math.isfinite(frame_s) or frame_s<=0 or len(t)<12:
        raise ValueError('动态相位至少需要12个可信连续前导')
    frame=np.rint((t-t[0])/frame_s).astype(int)
    if np.any(np.diff(frame)!=1):
        raise ValueError('动态相位存在前导缺口，拒绝跨缺口恢复')
    if np.max(abs((t-t[0])-frame*frame_s))>frame_s*.01:
        raise ValueError('相位前导时间不符合已验证帧时钟')
    increments=np.angle(np.exp(1j*np.diff(a)))
    velocity=np.unwrap(increments)
    change=np.diff(velocity)
    if np.max(abs(change))>=.8*np.pi:
        raise ValueError('前导间动态频偏变化接近周数歧义边界')
    shape=np.r_[a[0],a[0]+np.cumsum(velocity)]
    # Allow modest finite short-header model error only in this dynamic branch.
    # This is not a calibrated confidence bound. Alias margins and all
    # holdout/training checks remain unchanged.
    phase,evidence=_validate_phase_candidate(t,shape,f,s,frame_s,frame,interpolate=cubic_phase,slopes=cubic_phase_slopes,local_frequency_sigma=5.5)
    # A few large isolated changes must not hide under the P95 criterion.
    # Keep the existing .20 rad P95 and apply the same bound to every
    # omitted header in this new branch, including early calibration knots.
    leaveout=[]
    for k in range(1,len(t)-1):
        keep=np.arange(len(t))!=k
        leaveout.append(float(phase[k]-cubic_phase(t[keep],phase[keep],t[k])))
    if np.max(abs(np.asarray(leaveout)))>.20:
        raise ValueError('动态前导相位存在孤立跳变，拒绝局部周数假锁')
    evidence.update(tracking_strategy='known_header_dynamic_frequency',
        max_frequency_increment_phase_rad=float(np.max(abs(change))),
        max_interpolation_error_deg=float(np.degrees(np.max(abs(np.asarray(leaveout))))),
        relative_interval_turns=np.rint((velocity-increments)/(2*np.pi)).astype(int).tolist(),
        ambiguity_assumption='Inter-header frequency is smooth; an unobserved whole-cycle excursion inside unknown payload is not identifiable.',
        payload_used=False)
    return phase,evidence


def validate_training_phase(ids, eligible, delta, calibration):
    ids=np.asarray(ids);eligible=np.asarray(eligible);delta=np.asarray(delta)
    cal=ids<calibration;available_cal=eligible<calibration
    if np.count_nonzero(cal)<4 or np.count_nonzero(~cal)<6:
        raise ValueError('独立QAM训练不足以验证弱信号相位')
    coverage=[np.count_nonzero(cal)/max(1,np.count_nonzero(available_cal)),
              np.count_nonzero(~cal)/max(1,np.count_nonzero(~available_cal))]
    missing=~np.isin(eligible,ids);gap=longest=0
    for absent in missing:
        gap=gap+1 if absent else 0;longest=max(longest,gap)
    if min(coverage)<.8 or longest>2:
        raise ValueError('独立QAM训练验证覆盖不足或连续缺口过大')
    offset=float(np.angle(np.mean(np.exp(1j*delta[cal]))))
    calibration_error=np.angle(np.exp(1j*(delta[cal]-offset)))
    calibration_p95=float(np.percentile(abs(calibration_error),95))
    if calibration_p95>.20:raise ValueError('独立QAM校准区相位散布过大')
    residual=np.angle(np.exp(1j*(delta[~cal]-offset)))
    p95=float(np.percentile(abs(residual),95))
    if p95>.20:raise ValueError('独立QAM训练未通过PWM相位模型验证')
    return offset,p95,dict(training_calibration_phase_p95_deg=float(np.degrees(calibration_p95)),training_calibration_coverage=coverage[0],training_heldout_coverage=coverage[1],
        training_max_unchecked_frames=longest,training_phase_validation_headers=int(np.count_nonzero(~cal)))


def track_known_headers(x,fs,pwm,anchor,frame_period,training_context,cancel_event=None):
    if training_context is None:raise ValueError('弱信号相位回退需要独立已知QAM训练验证')
    period=fs/pwm['symbol_hz'];n=int(round(len(pwm['symbols'])*period))
    local=np.arange(n)/period;slot=np.minimum(local.astype(int),len(pwm['symbols'])-1);within=local-slot
    pulses=header_pulse_fractions(pwm)[slot];high=(within>=pulses[:,0])&(within<pulses.sum(axis=1))
    reach=max(.2,.5*pwm.get('observation_average_chips',0.))/pwm['total_chips']
    good=(within>reach)&(within<1-reach)&(abs(within-pulses[:,0])>reach)&(abs(within-pulses.sum(axis=1))>reach)
    first=math.ceil(-anchor/frame_period);last=math.floor((len(x)-n-2-anchor)/frame_period)
    times=[];angles=[];freq=[];errors=[];magnitudes=[];uncertainty=[];rejected=[]
    for k in range(first,last+1):
        _cancel(cancel_event);at=anchor+k*frame_period
        z=smooth_pwm_observation(_sample(x,at+np.arange(n)),fs,pwm)
        try:f,a,e,m,s=joint_header_fit(z,good,high,fs,period)
        except ValueError:rejected.append(at/fs);continue
        if e>pwm['max_error_percent'] or abs(f)>=pwm['max_frequency_hz'] or not np.all(np.isfinite([f,a,e,m,s])):
            rejected.append(at/fs);continue
        times.append((at+n/2)/fs);angles.append(a);freq.append(f);errors.append(e);magnitudes.append(m);uncertainty.append(s)
    t=np.asarray(times)
    try:
        phase,evidence=resolve_header_phase(t,angles,freq,uncertainty,frame_period/fs)
    except ValueError as original_error:
        # Preserve strict missing-header rejection. Dynamic unwrapping is not
        # a licence to interpolate across missing observations.
        if rejected or not math.isclose(training_context[3],10000.,rel_tol=.005):
            raise
        phase,evidence=resolve_dynamic_header_phase(t,angles,freq,uncertainty,frame_period/fs)
        evidence['static_rejection_reason']=str(original_error)
    interpolate=cubic_phase if evidence.get('tracking_strategy')=='known_header_dynamic_frequency' else pchip
    # An independent RF harmonic: only the four explicitly known QAM training
    # symbols. Fit one fixed PWM-to-training phase offset on calibration frames.
    from rx706_tdcm import harmonic_coefficients
    from rx706_qam_training import fit_training
    points,bound,tolerance,rate,selected,orientation,fit_dc=training_context
    delta=[];ids=[];eligible=[];orientations=[]
    for k,tt in enumerate(t):
        _cancel(cancel_event);start=tt*fs+n/2+pwm['gap_s']*fs
        if start+len(points)*fs/rate>=len(x):continue
        eligible.append(k)
        c,centers=harmonic_coefficients(x,fs,start,rate,selected,count=len(points),cancel_event=cancel_event)
        try:fit=fit_training(c,points,rate,bound,tolerance,orientation=orientation,fit_dc_offset=fit_dc,cancel_event=cancel_event)
        except ValueError:continue
        delta.append(float(np.angle(fit['gain'])-interpolate(t,phase,np.mean(centers)/fs)));ids.append(k);orientations.append(fit['conjugated'])
    if len(set(orientations))!=1:raise ValueError('独立QAM训练镜像在帧间不一致')
    offset,training_p95,coverage=validate_training_phase(ids,eligible,delta,evidence['calibration_headers'])
    evidence.update(coverage)
    end=(len(x)-1)/fs
    if t[0]>1.1*frame_period/fs or end-t[-1]>1.1*frame_period/fs:raise ValueError('边界相位外推超过一帧')
    dense=np.concatenate([np.linspace(a,b,33)[:-1] for a,b in zip(t[:-1],t[1:])]+[t[-1:]])
    out_t=np.r_[0.,dense,end];out_p=interpolate(t,phase,out_t)
    evidence.update(source='known_pwm_joint_fit_with_independent_qam_check',payload_used=False,
        header_count=len(t),rejected_header_times_s=rejected,header_times_s=t.tolist(),header_phase_rad=phase.tolist(),
        local_frequency_hz=freq,local_frequency_jackknife_se_hz=uncertainty,header_errors_percent=errors,
        training_phase_offset_rad=offset,training_phase_holdout_p95_deg=float(np.degrees(training_p95)),
        limitation='Smooth phase between known headers is assumed; arbitrary payload-only phase excursions are unobservable.')
    coherence=float(np.median(magnitudes)/max(np.sqrt(np.mean(abs(x[:min(len(x),n)])**2)),1e-15))
    return out_t,out_p,float(1/np.median(np.diff(t))),coherence,evidence

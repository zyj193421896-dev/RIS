"""Headerless two-state TDCM acquisition with temporal holdout validation.

Not a blind arbitrary modem: Fs, nominal symbol/base rate, harmonic and order
are priors. Repeat mode measures a repeated waveform; fixed mode requires the
actual clock (including manual ppm) and accepts nonrepeating data. All labels
remain candidates with quadrant, integer-symbol and protocol-frame ambiguity.
"""
import math
import time
from types import SimpleNamespace
import numpy as np
from rx706_spectrum import _cancel
from rx706_recovery import _read_span
from rx706_qam import decide
from rx706_tdcm import validate_tdcm,harmonic_coefficients
from rx706_tdcm_debug import (validate_compensation,_rotate_inplace,_decimate,
    _phase_track,_clock_repeat,_fit_grid,_evm)
from rx706_tdcm_auto import validate_search,_carrier_seeds,AutoSearchError,_fingerprint

ACQUISITION_MODES=('pwm','no_header_repeat','no_header_fixed')


def validate_acquisition(mode):
    if mode not in ACQUISITION_MODES:
        raise ValueError('TDCM获取方式须为pwm、no_header_repeat或no_header_fixed')
    return mode


def _two_state_quality(x):
    # Bounded evenly spaced samples, unfiltered original ADC cells. Fitting a
    # two-state RF model is independent of QAM decisions or unknown bit labels.
    z=x[np.linspace(0,len(x)-1,min(len(x),131072),dtype=int)]
    z=z-z.mean();power=float(np.mean(abs(z)**2))
    if power<1e-8:raise ValueError('无前导模式没有可用交流能量')
    direction=np.exp(.5j*np.angle(np.mean(z*z)))
    axis=(z/direction).real;split=float(np.median(axis))
    for _ in range(8):
        high=axis>split
        if min(high.mean(),1-high.mean())<.03:raise ValueError('没有两种充分出现的RF状态')
        low=z[~high].mean();top=z[high].mean();split=float(((low+top)/direction).real/2)
    separation=abs(top-low)
    error=float(np.sqrt(np.mean(abs(z-np.where(high,top,low))**2))/max(separation,1e-12))
    linearity=float(abs(np.mean(z*z))/max(power,1e-15))
    # Received pulse edges may ring and include broadband noise. Require an
    # independent RF line reference rather than unrealistically ideal levels;
    # downstream frozen lattice + repetition tests remain mandatory.
    if error>.4 or linearity<.35:
        raise ValueError(f'公共载波/两态参考不足：状态残差 {100*error:.2f}%，轴向相干度 {linearity:.3f}（须≤40%、≥0.35）')
    return dict(relative_rms=error,linearity=linearity,high_fraction=float(high.mean()),separation_adc=float(separation))


def _lattice(z,order):
    ideal,labels,_=decide(z,order);spacing=2/math.sqrt(2*(order-1)/3)
    geometry=float((abs(np.mean(np.exp(2j*np.pi*z.real/spacing)))+
                    abs(np.mean(np.exp(2j*np.pi*z.imag/spacing))))/2)
    return dict(nn_evm_percent=_evm(z,ideal),geometry=geometry,
                occupied_labels=int(len(np.unique(labels))))


def _repeat_clock(y,fs,expected,ppm,cancel):
    period,info=_clock_repeat(y,fs,expected,ppm,cancel)
    # The legacy estimator may retain only even lags of a false half-period.
    # Require the candidate's FIRST lag itself, and a small fitted lag residual.
    n=int(min(len(y)-math.ceil(period)-2,4*period))
    if n<16:raise ValueError('直接重复窗口不足')
    pos=np.arange(n)+period;grid=np.arange(len(y))
    a=y[:n]-np.mean(y[:n]);b=np.interp(pos,grid,y.real)+1j*np.interp(pos,grid,y.imag);b-=b.mean()
    coherence=float(abs(np.vdot(a,b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-15))
    residual=np.asarray(info['lag_residual_samples'])
    if coherence<.7 or np.percentile(abs(residual),90)>max(1.,.002*period):
        raise ValueError('候选首个重复间隔不匹配；拒绝只由偶数倍滞后形成的假半周期')
    info['first_lag_coherence']=coherence
    return period,info


def _coefficients(y,fs,start,rate,selected,n,cancel):
    return harmonic_coefficients(y,fs,start,rate,selected,count=n,cancel_event=cancel)[0]


def _timing(y,fs,start,rate,selected,ndata,ncal,repeat,order,cancel):
    """Fit a single global timing + complex gain on calibration cells only."""
    sps=fs/rate
    if repeat:
        frame=sps*ndata;length=max(16,int(round(frame)))
        grid=np.arange(len(y));pos=np.arange(length)*frame/length;mean=np.zeros(length,complex)
        for k in range(ncal):
            _cancel(cancel);p=start+k*frame+pos
            mean+=np.interp(p,grid,y.real)+1j*np.interp(p,grid,y.imag)
        mean/=ncal;yfit=np.tile(mean,3);fit_fs=fs*length/frame;at=length
        ns=ndata
    else:
        yfit=y;fit_fs=fs;at=start;ns=min(2048,ncal*ndata)
    best=None
    # Full symbol uncertainty, then local refinement. No per-symbol phase fit.
    offsets=np.linspace(0,1,51,endpoint=False)
    for step in (1/51,1/255):
        for offset in offsets:
            _cancel(cancel)
            c=_coefficients(yfit,fit_fs,at+offset*fit_fs/rate,rate,selected,ns,cancel)
            error,gain,ideal=_fit_grid(c,order,cancel)
            if best is None or error<best[0]:best=(error,float(offset),gain)
        offsets=best[1]+np.arange(-5,6)*step/5
    return best[1]


def _evaluate(y,fs,frame,rate,selected,ndata,nblocks,ncal,order,repeat,geometry,cancel):
    # Ignore RF-filter edges and leave one full block before acquisition.
    start=frame
    offset=_timing(y,fs,start,rate,selected,ndata,ncal,repeat,order,cancel)
    starts=start+np.arange(nblocks)*frame+offset*fs/rate
    parts=[_coefficients(y,fs,p,rate,selected,ndata,cancel) for p in starts]
    matrix=np.array(parts);train=matrix[:ncal]
    fit=train.mean(axis=0) if repeat else train.ravel()[:2048]
    _,gain,_=_fit_grid(fit,order,cancel)
    normalized=matrix/gain;template=normalized[:ncal].mean(axis=0) if repeat else None
    validation=normalized[ncal:].ravel();metric=_lattice(validation,order)
    metric['repeat_evm_percent']=_evm(normalized[ncal:],np.broadcast_to(template,normalized[ncal:].shape)) if repeat else None
    metric['calibration_blocks']=ncal;metric['validation_blocks']=nblocks-ncal
    if metric['geometry']<geometry or metric['occupied_labels']<order//2:
        raise ValueError(f"独立选择段网格不足：结构 {metric['geometry']:.3f}，标签 {metric['occupied_labels']}/{order}")
    return dict(offset=offset,gain=gain,starts=starts,metrics=metric,template=template)


def recover_headerless_tdcm(spectrum,*,duration_s=1,center_hint_hz=-200000,
        symbol_hz=19531.25,order=256,tdcm_options=None,preamble_options=None,
        cancel_event=None,progress=None):
    source=dict(tdcm_options or {});mode=validate_acquisition(source.get('acquisition_mode','pwm'))
    if mode=='pwm':raise ValueError('无前导接收入口不接受PWM模式')
    if preamble_options is not None:
        raise ValueError('本无前导入口不使用已知QAM训练；请关闭训练，结果保留映射歧义')
    if order not in (16,64,256):raise ValueError('QAM阶数须为16、64或256')
    fs=spectrum.sample_rate_hz;cfg=validate_tdcm(fs,symbol_hz,source)
    if cfg['normalization']!='blind':raise ValueError('无PWM前导须选择盲归一化，不能使用PWM两态幅相标定')
    comp=validate_compensation(source.get('compensation',{}));search=validate_search(source.get('auto_search',{}))
    if center_hint_hz is None or not math.isfinite(center_hint_hz):raise ValueError('无前导模式仍须给定有限的原载波搜索中心')
    repeat=mode=='no_header_repeat';automatic=search['enabled']
    if not repeat and comp['clock_mode']=='repeat':raise ValueError('随机载荷模式不能使用整帧重复时钟；请用配置周期或手动ppm')
    if source.get('payload_symbols',0):raise ValueError('无前导模式请将提取符号数设为0，使用每块窗口数及最多块数')
    ndata_values=search['frame_candidates'] if automatic and repeat else [comp['frame_symbols']]
    phases=('square','mean') if automatic else (comp['phase_mode'],)
    if not automatic and comp['phase_mode']=='off':
        raise ValueError('无前导入口须启用两态平方载波或载波均值；不能把未知公共相位当作已补偿')
    input_stat=spectrum.path.stat()
    started=time.monotonic();raw,count,clipped=_read_span(spectrum,duration_s,center_hint_hz,cancel_event,
        max_samples=min(math.ceil(fs),comp['max_input_samples']))
    fraction=search['selection_fraction'];prefix=int(len(raw)*fraction)
    if prefix<4096:raise ValueError('无前导独立选择片段不足4096点')
    if automatic:seeds,bounds=_carrier_seeds(raw[:prefix],fs,center_hint_hz,search['carrier_span_hz'],search['max_carriers'],cancel_event)
    else:seeds=[dict(carrier_hz=float(center_hint_hz),source='手动中心')];bounds=dict(low_hz=center_hint_hz,high_hz=center_hint_hz)
    rows=[];eligible=[];budget=False
    def update(message):
        _cancel(cancel_event)
        if progress:progress(message)
    for seed in seeds:
        for phase_mode in phases:
            update(f'无前导：载波 {seed["carrier_hz"]:+.2f} Hz / {phase_mode}')
            x=raw.copy();_rotate_inplace(x,fs,seed['carrier_hz'],cancel_event=cancel_event)
            y,coarse_fs,factor=_decimate(x,fs,max(500000.,48*cfg['base_hz']),cancel_event)
            before=y.copy();po=dict(comp,phase_mode=phase_mode)
            try:
                pt,phase,actual_update,coherence=_phase_track(y,coarse_fs,po,cancel_event)
                _rotate_inplace(y,coarse_fs,phase_t=pt,phase=phase,cancel_event=cancel_event)
                _rotate_inplace(x,fs,phase_t=pt,phase=phase,cancel_event=cancel_event)
                state=_two_state_quality(x[:prefix])
            except ValueError as exc:
                rows.append(dict(id=f'R{len(rows)+1}',carrier_hz=seed['carrier_hz'],phase_mode=phase_mode,status='拒绝',reason=str(exc)))
                continue
            for ndata in ndata_values:
                if len(rows)>=search['max_trials'] or time.monotonic()-started>search['max_seconds']:
                    budget=True;break
                row=dict(id=f'R{len(rows)+1}',carrier_hz=seed['carrier_hz'],phase_mode=phase_mode,
                         block_symbols=ndata,status='拒绝',reason='')
                rows.append(row)
                try:
                    if repeat:
                        frame,clock=_repeat_clock(y[:prefix//factor],coarse_fs,coarse_fs/symbol_hz*ndata,
                                                  search['clock_ppm'] if automatic else comp['clock_search_ppm'],cancel_event)
                        rate=coarse_fs*ndata/frame
                    else:
                        scale=1+comp['period_ppm']*1e-6 if comp['clock_mode']=='manual' else 1.
                        rate=symbol_hz/scale;frame=coarse_fs/rate*ndata
                        clock=dict(note='用户提供的固定时钟；不是随机载荷的自动钟差估计',lags=[],lag_residual_samples=[])
                    selected=cfg['harmonic']*cfg['base_hz']*rate/symbol_hz
                    if abs(seed['carrier_hz'])+abs(selected)+rate>=fs/2:raise ValueError('目标谐波带宽超出采样范围')
                    nblocks=min(int((prefix/factor-frame)/frame)-2,comp['max_frames'])
                    ncal=min(comp['calibration_frames'],max(2,nblocks//2))
                    if nblocks-ncal<3:raise ValueError('标定与独立选择至少需要5个完整窗口块')
                    fit=_evaluate(y,coarse_fs,frame,rate,selected,ndata,nblocks,ncal,order,repeat,search['minimum_geometry'],cancel_event)
                    m=fit['metrics'];score=m['repeat_evm_percent'] if repeat else m['nn_evm_percent']
                    if score>comp['quality_evm_percent'] or m['nn_evm_percent']>comp['quality_evm_percent']:
                        raise ValueError(f'独立选择误差 {score:.3f}% 未通过上限')
                    row.update(status='可用候选',metrics=m,period_samples=fs/rate,score=score)
                    eligible.append(dict(row=row.copy(),seed=seed,phase_mode=phase_mode,rate=rate,frame=frame,
                        factor=factor,coarse_fs=coarse_fs,ndata=ndata,ncal=ncal,fit=fit,clock=clock))
                except ValueError as exc:row['reason']=str(exc)
            del x,y,before
            if budget:break
        if budget:break
    failure=dict(status='failed',mode=mode,candidates=rows,ranges=search,carrier_bounds=bounds,
                 note='无PWM获取未锁定；最近邻误差不是BER')
    if not eligible:raise AutoSearchError('无前导模式没有组合通过两态模型及独立网格/重复验证',failure)
    # Multiples describe the same repeated signal, not distinct physical frame
    # lengths. Prefer the shortest validated interval within 0.5 EVM point.
    minimum=min(v['row']['score'] for v in eligible)
    close=[v for v in eligible if v['row']['score']<=minimum+.5]
    winner=min(close,key=lambda v:(v['ndata'],v['row']['score'])) if repeat else min(eligible,key=lambda v:v['row']['score'])
    seed=winner['seed'];phase_mode=winner['phase_mode'];rate=winner['rate'];ndata=winner['ndata'];ncal=winner['ncal']
    frame=winner['frame'];factor=winner['factor'];coarse_fs=winner['coarse_fs'];selected=cfg['harmonic']*cfg['base_hz']*rate/symbol_hz
    update('无前导：冻结已选模型，提取原始采样点并验证最终后段')
    x=raw.copy();_rotate_inplace(x,fs,seed['carrier_hz'],cancel_event=cancel_event)
    y,_,_=_decimate(x,fs,max(500000.,48*cfg['base_hz']),cancel_event);before=y.copy()
    pt,phase,actual_update,coherence=_phase_track(y,coarse_fs,dict(comp,phase_mode=phase_mode),cancel_event)
    _rotate_inplace(y,coarse_fs,phase_t=pt,phase=phase,cancel_event=cancel_event)
    _rotate_inplace(x,fs,phase_t=pt,phase=phase,cancel_event=cancel_event)
    frame_raw=frame*factor;offset=winner['fit']['offset']+comp['timing_offset_symbols']
    anchor=frame_raw+offset*fs/rate
    nfull=min(int((len(x)-anchor)/frame_raw),comp['max_frames'])
    starts=anchor+np.arange(nfull)*frame_raw
    coefficients=np.array([_coefficients(x,fs,p,rate,selected,ndata,cancel_event) for p in starts])
    # Refit only a global gain on original ADC calibration cells. Clock/timing
    # were frozen above; final holdout does not participate in either decision.
    ctrain=coefficients[:ncal].mean(axis=0) if repeat else coefficients[:ncal].ravel()[:2048]
    _,gain,_=_fit_grid(ctrain,order,cancel_event)
    z=coefficients/gain/cfg['amplitude_scale']*np.exp(-1j*math.radians(cfg['phase_deg']))
    template=z[:ncal].mean(axis=0) if repeat else None
    select=(starts>=starts[ncal])&(starts+frame_raw<=prefix)
    test=starts>=prefix
    if np.count_nonzero(test)<3:raise AutoSearchError('最终独立后段不足3个完整窗口块',failure)
    selection=_lattice(z[select].ravel(),order);final=_lattice(z[test].ravel(),order)
    if repeat:
        selection['repeat_evm_percent']=_evm(z[select],np.broadcast_to(template,z[select].shape))
        final['repeat_evm_percent']=_evm(z[test],np.broadcast_to(template,z[test].shape))
    else:selection['repeat_evm_percent']=None;final['repeat_evm_percent']=None
    threshold=comp['quality_evm_percent']
    passed=all(m['geometry']>=search['minimum_geometry'] and m['nn_evm_percent']<=threshold and
               m['occupied_labels']>=order//2 and (not repeat or m['repeat_evm_percent']<=threshold) for m in (selection,final))
    state_test=_two_state_quality(x[prefix:]);status='candidate_validated' if passed else 'test_failed'
    adopted=dict(qam_order=order,bits_per_symbol=int(math.log2(order)),acquisition_mode=mode,carrier_seed_hz=seed['carrier_hz'],phase_mode=phase_mode,
        clock_mode='repeat' if repeat else comp['clock_mode'],period_samples=fs/rate,period_ppm=(symbol_hz/rate-1)*1e6,
        symbol_hz=rate,base_hz=cfg['base_hz']*rate/symbol_hz,harmonic=cfg['harmonic'],block_symbols=ndata,
        timing_offset_symbols=offset,anchor_sample=anchor,gain_real=float(gain.real),gain_imag=float(gain.imag),
        phase_update_hz=actual_update,phase_bandwidth_hz=comp['phase_bandwidth_hz'],normalization='blind',
        amplitude_scale=cfg['amplitude_scale'],manual_phase_deg=cfg['phase_deg'],
        timing_search_symbols=[0,1],block_processing='all_for_independent_validation',
        pwm_used=False,training_used=False)
    acquisition=dict(status=status,mode=mode,adopted=adopted,ranges=search,carrier_bounds=bounds,
        candidates=rows,winner_id=winner['row']['id'],selection_metrics=selection,final_test_metrics=final,
        partitions=dict(calibration_blocks=ncal,selection_end_sample=prefix,final_start_sample=prefix),
        two_state_selection=_two_state_quality(x[:prefix]),two_state_test=state_test,budget_stopped=budget,
        input_fingerprint=_fingerprint(spectrum,count),
        reproducible_options=dict(duration_s=duration_s,center_hint_hz=center_hint_hz,symbol_hz=symbol_hz,
            order=order,tdcm_options={k:v for k,v in source.items() if not k.startswith('_')}),
        note='最短通过的重复窗口不是已识别协议帧；随机模式不估计钟差。独立验证检验结构，不验证未知比特；90°及整数符号偏移歧义仍在。')
    rows_out=[];flat=z.ravel();ideal,labels,bits=decide(flat,order)
    for k,p in enumerate(starts):
        row=dict(index=k,header_time_s=None,data_time_s=(p+spectrum.first_tuple)/fs,header_error_percent=None,
                 pilot_evm_percent=None,valid=passed,calibration=k<ncal,symbols=ndata,
                 nn_evm_percent=_evm(z[k],decide(z[k],order)[0]),
                 template_evm_percent=_evm(z[k],template) if repeat else None)
        rows_out.append(row)
    origins=spectrum.first_tuple/fs;times=(starts[:,None]+(np.arange(ndata)+.5)*fs/rate+spectrum.first_tuple)/fs
    bc=_coefficients(before,coarse_fs,frame,symbol_hz,cfg['harmonic']*cfg['base_hz'],min(8192,int((len(before)-frame)/(coarse_fs/symbol_hz))),cancel_event)
    cg=gain*np.sinc(selected/coarse_fs)/np.sinc(selected/fs)*np.exp(-1j*np.pi*selected*(1/coarse_fs-1/fs))
    warning=['没有发端净荷参考，未测BER；输出只有候选Gray标签/比特。',
        '无前导不能确定协议帧起点；存在整数符号移位和90°映射歧义。',
        '两态RF公共相位跟踪不等于逐个QAM符号消相位。']
    if not passed:warning.append('最终独立后段或重新提取选择段未通过；不重新选择模型，不得将本结果标作锁定。')
    if state_test['relative_rms']>.2:warning.append('接收两态有明显噪声/振铃；模型状态残差偏大，结构结论依赖独立后段网格/重复验证。')
    if spectrum.rate_source!='同名 JSON':warning.append('Fs来自手动/默认值，速率和Hz依赖该假设。')
    d=dict(options=dict(comp,phase_mode=phase_mode),quality='无前导结构通过；盲估候选（未验证比特）' if passed else '无前导候选未通过独立验证',
        structural_validation=passed,ber=None,rate_source=spectrum.rate_source,warnings=warning,clock=winner['clock'],
        nominal_symbol_hz=symbol_hz,measured_symbol_hz=rate,period_samples=fs/rate,period_ppm=adopted['period_ppm'],
        timing_offset_symbols=offset,phase_time_s=pt+origins,phase_rad=phase,frequency_hz=np.gradient(phase,pt)/(2*np.pi)+seed['carrier_hz']-center_hint_hz,
        phase_update_hz=actual_update,phase_coherence=coherence,before_scatter=bc/cg,gain_iq=gain,
        gain_source='前段盲网格复增益；无PWM、无真实训练，保留旋转歧义',heldout_template_evm_percent=final['repeat_evm_percent'],
        nearest_evm_percent=_evm(flat,ideal),occupied_labels=len(np.unique(labels)),frames=rows_out,frame_period_samples=frame_raw,
        parameter_source='Fs/标称速率/谐波为用户先验；两态载波参考；前段盲标定；最终后段独立验证',acquisition=acquisition)
    details=dict(cfg,original_carrier_hz=center_hint_hz,corrected_carrier_hz=seed['carrier_hz'],target_hz=seed['carrier_hz']+selected,
        coefficient_units='ADC counts',calibration_note=d['gain_source'],acquisition_mode=mode)
    final_stat=spectrum.path.stat()
    if (input_stat.st_size,input_stat.st_mtime_ns)!=(final_stat.st_size,final_stat.st_mtime_ns):
        raise ValueError('分析期间输入文件发生改变，丢弃本次无前导结果')
    result=SimpleNamespace(kind='qam',carrier_hz=center_hint_hz,order=order,symbol_hz=rate,scatter=flat,
        ideal=ideal,labels=labels,bits=bits,evm_percent=d['nearest_evm_percent'],time_s=times.ravel(),
        first_tuple=spectrum.first_tuple,input_samples=count,clipped=clipped,cutoff_hz=rate/2,decimation=1,output_rate_hz=rate,
        preamble=None,pwm_preamble=None,tdcm=details,coefficients=coefficients.ravel(),raw_coefficients=coefficients.ravel().copy(),
        diagnostics=d,frame_ids=np.repeat(np.arange(nfull),ndata),frames=rows_out)

    from rx706_qam_payload import attach_payload_debug
    return attach_payload_debug(result)


def replay_headerless_config(spectrum,recipe,*,cancel_event=None):
    if not isinstance(recipe,dict) or recipe.get('version')!=1 or recipe.get('method')!='headerless_tdcm_config':
        raise ValueError('无前导配置格式不支持')
    fp=recipe.get('input_fingerprint',{})
    try:count=int(fp['count'])
    except (KeyError,TypeError,ValueError):raise ValueError('无前导配置缺少输入指纹') from None
    if not 4096<=count<=33554432:raise ValueError('无前导配置输入长度无效')
    if _fingerprint(spectrum,count)!=fp:raise ValueError('无前导配置的IQ片段指纹/通道/Fs不匹配')
    kwargs=recipe.get('reproducible_options')
    if not isinstance(kwargs,dict):raise ValueError('无前导配置缺少处理参数')
    r=recover_headerless_tdcm(spectrum,**kwargs,cancel_event=cancel_event)
    r.diagnostics['warnings'].append('已核验输入指纹并按相同配置重新估计；此JSON不是冻结标定模板。')
    return r

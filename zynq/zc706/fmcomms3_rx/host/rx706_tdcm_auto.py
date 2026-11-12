"""Bounded TDCM model/compensation search with frozen, temporal holdout tests.

Assumptions: known sample format/Fs, nominal symbol/base/PWM rate, known PWM
sequence, and a repeated full waveform. This is not a blind arbitrary modem.
All scores describe repeated structure; no unknown payload BER is inferred.
"""
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from rx706_spectrum import _cancel,AnalysisCancelled
from rx706_recovery import _read_span
from rx706_qam import decide
from rx706_qam_pwm import acquire_pwm_header,validate_pwm_header
from rx706_tdcm import validate_tdcm,_pwm_reference
from rx706_tdcm_debug import (validate_compensation,recover_compensated_tdcm,
    _decimate,_rotate_inplace,_phase_track,_clock_repeat)


class AutoSearchError(ValueError):
    def __init__(self,message,diagnostics=None):
        super().__init__(message);self.diagnostics=diagnostics or {}


def validate_search(options):
    if not isinstance(options,dict):raise ValueError('自动搜索配置须为对象')
    o=dict(options)
    if not isinstance(o.get('enabled',False),bool):raise ValueError('自动搜索enabled须为布尔值')
    def number(key,default,lo,hi,integer=False):
        try:v=float(o.get(key,default))
        except (ValueError,TypeError):raise ValueError('自动搜索'+key+'须为数值') from None
        if not math.isfinite(v) or not lo<=v<=hi or (integer and not v.is_integer()):raise ValueError(f'自动搜索{key}须在{lo}～{hi}'+('内取整数' if integer else '内'))
        return int(v) if integer else v
    text=o.get('frame_candidates','64 128 256 512')
    try:frames=[int(x) for x in text.replace(',',' ').split()] if isinstance(text,str) else [int(x) for x in text]
    except (ValueError,TypeError):raise ValueError('每帧窗口候选须为整数列表') from None
    if not isinstance(text,str) and any(float(v)!=int(v) for v in text):raise ValueError('每帧窗口候选不能为小数')
    frames=list(dict.fromkeys(frames))
    if not frames or len(frames)>6 or any(not 16<=v<=2048 for v in frames):raise ValueError('每帧窗口候选须为1～6个16～2048的整数')
    o.update(enabled=o.get('enabled',False),frame_candidates=frames,
        carrier_span_hz=number('carrier_span_hz',500000,1000,30000000),
        clock_ppm=number('clock_ppm',1000,1,5000),
        max_carriers=number('max_carriers',3,1,4,True),
        max_trials=number('max_trials',12,1,24,True),
        max_seconds=number('max_seconds',60,5,180),
        selection_fraction=number('selection_fraction',.6,.45,.75),
        minimum_coverage=number('minimum_coverage',.8,.5,1),
        minimum_geometry=number('minimum_geometry',.55,.3,.95))
    return o


def _fingerprint(spectrum,count):
    h=hashlib.sha256()
    with spectrum.path.open('rb') as stream:
        stream.seek(spectrum.first_tuple*8);remaining=count*8
        while remaining:
            data=stream.read(min(4*1024*1024,remaining))
            if not data:raise ValueError('输入在指纹校验时被截短')
            h.update(data);remaining-=len(data)
    return dict(sha256=h.hexdigest(),count=count,first_tuple=spectrum.first_tuple,
                channel=spectrum.channel,sample_rate_hz=spectrum.sample_rate_hz,file_bytes=spectrum.total_tuples*8)


def _carrier_seeds(x,fs,hint,span,budget,cancel_event):
    length=min(262144,1 << (len(x).bit_length()-1))
    if length<4096:raise ValueError('载波候选搜索至少需要4096点')
    window=np.hanning(length+1)[:-1];freq=np.fft.fftfreq(length,1/fs)
    seeds=[];lo=max(-fs/2,hint-span);hi=min(fs/2,hint+span)
    for square in (False,True):
        _cancel(cancel_event);power=np.zeros(length)
        for at in (0,max(0,len(x)-length)):
            y=x[at:at+length];y=y*y if square else y
            power+=abs(np.fft.fft(y*window))**2
        floor=max(float(np.median(power)),1e-30)
        peaks=np.flatnonzero((power>np.roll(power,1))&(power>=np.roll(power,-1)))
        for j in peaks[np.argsort(power[peaks])[-32:][::-1]]:
            snr=10*math.log10(max(power[j]/floor,1e-30))
            if snr<12:continue
            logs=np.log(np.maximum(power[[(j-1)%length,j,(j+1)%length]],1e-30))
            denom=logs[0]-2*logs[1]+logs[2]
            delta=float(.5*(logs[0]-logs[2])/denom) if denom else 0.
            f=freq[j]+np.clip(delta,-.5,.5)*fs/length
            candidates=[f] if not square else [f/2,f/2+fs/2,f/2-fs/2]
            for value in candidates:
                if lo<=value<=hi:
                    seeds.append(dict(carrier_hz=float(value),spectral_contrast_db=float(snr),source='平方谱' if square else '原始谱'))
    # Each spectrum's contrast is a proposal score, not calibrated SNR.
    seeds.sort(key=lambda v:-v['spectral_contrast_db'])
    selected=[]
    for seed in seeds:
        if all(abs(seed['carrier_hz']-v['carrier_hz'])>500 for v in selected):selected.append(seed)
        if len(selected)>=budget:break
    if not selected or (len(selected)<budget and not any(abs(v['carrier_hz']-hint)<500 for v in selected)):
        selected.append(dict(carrier_hz=float(hint),spectral_contrast_db=None,source='配置中心'))
    return selected,dict(low_hz=lo,high_hz=hi,nfft=length,note='有限谱峰候选；谱对比度不是SNR；不保证全频段最优')


def _frame_matrix(result):
    ids=np.unique(result.frame_ids)
    arrays=[result.scatter[result.frame_ids==k] for k in ids]
    if not arrays:return ids,np.empty((0,0),complex)
    length=len(arrays[0]);good=[i for i,z in enumerate(arrays) if len(z)==length]
    return ids[good],np.array([arrays[i] for i in good])


def _metrics(result,order,calibration_frames,minimum_geometry,template=None,after_s=None):
    ids,matrix=_frame_matrix(result)
    if len(matrix)<6:raise ValueError('独立标定/选择至少需要6个有效帧')
    ntrain=min(calibration_frames,max(2,int(len(matrix)*.45)))
    train=matrix[:ntrain]
    if template is None:template=train.mean(axis=0)
    if len(template)!=matrix.shape[1]:raise ValueError('冻结模板长度与数据窗口数不符')
    selected=np.arange(len(matrix))>=ntrain
    if after_s is not None:
        starts={v['index']:v['data_time_s'] for v in result.frames}
        selected=np.array([starts[k]>=after_s for k in ids])
    check=matrix[selected]
    if len(check)<3:raise ValueError('独立验证至少需要3个有效数据帧')
    denominator=max(float(np.mean(abs(template)**2)),1e-30)
    evms=100*np.sqrt(np.mean(abs(check-template)**2,axis=1)/denominator)
    averaged=train.mean(axis=0);ideal,labels,_=decide(averaged,order)
    grid_evm=float(100*np.sqrt(np.mean(abs(averaged-ideal)**2)/np.mean(abs(ideal)**2)))
    spacing=2/math.sqrt(2*(order-1)/3)
    geometry=float((abs(np.mean(np.exp(2j*np.pi*averaged.real/spacing)))+abs(np.mean(np.exp(2j*np.pi*averaged.imag/spacing))))/2)
    occupied=len(np.unique(labels))
    return dict(repeat_evm_percent=float(np.median(evms)),repeat_evm_p90_percent=float(np.percentile(evms,90)),
        geometry=geometry,training_grid_evm_percent=grid_evm,occupied_labels=occupied,
        calibration_frames=ntrain,validation_frames=len(check),template_power=denominator),template


def _json_template(z):return [[float(v.real),float(v.imag)] for v in z]


def _candidate_row(index,stage,carrier,frame,phase,clock,gain):
    return dict(id=index,stage=stage,carrier_hz=float(carrier),frame_symbols=int(frame),
                phase_mode=phase,clock_mode=clock,gain_mode=gain,status='待评估',reason='')


def search_compensations(spectrum,*,duration_s=1,center_hint_hz=-200000,symbol_hz=19531.25,
        order=256,tdcm_options=None,pwm_preamble_options=None,preamble_options=None,
        cancel_event=None,progress=None):
    started=time.monotonic();source=dict(tdcm_options or {})
    input_stat=spectrum.path.stat()
    search=validate_search(source.get('auto_search',{}));cfg=validate_tdcm(spectrum.sample_rate_hz,symbol_hz,source)
    comp=validate_compensation(source.get('compensation',{}));pwm=validate_pwm_header(pwm_preamble_options or dict(symbol_hz=19531.25),spectrum.sample_rate_hz)
    fs=spectrum.sample_rate_hz
    if center_hint_hz is None or not math.isfinite(center_hint_hz):raise ValueError('自动搜索仍需有限的载波范围中心')
    if order not in (16,64,256):raise ValueError('自动搜索须先指定16/64/256QAM阶数')
    if source.get('payload_symbols',0):raise ValueError('自动搜索请将原“提取符号数”设为0，由每帧候选窗口列表决定')
    total=min(round(duration_s*fs),spectrum.total_tuples-spectrum.first_tuple,comp['max_input_samples'],math.ceil(fs))
    selection_count=int(total*search['selection_fraction'])
    prefix_s=selection_count/fs
    if prefix_s<.001:raise ValueError('自动搜索片段不足')
    raw,count,_=_read_span(spectrum,prefix_s,center_hint_hz,cancel_event,max_samples=selection_count)
    seeds,bounds=_carrier_seeds(raw,fs,center_hint_hz,search['carrier_span_hz'],search['max_carriers'],cancel_event)
    rows=[];hypotheses=[]
    def update(message):
        _cancel(cancel_event)
        if progress:progress(message)
    pwm_args=dict(pwm,symbols=' '.join(map(str,pwm['symbols'])),search_s=min(pwm['search_s'],prefix_s))
    for ci,seed in enumerate(seeds):
        update(f'自动补偿：载波/帧模型 {ci+1}/{len(seeds)}')
        if time.monotonic()-started>search['max_seconds']:break
        x=raw.copy();carrier=seed['carrier_hz']
        try:
            _rotate_inplace(x,fs,carrier,cancel_event=cancel_event)
            sync=acquire_pwm_header(x,fs,pwm_args,cancel_event)
            cfo,_,_,header_error=_pwm_reference(x,fs,sync,pwm,cancel_event)
            _rotate_inplace(x,fs,cfo,cancel_event=cancel_event)
            y,coarse_fs,factor=_decimate(x,fs,max(500000.,48*cfg['base_hz']),cancel_event)
        except ValueError as exc:
            row=_candidate_row(f'C{ci+1}','粗搜索',carrier,comp['frame_symbols'],'—','—','—')
            row.update(status='拒绝',reason=str(exc));rows.append(row);continue
        finally:
            del x
        for phase_mode in ('off','mean','square'):
            try:
                po=dict(comp,phase_mode=phase_mode)
                pt,phase,_,strength=_phase_track(y,coarse_fs,po,cancel_event)
                z=y.copy()
                if phase_mode!='off':_rotate_inplace(z,coarse_fs,phase_t=pt,phase=phase,cancel_event=cancel_event)
            except ValueError as exc:
                row=_candidate_row(f'C{ci+1}-{phase_mode}','粗搜索',carrier,comp['frame_symbols'],phase_mode,'—','—')
                row.update(status='拒绝',reason=str(exc));rows.append(row);continue
            for frame in search['frame_candidates']:
                row=_candidate_row(f'C{ci+1}-{phase_mode}-{frame}','粗搜索',carrier,frame,phase_mode,'repeat','—')
                rows.append(row)
                try:
                    expected=len(pwm['symbols'])*fs/pwm['symbol_hz']+pwm['gap_s']*fs+frame*fs/symbol_hz
                    period,clock=_clock_repeat(z,coarse_fs,expected/factor,search['clock_ppm'],cancel_event)
                    prefix_frames=int(count/(period*factor));calibration=min(comp['calibration_frames'],max(2,int(prefix_frames*.4)))
                    if prefix_frames-calibration<3:raise ValueError('标定后不足3个方案选择帧')
                    row.update(status='进入候选',coherence=clock['coherence'],period_ppm=(period*factor/expected-1)*1e6)
                    normals=('pwm','blind') if preamble_options is None else ('pilot',)
                    for clock_mode in ('repeat','off'):
                        for normal in normals:
                            hypotheses.append(dict(carrier=carrier,frame=frame,phase=phase_mode,clock=clock_mode,
                                normal=normal,calibration=calibration,coherence=clock['coherence'],header_error=header_error))
                except ValueError as exc:row.update(status='拒绝',reason=str(exc))
            del z
        del y
    del raw
    # High repeat correlation is a model proposal, never the final acceptance.
    hypotheses.sort(key=lambda h:(-h['coherence'],h['header_error'],h['normal']!='blind',h['clock']!='repeat'))
    # Round-robin top model families so a single phase variant cannot consume
    # the entire refinement budget with duplicate combinations.
    families={}
    for h in hypotheses:families.setdefault((h['carrier'],h['frame'],h['phase']),[]).append(h)
    ordered=[]
    while any(families.values()):
        for values in families.values():
            if values:ordered.append(values.pop(0))
    accepted=[];budget_stopped=False
    for ti,h in enumerate(ordered[:search['max_trials']]):
        if time.monotonic()-started>search['max_seconds']:budget_stopped=True;break
        update(f"自动补偿：组合 {ti+1}/{min(len(ordered),search['max_trials'])} · {h['phase']} / {h['clock']} / {h['normal']}")
        row=_candidate_row(f'T{ti+1}','精评',h['carrier'],h['frame'],h['phase'],h['clock'],h['normal']);rows.append(row)
        trial_comp=dict(comp,enabled=True,clock_mode=h['clock'],period_ppm=0,clock_search_ppm=search['clock_ppm'],
            phase_mode=h['phase'],frame_mode='all',frame_symbols=h['frame'],calibration_frames=h['calibration'],
            fit_timing=(h['clock']=='repeat' and h['normal']=='blind' and preamble_options is None),timing_offset_symbols=0)
        trial_tdcm=dict(cfg,payload_symbols=0,normalization='pwm' if h['normal']=='pwm' else 'blind',
                        amplitude_scale=1.,phase_deg=0.,compensation=trial_comp)
        try:
            trial=recover_compensated_tdcm(spectrum,duration_s=prefix_s,center_hint_hz=h['carrier'],symbol_hz=symbol_hz,
                order=order,tdcm_options=trial_tdcm,pwm_preamble_options=pwm_args,preamble_options=preamble_options,cancel_event=cancel_event)
            if getattr(trial,'payload_debug',{}).get('status')=='frame_length_mismatch':raise ValueError('物理净荷/额外尾窗口数与配置不符')
            metrics,template=_metrics(trial,order,h['calibration'],search['minimum_geometry'])
            coverage=sum(bool(v['valid']) for v in trial.frames)/max(1,len(trial.frames))
            row.update(metrics,coverage=coverage,period_ppm=trial.diagnostics['period_ppm'],status='通过选择验证')
            if coverage<search['minimum_coverage']:raise ValueError('有效帧覆盖率不足，不能只用少数好帧选方案')
            if preamble_options is None:
                if metrics['geometry']<search['minimum_geometry'] or metrics['occupied_labels']<min(order//2,h['frame']//2):raise ValueError('前段未形成可信QAM格点结构')
                if metrics['training_grid_evm_percent']>comp['quality_evm_percent']:raise ValueError('前段格点拟合误差超过阈值')
            else:
                pilots=[v['pilot_evm_percent'] for v in trial.frames if v['valid'] and v['pilot_evm_percent'] is not None]
                if not pilots:raise ValueError('没有通过的真实QAM训练')
                metrics['known_training_evm_percent']=float(np.median(pilots));row.update(metrics)
            if metrics['repeat_evm_percent']>comp['quality_evm_percent']:raise ValueError('独立重复验证误差超过阈值')
            complexity=(h['clock']!='off')+(h['phase']!='off')+(h['normal']=='blind')
            score=metrics['repeat_evm_percent']+5*(1-coverage)
            accepted.append(dict(row=row,result=trial,tdcm=trial_tdcm,template=template,score=score,complexity=complexity,carrier=h['carrier']))
        except ValueError as exc:row.update(status='拒绝',reason=str(exc))
    if not accepted:
        reasons='；'.join(v['reason'] for v in rows if v.get('reason'))
        raise AutoSearchError('自动补偿未找到通过独立选择验证的方案。请核对Fs、前导/帧模型及搜索范围。'+reasons[:800],
            dict(status='failed',ranges=search,candidates=rows,carrier_bounds=bounds))
    minimum=min(v['score'] for v in accepted)
    near=[v for v in accepted if v['score']<=minimum+.15]
    winner=min(near,key=lambda v:(v['complexity'],v['score']))
    trial=winner['result'];d=trial.diagnostics;effective=dict(winner['tdcm'])
    final_comp=dict(effective['compensation'],clock_mode='manual',period_ppm=d['period_ppm'],
                    fit_timing=False,timing_offset_symbols=d['timing_offset_symbols'])
    effective['compensation']=final_comp
    if preamble_options is None:
        gain=d['gain_iq']
        effective['frozen_calibration']=dict(gain_real=float(gain.real),gain_imag=float(gain.imag),
            template=_json_template(winner['template']),note='只由前段标定帧估计，最终验证段未用于拟合或选择')
    recipe=dict(version=1,method='bounded_repeated_tdcm',duration_s=duration_s,center_hint_hz=winner['carrier'],
        symbol_hz=symbol_hz,order=order,tdcm_options=effective,pwm_preamble_options=pwm_args,
        preamble_options=preamble_options,input_fingerprint=_fingerprint(spectrum,total))
    update('自动补偿：方案已冻结，验证未参与选择的后段')
    final=replay_recipe(spectrum,recipe,cancel_event=cancel_event)
    test_start=spectrum.first_tuple/fs+selection_count/fs+float(final.diagnostics['frame_period_samples'])/fs
    try:
        test_metrics,_=_metrics(final,order,winner['row']['calibration_frames'],search['minimum_geometry'],
                                template=winner['template'],after_s=test_start)
    except ValueError as exc:
        raise AutoSearchError('最终独立后段无法验证，未采用自动方案：'+str(exc),
            dict(status='test_failed',ranges=search,candidates=rows,carrier_bounds=bounds,
                 note='最终后段不足；未使用它反复选择其他组合')) from None
    tested=[v for v in final.frames if v['data_time_s']>=test_start]
    test_coverage=sum(bool(v['valid']) for v in tested)/max(1,len(tested))
    passed=(test_metrics['repeat_evm_percent']<=comp['quality_evm_percent'] and test_coverage>=search['minimum_coverage'])
    incompatible=[v for v in accepted if abs(v['score']-winner['score'])<.5 and
                  (v['row']['frame_symbols']!=winner['row']['frame_symbols'] or abs(v['carrier']-winner['carrier'])>cfg['base_hz']/4)]
    ambiguous=bool(incompatible)
    status='selected' if passed and not ambiguous else 'ambiguous' if ambiguous else 'test_failed'
    adopted=dict(qam_order=order,bits_per_symbol=int(math.log2(order)),nominal_symbol_hz=symbol_hz,measured_symbol_hz=final.symbol_hz,
        harmonic=cfg['harmonic'],measured_base_hz=final.tdcm['base_hz'],
        sample_rate_hz=fs,sample_rate_source=spectrum.rate_source,carrier_seed_hz=winner['carrier'],fixed_cfo_hz=final.pwm_preamble['frequency_hz'],
        corrected_carrier_reference_hz=final.tdcm['corrected_carrier_hz'],phase_mode=final_comp['phase_mode'],
        phase_update_hz=final.diagnostics['phase_update_hz'],phase_bandwidth_hz=final_comp['phase_bandwidth_hz'],
        clock_switch=winner['row']['clock_mode'],period_ppm=final_comp['period_ppm'],period_samples=final.diagnostics['period_samples'],
        timing_offset_symbols=final_comp['timing_offset_symbols'],gain_mode=winner['row']['gain_mode'],
        gain_real=float(final.diagnostics['gain_iq'].real),gain_imag=float(final.diagnostics['gain_iq'].imag),
        frame_symbols=winner['row']['frame_symbols'])
    recipe['adopted']=adopted
    recipe['search_ranges']=search
    recipe['search_outcome']=dict(status=status,selection_evm_percent=winner['row']['repeat_evm_percent'],final_test_evm_percent=test_metrics['repeat_evm_percent'],note='重复结构验证；未测净荷BER')
    train_ids=np.unique(trial.frame_ids)[:winner['row']['calibration_frames']].tolist()
    validation_ids=np.unique(trial.frame_ids)[winner['row']['calibration_frames']:].tolist()
    test_ids=[int(v['index']) for v in tested if v['valid']]
    final.diagnostics['auto_search']=dict(status=status,adopted=adopted,
        partitions=dict(calibration_frame_ids=train_ids,selection_frame_ids=validation_ids,final_test_frame_ids=test_ids),recipe=recipe,ranges=search,
        carrier_bounds=bounds,carrier_seeds=seeds,candidates=rows,winner_id=winner['row']['id'],
        evaluated_trials=sum(v['stage']=='精评' for v in rows),proposed_combinations=len(ordered),
        budget_stopped=budget_stopped or len(ordered)>search['max_trials'],selection_score=winner['score'],
        selection_metrics={k:v for k,v in winner['row'].items() if k not in ('reason',)},
        final_test_metrics=test_metrics,final_test_coverage=test_coverage,test_start_s=test_start,
        selection_end_s=spectrum.first_tuple/fs+selection_count/fs,elapsed_s=time.monotonic()-started,
        note='仅限指定范围、帧模型及已评估组合；最终数据窗口不用于拟合/重选，独立RF参考和真实导频仍可用于接收跟踪；不是BER或全局最优')
    current_stat=spectrum.path.stat()
    if (current_stat.st_size,current_stat.st_mtime_ns)!=(input_stat.st_size,input_stat.st_mtime_ns):
        raise AutoSearchError('输入文件在搜索期间变化，结果未采用；请使用下载完成的文件。')
    final.diagnostics['structural_validation']=passed and not ambiguous
    final.diagnostics['quality']='自动组合通过独立重复验证；净荷仍为候选' if status=='selected' else '自动搜索存在模型歧义；方案仅供检查' if ambiguous else '自动方案未通过最终后段验证；尚未确认'
    if not passed:final.diagnostics['warnings'].append('最终后段未通过；没有使用该后段反复挑选其他方案。')
    if ambiguous:final.diagnostics['warnings'].append('不同载波或帧模型表现接近，需发端信息消除歧义。')
    return final


def replay_recipe(spectrum,recipe,*,cancel_event=None):
    if not isinstance(recipe,dict) or recipe.get('version')!=1 or recipe.get('method')!='bounded_repeated_tdcm':raise ValueError('不支持的自动补偿方案格式')
    fp=recipe.get('input_fingerprint',{})
    if not isinstance(fp,dict):raise ValueError('复现方案输入指纹须为对象')
    input_stat=spectrum.path.stat()
    if any(fp.get(k)!=v for k,v in [('sample_rate_hz',spectrum.sample_rate_hz),('channel',spectrum.channel),('first_tuple',spectrum.first_tuple),('file_bytes',spectrum.total_tuples*8)]):raise ValueError('复现方案与当前采样率、通道、起点或文件长度不匹配')
    if not isinstance(fp.get('count'),int) or not 1<=fp['count']<=33554432:raise ValueError('复现方案输入范围无效')
    if _fingerprint(spectrum,fp['count'])['sha256']!=fp.get('sha256'):raise ValueError('输入指纹不同，不能把旧文件的冻结标定套到新文件')
    allowed=('duration_s','center_hint_hz','symbol_hz','order','tdcm_options','pwm_preamble_options','preamble_options')
    kwargs={k:recipe[k] for k in allowed if k in recipe}
    try:
        seconds=float(kwargs.get('duration_s',1))
        limit=validate_compensation(kwargs.get('tdcm_options',{}).get('compensation',{}))['max_input_samples']
        actual=min(round(seconds*spectrum.sample_rate_hz),spectrum.total_tuples-spectrum.first_tuple,limit,math.ceil(spectrum.sample_rate_hz))
    except (ValueError,TypeError,AttributeError):raise ValueError('复现方案处理范围无效') from None
    if actual!=fp['count']:raise ValueError('复现范围与输入指纹范围不一致')
    result=recover_compensated_tdcm(spectrum,cancel_event=cancel_event,**kwargs)
    current_stat=spectrum.path.stat()
    if (current_stat.st_size,current_stat.st_mtime_ns)!=(input_stat.st_size,input_stat.st_mtime_ns):raise ValueError('输入文件在复算期间变化')
    result.diagnostics['replay_recipe']=recipe
    if recipe.get('search_outcome',{}).get('status') in ('test_failed','ambiguous'):
        result.diagnostics['structural_validation']=False
        result.diagnostics['quality']='原自动方案未通过独立验证；本次仅复算候选'
    return result

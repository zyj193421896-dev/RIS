"""Fingerprint-locked replay of a known-training central-window capture.

This optional method preserves the legacy receivers. It recomputes from raw
IQ, fits only prefix pilots, and separates single windows from frame means.
"""
import math
from types import SimpleNamespace
import numpy as np

METHOD = 'frozen_central_tdcm_capture'


def _evm(z, reference):
    return float(100*np.sqrt(np.mean(abs(z-reference)**2)/max(float(np.mean(abs(reference)**2)), 1e-30)))


def replay_frozen_capture(spectrum, recipe, *, cancel_event=None):
    from rx706_recovery import _read_span
    from rx706_spectrum import _cancel
    from rx706_tdcm import harmonic_coefficients
    from rx706_tdcm_auto import _fingerprint
    from rx706_tdcm_debug import _rotate_inplace
    from rx706_qam import decide
    from rx706_qam_preamble import parse_preamble
    from rx706_qam_training import fit_training, normalize_training
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise ValueError('中央窗冻结回放需要SciPy；请使用已安装SciPy的Python环境启动接收GUI') from exc
    if not isinstance(recipe, dict) or recipe.get('version') != 1 or recipe.get('method') != METHOD:
        raise ValueError('不支持的中央窗冻结方案格式')
    fp=recipe.get('input_fingerprint',{});cfg=recipe.get('central_window',{})
    if not isinstance(fp,dict) or not isinstance(cfg,dict):raise ValueError('中央窗方案参数须为对象')
    fs=spectrum.sample_rate_hz;before=spectrum.path.stat()
    for key,value in [('sample_rate_hz',fs),('channel',spectrum.channel),('first_tuple',spectrum.first_tuple),('file_bytes',spectrum.total_tuples*8)]:
        if fp.get(key)!=value:raise ValueError('中央窗方案与当前采样率、通道、起点或文件长度不匹配')
    count=fp.get('count')
    if type(count) is not int or not 4096<=count<=min(33554432,math.ceil(fs),spectrum.total_tuples-spectrum.first_tuple):
        raise ValueError('中央窗方案输入范围无效')
    def number(key,low,high,integer=False):
        v=cfg.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not low<=v<=high or (integer and int(v)!=v):
            raise ValueError('中央窗参数无效：'+key)
        return int(v) if integer else float(v)
    sps=number('period_samples',2,1000000);rate=fs/sps
    frame=number('frame_symbols',2,65536,True);start=number('training_start_sample',0,count-1)
    ncal=number('calibration_frames',1,128,True);holdout=number('holdout_start_frame',ncal,2048,True)
    block=number('phase_block_samples',16,count//8,True)
    window=number('phase_savgol_window',3,min(4095,count//block),True)
    degree=number('phase_savgol_order',0,min(8,window-1),True)
    if window%2!=1:raise ValueError('相位平滑窗须为奇数')
    seed=number('carrier_seed_hz',-fs/2+1,fs/2-1);residual=number('reflection_residual_hz',-fs/4,fs/4)
    harmonic=number('harmonic',-32,32,True)
    if harmonic==0 or abs(harmonic*rate)>=fs/2:raise ValueError('目标谐波超出采样范围')
    order=recipe.get('order');view=recipe.get('view','coherent_mean')
    if order != 64 or view not in ('coherent_mean','individual'):raise ValueError('中央窗冻结回放仅支持64QAM及已知显示方式')
    nominal=recipe.get('symbol_hz')
    if isinstance(nominal,bool) or not isinstance(nominal,(int,float)) or not math.isfinite(nominal) or not 0<nominal<fs/2:raise ValueError('中央窗标称符号率无效')
    def indices(key):
        value=cfg.get(key)
        if not isinstance(value,list) or not value or any(type(v) is not int or not 0<=v<frame for v in value) or len(set(value))!=len(value):
            raise ValueError('中央窗符号索引无效：'+key)
        return np.asarray(value,dtype=int)
    ti=indices('training_window_indices');pi=indices('payload_window_indices')
    if np.intersect1d(ti,pi).size:raise ValueError('训练与净荷窗口不可重叠')
    expected=parse_preamble(recipe.get('preamble_options',{}).get('symbols',''),'hex',order)
    if len(expected)<=int(ti.max()):raise ValueError('已知训练长度与中央窗索引不匹配')
    points=expected[ti]
    if np.linalg.matrix_rank(np.column_stack([points.real,points.imag]))<2:raise ValueError('中央窗训练不能确定幅相映射')
    _cancel(cancel_event)
    if _fingerprint(spectrum,count)['sha256']!=fp.get('sha256'):raise ValueError('输入指纹不同，不能将旧中央窗标定用于其他数据')
    x,actual,clipped=_read_span(spectrum,count/fs,seed,cancel_event,max_samples=count)
    if actual!=count:raise ValueError('中央窗回放范围不一致')
    _rotate_inplace(x,fs,seed,cancel_event=cancel_event)
    blocks=count//block;phase_t=(np.arange(blocks)+.5)*block/fs
    means=x[:blocks*block].reshape(blocks,block).mean(axis=1)
    phase=savgol_filter(np.unwrap(np.angle(means)),window,degree)
    _rotate_inplace(x,fs,phase_t=phase_t,phase=phase,cancel_event=cancel_event)
    _rotate_inplace(x,fs,residual,cancel_event=cancel_event)
    coefficients,centres=harmonic_coefficients(x,fs,start,rate,harmonic*rate,cancel_event=cancel_event)
    nf=len(coefficients)//frame
    if nf<=holdout+3 or nf>2048:raise ValueError('中央窗完整帧数无效，独立后段至少4帧且总帧数不超过2048')
    matrix=coefficients[:nf*frame].reshape(nf,frame);ct=centres[:nf*frame].reshape(nf,frame)
    fit=fit_training(matrix[:ncal,ti].mean(axis=0),points,rate,1000,50,orientation='normal',fit_dc_offset=True)
    frozen=cfg.get('frozen_calibration')
    if frozen:
        try:g=complex(*frozen['gain']);d=complex(*frozen['offset'])
        except (KeyError,TypeError,ValueError):raise ValueError('冻结标定格式无效') from None
        if not np.allclose([fit['gain'],fit['offset']],[g,d],rtol=1e-9,atol=1e-9):raise ValueError('前段已知训练复算与冻结标定不一致')
    normalized=normalize_training(matrix,fit);pilots=normalized[:,ti];payload=normalized[:,pi]
    final=payload[holdout:];mean=final.mean(axis=0);mean_ideal,mean_labels,mean_bits=decide(mean,order)
    mid=(nf+holdout)//2;left=payload[holdout:mid].mean(axis=0);right=payload[mid:].mean(axis=0)
    single_ideal=decide(final.ravel(),order)[0]
    metrics=dict(training_mean_evm_percent=_evm(pilots[holdout:].mean(axis=0),points),
        training_single_evm_percent=_evm(pilots[holdout:],np.broadcast_to(points,pilots[holdout:].shape)),
        single_window_nn_evm_percent=_evm(final.ravel(),single_ideal),coherent_mean_nn_evm_percent=_evm(mean,mean_ideal),
        halves_repeat_evm_percent=_evm(left,right),
        halves_label_agreement_fraction=float(np.mean(decide(left,order)[1]==decide(right,order)[1])),
        calibration_frames=ncal,holdout_start_frame=holdout,holdout_frames=nf-holdout,complete_frames=nf,
        observed_windows_per_frame=len(pi),ber=None)
    if view=='coherent_mean':
        scatter=mean;times=ct[holdout:,pi].mean(axis=0);ids=np.full(len(pi),-1,int);c=matrix[holdout:,pi].mean(axis=0)
    else:
        scatter=final.ravel();times=ct[holdout:,pi].ravel();ids=np.repeat(np.arange(holdout,nf),len(pi));c=matrix[holdout:,pi].ravel()
    ideal,labels,bits=decide(scatter,order)
    frames=[];training=[]
    for k in range(nf):
        _cancel(cancel_event)
        pe=_evm(pilots[k],points)
        frames.append(dict(index=k,data_time_s=float((ct[k,pi[0]]+spectrum.first_tuple)/fs),header_error_percent=None,
            pilot_evm_percent=pe,valid=pe<=50,calibration=k<ncal,symbols=len(pi),
            nn_evm_percent=_evm(payload[k],decide(payload[k],order)[0]),template_evm_percent=_evm(payload[k],mean),
            note='中央窗候选；此标志不用于剔除帧'))
        for j,v in enumerate(pilots[k]):
            training.append(dict(frame=k,training_index=int(ti[j]),time_s=float((ct[k,ti[j]]+spectrum.first_tuple)/fs),
                expected_hex=f'{int(decide(points,order)[1][j]):02X}',observed_hex=f'{int(decide(np.array([v]),order)[1][0]):02X}',
                normalized_i=float(v.real),normalized_q=float(v.imag),training_passed=pe<=50,conjugated=False,evm_percent=pe))
    after=spectrum.path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('输入文件在中央窗复算期间发生变化')
    phase_frequency=seed+residual+np.gradient(phase,phase_t)/(2*np.pi)
    diag=dict(frozen_capture=dict(metrics=metrics,view=view,recipe=recipe),replay_recipe=recipe,
        quality='冻结中央窗回放；重复图样候选，逐帧解码未确认；未测BER',
        period_samples=sps,period_ppm=(sps/(fs/recipe['symbol_hz'])-1)*1e6,measured_symbol_hz=rate,
        gain_iq=fit['gain'],gain_source=f'前{ncal}帧已知训练公共复增益和零点；未用净荷拟合',frames=frames,
        phase_time_s=phase_t+spectrum.first_tuple/fs,phase_rad=phase,frequency_hz=phase_frequency,
        warnings=['采样率来自项目协议，本数据没有采集元数据。','相干均值不是单次符号性能；判决EVM不是准确率或BER。',
                  '20% PWM模板未通过；本方案回放已冻结的重复帧/训练边界，不声称PWM已锁定。',
                  '每个重复对仅取一个中央窗，不合并或证明两次物理判决一致。'],
        structural_validation=False,ber=None)
    return SimpleNamespace(kind='qam',order=order,carrier_hz=seed,symbol_hz=rate,scatter=scatter,ideal=ideal,labels=labels,bits=bits,
        evm_percent=_evm(scatter,ideal),time_s=(times+spectrum.first_tuple)/fs,first_tuple=spectrum.first_tuple,
        input_samples=count,clipped=clipped,cutoff_hz=rate/2,decimation=1,output_rate_hz=rate,
        preamble=None,pwm_preamble=None,coefficients=c,raw_coefficients=matrix,diagnostics=diag,frames=frames,
        frame_ids=ids,training_records=training,payload_pairs=[],tail_records=[],
        tdcm=dict(base_hz=rate,harmonic=harmonic,corrected_carrier_hz=float(np.median(phase_frequency)),target_hz=float(np.median(phase_frequency))+harmonic*rate),
        frozen_payload=payload,frozen_pilots=pilots,frozen_mean=mean,frozen_points=points,frozen_halves=(left,right))


def summary_frozen_capture(r,spectrum):
    f=r.diagnostics['frozen_capture'];m=f['metrics'];recipe=f['recipe'];cfg=recipe['central_window']
    view=f'{m["holdout_frames"]}帧相干均值（每个逻辑位置一个候选）' if f['view']=='coherent_mean' else '独立后段单次中央窗（未作跨帧平均）'
    return (f'输入：{spectrum.path} · RX{spectrum.channel} · 起点{spectrum.first_tuple}\n'
        f'64QAM 冻结中央窗回放 · 显示：{view}\n'
        f'当前显示最近邻EVM {r.evm_percent:.6f}%；未测BER\n'
        f'独立训练均值参考EVM {m["training_mean_evm_percent"]:.6f}%\n'
        f'净荷单次中央窗最近邻EVM {m["single_window_nn_evm_percent"]:.6f}%\n'
        f'净荷{m["holdout_frames"]}帧相干均值最近邻EVM {m["coherent_mean_nn_evm_percent"]:.6f}%\n'
        f'独立验证两半均值标签一致率 {100*m["halves_label_agreement_fraction"]:.6f}%（不是准确率）\n'
        f'采样率 {spectrum.sample_rate_hz:g} Hz；输入{spectrum.total_tuples}点；来源：{spectrum.rate_source}\n'
        f'周期 {r.diagnostics["period_samples"]:.9f}点；周期偏差 {r.diagnostics["period_ppm"]:+.6f} ppm\n'
        f'3PWM + 00 00 04 04 24 24 20 20 + 128物理净荷；139窗/帧先验。\n'
        f'每重复对取一个中央谐波窗，4个训练观测、64个净荷观测/帧；不是128个物理窗全量解码。\n'
        f'前{m["calibration_frames"]}帧拟合公共复增益和零点；时钟/时序获取输入至4,194,304点。\n'
        f'最终验证从第{m["holdout_start_frame"]}帧开始，共{m["holdout_frames"]}帧；不按训练EVM筛除任何帧。\n'
        f'载波：4096点块均值 + 21块二阶平滑；固定周期与中央窗锚点由JSON冻结。\n'
        f'中央窗训练起点 {cfg["training_start_sample"]:.9f}点；显示均值的时间是各帧采样中心的均值。\n'
        '通过“载入方案JSON并复算”重新载入此文件可再次复算；普通手动参数不包含全部冻结操作。\n'
        +'\n'.join(r.diagnostics['warnings'])+'\n候选HEX：'+' '.join(f'{int(v):02X}' for v in r.labels[:64]))


def figures_frozen_capture(r,spectrum):
    from matplotlib.figure import Figure
    from rx706_qam_debug_gui import scatter_indices
    from rx706_qam import decide
    f=r.diagnostics['frozen_capture'];m=f['metrics'];hold=m['holdout_start_frame'];payload=r.frozen_payload[hold:]
    levels=np.arange(-7,8,2)/np.sqrt(42);ideal=(levels[:,None]+1j*levels[None,:]).ravel()
    fig=Figure(figsize=(12,8),layout='constrained');axes=fig.subplots(2,2)
    single=payload.ravel();idx=scatter_indices(len(single));z=single[idx]
    axes[0,0].scatter(z.real,z.imag,s=2,alpha=.2);axes[0,0].set(xlim=(-2.2,2.2),ylim=(-2.2,2.2),title=f'Single windows: NN EVM {m["single_window_nn_evm_percent"]:.3f}% (display clipped)')
    z=r.frozen_mean;axes[0,1].scatter(z.real,z.imag,s=18)
    for j,v in enumerate(z):axes[0,1].annotate(f'{int(decide(z,r.order)[1][j]):02X}',(v.real,v.imag),fontsize=6,xytext=(3,3),textcoords='offset points')
    axes[0,1].set_title(f'{m["holdout_frames"]}-frame mean: NN EVM {m["coherent_mean_nn_evm_percent"]:.3f}%')
    p=r.frozen_pilots[hold:].mean(axis=0);axes[1,0].scatter(p.real,p.imag,s=60);axes[1,0].scatter(r.frozen_points.real,r.frozen_points.imag,marker='x',s=80)
    axes[1,0].set_title(f'Known pilot means: reference EVM {m["training_mean_evm_percent"]:.3f}%')
    left,right=r.frozen_halves;axes[1,1].scatter(left.real,left.imag,label='Earlier half',s=20);axes[1,1].scatter(right.real,right.imag,label='Later half',marker='x',s=20)
    axes[1,1].set_title(f'Label agreement {100*m["halves_label_agreement_fraction"]:.3f}% (not accuracy)');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_aspect('equal');ax.set_xlabel('I');ax.set_ylabel('Q')
    for ax in axes.flat[:2]:ax.scatter(ideal.real,ideal.imag,c='#c22',marker='+',s=20)
    diag=Figure(figsize=(12,8),layout='constrained');a=diag.subplots(2,2);d=r.diagnostics
    a[0,0].plot(d['phase_time_s'],d['frequency_hz']);a[0,0].set(title='Common carrier from independent block means',xlabel='Time (s)',ylabel='Baseband Hz')
    a[0,1].plot(d['phase_time_s'],d['phase_rad']);a[0,1].set(title='Smooth common phase correction',xlabel='Time (s)',ylabel='Radians')
    rows=r.frames;a[1,0].plot([v['index'] for v in rows],[v['pilot_evm_percent'] for v in rows],lw=.6);a[1,0].axvline(hold,c='red',ls='--');a[1,0].set(title='Per-frame pilot EVM; no frame rejection',xlabel='Frame',ylabel='EVM %')
    a[1,1].plot([v['index'] for v in rows],[v['nn_evm_percent'] for v in rows],lw=.6);a[1,1].axvline(hold,c='red',ls='--');a[1,1].set(title='Per-frame payload NN EVM; not BER',xlabel='Frame',ylabel='EVM %')
    for ax in a.flat:ax.grid(alpha=.2)
    return fig,diag

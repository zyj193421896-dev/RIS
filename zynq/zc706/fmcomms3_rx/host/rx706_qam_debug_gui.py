"""Isolated QAM compensation controls, report views and candidate exports."""
import base64
import csv
import html
import io
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk,scrolledtext,filedialog,messagebox
import numpy as np

CLOCK_MODES={'关闭（配置周期）':'off','手动周期ppm':'manual','自动：整帧重复图样':'repeat'}
PHASE_MODES={'关闭':'off','两态平方载波':'square','载波均值':'mean'}
FRAME_MODES={'单帧候选':'single','按配置分割全部帧':'all'}
ACQUISITION_MODES={'有PWM前导（原流程）':'pwm','无前导：重复图样':'no_header_repeat','无前导：已知固定时钟/随机载荷':'no_header_fixed'}


class CompensationControls:
    def __init__(self,parent,profile_callback=None,calibration_variables=None,recipe_callback=None,search_callback=None,training_profile_callback=None,paired_profile_callback=None,save_bound_callback=None):
        self.calibration_variables=calibration_variables or {}
        self.search_callback=search_callback
        self.acquisition_mode=tk.StringVar(value=next(iter(ACQUISITION_MODES)))
        self.acquisition_mode.trace_add('write',self._acquisition_changed)
        self.widgets=[];self.enabled=tk.BooleanVar(value=False);self.auto_open=tk.BooleanVar(value=True)
        self.clock_mode=tk.StringVar(value=next(iter(CLOCK_MODES)))
        self.phase_mode=tk.StringVar(value=next(iter(PHASE_MODES)))
        self.frame_mode=tk.StringVar(value=next(iter(FRAME_MODES)))
        self.fit_timing=tk.BooleanVar(value=False)
        self.search_enabled=tk.BooleanVar(value=False)
        self.search_enabled.trace_add('write',self._search_changed)
        self.search_values={k:tk.StringVar(value=v) for k,v in dict(carrier_span_khz='500',clock_ppm='1000',frame_candidates='64 128 256 512',max_carriers='3',max_trials='12',max_seconds='60',minimum_coverage='0.8',minimum_geometry='0.55').items()}
        values=dict(period_ppm='0',clock_search_ppm='200',phase_update_hz='2000',phase_bandwidth_hz='300',
                    frame_symbols='256',max_frames='128',max_input_samples='16777216',calibration_frames='20',timing_offset_symbols='0',
                    timing_search_symbols='0.05',quality_evm_percent='8')
        self.values={k:tk.StringVar(value=v) for k,v in values.items()}
        canvas=tk.Canvas(parent,highlightthickness=0,width=340)
        scrollbar=ttk.Scrollbar(parent,orient='vertical',command=canvas.yview)
        scrollbar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        body=ttk.Frame(canvas,padding=8);wid=canvas.create_window((0,0),window=body,anchor='nw')
        body.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(wid,width=e.width))
        canvas.bind('<MouseWheel>',lambda e:canvas.yview_scroll(-int(e.delta/120),'units'))
        def check(text,var):
            w=ttk.Checkbutton(body,text=text,variable=var);w.pack(anchor='w',pady=5);self.widgets.append(w)
        check('启用 TDCM / QAM 独立补偿',self.enabled)
        ttk.Label(body,text='默认关闭，保持旧流程。增强仅用于TDCM谐波路径；普通QAM保留匹配滤波及已知训练流程。',wraplength=290).pack(fill='x')
        if profile_callback:
            w=ttk.Button(body,text='有前导重复图样配置（按当前阶数）',command=lambda:profile_callback(self))
            w.pack(fill='x',pady=8);self.widgets.append(w)
        if training_profile_callback:
            w=ttk.Button(body,text='3PWM＋四角训练（按当前阶数）',command=training_profile_callback)
            w.pack(fill='x',pady=5);self.widgets.append(w)
        if paired_profile_callback:
            w=ttk.Button(body,text='64QAM当前帧：3+4+4+128',command=paired_profile_callback)
            w.pack(fill='x',pady=5);self.widgets.append(w)
        def group(title):
            box=ttk.LabelFrame(body,text=title,padding=7);box.pack(fill='x',pady=6);box.columnconfigure(1,weight=1);return box
        def field(box,label,key,choices=None):
            row=box.grid_size()[1];var=key if isinstance(key,tk.Variable) else self.values[key]
            ttk.Label(box,text=label,wraplength=150).grid(row=row,column=0,sticky='w',pady=4)
            w=ttk.Combobox(box,textvariable=var,values=choices,state='readonly',width=15) if choices else ttk.Entry(box,textvariable=var,width=15)
            w.grid(row=row,column=1,sticky='ew',pady=4);self.widgets.append(w)
        acquisition_box=group('0 · 前导与数据获取')
        field(acquisition_box,'获取方式',self.acquisition_mode,tuple(ACQUISITION_MODES))
        ttk.Label(acquisition_box,text='无前导仅支持两态TDCM：重复图样可估周期；随机载荷须输入真实时钟/ppm。固定拟合完整符号起点并分块验证；下方单帧/窄窗仅用于有前导。块边界不是协议帧，标签/比特始终是候选。',wraplength=285).grid(row=1,column=0,columnspan=2,sticky='w')
        auto_box=group('自动补偿：有限范围组合搜索')
        check('自动搜索并验证补偿组合',self.search_enabled)
        field(auto_box,'载波中心 ±kHz',self.search_values['carrier_span_khz'])
        field(auto_box,'自动周期搜索 ±ppm',self.search_values['clock_ppm'])
        field(auto_box,'每帧QAM窗口候选',self.search_values['frame_candidates'])
        field(auto_box,'最多载波候选',self.search_values['max_carriers'])
        field(auto_box,'最多精评组合',self.search_values['max_trials'])
        field(auto_box,'搜索预算 (s)',self.search_values['max_seconds'])
        field(auto_box,'最低有效帧比例',self.search_values['minimum_coverage'])
        field(auto_box,'最低格点结构分数',self.search_values['minimum_geometry'])
        button=ttk.Button(body,text='启用自动搜索（使用当前先验）',command=self.enable_search)
        button.pack(fill='x',pady=5);self.widgets.append(button)
        if recipe_callback:
            button=ttk.Button(body,text='载入参数 / 冻结方案 JSON…',command=recipe_callback)
            button.pack(fill='x',pady=5);self.widgets.append(button)
        if save_bound_callback:
            button=ttk.Button(body,text='保存本次IQ绑定复算JSON…',command=save_bound_callback)
            button.pack(fill='x',pady=5);self.widgets.append(button)
        ttk.Label(body,text='须提供Fs、标称速率及谐波。有前导模式另需PWM；无前导模式搜索两态载波、相位及重复窗口。随机载荷保持配置时钟。保留最终后段验证；采用值以结果为准。',wraplength=290).pack(fill='x')
        box=group('1 · 符号时钟与窗口')
        field(box,'周期校正方式',self.clock_mode,tuple(CLOCK_MODES))
        field(box,'周期增大ppm（正值降速）','period_ppm')
        field(box,'重复周期搜索 ±ppm','clock_search_ppm')
        field(box,'手动窗口偏移（符号）','timing_offset_symbols')
        field(box,'盲定时搜索 ±符号','timing_search_symbols')
        check('在前段重复图样拟合窗口起点',self.fit_timing)
        ttk.Label(body,text='自动模式要求整帧波形重复。盲定时还需选择“盲归一化”，不得用未知序列冒充QAM训练。',wraplength=290).pack(fill='x')
        box=group('2 · 公共载波相位')
        field(box,'独立载波参考',self.phase_mode,tuple(PHASE_MODES))
        field(box,'最低相位更新率 (Hz)','phase_update_hz')
        field(box,'相位参考低通带宽 (Hz)','phase_bandwidth_hz')
        ttk.Label(body,text='从原载波参考估计公共相位，保留每符号的数据相位。更新率和带宽不同；弱参考或混叠会拒绝处理。',wraplength=290).pack(fill='x')
        box=group('3 · 帧、标定与判据')
        field(box,'帧处理',self.frame_mode,tuple(FRAME_MODES))
        field(box,'每帧QAM窗口（含训练，允许短载荷）','frame_symbols')
        field(box,'最多处理帧数','max_frames')
        field(box,'输入采样点上限','max_input_samples')
        field(box,'前段标定帧数','calibration_frames')
        field(box,'结构验证EVM上限 (%)','quality_evm_percent')
        shared=self.calibration_variables
        if shared:
            field(box,'目标谐波校准',shared['normalization'],('PWM两态参考','盲归一化'))
            field(box,'目标谐波幅度系数',shared['amplitude'])
            field(box,'目标谐波相位修正 (°)',shared['phase'])
            check('启用真实QAM训练（须实际发送）',shared['training_enabled'])
            training_box=group('4 · 真实QAM训练（与原设置同步）')
            field(training_box,'训练表示',shared['training_format'],('符号值（Gray）','HEX字节（00–FF）','I/Q电平'))
            field(training_box,'实际训练序列',shared['training_symbols'])
            field(training_box,'训练EVM上限 (%)',shared['training_evm'])
            field(training_box,'训练残余频偏 ±kHz',shared['training_frequency'])
            if 'training_orientation' in shared:
                field(training_box,'训练I/Q镜像',shared['training_orientation'],('不共轭（原流程）','自动辨识I/Q镜像','固定共轭'))
                check('用已知训练校正复零点',shared['training_offset'])
            ttk.Label(training_box,text='四角顺序左下、左上、右上、右下：256QAM=00 08 88 80；64QAM=00 04 24 20。须实际发送该序列。每帧窗口数=真实净荷数+训练长度，不由阶数决定。64QAM每符号6bit，HEX仅为00～3F存储值。',wraplength=280).grid(row=training_box.grid_size()[1],column=0,columnspan=2,sticky='w')
        ttk.Label(body,text='目标谐波幅度/相位沿用TDCM设置；真实QAM训练优先。后段模板仅检验重复结构，最近邻EVM/候选标签均不是BER。',wraplength=290).pack(fill='x')
        check('分析完成后打开QAM诊断报告',self.auto_open)

    def _acquisition_changed(self,*args):
        mode=ACQUISITION_MODES[self.acquisition_mode.get()]
        if mode!='pwm':
            self.enabled.set(True);self.phase_mode.set('两态平方载波')
            self.clock_mode.set('自动：整帧重复图样' if mode=='no_header_repeat' else '手动周期ppm')
            if self.calibration_variables:self.calibration_variables['normalization'].set('盲归一化')
            if self.search_callback:self.search_callback()

    def acquisition_options(self):
        return ACQUISITION_MODES[self.acquisition_mode.get()]

    def options(self):
        from rx706_tdcm_debug import validate_compensation
        options={k:v.get() for k,v in self.values.items()}
        options.update(enabled=self.enabled.get(),clock_mode=CLOCK_MODES[self.clock_mode.get()],
                       phase_mode=PHASE_MODES[self.phase_mode.get()],frame_mode=FRAME_MODES[self.frame_mode.get()],
                       fit_timing=self.fit_timing.get())
        return validate_compensation(options)

    def _search_changed(self,*args):
        if self.search_enabled.get():
            self.enabled.set(True)
            if self.search_callback:self.search_callback()

    def enable_search(self):
        self.enabled.set(True);self.search_enabled.set(True)

    def search_options(self):
        from rx706_tdcm_auto import validate_search
        data={k:v.get() for k,v in self.search_values.items() if k!='carrier_span_khz'}
        try:data['carrier_span_hz']=float(self.search_values['carrier_span_khz'].get())*1000
        except ValueError:raise ValueError('载波搜索半宽须为kHz数值') from None
        data['enabled']=self.search_enabled.get()
        return validate_search(data)

    def repeated_profile(self,frame_symbols=None):
        self.acquisition_mode.set('有PWM前导（原流程）')
        self.search_enabled.set(False)
        self.enabled.set(True);self.clock_mode.set('自动：整帧重复图样');self.phase_mode.set('两态平方载波')
        self.frame_mode.set('按配置分割全部帧');self.fit_timing.set(True)
        for k,v in dict(period_ppm='0',clock_search_ppm='200',phase_update_hz='2000',phase_bandwidth_hz='300',
                        frame_symbols='256',max_frames='128',max_input_samples='16777216',calibration_frames='20',timing_offset_symbols='0',
                        timing_search_symbols='0.05',quality_evm_percent='8').items():self.values[k].set(v)
        if frame_symbols is not None:self.values['frame_symbols'].set(str(frame_symbols))


def summary_text(r,spectrum):
    d=getattr(r,'diagnostics',None)
    if d and d.get('frozen_capture'):
        from rx706_tdcm_frozen import summary_frozen_capture
        return summary_frozen_capture(r,spectrum)
    text=f'输入：{spectrum.path} · RX{spectrum.channel} · 起点{spectrum.first_tuple}\n'
    from rx706_qam_profiles import mapping_note
    text+=mapping_note(r.order)+'\n'
    text+=f'{r.order}QAM · {len(r.scatter):,} 个候选数据符号\n最近邻判决EVM {r.evm_percent:.3f}% · 未测BER\n'
    layout=getattr(r,'payload_debug',None)
    if layout:
        text+=f"物理后训练符号 {layout['physical_symbols']}；重复对/逻辑位置 {layout['pair_positions']}；一致 {layout['agreeing_pairs']}，不一致 {layout['disagreeing_pairs']}；额外尾符号 {layout['tail_symbols']}\n"
        text+=layout['policy']+'\nCRC调试：'+layout['crc']['status']+'；'+layout['crc']['note']+'\n'
    text+='净荷候选HEX：'+' '.join(f'{int(v):02X}' for v in r.labels[:48])+'\n'
    from rx706_qam import decision_margins
    spacing=2/np.sqrt(2*(r.order-1)/3);near=int(np.count_nonzero(decision_margins(r.scatter,r.order)<.1*spacing))
    text+=f'靠近判决边界（距边界不足格距10%）：{near}/{len(r.scatter)}；这是几何提示，不是误码率。\n'
    text+=f'采样率 {spectrum.sample_rate_hz:g} Hz；来源：{spectrum.rate_source}\n'
    text+=f'本次恢复输入 {r.input_samples:,} 点 / {r.input_samples/spectrum.sample_rate_hz:.6f} s（时间依赖Fs）\n'
    if not d:return text+'原有静态接收流程；无增强补偿轨迹。请核对发端映射和真实训练。'
    text+=f"\n状态：{d['quality']}\n实测/采用周期 {d['period_samples']:.9f} 点；周期偏差 {d['period_ppm']:+.5f} ppm\n"
    stages=d.get('processing_stages')
    if stages:
        text+='PWM前导：已匹配候选；符号窗口：已提取候选复系数\n'
        verdict={'training_matched':'实际QAM训练匹配',
                 'repeat_structure_passed':'重复结构通过，映射仍需核对',
                 'sparse_repeat_verified_mapping_unverified':'重复波形通过；稀疏载荷的QAM映射未验证',
                 'unvalidated':'尚无充分独立验证'}
        text+='质量验证：'+verdict[stages['quality_validation']]+'；比特验证：未测\n'
    training=d.get('training_summary')
    if training:text+='已知训练 / 映射：'+json.dumps(training,ensure_ascii=False,indent=2)+'\n'
    text+=f"符号率 {d['measured_symbol_hz']:.9f} Sym/s；窗口偏移 {d['timing_offset_symbols']:+.6f} 符号\n"
    text+=f"载波参考 {d['options']['phase_mode']}；实际更新率 {d['phase_update_hz']:.3f} Hz；参考强度 {d['phase_coherence']:.4f}\n"
    text+=f"完整窗口块 {len(r.frames)}；结构判据通过 {sum(x['valid'] for x in r.frames)}；覆盖网格标签 {d['occupied_labels']}/{r.order}\n" if d.get('acquisition') else f"完整帧 {len(r.frames)}；前导/训练通过 {sum(x['valid'] for x in r.frames)}；覆盖网格标签 {d['occupied_labels']}/{r.order}\n"
    held=d['heldout_template_evm_percent'];text+=f"后段重复模板EVM：{held:.3f}%（非发端参考）\n" if held is not None else '后段重复模板EVM：无独立可用验证\n'
    text+=f"标定：{d['gain_source']}\n复增益 {d['gain_iq'].real:+.6g}{d['gain_iq'].imag:+.6g}j ADC counts\n"
    acquisition=d.get('acquisition')
    if acquisition:
        text+='\n无PWM获取：'+acquisition['status']+'\n采用方案：'+json.dumps(acquisition['adopted'],ensure_ascii=False,indent=2)+'\n'
        text+='选择段指标：'+json.dumps(acquisition['selection_metrics'],ensure_ascii=False)+'\n最终独立后段：'+json.dumps(acquisition['final_test_metrics'],ensure_ascii=False)+'\n'
        text+=acquisition['note']+'\n'
    search=d.get('auto_search')
    if search:
        text+='\n自动组合搜索：'+search['status']+'（仅限已评估范围）\n'
        text+='采用组合：'+json.dumps(search['adopted'],ensure_ascii=False,indent=2)+'\n'
        text+=f"选择段重复EVM {search['selection_metrics']['repeat_evm_percent']:.3f}%（评分 {search['selection_score']:.3f}）；最终后段 {search['final_test_metrics']['repeat_evm_percent']:.3f}%；最终帧覆盖率 {100*search['final_test_coverage']:.1f}%\n"
        text+=f"精评 {search['evaluated_trials']}/{search['proposed_combinations']} 组合；预算截断 {search['budget_stopped']}\n"
        text+=search['note']+'\n'
    elif d.get('replay_recipe'):
        text+='\n已按输入指纹核验并复现冻结方案；未重新搜索其他组合。\n'
        text+='复现采用数值：'+json.dumps(d['replay_recipe'].get('adopted',{}),ensure_ascii=False,indent=2)+'\n'
    text+='\n参数来源：'+d['parameter_source']+'\n'+'\n'.join(d['warnings'])
    return text


def scatter_indices(count,budget=12000,labels=None):
    """Avoid periodic-stride aliasing of repeated constellation sweeps."""
    if count<=budget:return np.arange(count)
    essential=np.unique(labels,return_index=True)[1] if labels is not None else np.empty(0,int)
    essential=essential[:budget]
    available=np.setdiff1d(np.arange(count),essential,assume_unique=True)
    chosen=np.random.default_rng(0).choice(available,budget-len(essential),replace=False)
    return np.sort(np.r_[essential,chosen])


def report_figures(r,spectrum):
    if getattr(r,'diagnostics',{}).get('frozen_capture'):
        from rx706_tdcm_frozen import figures_frozen_capture
        return figures_frozen_capture(r,spectrum)
    from matplotlib.figure import Figure
    d=getattr(r,'diagnostics',{})
    fig=Figure(figsize=(12,8),layout='constrained');axes=fig.subplots(2,2)
    from rx706_spectrum import display_trace
    frequency,power=display_trace(spectrum,max_points=12000)
    axes[0,0].plot(frequency*1000,power,lw=.6)
    axes[0,0].set(title='Input spectrum (coherent bin levels)',xlabel='Baseband kHz',ylabel='dBFS / bin')
    for aa,z,title in [(axes[0,1],d.get('before_scatter',np.empty(0)),'Before: fixed CFO / nominal clock'),
                       (axes[1,0],r.scatter,'After: actual coefficients (no snapping)')]:
        idx=scatter_indices(len(z),labels=r.labels if z is r.scatter else None);v=z[idx]
        aa.scatter(v.real,v.imag,s=2,alpha=.3);aa.set(title=title,xlabel='I',ylabel='Q');aa.set_aspect('equal')
    n=min(1000,len(r.scatter));axes[1,1].plot(r.time_s[:n],r.scatter.real[:n],'.-',ms=2,lw=.5,label='I')
    axes[1,1].plot(r.time_s[:n],r.scatter.imag[:n],'.-',ms=2,lw=.5,label='Q');axes[1,1].legend()
    axes[1,1].set(title='Recovered candidate symbols (first 1000)',xlabel='Original time (s)')
    for aa in axes.ravel():aa.grid(alpha=.2)
    diag=Figure(figsize=(12,8),layout='constrained');a=diag.subplots(2,2)
    t=d.get('phase_time_s',[]);a[0,0].plot(t,d.get('phase_rad',[]));a[0,0].set(title='Continuous common phase correction',xlabel='Original time (s)',ylabel='radians')
    a[0,1].plot(t,d.get('frequency_hz',[]));a[0,1].set(title='Carrier frequency relative to nominal',xlabel='Original time (s)',ylabel='Hz')
    clock=d.get('clock',{});a[1,0].plot(clock.get('lags',[]),clock.get('lag_residual_samples',[]),'o-')
    a[1,0].set(title='Repeat clock fit residual (coarse sample cells)',xlabel='Frame lag',ylabel='samples')
    rows=d.get('frames',[])
    if rows:
        valid=[v for v in rows if v['nn_evm_percent'] is not None]
        a[1,1].plot([v['index'] for v in valid],[v['nn_evm_percent'] for v in valid],'o-',label='Nearest grid')
        test=[v for v in valid if v['template_evm_percent'] is not None and not v['calibration']]
        a[1,1].plot([v['index'] for v in test],[v['template_evm_percent'] for v in test],'o-',label='Heldout repeat template')
        a[1,1].legend(fontsize=8)
    a[1,1].set(title='Per-frame errors; no payload BER measured',xlabel='Frame',ylabel='EVM (%)')
    for aa in a.ravel():aa.grid(alpha=.2)
    return fig,diag


def math_ceil(x):
    return int(np.ceil(x))


def export_symbols_csv(r,path,cancel_event=None):
    from rx706_qam import decision_margins
    ids=getattr(r,'frame_ids',np.full(len(r.scatter),-1,int))
    margins=decision_margins(r.scatter,r.order)
    coeff=getattr(r,'coefficients',np.full(len(r.scatter),complex(float('nan'),float('nan'))))
    with open(path,'w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        from rx706_qam_profiles import mapping_note
        writer.writerow(['# 候选结果；未验证发端映射；未测BER；'+mapping_note(r.order)])
        writer.writerow(['index','frame','time_s','coefficient_i_counts','coefficient_q_counts','normalized_i','normalized_q','candidate_gray_label','candidate_bits','nearest_error','decision_boundary_margin','candidate_hex'])
        for j,z in enumerate(r.scatter):
            if j%1024==0 and cancel_event is not None and cancel_event.is_set():raise RuntimeError('导出已取消；可能保留部分输出文件')
            writer.writerow([j,int(ids[j]),f'{r.time_s[j]:.12g}',coeff[j].real,coeff[j].imag,z.real,z.imag,int(r.labels[j]),''.join(map(str,r.bits[j])),abs(z-r.ideal[j]),margins[j],f'{int(r.labels[j]):02X}'])


def export_training_csv(r,path,cancel_event=None):
    rows=getattr(r,'training_records',[])
    if not rows:raise ValueError('当前结果没有单独的已知训练记录')
    with open(path,'w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['# 已知训练；与净荷分离；训练匹配不代表净荷BER'])
        fields=list(rows[0]);w.writerow(fields)
        for k,row in enumerate(rows):
            if k%1024==0 and cancel_event is not None and cancel_event.is_set():raise RuntimeError('训练导出已取消')
            w.writerow([row.get(name) for name in fields])


def export_payload_pairs_csv(r,path,cancel_event=None):
    rows=getattr(r,'payload_pairs',[])
    if not rows:raise ValueError('没有已对齐的逐帧重复对，不能导出逻辑候选')
    keys=('frame','logical_index','left_physical_index','right_physical_index','left_time_s','right_time_s','left_hex','right_hex','agree','alignment_valid','logical_hex','normalized_distance')
    with open(path,'w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['# 独立重复对；物理符号不合并；不一致逻辑候选为空；未测BER'])
        w.writerow(keys)
        for k,row in enumerate(rows):
            if k%1024==0 and cancel_event is not None and cancel_event.is_set():raise RuntimeError('导出已取消')
            w.writerow([row.get(key) for key in keys])


def export_tail_csv(r,path,cancel_event=None):
    rows=getattr(r,'tail_records',[])
    if not rows:raise ValueError('没有配置并提取到额外尾符号；CRC仍未验证')
    keys=('frame','tail_index','physical_index','time_s','candidate_hex','candidate_bits','normalized_i','normalized_q','classification')
    with open(path,'w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['# 原始额外尾符号；不认定为CRC；无算法参数，不作CRC通过/失败判定'])
        w.writerow(keys)
        for k,row in enumerate(rows):
            if k%1024==0 and cancel_event is not None and cancel_event.is_set():raise RuntimeError('导出已取消')
            w.writerow([row.get(key) for key in keys])


def export_recipe_json(r,path):
    d=getattr(r,'diagnostics',{})
    acquisition=d.get('acquisition')
    recipe=(dict(version=1,method='headerless_tdcm_config',**acquisition) if acquisition else None) or d.get('auto_search',{}).get('recipe') or d.get('replay_recipe')
    if recipe is None:raise ValueError('当前结果没有自动搜索冻结方案')
    with open(path,'w',encoding='utf-8') as stream:json.dump(recipe,stream,ensure_ascii=False,indent=2)


def export_report_html(r,spectrum,path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    parts=[]
    for fig in report_figures(r,spectrum):
        FigureCanvasAgg(fig);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=125)
        parts.append('<img alt="QAM diagnostics" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">')
    text=html.escape(summary_text(r,spectrum));rows=getattr(r,'frames',[])
    frame_table='<table><tr><th>窗口块/帧</th><th>数据起点(s)</th><th>结果判据通过</th><th>最近邻EVM%</th><th>重复模板EVM%</th><th>用于标定</th></tr>'
    for v in rows:
        frame_table+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [v['index'],v['data_time_s'],v['valid'],v['nn_evm_percent'],v['template_evm_percent'],v['calibration']])+'</tr>'
    frame_table+='</table>'
    diagnostics=getattr(r,'diagnostics',{})
    settings=html.escape(json.dumps(diagnostics.get('options',{}),ensure_ascii=False,indent=2))
    if diagnostics.get('acquisition'):
        settings+='\n\n无PWM获取方案/范围/候选：\n'+html.escape(json.dumps(diagnostics['acquisition'],ensure_ascii=False,indent=2))
    search=diagnostics.get('auto_search')
    if search:
        settings+='\n\n自动采用方案/范围/候选：\n'+html.escape(json.dumps(search,ensure_ascii=False,indent=2))
    training_rows=getattr(r,'training_records',[])
    training_table=''
    if training_rows:
        training_table='<h2>已知训练（与净荷分离）</h2><p>训练匹配不代表净荷BER。下表最多128行；完整记录可另存训练CSV。</p><table><tr><th>帧</th><th>训练序号</th><th>期望HEX</th><th>观察HEX</th><th>归一化I</th><th>归一化Q</th><th>训练通过</th><th>镜像校正</th></tr>'
        for row in training_rows[:128]:
            training_table+='<tr>'+''.join('<td>'+html.escape(str(row.get(k)))+'</td>' for k in ('frame','training_index','expected_hex','observed_hex','normalized_i','normalized_q','training_passed','conjugated'))+'</tr>'
        training_table+='</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>QAM解调诊断</title><style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;color:#21384a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6fa;padding:16px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd8e3;padding:6px}h2{margin-top:32px}</style><h1>QAM解调与补偿诊断</h1><pre>'+text+'</pre>'+''.join(parts)+'<h2>逐帧信息</h2>'+frame_table+'<h2>补偿配置</h2><pre>'+settings+'</pre><p>图中为真实窗口复系数；未将坐标吸附至网格。此报告不包含原始IQ文件。</p></html>'
    layout=getattr(r,'payload_debug',None)
    if layout:
        detail='<h2>重复净荷 / 额外帧尾 / CRC</h2><pre>'+html.escape(json.dumps(layout,ensure_ascii=False,indent=2))+'</pre>'
        for title,rows,keys in [('重复对（前128行；物理符号不合并）',getattr(r,'payload_pairs',[]),('frame','logical_index','left_hex','right_hex','logical_hex','agree')),('未分类额外尾符号（前128行）',getattr(r,'tail_records',[]),('frame','tail_index','candidate_hex','candidate_bits','classification'))]:
            if rows:
                detail+='<h3>'+title+'</h3><table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
                for row in rows[:128]:detail+='<tr>'+''.join('<td>'+html.escape(str(row.get(k)))+'</td>' for k in keys)+'</tr>'
                detail+='</table>'
        page=page.replace('<h2>逐帧信息</h2>',detail+'<h2>逐帧信息</h2>')
    if training_table:page=page.replace('<h2>逐帧信息</h2>',training_table+'<h2>逐帧信息</h2>')
    with open(path,'w',encoding='utf-8') as stream:stream.write(page)


class QAMReportWindow:
    def __init__(self,parent,r,spectrum):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        from rx706_plot_navigation import PlotNavigation
        self.window=tk.Toplevel(parent);self.window.title('QAM 解调与补偿诊断 · 候选结果')
        self.window.geometry('1300x900');self.window.minsize(900,650)
        self.closed=False;self.poll_id=None;self.events=queue.Queue();self.cancel=threading.Event();self.export_thread=None
        self.r=r;self.spectrum=spectrum;self.page=0;self.page_size=500;self.frame_filter=tk.StringVar(value='全部')
        self.status=tk.StringVar(value='真实复系数 · 候选Gray标签/比特 · 未测BER')
        self.navigation=[];self.canvases=[]
        bar=ttk.Frame(self.window,padding=8);bar.pack(fill='x')
        ttk.Button(bar,text='导出完整HTML报告…',command=lambda:self.export('html')).pack(side='left',padx=5)
        ttk.Button(bar,text='导出全部候选符号CSV…',command=lambda:self.export('csv')).pack(side='left',padx=5)
        if getattr(r,'diagnostics',{}).get('acquisition'):
            ttk.Button(bar,text='导出无前导配置JSON…',command=lambda:self.export('json')).pack(side='left',padx=5)
        elif getattr(r,'diagnostics',{}).get('auto_search') or getattr(r,'diagnostics',{}).get('replay_recipe'):
            ttk.Button(bar,text='导出冻结方案JSON…',command=lambda:self.export('json')).pack(side='left',padx=5)
        ttk.Label(bar,textvariable=self.status).pack(side='left',padx=10)
        self.notebook=ttk.Notebook(self.window);self.notebook.pack(fill='both',expand=True,padx=8,pady=8)
        info=ttk.Frame(self.notebook);self.notebook.add(info,text='结果 / 可信度')
        text=scrolledtext.ScrolledText(info,wrap='word',font='TkDefaultFont');text.pack(fill='both',expand=True)
        text.insert('1.0',summary_text(r,spectrum));text.configure(state='disabled')
        figs=report_figures(r,spectrum)
        self.figures=figs
        for title,fig in zip(['频谱 / 补偿前后 / 符号','公共相位 / 时钟 / 逐帧误差'],figs):
            tab=ttk.Frame(self.notebook);self.notebook.add(tab,text=title)
            canvas=FigureCanvasTkAgg(fig,master=tab);toolbar=NavigationToolbar2Tk(canvas,tab,pack_toolbar=False)
            toolbar.pack(side='bottom',fill='x');canvas.get_tk_widget().pack(fill='both',expand=True)
            self.navigation.append(PlotNavigation(canvas,toolbar));self.canvases.append(canvas);canvas.draw_idle()
        frames=ttk.Frame(self.notebook);self.notebook.add(frames,text='逐帧信息')
        cols=('frame','data_s','header_error','pilot_error','check','symbols','nn','template','calibration')
        headerless=bool(getattr(r,'diagnostics',{}).get('acquisition'))
        self.frame_tree=self._table(frames,cols,('窗口块' if headerless else '帧','数据窗口起点(s)','前导误差%','QAM训练EVM%','结构判据' if headerless else '前导/训练','符号数','最近邻EVM%','模板EVM%','标定块' if headerless else '标定帧'))
        for row in getattr(r,'frames',[]):
            def fmt(v):return '—' if v is None else f'{v:.4f}'
            self.frame_tree.insert('','end',values=(row['index'],f"{row['data_time_s']:.9f}",fmt(row['header_error_percent']),fmt(row['pilot_evm_percent']),
                '通过' if row['valid'] else '未通过',row['symbols'],fmt(row['nn_evm_percent']),fmt(row['template_evm_percent']),'是' if row['calibration'] else '否'))
        symbols=ttk.Frame(self.notebook);self.notebook.add(symbols,text='候选符号 / 比特')
        controls=ttk.Frame(symbols);controls.pack(fill='x',pady=6)
        ttk.Label(controls,text='帧筛选').pack(side='left')
        ids=sorted(set(getattr(r,'frame_ids',[])))
        cb=ttk.Combobox(controls,textvariable=self.frame_filter,values=['全部']+[str(int(k)) for k in ids],state='readonly',width=10)
        cb.pack(side='left',padx=5);cb.bind('<<ComboboxSelected>>',lambda e:self.set_page(0))
        ttk.Button(controls,text='上一页',command=lambda:self.set_page(self.page-1)).pack(side='left',padx=4)
        ttk.Button(controls,text='下一页',command=lambda:self.set_page(self.page+1)).pack(side='left',padx=4)
        self.page_text=tk.StringVar();ttk.Label(controls,textvariable=self.page_text).pack(side='left',padx=8)
        self.symbol_tree=self._table(symbols,('i','frame','time','I','Q','label','bits','error','margin'),
            ('净荷符号序号','帧','时间(s)','归一化I','归一化Q','候选HEX','候选比特','最近邻误差','距判决边界'))
        training_rows=getattr(r,'training_records',[])
        if training_rows:
            training_tab=ttk.Frame(self.notebook);self.notebook.add(training_tab,text='已知训练 / 映射')
            ttk.Label(training_tab,text='训练与净荷分开。显示前500条训练记录；完整记录可导出。训练匹配不代表净荷BER。').pack(fill='x',pady=5)
            ttk.Button(training_tab,text='导出完整训练CSV…',command=lambda:self.export('training')).pack(anchor='w',padx=5,pady=5)
            self.training_tree=self._table(training_tab,('frame','index','expected','observed','I','Q','pass','mirror'),
                ('帧','训练序号','期望HEX','观察HEX','归一化I','归一化Q','训练通过','共轭校正'))
            for row in training_rows[:500]:
                self.training_tree.insert('','end',values=(row['frame'],row['training_index'],row['expected_hex'],row['observed_hex'] or '—',
                    '—' if row['normalized_i'] is None else f"{row['normalized_i']:.6f}",
                    '—' if row['normalized_q'] is None else f"{row['normalized_q']:.6f}",row['training_passed'],row['conjugated']))
        layout=getattr(r,'payload_debug',None)
        if layout:
            payload_tab=ttk.Frame(self.notebook);self.notebook.add(payload_tab,text='重复对 / 帧尾 / CRC')
            info=scrolledtext.ScrolledText(payload_tab,height=7,wrap='word');info.pack(fill='x')
            info.insert('1.0',json.dumps(layout,ensure_ascii=False,indent=2));info.configure(state='disabled')
            if getattr(r,'payload_pairs',[]):
                ttk.Button(payload_tab,text='导出全部重复对CSV…',command=lambda:self.export('pairs')).pack(anchor='w')
                self.pair_tree=self._table(payload_tab,('frame','logical','left','right','candidate','agree'),('帧','逻辑位置','左物理HEX','右物理HEX','一致候选HEX','是否一致'))
                for row in r.payload_pairs[:500]:self.pair_tree.insert('','end',values=(row['frame'],row['logical_index'],row['left_hex'],row['right_hex'],row['logical_hex'] or '?',row['agree']))
            if getattr(r,'tail_records',[]):
                ttk.Button(payload_tab,text='导出额外尾符号CSV…',command=lambda:self.export('tail')).pack(anchor='w')
                self.tail_tree=self._table(payload_tab,('frame','index','hex','bits'),('帧','尾部物理序号','候选HEX（未分类）','候选比特'))
                for row in r.tail_records[:500]:self.tail_tree.insert('','end',values=(row['frame'],row['tail_index'],row['candidate_hex'],row['candidate_bits']))
        search=getattr(r,'diagnostics',{}).get('auto_search')
        acquisition=getattr(r,'diagnostics',{}).get('acquisition')
        if acquisition:
            acquisition_tab=ttk.Frame(self.notebook);self.notebook.add(acquisition_tab,text='无前导方案 / 验证')
            build_acquisition_table(acquisition_tab,acquisition)
        if search:
            auto_tab=ttk.Frame(self.notebook);self.notebook.add(auto_tab,text='自动方案 / 候选组合')
            self.search_tree=build_search_table(auto_tab,search)
        self.set_page(0);self.notebook.select(auto_tab if search else acquisition_tab if acquisition else 1);self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.poll_id=self.window.after(100,self.poll)

    def _table(self,parent,cols,headers):
        box=ttk.Frame(parent);box.pack(fill='both',expand=True)
        tree=ttk.Treeview(box,columns=cols,show='headings')
        for c,h in zip(cols,headers):tree.heading(c,text=h);tree.column(c,width=135,minwidth=95,stretch=False)
        vertical=ttk.Scrollbar(box,orient='vertical',command=tree.yview);horizontal=ttk.Scrollbar(box,orient='horizontal',command=tree.xview)
        tree.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        vertical.pack(side='right',fill='y');horizontal.pack(side='bottom',fill='x');tree.pack(fill='both',expand=True)
        return tree

    def set_page(self,page):
        from rx706_qam import decision_margins
        r=self.r;ids=np.arange(len(r.scatter));frames=getattr(r,'frame_ids',np.full(len(ids),-1))
        if self.frame_filter.get()!='全部':ids=ids[frames==int(self.frame_filter.get())]
        pages=max(1,math_ceil(len(ids)/self.page_size));self.page=max(0,min(page,pages-1))
        self.symbol_tree.delete(*self.symbol_tree.get_children())
        for j in ids[self.page*self.page_size:(self.page+1)*self.page_size]:
            z=r.scatter[j];margin=decision_margins(np.array([z]),r.order)[0]
            self.symbol_tree.insert('','end',values=(j,frames[j],f'{r.time_s[j]:.9f}',f'{z.real:.6f}',f'{z.imag:.6f}',f'{int(r.labels[j]):02X}',''.join(map(str,r.bits[j])),f'{abs(z-r.ideal[j]):.5f}',f'{margin:.6f}'))
        self.page_text.set(f'{self.page+1}/{pages}页 · {len(ids):,}符号 · 每页最多{self.page_size}')

    def export(self,kind):
        if self.export_thread is not None and self.export_thread.is_alive():return
        extension='csv' if kind in ('training','pairs','tail') else kind
        path=filedialog.asksaveasfilename(parent=self.window,title='导出QAM候选诊断',defaultextension='.'+extension,
            filetypes=[(extension.upper(),'*.'+extension)])
        if not path:return
        self.status.set('正在后台导出…')
        def work():
            try:
                if kind=='html':export_report_html(self.r,self.spectrum,path)
                elif kind=='json':export_recipe_json(self.r,path)
                elif kind=='training':export_training_csv(self.r,path,self.cancel)
                elif kind=='pairs':export_payload_pairs_csv(self.r,path,self.cancel)
                elif kind=='tail':export_tail_csv(self.r,path,self.cancel)
                else:export_symbols_csv(self.r,path,self.cancel)
                self.events.put((True,path))
            except Exception as exc:self.events.put((False,str(exc)))
        self.export_thread=threading.Thread(target=work,daemon=True,name='qam-export');self.export_thread.start()

    def poll(self):
        if self.closed:return
        try:
            ok,value=self.events.get_nowait();self.status.set('已导出：'+value if ok else '导出失败：'+value)
        except queue.Empty:pass
        self.poll_id=self.window.after(100,self.poll)

    def close(self):
        if self.closed:return
        self.closed=True;self.cancel.set()
        if self.poll_id:self.window.after_cancel(self.poll_id)
        for nav in self.navigation:nav.close()
        for canvas in self.canvases:
            idle=getattr(canvas,'_idle_draw_id',None)
            if idle:
                try:canvas.get_tk_widget().after_cancel(idle)
                except tk.TclError:pass
                canvas._idle_draw_id=None
        self.window.destroy()


def build_search_table(parent,search):
    info=scrolledtext.ScrolledText(parent,wrap='word',height=12,font='TkDefaultFont')
    info.pack(fill='x',padx=6,pady=6)
    detail=dict(status=search.get('status'),adopted=search.get('adopted'),ranges=search.get('ranges'),
                carrier_bounds=search.get('carrier_bounds'),note=search.get('note'))
    verdict={'selected':'通过独立后段验证','ambiguous':'不同模型无法唯一选择','test_failed':'最终后段未通过，方案仅供检查','failed':'没有组合通过验证'}.get(search.get('status'),'复算/候选结果')
    info.insert('1.0','当前结果：'+verdict+'\n'+json.dumps(detail,ensure_ascii=False,indent=2));info.configure(state='disabled')
    box=ttk.Frame(parent);box.pack(fill='both',expand=True)
    cols=('id','stage','carrier','frame','phase','clock','gain','error','coverage','geometry','status','reason')
    tree=ttk.Treeview(box,columns=cols,show='headings')
    for key,title in zip(cols,('候选','阶段','载波(kHz)','窗口数','相位','时钟','幅相','选择EVM%','覆盖率','格点相干度','状态','原因')):
        tree.heading(key,text=title);tree.column(key,width=130,minwidth=80,stretch=False)
    vertical=ttk.Scrollbar(box,orient='vertical',command=tree.yview);vertical.pack(side='right',fill='y')
    horizontal=ttk.Scrollbar(box,orient='horizontal',command=tree.xview);horizontal.pack(side='bottom',fill='x')
    tree.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set);tree.pack(fill='both',expand=True)
    tree.tag_configure('winner',background='#dff2e6' if search.get('status')=='selected' else '#fff1cc')
    winner=None
    for row in search.get('candidates',[]):
        def fmt(key):
            v=row.get(key);return '—' if v is None else f'{v:.4f}'
        item=tree.insert('','end',tags=('winner',) if row['id']==search.get('winner_id') else (),values=(row['id'],row['stage'],f"{row['carrier_hz']/1000:+.5f}",row['frame_symbols'],
            row['phase_mode'],row['clock_mode'],row['gain_mode'],fmt('repeat_evm_percent'),fmt('coverage'),fmt('geometry'),row['status'],row.get('reason','')))
        if row['id']==search.get('winner_id'):winner=item
    if winner:tree.selection_set(winner);tree.see(winner)
    return tree


class AutoSearchFailureWindow:
    def __init__(self,parent,diagnostics):
        self.window=tk.Toplevel(parent);self.window.title('补偿获取未通过 · 搜索诊断')
        self.window.geometry('1250x700');self.closed=False;self.r=None;self.failure=diagnostics
        (build_acquisition_table if diagnostics.get('mode','').startswith('no_header') else build_search_table)(self.window,diagnostics);self.window.protocol('WM_DELETE_WINDOW',self.close)
    def close(self):
        if not self.closed:self.closed=True;self.window.destroy()


def build_acquisition_table(parent,acquisition):
    verdict={'candidate_validated':'独立结构验证通过；比特仍为候选',
             'test_failed':'独立验证未通过；不得当作锁定',
             'failed':'没有方案通过获取/验证'}.get(acquisition.get('status'),'候选结果')
    info=scrolledtext.ScrolledText(parent,wrap='word',height=16,font='TkDefaultFont')
    info.pack(fill='x',padx=6,pady=6)
    details={k:acquisition[k] for k in ('adopted','selection_metrics','final_test_metrics','ranges','two_state_selection','two_state_test') if k in acquisition}
    info.insert('1.0','无PWM获取：'+verdict+'\n真实BER：未测；90°旋转及协议帧边界未验证。\n'+json.dumps(details,ensure_ascii=False,indent=2)+'\n'+acquisition.get('note',''))
    info.configure(state='disabled')
    box=ttk.Frame(parent);box.pack(fill='both',expand=True)
    cols=('id','carrier','phase','block','period','error','geometry','status','reason')
    tree=ttk.Treeview(box,columns=cols,show='headings')
    for c,h in zip(cols,('候选','载波(kHz)','公共相位','窗口符号数','周期(原始点)','选择误差%','网格结构','状态','原因')):
        tree.heading(c,text=h);tree.column(c,width=145,minwidth=85,stretch=False)
    v=ttk.Scrollbar(box,orient='vertical',command=tree.yview);v.pack(side='right',fill='y')
    h=ttk.Scrollbar(box,orient='horizontal',command=tree.xview);h.pack(side='bottom',fill='x')
    tree.configure(yscrollcommand=v.set,xscrollcommand=h.set);tree.pack(fill='both',expand=True)
    tree.tag_configure('adopted',background='#dff2e6' if acquisition.get('status')=='candidate_validated' else '#fff1cc')
    for row in acquisition.get('candidates',[]):
        def fmt(v):return '—' if v is None else f'{v:.5f}'
        metrics=row.get('metrics',{});selected=row.get('id')==acquisition.get('winner_id') and acquisition.get('winner_id') is not None
        item=tree.insert('','end',tags=('adopted',) if selected else (),values=(row.get('id','—'),fmt(row.get('carrier_hz',0)/1000),row.get('phase_mode','—'),
            row.get('block_symbols','—'),fmt(row.get('period_samples')),fmt(row.get('score')),fmt(metrics.get('geometry')),row.get('status','—'),row.get('reason','')))
        if selected:tree.selection_set(item);tree.see(item)
    return tree

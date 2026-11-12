"""Three task-oriented front doors; decoder and expert controls stay unchanged."""
import copy
import json
from pathlib import Path
from rx706_ui_contrast import bright_hint
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

AREAS = ('PWM测速', '256QAM验证', '三用户验证')
FIXED_PREAMBLE10_PROFILE = Path(__file__).resolve().parent/'frame_profiles/fixed_preamble10/frame.json'
FIXED_PREAMBLE10_SHA256 = 'afa8d1b0f87b964c63daee8cc410f6447f794c293cba1cefae811fe1ce4ca2a0'
TDCM_PROFILE_SCHEMA = 'rx706_tdcm_settings_v1'
QAM_FIELDS = ('channel', 'rate', 'recovery_ms', 'first_tuple', 'qam_order',
    'tdcm_symbol_khz', 'tdcm_base_khz', 'tdcm_pwm_khz', 'tdcm_carrier_mhz',
    'tdcm_harmonic', 'tdcm_normalization', 'tdcm_amplitude', 'tdcm_phase', 'tdcm_count',
    'qam_preamble_enabled', 'qam_preamble_format', 'qam_preamble_symbols',
    'qam_preamble_frequency_khz', 'qam_preamble_evm', 'qam_preamble_search_ms',
    'qam_pwm_symbols', 'qam_pwm_gap_us', 'qam_pwm_search_ms', 'qam_pwm_error',
    'qam_pwm_width_offset', 'qam_pwm_frequency_khz', 'qam_pwm_qam_window',
    'pwm2_total_chips', 'pwm2_prefix_chips', 'pwm2_guard_chips')


def validate_qam_profile(config):
    """Do not silently downgrade extended, IQ-bound historical replay recipes."""
    if not isinstance(config, dict) or config.get('schema') != TDCM_PROFILE_SCHEMA:
        raise ValueError('不是当前TDCM设置文件；历史IQ绑定回放需专用回放入口')
    if set(config) != {'schema', 'settings'} or not isinstance(config['settings'], dict):
        raise ValueError('TDCM设置包含不支持的字段')
    settings = config['settings']
    unknown, missing = set(settings)-set(QAM_FIELDS), set(QAM_FIELDS)-set(settings)
    if unknown or missing:
        raise ValueError('TDCM设置字段不匹配：'+', '.join(sorted(unknown | missing)))
    if str(settings['qam_order']) != '256':
        raise ValueError('本功能区要求256QAM')
    # Validate every value before changing the live UI.
    import math
    text_fields = {'channel', 'tdcm_normalization', 'qam_preamble_format',
                   'qam_preamble_symbols', 'qam_pwm_symbols'}
    for key, value in settings.items():
        if key == 'qam_preamble_enabled':
            if not isinstance(value, bool):
                raise ValueError('qam_preamble_enabled须为布尔值')
        elif key not in text_fields:
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(key+'须为有限数值')
    if settings['channel'] not in ('RX1', 'RX2'):
        raise ValueError('通道须为RX1或RX2')
    if settings['tdcm_normalization'] not in ('PWM两态参考', '盲归一化'):
        raise ValueError('未知TDCM归一化方式')
    if settings['qam_preamble_format'] not in ('符号值（Gray）', 'I/Q电平'):
        raise ValueError('未知QAM训练表示方式')
    fs, rate = float(settings['rate'])*1e6, float(settings['tdcm_symbol_khz'])*1000
    if fs <= 0 or float(settings['recovery_ms']) <= 0 or float(settings['first_tuple']) < 0 or not float(settings['first_tuple']).is_integer():
        raise ValueError('采样率/时长须为正数，起点须为非负整数')
    from rx706_tdcm import validate_tdcm
    validate_tdcm(fs, rate, dict(base_hz=float(settings['tdcm_base_khz'])*1000,
        harmonic=float(settings['tdcm_harmonic']), amplitude_scale=float(settings['tdcm_amplitude']),
        phase_deg=float(settings['tdcm_phase']), payload_symbols=float(settings['tdcm_count']),
        normalization='pwm' if settings['tdcm_normalization'] == 'PWM两态参考' else 'blind'))
    from rx706_qam_pwm import validate_pwm_header
    for key in ('pwm2_total_chips', 'pwm2_prefix_chips', 'pwm2_guard_chips'):
        if not float(settings[key]).is_integer():
            raise ValueError(key+'须为整数')
    validate_pwm_header(dict(symbols=settings['qam_pwm_symbols'],
        total_chips=int(settings['pwm2_total_chips']), prefix_chips=int(settings['pwm2_prefix_chips']),
        guard_chips=int(settings['pwm2_guard_chips']), symbol_hz=float(settings['tdcm_pwm_khz'])*1000,
        gap_s=float(settings['qam_pwm_gap_us'])*1e-6, search_s=float(settings['qam_pwm_search_ms'])*1e-3,
        max_error_percent=float(settings['qam_pwm_error']), width_offset_chips=float(settings['qam_pwm_width_offset']),
        max_frequency_hz=float(settings['qam_pwm_frequency_khz'])*1000,
        qam_search_symbols=float(settings['qam_pwm_qam_window'])), fs)
    if settings['qam_preamble_enabled']:
        from rx706_qam_preamble import validate_preamble
        validate_preamble(dict(symbols=settings['qam_preamble_symbols'],
            representation='iq' if settings['qam_preamble_format'] == 'I/Q电平' else 'labels',
            max_frequency_hz=float(settings['qam_preamble_frequency_khz'])*1000,
            max_evm_percent=float(settings['qam_preamble_evm']),
            search_s=float(settings['qam_preamble_search_ms'])*1e-3), 256, rate)
    return copy.deepcopy(settings)


def auto_acquisition_note(recovery):
    """One consistent measured-acquisition label for the visible result panels."""
    auto = (getattr(recovery, 'diagnostics', None) or {}).get('auto_acquisition')
    if not auto:
        return ''
    return (f"自动搜索：原载波 {auto['carrier_hz']/1000:+.6f} kHz；"
            f"前导实测周期偏差 {auto['period_ppm']:+.3f} ppm")


def qam_training_note(recovery):
    """Training-only fit error must never be presented as payload quality."""
    import math
    from statistics import median
    rows = getattr(recovery, 'frames', None) or []
    values = [float(row['pilot_evm_percent']) for row in rows
              if row.get('valid') and row.get('pilot_evm_percent') is not None
              and math.isfinite(float(row['pilot_evm_percent']))]
    if not values:
        if rows:
            return '已知训练EVM 无有效帧'
        sync = getattr(recovery, 'preamble', None) or {}
        evm = sync.get('evm_percent')
        if evm is not None and math.isfinite(float(evm)):
            return f'已知训练EVM {float(evm):.2f}%（当前匹配）'
        return '已知训练EVM 未测'
    records = getattr(recovery, 'training_records', None) or []
    expected = {row.get('expected_hex') for row in records}
    corners = expected == {'00', '08', '88', '80'} and getattr(recovery, 'order', None) == 256
    name = '四角训练EVM' if corners else '已知训练EVM'
    return f'{name} {median(values):.2f}%（有效帧中位数）'


def auto_qam_summary(recovery):
    """Describe the effective auto path, not the legacy static decoder."""
    d = getattr(recovery, 'diagnostics', None) or {}
    auto = d.get('auto_acquisition')
    if not auto:
        return ''
    options = auto.get('effective_recovery', {}).get('tdcm_options', {}).get('compensation', d.get('options', {}))
    frames = ('已逐帧提取净荷，剔除各帧PWM前导和已知QAM训练。'
              if options.get('frame_mode') == 'all' else '按配置的单帧窗口提取净荷。')
    phase_mode = options.get('phase_mode', 'off')
    dynamic = (d.get('phase_tracking') or {}).get('tracking_strategy') == 'known_header_dynamic_frequency'
    phase = ('10kHz专用动态频偏解缠＋三次相位插值；已通过已知前导逐头留一和独立QAM训练验证。'
             if dynamic else '已知PWM前导相位跟踪；仅用已知前导估计公共相位，使用各帧已知训练校准幅相。'
             if phase_mode == 'pwm_headers' else
             '未启用连续公共相位跟踪；使用各帧已知训练校准幅相。' if phase_mode == 'off' else
             '连续公共相位补偿：'+str(phase_mode)+'；使用已知训练校准幅相。')
    if dynamic:
        erased=(d.get('phase_tracking') or {}).get('boundary_erased_frames',[])
        phase+=f'双侧前导不足的边界帧擦除{len(erased)}帧；不外推未知净荷相位。'
    preview = ' '.join(''.join(map(str, b)) for b in recovery.bits[:24])
    return (auto_acquisition_note(recovery)+'\n'
            + f'补偿后的{recovery.order}QAM净荷星座：{len(recovery.scatter)}个候选符号；'
            + f'净荷最近邻EVM {recovery.evm_percent:.2f}%；'+qam_training_note(recovery)+'。\n'
            + '已由已知PWM前导拟合并补偿固定时钟偏差，非连续时钟漂移跟踪。'+phase+'\n'
            + frames+'仅以已知前导与训练选参，未知净荷未作为参考；BER未测。\n'
            + '最近邻EVM不等于发端参考EVM或BER；训练误差不能代表全部净荷。\n'
            + 'Gray映射须与发端一致。候选净荷比特：'+preview)


class ReceiverWorkflows:
    def __init__(self, view, outer, before):
        self.view = view
        self.area = AREAS[0]
        self.states = {}
        self.confirmed = None
        self.replay = None
        self.portable = None
        self.label = tk.StringVar(value='待载入发送配置')
        self.metrics = tk.StringVar(value='码元率 —   BER 未测')
        self.loading = False
        self.bar = ttk.Frame(outer)
        self.bar.pack(fill='x', before=before, pady=(0, 8))
        self.area_var = tk.StringVar(value=self.area)
        self.area_buttons = []
        for name in AREAS:
            button = ttk.Radiobutton(self.bar, text=name, variable=self.area_var, value=name,
                                      command=lambda n=name: self.switch(n))
            button.pack(side='left', padx=(0, 24))
            self.area_buttons.append(button)
            view.controls.append(button)
        self.defaults = self.snapshot()
        self.panel = ttk.LabelFrame(view.parameter_body, text='发送配置', padding=8)
        self.panel.pack(fill='x', before=view._parameter_groups[0][0], pady=(0, 9))
        ttk.Label(self.panel, textvariable=self.label, wraplength=295).pack(fill='x', pady=(0, 6))
        actions = ttk.Frame(self.panel)
        actions.pack(fill='x')
        self.load_button = ttk.Button(actions, text='载入配置…', command=self.load)
        self.load_button.pack(side='left', fill='x', expand=True)
        self.edit_button = ttk.Button(actions, text='帧向导…', command=self.edit)
        self.edit_button.pack(side='left', fill='x', expand=True)
        self.new_protocol_button = ttk.Button(self.panel, text='旧66符号 · 10前导…', command=self.load_new_pwm_protocol)
        self.plus_symbols_button = ttk.Button(self.panel, text='新78符号 · 配置帧…', command=self.configure_plus_symbols)
        self.qam_rate_button = ttk.Button(self.panel, text='设置QAM速率…', command=self.set_qam_rate)
        self.template_button = ttk.Button(self.panel, text='从三用户模板开始…', command=self.template)
        self.confirm_button = ttk.Button(self.panel, text='确认当前配置与发送端一致', command=self.confirm)
        self.confirm_button.pack(fill='x', pady=(6, 0))
        self.save_button = ttk.Button(self.panel, text='保存配置…', command=self.save)
        self.save_button.pack(fill='x', pady=(6, 0))
        toggle = ttk.Checkbutton(self.panel, text='高级参数', variable=view.show_advanced,
                                  command=view.refresh_parameter_visibility)
        toggle.pack(anchor='w', pady=(6, 0))
        self.auto_acquisition_toggle = ttk.Checkbutton(self.panel,
            text='自动搜索原载波与前导时钟', variable=view.qam_auto_acquisition)
        view.controls.append(self.auto_acquisition_toggle)
        self.reference_toggle = ttk.Checkbutton(self.panel, text='按已知净荷测BER', variable=view.pwm_frame_reference_enabled)
        view.controls.append(self.reference_toggle)
        self.note = ttk.Label(self.panel, text='', wraplength=295)
        self.note.pack(fill='x', pady=(4, 0))
        self.expert_button = ttk.Button(self.panel, text='按高级参数分析', command=self.expert_run)
        view.controls.append(self.expert_button)
        for widget in (self.load_button, self.edit_button, self.template_button, self.new_protocol_button, self.plus_symbols_button,
                       self.confirm_button, self.save_button, self.qam_rate_button, toggle):
            view.controls.append(widget)
        # Short outputs remain in view, full analysis lives in Statistics.
        self.result_label = ttk.Label(outer, textvariable=self.metrics,
                                     justify='left', wraplength=1100)
        bright_hint(self.result_label)
        self.result_label.pack(fill='x', before=view.panes, pady=(0, 5))
        self.refresh()

    def settings_names(self):
        return [name for name, value in vars(self.view).items()
                if isinstance(value, tk.Variable) and not name.endswith('_summary') and (name.startswith(('pwm', 'qam', 'tdcm')) or
                    name in ('filename', 'channel', 'rate', 'nfft', 'averages', 'first_tuple',
                             'remove_dc', 'recovery_ms', 'center_hint', 'constellation_mode', 'cycle_khz'))]

    def snapshot(self):
        v = self.view
        return {'variables': {name: getattr(v, name).get() for name in self.settings_names()},
                'users': copy.deepcopy(v.pwm_frame_users),
                'calibration': copy.deepcopy(v.pwm_cycle_calibration),
                'air_crc12': getattr(v, 'pwm_air_crc12', False),
                'nearest': v.pwm_nearest_decision, 'source_sha256': v._fixed_source_sha256,
                'confirmed': self.confirmed, 'replay': copy.deepcopy(self.replay),
                'portable': copy.deepcopy(self.portable)}

    def signature(self):
        state = self.snapshot()
        state.pop('confirmed')
        state['variables'].pop('filename', None)
        for key in ('rate', 'recovery_ms'):
            try:
                state['variables'][key] = float(state['variables'][key])
            except (ValueError, TypeError):
                pass
        # Binding to a capture is checked separately by the original decoder.
        state.pop('source_sha256')
        return json.dumps(state, sort_keys=True, ensure_ascii=False)

    def editor_open(self):
        v = self.view
        editors = [getattr(v, '_workflow_editor', None), getattr(getattr(v, '_frame_builder', None), 'window', None)]
        return any(w is not None and w.winfo_exists() for w in editors)

    def switch(self, area):
        v = self.view
        if v.busy or self.editor_open() or area not in AREAS:
            self.area_var.set(self.area)
            return False
        if area == self.area:
            return True
        self.states[self.area] = self.snapshot()
        state = copy.deepcopy(self.states.get(area, self.defaults))
        if area == AREAS[1] and area not in self.states:
            from rx706_qam10_profile import build_qam10_config
            state['portable'] = build_qam10_config()
            state['replay'] = None
            state['confirmed'] = None
        self.area = area
        self.area_var.set(area)
        self.loading = v._workflow_restoring = True
        try:
            for name, value in state['variables'].items():
                getattr(v, name).set(value)
            v.pwm_frame_users = state['users']
            v.pwm_cycle_calibration = state['calibration']
            v.pwm_air_crc12 = state.get('air_crc12', False)
            v.pwm_nearest_decision = state['nearest']
            v._fixed_source_sha256 = state['source_sha256']
            v.constellation_mode.set('TDCM 谐波QAM' if area == AREAS[1] else 'PWM（占空比）')
            if area == AREAS[1]:
                v.qam_order.set('256')
                v.pwm2_input_mode.set('时间片协议')
            self.confirmed = state['confirmed']
            self.replay = state.get('replay')
            self.portable = state.get('portable')
            if self.portable is not None:
                self.sync_qam_rate_fields()
        finally:
            self.loading = v._workflow_restoring = False
        v._receiver_changed()
        v._invalidate_analysis()
        v.status.set('请选择bin与本次发送配置。')
        v.refresh_parameter_visibility()
        self.refresh()
        return True

    def accepted(self):
        if self.loading:
            return
        self.confirmed = self.signature()
        self.refresh()

    def confirm(self):
        if self.view.busy:
            return False
        try:
            if self.replay is not None and self.confirmed != self.signature():
                raise ValueError('绑定回放参数已改变，请重新载入方案')
            self.validate()
        except (ValueError, TypeError, KeyError) as exc:
            self.view.status.set('配置错误：'+str(exc))
            return False
        self.accepted()
        self.view.status.set('配置已确认；选择bin后运行。')
        return True

    def validate(self):
        v = self.view
        if self.area == AREAS[1] and v.constellation_mode.get() != 'TDCM 谐波QAM':
            raise ValueError('256QAM功能区需要TDCM模式')
        if self.area == AREAS[1] and self.replay is not None:
            from rx706_qam_replay_adapter import validate_replay
            return validate_replay(self.replay, input_path=v.filename.get())
        if self.area == AREAS[1] and self.portable is not None:
            from rx706_qam_config import validate_parameter_config
            return validate_parameter_config(self.portable)
        if self.area == AREAS[1]:
            return validate_qam_profile(self.qam_profile())
        if v.constellation_mode.get() != 'PWM（占空比）' or v.pwm2_input_mode.get() != '脉冲宽度与时延':
            raise ValueError('请载入脉冲宽度与时延帧配置')
        pulse = v.pulse_config()
        if self.area == AREAS[2] and int(v.pwm_frame_user_count.get()) != 3:
            raise ValueError('三用户验证需要恰好3个用户')
        return pulse

    def run(self):
        v = self.view
        if v.busy or v.closed or self.editor_open():
            return False
        try:
            if not v.filename.get().strip() or not Path(v.filename.get()).is_file():
                raise ValueError('请选择有效bin文件')
            if self.confirmed != self.signature():
                raise ValueError('请载入本次发送配置，或确认已编辑的配置')
            config = self.validate()
        except (ValueError, TypeError, KeyError) as exc:
            v._invalidate_analysis()
            v.status.set(str(exc))
            self.refresh()
            return False
        if self.area == AREAS[1] and self.replay is not None:
            self.run_replay()
        elif self.area == AREAS[1] and self.portable is not None:
            return self.run_portable(config)
        elif self.area == AREAS[1]:
            v.analyze()
        else:
            if not v.pwm_pulse_codebook.get().strip():
                # Explicit pulse geometry deterministically defines this table.
                v.pwm_pulse_codebook.set('\n'.join(f'{d:g} {w:g}' for d, w in config['codebook']))
                self.accepted()
            v.start_pwm_autotune()
        return True

    def expert_run(self):
        v = self.view
        if v.busy or v.closed or self.editor_open() or self.replay is not None or self.portable is not None:
            return False
        if self.area == AREAS[1] and (v.constellation_mode.get() != 'TDCM 谐波QAM' or v.qam_order.get() != '256'):
            v.status.set('256QAM功能区要求TDCM 256QAM；旧PWM请在PWM功能区使用高级参数。')
            return False
        v.analyze()
        return True

    def run_replay(self):
        import threading
        from rx706_spectrum import AnalysisCancelled, display_trace
        from rx706_qam_replay_adapter import run_replay
        v = self.view
        recipe = copy.deepcopy(self.replay)
        v._invalidate_analysis()
        v._analysis_token = v._autotune_token
        v.busy = True
        v.cancel.clear()
        v._buttons()
        v.status.set('正在校验文件并按绑定方案复算…')
        def worker():
            try:
                spectrum, recovered = run_replay(recipe, cancel_event=v.cancel)
                v.events.put(('result', (spectrum, display_trace(spectrum), recovered, None)))
            except AnalysisCancelled as exc:
                v.events.put(('cancelled', str(exc)))
            except Exception as exc:
                v.events.put(('error', str(exc)))
        v.thread = threading.Thread(target=worker, daemon=True, name='rx706-bound-replay')
        v.thread.start()

    def run_portable(self, config):
        """Keep transmitter parameters separate from the current capture controls."""
        import math
        import threading
        v = self.view
        try:
            options = dict(channel=1 if v.channel.get() == 'RX1' else 2,
                nfft=int(v.nfft.get()), averages=int(v.averages.get()),
                first_tuple=int(v.first_tuple.get()), sample_rate_hz=float(v.rate.get())*1e6,
                remove_dc=v.remove_dc.get())
            recovery = {key: copy.deepcopy(config[key]) for key in
                ('order', 'symbol_hz', 'tdcm_options', 'pwm_preamble_options', 'preamble_options')}
            automatic = v.qam_auto_acquisition.get()
            try:
                center_hint = float(v.tdcm_carrier_mhz.get())*1e6
                if automatic and not math.isfinite(center_hint):
                    center_hint = None
            except (ValueError, TypeError, OverflowError):
                if not automatic:
                    raise ValueError('手动模式需要有效的原载波频率')
                center_hint = None
            recovery.update(kind='qam', auto_acquisition=automatic,
                            duration_s=float(v.recovery_ms.get())/1000,
                            center_hint_hz=center_hint)
            if (options['channel'] not in (1, 2) or v.channel.get() not in ('RX1', 'RX2') or
                    options['first_tuple'] < 0 or options['nfft'] < 1 or options['averages'] < 1 or
                    not math.isfinite(options['sample_rate_hz']) or options['sample_rate_hz'] <= 0 or
                    not math.isfinite(recovery['duration_s']) or recovery['duration_s'] <= 0 or
                    (not automatic and not math.isfinite(recovery['center_hint_hz']))):
                raise ValueError('采集参数须有效，采样率和时长须为有限正数')
            from rx706_tdcm import validate_tdcm
            from rx706_qam_pwm import validate_pwm_header
            validate_tdcm(options['sample_rate_hz'], recovery['symbol_hz'], recovery['tdcm_options'])
            validate_pwm_header(recovery['pwm_preamble_options'], options['sample_rate_hz'])
        except (ValueError, TypeError, OverflowError) as exc:
            v.status.set('采集参数错误：'+str(exc))
            return False
        v._invalidate_analysis()
        v._analysis_token = v._autotune_token
        v.busy = True
        v.cancel.clear()
        v._buttons()
        v.status.set('正在搜索原载波并拟合已知PWM前导时钟…' if recovery['auto_acquisition'] else
                     '正在按完整通用TDCM配置及手动原载波分析当前采样…')
        v.thread = threading.Thread(target=v._analyze_worker,
            args=(v.filename.get().strip(), options, recovery), daemon=True, name='rx706-portable-qam')
        v.thread.start()
        return True

    def show_replay(self):
        from tkinter import scrolledtext
        dialog = tk.Toplevel(self.view.window)
        dialog.title('通用TDCM参数（只读；编辑JSON后重新载入）' if self.portable is not None else '绑定回放参数（只读）')
        dialog.geometry('760x600')
        text = scrolledtext.ScrolledText(dialog, wrap='word')
        text.pack(fill='both', expand=True)
        text.insert('1.0', json.dumps(self.portable if self.portable is not None else self.replay, ensure_ascii=False, indent=2))
        text.configure(state='disabled')

    def qam_profile(self):
        return {'schema': TDCM_PROFILE_SCHEMA,
                'settings': {key: getattr(self.view, key).get() for key in QAM_FIELDS}}

    def configure_plus_symbols(self):
        """Open a draft only. The uploaded alphabet says nothing about frame length."""
        v = self.view
        if v.busy or v.closed or self.editor_open() or self.area != AREAS[0]:
            return False
        try:
            from rx706_pwm_symbols import plus_fields
            from rx706_pwm_frame_builder import FrameBuilderDialog
            fields = plus_fields()
            v._frame_builder = FrameBuilderDialog(v.window, fields=fields,
                apply_callback=self.apply_plus_symbols)
            v._frame_builder.note.set('78项是码表大小，不是净荷长度。请填写实际净荷区长度、核对发端码元率；10前导已按物理波形换号，参考默认关闭。')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror('78符号草稿未打开', str(exc), parent=v.window)
            return False
        return True

    def apply_plus_symbols(self, profile):
        v = self.view
        if v.busy or v.closed or self.area != AREAS[0]:
            return False
        if not v.load_frame_config(config=profile):
            return False
        v.filename.set('')
        v._fixed_source_sha256 = None
        self.replay = None
        v.first_tuple.set('0')
        v.pwm_decimation.set('1')
        v.recovery_ms.set('2000')
        self.confirmed = None
        v._receiver_changed()
        v._invalidate_analysis()
        self.refresh()
        v.status.set('已应用新符号配置，已清除旧BIN；选择本次新采样并确认发送配置后运行。自动搜索支持完整2秒。')
        return True

    def load_new_pwm_protocol(self):
        """Select the validated new sender protocol, never reuse an old capture."""
        v = self.view
        if v.busy or v.closed or self.editor_open() or self.area != AREAS[0]:
            return False
        try:
            import hashlib
            from rx706_pwm_pulse import validate_pulse_options
            data = FIXED_PREAMBLE10_PROFILE.read_bytes()
            if hashlib.sha256(data).hexdigest() != FIXED_PREAMBLE10_SHA256:
                raise ValueError('新协议模板与已验证版本不一致；请用载入配置明确选择其他协议')
            config = json.loads(data.decode('utf-8-sig'))
            if config.get('source_sha256') or config.get('reference_enabled'):
                raise ValueError('新协议入口必须使用无采样绑定、无已知参考的模板')
            validate_pulse_options(dict(config, step_percent=config['duty_step'],
                symbol_count=config['duty_count'], frame_options=dict(config, reference_format='symbols')),
                float(config['symbol_khz'])*1000)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror('新协议未载入', str(exc), parent=v.window)
            return False
        if not v.load_frame_config(config=config):
            return False
        v.filename.set('')
        v._fixed_source_sha256 = None
        self.replay = None
        v.first_tuple.set('0')
        v.pwm_decimation.set('1')
        v.recovery_ms.set('1000')
        self.confirmed = None
        v._receiver_changed()
        v._invalidate_analysis()
        self.refresh()
        v.summary.set('10前导新协议：10个身份前导＋66个净荷符号；无旧校准、无已知参考。请采集并选择匹配新协议的新BIN。')
        v.status.set('已载入10前导新协议，已清除旧BIN；请选择新采样并确认发送配置。已知0～65参考须另行载入对应BER配置。')
        return True

    def load(self, filename=None):
        v = self.view
        if v.busy:
            return False
        filename = filename or filedialog.askopenfilename(parent=v.window, title='载入本次发送配置',
                                                         filetypes=[('JSON', '*.json')])
        if not filename:
            return False
        try:
            config = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
            if self.area == AREAS[1] and config.get('method') == 'iq_bound_tdcm_replay':
                from rx706_qam_replay_adapter import load_replay
                recipe = load_replay(filename, input_path=v.filename.get().strip() or None)
                self.loading = v._workflow_restoring = True
                try:
                    v.filename.set(recipe['input_path'])
                    for key, variable in (('first_tuple', v.first_tuple), ('nfft', v.nfft), ('averages', v.averages)):
                        variable.set(str(recipe['spectrum'][key]))
                    v.channel.set('RX'+str(recipe['spectrum']['channel']))
                    v.rate.set(str(recipe['spectrum']['sample_rate_hz']/1e6))
                    v.remove_dc.set(recipe['spectrum']['remove_dc'])
                    v.recovery_ms.set(str(recipe['recovery']['duration_s']*1000))
                    v.qam_order.set(str(recipe['recovery']['order']))
                    v._fixed_source_sha256 = None
                finally:
                    self.loading = v._workflow_restoring = False
                self.replay = recipe
                self.portable = None
                v._receiver_changed()
                v._invalidate_analysis()
            elif self.area == AREAS[1] and config.get('method') == 'tdcm_parameter_config':
                from rx706_qam_config import read_parameter_config
                portable = read_parameter_config(filename)
                if portable['order'] != 256:
                    raise ValueError('256QAM功能区只接受256QAM通用配置')
                self.portable = portable
                self.sync_qam_rate_fields()
                self.replay = None
                v._fixed_source_sha256 = None
                v._invalidate_analysis()
            elif self.area == AREAS[1]:
                settings = validate_qam_profile(config)
                self.replay = None
                self.portable = None
                self.loading = v._workflow_restoring = True
                try:
                    for key, value in settings.items():
                        getattr(v, key).set(value)
                finally:
                    self.loading = v._workflow_restoring = False
                v._receiver_changed()
                v._invalidate_analysis()
            else:
                if self.area == AREAS[2] and config.get('user_count') not in (3, '3'):
                    raise ValueError('请选择三用户发送配置')
                if not v.load_frame_config(filename=filename):
                    return False
            self.accepted()
            v.status.set('已载入：'+Path(filename).name)
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            # Failed portable loads are transactional: retain the last valid
            # configuration and confirmation, never partially replace options.
            if getattr(self, 'portable', None) is None:
                self.confirmed = None
            v._invalidate_analysis()
            self.refresh()
            messagebox.showerror('配置未载入', str(exc), parent=v.window)
            return False

    def sync_qam_rate_fields(self):
        """Mirror authoritative JSON rates without firing edit invalidations."""
        v = self.view
        old_loading, old_restoring = self.loading, getattr(v, '_workflow_restoring', False)
        self.loading = v._workflow_restoring = True
        try:
            p = self.portable
            for field, value in (('tdcm_symbol_khz', p['symbol_hz']),
                    ('tdcm_base_khz', p['tdcm_options']['base_hz']),
                    ('tdcm_pwm_khz', p['pwm_preamble_options']['symbol_hz'])):
                variable = getattr(v, field, None)
                if variable is not None:
                    variable.set(f'{value/1000:g}')
        finally:
            self.loading, v._workflow_restoring = old_loading, old_restoring

    def set_qam_rate(self, rate_khz=None):
        if self.view.busy or self.editor_open() or self.area != AREAS[1] or self.portable is None:
            return False
        if rate_khz is None:
            rate_khz = simpledialog.askfloat('设置QAM速率',
                'QAM符号率（kBaud），同步缩放PWM前导和调制基频。\n保留两者原比例；须与实际发端一致。',
                initialvalue=self.portable['symbol_hz']/1000, parent=self.view.window)
        if rate_khz is None:
            return False
        try:
            from rx706_qam_config import rescale_parameter_rate
            updated = rescale_parameter_rate(self.portable, rate_khz*1000)
        except (ValueError, TypeError) as exc:
            messagebox.showerror('速率未修改', str(exc), parent=self.view.window)
            return False
        self.portable = updated
        self.sync_qam_rate_fields()
        self.confirmed = None
        self.view._invalidate_analysis()
        self.refresh()
        self.view.status.set('速率已同步调整；请确认与发送端一致后运行。旧JSON未覆盖。')
        return True

    def save(self):
        if self.view.busy:
            return
        if self.area != AREAS[1]:
            return self.view.save_frame_config()
        try:
            config = copy.deepcopy(self.portable if self.portable is not None else self.replay) if self.portable is not None or self.replay is not None else self.qam_profile()
            if self.portable is not None:
                from rx706_qam_config import validate_parameter_config
                config = validate_parameter_config(config)
            elif self.replay is None:
                validate_qam_profile(config)
            filename = filedialog.asksaveasfilename(parent=self.view.window, title='保存TDCM设置',
                defaultextension='.json', filetypes=[('JSON', '*.json')])
            if filename:
                Path(filename).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror('保存失败', str(exc), parent=self.view.window)

    def edit(self):
        if self.view.busy:
            return
        if self.area == AREAS[2]:
            self.view.configure_frame_users()
        elif self.area == AREAS[1] and (self.replay is not None or self.portable is not None):
            self.show_replay()
        elif self.area == AREAS[1]:
            self.view.show_advanced.set(True)
            self.view.refresh_parameter_visibility()
        else:
            self.view.generate_frame_config()

    def template(self):
        v = self.view
        if v.busy:
            return
        if v.load_frame_config(filename=Path(__file__).with_name('pwm_frame_three_users.json')):
            self.confirmed = None
            v.status.set('模板待确认：请编辑实际身份前导与净荷。')
            self.refresh()
            v.configure_frame_users()

    def refresh(self):
        if not hasattr(self, 'panel'):
            return
        v = self.view
        ready = self.confirmed is not None and self.confirmed == self.signature()
        self.label.set(('通用TDCM配置已确认' if self.portable is not None else '绑定回放已确认' if self.replay is not None else '发送配置已确认') if ready else '发送配置待确认')
        self.panel.configure(text='TDCM设置' if self.area == AREAS[1] else '发送帧配置')
        self.edit_button.configure(text=('查看完整配置…' if self.portable is not None else '查看回放…' if self.replay is not None else '编辑参数…' if self.area == AREAS[1] else
                                         '用户与参考…' if self.area == AREAS[2] else '帧向导…'))
        self.qam_rate_button.pack_forget()
        if self.area == AREAS[1] and self.portable is not None:
            self.qam_rate_button.pack(fill="x", before=self.confirm_button, pady=(6, 0))
        self.auto_acquisition_toggle.pack_forget()
        if self.area == AREAS[1] and self.portable is not None:
            self.auto_acquisition_toggle.pack(anchor='w', before=self.note, pady=(4, 0))
        self.reference_toggle.pack_forget()
        if self.area != AREAS[1]:
            self.reference_toggle.pack(anchor='w', before=self.note, pady=(4, 0))
        self.new_protocol_button.pack_forget()
        self.plus_symbols_button.pack_forget()
        if self.area == AREAS[0]:
            self.new_protocol_button.pack(fill='x', before=self.confirm_button, pady=(6, 0))
            self.plus_symbols_button.pack(fill='x', before=self.confirm_button, pady=(6, 0))
        self.template_button.pack_forget()
        if self.area == AREAS[2]:
            self.template_button.pack(fill='x', before=self.confirm_button, pady=(6, 0))
        self.note.configure(text=('TDCM单谐波 · EVM不等于BER' if self.area == AREAS[1] else
                                  '顺序身份块 · 非三路射频' if self.area == AREAS[2] else '整文件码元率 / BER'))
        if self.area == AREAS[1] and self.portable is not None:
            p = self.portable
            slices = p.get('transmitter', {}).get('qam_slices')
            slice_text = f'QAM {slices} slice · ' if slices is not None else ''
            self.note.configure(text=(f"通用配置：{p['order']}QAM · {p['symbol_hz']/1000:g} kBaud · "
                f"{slice_text}"
                f"{len(p['pwm_preamble_options']['symbols'].split())}个PWM前导 · "
                f"{len(p['preamble_options']['symbols'].split())}个训练 · "
                f"{p['tdcm_options']['payload_data_symbols']}个净荷。\n"
                '速率可用“设置QAM速率”同步调整；其他参数以JSON为准。采样文件、Fs、通道与起点使用当前采集设置。'
                + ('原载波输入仅作为搜索提示，时钟由已知前导估计。' if v.qam_auto_acquisition.get() else
                   '手动模式：使用当前原载波和JSON时钟补偿设置。')+'EVM不等于BER。'))
        v.pwm_symbols_button.configure(text='QAM 符号表 / 比特流…' if self.area == AREAS[1] else 'PWM 符号表 / 比特流…')
        v.analyze_button.configure(text='运行接收')
        v.frame_builder_button.pack_forget()
        if self.area == AREAS[0]:
            v.frame_builder_button.pack(side='left', before=v.analyze_button, padx=(0, 8))
        v.fixed_pwm_button.pack_forget()
        self.results()

    def visibility(self):
        v = self.view
        self.expert_button.pack_forget()
        if v.show_advanced.get() and self.replay is None and self.portable is None:
            self.expert_button.pack(fill='x', pady=(6, 0))
        if v.show_advanced.get() and self.replay is None:
            return
        for box, _ in v._parameter_groups:
            box.pack_forget()
        # Main front door stays uncluttered. Every original setting is available
        # with the single Advanced toggle, including legacy expert modes.
        if hasattr(v, 'parameter_footer'):
            v.parameter_footer.pack_forget()

    def results(self):
        r = self.view.recovery
        if r is None:
            self.metrics.set('EVM —   BER 未测' if self.area == AREAS[1] else '物理码率等效上限 — Mbit/s（含前导/训练）   BER 未测')
            return
        if getattr(r, 'kind', None) == 'qam':
            note = auto_acquisition_note(r)
            self.metrics.set(f'TDCM {r.order}QAM   {len(r.scatter)} 净荷符号   净荷最近邻EVM {r.evm_percent:.2f}%   BER 未测'
                             + '\n'+qam_training_note(r) + ('\n'+note if note else ''))
        elif getattr(r, 'pulse_options', None):
            from rx706_pwm_autotune import file_evaluation
            e = file_evaluation(r)
            s = e['reference_overall']
            ber = f"{s['ber']:.4%}" if s['ber'] is not None else '未测'
            bler = f"{s['bler']:.3%}" if s['bler'] is not None else '未测'
            rate = e.get('decided_payload_symbol_rate_hz')
            duration = e.get('analyzed_duration_s')
            useful = f'{rate/1000:.3f} ksym/s' if rate is not None else '未测（缺少采样时长）'
            interval = f' · {duration:.6g} s' if duration is not None else ''
            gross = e.get('physical_equivalent_ceiling_bps')
            physical = f'{gross/1e6:.6f} Mbit/s' if gross is not None else '未测（缺少实测时钟）'
            equivalent = e.get('decided_payload_equivalent_ceiling_bps')
            effective = f'{equivalent/1e6:.6f} Mbit/s' if equivalent is not None else '未测'
            symbol = e.get('physical_symbol_rate_hz')
            clock = f'{symbol/1000:.3f} kBaud' if symbol is not None else '未测'
            text = (f"物理码率等效上限 {physical}（含前导/训练）   码元率 {clock}"
                    f"\n已判决净荷等效上限 {effective}（{useful}{interval}，非无误码保证）"
                    f"\n编号标签BER {ber}   BLER {bler}   净荷擦除 {e['erased_payload_symbols']}   实际传输/无误码净比特率 未知（映射/纠错未定义）")
            if s['ber'] is not None:
                text += '   （按配置参考，不含擦除）'
            if self.area == AREAS[2] and r.frame_check is not None:
                for user in r.frame_check.per_user:
                    rate = user.get('ber')
                    value = f'{rate:.4%}' if rate is not None else '未测'
                    rows = r.frames
                    mask = rows['block_complete'] & (rows['frame_part'] == '净荷') & (rows['user_id'] == user['user_id'])
                    total = int(mask.sum())
                    erased = total-int((mask & rows['valid'] & rows['identity_ok']).sum())
                    bler = f"{user['bler']:.3%}" if user.get('bler') is not None else '未测'
                    text += f"\n{user['name']}   BER {value}   BLER {bler}   净荷擦除 {erased}/{total}符号"
            crc_report = (r.pulse_options or {}).get('air_crc12_report')
            if crc_report:
                text = 'CRC12统计：仅首次锁定后的预期时隙，全传输BLER未知'
                for uid, stat in list(crc_report['users'].items()) + [('总计', crc_report['overall'])]:
                    value = stat['observed_scheduled_bler']
                    label = f'{value:.3%}' if value is not None else '未测'
                    name = f"U{uid}" if uid != "总计" else "总计"
                    text += f"\n{name} CRC窗口BLER {label}；预期 {stat['expected_complete_slots']} / 通过 {stat['expected_complete_slots']-stat['failed_slots']} / 失败 {stat['failed_slots']}"
            self.metrics.set(text)

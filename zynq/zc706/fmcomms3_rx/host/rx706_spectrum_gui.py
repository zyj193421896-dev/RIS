"""Optional Tk/Matplotlib window for local capture analysis; no board traffic."""
import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from rx706_spectrum import AnalysisCancelled, analyze_bin, display_trace

RECOVERY_MODE = "周期方波载波恢复"
PWM_MODE = "PWM v1（旧协议）"
PWM2_MODE = "PWM（占空比）"
PWM_MODES = (PWM_MODE, PWM2_MODE)
QAM_MODE = "QAM 解调"
QAM_PWM_MODE = "PWM前导 + QAM"
TDCM_MODE = "TDCM 谐波QAM"
FOUR_CORNER_MODE = "QAM：3PWM＋四角训练"
PAIRED_64_MODE = "64QAM：3PWM＋双遍四角＋成对净荷"
QAM256_120K_MODE = "256QAM：120k／64片／3+4+16"
QAM256_120K_LEGACY_MODE = "256QAM：120k／64片／旧3+8+128"
TDCM_MODES = (TDCM_MODE,FOUR_CORNER_MODE,PAIRED_64_MODE,QAM256_120K_MODE,QAM256_120K_LEGACY_MODE)
QAM_MODES = (QAM_MODE, QAM_PWM_MODE, *TDCM_MODES)
RECOVERY_MODES = (RECOVERY_MODE, PWM_MODE, PWM2_MODE, *QAM_MODES)
MODE_KINDS = {RECOVERY_MODE: 'periodic', PWM_MODE: 'pwm', PWM2_MODE: 'pwm2',
              QAM_MODE: 'qam', QAM_PWM_MODE: 'qam', TDCM_MODE: 'qam',FOUR_CORNER_MODE:'qam',PAIRED_64_MODE:'qam',QAM256_120K_MODE:'qam',QAM256_120K_LEGACY_MODE:'qam'}
PLOT_NAMES = ("总览", "频谱", "星座", "IQ 波形", "PWM 包络")
PWM150_PRESET = '150k：净荷24%～72%，前导24%×3'
PWM170_PRESET = '169.971671k：20%步长7%，8档温度计码'


class SpectrumWindow:
    def __init__(self, parent, filename=""):
        # Lazy imports keep the board-control UI usable without plotting packages.
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.window = tk.Toplevel(parent)
        self.window.title("接收信号分析 · IQ / 星座 / 频谱")
        self.window.geometry("1260x940")
        self.window.minsize(960, 700)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = self.busy = False
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.thread = None
        self.result = None
        self.recovery = None
        self._next_qam_recipe = None
        self._open_pwm_after_analysis = False
        self._pwm_dialog = None
        self._pwm_dialog_result = None
        self._qam_report = None
        self.poll_after = None
        self.filename = tk.StringVar(value=str(filename))
        self.channel = tk.StringVar(value="RX1")
        self.nfft = tk.StringVar(value="65536")
        self.averages = tk.StringVar(value="16")
        self.first_tuple = tk.StringVar(value="0")
        self.rate = tk.StringVar(value="15.36")
        self.remove_dc = tk.BooleanVar(value=True)
        self.constellation_mode = tk.StringVar(value=PWM2_MODE)
        self.cycle_khz = tk.StringVar(value="100")
        self.recovery_ms = tk.StringVar(value="1000")
        self.center_hint = tk.StringVar(value="自动")
        self.qam_order = tk.StringVar(value="256")
        self._last_qam_order = 256
        self.four_corner_payload = tk.StringVar(value="")
        self.qam_training_repeats = tk.StringVar(value="1")
        self.qam_payload_repeat = tk.StringVar(value="1")
        self.qam_extra_tail_symbols = tk.StringVar(value="0")
        self.qam_crc_enabled = tk.BooleanVar(value=False)
        self.qam_symbol_khz = tk.StringVar(value="100")
        self.qam_rolloff = tk.StringVar(value="0.35")
        self.qam_pulse = tk.StringVar(value="RRC")
        self.qam_carrier_mhz = tk.StringVar(value="0")
        self.qam_preamble_enabled = tk.BooleanVar(value=False)
        self.qam_preamble_format = tk.StringVar(value="符号值（Gray）")
        self.qam_preamble_symbols = tk.StringVar(value="0 0 0")
        self.qam_preamble_frequency_khz = tk.StringVar(value="1")
        self.qam_preamble_evm = tk.StringVar(value="10")
        self.qam_preamble_search_ms = tk.StringVar(value="1000")
        self.qam_training_orientation = tk.StringVar(value='不共轭（原流程）')
        self.qam_training_offset = tk.BooleanVar(value=False)
        self.qam_pwm_symbols = tk.StringVar(value="0 0 0")
        self.qam_pwm_gap_us = tk.StringVar(value="0")
        self.qam_pwm_search_ms = tk.StringVar(value="1000")
        self.qam_pwm_error = tk.StringVar(value="25")
        self.qam_pwm_width_offset = tk.StringVar(value="0")
        self.qam_pwm_frequency_khz = tk.StringVar(value="1")
        self.qam_pwm_qam_window = tk.StringVar(value="8")
        self.tdcm_symbol_khz = tk.StringVar(value="19.53125")
        self.tdcm_base_khz = tk.StringVar(value="19.53125")
        self.tdcm_pwm_khz = tk.StringVar(value="19.53125")
        self.tdcm_carrier_mhz = tk.StringVar(value="-0.2")
        self.tdcm_harmonic = tk.StringVar(value="1")
        self.tdcm_normalization = tk.StringVar(value="PWM两态参考")
        self.tdcm_amplitude = tk.StringVar(value="1")
        self.tdcm_phase = tk.StringVar(value="0")
        self.tdcm_count = tk.StringVar(value="0")
        self.pwm_symbol_khz = tk.StringVar(value="169.971671")
        self.pwm2_total_chips = tk.StringVar(value="25")
        self.pwm2_prefix_chips = tk.StringVar(value="5")
        self.pwm2_guard_chips = tk.StringVar(value="8")
        self.pwm2_preset = tk.StringVar(value="自定义")
        self.pwm2_layout_summary = tk.StringVar()
        self.pwm2_decision_mode = tk.StringVar(value="连续脉宽")
        self.pwm2_training_enabled = tk.BooleanVar(value=True)
        self.pwm2_training_tolerance = tk.StringVar(value="0.5")
        self.pwm2_preamble_start = tk.StringVar(value="20")
        self.pwm2_preamble_step = tk.StringVar(value="0")
        self.pwm2_preamble_count = tk.StringVar(value="3")
        self.pwm2_preamble_summary = tk.StringVar()
        self.pwm2_payload_count = tk.StringVar(value="自动")
        self.pwm2_auto_scale = tk.BooleanVar(value=True)
        self.pwm2_clock_width = tk.BooleanVar(value=True)
        self.pwm2_mapping = tk.StringVar(value="高片数减前高")
        self.pwm2_width_table = tk.StringVar(value="6 7 8 9 10 11 12 13 14 15 16 17 18")
        self.show_advanced = tk.BooleanVar(value=False)
        self.pwm2_input_mode = tk.StringVar(value='脉冲宽度与时延')
        self.pwm_duty_start = tk.StringVar(value='0')
        self.pwm_duty_step = tk.StringVar(value='5')
        self.pwm_duty_count = tk.StringVar(value='8')
        self.pwm_min_high_chips = tk.StringVar(value='6')
        self.pwm_joint_delay = tk.BooleanVar(value=True)
        self.pwm_low_guard_chips = tk.StringVar(value='1')
        self.pwm_pulse_codebook = tk.StringVar(value='')
        self.pwm_pulse_tolerance = tk.StringVar(value='0.4')
        self.pwm_nearest_decision = tk.BooleanVar(value=True)
        self.pwm_duty_summary = tk.StringVar()
        self.pwm_auto_search = tk.BooleanVar(value=True)
        self.pwm_code_chips = tk.StringVar(value="8")
        self.pwm_guard_chips = tk.StringVar(value="2")
        self.pwm_polarity = tk.StringVar(value="自动（PWM约束）")
        self.pwm_decimation = tk.StringVar(value="1")
        self.pwm_tolerance = tk.StringVar(value="0.30")
        self.pwm_hysteresis = tk.StringVar(value="0.10")
        self.pwm_glitch_chips = tk.StringVar(value="0.15")
        self.pwm_reference_enabled = tk.BooleanVar(value=False)
        self.pwm_reference = tk.StringVar(value="7 6 4 2")
        self.pwm_sync_symbols = tk.StringVar(value="64")
        from rx706_pwm_frames import default_users, DEFAULT_TRAINING
        self.pwm_frame_enabled = tk.BooleanVar(value=True)
        self.pwm_frame_user_count = tk.StringVar(value='1')
        self.pwm_frame_startup_count = tk.StringVar(value='0')
        self.pwm_frame_startup_pattern = tk.StringVar(value=DEFAULT_TRAINING)
        self.pwm_frame_training_count = tk.StringVar(value='0')
        self.pwm_frame_training_pattern = tk.StringVar(value=DEFAULT_TRAINING)
        self.pwm_frame_search = tk.StringVar(value='4096')
        self.pwm_frame_identity_errors = tk.StringVar(value='1')
        self.pwm_frame_training_tolerance = tk.StringVar(value='0.3')
        self.pwm_frame_reference_enabled = tk.BooleanVar(value=False)
        self.pwm_frame_reference_format = tk.StringVar(value='符号值')
        self.pwm_frame_reference_repeat = tk.BooleanVar(value=True)
        self.pwm_frame_reference_reset = tk.BooleanVar(value=True)
        self.pwm_frame_users = default_users()
        self.auto = tk.BooleanVar(value=True)
        self.pending_download = None
        self.status = tk.StringVar(value="选择已下载的 bin，点击分析；无需连接板卡")
        self.summary = tk.StringVar(value="频率轴为 I+jQ 的有符号基带频率；0 Hz 为接收 LO。")
        self._build_ui(Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)

    def _build_ui(self, Figure, FigureCanvasTkAgg, NavigationToolbar2Tk):
        from rx706_plot_navigation import PlotNavigation
        self.controls = []
        self.plot_view = tk.StringVar(value="总览")
        self.brief_status = tk.StringVar(value="选择 IQ 文件后分析；滚轮缩放、左键拖动、双击单图放大")
        self.fullscreen = False
        self.sidebar_visible = True
        self.sidebar_width = 360
        self._active_plot_name = "总览"
        self._overview_limits = None
        self.window.geometry("1500x960")
        self.window.minsize(1000, 650)
        outer = ttk.Frame(self.window, padding=8)
        outer.pack(fill="both", expand=True)
        path_row = ttk.Frame(outer)
        path_row.pack(fill="x", pady=(0, 6))
        ttk.Label(path_row, text="IQ 文件").pack(side="left", padx=(0, 6))
        entry = ttk.Entry(path_row, textvariable=self.filename)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        browse = ttk.Button(path_row, text="打开 bin…", command=self.browse)
        browse.pack(side="left", padx=(0, 8))
        self.controls.extend((entry, browse))
        self.analyze_button = ttk.Button(path_row, text="分析信号", command=self.analyze)
        self.analyze_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(path_row, text="停止", command=self.cancel.set, state="disabled")
        self.stop_button.pack(side="left")
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 4))
        self.sidebar_button = ttk.Button(bar, text="隐藏侧栏", command=self.toggle_sidebar)
        self.sidebar_button.pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="图表视图").pack(side="left", padx=(0, 5))
        view = ttk.Combobox(bar, textvariable=self.plot_view, values=PLOT_NAMES, state="readonly", width=9)
        view.pack(side="left", padx=(0, 10))
        view.bind("<<ComboboxSelected>>", lambda event: self.set_plot_view())
        ttk.Button(bar, text="复位视图 (R)", command=lambda: self.navigation.reset()).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="最大化", command=self.maximize).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="全屏 (F11)", command=self.toggle_fullscreen).pack(side="left", padx=(0, 12))
        self.pwm_symbols_button = ttk.Button(bar, text="PWM 符号表 / 比特流…", command=self.show_pwm_symbols)
        self.pwm_symbols_button.pack(side="left", padx=(0, 10))
        self.qam_report_button = ttk.Button(bar, text="QAM 解调诊断…", command=self.show_qam_report)
        self.qam_report_button.pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="滚轮缩放 · 拖动平移 · 双击放大/返回").pack(side="left")
        ttk.Label(outer, textvariable=self.brief_status, foreground="#24486c", anchor="w").pack(fill="x", pady=(0, 4))
        self.panes = ttk.Panedwindow(outer, orient="horizontal")
        self.panes.pack(fill="both", expand=True)
        self.sidebar = ttk.Frame(self.panes, width=self.sidebar_width)
        chart = ttk.Frame(self.panes)
        self.panes.add(self.sidebar, weight=0)
        self.panes.add(chart, weight=1)
        notebook = ttk.Notebook(self.sidebar)
        notebook.pack(fill="both", expand=True, padx=(0, 6))
        parameters = ttk.Frame(notebook)
        details = ttk.Frame(notebook)
        notebook.add(parameters, text="分析参数")
        notebook.add(details, text="统计 / 状态")
        from rx706_qam_debug_gui import CompensationControls
        compensation_tab = ttk.Frame(notebook)
        notebook.add(compensation_tab, text="QAM 补偿（64 / 256）")
        self.qam_compensation = CompensationControls(compensation_tab,self.apply_qam_debug_profile,
            dict(normalization=self.tdcm_normalization,amplitude=self.tdcm_amplitude,phase=self.tdcm_phase,
                 training_enabled=self.qam_preamble_enabled,training_format=self.qam_preamble_format,
                 training_symbols=self.qam_preamble_symbols,training_evm=self.qam_preamble_evm,
                 training_frequency=self.qam_preamble_frequency_khz,training_orientation=self.qam_training_orientation,
                 training_offset=self.qam_training_offset,order=self.qam_order),recipe_callback=self.load_qam_recipe,search_callback=self.activate_qam_auto_search,
                 training_profile_callback=self.apply_four_corner_training_profile,paired_profile_callback=self.apply_paired_64_frame_profile,save_bound_callback=self.save_iq_bound_recipe)
        self.controls.extend(self.qam_compensation.widgets)
        self.info_text = scrolledtext.ScrolledText(details, wrap="word", width=32, state="disabled", font="TkDefaultFont")
        self.info_text.pack(fill="both", expand=True)
        self.parameter_canvas = tk.Canvas(parameters, highlightthickness=0, width=340)
        scrollbar = ttk.Scrollbar(parameters, orient="vertical", command=self.parameter_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.parameter_canvas.pack(side="left", fill="both", expand=True)
        self.parameter_canvas.configure(yscrollcommand=scrollbar.set)
        body = ttk.Frame(self.parameter_canvas, padding=(4, 4, 8, 10))
        body_id = self.parameter_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox("all")))
        self.parameter_canvas.bind("<Configure>", lambda event: self.parameter_canvas.itemconfigure(body_id, width=event.width))

        self._parameter_groups = []
        self._parameter_titles = {}
        self._advanced_fields = []
        mode_groups = {
            '载波恢复设置': RECOVERY_MODES,
            'QAM 解调设置': QAM_MODES,
            'QAM：可配置前导 / 导频': QAM_MODES,
            '混合模式：PWM前导 + QAM': (QAM_PWM_MODE, *TDCM_MODES),
            'TDCM：按谐波复系数解调': TDCM_MODES,
            'PWM 脉宽与信息解调': (*PWM_MODES, QAM_PWM_MODE, *TDCM_MODES),
            'PWM version2：协议与映射': (PWM2_MODE, QAM_PWM_MODE, *TDCM_MODES),
            'version2：前导与恢复': (PWM2_MODE,),
            '已知循环校验 / BER': PWM_MODES,
            '最简帧：公共训练与用户数据': (PWM2_MODE,),
        }

        def group(title):
            box = ttk.LabelFrame(body, text=title, padding=8)
            box.pack(fill="x", pady=(0, 9))
            box.columnconfigure(1, weight=1)
            self._parameter_groups.append((box, mode_groups.get(title)))
            self._parameter_titles[box] = title
            return box

        def field(box, label, variable, values=None, callback=None, advanced=False, modes=None):
            row = box.grid_size()[1]
            caption = ttk.Label(box, text=label, wraplength=140)
            caption.grid(row=row, column=0, sticky="w", padx=(0, 7), pady=4)
            widget = (ttk.Combobox(box, textvariable=variable, values=values, state="readonly", width=16)
                      if values else ttk.Entry(box, textvariable=variable, width=16))
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            if callback:
                widget.bind("<<ComboboxSelected>>", callback)
            self.controls.append(widget)
            if advanced or modes:
                self._advanced_fields.extend((w, advanced, modes) for w in (caption, widget))
            return widget

        box = group("信号解调模式")
        field(box, "解调模式", self.constellation_mode,
              ("原始 IQ", *RECOVERY_MODES), self.on_mode_changed)
        field(box, "快速预设", self.pwm2_preset,
              ("自定义", PWM150_PRESET, PWM170_PRESET, "20片：8+10+2", "50片：20+25+5", "50片：8+40+2"), self.apply_pwm2_preset)
        toggle = ttk.Checkbutton(box, text="显示高级参数", variable=self.show_advanced,
                                 command=self.refresh_parameter_visibility)
        toggle.grid(row=2, column=0, columnspan=2, sticky='w')
        self.controls.append(toggle)
        for row, label, action in ((3, '载入PWM IQ绑定JSON并分析…', self.load_pwm_recipe),
                                   (4, '保存本次PWM IQ绑定JSON…', self.save_pwm_recipe)):
            widget = ttk.Button(box, text=label, command=action)
            widget.grid(row=row, column=0, columnspan=2, sticky='ew', pady=3)
            self.controls.append(widget)
        box = group("采样与频谱")
        field(box, "通道", self.channel, ("RX1", "RX2"))
        field(box, "FFT 点数", self.nfft, tuple(str(1 << p) for p in range(8, 21)), advanced=True)
        field(box, "平均段数", self.averages, advanced=True)
        field(box, "起始采样点", self.first_tuple, advanced=True)
        field(box, "采样率 (MS/s)", self.rate)
        dc = ttk.Checkbutton(box, text="频谱去直流", variable=self.remove_dc)
        dc.grid(row=5, column=0, columnspan=2, sticky="w", pady=4)
        self.controls.append(dc)
        self._advanced_fields.append((dc, True, None))
        ttk.Checkbutton(box, text="新下载文件自动分析", variable=self.auto).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        box = group("载波恢复设置")
        field(box, "完整周期频率 (kHz，仅方波)", self.cycle_khz, modes=(RECOVERY_MODE,))
        field(box, "恢复时长 (ms)", self.recovery_ms)
        field(box, "基带载波 (MHz / 自动)", self.center_hint, advanced=True)
        box = group("QAM 解调设置")
        field(box, "QAM阶数", self.qam_order, ("16", "64", "256"), self.on_qam_order_changed)
        field(box, "符号率 (kSym/s)", self.qam_symbol_khz)
        field(box, "基带载波频偏 (MHz)", self.qam_carrier_mhz)
        field(box, "发端脉冲形状", self.qam_pulse, ("RRC", "Rectangular"))
        field(box, "RRC滚降系数", self.qam_rolloff)
        ttk.Label(box, text="载波已居中填0；参数须与发端一致。静态相位/定时估计，存在90°歧义，未测BER。", wraplength=290).grid(row=5, column=0, columnspan=2)
        field(box, "四角训练每帧物理净荷数", self.four_corner_payload)
        field(box, "四角预设训练遍数", self.qam_training_repeats,('1','2'))
        field(box, "净荷连续重复数", self.qam_payload_repeat,('1','2'))
        field(box, "每帧额外尾符号数", self.qam_extra_tail_symbols)
        row=box.grid_size()[1]
        crc=ttk.Checkbutton(box,text='CRC / TDCM帧尾调试（默认关闭）',variable=self.qam_crc_enabled)
        crc.grid(row=row,column=0,columnspan=2,sticky='w');self.controls.append(crc)
        ttk.Label(box,text='CRC参数未知时只显示未验证；不猜算法、不判通过/失败。尾部长度须观察后手动填写，原始物理符号保留。',wraplength=290).grid(row=row+1,column=0,columnspan=2,sticky='w')
        box = group("QAM：可配置前导 / 导频")
        pilot = ttk.Checkbutton(box, text="启用已知前导同步与频偏校正", variable=self.qam_preamble_enabled)
        pilot.grid(row=0, column=0, columnspan=2, sticky="w")
        self.controls.append(pilot)
        field(box, "前导表示方式", self.qam_preamble_format, ("符号值（Gray）", "HEX字节（00–FF）", "I/Q电平"))
        field(box, "前导序列（空格分隔）", self.qam_preamble_symbols)
        field(box, "残余频偏范围 ±kHz", self.qam_preamble_frequency_khz)
        field(box, "前导匹配EVM上限 (%)", self.qam_preamble_evm)
        field(box, "前导搜索时长 (ms)", self.qam_preamble_search_ms)
        ttk.Label(box, text="填3～256个已知符号；默认0 0 0仅为示例，须与发端一致。"
                  "符号值支持0b二进制、0x十六进制，使用本机Gray映射；I/Q按未归一化奇数电平填写，"
                  "如 -15,-15 -15,-15 -15,-15。去除本次匹配前导后输出后续符号。"
                  "三符号可能误匹配；仅估计固定频偏和相位，不持续跟踪，不自动剔除后续重复前导。",
                  wraplength=290).grid(row=9, column=0, columnspan=2, sticky="w")
        box = group("混合模式：PWM前导 + QAM")
        field(box, "PWM前导符号值", self.qam_pwm_symbols)
        field(box, "PWM→QAM间隔 (µs)", self.qam_pwm_gap_us)
        field(box, "PWM搜索时长 (ms)", self.qam_pwm_search_ms)
        field(box, "PWM拟合误差上限 (%)", self.qam_pwm_error)
        field(box, "高电平脉宽偏移 (片)", self.qam_pwm_width_offset)
        field(box, "PWM残余频偏范围 ±kHz", self.qam_pwm_frequency_khz)
        field(box, "QAM起始搜索窗 (符号)", self.qam_pwm_qam_window)
        ttk.Label(box, text="用于TDCM的PWM前导。PWM符号率沿用TDCM设置，"
                  "片数沿用version2（默认25/5/8）；此默认下0 0 0表示三个20%符号。"
                  "按已知序列拟合两态IQ，支持等幅反相；无需8个PWM符号。"
                  "间隔从最后一个PWM完整周期结束算起；脉宽偏移=实测高片数−标称高片数。"
                  "PWM估计粗频偏，不提供QAM绝对相位。上方QAM前导可选；勾选后须在PWM之后发送对应QAM训练符号。"
                  "不勾选则直接盲解调QAM，仍有90°歧义。当前仅处理一帧候选。",
                  wraplength=290).grid(row=9, column=0, columnspan=2, sticky="w")
        box = group("TDCM：按谐波复系数解调")
        field(box, "QAM符号率 (kSym/s)", self.tdcm_symbol_khz)
        field(box, "谐波基频 (kHz)", self.tdcm_base_khz)
        field(box, "PWM前导频率 (kHz)", self.tdcm_pwm_khz)
        field(box, "原载波基带位置 (MHz)", self.tdcm_carrier_mhz)
        field(box, "目标谐波阶数（带正负）", self.tdcm_harmonic)
        field(box, "归一化 / 校准", self.tdcm_normalization, ("PWM两态参考", "盲归一化"))
        field(box, "幅度系数", self.tdcm_amplitude)
        field(box, "相位修正 (°)", self.tdcm_phase)
        field(box, "提取符号数（0到文件尾）", self.tdcm_count)
        ttk.Label(box, text="选择“TDCM 谐波QAM”生效。默认+1阶、19.53125kHz，每51.2µs提取一个复系数；"
                  "原载波填−0.2时，+1阶约−0.18046875MHz，不能把+1阶位置填为原载波。"
                  "PWM序列、片数、切换间隔和容差沿用混合模式；此处单独配置前导频率。"
                  "不使用RRC/矩形脉冲匹配设置。PWM两态参考假定：高电平对应Γ1、低电平Γ0，"
                  "方形QAM最远角点对应该阶谐波的理论最大幅度；幅度系数和相位修正须按发端约定核对。"
                  "盲归一化仍有90°歧义。可选QAM训练须实际发在PWM后；不自动分割后续帧。",
                  wraplength=290).grid(row=9, column=0, columnspan=2, sticky="w")
        box = group("PWM 脉宽与信息解调")
        field(box, "方波频率 (kHz)", self.pwm_symbol_khz)
        simple_fields = []
        for label, variable in (('占空比步长 (百分点)', self.pwm_duty_step),
                                ('符号数量', self.pwm_duty_count)):
            entry = field(box, label, variable)
            if variable is self.pwm_duty_count:
                self._pwm_count_widget = entry
            simple_fields.extend(box.grid_slaves(row=entry.grid_info()['row']))
        duty_summary = ttk.Label(box, textvariable=self.pwm_duty_summary, wraplength=290)
        duty_summary.grid(row=3, column=0, columnspan=2, sticky='w')
        simple_fields.append(duty_summary)
        self._simple_pwm_fields = simple_fields
        self._pulse_fields = []
        for label, variable in (('最小高电平宽度 (片)', self.pwm_min_high_chips),
                                ('末尾低电平预留 (片)', self.pwm_low_guard_chips)):
            entry = field(box, label, variable)
            self._pulse_fields.extend(box.grid_slaves(row=entry.grid_info()['row']))
        joint = ttk.Checkbutton(box, text='按时延＋脉宽联合码表解码', variable=self.pwm_joint_delay,
                                command=self.refresh_parameter_visibility)
        joint.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='w')
        self._pulse_fields.append(joint)
        self.controls.append(joint)
        codebook = ttk.Button(box, text='查看 / 编辑延迟与脉宽码表…', command=self.configure_pulse_codebook)
        codebook.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=4)
        self._pulse_fields.append(codebook)
        self.controls.append(codebook)
        entry = field(box, '码字边沿容差 (片)', self.pwm_pulse_tolerance)
        self._pulse_fields.extend(box.grid_slaves(row=entry.grid_info()['row']))
        nearest = ttk.Checkbutton(box, text='超容差仍输出最近码字估计', variable=self.pwm_nearest_decision)
        nearest.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='w')
        self._pulse_fields.append(nearest)
        self.controls.append(nearest)
        for variable in (self.pwm_duty_start, self.pwm_duty_step, self.pwm_duty_count,
                         self.pwm_min_high_chips, self.pwm_joint_delay, self.pwm_pulse_codebook):
            variable.trace_add('write', lambda *args: self.update_duty_summary())
        self.pwm_low_guard_chips.trace_add('write', lambda *args: self.update_duty_summary())
        self.pwm_symbol_khz.trace_add('write', lambda *args: self.update_duty_summary())
        self.update_duty_summary()
        field(box, "v1编码时间片数", self.pwm_code_chips, ("2", "4", "8", "16"), modes=(PWM_MODE,))
        field(box, "v1末尾保护片数", self.pwm_guard_chips, modes=(PWM_MODE,))
        field(box, "控制状态 1 对应", self.pwm_polarity, ("自动（PWM约束）", "A=1", "B=1"), advanced=True)
        field(box, "处理抽取倍数", self.pwm_decimation, ("1", "2", "4", "8", "16"), advanced=True)
        field(box, "脉宽容差 (时间片)", self.pwm_tolerance, advanced=True)
        field(box, "两态滞回半宽 (比例)", self.pwm_hysteresis, advanced=True)
        field(box, "单次毛刺上限 (时间片)", self.pwm_glitch_chips, advanced=True)
        explanation = ttk.Label(box, text="v1：默认100kHz对应当前固定方波测试；1Mbps目标需填333.333333kHz，不能直接用于100kHz文件。"
                  "8+2=10片，10%～80%对应符号0～7；全0为空闲。"
                  "先选“PWM v1（旧协议）”再分析。原速率即可解调；可选抽取会先抗混叠滤波，"
                  "抽取后每片至少3点，建议4点以上；3～4点会显示计时余量提示。"
                  "毛刺上限0为严格模式；仅忽略闭合短回跳，累计不超过0.30片。"
                  "毛刺告警不等于误码。A/B映射需由训练序列核对。", wraplength=290)
        explanation.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._advanced_fields.append((explanation, True, None))
        search = ttk.Checkbutton(box, text="自动搜索有效PWM片段", variable=self.pwm_auto_search)
        self._pwm_search_widget = search
        search.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky="w", pady=4)
        self.controls.append(search)
        self._advanced_fields.append((search, True, None))
        box = group("PWM version2：协议与映射")
        field(box, "协议输入方式", self.pwm2_input_mode, ('脉冲宽度与时延', '占空比范围', '时间片协议'))
        self._time_protocol_fields = []
        field(box, "每符号总片数", self.pwm2_total_chips)
        field(box, "前置固定高片数", self.pwm2_prefix_chips)
        field(box, "末尾固定低片数", self.pwm2_guard_chips)
        field(box, "符号判决依据", self.pwm2_decision_mode, ("时间片码型", "连续脉宽"))
        field(box, "符号映射", self.pwm2_mapping, ("高片数减前高", "自定义高片数表"))
        table_entry = field(box, "符号0、1…对应高片数", self.pwm2_width_table)
        self._mapping_fields = box.grid_slaves(row=table_entry.grid_info()['row'])
        self._time_protocol_fields = [w for w in box.winfo_children() if int(w.grid_info()['row']) > 0]
        ttk.Label(box, textvariable=self.pwm2_layout_summary, wraplength=290).grid(row=8, column=0, columnspan=2, sticky='w')
        for variable in (self.pwm2_total_chips, self.pwm2_prefix_chips, self.pwm2_guard_chips,
                         self.pwm_symbol_khz, self.tdcm_pwm_khz, self.constellation_mode, self.rate, self.pwm_decimation):
            variable.trace_add('write', lambda *args: self.update_pwm2_layout())
        self.update_pwm2_layout()
        box = group("version2：前导与恢复")
        self._training_box = box
        train = ttk.Checkbutton(box, text="启用已知前导校准", variable=self.pwm2_training_enabled)
        train.grid(row=0, column=0, columnspan=2, sticky='w')
        self.controls.append(train)
        field(box, "前导起始占空比 (%)", self.pwm2_preamble_start)
        field(box, "前导占空比步长 (百分点)", self.pwm2_preamble_step)
        field(box, "前导符号数量", self.pwm2_preamble_count)
        ttk.Label(box, textvariable=self.pwm2_preamble_summary, wraplength=290).grid(row=4, column=0, columnspan=2, sticky="w")
        field(box, "每帧净荷数（自动/0/整数）", self.pwm2_payload_count)
        fit = ttk.Checkbutton(box, text="自动标定宽度档位（需覆盖全部档位）", variable=self.pwm2_auto_scale)
        fit.grid(row=6, column=0, columnspan=2, sticky='w', pady=4)
        self.controls.append(fit)
        field(box, "前导单符号拟合误差上限 (片)", self.pwm2_training_tolerance, advanced=True)
        clock = ttk.Checkbutton(box, text="使用拟合时钟测宽，减少起点抖动", variable=self.pwm2_clock_width)
        clock.grid(row=8, column=0, columnspan=2, sticky='w')
        self.controls.append(clock)
        self._advanced_fields.append((clock, True, None))
        ttk.Label(box, text="补偿请选择连续脉宽。前导匹配另计采样率决定的测宽余量。前导可同宽或多宽；帧长固定时填写净荷长度，避免前导与净荷同宽造成错位。自动档位标定只用先前观测，不使用参考循环纠错。", wraplength=290).grid(row=9, column=0, columnspan=2, sticky='w')
        self.pwm2_training_enabled.trace_add('write', lambda *args: self.refresh_parameter_visibility())
        self.pwm2_mapping.trace_add('write', lambda *args: self.refresh_parameter_visibility())
        self.pwm2_input_mode.trace_add('write', lambda *args: (self.refresh_parameter_visibility(), self.update_pwm2_layout()))
        for variable in (self.pwm2_training_enabled, self.pwm2_preamble_start, self.pwm2_payload_count,
                         self.pwm2_preamble_step, self.pwm2_preamble_count, self.pwm2_clock_width,
                         self.pwm2_auto_scale):
            variable.trace_add('write', lambda *args: self.update_duty_summary())
        box = group('最简帧：公共训练与用户数据')
        box.configure(text='用户身份前导与数据块')
        enable = ttk.Checkbutton(box, text='启用最简帧（取消可测试旧格式）', variable=self.pwm_frame_enabled,
                                 command=self.refresh_parameter_visibility)
        enable.grid(row=0, column=0, columnspan=2, sticky='w')
        self._frame_enable_widget = enable
        self.controls.append(enable)
        field(box, '用户数（默认单用户）', self.pwm_frame_user_count)
        configure = ttk.Button(box, text='用户身份与净荷 / 已知数据…', command=self.configure_frame_users)
        configure.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=4)
        self.controls.append(configure)
        check = ttk.Checkbutton(box, text='启用已知净荷 BER / 误块率测试', variable=self.pwm_frame_reference_enabled)
        check.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='w', pady=4)
        self.controls.append(check)
        field(box, '已知净荷表示方式', self.pwm_frame_reference_format, ('符号值', '二进制比特'))
        for label, variable in (('循环重复已知数据', self.pwm_frame_reference_repeat),
                                ('每块从参考起点开始（否则按用户连续）', self.pwm_frame_reference_reset)):
            widget = ttk.Checkbutton(box, text=label, variable=variable)
            widget.grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='w')
            self.controls.append(widget)
        field(box, '首身份搜索范围（周期；0全段）', self.pwm_frame_search, advanced=True)
        field(box, '身份码最大错误符号数', self.pwm_frame_identity_errors, advanced=True)
        row = box.grid_size()[1]
        for col, label, action in ((0, '保存帧配置…', self.save_frame_config), (1, '载入帧配置…', self.load_frame_config)):
            widget = ttk.Button(box, text=label, command=action)
            widget.grid(row=row, column=col, sticky='ew', pady=4)
            self.controls.append(widget)
        ttk.Label(box, text='仅循环发送“用户身份前导＋净荷”，没有缓冲和公共训练。前导用于身份识别、符号定时及下降沿拖尾校正。前导长度由序列项数决定，数据长度逐用户配置；默认单用户。',
                  wraplength=290).grid(row=box.grid_size()[1], column=0, columnspan=2, sticky='w')
        box = group("已知循环校验 / BER")
        check = ttk.Checkbutton(box, text="启用已知发端符号循环", variable=self.pwm_reference_enabled)
        check.grid(row=0, column=0, columnspan=2, sticky='w', pady=4)
        self.controls.append(check)
        field(box, "循环符号值（v2用空格分隔）", self.pwm_reference)
        field(box, "同步窗 (符号数)", self.pwm_sync_symbols)
        ttk.Label(box, text="填写解码符号值，参考仅用于校验。高片数6～18在前高6配置下对应符号0～12。擦除单列；训练不计BER。", wraplength=290).grid(row=3, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.pwm_frame_enabled.trace_add('write', lambda *args: (self.refresh_parameter_visibility(), self.update_duty_summary()))
        for variable in (self.pwm_frame_user_count, self.pwm_frame_startup_count, self.pwm_frame_training_count):
            variable.trace_add('write', lambda *args: self.update_duty_summary())
        self._geometry_family = 'qam' if self.constellation_mode.get() in QAM_MODES else 'pwm'
        self._geometry_saved = {}
        self.constellation_mode.trace_add('write', self._on_geometry_mode_changed)
        self.refresh_parameter_visibility()
        self.parameter_footer = ttk.Label(body, text="采样率优先使用文件同名JSON。修改参数后点分析；详细指标在“统计 / 状态”。", wraplength=300)
        self.parameter_footer.pack(fill="x")
        self._sidebar_wheel_tag = f"RxSidebarWheel{id(self)}"
        self.window.bind_class(self._sidebar_wheel_tag, "<MouseWheel>", self._sidebar_wheel)
        self.window.bind_class(self._sidebar_wheel_tag, "<Button-4>", self._sidebar_wheel)
        self.window.bind_class(self._sidebar_wheel_tag, "<Button-5>", self._sidebar_wheel)
        def tag_widgets(widget):
            widget.bindtags((self._sidebar_wheel_tag, *widget.bindtags()))
            for child in widget.winfo_children():
                tag_widgets(child)
        tag_widgets(self.parameter_canvas)

        self.figure = Figure(figsize=(12, 8), dpi=100, layout="constrained")
        self._create_plot_axes()
        self._empty_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.toolbar = NavigationToolbar2Tk(self.canvas, chart, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.get_tk_widget().bind("<Enter>", lambda event: self.canvas.get_tk_widget().focus_set())
        self.navigation = PlotNavigation(self.canvas, self.toolbar, self.double_click_plot, self._apply_view_layout)
        self.canvas.draw()
        self._info_traces = [(v, v.trace_add("write", self._refresh_info)) for v in (self.status, self.summary)]
        self._refresh_info()
        self.window.bind("<F11>", self.toggle_fullscreen)
        self.window.bind("<Escape>", self.exit_fullscreen)
        self.maximize()
        self.poll_after = self.window.after(50, self.poll)

    def _create_plot_axes(self):
        grid = self.figure.add_gridspec(2, 2, height_ratios=(1, 1.2))
        self.axes = self.figure.add_subplot(grid[0, :])
        self.iq_axes = self.figure.add_subplot(grid[1, 0])
        self.wave_axes = self.figure.add_subplot(grid[1, 1])
        self._plot_axes = {"频谱": self.axes, "星座": self.iq_axes,
                           "IQ 波形": self.wave_axes, "PWM 包络": self.wave_axes}
        self._overview_specs = {ax: ax.get_subplotspec() for ax in self.figure.axes}
        self._single_spec = self.figure.add_gridspec(1, 1)[0, 0]

    def _sidebar_wheel(self, event):
        steps = -1 if getattr(event, "num", None) == 4 else 1 if getattr(event, "num", None) == 5 else -int(event.delta / 120)
        if steps:
            self.parameter_canvas.yview_scroll(steps, "units")
        return "break"

    def _refresh_info(self, *args):
        if self.closed:
            return
        text = self.status.get()+"\n\n"+self.summary.get()
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")
        if self.busy:
            brief = self.status.get().splitlines()[0]
        elif "未锁定" in self.status.get() or "失败" in self.status.get():
            brief = self.status.get().splitlines()[-1]
        elif self.recovery is not None:
            r = self.recovery
            if r.kind == 'qam':
                brief = f'{r.order}QAM | {len(r.scatter)} symbols | decision EVM {r.evm_percent:.2f}% | 未测BER'
            elif r.kind in ('pwm', 'pwm2'):
                good = int(r.frames['valid'].sum())
                warned = int((r.frames['valid'] & (r.frames['glitch_count'] > 0)).sum())
                brief = (f"{r.protocol_label} [{('码型' if r.decision_mode == 'pattern' else '脉宽')}] | 符号率 {r.measured_symbol_rate_hz/1000:.4f} kHz | "
                         f"有效 {good}/{len(r.frames)}（毛刺告警{warned}） | {r.mapping_note} | 原Fs / {r.decimation}")
                if r.frame_check is not None:
                    stats = r.frame_check.overall
                    brief += (f" | 错位{stats['bit_errors']}/{stats['compared_bits']}，"
                              f"错块{stats['bad_blocks']}/{stats['evaluated_blocks']}，擦除{stats['erased_bits']}位")
                elif r.reference_check is not None:
                    c = r.reference_check
                    brief += (f" | 错比特 {c.bit_errors}/{c.compared_bits}，擦除 {c.erasures}/{c.evaluation_symbols}"
                              if c.locked else " | 参考循环未同步，未测BER")
            else:
                brief = f"周期恢复 | 载波 {r.carrier_hz/1e6:+.6f} MHz | 周期 {r.cycle_hz/1000:.4f} kHz | 两态相位差 {abs(r.phase_difference_deg):.2f}° | 幅度差 {r.amplitude_difference_db:+.2f} dB"
        else:
            brief = self.status.get().splitlines()[0] if self.status.get() else ""
        self.brief_status.set(brief[:180])

    def maximize(self, event=None):
        try:
            self.window.state("zoomed")
        except tk.TclError:
            self.window.geometry(f"{self.window.winfo_screenwidth()}x{max(650,self.window.winfo_screenheight()-80)}+0+0")

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.window.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen(self, event=None):
        self.fullscreen = False
        self.window.attributes("-fullscreen", False)

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_width = max(250, self.panes.sashpos(0))
            self.panes.forget(self.sidebar)
            self.sidebar_button.configure(text="显示侧栏")
        else:
            self.panes.insert(0, self.sidebar, weight=0)
            self.panes.sashpos(0, self.sidebar_width)
            self.sidebar_button.configure(text="隐藏侧栏")
        self.sidebar_visible = not self.sidebar_visible

    def _apply_view_layout(self):
        if not hasattr(self, "_overview_specs"):
            return
        chosen = self._plot_axes.get(self.plot_view.get())
        for ax, spec in self._overview_specs.items():
            visible = chosen is None or ax is chosen
            ax.set_visible(visible)
            ax.set_in_layout(visible)
            ax.set_subplotspec(self._single_spec if ax is chosen else spec)
        if chosen is self.iq_axes:
            chosen.set_aspect("equal", adjustable="box")
            # Expand the bounds once to fill the large panel without allowing
            # Matplotlib's aspect solver to override cursor zoom/pan limits.
            panel = self._single_spec.get_position(self.figure)
            fw,fh = self.figure.get_size_inches()
            ratio = panel.width*fw/(panel.height*fh)
            xlim,ylim = chosen.get_xlim(),chosen.get_ylim()
            cx,cy = sum(xlim)/2,sum(ylim)/2
            width,height = abs(xlim[1]-xlim[0]),abs(ylim[1]-ylim[0])
            width = max(width,height*ratio)
            height = max(height,width/ratio)
            chosen.set_xlim(cx-width/2,cx+width/2)
            chosen.set_ylim(cy-height/2,cy+height/2)
        elif chosen is None:
            self.iq_axes.set_aspect("equal", adjustable="box")
            # Fill the overview panel while retaining identical I/Q units.
            # Two nearly antipodal states otherwise collapse the axes to a
            # thin strip, moving titles and labels when limits change.
            panel = self._overview_specs[self.iq_axes].get_position(self.figure)
            fw,fh = self.figure.get_size_inches()
            ratio = panel.width*fw/(panel.height*fh)
            xlim,ylim = self.iq_axes.get_xlim(),self.iq_axes.get_ylim()
            cx,cy = sum(xlim)/2,sum(ylim)/2
            width,height = abs(xlim[1]-xlim[0]),abs(ylim[1]-ylim[0])
            width = max(width,height*ratio)
            height = max(height,width/ratio)
            self.iq_axes.set_xlim(cx-width/2,cx+width/2)
            self.iq_axes.set_ylim(cy-height/2,cy+height/2)

    def _plots_changed(self):
        for ax in self.figure.axes:
            ax.title.set_fontsize(10)
            ax.title.set_wrap(True)
        if hasattr(self, "_overview_specs"):
            self._overview_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.figure.axes}
            self._apply_view_layout()
        navigation = getattr(self, "navigation", None)
        if navigation is not None:
            navigation.new_data()
        self.canvas.draw_idle()

    def set_plot_view(self, name=None):
        if name is not None:
            self.plot_view.set(name)
        previous = self._active_plot_name
        if previous == "总览" and self.plot_view.get() != "总览":
            self._overview_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self.figure.axes}
        elif self.plot_view.get() == "总览" and self._overview_limits is not None:
            for ax, (xlim,ylim) in self._overview_limits.items():
                ax.set_xlim(xlim); ax.set_ylim(ylim)
        self._apply_view_layout()
        self._active_plot_name = self.plot_view.get()
        self.canvas.draw_idle()

    def double_click_plot(self, ax):
        name = next((name for name,item in self._plot_axes.items() if item is ax), "总览")
        self.set_plot_view(name if self.plot_view.get() == "总览" else "总览")

    def _empty_axes(self):
        for ax in (self.axes, self.iq_axes, self.wave_axes):
            ax.clear()
            ax.grid(alpha=.25)
        self.axes.set(xlabel='Baseband frequency (MHz), I + jQ', ylabel='Amplitude (dBFS / FFT bin)', ylim=(-140,5))
        self.iq_axes.set(title='Raw IQ scatter', xlabel='I (ADC counts)', ylabel='Q (ADC counts)')
        self.iq_axes.set_aspect('equal', adjustable='box')
        self.wave_axes.set(title='Contiguous IQ waveform', xlabel='Time from selection start (us)', ylabel='ADC counts')

    def draw_constellation(self):
        if self.result is None or self.closed:
            return
        import numpy as np
        result = self.result
        expected_kind = MODE_KINDS.get(self.constellation_mode.get())
        if self.recovery is not None and expected_kind == self.recovery.kind:
            self.draw_recovered()
            return
        x = result.constellation
        self.iq_axes.clear()
        self.iq_axes.scatter(x.real, x.imag, s=3, alpha=.25, linewidths=0, rasterized=True)
        self.iq_axes.set(title=f'Raw IQ | {len(x):,} samples', xlabel='I (ADC counts)', ylabel='Q (ADC counts)')
        self.iq_axes.set_aspect('equal', adjustable='box')
        self.iq_axes.grid(alpha=.25)
        z = result.waveform
        time_us = np.arange(len(z))/result.sample_rate_hz*1e6
        self.wave_axes.clear()
        for y, label in ((z.real,'I'), (z.imag,'Q'), (abs(z),'Magnitude')):
            self.wave_axes.plot(time_us, y, lw=.6, label=label)
        self.wave_axes.set(title='Raw IQ waveform', xlabel='Time from selection start (us)', ylabel='ADC counts')
        self.wave_axes.legend(fontsize=7)
        self.wave_axes.grid(alpha=.25)
        self._plots_changed()

    def draw_qam(self):
        import numpy as np
        r = self.recovery
        for ax in (self.iq_axes, self.wave_axes):
            ax.clear()
            ax.grid(alpha=.25)
        z = r.scatter
        from rx706_qam_debug_gui import scatter_indices
        selected = scatter_indices(len(z),budget=20000,labels=r.labels)
        self.iq_axes.scatter(z.real[selected], z.imag[selected], s=3, alpha=.3)
        grid = np.unique(r.ideal)
        self.iq_axes.plot(grid.real, grid.imag, 'rx', ms=4)
        prefix = f"TDCM {r.tdcm['harmonic']:+d} | " if getattr(r,'tdcm',None) is not None else ''
        self.iq_axes.set(title=prefix+f'{r.order}QAM | decision EVM {r.evm_percent:.2f}%', xlabel='Normalized I', ylabel='Normalized Q')
        self.iq_axes.set_aspect('equal', adjustable='box')
        t = (r.time_s-r.time_s[0])*1e6
        self.wave_axes.plot(t[:1000], z.real[:1000], '.-', lw=.5, label='I')
        self.wave_axes.plot(t[:1000], z.imag[:1000], '.-', lw=.5, label='Q')
        self.wave_axes.set(title='TDCM harmonic symbols' if prefix else 'QAM symbol samples', xlabel='Time (us)')
        self.wave_axes.legend(fontsize=7)
        self._plots_changed()

    def _on_geometry_mode_changed(self, *args):
        """Keep standalone PWM geometry separate from known QAM headers."""
        family = 'qam' if self.constellation_mode.get() in QAM_MODES else 'pwm'
        if family != self._geometry_family:
            variables = (self.pwm2_total_chips, self.pwm2_prefix_chips, self.pwm2_guard_chips)
            self._geometry_saved[self._geometry_family] = tuple(v.get() for v in variables)
            self._geometry_family = family
            saved = self._geometry_saved.get(family)
            if saved is not None:
                for variable, value in zip(variables, saved):
                    variable.set(value)
        self.refresh_parameter_visibility()
        self.update_pwm2_layout()

    def refresh_parameter_visibility(self):
        """Show only settings used by the selected demodulator."""
        if not hasattr(self, '_parameter_groups'):
            return
        mode, advanced = self.constellation_mode.get(), self.show_advanced.get()
        for box, modes in self._parameter_groups:
            box.pack_forget()
            advanced_group = (mode == PWM2_MODE and self._parameter_titles[box] in (
                '采样与频谱', '载波恢复设置', 'PWM version2：协议与映射',
                'version2：前导与恢复', '已知循环校验 / BER'))
            legacy_hidden = (mode == PWM2_MODE and (self.pwm_frame_enabled.get() or self.pwm2_input_mode.get() == '脉冲宽度与时延') and
                             self._parameter_titles[box] in ('version2：前导与恢复', '已知循环校验 / BER'))
            if (modes is None or mode in modes) and (advanced or not advanced_group) and not legacy_hidden:
                box.pack(fill='x', pady=(0, 9))
        for widget in self._training_box.winfo_children():
            if widget.grid_info() or widget.winfo_manager() == '':
                widget.grid()
        for widget, needs_advanced, modes in self._advanced_fields:
            if (not needs_advanced or advanced) and (modes is None or mode in modes):
                widget.grid()
            else:
                widget.grid_remove()
        if not self.pwm2_training_enabled.get():
            for widget in self._training_box.winfo_children():
                if int(widget.grid_info().get('row', 0)) != 0:
                    widget.grid_remove()
        if self.pwm2_mapping.get() != '自定义高片数表':
            for widget in self._mapping_fields:
                widget.grid_remove()
        else:
            for widget in self._mapping_fields:
                widget.grid()
        pulse = mode == PWM2_MODE and self.pwm2_input_mode.get() == '脉冲宽度与时延'
        simple = mode == PWM2_MODE and self.pwm2_input_mode.get() in ('占空比范围', '脉冲宽度与时延')
        for widget in self._simple_pwm_fields:
            widget.grid() if simple else widget.grid_remove()
        for widget in self._time_protocol_fields:
            if mode == PWM2_MODE and self.pwm2_input_mode.get() != '时间片协议':
                widget.grid_remove()
            elif widget not in self._mapping_fields or self.pwm2_mapping.get() == '自定义高片数表':
                widget.grid()
        if mode == PWM2_MODE and self.pwm_frame_enabled.get():
            self._pwm_search_widget.grid_remove()
        for widget in self._pulse_fields:
            widget.grid() if pulse else widget.grid_remove()
        self._frame_enable_widget.grid_remove() if pulse else self._frame_enable_widget.grid()
        self._pwm_count_widget.configure(state='readonly' if pulse and self.pwm_joint_delay.get() else 'normal')
        if hasattr(self, 'parameter_footer'):
            self.parameter_footer.pack_forget()
            self.parameter_footer.pack(fill='x')

    def update_duty_summary(self):
        if self.constellation_mode.get() != PWM2_MODE:
            self.update_pwm2_layout()
            return
        from rx706_pwm_protocol import duty_protocol
        try:
            if self.pwm2_input_mode.get() == '脉冲宽度与时延':
                config = self.pulse_config()
                total = config['total_chips']
                period = 1000/float(self.pwm_symbol_khz.get())
                minimum = config['min_high_chips']
                if self.pwm_joint_delay.get() and self.pwm_duty_count.get() != str(config['symbol_count']):
                    self.pwm_duty_count.set(str(config['symbol_count']))
                rates = '；'.join(f"{u['name']} {r/1000:.3f} kbit/s" for u, r in
                                 zip(config['frame_options']['users'], config['user_information_rates']))
                self.pwm_duty_summary.set(f"步长{config['step_percent']:g}%＝{total}片；最小高电平{minimum}片＝{minimum*config['step_percent']:g}%。\n"
                    f"当前周期{period:.3f}µs，最短高电平{minimum/total*period:.3f}µs；"
                    '拖尾及周期下限待前导估计。\n'
                    f"末尾低电平预留{config['low_guard_chips']}片；合法组合{config['available_combinations']}种，当前码表{config['symbol_count']}种。\n"
                    f"每符号信息量 log₂(M)＝{config['information_bits']:.6f} bit。\n"
                    f"含前导和时分开销的单用户净荷信息率上限：{rates}。\n"
                    f"逐符号固定整数映射最多{config['integer_payload_bits']} bit（需选2的整数次幂个码字）；当前输出为符号编号的二进制标记。\n"
                    '拖尾完全由各用户前导估计；数据脉冲可延后。')
                self.pwm2_layout_summary.set(self.pwm_duty_summary.get())
                return
            preamble_text = self.preamble_summary()
            _, values = duty_protocol(self.pwm_duty_start.get(), self.pwm_duty_step.get(),
                                      self.pwm_duty_count.get())
            self.pwm_duty_summary.set('结束占空比 '+f'{values[-1]:g}%'+
                '\n符号0起：'+'、'.join(f'{v:g}%' for v in values[:16])+('…' if len(values)>16 else '')+
                '\n最近档位判决；超容差仍恢复并标注。\n'+preamble_text)
        except ValueError as exc:
            self.pwm_duty_summary.set(str(exc))
        if self.pwm2_input_mode.get() == '脉冲宽度与时延':
            self.pwm2_layout_summary.set(self.pwm_duty_summary.get())
            return
        self.update_pwm2_layout()

    def preamble_summary(self):
        if self.pwm2_input_mode.get() == '脉冲宽度与时延':
            return f'用户身份前导＋数据；{self.pwm_frame_user_count.get()}用户，长度逐用户配置。'
        if self.pwm_frame_enabled.get():
            return (f'最简帧：缓冲{self.pwm_frame_startup_count.get()}＋训练{self.pwm_frame_training_count.get()}；'
                    f'{self.pwm_frame_user_count.get()}用户；身份码与净荷长度逐用户配置。')
        from rx706_pwm_protocol import preamble_duties
        try:
            values = preamble_duties(self.pwm2_preamble_start.get(), self.pwm2_preamble_step.get(),
                                     self.pwm2_preamble_count.get())
        except ValueError as exc:
            self.pwm2_preamble_summary.set(str(exc))
            raise
        text = ('前导：'+'、'.join(f'{v:g}%' for v in values[:16])+('…' if len(values)>16 else '')+
                f'；结束 {values[-1]:g}%；共{len(values)}符号。')
        self.pwm2_preamble_summary.set(text)
        payload = self.pwm2_payload_count.get()
        if payload == '自动':
            if self.pwm2_input_mode.get() == '占空比范围':
                payload = self.pwm_duty_count.get()
            else:
                payload = '自动（按映射档位数）'
        return (text + f'\n每帧净荷 {payload}；' +
                ('矫正已启用；' if self.pwm2_training_enabled.get() else '矫正已手动关闭；') +
                ('拟合时钟测宽。' if self.pwm2_clock_width.get() else '边沿测宽。') +
                ('\n自动档位标定已启用；覆盖全部档位后生效，初期仅前导偏移。'
                 if self.pwm2_auto_scale.get() else '\n自动档位标定已关闭；同宽前导只校准偏移。'))

    def apply_pwm2_preset(self, event=None):
        selected = self.pwm2_preset.get()
        if selected != '自定义':
            self.pwm_frame_enabled.set(False)
        layout = {'20片：8+10+2': (20, 8, 2),
                  '50片：20+25+5': (50, 20, 5),
                  '50片：8+40+2': (50, 8, 2), PWM150_PRESET: (25, 6, 7)}.get(selected)
        if layout:
            self.constellation_mode.set(PWM2_MODE)
            for variable, value in zip((self.pwm2_total_chips, self.pwm2_prefix_chips,
                                        self.pwm2_guard_chips), layout):
                variable.set(str(value))
            self.pwm2_preset.set(selected)
        if selected == PWM150_PRESET:
            self.constellation_mode.set(PWM2_MODE)
            for name, value in dict(pwm_symbol_khz='150', pwm2_preamble_start='24', pwm2_preamble_step='0', pwm2_preamble_count='3', pwm2_payload_count='自动',
                    pwm2_decision_mode='连续脉宽', pwm2_training_tolerance='0.8',
                    pwm_tolerance='0.35', pwm_decimation='1', pwm_polarity='自动（PWM约束）',
                    pwm2_mapping='高片数减前高', pwm_sync_symbols='256',
                    pwm2_input_mode='占空比范围', pwm_duty_start='24', pwm_duty_step='4', pwm_duty_count='13',
                    pwm_reference=' '.join(map(str, range(13)))).items():
                getattr(self, name).set(value)
            self.pwm2_training_enabled.set(True)
            self.pwm2_auto_scale.set(True)
            self.pwm2_clock_width.set(True)
            self.pwm_reference_enabled.set(True)
        elif selected == PWM170_PRESET:
            self.constellation_mode.set(PWM2_MODE)
            for name, value in dict(pwm_symbol_khz='169.971671', pwm_duty_start='20',
                    pwm_duty_step='7', pwm_duty_count='8', pwm2_input_mode='占空比范围',
                    pwm2_preamble_start='20', pwm2_preamble_step='0', pwm2_preamble_count='3',
                    pwm2_payload_count='8', pwm2_training_tolerance='0.5',
                    pwm2_decision_mode='连续脉宽', pwm_tolerance='0.30', pwm_decimation='1',
                    pwm_polarity='自动（PWM约束）').items():
                getattr(self, name).set(value)
            self.pwm2_training_enabled.set(True)
            self.pwm2_clock_width.set(True)
            self.pwm2_auto_scale.set(True)
            self.pwm_reference_enabled.set(False)
        elif layout:
            self.pwm2_training_enabled.set(True)
            self.pwm2_clock_width.set(True)
            self.pwm2_decision_mode.set("连续脉宽")
            self.pwm_reference_enabled.set(False)
            self.pwm2_mapping.set('高片数减前高')
            self.pwm2_input_mode.set('时间片协议')
        if hasattr(self, '_parameter_groups'):
            self.refresh_parameter_visibility()
        self.update_pwm2_layout()

    def frame_config(self, widths=None, users=None):
        from rx706_pwm_frames import default_users, validate_frame_options
        if widths is None:
            if self.pwm2_input_mode.get() == '脉冲宽度与时延':
                widths = tuple(range(32768))
            else:
                from rx706_pwm_protocol import duty_protocol
                compiled, _ = duty_protocol(self.pwm_duty_start.get(), self.pwm_duty_step.get(), self.pwm_duty_count.get())
                widths = compiled['symbol_widths']
        count = int(self.pwm_frame_user_count.get())
        if not 1 <= count <= 16:
            raise ValueError('用户数须为1～16')
        if len(self.pwm_frame_users) < count:
            alphabet = min(len(widths), int(self.pwm_duty_count.get()))
            self.pwm_frame_users.extend(default_users(count, alphabet)[len(self.pwm_frame_users):])
        config = dict(user_count=count, users=(self.pwm_frame_users if users is None else users)[:count],
            startup_symbols=0, startup_pattern='', training_symbols=0, training_pattern='',
            search_symbols=self.pwm_frame_search.get(), identity_errors=self.pwm_frame_identity_errors.get(),
            training_tolerance=self.pwm_frame_training_tolerance.get(), clock_width=self.pwm2_clock_width.get(),
            reference_enabled=self.pwm_frame_reference_enabled.get(),
            reference_format='bits' if self.pwm_frame_reference_format.get() == '二进制比特' else 'symbols',
            reference_repeat=self.pwm_frame_reference_repeat.get(), reference_reset=self.pwm_frame_reference_reset.get())
        return validate_frame_options(config, widths)

    def pulse_config(self):
        from rx706_pwm_pulse import validate_pulse_options
        config = dict(step_percent=self.pwm_duty_step.get(), symbol_count=int(self.pwm_duty_count.get()),
            min_high_chips=int(self.pwm_min_high_chips.get()),
            low_guard_chips=int(self.pwm_low_guard_chips.get()),
            joint_delay=self.pwm_joint_delay.get(),
            codebook=self.pwm_pulse_codebook.get(), frame_options=self.frame_config())
        if hasattr(self, 'pwm_pulse_tolerance'):
            config['tolerance_chips'] = float(self.pwm_pulse_tolerance.get())
            config['nearest_decision'] = self.pwm_nearest_decision.get()
        return validate_pulse_options(config, float(self.pwm_symbol_khz.get())*1000)

    def configure_pulse_codebook(self):
        dialog = tk.Toplevel(self.window)
        dialog.title('符号值 → 延迟片数、高电平片数')
        dialog.geometry('650x520')
        dialog.transient(self.window)
        box = ttk.Frame(dialog, padding=12)
        box.pack(fill='both', expand=True)
        ttk.Label(box, text='每行“延迟片数 高电平片数”，按行对应符号0、1…；起点为0。联合码表时两项共同决定符号；仅脉宽映射时数据时延单列恢复。前导总按此表的已知延迟定位。', wraplength=610).pack(fill='x')
        text = scrolledtext.ScrolledText(box, height=18)
        text.pack(fill='both', expand=True, pady=8)
        value = self.pwm_pulse_codebook.get()
        if not value:
            try:
                config = self.pulse_config()
                value = '\n'.join(f'{d:g} {w:g}' for d, w in config['codebook'])
            except ValueError:
                value = ''
        text.insert('1.0', value)
        def save():
            old = self.pwm_pulse_codebook.get()
            self.pwm_pulse_codebook.set(text.get('1.0', 'end').strip())
            try:
                self.pulse_config()
            except ValueError as exc:
                self.pwm_pulse_codebook.set(old)
                messagebox.showerror('码表错误', str(exc), parent=dialog)
                return
            dialog.destroy()
        ttk.Button(box, text='应用码表', command=save).pack(side='right')
        ttk.Button(box, text='恢复全部合法时延/脉宽组合', command=lambda: (self.pwm_pulse_codebook.set(''), self.pwm_joint_delay.set(True), dialog.destroy())).pack(side='left')

    def configure_frame_users(self):
        """Edit each user without coupling marker/payload lengths across users."""
        from pathlib import Path
        from rx706_pwm_frames import default_users, parse_symbols
        try:
            count, alphabet = int(self.pwm_frame_user_count.get()), int(self.pwm_duty_count.get())
            if not 1 <= count <= 16 or not 2 <= alphabet <= 32768:
                raise ValueError('用户数须为1～16，码字数须为2～32768')
        except ValueError as exc:
            messagebox.showerror('配置错误', str(exc), parent=self.window)
            return
        if len(self.pwm_frame_users) < count:
            self.pwm_frame_users.extend(default_users(count, alphabet)[len(self.pwm_frame_users):])
        users = [dict(user) for user in self.pwm_frame_users]
        dialog = tk.Toplevel(self.window)
        dialog.title('用户身份与净荷 / 已知数据')
        dialog.geometry('780x600')
        dialog.transient(self.window)
        box = ttk.Frame(dialog, padding=12)
        box.pack(fill='both', expand=True)
        current, name, identity, length = (tk.StringVar() for _ in range(4))
        selected = [0]
        chooser = ttk.Combobox(box, textvariable=current, state='readonly', values=[str(i+1) for i in range(count)])
        chooser.pack(fill='x')
        for label, variable in (('用户名称', name), ('身份码：空格分隔的PWM符号值（序列项数就是长度）', identity),
                                ('每块净荷长度（PWM符号数）', length)):
            ttk.Label(box, text=label).pack(anchor='w', pady=(10, 2))
            ttk.Entry(box, textvariable=variable).pack(fill='x')
        ttk.Label(box, text='已知净荷参考（可留空）；格式及循环规则沿用主窗口设置').pack(anchor='w', pady=(10, 2))
        reference = scrolledtext.ScrolledText(box, height=9, wrap='word')
        reference.pack(fill='both', expand=True)
        ttk.Label(box, text='填写真实发端数据；符号值用空格分隔，比特用0/1。也可载入UTF-8文本文件。身份码和参考都可自定义；参考仅用于测试，不改变接收结果。',
                  wraplength=730).pack(fill='x', pady=8)

        def store_user():
            code = parse_symbols(identity.get(), alphabet)
            size = int(length.get())
            if not 4 <= len(code) <= 256 or not 1 <= size <= 1000000:
                raise ValueError('身份码长度须为4～256，净荷长度须为1～1000000')
            users[selected[0]] = dict(name=name.get().strip() or f'U{selected[0]+1}', identity=' '.join(map(str, code)),
                                      payload_symbols=size, reference=reference.get('1.0', 'end').strip())

        def show_user(at):
            selected[0] = at
            user = users[at]
            current.set(str(at+1))
            name.set(user['name'])
            code = user['identity']
            identity.set(code if isinstance(code, str) else ' '.join(map(str, code)))
            length.set(str(user['payload_symbols']))
            reference.delete('1.0', 'end')
            value = user.get('reference', '')
            reference.insert('1.0', value if isinstance(value, str) else ' '.join(map(str, value)))

        def switch(event):
            at = int(current.get())-1
            try:
                store_user()
            except ValueError as exc:
                current.set(str(selected[0]+1))
                messagebox.showerror('配置错误', str(exc), parent=dialog)
                return
            show_user(at)

        def load_reference():
            filename = filedialog.askopenfilename(parent=dialog, title='选择已知净荷文本', filetypes=[('UTF-8文本', '*.txt'), ('所有文件', '*.*')])
            if filename:
                reference.delete('1.0', 'end')
                reference.insert('1.0', '@'+str(Path(filename).resolve()))

        def save():
            try:
                store_user()
                self.frame_config(users=users)
            except (ValueError, OSError) as exc:
                messagebox.showerror('配置错误', str(exc), parent=dialog)
                return
            self.pwm_frame_users = users
            self.update_duty_summary()
            dialog.destroy()

        chooser.bind('<<ComboboxSelected>>', switch)
        buttons = ttk.Frame(box)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='载入已知数据文件…', command=load_reference).pack(side='left')
        ttk.Button(buttons, text='保存所有用户', command=save).pack(side='right')
        show_user(0)

    def save_frame_config(self):
        try:
            pulse = self.pwm2_input_mode.get() == '脉冲宽度与时延'
            self.pulse_config() if pulse else self.frame_config()
            config = dict(user_count=int(self.pwm_frame_user_count.get()), users=self.pwm_frame_users,
                startup_symbols=self.pwm_frame_startup_count.get(), startup_pattern=self.pwm_frame_startup_pattern.get(),
                training_symbols=self.pwm_frame_training_count.get(), training_pattern=self.pwm_frame_training_pattern.get(),
                search_symbols=self.pwm_frame_search.get(), identity_errors=self.pwm_frame_identity_errors.get(),
                training_tolerance=self.pwm_frame_training_tolerance.get(),
                reference_enabled=self.pwm_frame_reference_enabled.get(), reference_format=self.pwm_frame_reference_format.get(),
                reference_repeat=self.pwm_frame_reference_repeat.get(), reference_reset=self.pwm_frame_reference_reset.get(),
                clock_width=self.pwm2_clock_width.get(),
                symbol_khz=self.pwm_symbol_khz.get(), duty_start=self.pwm_duty_start.get(),
                duty_step=self.pwm_duty_step.get(), duty_count=self.pwm_duty_count.get())
            if pulse:
                for key in ('startup_symbols', 'startup_pattern', 'training_symbols', 'training_pattern', 'duty_start'):
                    config.pop(key, None)
                config.update(schema_version=3, mode='脉冲宽度与时延',
                    min_high_chips=self.pwm_min_high_chips.get(), low_guard_chips=self.pwm_low_guard_chips.get(),
                    joint_delay=self.pwm_joint_delay.get(), codebook=self.pwm_pulse_codebook.get())
            filename = filedialog.asksaveasfilename(parent=self.window, title='保存发收端共用的帧配置', defaultextension='.json', filetypes=[('帧配置', '*.json')])
            if filename:
                from pathlib import Path
                Path(filename).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        except (OSError, ValueError) as exc:
            messagebox.showerror('保存失败', str(exc), parent=self.window)

    def load_frame_config(self):
        filename = filedialog.askopenfilename(parent=self.window, title='载入帧配置', filetypes=[('帧配置', '*.json')])
        if not filename:
            return
        from pathlib import Path
        try:
            config = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
            mapping = dict(user_count='pwm_frame_user_count', startup_symbols='pwm_frame_startup_count',
                startup_pattern='pwm_frame_startup_pattern', training_symbols='pwm_frame_training_count',
                training_pattern='pwm_frame_training_pattern', search_symbols='pwm_frame_search',
                identity_errors='pwm_frame_identity_errors', training_tolerance='pwm_frame_training_tolerance',
                reference_enabled='pwm_frame_reference_enabled', reference_format='pwm_frame_reference_format',
                reference_repeat='pwm_frame_reference_repeat', reference_reset='pwm_frame_reference_reset',
                clock_width='pwm2_clock_width',
                symbol_khz='pwm_symbol_khz', duty_start='pwm_duty_start', duty_step='pwm_duty_step', duty_count='pwm_duty_count')
            from rx706_pwm_protocol import duty_protocol
            from rx706_pwm_frames import validate_frame_options
            if config.get('mode') == '脉冲宽度与时延':
                from rx706_pwm_pulse import validate_pulse_options
                if config.get('schema_version', 2) < 3 and 'tail_us' in config and not config.get('codebook'):
                    # Preserve the old sender's automatic codebook while removing
                    # its fixed receive-tail assumption.
                    import math
                    config['low_guard_chips'] = int(config.get('low_guard_chips', 1))+math.ceil(
                        float(config['tail_us'])*1e-6*float(config['symbol_khz'])*1000*100/float(config['duty_step'])-1e-9)
                frames = dict(config, reference_format='bits' if config['reference_format'] == '二进制比特' else 'symbols')
                validate_pulse_options(dict(config, step_percent=config['duty_step'], symbol_count=config['duty_count'],
                                           frame_options=frames), float(config['symbol_khz'])*1000)
                mapping.update(min_high_chips='pwm_min_high_chips', low_guard_chips='pwm_low_guard_chips',
                    joint_delay='pwm_joint_delay',
                    codebook='pwm_pulse_codebook')
                self.pwm_frame_users = config['users']
                self.pwm2_input_mode.set('脉冲宽度与时延')
                for key, attribute in mapping.items():
                    if key in config:
                        getattr(self, attribute).set(config[key])
                self.pwm_frame_enabled.set(True)
                self.constellation_mode.set(PWM2_MODE)
                self.refresh_parameter_visibility()
                self.update_duty_summary()
                return
            widths = duty_protocol(config['duty_start'], config['duty_step'], config['duty_count'])[0]['symbol_widths']
            validate_frame_options(dict(config, reference_format='bits' if config['reference_format'] == '二进制比特' else 'symbols'), widths)
            self.pwm_frame_users = config['users']
            for key, attribute in mapping.items():
                getattr(self, attribute).set(config[key])
            self.pwm_frame_enabled.set(True)
            self.pwm2_input_mode.set('占空比范围')
            self.constellation_mode.set(PWM2_MODE)
            self.refresh_parameter_visibility()
            self.update_duty_summary()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror('载入失败', str(exc), parent=self.window)

    def pwm2_protocol_options(self):
        if self.pwm2_input_mode.get() == '脉冲宽度与时延':
            return dict(kind='pwm2', pulse_options=self.pulse_config())
        from rx706_pwm import pwm2_layout
        from rx706_pwm_protocol import parse_widths
        from rx706_pwm_training import validate_training
        simple = self.pwm2_input_mode.get() == '占空比范围'
        if simple:
            from rx706_pwm_protocol import duty_protocol
            compiled, _ = duty_protocol(self.pwm_duty_start.get(), self.pwm_duty_step.get(), self.pwm_duty_count.get())
            total, prefix, guard = (compiled[k] for k in ('total_chips', 'prefix_chips', 'guard_chips'))
            widths = compiled['symbol_widths']
        else:
            total = int(self.pwm2_total_chips.get())
            prefix = int(self.pwm2_prefix_chips.get())
            guard = int(self.pwm2_guard_chips.get())
            widths = (parse_widths(self.pwm2_width_table.get(), total,
                              minimum=prefix, maximum=total-guard)
                      if self.pwm2_mapping.get() == '自定义高片数表' else tuple(range(prefix, total-guard+1)))
        code = pwm2_layout(total, prefix, guard)
        if len(set(widths)) != len(widths):
            raise ValueError('符号映射表中的高片数不能重复')
        result = dict(kind='pwm2', total_chips=total, prefix_chips=prefix, code_chips=code,
                      guard_chips=guard, symbol_widths=widths,
                      output_binary=simple,
                      decision_mode='pattern' if not simple and self.pwm2_decision_mode.get() == '时间片码型' else 'width')
        if self.pwm_frame_enabled.get():
            if not simple:
                raise ValueError('最简帧使用占空比范围输入；时间片协议请取消最简帧')
            config = self.frame_config(widths)
            result['training_options'] = dict(frame_options=config, alphabet_widths=widths)
        elif self.pwm2_training_enabled.get():
            from rx706_pwm_protocol import preamble_duties
            duties = preamble_duties(self.pwm2_preamble_start.get(), self.pwm2_preamble_step.get(),
                                     self.pwm2_preamble_count.get())
            preamble = tuple(v*total/100 for v in duties)
            training = dict(preamble_widths=preamble, alphabet_widths=widths,
                            payload_symbols=len(widths) if self.pwm2_payload_count.get() == '自动' else int(self.pwm2_payload_count.get()),
                            same_tolerance_chips=float(self.pwm2_training_tolerance.get()),
                            auto_scale=self.pwm2_auto_scale.get(), clock_width=self.pwm2_clock_width.get())
            validate_training(total, prefix, code, training)
            result['training_options'] = training
        return result

    def update_pwm2_layout(self):
        from rx706_pwm import pwm2_layout
        from rx706_qam_pwm import pwm_header_layout
        try:
            if self.constellation_mode.get() == PWM2_MODE and self.pwm2_input_mode.get() == '脉冲宽度与时延':
                self.update_duty_summary()
                return
            if self.constellation_mode.get() == PWM2_MODE and self.pwm2_input_mode.get() == '占空比范围':
                from rx706_pwm_protocol import duty_protocol
                import math
                _, duties = duty_protocol(self.pwm_duty_start.get(), self.pwm_duty_step.get(), self.pwm_duty_count.get())
                self.pwm2_layout_summary.set(f'当前按占空比输入：{duties[0]:g}%～{duties[-1]:g}%，'
                    f'{len(duties)}种符号，每符号{math.ceil(math.log2(len(duties)))}位二进制。'
                    '需要自定义高片数映射时切换到时间片协议。')
                return
            total, prefix, guard = (int(v.get()) for v in
                (self.pwm2_total_chips, self.pwm2_prefix_chips, self.pwm2_guard_chips))
            header = self.constellation_mode.get() in QAM_MODES
            code = (pwm_header_layout if header else pwm2_layout)(total, prefix, guard)
            preset = {'20片：8+10+2': (20, 8, 2), '50片：20+25+5': (50, 20, 5),
                      '50片：8+40+2': (50, 8, 2), PWM150_PRESET: (25, 6, 7)}.get(self.pwm2_preset.get())
            if preset is not None and preset != (total, prefix, guard):
                self.pwm2_preset.set('自定义')
            text = (f'合法高片数{prefix}～{total-guard}；默认符号=高片数−{prefix}。'
                    f'{prefix}+{code}+{guard}={total}片；{code+1}态，k=0～{code}。'
                    f'占空比{100*prefix/total:g}%～{100*(total-guard)/total:g}%，步进{100/total:g}%。')
            if header and code == 0:text += ' 固定已知前导，不用于独立PWM v2数据解调。'
            rate = self.tdcm_pwm_khz if self.constellation_mode.get() in TDCM_MODES else self.pwm_symbol_khz
            fs, symbol = float(self.rate.get())*1e6, float(rate.get())*1000
            decimation = 1 if header else int(self.pwm_decimation.get())
            if fs > 0 and symbol > 0 and decimation > 0:
                points = fs/(symbol*total*decimation)
                text += f' 每片{points:.3f}个处理点。'
                minimum = 2 if header else 3
                if points < minimum:
                    text += f'不足{minimum}点，请降低符号率或抽取倍数。'
                elif points < 4:
                    text += '低于建议4点，计时余量较小。'
            self.pwm2_layout_summary.set(text)
        except (ValueError, ZeroDivisionError) as exc:
            self.pwm2_layout_summary.set(f'参数待检查：{exc}')

    def on_qam_order_changed(self, event=None):
        from rx706_qam_profiles import four_corner_training_hex
        order=int(self.qam_order.get());old=getattr(self,'_last_qam_order',256)
        # Only a generated preset may be translated. User training is never
        # guessed or truncated to fit another alphabet.
        if (getattr(self,'_four_corner_profile_order',None)==old and
                self.qam_preamble_format.get()=='HEX字节（00–FF）' and
                self.qam_preamble_symbols.get().upper()==four_corner_training_hex(old,int(self.qam_training_repeats.get()))):
            self.qam_preamble_symbols.set(four_corner_training_hex(order,int(self.qam_training_repeats.get())))
            self._four_corner_profile_order=order
        self._last_qam_order=order;self._next_qam_recipe=None
        if self.constellation_mode.get()==PAIRED_64_MODE and order!=64:self.constellation_mode.set(TDCM_MODE)
        self.status.set(f'已选择{order}QAM；训练、符号率和帧长须与发端一致，修改后重新分析。')
        if self.recovery is not None and getattr(self.recovery,'order',order)!=order:
            self.recovery=None;self.draw_constellation();self.canvas.draw_idle()

    def on_mode_changed(self, event=None):
        if self.closed or self.busy:
            return
        if hasattr(self, '_parameter_groups'):
            self.refresh_parameter_visibility()
            self.update_pwm2_layout()
        if self.constellation_mode.get() in (QAM256_120K_MODE,QAM256_120K_LEGACY_MODE):
            apply = self.apply_qam256_120k_frame_profile if self.constellation_mode.get()==QAM256_120K_MODE else self.apply_qam256_120k_legacy_frame_profile
            if apply() is False:return
            if self.filename.get().strip():self.analyze()
            return
        if self.constellation_mode.get()==PAIRED_64_MODE:
            if self.apply_paired_64_frame_profile() is False:return
            if self.filename.get().strip():self.analyze()
            return
        if self.constellation_mode.get()==FOUR_CORNER_MODE:
            if self.apply_four_corner_training_profile() is False:return
            if self.filename.get().strip():self.analyze()
            return
        expected_kind = MODE_KINDS.get(self.constellation_mode.get())
        wrong_header = (expected_kind == 'qam' and self.recovery is not None and
                        (self.constellation_mode.get() == QAM_PWM_MODE or
                         (self.constellation_mode.get() in TDCM_MODES and self.qam_compensation.acquisition_options()=='pwm')) !=
                        (getattr(self.recovery, 'pwm_preamble', None) is not None))
        wrong_tdcm = (expected_kind == 'qam' and self.recovery is not None and
                      (self.constellation_mode.get() in TDCM_MODES) != (getattr(self.recovery,'tdcm',None) is not None))
        if self.constellation_mode.get() in RECOVERY_MODES and (self.recovery is None or self.recovery.kind != expected_kind or wrong_header or wrong_tdcm):
            if self.constellation_mode.get() in QAM_MODES:
                self.analyze()
                return
            self.status.set('已切换模式；核对当前参数后点击“分析信号”。')
            if self.result is not None:
                self.draw_constellation()
        else:
            self.draw_constellation()

    def draw_recovered(self):
        import numpy as np
        r = self.recovery
        if r.kind == 'qam':
            self.draw_qam()
            return
        pwm = r.kind in ('pwm','pwm2')
        self.iq_axes.clear()
        groups = [(-1,'Transitions','#aaaaaa'), (0,'State A','#2879ad'), (1,'State B','#cf7130')]
        for state, label, color in groups:
            z = r.scatter[r.scatter_states == state]
            self.iq_axes.scatter(z.real,z.imag,s=3,alpha=.2,linewidths=0,color=color,label=label)
        self.iq_axes.plot(r.centers.real,r.centers.imag,'kx',markersize=7)
        title = 'Recovered PWM IQ' if pwm else 'Recovered A/B IQ'
        if pwm and not r.decode_locked:
            title += ' | code NOT locked'
        self.iq_axes.set(title=title,xlabel='I (ADC counts)',ylabel='Q (ADC counts)')
        xlo,xhi = np.percentile(r.scatter.real,[.5,99.5])
        ylo,yhi = np.percentile(r.scatter.imag,[.5,99.5])
        distance = abs(r.centers[1]-r.centers[0])
        xpad,ypad = max((xhi-xlo)*.12,1.),max((yhi-ylo)*.2,distance*.05,1.)
        self.iq_axes.set_xlim(xlo-xpad,xhi+xpad)
        self.iq_axes.set_ylim(ylo-ypad,yhi+ypad)
        outside = np.count_nonzero((r.scatter.real < xlo-xpad)|(r.scatter.real > xhi+xpad)|
                                  (r.scatter.imag < ylo-ypad)|(r.scatter.imag > yhi+ypad))
        self.iq_axes.text(.01,.02,f'{outside}/{len(r.scatter)} samples outside focused view',fontsize=6,transform=self.iq_axes.transAxes)
        self.iq_axes.set_aspect('equal',adjustable='box')
        self.iq_axes.grid(alpha=.25)
        self.iq_axes.legend(fontsize=6,loc='upper center',ncol=3)
        self.wave_axes.clear()
        t = (r.waveform_time_s-r.waveform_time_s[0])*1e6
        if pwm:
            self.wave_axes.plot(t,r.decision_metric,color='#aaaaaa',lw=.6,label='IQ projection')
            self.wave_axes.step(t,r.levels.astype(int),where='post',lw=1,label='Control 0/1')
            plot_label = f'PWM duty ({len(r.symbol_widths)} symbols)' if r.output_binary else r.protocol_label
            self.wave_axes.set(title=plot_label+'\nControl envelope | P = pilot',xlabel='Time from view start (us)',ylabel='Control level',ylim=(-.25,1.7))
            shown = r.frames[(r.frames['start_s'] >= r.level_time_s[0]) & (r.frames['start_s'] <= r.level_time_s[-1])]
            for row in shown:
                at = (row['start_s']-r.level_time_s[0])*1e6
                label = str(row['symbol'])+('!' if row['glitch_count'] else '') if row['valid'] else ('P' if row['symbol'] == -3 else 'idle' if row['symbol'] == -2 else '?')
                self.wave_axes.axvline(at,color='#479160',lw=.5,alpha=.5)
                self.wave_axes.text(at,1.12,label,fontsize=7)
        else:
            z = r.waveform
            for y,label in ((z.real,'I'),(z.imag,'Q'),(abs(z),'Magnitude')):
                self.wave_axes.plot(t,y,lw=.7,label=label)
            self.wave_axes.set(title='Recovered IQ waveform',xlabel='Time from view start (us)',ylabel='ADC counts')
        self.wave_axes.legend(fontsize=7,loc='upper right')
        self.wave_axes.grid(alpha=.25)
        self._plots_changed()

    def browse(self):
        filename = filedialog.askopenfilename(parent=self.window, title="选择下载完成的 IQ bin 文件",
            filetypes=[("原始 IQ", "*.bin"), ("所有文件", "*")])
        if filename:
            self.filename.set(filename)

    def offer_download(self, filename):
        if self.closed:
            return
        if self.busy:
            self.pending_download = str(filename)
            return
        self.filename.set(str(filename))
        if self.auto.get():
            self.analyze()

    def _buttons(self):
        for widget in self.controls:
            widget.configure(state="disabled" if self.busy else
                             ("readonly" if isinstance(widget, ttk.Combobox) else "normal"))
        self.analyze_button.configure(state="disabled" if self.busy else "normal")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        if not self.busy:
            self.refresh_parameter_visibility()

    def analyze(self):
        if self.closed or self.busy:
            return
        try:
            name = self.filename.get().strip()
            if not name:
                raise ValueError("请选择已下载的 bin 文件")
            options = dict(channel=1 if self.channel.get() == "RX1" else 2,
                nfft=int(self.nfft.get()), averages=int(self.averages.get()),
                first_tuple=int(self.first_tuple.get()), sample_rate_hz=float(self.rate.get()) * 1e6,
                remove_dc=self.remove_dc.get())
            recovery_options = None
            bound = getattr(self,'_next_iq_bound_recipe',None)
            replay = getattr(self,'_next_qam_recipe',None)
            if bound is not None:
                self._next_iq_bound_recipe = None
                options = dict(bound['spectrum'])
                recovery_options = dict(kind='qam', _iq_bound_recipe=bound, **bound['recovery'])
            elif replay is not None:
                self._next_qam_recipe = None
                recovery_options = dict(kind='qam',tdcm_options=dict(replay_recipe=replay))
            elif self.constellation_mode.get() in RECOVERY_MODES:
                center = self.center_hint.get().strip()
                recovery_options = dict(duration_s=float(self.recovery_ms.get())/1000,
                    center_hint_hz=None if center.lower() in ('自动', 'auto', '') else float(center)*1e6)
                if self.constellation_mode.get() in QAM_MODES:
                    recovery_options.update(kind='qam', order=int(self.qam_order.get()),
                        symbol_hz=float(self.qam_symbol_khz.get())*1000,
                        rolloff=float(self.qam_rolloff.get()), pulse=self.qam_pulse.get(),
                        center_hint_hz=float(self.qam_carrier_mhz.get())*1e6)
                    if self.constellation_mode.get() in TDCM_MODES:
                        recovery_options.update(symbol_hz=float(self.tdcm_symbol_khz.get())*1000,
                            center_hint_hz=float(self.tdcm_carrier_mhz.get())*1e6)
                    if self.qam_preamble_enabled.get():
                        from rx706_qam_preamble import validate_preamble
                        preamble = dict(symbols=self.qam_preamble_symbols.get(),
                            representation='iq' if self.qam_preamble_format.get() == 'I/Q电平' else 'hex' if self.qam_preamble_format.get()=='HEX字节（00–FF）' else 'labels',
                            max_frequency_hz=float(self.qam_preamble_frequency_khz.get())*1000,
                            max_evm_percent=float(self.qam_preamble_evm.get()),
                            search_s=float(self.qam_preamble_search_ms.get())/1000,
                            orientation={'不共轭（原流程）':'normal','自动辨识I/Q镜像':'auto','固定共轭':'conjugate'}[self.qam_training_orientation.get()],
                            fit_dc_offset=self.qam_training_offset.get())
                        validate_preamble(preamble, recovery_options['order'], recovery_options['symbol_hz'])
                        if self.constellation_mode.get() not in TDCM_MODES and (preamble['orientation']!='normal' or preamble['fit_dc_offset']):
                            raise ValueError('镜像/零点训练校准用于TDCM谐波QAM；普通QAM请使用原训练方式')
                        recovery_options['preamble_options'] = preamble
                    if (self.constellation_mode.get() == QAM_PWM_MODE or
                            (self.constellation_mode.get() in TDCM_MODES and self.qam_compensation.acquisition_options()=='pwm')):
                        from rx706_qam_pwm import validate_pwm_header
                        pwm = dict(symbols=self.qam_pwm_symbols.get(),
                            total_chips=int(self.pwm2_total_chips.get()),
                            prefix_chips=int(self.pwm2_prefix_chips.get()),
                            guard_chips=int(self.pwm2_guard_chips.get()),
                            symbol_hz=float(self.tdcm_pwm_khz.get() if self.constellation_mode.get() in TDCM_MODES else self.pwm_symbol_khz.get())*1000,
                            gap_s=float(self.qam_pwm_gap_us.get())/1e6,
                            search_s=float(self.qam_pwm_search_ms.get())/1000,
                            max_error_percent=float(self.qam_pwm_error.get()),
                            width_offset_chips=float(self.qam_pwm_width_offset.get()),
                            max_frequency_hz=float(self.qam_pwm_frequency_khz.get())*1000,
                            qam_search_symbols=float(self.qam_pwm_qam_window.get()))
                        validate_pwm_header(pwm, options['sample_rate_hz'])
                        recovery_options['pwm_preamble_options'] = pwm
                    if self.constellation_mode.get() in TDCM_MODES:
                        from rx706_tdcm import validate_tdcm
                        if self.constellation_mode.get() in (QAM256_120K_MODE,QAM256_120K_LEGACY_MODE):
                            if int(self.qam_order.get())!=256:raise ValueError('120k／64片预设须为256QAM；其他阶数使用通用训练模式')
                        if self.constellation_mode.get()==QAM256_120K_MODE:
                            if int(self.qam_extra_tail_symbols.get())!=0 or self.qam_crc_enabled.get():
                                raise ValueError('当前3+4+16明文帧无CRC和额外尾部，请关闭CRC调试并将尾符号数设为0')
                        if self.constellation_mode.get()==PAIRED_64_MODE:
                            if int(self.qam_order.get())!=64:raise ValueError('当前成对帧预设须为64QAM')
                            if self.four_corner_payload.get().strip()!='128':raise ValueError('当前成对帧固定128物理净荷；其他长度使用通用训练模式')
                            self.qam_compensation.values['frame_symbols'].set(str(136+int(self.qam_extra_tail_symbols.get())))
                        data_symbols=int(self.four_corner_payload.get()) if self.four_corner_payload.get().strip() else 0
                        tdcm = dict(payload_repeat=int(self.qam_payload_repeat.get()),payload_data_symbols=data_symbols,
                            payload_tail_symbols=int(self.qam_extra_tail_symbols.get()),crc_debug=dict(enabled=self.qam_crc_enabled.get()),acquisition_mode=self.qam_compensation.acquisition_options(),base_hz=float(self.tdcm_base_khz.get())*1000,
                            harmonic=int(self.tdcm_harmonic.get()),
                            normalization='pwm' if self.tdcm_normalization.get() == 'PWM两态参考' else 'blind',
                            amplitude_scale=float(self.tdcm_amplitude.get()),phase_deg=float(self.tdcm_phase.get()),
                            payload_symbols=int(self.tdcm_count.get()))
                        validate_tdcm(options['sample_rate_hz'],recovery_options['symbol_hz'],tdcm)
                        if self.qam_compensation.enabled.get():
                            tdcm['compensation'] = self.qam_compensation.options()
                            if self.qam_compensation.search_enabled.get():
                                tdcm['auto_search'] = self.qam_compensation.search_options()
                                tdcm['payload_symbols'] = 0  # Auto frame candidates supersede the manual count.
                        recovery_options['tdcm_options'] = tdcm
                elif self.constellation_mode.get() in PWM_MODES:
                    recovery_options.update(kind='pwm', symbol_hz=float(self.pwm_symbol_khz.get())*1000,
                        code_chips=10 if self.constellation_mode.get() == PWM2_MODE else int(self.pwm_code_chips.get()),
                        guard_chips=2 if self.constellation_mode.get() == PWM2_MODE else int(self.pwm_guard_chips.get()),
                        state_one={'A=1': 'A', 'B=1': 'B'}.get(self.pwm_polarity.get(), 'auto'),
                        decimation=int(self.pwm_decimation.get()), width_tolerance=float(self.pwm_tolerance.get()),
                        hysteresis=float(self.pwm_hysteresis.get()),
                        glitch_tolerance_chips=float(self.pwm_glitch_chips.get()))
                    recovery_options['auto_search'] = self.pwm_auto_search.get()
                    if self.constellation_mode.get() == PWM2_MODE:
                        recovery_options.update(self.pwm2_protocol_options())
                    if self.pwm_reference_enabled.get() and not (self.constellation_mode.get() == PWM2_MODE and self.pwm_frame_enabled.get()):
                        from rx706_pwm_reference import parse_reference
                        sync = int(self.pwm_sync_symbols.get())
                        if not 8 <= sync <= 4096:
                            raise ValueError('同步窗须为 8～4096 个符号')
                        recovery_options['reference_options'] = dict(
                            sequence=parse_reference(self.pwm_reference.get(), len(recovery_options['symbol_widths']) if self.constellation_mode.get() == PWM2_MODE else int(self.pwm_code_chips.get())),
                            sync_symbols=sync)
                else:
                    recovery_options['cycle_hz'] = float(self.cycle_khz.get())*1000
        except ValueError as exc:
            self.status.set(f"参数错误：{exc}")
            return
        dialog = getattr(self, '_pwm_dialog', None)
        if dialog is not None and dialog.winfo_exists():
            dialog.title('PWM 上次分析结果 · 正在重新分析')
            self._open_pwm_after_analysis = True
        self.result = None
        self.recovery = None
        self._last_pwm_recipe = None
        self._last_iq_request = (name, dict(options), dict(recovery_options or {}))
        self._empty_axes()
        self.axes.set_title("Analyzing local IQ file...")
        self.toolbar.update()
        self.canvas.draw()
        self.busy = True
        self.cancel.clear()
        self.status.set("正在分块读取和分析本地文件…")
        self.summary.set("")
        self._buttons()
        self.thread = threading.Thread(target=self._analyze_worker, args=(name, options, recovery_options),
                                       daemon=True, name="rx706-fft")
        self.thread.start()

    def _analyze_worker(self, name, options, recovery_options=None):
        try:
            bound = (recovery_options or {}).get('_iq_bound_recipe')
            if bound is not None:
                from rx706_iq_replay import verify_input
                verify_input(bound)
            from pathlib import Path
            source_before=Path(name).stat()
            result = analyze_bin(name, cancel_event=self.cancel, **options)
            if bound is not None and result.sample_rate_hz != bound['spectrum']['sample_rate_hz']:
                raise ValueError('同名采集元数据的采样率与绑定方案不一致')
            recovery, recovery_error = None, None
            if recovery_options is not None:
                from rx706_recovery import recover_periodic
                try:
                    recovery_options = dict(recovery_options)
                    recovery_options.pop('_iq_bound_recipe',None)
                    kind = recovery_options.pop('kind','periodic')
                    if kind == 'qam':
                        from rx706_qam import recover_qam
                        tdcm_options = dict(recovery_options.get('tdcm_options') or {})
                        if tdcm_options.get('auto_search',{}).get('enabled') or tdcm_options.get('acquisition_mode','pwm')!='pwm':
                            tdcm_options['_progress_callback'] = lambda message: self.events.put(('progress',message))
                            recovery_options['tdcm_options'] = tdcm_options
                        recovery = recover_qam(result, cancel_event=self.cancel, **recovery_options)
                    elif kind in ('pwm', 'pwm2'):
                        from rx706_pwm import recover_pwm, recover_pwm2
                        reference_options = recovery_options.pop('reference_options', None)
                        recovery_options.setdefault('allow_unlocked', True)
                        recovery = (recover_pwm2 if kind == 'pwm2' else recover_pwm)(result, cancel_event=self.cancel, **recovery_options)
                        if reference_options is not None and recovery.decode_locked:
                            from rx706_pwm_reference import check_reference
                            recovery.reference_check = check_reference(recovery.frames,
                                bits_per_symbol=recovery.bits_per_symbol,
                                symbol_bitwords=recovery.symbol_bitwords, **reference_options)
                    else:
                        recovery = recover_periodic(result, cancel_event=self.cancel, **recovery_options)
                except ValueError as exc:
                    recovery_error = str(exc)
                    if getattr(exc,'diagnostics',None):
                        result.auto_search_failure = exc.diagnostics
            if bound is not None and recovery is not None and getattr(recovery,'diagnostics',None):
                recovery.diagnostics.setdefault('warnings',[]).append(bound.get('caveat','训练及映射须与发端确认；未测BER。'))
            source_after=Path(name).stat()
            if (source_before.st_size,source_before.st_mtime_ns)!=(source_after.st_size,source_after.st_mtime_ns):
                raise ValueError('IQ文件在分析期间发生变化，结果已丢弃')
            result._source_stat=(source_after.st_size,source_after.st_mtime_ns)
            if bound is not None:
                verify_input(bound)
            # Prepare a bounded plot trace in the worker as well.
            self.events.put(("result", (result, display_trace(result), recovery, recovery_error)))
        except AnalysisCancelled as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def poll(self):
        if self.closed:
            return
        try:
            kind, value = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == 'progress':
                self.status.set(value)
                self.poll_after = self.window.after(50,self.poll)
                return
            if kind in ('pwm_recipe_saved', 'pwm_recipe_save_error', 'pwm_recipe_save_cancelled'):
                self.busy = False
                self._buttons()
                self.status.set(('已保存PWM IQ绑定方案：' if kind == 'pwm_recipe_saved' else 'PWM方案保存未完成：') + value)
                self.poll_after = self.window.after(50, self.poll)
                return
            if kind == 'pwm_recipe_result':
                recipe, value = value
                self.apply_pwm_recipe_parameters(recipe)
                self._last_pwm_recipe = recipe
                self._last_iq_request = (recipe['input_path'], dict(recipe['spectrum']),
                                         dict(kind='pwm', **recipe['recovery']))
                kind = 'result'
            self.busy = False
            self._buttons()
            self._empty_axes()
            self.toolbar.update()
            if kind == "result":
                result, (freq, power), self.recovery, recovery_error = value
                self.result = result
                self.rate.set(f"{result.sample_rate_hz / 1e6:g}")
                self.axes.plot(freq, power, color="#2467a4", linewidth=0.8)
                self.axes.set_xlim(-result.sample_rate_hz / 2e6, result.sample_rate_hz / 2e6)
                self.axes.set_ylim(max(-180, float(result.dbfs.max()) - 110), max(5, float(result.dbfs.max()) + 5))
                self.axes.set_title(f"RX{result.channel} | Hann | N={result.nfft:,} | averages={result.averages}")
                self.draw_constellation()
                if self.recovery is not None:
                    tdcm = getattr(self.recovery,'tdcm',None)
                    carrier = self.recovery.carrier_hz if tdcm is None else tdcm['corrected_carrier_hz']
                    self.axes.axvline(carrier/1e6, color='#479160', linestyle='--',
                                     linewidth=1, label='RF carrier reference' if tdcm is not None else 'Recovered carrier center')
                    if tdcm is not None:
                        self.axes.axvline(tdcm['target_hz']/1e6,color='#cf8518',linestyle=':',
                                         linewidth=1,label=f"TDCM {tdcm['harmonic']:+d} harmonic")
                    self.axes.legend(fontsize=7)
                if result.peak_hz is not None:
                    self.axes.plot(result.peak_hz / 1e6, result.peak_dbfs, "o", color="#d15d32", markersize=5)
                    peak = f"主峰 {result.peak_hz / 1e6:+.6f} MHz / {result.peak_dbfs:.2f} dBFS"
                else:
                    peak = "所分析片段没有可显示的非零信号"
                rf = (f"；配置 LO + 主峰 ≈ {(result.lo_hz + result.peak_hz)/1e6:.6f} MHz"
                      if result.lo_hz is not None and result.peak_hz is not None else "；RF 绝对频率未知（无 LO 元数据）")
                occupied = (f"；99% 频带 {result.occupied_99_hz[0]/1e6:+.4f}～{result.occupied_99_hz[1]/1e6:+.4f} MHz"
                            if result.occupied_99_hz is not None else "")
                peak_list = " / ".join(f"{f/1e6:+.4f} MHz ({p:.1f} dBFS/bin)" for f,p in result.peaks)
                papr = f"{result.papr_db:.2f} dB" if result.papr_db is not None else "无信号"
                self.status.set(f"RX{result.channel}：{peak}{rf}\n"
                    f"总功率 {result.power_dbfs:.2f} dBFS；分段去直流功率 {result.ac_power_dbfs:.2f} dBFS；"
                    f"RMS {result.rms_counts:.2f} counts；PAPR {papr}；"
                    f"削顶 {result.clipped:,}/{result.samples:,} ({result.clipped/result.samples:.3%}){occupied}\n"
                    f"局部频谱峰：{peak_list or '无'}")
                self.summary.set(f"Fs={result.sample_rate_hz / 1e6:g} MSPS（{result.rate_source}）；"
                    f"Δf={result.sample_rate_hz / result.nfft:.3f} Hz；"
                    f"分析点 [{result.first_tuple:,}, {result.first_tuple + result.samples:,}) / 共 {result.total_tuples:,} 点；"
                    f"时长 {result.samples / result.sample_rate_hz:.6g} s；去直流={'是' if result.remove_dc else '否'}。"
                    "功率参考为 |I+jQ|²/2048²，非 dBm；频带包含噪声和干扰。"
                    "原始星座为有界抽样的IQ散点；请选择解调模式恢复符号。")
                if self.recovery is not None:
                    r = self.recovery
                    if r.kind == 'qam':
                        self.status.set(self.status.get()+f'\n{r.order}QAM：{len(r.scatter)}符号；判决EVM {r.evm_percent:.2f}%；未测BER')
                    elif r.kind in ('pwm', 'pwm2'):
                        good = int(r.frames['valid'].sum())
                        candidates = int((r.frames['candidate_symbol'] >= 0).sum())
                        idle = int((r.frames['symbol'] == -2).sum())
                        training_count = int(r.frames['training'].sum()) if r.training_calibration is not None else 0
                        warned = int((r.frames['valid'] & (r.frames['glitch_count'] > 0)).sum())
                        protocol_status = r.protocol_label if r.version == 2 else f'PWM：{r.code_chips}+{r.guard_chips}片'
                        self.status.set(self.status.get()+f"\n{protocol_status}；"
                            f"符号率 {r.measured_symbol_rate_hz/1000:.6f} kHz；有效 {good}（毛刺告警{warned}），空闲 {idle}，"
                            f"无效 {len(r.frames)-good-idle-training_count}；训练 {training_count}；候选符号 {candidates}；两态分离度 {r.separation:.1f}；{r.mapping_note}")
                        if not r.decode_locked:
                            self.status.set(self.status.get()+'\nPWM编码未锁定（两态已恢复）：'+r.decode_note)
                    else:
                        self.status.set(self.status.get() +
                        f"\n周期恢复：载波中心 {r.carrier_hz/1e6:+.6f} MHz；完整周期 {r.cycle_hz/1000:.4f} kHz；"
                        f"相位差 {abs(r.phase_difference_deg):.2f}°；幅度差(B/A) {r.amplitude_difference_db:+.2f} dB；"
                        f"两簇距离/簇内 RMS {r.separation:.1f}")
                    self.summary.set(f"原始频谱范围 {result.first_tuple/result.sample_rate_hz:.6g}～"
                        f"{(result.first_tuple+result.samples)/result.sample_rate_hz:.6g} s；"
                        f"恢复范围 {r.first_tuple/result.sample_rate_hz:.6g}～{(r.first_tuple+r.input_samples)/result.sample_rate_hz:.6g} s；"
                        f"恢复片段削顶 {r.clipped}/{r.input_samples}；低通截止 {r.cutoff_hz/1000:.1f} kHz；"
                        f"抽取 {r.decimation} 倍，输出 {r.output_rate_hz/1e6:g} MS/s。"
                        + ("PWM按两态IQ距离恢复电平。首尾不完整符号不解码，内部无效符号保留?。"
                           if r.kind in ('pwm','pwm2') else
                           "周期模式恢复相对A/B状态，未确认绝对0/1。") + "功率非dBm。")
                    if r.kind == 'qam':
                        preview = ' '.join(f'{int(v):02X}' for v in r.labels[:24])
                        self.summary.set(f'{r.order}QAM；符号率 {r.symbol_hz/1000:g} kSym/s；载波频偏 {r.carrier_hz/1e6:g} MHz。\n静态相位和定时估计；未跟踪时钟漂移或残余频偏；盲接收存在90°旋转歧义。\n判决EVM不能替代参考EVM或证明锁定。Gray映射：I后Q，各轴从负到正，MSB优先，须与发端一致。\n候选HEX（未测净荷BER）：{preview}')
                        sync = getattr(r, 'preamble', None)
                        if sync is not None:
                            self.status.set(self.status.get()+f"\n前导候选 {sync['length']}符号；位置 {sync['time_s']*1e6:.3f} µs；残余频偏 {sync['frequency_hz']:+.3f} Hz；前导EVM {sync['evm_percent']:.2f}%")
                            self.summary.set(f"{r.order}QAM；已按配置前导进行定时、固定频偏与相位校正。\n"
                                f"前导采样中心 {sync['time_s']:.9f} s；净荷首符号采样中心 {sync['payload_time_s']:.9f} s。\n"
                                f"前导 {sync['length']}符号；匹配EVM {sync['evm_percent']:.2f}%；残余频偏 {sync['frequency_hz']:+.3f} Hz；"
                                f"总校正频偏 {sync['total_frequency_hz']:+.3f} Hz；初始相位 {sync['phase_deg']:+.2f}°。\n"
                                "前导用于消除90°相位歧义；短前导可能误匹配，结果须与发端核对。"
                                "未跟踪时钟漂移、时变相位或频偏；未自动剔除后续前导。\n"
                                "Gray映射：I后Q，各轴从负到正，MSB优先；前导已排除，未测BER。"
                                f"判决EVM不能证明锁定。\n候选净荷比特：{preview}")
                        pwm_sync = getattr(r, 'pwm_preamble', None)
                        if pwm_sync is not None:
                            if sync is None:
                                self.summary.set(f'{r.order}QAM；PWM前导定位后进行盲定时和静态相位估计。\n'
                                    '未使用QAM训练符号，90°相位歧义未消除；PWM两态不能确定QAM的绝对相位。\n'
                                    '仅校正PWM估计的固定粗频偏；未持续跟踪残余频偏、时变相位或时钟漂移。\n'
                                    'Gray映射：I后Q，各轴从负到正，MSB优先；判决EVM不能证明锁定。\n'
                                    f'候选数据（未测BER）：{preview}')
                            pwm_note = (f"PWM前导候选 {len(pwm_sync['symbols'])}符号；"
                                f"起点 {pwm_sync['time_s']*1e6:.3f} µs；完整前导结束 {pwm_sync['end_time_s']*1e6:.3f} µs；"
                                f"QAM边界 {pwm_sync['qam_time_s']*1e6:.3f} µs。\n"
                                f"PWM两态拟合误差 {pwm_sync['error_percent']:.2f}%；"
                                f"PWM粗频偏校正 {pwm_sync['frequency_hz']:+.3f} Hz；间隔 {pwm_sync['gap_s']*1e6:g} µs。")
                            self.status.set(self.status.get()+'\n'+pwm_note)
                            self.summary.set(pwm_note+'\n'+self.summary.get()+
                                '\nPWM及切换间隔不输出为QAM数据；未自动分割后续帧。')
                        tdcm = getattr(r,'tdcm',None)
                        if tdcm is not None and pwm_sync is not None:
                            self.status.set(self.status.get()+f"\nTDCM {tdcm['harmonic']:+d}阶；每符号 {1e6/r.symbol_hz:g} µs；目标频点 {tdcm['target_hz']/1000:+.6f} kHz")
                            self.summary.set(f"TDCM {tdcm['harmonic']:+d}阶 → {r.order}QAM；基频 {tdcm['base_hz']/1000:g} kHz；"
                                f"符号率 {r.symbol_hz/1000:g} kSym/s。\n原载波设置 {r.carrier_hz/1000:+.6f} kHz；"
                                f"校正后 {tdcm['corrected_carrier_hz']/1000:+.6f} kHz；目标谐波 {tdcm['target_hz']/1000:+.6f} kHz。\n"
                                f"PWM前导候选 {pwm_sync['time_s']:.9f} s；QAM边界 {pwm_sync['qam_time_s']:.9f} s；"
                                f"稳定片段拟合误差 {pwm_sync['error_percent']:.2f}%。\n"
                                f"每 {1e6/r.symbol_hz:g} µs完整窗口提取复系数；每符号{tdcm['cycles_per_symbol']}个调制周期；首尾不完整窗口不输出。\n"
                                f"校准：{tdcm['calibration_note']}；幅度系数 {tdcm['amplitude_scale']:g}；相位修正 {tdcm['phase_deg']:g}°。\n"
                                "PWM两态校准须符合发端的幅度/时移约定，不能据此保证映射正确；后续重复前导不自动剔除。"
                                "未持续跟踪载波/时钟漂移；判决EVM不能证明锁定，未测BER。\n"
                                f"Gray映射：I后Q，各轴从负到正，MSB优先。候选HEX：{preview}")
                    if r.kind in ('pwm', 'pwm2'):
                        nominal = r.symbol_rate_hz*r.bits_per_symbol
                        preview = ' | '.join(r.frames['bits' if r.decode_locked else 'candidate_bits'][:32])
                        preview_label = '分组比特流预览' if r.decode_locked else '候选分组比特流预览（允许错误，不用于BER）'
                        self.summary.set(self.summary.get()+f"\n每符号 {r.bits_per_symbol} bit；时间片速率 "
                            f"{r.symbol_rate_hz*(r.total_chips)/1e6:.6g} MHz；"
                            f"每片 {r.output_rate_hz/(r.symbol_rate_hz*(r.total_chips)):.3f} 个处理点。"
                            f"\n理论编码速率 {nominal/1e6:.6g} Mbps；片段有效判决比特率 {r.decoded_bit_rate/1e6:.6g} Mbps（非净荷吞吐/BER）。"
                            f"\n单次毛刺上限 {r.glitch_tolerance_chips:.3g} 片，累计上限0.30片。"
                            "可判决的短毛刺单独告警；误码以参考序列比对为准。波形保留原始电平跳变。"
                            f"\n{preview_label}：{preview}\n点击“PWM 符号表 / 比特流…”查看 T1/T0 和导出。")
                        reference_summary = (r.reference_check.summary() if r.reference_check is not None else
                            '未测 BER；如发端循环发送7642，请勾选“启用已知发端符号循环”后重新分析。')
                        self.summary.set(self.summary.get()+'\n'+reference_summary)
                        self.summary.set(self.summary.get()+f'\n当前每片 {r.output_rate_hz/(r.symbol_rate_hz*(r.total_chips)):.3f} 个IQ点。'
                            +(r.sampling_note or ''))
                        if r.calibration_start_s > r.first_tuple/result.sample_rate_hz:
                            self.summary.set(self.summary.get()+f'\n首窗未能稳定分出两态，载波初始化改用文件 {r.calibration_start_s*1e3:.3f} ms 处窗口；'
                                '原始起始段仍保留并参与波形/编码检查，没有删样或补位。')
                        if not r.decode_locked:
                            self.summary.set(self.summary.get()+'\n'+r.decode_note+
                                '候选符号按最近脉宽估计并保留失败原因；已确认信息仍为?，候选不据此计算BER。')
                        if r.state_fit_note:
                            self.summary.set(self.summary.get()+'\n'+r.state_fit_note)
                        if r.search_note:
                            self.summary.set(self.summary.get()+'\n'+r.search_note)
                        if r.version == 2:
                            import math
                            reference_summary = (r.reference_check.summary() if r.reference_check is not None else
                                f'未测BER。可填写已知符号循环（k=0～{r.code_chips}，必须用空格分隔），启用校验。')
                            preview_label = f'中间{r.code_chips}片码型' if r.decode_locked else f'候选中间{r.code_chips}片码型（非确认信息）'
                            states = len(r.symbol_widths) if r.symbol_widths is not None else r.code_chips+1
                            self.summary.set(
                                f'PWM version2：{r.prefix_chips}片固定高 + {r.code_chips}片可调 + {r.guard_chips}片固定低，共{r.total_chips}片。\n'
                                f'判决依据：{"时间片码型（不使用连续脉宽取整）" if r.decision_mode == "pattern" else "连续脉宽"}。\n'
                                f'符号k={"码型连续1片数" if r.decision_mode == "pattern" else "高片数"}−{r.prefix_chips}，范围0～{r.code_chips}；占空比{100*r.prefix_chips/r.total_chips:g}%～{100*(r.prefix_chips+r.code_chips)/r.total_chips:g}%，步进{100/r.total_chips:g}%。\n'
                                f'时间片速率 {r.symbol_rate_hz*r.total_chips/1e6:.6g} MHz；每片 {r.output_rate_hz/(r.symbol_rate_hz*r.total_chips):.3f} 个处理点。\n'
                                f'{states}态信息量上限 log2({states})≈{math.log2(states):.4f} bit/符号；'
                                f'理想信息率上限 {r.symbol_rate_hz*math.log2(states)/1e6:.6g} Mbps（非实测净荷速率）。\n'
                                f'输出的{r.code_chips}片码型受连续1后接0的规则约束，不是同等数量的独立信息比特。'
                                f'全{r.total_chips}片低为空闲；中间{r.code_chips}片全0是有效符号k=0。\n'
                                f'{preview_label}：{preview}\n{reference_summary}\n'
                                f'恢复范围 {r.first_tuple/result.sample_rate_hz:.6f}～'
                                f'{(r.first_tuple+r.input_samples)/result.sample_rate_hz:.6f} s；'
                                f'削顶 {r.clipped}/{r.input_samples}。\n'
                                f'{r.search_note}\n{r.sampling_note}\n{r.state_fit_note}\n{r.decode_note}')
                            if r.training_calibration is not None:
                                calibration = r.training_calibration
                                offsets = calibration['corrections_chips']
                                offset_text = (f'{min(offsets):+.4f}～{max(offsets):+.4f}片' if offsets else '尚无可用补偿')
                                configured = calibration.get('preamble_widths') is not None
                                label = f"前导高片数 {calibration['preamble_widths']}" if configured else '20%三连训练补偿'
                                self.summary.set(self.summary.get()+
                                    f"\n{label}：{calibration['updates']}个前导区段，排除{calibration['training_symbols']}个训练符号；补偿范围{offset_text}。"
                                    '\n原始脉宽不改写。P为前导，不输出净荷；受损前导沿用先前校准。'
                                    f"\n比例范围{calibration.get('scale_range', [])}；{calibration.get('spacing_symbols', 0)}行使用档位标定。"
                                    '\n参考循环只用于独立校验，不用于补偿或纠错；擦除单列。')
                                if r.decision_mode == 'pattern':
                                    self.summary.set(self.summary.get()+'\n当前按原始时间片码型判决；脉宽补偿列仅供对照，不影响码型符号值。')
                            if r.symbol_widths is not None:
                                self.summary.set(self.summary.get()+'\n当前符号映射（符号:高片数）：'+
                                    '  '.join(f'{k}:{w}' for k, w in enumerate(r.symbol_widths)))
                            if r.output_binary:
                                duties = [100*w/r.total_chips for w in r.symbol_widths]
                                calibration_note = ''
                                if r.training_calibration is not None:
                                    cal = r.training_calibration
                                    calibration_note = (f"\n{r.calibration_description}\n排除{cal['training_symbols']}个前导符号。"
                                        f"比例范围{cal['scale_range']}；{cal['spacing_symbols']}行使用档位标定。")
                                self.summary.set(f'PWM占空比解调；实测频率 {r.measured_symbol_rate_hz/1000:.5f} kHz。\n'
                                    f'{states}种符号，符号0～{states-1}，每符号输出{r.bits_per_symbol}位二进制。\n'
                                    '符号→占空比：'+'  '.join(f'{k}→{v:g}%' for k, v in enumerate(duties))+
                                    '\n'+reference_summary+calibration_note+
                                    '\n先前观测用于校准；参考循环不参与纠错。原始数据保留，擦除单列。\n'+
                                    f'{r.search_note}\n{r.sampling_note}\n{r.decode_note}')
                            if r.frame_check is not None:
                                self.summary.set('PWM占空比最简帧；'+r.calibration_description+'\n'+
                                    r.frame_check.summary()+'\n'+r.sampling_note+'\n'+r.decode_note)
                            if r.pulse_options is not None:
                                pulse = r.pulse_options
                                tails = [c['tail_us'] for c in r.pulse_calibrations]
                                rates = '；'.join(f"{u['name']} {rate/1000:.3f} kbit/s" for u, rate in
                                    zip(r.frame_options['users'], pulse['user_information_rates']))
                                self.summary.set(r.calibration_description+'\n'+
                                    f"当前码表{pulse['symbol_count']}种；log₂(M)={pulse['information_bits']:.6f} bit/符号。\n"
                                    f"含前导和时分开销的单用户净荷信息率上限：{rates}。\n"
                                    f"输出{r.bits_per_symbol}位符号编号，仅{pulse['symbol_count']}种有效，不等于同位数独立净荷。\n"
                                    f"成功前导{len(tails)}个，实测拖尾范围{min(tails):.4f}～{max(tails):.4f}µs。\n"+
                                    f"按实测拖尾保守评估，完整落在符号内的码字最少{min(c['conservative_codewords'] for c in r.pulse_calibrations)}种；发送码表编号保持不变。\n"+
                                    r.frame_check.summary()+'\n'+r.sampling_note+'\n'+r.decode_note)
                elif recovery_error:
                    prefix = 'QAM 解调' if self.constellation_mode.get() in QAM_MODES else ('PWM 解调' if self.constellation_mode.get() in PWM_MODES else '周期恢复')
                    self.status.set(self.status.get() + f"\n{prefix}未锁定：" + recovery_error)
                    self.summary.set(self.summary.get() + " 已显示原始 IQ，未强行生成星座簇。")
            else:
                self.axes.set_title("Analysis stopped" if kind == "cancelled" else "Analysis failed")
                self.status.set(("" if kind == "cancelled" else "分析失败：") + value)
            if kind == 'result' and self.recovery is not None and getattr(self.recovery,'diagnostics',None):
                from rx706_qam_debug_gui import summary_text
                self.summary.set(summary_text(self.recovery,self.result))
                self.status.set(self.status.get()+'\n'+self.recovery.diagnostics['quality'])
            if kind == 'result' and self.recovery is not None and getattr(self.recovery, 'replay_warning', None):
                self.summary.set(self.summary.get().replace('；擦除', '；参考覆盖范围擦除') + '\n' + self.recovery.replay_warning)
                from rx706_pwm_replay import payload_diagnostics
                counts = payload_diagnostics(self.recovery)
                self.status.set(self.status.get() + f"\n净荷擦除{counts['erased_payload_symbols']}/{counts['payload_symbols']}符号；"
                                + ('BER未测' if not self._last_pwm_recipe['recovery']['pulse_options']['frame_options']['reference_enabled'] else 'BER仅按已确认参考计分'))
                if not self.recovery.decode_locked or not self.recovery.frames['valid'].any():
                    self.status.set(self.status.get() + '\nPWM未锁定或无有效净荷；不能据此报告成功解码或BER。')
            self.canvas.draw()
            if (kind == 'result' and self.recovery is not None and getattr(self.recovery,'diagnostics',None)
                    and hasattr(self,'qam_compensation') and self.qam_compensation.auto_open.get()):
                self.show_qam_report()
            elif (kind == 'result' and getattr(self.result,'auto_search_failure',None)
                    and hasattr(self,'qam_compensation') and self.qam_compensation.auto_open.get()):
                self.show_qam_report()
            if getattr(self, '_open_pwm_after_analysis', False):
                self._open_pwm_after_analysis = False
                if kind == 'result' and self.recovery is not None and self.recovery.kind in ('pwm', 'pwm2'):
                    self.show_pwm_symbols()
                elif kind != 'cancelled':
                    detail = str(recovery_error) if kind == 'result' and recovery_error else self.status.get()
                    slots = f'{self.pwm2_total_chips.get()}片/前高{self.pwm2_prefix_chips.get()}/后低{self.pwm2_guard_chips.get()}' if self.constellation_mode.get() == PWM2_MODE else (
                        f'{self.pwm_code_chips.get()}+{self.pwm_guard_chips.get()}') if hasattr(self, 'pwm_code_chips') else ''
                    config = (f'\n\n本次设置：{self.constellation_mode.get()} {self.pwm_symbol_khz.get()} kHz，'
                              f'{slots}片，'
                              f'{self.pwm_polarity.get()}；抽取{self.pwm_decimation.get()}倍，Fs={self.rate.get()} MS/s。') if hasattr(self, 'pwm_symbol_khz') else ''
                    advice = ('\n\n实际恢复片段触限过多：请降低输入电平或接收增益后重新采集；修改PWM参数不能还原已削顶的采样。'
                              if '恢复片段削顶' in detail else '\n\n请核对实际 PWM 符号率、保护片数和 A/B 对应关系后重试。')
                    messagebox.showwarning('PWM 解调未完成', detail+config+advice, parent=self.window)
            if self.pending_download is not None:
                filename, self.pending_download = self.pending_download, None
                self.offer_download(filename)
        self.poll_after = self.window.after(50, self.poll)

    def apply_qam_debug_profile(self, panel):
        """Explicit user preset; estimates the clock rather than hard-coding this file."""
        order=int(self.qam_order.get())
        self.qam_payload_repeat.set('1');self.qam_training_repeats.set('1')
        panel.repeated_profile(frame_symbols=panel.values['frame_symbols'].get() if order!=256 else None)
        self.constellation_mode.set(TDCM_MODE)
        if order!=256:
            panel.search_values['frame_candidates'].set(panel.values['frame_symbols'].get())
            self.qam_preamble_enabled.set(False);self.tdcm_normalization.set('盲归一化')
            self.status.set(f'已为{order}QAM启用重复图样补偿；保留当前速率、载波和窗口数，不推断发端帧长。')
            return
        self.tdcm_symbol_khz.set('19.53125')
        self.tdcm_base_khz.set('19.53125')
        self.tdcm_pwm_khz.set('19.53125')
        self.tdcm_carrier_mhz.set('-0.2')
        self.tdcm_harmonic.set('1')
        self.tdcm_normalization.set('盲归一化')
        self.tdcm_amplitude.set('1'); self.tdcm_phase.set('0'); self.tdcm_count.set('0')
        self.qam_pwm_symbols.set('0 0 0'); self.qam_pwm_gap_us.set('0')
        self.qam_pwm_search_ms.set('1000'); self.qam_pwm_error.set('25')
        self.qam_pwm_width_offset.set('0'); self.qam_pwm_frequency_khz.set('1')
        self.qam_pwm_qam_window.set('8')
        self.pwm2_total_chips.set('25'); self.pwm2_prefix_chips.set('5'); self.pwm2_guard_chips.set('8')
        self.qam_preamble_enabled.set(False)
        self.status.set('已载入重复图样假设：3个PWM + 256个QAM窗口；Fs须核对。点击分析信号。')

    def activate_qam_auto_search(self):
        if self.constellation_mode.get() not in (FOUR_CORNER_MODE,PAIRED_64_MODE) or self.qam_compensation.acquisition_options()!='pwm':
            self.constellation_mode.set(TDCM_MODE)
        self.status.set('TDCM获取方式已设置；先核对Fs、标称速率、获取方式与搜索范围，再点击分析信号。')

    def apply_four_corner_training_profile(self):
        from rx706_qam_preamble import parse_preamble
        from rx706_qam_profiles import four_corner_training_hex,four_corner_payload
        p=self.qam_compensation;order=int(self.qam_order.get())
        try:total=int(p.values['frame_symbols'].get())
        except (ValueError,TypeError):self.status.set('每帧QAM窗口数须为整数');return False
        old=0
        if self.qam_preamble_enabled.get():
            try:old=len(parse_preamble(self.qam_preamble_symbols.get(),'iq' if self.qam_preamble_format.get()=='I/Q电平' else 'hex' if self.qam_preamble_format.get()=='HEX字节（00–FF）' else 'labels',int(self.qam_order.get())))
            except ValueError:old=0
        try:
            payload=four_corner_payload(order,total,old,getattr(self,'four_corner_payload',None).get() if hasattr(self,'four_corner_payload') else '')
        except ValueError as exc:self.status.set(str(exc));return False
        repetitions=int(self.qam_training_repeats.get());corners=four_corner_training_hex(order,repetitions)
        tail=int(self.qam_extra_tail_symbols.get());windows=payload+4*repetitions+tail
        self.constellation_mode.set(FOUR_CORNER_MODE);self._four_corner_profile_order=order;self._last_qam_order=order
        p.acquisition_mode.set('有PWM前导（原流程）');p.enabled.set(True);p.search_enabled.set(False)
        p.clock_mode.set('自动：整帧重复图样');p.phase_mode.set('两态平方载波');p.frame_mode.set('按配置分割全部帧');p.fit_timing.set(False)
        p.values['frame_symbols'].set(str(windows));p.values['max_frames'].set('2048');p.values['clock_search_ppm'].set('1000')
        self.qam_preamble_enabled.set(True);self.qam_preamble_format.set('HEX字节（00–FF）')
        self.qam_preamble_symbols.set(corners);self.qam_preamble_evm.set('10')
        p.search_values['frame_candidates'].set(str(windows))
        self.qam_training_orientation.set('自动辨识I/Q镜像');self.qam_training_offset.set(True)
        self.qam_pwm_symbols.set('0 0 0');self.pwm2_total_chips.set('25');self.pwm2_prefix_chips.set('5');self.pwm2_guard_chips.set('8')
        self.qam_pwm_gap_us.set('0');self.qam_pwm_qam_window.set('0');self.tdcm_normalization.set('盲归一化')
        self._next_qam_recipe=None
        self.status.set(f'已配置{order}QAM：3×20% PWM + {corners}训练 + {payload}个净荷窗口；这是接收假设，实际发端训练/帧长须核对。')
        return True

    def apply_qam256_120k_frame_profile(self):
        """Current 3+4+16 plaintext frame; do not assume repeated payload."""
        self.qam_order.set('256');self.four_corner_payload.set('16')
        self.qam_training_repeats.set('1');self.qam_payload_repeat.set('1')
        self.qam_extra_tail_symbols.set('0');self.qam_crc_enabled.set(False)
        if self.apply_four_corner_training_profile() is False:return False
        self.constellation_mode.set(QAM256_120K_MODE)
        self.qam_compensation.clock_mode.set('关闭（配置周期）')
        self.qam_compensation.phase_mode.set('关闭')
        self.qam_compensation.values['period_ppm'].set('0')
        self.tdcm_symbol_khz.set('120');self.tdcm_base_khz.set('120');self.tdcm_pwm_khz.set('120')
        self.tdcm_harmonic.set('1');self.tdcm_count.set('0')
        self.pwm2_total_chips.set('64');self.pwm2_prefix_chips.set('16');self.pwm2_guard_chips.set('48')
        self.qam_pwm_width_offset.set('0');self.qam_pwm_qam_window.set('0')
        self.update_pwm2_layout()
        self.status.set('256QAM：120 kSym/s，K=1；3×64片25%PWM + 单遍00 08 88 80 + 16物理明文符号。每帧23周期，无CRC，不假定净荷重复；Fs和原载波须核对。')
        return True

    def apply_qam256_120k_legacy_frame_profile(self):
        """Confirmed 3+8+128 frame; preserve actual Fs and original carrier."""
        self.qam_order.set('256');self.four_corner_payload.set('128')
        self.qam_training_repeats.set('2');self.qam_payload_repeat.set('1')
        self.qam_extra_tail_symbols.set('0');self.qam_crc_enabled.set(False)
        if self.apply_four_corner_training_profile() is False:return False
        self.constellation_mode.set(QAM256_120K_LEGACY_MODE)
        self.tdcm_symbol_khz.set('120');self.tdcm_base_khz.set('120');self.tdcm_pwm_khz.set('120')
        self.tdcm_harmonic.set('1');self.tdcm_count.set('0')
        self.pwm2_total_chips.set('64');self.pwm2_prefix_chips.set('16');self.pwm2_guard_chips.set('48')
        self.qam_pwm_width_offset.set('0');self.qam_pwm_qam_window.set('0')
        self.update_pwm2_layout()
        self.status.set('256QAM：120 kSym/s，K=1；3×64片25%PWM + 两遍00 08 88 80 + 128物理净荷。每帧139周期；Fs和原载波仍须按采集核对。')
        return True

    def apply_paired_64_frame_profile(self):
        self.qam_order.set('64');self.four_corner_payload.set('128')
        self.qam_training_repeats.set('2');self.qam_payload_repeat.set('2')
        if self.apply_four_corner_training_profile() is False:return False
        self.constellation_mode.set(PAIRED_64_MODE)
        self.status.set('64QAM当前帧：3PWM + 两遍00 04 24 20 + 128物理净荷（64重复对）；额外尾符号按手动配置。CRC参数未知，不校验。')
        return True

    def load_pwm_recipe(self):
        """Read only small JSON on Tk's thread; hashing/recovery stays in a worker."""
        if self.closed or self.busy:
            return
        filename = filedialog.askopenfilename(parent=self.window, title='载入PWM IQ绑定JSON并分析',
                                              filetypes=[('JSON', '*.json')])
        if not filename:
            return
        try:
            from rx706_pwm_replay import read_recipe
            recipe = read_recipe(filename)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror('无法载入PWM方案', str(exc), parent=self.window)
            return
        # Keep a visible previous result until replacement succeeds. Destroying
        # it here made the PWM window disappear on a replay click (even on failure).
        dialog = getattr(self, '_pwm_dialog', None)
        reopen_pwm = dialog is not None and bool(dialog.winfo_exists())
        if reopen_pwm:
            dialog.title('PWM 上次分析结果 · 正在重新分析')
        self.result = self.recovery = self._last_iq_request = self._last_pwm_recipe = None
        self._next_qam_recipe = self._next_iq_bound_recipe = None
        self._open_pwm_after_analysis = reopen_pwm
        self._empty_axes()
        self.toolbar.update()
        self.canvas.draw()
        self.busy = True
        self.cancel.clear()
        self.status.set('正在后台校验PWM绑定IQ，校验通过后恢复参数并分析…')
        self.summary.set('')
        self._buttons()
        self.thread = threading.Thread(target=self._pwm_recipe_worker, args=(recipe,),
                                       daemon=True, name='rx706-pwm-replay')
        self.thread.start()

    def _pwm_recipe_worker(self, recipe):
        try:
            from rx706_pwm_replay import run_recipe
            result, recovery = run_recipe(recipe, cancel_event=self.cancel,
                progress=lambda text: self.events.put(('progress', text)))
            self.events.put(('pwm_recipe_result', (recipe, (result, display_trace(result), recovery, None))))
        except AnalysisCancelled as exc:
            self.events.put(('cancelled', str(exc)))
        except Exception as exc:
            self.events.put(('error', str(exc)))

    def apply_pwm_recipe_parameters(self, recipe):
        """Display the exact validated worker snapshot, independent of old QAM state."""
        import copy
        sp, rec = recipe['spectrum'], recipe['recovery']
        pulse, frames = rec['pulse_options'], rec['pulse_options']['frame_options']
        self.constellation_mode.set(PWM2_MODE)
        self.pwm2_input_mode.set('脉冲宽度与时延')
        self.pwm_frame_enabled.set(True)
        self.filename.set(recipe['input_path'])
        self.rate.set(str(sp['sample_rate_hz'] / 1e6))
        self.channel.set('RX' + str(sp['channel']))
        self.first_tuple.set(str(sp['first_tuple']))
        self.nfft.set(str(sp['nfft']))
        self.averages.set(str(sp['averages']))
        self.remove_dc.set(sp['remove_dc'])
        self.recovery_ms.set(str(rec['duration_s'] * 1000))
        self.center_hint.set('自动' if rec['center_hint_hz'] is None else str(rec['center_hint_hz'] / 1e6))
        self.pwm_symbol_khz.set(str(rec['symbol_hz'] / 1000))
        for key, variable in (('decimation', self.pwm_decimation), ('width_tolerance', self.pwm_tolerance),
                              ('hysteresis', self.pwm_hysteresis), ('glitch_tolerance_chips', self.pwm_glitch_chips)):
            variable.set(str(rec[key]))
        self.pwm_polarity.set({'A': 'A=1', 'B': 'B=1', 'auto': '自动（PWM约束）'}[rec['state_one']])
        self.pwm_auto_search.set(False)
        self.pwm_reference_enabled.set(False)
        for key, variable in (('step_percent', self.pwm_duty_step), ('symbol_count', self.pwm_duty_count),
                              ('min_high_chips', self.pwm_min_high_chips), ('low_guard_chips', self.pwm_low_guard_chips)):
            variable.set(str(pulse[key]))
        self.pwm_joint_delay.set(pulse['joint_delay'])
        self.pwm_pulse_tolerance.set(str(pulse['tolerance_chips']))
        self.pwm_nearest_decision.set(pulse['nearest_decision'])
        self.pwm_pulse_codebook.set('\n'.join(f'{d:g} {w:g}' for d, w in pulse['codebook']))
        self.pwm_frame_users = copy.deepcopy(frames['users'])
        for key, variable in (('user_count', self.pwm_frame_user_count), ('search_symbols', self.pwm_frame_search),
                              ('identity_errors', self.pwm_frame_identity_errors), ('training_tolerance', self.pwm_frame_training_tolerance)):
            variable.set(str(frames[key]))
        self.pwm_frame_reference_enabled.set(frames['reference_enabled'])
        self.pwm_frame_reference_format.set('二进制比特' if frames['reference_format'] == 'bits' else '符号值')
        self.pwm_frame_reference_repeat.set(frames['reference_repeat'])
        self.pwm_frame_reference_reset.set(frames['reference_reset'])
        self.pwm2_clock_width.set(frames['clock_width'])
        self.refresh_parameter_visibility()
        self.update_duty_summary()

    def save_pwm_recipe(self):
        """Export the last completed PWM analysis, never current edited controls."""
        if self.closed or self.busy:
            return
        import copy
        from pathlib import Path
        request = getattr(self, '_last_iq_request', None)
        if not request or self.result is None or self.recovery is None or not getattr(self.recovery, 'pulse_options', None):
            messagebox.showerror('无法保存', '请先完成一次PWM脉冲宽度与时延分析。', parent=self.window)
            return
        filename = filedialog.asksaveasfilename(parent=self.window, title='保存本次PWM IQ绑定JSON',
            initialfile=Path(request[0]).stem + '_pwm_replay.json', defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not filename:
            return
        snapshot = copy.deepcopy(request)
        snapshot[1]['sample_rate_hz'] = self.result.sample_rate_hz
        source_stat = getattr(self.result, '_source_stat', None)
        prior_recipe = copy.deepcopy(getattr(self, '_last_pwm_recipe', None))
        self.busy = True
        self.cancel.clear()
        self.status.set('正在后台校验并保存本次PWM分析快照…')
        self._buttons()
        self.thread = threading.Thread(target=self._save_pwm_recipe_worker,
            args=(snapshot, source_stat, prior_recipe, filename), daemon=True, name='rx706-pwm-recipe-save')
        self.thread.start()

    def _save_pwm_recipe_worker(self, request, source_stat, prior_recipe, filename):
        try:
            from pathlib import Path
            from rx706_pwm_replay import make_recipe, verify_input
            from rx706_spectrum import _cancel
            name, sp, rec = request
            def unchanged():
                stat = Path(name).stat()
                if (stat.st_size, stat.st_mtime_ns) != source_stat:
                    raise ValueError('IQ文件自分析后已变化，请重新分析后再保存')
            unchanged()
            if prior_recipe is not None:
                recipe = prior_recipe
            else:
                rec = {k: v for k, v in rec.items() if k in ('duration_s', 'symbol_hz', 'center_hint_hz',
                    'decimation', 'state_one', 'width_tolerance', 'hysteresis', 'glitch_tolerance_chips', 'pulse_options')}
                # A manual BER checkbox is not proof the transmitter reference was verified.
                rec['pulse_options']['frame_options']['reference_enabled'] = False
                recipe = make_recipe(name, cancel_event=self.cancel, **sp, **rec)
            verify_input(recipe, self.cancel)
            unchanged()
            _cancel(self.cancel)
            with Path(filename).open('x', encoding='utf-8') as stream:
                json.dump(recipe, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            self.events.put(('pwm_recipe_saved', filename + ('；未核验参考默认关闭' if prior_recipe is None else '')))
        except AnalysisCancelled as exc:
            self.events.put(('pwm_recipe_save_cancelled', str(exc)))
        except Exception as exc:
            self.events.put(('pwm_recipe_save_error', str(exc)))

    def load_qam_recipe(self):
        if self.closed or self.busy:
            return
        import json
        from pathlib import Path
        filename=filedialog.askopenfilename(parent=self.window,title='载入通用参数或同一IQ片段的冻结方案',filetypes=[('JSON','*.json')])
        if not filename:return
        try:
            path=Path(filename)
            if path.stat().st_size>512*1024:raise ValueError('方案JSON超过512KiB')
            recipe=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(recipe,dict):raise ValueError('方案须为JSON对象')
            if recipe.get('method')=='iq_bound_tdcm_replay':
                from rx706_iq_replay import validate_recipe
                recipe=validate_recipe(recipe)
                if not Path(recipe['input_path']).is_file():raise ValueError('绑定的IQ文件不存在：'+recipe['input_path'])
                sp=recipe['spectrum']; rec=recipe['recovery']
                self.filename.set(recipe['input_path']);self.rate.set(str(sp['sample_rate_hz']/1e6))
                self.channel.set('RX1' if sp['channel']==1 else 'RX2');self.first_tuple.set(str(sp['first_tuple']))
                self.nfft.set(str(sp['nfft']));self.averages.set(str(sp['averages']));self.remove_dc.set(sp['remove_dc'])
                self.apply_qam_recipe_parameters(dict(method='iq_bound_tdcm_replay',**rec))
                self.constellation_mode.set(TDCM_MODE);self.qam_compensation.auto_open.set(True)
                self._next_qam_recipe=None;self._next_iq_bound_recipe=recipe
                self.analyze();return
            if recipe.get('method')=='tdcm_parameter_config':
                self.apply_qam_parameter_config(recipe)
                return  # Parameters only: never analyze or replay an IQ file on load.
            if recipe.get('version')!=1 or recipe.get('method') not in ('bounded_repeated_tdcm','headerless_tdcm_config','frozen_central_tdcm_capture'):raise ValueError('方案格式不支持')
            fp=recipe['input_fingerprint']
            self.rate.set(str(float(fp['sample_rate_hz'])/1e6))
            self.channel.set('RX1' if fp['channel']==1 else 'RX2');self.first_tuple.set(str(fp['first_tuple']))
            self.constellation_mode.set(TDCM_MODE)
            self.apply_qam_recipe_parameters(recipe)
            if recipe.get('method')=='frozen_central_tdcm_capture':
                if recipe.get('input_path'):self.filename.set(str(Path(recipe['input_path']).expanduser()))
                self.qam_compensation.auto_open.set(True)
            self._next_qam_recipe=recipe
        except (OSError,ValueError,TypeError,KeyError) as exc:
            messagebox.showerror('无法载入方案',str(exc),parent=self.window);return
        self.analyze()

    def save_iq_bound_recipe(self):
        """Save the exact successful TDCM run, not edited controls after it."""
        if self.closed or self.busy:return
        request=getattr(self,'_last_iq_request',None)
        if not request or self.result is None or self.recovery is None or self.recovery.kind!='qam':
            messagebox.showerror('无法保存','请先成功完成一次有PWM前导的TDCM补偿分析。',parent=self.window);return
        name,sp,rec=request
        if rec.get('_iq_bound_recipe'):
            rec=dict(kind='qam',**rec['_iq_bound_recipe']['recovery'])
        try:
            from rx706_iq_replay import make_recipe
            from pathlib import Path
            cfg=dict(version=1,method='tdcm_parameter_config',order=rec['order'],symbol_hz=rec['symbol_hz'],
                     tdcm_options=rec['tdcm_options'],pwm_preamble_options=rec['pwm_preamble_options'],preamble_options=rec['preamble_options'])
            # Use the actual sample rate (metadata can override the requested fallback).
            source=Path(name).stat()
            if (source.st_size,source.st_mtime_ns)!=getattr(self.result,'_source_stat',None):
                raise ValueError('IQ文件自分析后已变化，请重新分析后再保存')
            recipe=make_recipe(name,cfg,sample_rate_hz=self.result.sample_rate_hz,channel=sp['channel'],
                first_tuple=sp['first_tuple'],duration_s=rec['duration_s'],center_hint_hz=rec['center_hint_hz'],
                nfft=sp['nfft'],averages=sp['averages'],remove_dc=sp['remove_dc'])
            after=Path(name).stat()
            if (after.st_size,after.st_mtime_ns)!=self.result._source_stat:
                raise ValueError('IQ文件自分析后已变化，请重新分析后再保存')
            filename=filedialog.asksaveasfilename(parent=self.window,title='保存IQ绑定复算方案',defaultextension='.json',
                initialfile=Path(name).stem+'_replay.json',filetypes=[('JSON','*.json')])
            if not filename:return
            import json
            with open(filename,'x',encoding='utf-8') as stream:
                json.dump(recipe,stream,ensure_ascii=False,indent=2);stream.write('\n')
            self.status.set('已保存IQ绑定复算方案：'+filename)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            messagebox.showerror('无法保存',str(exc),parent=self.window)

    def apply_qam_parameter_config(self, config):
        """Validate before updating; preserve file/Fs/carrier/channel/start."""
        from rx706_qam_config import validate_parameter_config
        parameters=validate_parameter_config(config)
        self.apply_qam_recipe_parameters(parameters)
        self.constellation_mode.set(TDCM_MODE)
        self._next_qam_recipe=None
        self.recovery=None
        self.update_pwm2_layout()
        self.status.set('通用参数已载入；当前IQ文件、Fs、原载波、通道及起点保持不变。请核对后点击分析信号；没有自动分析。')
        return True

    def apply_qam_recipe_parameters(self, recipe):
        """Show the recipe's actual settings before its one-shot replay."""
        from rx706_qam_debug_gui import ACQUISITION_MODES,CLOCK_MODES,PHASE_MODES,FRAME_MODES
        if self.constellation_mode.get() not in QAM_MODES:
            self.constellation_mode.set(TDCM_MODE)
        args=recipe.get('reproducible_options',{}) if recipe['method']=='headerless_tdcm_config' else recipe
        td=args.get('tdcm_options',{});comp=td.get('compensation',{});p=self.qam_compensation
        def number(variable,value): variable.set(f'{float(value):.12g}')
        if 'duration_s' in args:number(self.recovery_ms,1000*args['duration_s'])
        if 'order' in args:
            self.qam_order.set(str(int(args['order'])));self._last_qam_order=int(args['order'])
        self._four_corner_profile_order=None
        if 'symbol_hz' in args:number(self.tdcm_symbol_khz,args['symbol_hz']/1000)
        if 'center_hint_hz' in args:number(self.tdcm_carrier_mhz,args['center_hint_hz']/1e6)
        for key,var,scale in [('base_hz',self.tdcm_base_khz,1000),('harmonic',self.tdcm_harmonic,1),
                ('amplitude_scale',self.tdcm_amplitude,1),('phase_deg',self.tdcm_phase,1),('payload_symbols',self.tdcm_count,1)]:
            if key in td:number(var,td[key]/scale)
        if 'normalization' in td:self.tdcm_normalization.set('PWM两态参考' if td['normalization']=='pwm' else '盲归一化')
        mode=td.get('acquisition_mode','pwm')
        p.acquisition_mode.set(next(label for label,value in ACQUISITION_MODES.items() if value==mode))
        p.enabled.set(bool(comp.get('enabled',False)))
        p.search_enabled.set(bool(td.get('auto_search',{}).get('enabled',False)))
        for key,var,modes in [('clock_mode',p.clock_mode,CLOCK_MODES),('phase_mode',p.phase_mode,PHASE_MODES),('frame_mode',p.frame_mode,FRAME_MODES)]:
            if key in comp:var.set(next(label for label,value in modes.items() if value==comp[key]))
        for key,var in p.values.items():
            if key in comp:number(var,comp[key])
        p.fit_timing.set(bool(comp.get('fit_timing',False)))
        self.qam_payload_repeat.set(str(td.get('payload_repeat',1)))
        self.qam_extra_tail_symbols.set(str(td.get('payload_tail_symbols',0)))
        self.qam_crc_enabled.set(bool(td.get('crc_debug',{}).get('enabled',False)))
        search=td.get('auto_search',{})
        for key,var in p.search_values.items():
            if key=='carrier_span_khz' and 'carrier_span_hz' in search:number(var,search['carrier_span_hz']/1000)
            elif key in search:
                value=search[key];var.set(' '.join(map(str,value)) if isinstance(value,list) else str(value))
        training=args.get('preamble_options');self.qam_preamble_enabled.set(training is not None)
        if training:
            self.qam_preamble_symbols.set(str(training.get('symbols','')))
            self.qam_preamble_format.set('I/Q电平' if training.get('representation')=='iq' else 'HEX字节（00–FF）' if training.get('representation')=='hex' else '符号值（Gray）')
            self.qam_training_orientation.set({'normal':'不共轭（原流程）','auto':'自动辨识I/Q镜像','conjugate':'固定共轭'}[training.get('orientation','normal')])
            self.qam_training_offset.set(bool(training.get('fit_dc_offset',False)))
            from rx706_qam_preamble import parse_preamble
            from rx706_qam import decide
            expected=parse_preamble(training.get('symbols',''),training.get('representation','labels'),int(args.get('order',256)))
            from rx706_qam_profiles import corner_repetitions
            order=int(args.get('order',256))
            repetitions=corner_repetitions(decide(expected,order)[1].tolist(),order)
            if repetitions:
                self.qam_training_repeats.set(str(repetitions))
                self.constellation_mode.set(PAIRED_64_MODE if order==64 and repetitions==2 and td.get('payload_repeat')==2 and td.get('payload_data_symbols')==128 else FOUR_CORNER_MODE);self._four_corner_profile_order=order
                if hasattr(self,'four_corner_payload'):self.four_corner_payload.set(str(td.get('payload_data_symbols') or int(comp.get('frame_symbols',256))-len(expected)-int(td.get('payload_tail_symbols',0))))
            self._last_qam_order=order
            for key,var,scale in [('max_frequency_hz',self.qam_preamble_frequency_khz,1000),('max_evm_percent',self.qam_preamble_evm,1),('search_s',self.qam_preamble_search_ms,.001)]:
                if key in training:number(var,training[key]/scale)
        pwm=args.get('pwm_preamble_options',{})
        if pwm:
            values=pwm.get('symbols','0 0 0');self.qam_pwm_symbols.set(' '.join(map(str,values)) if isinstance(values,list) else str(values))
            for key,var,scale in [('symbol_hz',self.tdcm_pwm_khz,1000),('gap_s',self.qam_pwm_gap_us,.000001),
                    ('search_s',self.qam_pwm_search_ms,.001),('max_error_percent',self.qam_pwm_error,1),
                    ('max_frequency_hz',self.qam_pwm_frequency_khz,1000),('width_offset_chips',self.qam_pwm_width_offset,1),
                    ('qam_search_symbols',self.qam_pwm_qam_window,1),('total_chips',self.pwm2_total_chips,1),
                    ('prefix_chips',self.pwm2_prefix_chips,1),('guard_chips',self.pwm2_guard_chips,1)]:
                if key in pwm:number(var,pwm[key]/scale)

    def show_qam_report(self):
        if self.closed:
            return
        if self.busy:
            messagebox.showinfo('分析进行中','请等待当前分析完成后打开QAM诊断。',parent=self.window)
            return
        failure = getattr(self.result,'auto_search_failure',None) if self.result is not None else None
        if self.recovery is None and failure:
            previous=getattr(self,'_qam_report',None)
            if previous is not None and not previous.closed:
                if getattr(previous,'failure',None) is failure:
                    previous.window.lift();return previous
                previous.close()
            from rx706_qam_debug_gui import AutoSearchFailureWindow
            self._qam_report=AutoSearchFailureWindow(self.window,failure)
            return self._qam_report
        if self.recovery is None or self.recovery.kind != 'qam':
            messagebox.showinfo('QAM诊断','选择QAM或TDCM模式并完成分析后，可查看候选符号和补偿诊断。',parent=self.window)
            return
        previous = getattr(self,'_qam_report',None)
        if previous is not None and not previous.closed:
            if previous.r is self.recovery:
                previous.window.deiconify(); previous.window.lift()
                return previous
            previous.close()
        from rx706_qam_debug_gui import QAMReportWindow
        self._qam_report = QAMReportWindow(self.window,self.recovery,self.result)
        return self._qam_report

    def show_pwm_symbols(self):
        """Open a bounded table; exports include all rows and preserve erasures."""
        if self.closed:
            return
        if self.busy:
            pwm_running = self.constellation_mode.get() in PWM_MODES
            if pwm_running:
                self._open_pwm_after_analysis = True
            messagebox.showinfo('分析进行中',
                'PWM 正在分析，完成后会自动打开符号表。' if pwm_running else
                '当前分析尚未结束，请等待完成后再点击 PWM 符号表。', parent=self.window)
            return
        r = self.recovery
        if r is None or r.kind not in ('pwm', 'pwm2'):
            if not self.filename.get().strip():
                self.browse()
                if not self.filename.get().strip():
                    return
            if self.constellation_mode.get() not in PWM_MODES:
                self.constellation_mode.set(PWM2_MODE)
                if hasattr(self, '_parameter_groups'):
                    self.refresh_parameter_visibility()
            self._open_pwm_after_analysis = True
            self.analyze()
            if not self.busy:
                self._open_pwm_after_analysis = False
                messagebox.showwarning('无法开始 PWM 解调', self.status.get(), parent=self.window)
            return
        existing = getattr(self, '_pwm_dialog', None)
        if existing is not None and existing.winfo_exists():
            if getattr(self, '_pwm_dialog_result', None) is r:
                existing.deiconify()
                existing.lift()
                existing.focus_set()
                return existing
            existing.destroy()
        import csv
        from pathlib import Path
        source_name = str(self.result.path) if self.result is not None else self.filename.get()
        dialog = tk.Toplevel(self.window)
        self._pwm_dialog = dialog
        self._pwm_dialog_result = r
        dialog.title("PWM 符号持续时间与信息 · "+Path(source_name).name)
        dialog.transient(self.window)
        dialog.geometry("1360x680")
        dialog.minsize(900, 450)
        outer = ttk.Frame(dialog, padding=10)
        outer.pack(fill='both', expand=True)
        good = int(r.frames['valid'].sum())
        idle = int((r.frames['symbol'] == -2).sum())
        training_count = int(r.frames['training'].sum()) if r.training_calibration is not None else 0
        warned = int((r.frames['valid'] & (r.frames['glitch_count'] > 0)).sum())
        reference = r.reference_check
        estimated = (int((r.frames['decision_estimated'] & r.frames['valid']).sum())
                     if r.pulse_options is not None else 0)
        reference_summary = reference.summary() if reference is not None else '未测BER（未启用已知发端符号循环）。'
        if not r.decode_locked:
            reference_summary = ('PWM编码未锁定：'+r.decode_note+
                '\n已展示候选符号/比特（允许错误），候选不受检查失败拦截；已确认信息保持?，不计算BER。')
        ttk.Label(outer, text=f"{r.protocol_label}；{r.total_chips} 片 / 符号；"
            f"有效 {good}（其中毛刺告警{warned}），空闲 {idle}，训练 {training_count}，无效 {len(r.frames)-good-idle-training_count}；{r.mapping_note}。"
            "? 为擦除，- 为空闲或前导。\n"+r.calibration_description+'\n'+reference_summary,
            wraplength=1280).pack(fill='x', pady=(0, 8))
        if r.pulse_options is not None:
            ttk.Label(outer, text=f'最近码字估计 {estimated} 个：超容差仍显示符号和比特，金色标记；这些判决参与BER/误块率。缺脉冲或身份失败仍为擦除。',
                      wraplength=1280).pack(fill='x', pady=(0, 6))
        selected_user = tk.StringVar(value='全部用户')
        user_filter = None
        if r.frame_check is not None:
            user_filter = ttk.Combobox(outer, textvariable=selected_user, state='readonly',
                values=['全部用户']+[f"{i+1}: {u['name']}" for i, u in enumerate(r.frame_options['users'])])
            user_filter.pack(fill='x', pady=(0, 6))

        def selected_frames(payload_only=False):
            rows = r.frames
            if r.frame_check is not None:
                if selected_user.get() != '全部用户':
                    rows = rows[rows['user_id'] == int(selected_user.get().split(':', 1)[0])]
                if payload_only:
                    rows = rows[rows['frame_part'] == '净荷']
            elif payload_only and r.training_calibration is not None:
                rows = rows[~rows['training']]
            return rows
        headers = ('符号序号', '起始时间(μs)', '状态1持续(μs)', '状态0持续(μs)',
                   '原始高片数', '原始占空比(%)', '原始时间片码型', '码型中1的片数', '原始码型推断符号',
                   '脉宽判决用高片数', '脉宽候选符号（允许错误）', '候选比特', '候选依据',
                   '符号值', '信息比特', '判决', '恢复高片数', '判决占空比(%)', '恢复占空比(%)',
                   '忽略毛刺数', '累计忽略(ns)', '最大忽略(ns)', '波形告警（单独统计）',
                   '参考符号', '参考比特', '序列校验', '错比特数（同步不计入BER）')
        if r.version == 2 and not r.output_binary:
            headers = tuple({'信息比特': f'中间{r.code_chips}片码型', '候选比特': f'候选{r.code_chips}片码型',
                             '参考比特': f'参考{r.code_chips}片码型'}.get(title, title) for title in headers)
        if r.decision_mode == 'pattern':
            headers = tuple({'脉宽判决用高片数': '脉宽分支高片数（对照）',
                             '脉宽候选符号（允许错误）': '码型候选符号（允许错误）'}.get(title, title) for title in headers)
        field_names = ('index', 'start_s', 'one_us', 'zero_us', 'high_chips',
                       'duty_percent', 'pattern', 'pattern_ones', 'pattern_symbol', 'decision_high_chips',
                       'candidate_symbol', 'candidate_bits', 'candidate_source',
                       'symbol', 'bits', 'reason', 'decoded_high_chips', 'decision_duty_percent', 'decoded_duty_percent',
                       'glitch_count', 'glitch_us', 'glitch_max_us', 'warning',
                       'expected_symbol', 'expected_bits', 'reference_status', 'bit_errors')
        if r.training_calibration is not None:
            headers += ('训练段', '校准编号', '脉宽补偿(片)', '补偿后高片数', '补偿后T1(μs)', '补偿后T0(μs)', '原始脉宽检查（非参考验证）', '测宽比例', '校准依据', '拟合时钟测宽(片)')
            field_names += ('training', 'calibration_id', 'width_correction_chips', 'corrected_high_chips',
                            'corrected_one_us', 'corrected_zero_us', 'raw_reason',
                            'width_scale', 'calibration_source', 'clock_high_chips')
        if r.output_binary and r.pulse_options is None:
            headers += (f'{len(r.symbol_widths)}位温度计码（档位映射）',)
            field_names += ('duty_thermometer',)
        if r.pulse_options is not None:
            headers += ('上升沿时间(µs)', '下降沿时间(µs)', '脉冲延迟(片)', '脉冲延迟(µs)',
                        '原始高脉冲(µs)', '前导估计拖尾(µs)', '前导残差(µs)', '估计判决', '最近码字距离(片)')
            field_names += ('rise_s', 'fall_s', 'delay_chips', 'delay_us', 'raw_pulse_us', 'tail_us',
                            'calibration_error_us', 'decision_estimated', 'decision_distance_chips')
        if r.frame_check is not None:
            headers += ('帧内区段', '用户编号', '块编号', '块内净荷序号', '身份码有效', '完整块')
            field_names += ('frame_part', 'user_id', 'block_id', 'payload_index', 'identity_ok', 'block_complete')
        table_box = ttk.Frame(outer)
        table_box.pack(fill='both', expand=True)
        tree = ttk.Treeview(table_box, columns=field_names, show='headings', height=15)
        vertical = ttk.Scrollbar(table_box, orient='vertical', command=tree.yview)
        horizontal = ttk.Scrollbar(table_box, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side='right', fill='y')
        horizontal.pack(side='bottom', fill='x')
        tree.pack(fill='both', expand=True)
        from rx706_table_layout import configure_symbol_table
        restore_columns = configure_symbol_table(tree, field_names, headers, r.total_chips, r.code_chips)
        compact = tuple(key for key in ('index', 'start_s', 'high_chips', 'decision_high_chips',
                         'decoded_high_chips', 'symbol', 'reason', 'reference_status') if key in field_names)
        if r.output_binary:
            compact = ('index', 'start_s', 'duty_percent', 'decision_duty_percent',
                       'decoded_duty_percent', 'symbol', 'duty_thermometer', 'bits', 'reason')
        if r.frame_check is not None:
            compact = ('index', 'frame_part', 'user_id', 'block_id', 'payload_index',
                       'symbol', 'bits', 'reference_status', 'bit_errors', 'reason')
        if r.pulse_options is not None:
            compact = ('index', 'user_id', 'block_id', 'delay_chips', 'corrected_high_chips',
                       'tail_us', 'symbol', 'bits', 'decision_estimated', 'reference_status', 'reason')
        tree.configure(displaycolumns=compact)
        tree.tag_configure('invalid', foreground='#b64720')
        tree.tag_configure('idle', foreground='#777777')
        tree.tag_configure('warning', foreground='#a56a00')
        tree.tag_configure('mismatch', foreground='#c02020')
        tree.tag_configure('training', foreground='#355fb5')

        def values(row, numeric=False):
            from rx706_pwm import pattern_symbol, decision_width_chips
            start = float(row['start_s'])*1e6
            position = int(row['index'])-int(r.frames['index'][0])
            if row['symbol'] == -3:
                check_values = ('', '', '训练/不统计', '')
            elif reference is not None and reference.locked:
                expected = int(reference.expected[position])
                errors = int(reference.row_bit_errors[position])
                expected_bits = (reference.row_expected_bits[position] if r.frame_check is not None else
                    reference.bitwords[expected] if reference.bitwords is not None else format(expected, f'0{r.bits_per_symbol}b'))
                check_values = (expected if expected >= 0 else '', expected_bits,
                                str(reference.row_status[position]), errors if errors >= 0 else '')
            else:
                check_values = ('', '', '未同步' if reference is not None else '未启用', '')
            calibration_values = ()
            if r.training_calibration is not None:
                calibration_values = (bool(row['training']), int(row['calibration_id']),
                    *(float(row[key]) if numeric else f'{row[key]:.4f}' for key in
                      ('width_correction_chips', 'corrected_high_chips', 'corrected_one_us', 'corrected_zero_us')),
                    str(row['raw_reason']), float(row['width_scale']), str(row['calibration_source']),
                    float(row['clock_high_chips']))
            raw_pattern = str(row['pattern'])
            raw_k = pattern_symbol(raw_pattern, r.total_chips, r.code_chips, r.prefix_chips)
            if raw_k >= 0 and r.symbol_widths is not None:
                raw_k = r.width_symbol(raw_k+r.prefix_chips)
            raw_value = raw_k if raw_k >= 0 else '空闲' if raw_k == -2 else '码型不符'
            used_width = decision_width_chips(row)
            decoded_width = ((r.symbol_widths[int(row['symbol'])] if r.symbol_widths is not None else
                              int(row['symbol'])+(r.prefix_chips or 1)) if row['valid'] else None)
            return (int(row['index']), start if numeric else f'{start:.4f}',
                    *(float(row[key]) if numeric else f'{row[key]:.4f}'
                      for key in ('one_us', 'zero_us', 'high_chips', 'duty_percent')),
                    raw_pattern, raw_pattern.count('1'), raw_value, used_width if numeric else f'{used_width:.4f}',
                    int(row['candidate_symbol']) if row['candidate_symbol'] >= 0 else ('训练' if row['candidate_symbol'] == -3 else '空闲' if row['candidate_symbol'] == -2 else '?'),
                    str(row['candidate_bits']), str(row['candidate_source']),
                    int(row['symbol']) if row['symbol'] >= 0 else ('训练' if row['symbol'] == -3 else '空闲' if row['symbol'] == -2 else '无效'),
                    str(row['bits']), str(row['reason']),
                    decoded_width if decoded_width is not None else '',
                    100*used_width/r.total_chips if numeric else f'{100*used_width/r.total_chips:.3f}',
                    (100*decoded_width/r.total_chips if numeric else f'{100*decoded_width/r.total_chips:.3f}') if decoded_width is not None else '',
                    int(row['glitch_count']),
                    *(float(row[key])*1e3 if numeric else f'{row[key]*1e3:.2f}' for key in ('glitch_us', 'glitch_max_us')),
                    str(row['warning']), *check_values, *calibration_values,
                    *((r.duty_thermometer(row),) if r.output_binary and r.pulse_options is None else ()),
                    *((float(row['rise_s'])*1e6, float(row['fall_s'])*1e6,
                       *(float(row[key]) for key in ('delay_chips', 'delay_us', 'raw_pulse_us', 'tail_us', 'calibration_error_us')),
                       bool(row['decision_estimated']) if numeric else '是' if row['decision_estimated'] else '否',
                       float(row['decision_distance_chips']))
                      if r.pulse_options is not None else ()),
                    *((str(row['frame_part']), int(row['user_id']), int(row['block_id']),
                       int(row['payload_index']) if row['payload_index'] >= 0 else '',
                       bool(row['identity_ok']), bool(row['block_complete'])) if r.frame_check is not None else ()))

        def populate_table():
            tree.delete(*tree.get_children())
            for row in selected_frames()[:2000]:
                tag = ('warning' if row['glitch_count'] else '') if row['valid'] else ('training' if row['symbol'] == -3 else 'idle' if row['symbol'] == -2 else 'invalid')
                if row['valid'] and '超容差' in row['reason']:
                    tag = 'warning'
                if row['valid'] and r.pulse_options is not None and row['decision_estimated']:
                    tag = 'warning'
                position = int(row['index'])-int(r.frames['index'][0])
                if reference is not None and reference.locked and '不匹配' in reference.row_status[position]:
                    tag = 'mismatch'
                tree.insert('', 'end', values=values(row), tags=(tag,) if tag else ())
        populate_table()
        ttk.Label(outer, text=f"表格显示前 {min(2000,len(r.frames))} 行，共 {len(r.frames)} 行；CSV 导出全部结果。"
            "起始时间相对原始采集文件；T1/T0取忽略短毛刺后的主脉冲边沿。"
            + ("当前按时间片码型判决，候选与码型推断一致；脉宽和补偿列仅供对照。" if r.decision_mode == 'pattern' else
               "原始码型按每片多数采样显示，与连续边沿测宽是两种估计；脉宽候选按判决用高片数取整，可能与码型推断不同。") +
            "金色为毛刺告警或最近码字估计；红色为参考不匹配；最近估计仍计入BER。", wraplength=1250).pack(fill='x', pady=6)
        show_candidates = tk.BooleanVar(value=not r.decode_locked)
        ttk.Label(outer, text=(f'中间{r.code_chips}片码型预览（前2000个符号；受约束码型）' if r.version == 2 and not r.output_binary else
                              '分组比特预览（前2000个符号，MSB优先；候选含未通过检查的估计值）')).pack(anchor='w')
        text = scrolledtext.ScrolledText(outer, height=5, wrap='word')
        text.pack(fill='x', pady=4)
        def refresh_preview():
            text.configure(state='normal')
            text.delete('1.0', 'end')
            text.insert('1.0', ' | '.join(selected_frames(payload_only=r.frame_check is not None)['candidate_bits' if show_candidates.get() else 'bits'][:2000]))
            text.configure(state='disabled')
        refresh_preview()
        if user_filter is not None:
            user_filter.bind('<<ComboboxSelected>>', lambda event: (populate_table(), refresh_preview()))
        ttk.Checkbutton(outer, text='预览候选比特（允许错误，不用于BER）', variable=show_candidates,
                        command=refresh_preview).pack(anchor='w')
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x', pady=(6, 0))
        show_details = tk.BooleanVar(value=False)
        ttk.Checkbutton(buttons, text='显示全部诊断列', variable=show_details,
                        command=lambda: tree.configure(displaycolumns=field_names if show_details.get() else compact)).pack(side='right', padx=8)
        ttk.Button(buttons, text='复位列宽', command=restore_columns).pack(side='right')

        def export(kind):
            is_csv = kind == 'csv'
            extension = 'csv' if is_csv else 'txt'
            suffix = '_pwm.csv' if is_csv else ('_candidate_bits.txt' if kind == 'candidate' else '_bits.txt')
            target = filedialog.asksaveasfilename(parent=dialog, title='导出全部 PWM 解调结果',
                initialfile=Path(source_name).stem+suffix,
                defaultextension='.'+extension, filetypes=[('PWM CSV' if is_csv else '分组比特流', '*.'+extension)])
            if not target:
                return
            try:
                if is_csv:
                    with open(target, 'w', newline='', encoding='utf-8-sig') as stream:
                        writer = csv.writer(stream)
                        writer.writerow(headers)
                        writer.writerows(values(row, numeric=True) for row in selected_frames())
                else:
                    with open(target, 'w', encoding='utf-8') as stream:
                        stream.write(f'# {r.protocol_label}; {r.total_chips} slots; {r.mapping_note}; ?=erasure; -=idle\n')
                        if r.version == 2 and not r.output_binary:
                            stream.write(f'# Middle {r.code_chips}-slot thermometer words; {r.code_chips+1} states, not independent payload bits.\n')
                            stream.write(f'# Decision basis: {r.decision_mode}; raw time-slot pattern and width measurements retained in CSV.\n')
                        stream.write('# '+reference_summary.replace('\n', '\n# ')+'\n')
                        if kind == 'candidate':
                            stream.write('# 候选比特：含检查失败的最近脉宽估计，仅供观察，不用于BER。\n')
                        exported = selected_frames(payload_only=True)
                        if r.training_calibration is not None:
                            stream.write('# Configured preambles excluded; calibration uses preceding observations only.\n')
                        stream.write(' | '.join(exported['candidate_bits' if kind == 'candidate' else 'bits'])+'\n')
            except OSError as exc:
                messagebox.showerror('导出失败', str(exc), parent=dialog)
                return
            messagebox.showinfo('导出完成', target, parent=dialog)

        ttk.Button(buttons, text='导出全部 CSV…', command=lambda: export('csv')).pack(side='left', padx=(0, 10))
        ttk.Button(buttons, text='导出分组比特流…', command=lambda: export('txt')).pack(side='left')
        ttk.Button(buttons, text='导出候选比特流…', command=lambda: export('candidate')).pack(side='left', padx=(10, 0))
        if r.frame_check is not None:
            def export_metrics():
                target = filedialog.asksaveasfilename(parent=dialog, title='导出逐块结果和整体指标',
                    initialfile=Path(source_name).stem+'_blocks.json', defaultextension='.json', filetypes=[('统计JSON', '*.json')])
                if not target:
                    return
                try:
                    report = dict(config=r.frame_options, locked=r.frame_check.locked,
                                  summary=r.frame_check.summary(), overall=r.frame_check.overall,
                                  per_user=r.frame_check.per_user, blocks=r.frame_check.blocks)
                    if r.pulse_options is not None:
                        report.update(pulse_config=r.pulse_options, calibrations=r.pulse_calibrations)
                    if getattr(r, 'replay_warning', None):
                        from rx706_pwm_replay import payload_diagnostics
                        report.update(warning=r.replay_warning, **payload_diagnostics(r))
                    Path(target).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                except OSError as exc:
                    messagebox.showerror('导出失败', str(exc), parent=dialog)
            ttk.Button(buttons, text='导出逐块统计…', command=export_metrics).pack(side='left', padx=(10, 0))
        dialog.update_idletasks()
        dialog.deiconify()
        dialog.lift()
        dialog.focus_set()
        return dialog

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        report = getattr(self,'_qam_report',None)
        if report is not None:
            report.close()
        if hasattr(self, "navigation"):
            self.navigation.close()
        canvas = getattr(self, 'canvas', None)
        idle_draw = getattr(canvas, '_idle_draw_id', None)
        if idle_draw is not None:
            try:
                canvas.get_tk_widget().after_cancel(idle_draw)
            except tk.TclError:
                pass
            canvas._idle_draw_id = None
        for variable, token in getattr(self, "_info_traces", []):
            variable.trace_remove("write", token)
        if hasattr(self, "_sidebar_wheel_tag"):
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.window.unbind_class(self._sidebar_wheel_tag, event)
        if self.poll_after:
            self.window.after_cancel(self.poll_after)
            self.poll_after = None
        self.window.destroy()

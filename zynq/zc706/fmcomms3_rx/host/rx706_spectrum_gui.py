"""Optional Tk/Matplotlib window for local capture analysis; no board traffic."""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from rx706_spectrum import AnalysisCancelled, analyze_bin, display_trace

RECOVERY_MODE = "周期方波载波恢复"
PWM_MODE = "PWM 脉宽解调"
PWM2_MODE = "PWM version2"
PWM_MODES = (PWM_MODE, PWM2_MODE)
QAM_MODE = "QAM 解调"
QAM_PWM_MODE = "PWM前导 + QAM"
TDCM_MODE = "TDCM 谐波QAM"
QAM_MODES = (QAM_MODE, QAM_PWM_MODE, TDCM_MODE)
RECOVERY_MODES = (RECOVERY_MODE, PWM_MODE, PWM2_MODE, *QAM_MODES)
MODE_KINDS = {RECOVERY_MODE: 'periodic', PWM_MODE: 'pwm', PWM2_MODE: 'pwm2',
              QAM_MODE: 'qam', QAM_PWM_MODE: 'qam', TDCM_MODE: 'qam'}
PLOT_NAMES = ("总览", "频谱", "星座", "IQ 波形", "PWM 包络")


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
        self._open_pwm_after_analysis = False
        self._pwm_dialog = None
        self._pwm_dialog_result = None
        self.poll_after = None
        self.filename = tk.StringVar(value=str(filename))
        self.channel = tk.StringVar(value="RX1")
        self.nfft = tk.StringVar(value="65536")
        self.averages = tk.StringVar(value="16")
        self.first_tuple = tk.StringVar(value="0")
        self.rate = tk.StringVar(value="15.36")
        self.remove_dc = tk.BooleanVar(value=True)
        self.constellation_mode = tk.StringVar(value="原始 IQ")
        self.cycle_khz = tk.StringVar(value="100")
        self.recovery_ms = tk.StringVar(value="1000")
        self.center_hint = tk.StringVar(value="自动")
        self.qam_order = tk.StringVar(value="256")
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
        self.pwm_symbol_khz = tk.StringVar(value="100")
        self.pwm2_total_chips = tk.StringVar(value="25")
        self.pwm2_prefix_chips = tk.StringVar(value="5")
        self.pwm2_guard_chips = tk.StringVar(value="8")
        self.pwm2_preset = tk.StringVar(value="自定义")
        self.pwm2_layout_summary = tk.StringVar()
        self.pwm2_decision_mode = tk.StringVar(value="时间片码型")
        self.pwm2_training_enabled = tk.BooleanVar(value=False)
        self.pwm2_training_tolerance = tk.StringVar(value="0.5")
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

        def group(title):
            box = ttk.LabelFrame(body, text=title, padding=8)
            box.pack(fill="x", pady=(0, 9))
            box.columnconfigure(1, weight=1)
            return box

        def field(box, label, variable, values=None, callback=None):
            row = box.grid_size()[1]
            ttk.Label(box, text=label, wraplength=140).grid(row=row, column=0, sticky="w", padx=(0, 7), pady=4)
            widget = (ttk.Combobox(box, textvariable=variable, values=values, state="readonly", width=16)
                      if values else ttk.Entry(box, textvariable=variable, width=16))
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            if callback:
                widget.bind("<<ComboboxSelected>>", callback)
            self.controls.append(widget)
            return widget

        box = group("采样与频谱")
        field(box, "通道", self.channel, ("RX1", "RX2"))
        field(box, "FFT 点数", self.nfft, tuple(str(1 << p) for p in range(8, 21)))
        field(box, "平均段数", self.averages)
        field(box, "起始采样点", self.first_tuple)
        field(box, "采样率 (MS/s)", self.rate)
        dc = ttk.Checkbutton(box, text="频谱去直流", variable=self.remove_dc)
        dc.grid(row=5, column=0, columnspan=2, sticky="w", pady=4)
        self.controls.append(dc)
        ttk.Checkbutton(box, text="新下载文件自动分析", variable=self.auto).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        box = group("信号解调模式")
        field(box, "星座处理", self.constellation_mode,
              ("原始 IQ", *RECOVERY_MODES), self.on_mode_changed)
        box = group("载波恢复设置")
        field(box, "完整周期频率 (kHz，仅方波)", self.cycle_khz)
        field(box, "恢复时长 (ms)", self.recovery_ms)
        field(box, "基带载波 (MHz / 自动)", self.center_hint)
        ttk.Label(box, text="周期切换选方波；PWM和QAM使用各自解调模式。修改后点分析。", wraplength=290).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        box = group("QAM 解调设置")
        field(box, "QAM阶数", self.qam_order, ("16", "64", "256"))
        field(box, "符号率 (kSym/s)", self.qam_symbol_khz)
        field(box, "基带载波频偏 (MHz)", self.qam_carrier_mhz)
        field(box, "发端脉冲形状", self.qam_pulse, ("RRC", "Rectangular"))
        field(box, "RRC滚降系数", self.qam_rolloff)
        ttk.Label(box, text="载波已居中填0；参数须与发端一致。静态相位/定时估计，存在90°歧义，未测BER。", wraplength=290).grid(row=5, column=0, columnspan=2)
        box = group("QAM：可配置前导 / 导频")
        pilot = ttk.Checkbutton(box, text="启用已知前导同步与频偏校正", variable=self.qam_preamble_enabled)
        pilot.grid(row=0, column=0, columnspan=2, sticky="w")
        self.controls.append(pilot)
        field(box, "前导表示方式", self.qam_preamble_format, ("符号值（Gray）", "I/Q电平"))
        field(box, "前导序列（空格分隔）", self.qam_preamble_symbols)
        field(box, "残余频偏范围 ±kHz", self.qam_preamble_frequency_khz)
        field(box, "前导匹配EVM上限 (%)", self.qam_preamble_evm)
        field(box, "前导搜索时长 (ms)", self.qam_preamble_search_ms)
        ttk.Label(box, text="填3～256个已知符号；默认0 0 0仅为示例，须与发端一致。"
                  "符号值支持0b二进制、0x十六进制，使用本机Gray映射；I/Q按未归一化奇数电平填写，"
                  "如 -15,-15 -15,-15 -15,-15。去除本次匹配前导后输出后续符号。"
                  "三符号可能误匹配；仅估计固定频偏和相位，不持续跟踪，不自动剔除后续重复前导。",
                  wraplength=290).grid(row=7, column=0, columnspan=2, sticky="w")
        box = group("混合模式：PWM前导 + QAM")
        field(box, "PWM前导符号值", self.qam_pwm_symbols)
        field(box, "PWM→QAM间隔 (µs)", self.qam_pwm_gap_us)
        field(box, "PWM搜索时长 (ms)", self.qam_pwm_search_ms)
        field(box, "PWM拟合误差上限 (%)", self.qam_pwm_error)
        field(box, "高电平脉宽偏移 (片)", self.qam_pwm_width_offset)
        field(box, "PWM残余频偏范围 ±kHz", self.qam_pwm_frequency_khz)
        field(box, "QAM起始搜索窗 (符号)", self.qam_pwm_qam_window)
        ttk.Label(box, text="星座处理选择“PWM前导 + QAM”生效。PWM符号率沿用下方PWM频率，"
                  "片数沿用version2（默认25/5/8）；此默认下0 0 0表示三个20%符号。"
                  "按已知序列拟合两态IQ，支持等幅反相；无需8个PWM符号。"
                  "间隔从最后一个PWM完整周期结束算起；脉宽偏移=实测高片数−标称高片数。"
                  "PWM估计粗频偏，不提供QAM绝对相位。上方QAM前导可选；勾选后须在PWM之后发送对应QAM训练符号。"
                  "不勾选则直接盲解调QAM，仍有90°歧义。当前仅处理一帧候选。",
                  wraplength=290).grid(row=7, column=0, columnspan=2, sticky="w")
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
        field(box, "符号率 / PWM频率 (kHz)", self.pwm_symbol_khz)
        field(box, "v1编码时间片数", self.pwm_code_chips, ("2", "4", "8", "16"))
        field(box, "v1末尾保护片数", self.pwm_guard_chips)
        field(box, "控制状态 1 对应", self.pwm_polarity, ("自动（PWM约束）", "A=1", "B=1"))
        field(box, "处理抽取倍数", self.pwm_decimation, ("1", "2", "4", "8", "16"))
        field(box, "脉宽容差 (时间片)", self.pwm_tolerance)
        field(box, "两态滞回半宽 (比例)", self.pwm_hysteresis)
        field(box, "单次毛刺上限 (时间片)", self.pwm_glitch_chips)
        ttk.Label(box, text="v1：默认100kHz对应当前固定方波测试；1Mbps目标需填333.333333kHz，不能直接用于100kHz文件。"
                  "8+2=10片，10%～80%对应符号0～7；全0为空闲。"
                  "先选“PWM 脉宽解调”再分析。原速率即可解调；可选抽取会先抗混叠滤波，"
                  "抽取后每片至少3点，建议4点以上；3～4点会显示计时余量提示。"
                  "毛刺上限0为严格模式；仅忽略闭合短回跳，累计不超过0.30片。"
                  "毛刺告警不等于误码。A/B映射需由训练序列核对。", wraplength=290).grid(
                      row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        search = ttk.Checkbutton(box, text="自动搜索有效PWM片段", variable=self.pwm_auto_search)
        search.grid(row=9, column=0, columnspan=2, sticky="w", pady=4)
        self.controls.append(search)
        ttk.Label(box, text="在起始采样点后的恢复时长内搜索；默认1秒。找到后保留前置余量重新解调，不预留或删除导频。", wraplength=290).grid(row=10, column=0, columnspan=2, sticky="w")
        box = group("PWM version2：可配置时间片")
        field(box, "预设", self.pwm2_preset,
              ("自定义", "20片：8+10+2", "50片：20+25+5", "50片：8+40+2"), self.apply_pwm2_preset)
        field(box, "每符号总片数", self.pwm2_total_chips)
        field(box, "前置固定高片数", self.pwm2_prefix_chips)
        field(box, "末尾固定低片数", self.pwm2_guard_chips)
        field(box, "符号判决依据", self.pwm2_decision_mode, ("时间片码型", "连续脉宽"))
        ttk.Label(box, textvariable=self.pwm2_layout_summary, wraplength=290).grid(row=5, column=0, columnspan=2, sticky='w')
        ttk.Label(box, text="中间片数=总片数−前高−后低，按连续k个1再接0编码。"
                  "符号率、A/B映射、抽取和容差沿用上方PWM设置。修改后重新分析；参数须与发端一致。"
                  "码型模式按中间片数判决，不使用脉宽取整；连续脉宽模式沿用原算法。"
                  "码型长度不是独立信息比特数。", wraplength=290).grid(row=6, column=0, columnspan=2, sticky='w')
        for variable in (self.pwm2_total_chips, self.pwm2_prefix_chips, self.pwm2_guard_chips,
                         self.pwm_symbol_khz, self.rate, self.pwm_decimation):
            variable.trace_add('write', lambda *args: self.update_pwm2_layout())
        self.update_pwm2_layout()
        box = group("version2：20%三连训练补偿")
        train = ttk.Checkbutton(box, text="启用三连同宽训练与脉宽补偿", variable=self.pwm2_training_enabled)
        train.grid(row=0, column=0, columnspan=2, sticky='w')
        self.controls.append(train)
        field(box, "三连宽度极差上限 (片)", self.pwm2_training_tolerance)
        ttk.Label(box, text="连续3个同宽符号即按20%训练段处理，不要求固定组长。"
                  "补偿=20%标准高片数−三连实测中位数；只补偿后续脉宽，直到下一训练段更新。"
                  "原始波形保留。训练不输出净荷、不计入BER。"
                  "码型判决时仅识别训练并记录补偿供对照，不对码型重复加补偿。"
                  "数据中出现三连同宽也会被识别为训练。", wraplength=290).grid(row=2, column=0, columnspan=2, sticky='w')
        box = group("已知循环校验 / BER")
        check = ttk.Checkbutton(box, text="启用已知发端符号循环", variable=self.pwm_reference_enabled)
        check.grid(row=0, column=0, columnspan=2, sticky='w', pady=4)
        self.controls.append(check)
        field(box, "循环符号值（v2用空格分隔）", self.pwm_reference)
        field(box, "同步窗 (符号数)", self.pwm_sync_symbols)
        ttk.Label(box, text="当前发端填写7 6 4 2（也支持7642）。前64个符号仅用于循环对齐；"
                  "其余符号逐位比对，BER仅统计已判决比特，擦除单列。参考序列不用于纠错。"
                  "修改后点分析。", wraplength=290).grid(row=3, column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Label(body, text="同名 JSON 的采样率优先。原始波形最多显示 4096 点；恢复波形显示约 10 个周期。详细指标在“统计 / 状态”页。", wraplength=300).pack(fill="x")
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
        if "未锁定" in self.status.get() or "失败" in self.status.get():
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
                if r.reference_check is not None:
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
        step = max(1,len(z)//20000)
        self.iq_axes.scatter(z.real[::step], z.imag[::step], s=3, alpha=.3)
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

    def apply_pwm2_preset(self, event=None):
        selected = self.pwm2_preset.get()
        layout = {'20片：8+10+2': (20, 8, 2),
                  '50片：20+25+5': (50, 20, 5),
                  '50片：8+40+2': (50, 8, 2)}.get(selected)
        if layout:
            for variable, value in zip((self.pwm2_total_chips, self.pwm2_prefix_chips,
                                        self.pwm2_guard_chips), layout):
                variable.set(str(value))
            self.pwm2_preset.set(selected)
        self.update_pwm2_layout()

    def update_pwm2_layout(self):
        from rx706_pwm import pwm2_layout
        try:
            total, prefix, guard = (int(v.get()) for v in
                (self.pwm2_total_chips, self.pwm2_prefix_chips, self.pwm2_guard_chips))
            code = pwm2_layout(total, prefix, guard)
            preset = {'20片：8+10+2': (20, 8, 2), '50片：20+25+5': (50, 20, 5),
                      '50片：8+40+2': (50, 8, 2)}.get(self.pwm2_preset.get())
            if preset is not None and preset != (total, prefix, guard):
                self.pwm2_preset.set('自定义')
            text = (f'{prefix}+{code}+{guard}={total}片；{code+1}态，k=0～{code}。'
                    f'占空比{100*prefix/total:g}%～{100*(total-guard)/total:g}%，步进{100/total:g}%。')
            fs, symbol, decimation = float(self.rate.get())*1e6, float(self.pwm_symbol_khz.get())*1000, int(self.pwm_decimation.get())
            if fs > 0 and symbol > 0 and decimation > 0:
                points = fs/(symbol*total*decimation)
                text += f' 每片{points:.3f}个处理点。'
                if points < 3:
                    text += '不足3点，请降低符号率或抽取倍数。'
                elif points < 4:
                    text += '低于建议4点，计时余量较小。'
            self.pwm2_layout_summary.set(text)
        except (ValueError, ZeroDivisionError) as exc:
            self.pwm2_layout_summary.set(f'参数待检查：{exc}')

    def on_mode_changed(self, event=None):
        if self.closed or self.busy:
            return
        expected_kind = MODE_KINDS.get(self.constellation_mode.get())
        wrong_header = (expected_kind == 'qam' and self.recovery is not None and
                        (self.constellation_mode.get() in (QAM_PWM_MODE,TDCM_MODE)) !=
                        (getattr(self.recovery, 'pwm_preamble', None) is not None))
        wrong_tdcm = (expected_kind == 'qam' and self.recovery is not None and
                      (self.constellation_mode.get() == TDCM_MODE) != (getattr(self.recovery,'tdcm',None) is not None))
        if self.constellation_mode.get() in RECOVERY_MODES and (self.recovery is None or self.recovery.kind != expected_kind or wrong_header or wrong_tdcm):
            self.analyze()
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
            self.wave_axes.set(title=r.protocol_label+'\nControl envelope | P = pilot',xlabel='Time from view start (us)',ylabel='Control level',ylim=(-.25,1.7))
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
            if self.constellation_mode.get() in RECOVERY_MODES:
                center = self.center_hint.get().strip()
                recovery_options = dict(duration_s=float(self.recovery_ms.get())/1000,
                    center_hint_hz=None if center.lower() in ('自动', 'auto', '') else float(center)*1e6)
                if self.constellation_mode.get() in QAM_MODES:
                    recovery_options.update(kind='qam', order=int(self.qam_order.get()),
                        symbol_hz=float(self.qam_symbol_khz.get())*1000,
                        rolloff=float(self.qam_rolloff.get()), pulse=self.qam_pulse.get(),
                        center_hint_hz=float(self.qam_carrier_mhz.get())*1e6)
                    if self.constellation_mode.get() == TDCM_MODE:
                        recovery_options.update(symbol_hz=float(self.tdcm_symbol_khz.get())*1000,
                            center_hint_hz=float(self.tdcm_carrier_mhz.get())*1e6)
                    if self.qam_preamble_enabled.get():
                        from rx706_qam_preamble import validate_preamble
                        preamble = dict(symbols=self.qam_preamble_symbols.get(),
                            representation='iq' if self.qam_preamble_format.get() == 'I/Q电平' else 'labels',
                            max_frequency_hz=float(self.qam_preamble_frequency_khz.get())*1000,
                            max_evm_percent=float(self.qam_preamble_evm.get()),
                            search_s=float(self.qam_preamble_search_ms.get())/1000)
                        validate_preamble(preamble, recovery_options['order'], recovery_options['symbol_hz'])
                        recovery_options['preamble_options'] = preamble
                    if self.constellation_mode.get() in (QAM_PWM_MODE,TDCM_MODE):
                        from rx706_qam_pwm import validate_pwm_header
                        pwm = dict(symbols=self.qam_pwm_symbols.get(),
                            total_chips=int(self.pwm2_total_chips.get()),
                            prefix_chips=int(self.pwm2_prefix_chips.get()),
                            guard_chips=int(self.pwm2_guard_chips.get()),
                            symbol_hz=float(self.tdcm_pwm_khz.get() if self.constellation_mode.get() == TDCM_MODE else self.pwm_symbol_khz.get())*1000,
                            gap_s=float(self.qam_pwm_gap_us.get())/1e6,
                            search_s=float(self.qam_pwm_search_ms.get())/1000,
                            max_error_percent=float(self.qam_pwm_error.get()),
                            width_offset_chips=float(self.qam_pwm_width_offset.get()),
                            max_frequency_hz=float(self.qam_pwm_frequency_khz.get())*1000,
                            qam_search_symbols=float(self.qam_pwm_qam_window.get()))
                        validate_pwm_header(pwm, options['sample_rate_hz'])
                        recovery_options['pwm_preamble_options'] = pwm
                    if self.constellation_mode.get() == TDCM_MODE:
                        from rx706_tdcm import validate_tdcm
                        tdcm = dict(base_hz=float(self.tdcm_base_khz.get())*1000,
                            harmonic=int(self.tdcm_harmonic.get()),
                            normalization='pwm' if self.tdcm_normalization.get() == 'PWM两态参考' else 'blind',
                            amplitude_scale=float(self.tdcm_amplitude.get()),phase_deg=float(self.tdcm_phase.get()),
                            payload_symbols=int(self.tdcm_count.get()))
                        validate_tdcm(options['sample_rate_hz'],recovery_options['symbol_hz'],tdcm)
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
                        from rx706_pwm import pwm2_layout
                        total = int(self.pwm2_total_chips.get())
                        prefix = int(self.pwm2_prefix_chips.get())
                        guard = int(self.pwm2_guard_chips.get())
                        code = pwm2_layout(total, prefix, guard)
                        recovery_options.update(kind='pwm2', total_chips=total,
                                                prefix_chips=prefix, code_chips=code, guard_chips=guard,
                                                decision_mode='pattern' if self.pwm2_decision_mode.get() == '时间片码型' else 'width')
                        if self.pwm2_training_enabled.get():
                            from rx706_pwm_training import validate_training
                            training = dict(same_tolerance_chips=float(self.pwm2_training_tolerance.get()))
                            validate_training(total, prefix, code, training)
                            recovery_options['training_options'] = training
                    if self.pwm_reference_enabled.get():
                        from rx706_pwm_reference import parse_reference
                        sync = int(self.pwm_sync_symbols.get())
                        if not 8 <= sync <= 4096:
                            raise ValueError('同步窗须为 8～4096 个符号')
                        recovery_options['reference_options'] = dict(
                            sequence=parse_reference(self.pwm_reference.get(), code+1 if self.constellation_mode.get() == PWM2_MODE else int(self.pwm_code_chips.get())),
                            sync_symbols=sync)
                else:
                    recovery_options['cycle_hz'] = float(self.cycle_khz.get())*1000
        except ValueError as exc:
            self.status.set(f"参数错误：{exc}")
            return
        self.result = None
        self.recovery = None
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
            result = analyze_bin(name, cancel_event=self.cancel, **options)
            recovery, recovery_error = None, None
            if recovery_options is not None:
                from rx706_recovery import recover_periodic
                try:
                    recovery_options = dict(recovery_options)
                    kind = recovery_options.pop('kind','periodic')
                    if kind == 'qam':
                        from rx706_qam import recover_qam
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
                        preview = ' '.join(''.join(map(str, b)) for b in r.bits[:24])
                        self.summary.set(f'{r.order}QAM；符号率 {r.symbol_hz/1000:g} kSym/s；载波频偏 {r.carrier_hz/1e6:g} MHz。\n静态相位和定时估计；未跟踪时钟漂移或残余频偏；90°旋转歧义未消除。\n判决EVM不能替代参考EVM或证明锁定。Gray映射：I后Q，各轴从负到正，MSB优先，须与发端一致。\n候选比特（未帧同步、未测BER）：{preview}')
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
                        if tdcm is not None:
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
                                f"Gray映射：I后Q，各轴从负到正，MSB优先。候选比特：{preview}")
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
                            states = r.code_chips+1
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
                                self.summary.set(self.summary.get()+
                                    f"\n20%三连训练补偿：{calibration['updates']}次更新，{calibration['training_symbols']}个训练符号；补偿范围{offset_text}。"
                                    '\n原始脉宽不改写；补偿后重做判决，不放宽容差。P为训练段，不输出净荷。'
                                    '数据中任意三连同宽也会触发训练；只有单一20%参考，无法估计比例误差或修复丢失边沿。'
                                    '\n参考循环只针对非训练符号，并在各训练段之间连续推进；不会逐组重新对齐参考。')
                                if r.decision_mode == 'pattern':
                                    self.summary.set(self.summary.get()+'\n当前按原始时间片码型判决；脉宽补偿列仅供对照，不影响码型符号值。')
                elif recovery_error:
                    prefix = 'QAM 解调' if self.constellation_mode.get() in QAM_MODES else ('PWM 解调' if self.constellation_mode.get() in PWM_MODES else '周期恢复')
                    self.status.set(self.status.get() + f"\n{prefix}未锁定：" + recovery_error)
                    self.summary.set(self.summary.get() + " 已显示原始 IQ，未强行生成星座簇。")
            else:
                self.axes.set_title("Analysis stopped" if kind == "cancelled" else "Analysis failed")
                self.status.set(("" if kind == "cancelled" else "分析失败：") + value)
            self.canvas.draw()
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
                self.constellation_mode.set(PWM_MODE)
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
        reference_summary = reference.summary() if reference is not None else '未测BER（未启用已知发端符号循环）。'
        if not r.decode_locked:
            reference_summary = ('PWM编码未锁定：'+r.decode_note+
                '\n已展示候选符号/比特（允许错误），候选不受检查失败拦截；已确认信息保持?，不计算BER。')
        ttk.Label(outer, text=f"{r.protocol_label}；{r.total_chips} 片 / 符号；"
            f"有效 {good}（其中毛刺告警{warned}），空闲 {idle}，训练 {training_count}，无效 {len(r.frames)-good-idle-training_count}；{r.mapping_note}。"
            "? 为擦除，- 为空闲。\n"+reference_summary, wraplength=1280).pack(fill='x', pady=(0, 8))
        headers = ('符号序号', '起始时间(μs)', '状态1持续(μs)', '状态0持续(μs)',
                   '原始高片数', '原始占空比(%)', '原始时间片码型', '码型中1的片数', '原始码型推断符号',
                   '脉宽判决用高片数', '脉宽候选符号（允许错误）', '候选比特', '候选依据',
                   '符号值', '信息比特', '判决',
                   '忽略毛刺数', '累计忽略(ns)', '最大忽略(ns)', '波形告警（单独统计）',
                   '参考符号', '参考比特', '序列校验', '错比特数（同步不计入BER）')
        if r.version == 2:
            headers = tuple({'信息比特': f'中间{r.code_chips}片码型', '候选比特': f'候选{r.code_chips}片码型',
                             '参考比特': f'参考{r.code_chips}片码型'}.get(title, title) for title in headers)
        if r.decision_mode == 'pattern':
            headers = tuple({'脉宽判决用高片数': '脉宽分支高片数（对照）',
                             '脉宽候选符号（允许错误）': '码型候选符号（允许错误）'}.get(title, title) for title in headers)
        field_names = ('index', 'start_s', 'one_us', 'zero_us', 'high_chips',
                       'duty_percent', 'pattern', 'pattern_ones', 'pattern_symbol', 'decision_high_chips',
                       'candidate_symbol', 'candidate_bits', 'candidate_source',
                       'symbol', 'bits', 'reason',
                       'glitch_count', 'glitch_us', 'glitch_max_us', 'warning',
                       'expected_symbol', 'expected_bits', 'reference_status', 'bit_errors')
        if r.training_calibration is not None:
            headers += ('训练段', '校准编号', '脉宽补偿(片)', '补偿后高片数', '补偿后T1(μs)', '补偿后T0(μs)', '原始脉宽检查（非参考验证）')
            field_names += ('training', 'calibration_id', 'width_correction_chips', 'corrected_high_chips',
                            'corrected_one_us', 'corrected_zero_us', 'raw_reason')
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
                expected_bits = reference.bitwords[expected] if reference.bitwords is not None else format(expected, f'0{r.bits_per_symbol}b')
                check_values = (expected, expected_bits,
                                str(reference.row_status[position]), errors if errors >= 0 else '')
            else:
                check_values = ('', '', '未同步' if reference is not None else '未启用', '')
            calibration_values = ()
            if r.training_calibration is not None:
                calibration_values = (bool(row['training']), int(row['calibration_id']),
                    *(float(row[key]) if numeric else f'{row[key]:.4f}' for key in
                      ('width_correction_chips', 'corrected_high_chips', 'corrected_one_us', 'corrected_zero_us')),
                    str(row['raw_reason']))
            raw_pattern = str(row['pattern'])
            raw_k = pattern_symbol(raw_pattern, r.total_chips, r.code_chips, r.prefix_chips)
            raw_value = raw_k if raw_k >= 0 else '空闲' if raw_k == -2 else '码型不符'
            used_width = decision_width_chips(row)
            return (int(row['index']), start if numeric else f'{start:.4f}',
                    *(float(row[key]) if numeric else f'{row[key]:.4f}'
                      for key in ('one_us', 'zero_us', 'high_chips', 'duty_percent')),
                    raw_pattern, raw_pattern.count('1'), raw_value, used_width if numeric else f'{used_width:.4f}',
                    int(row['candidate_symbol']) if row['candidate_symbol'] >= 0 else ('训练' if row['candidate_symbol'] == -3 else '空闲' if row['candidate_symbol'] == -2 else '?'),
                    str(row['candidate_bits']), str(row['candidate_source']),
                    int(row['symbol']) if row['symbol'] >= 0 else ('训练' if row['symbol'] == -3 else '空闲' if row['symbol'] == -2 else '无效'),
                    str(row['bits']), str(row['reason']), int(row['glitch_count']),
                    *(float(row[key])*1e3 if numeric else f'{row[key]*1e3:.2f}' for key in ('glitch_us', 'glitch_max_us')),
                    str(row['warning']), *check_values, *calibration_values)

        for row in r.frames[:2000]:
            tag = ('warning' if row['glitch_count'] else '') if row['valid'] else ('training' if row['symbol'] == -3 else 'idle' if row['symbol'] == -2 else 'invalid')
            position = int(row['index'])-int(r.frames['index'][0])
            if reference is not None and reference.locked and '不匹配' in reference.row_status[position]:
                tag = 'mismatch'
            tree.insert('', 'end', values=values(row), tags=(tag,) if tag else ())
        ttk.Label(outer, text=f"表格显示前 {min(2000,len(r.frames))} 行，共 {len(r.frames)} 行；CSV 导出全部结果。"
            "起始时间相对原始采集文件；T1/T0取忽略短毛刺后的主脉冲边沿。"
            + ("当前按时间片码型判决，候选与码型推断一致；脉宽和补偿列仅供对照。" if r.decision_mode == 'pattern' else
               "原始码型按每片多数采样显示，与连续边沿测宽是两种估计；脉宽候选按判决用高片数取整，可能与码型推断不同。") +
            "金色为可判决但有毛刺；红色为参考不匹配；同步行不计入BER。", wraplength=1250).pack(fill='x', pady=6)
        show_candidates = tk.BooleanVar(value=not r.decode_locked)
        ttk.Label(outer, text=(f'中间{r.code_chips}片码型预览（前2000个符号；受约束码型）' if r.version == 2 else
                              '分组比特预览（前2000个符号，MSB优先；候选含未通过检查的估计值）')).pack(anchor='w')
        text = scrolledtext.ScrolledText(outer, height=5, wrap='word')
        text.pack(fill='x', pady=4)
        def refresh_preview():
            text.configure(state='normal')
            text.delete('1.0', 'end')
            text.insert('1.0', ' | '.join(r.frames['candidate_bits' if show_candidates.get() else 'bits'][:2000]))
            text.configure(state='disabled')
        refresh_preview()
        ttk.Checkbutton(outer, text='预览候选比特（允许错误，不用于BER）', variable=show_candidates,
                        command=refresh_preview).pack(anchor='w')
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x', pady=(6, 0))
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
                        writer.writerows(values(row, numeric=True) for row in r.frames)
                else:
                    with open(target, 'w', encoding='utf-8') as stream:
                        stream.write(f'# {r.protocol_label}; {r.total_chips} slots; {r.mapping_note}; ?=erasure; -=idle\n')
                        if r.version == 2:
                            stream.write(f'# Middle {r.code_chips}-slot thermometer words; {r.code_chips+1} states, not independent payload bits.\n')
                            stream.write(f'# Decision basis: {r.decision_mode}; raw time-slot pattern and width measurements retained in CSV.\n')
                        stream.write('# '+reference_summary.replace('\n', '\n# ')+'\n')
                        if kind == 'candidate':
                            stream.write('# 候选比特：含检查失败的最近脉宽估计，仅供观察，不用于BER。\n')
                        exported = r.frames[~r.frames['training']] if r.training_calibration is not None else r.frames
                        if r.training_calibration is not None:
                            stream.write('# 20% pilot triples excluded; width corrections estimated only from preceding pilots.\n')
                        stream.write(' | '.join(exported['candidate_bits' if kind == 'candidate' else 'bits'])+'\n')
            except OSError as exc:
                messagebox.showerror('导出失败', str(exc), parent=dialog)
                return
            messagebox.showinfo('导出完成', target, parent=dialog)

        ttk.Button(buttons, text='导出全部 CSV…', command=lambda: export('csv')).pack(side='left', padx=(0, 10))
        ttk.Button(buttons, text='导出分组比特流…', command=lambda: export('txt')).pack(side='left')
        ttk.Button(buttons, text='导出候选比特流…', command=lambda: export('candidate')).pack(side='left', padx=(10, 0))
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

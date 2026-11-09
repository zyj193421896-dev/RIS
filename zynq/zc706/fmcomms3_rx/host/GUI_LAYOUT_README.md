# GUI布局与默认采集

主界面的采集长度默认设为“秒 / 1”，在15.36 MS/s、双通道I/Q int16格式下对应122880000字节、3750个完整包。可继续手动输入其他采集长度。信号分析的恢复时长默认为1000ms。

接收分析窗口使用三图布局：顶部频谱，底部星座与IQ/PWM包络。可单独放大图表；功率图和脉宽散点图已移除，功率统计与逐符号脉宽仍可在统计页、符号表和CSV中查看。默认PWM（占空比）；模式菜单同时保留原始IQ、周期方波恢复、PWM v1（旧协议）、新版PWM（占空比/时延＋脉宽）、普通QAM、PWM前导+QAM及全部TDCM/64QAM/256QAM预设。整合说明见 [PWM_QAM_INTEGRATION.md](PWM_QAM_INTEGRATION.md)。

PWM占空比默认使用时延＋脉宽码表，用户数为1，只循环发送各用户身份前导与净荷。步长决定总片数，最小高电平和末尾低电平预留可配置；拖尾仅由前导估计。信息率包含前导和用户时分开销。配置、码表与已知数据测试见 [PWM_PULSE_README.md](PWM_PULSE_README.md)。旧快速预设仍可选择旧帧逻辑。

图表拖动期间暂停自动布局，松开鼠标、离开画布或窗口尺寸变化时恢复，避免坐标轴随刻度文本宽度变化而偏移。双击切换单图，R复位；工具栏启用时优先使用工具栏操作。

符号表按当前字体测量列宽及行高，列宽不再随窗口缩小而被压缩。可拖动列分隔线扩大列宽，最小宽度保证标题及常见数值可读；用底部横向滚动条查看更多列。“复位列宽”恢复初始宽度。码型、候选、校准及完整结果导出不改变。

回归命令：在tests目录运行 `python -m unittest test_gui_layout test_plot_navigation test_signal_analysis_view test_pwm_gui test_pwm2 test_pwm_pattern test_pwm_training test_rx706_pwm test_pwm_search test_pwm_reference test_rx706_recovery test_rx706_qam`。`test_gui_layout`需要可用的Tcl/Tk运行时及显示环境；图表测试使用Agg，协议测试不访问板卡。

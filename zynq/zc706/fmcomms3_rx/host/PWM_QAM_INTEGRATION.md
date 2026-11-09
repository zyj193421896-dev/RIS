# PWM 与 QAM 接收功能整合

将收端分支 `feat/rx706-tdcm-qam` 的 `9659f9e` 更新与本机已有 QAM 改动整合，未修改发端码本工具或原始 IQ。

## 使用

在 `fmcomms3_rx` 目录运行 `/opt/anaconda3/bin/python host/rx706_gui.py`，打开 IQ 信号分析窗口。已经打开的窗口需要重启才能加载新代码。

- 新版 PWM：默认“PWM（占空比）”，支持时延＋脉宽联合码表、占空比范围、时间片协议、用户帧、参考校验及完整 CSV 导出。见 [PWM_PULSE_README.md](PWM_PULSE_README.md)、[PWM_MINIMAL_FRAME_README.md](PWM_MINIMAL_FRAME_README.md)。
- QAM：保留普通直接 IQ QAM、PWM前导＋QAM、TDCM 谐波、四角训练、64QAM 成对净荷、256QAM 120k 的 3+4+16 与旧 3+8+128。所有阶数、训练、补偿、参数 JSON、冻结复算和诊断入口保留。见 [QAM_README.md](QAM_README.md)、[QAM64_README.md](QAM64_README.md)、[QAM256_120K_README.md](QAM256_120K_README.md)。
- PWM 与 QAM 的前导片数分别记忆；切换到 QAM 时不会因为 PWM 的脉冲输入方式隐藏 QAM 前导参数。QAM 预设及加载 JSON 的新参数优先。
- QAM 模式继续沿用原有切换后分析行为；新版独立 PWM 切换后先核对参数，再点“分析信号”。当前 PWM 默认仍是独立脉冲解调。
- 已知 QAM 固定前导允许 64片/16高/48低、每片2点；独立 PWM 数据解调继续使用自身更严格的校验，两者不共用判决输出。

## 验证边界

集成回归包括两类解调的模式入口、参数可见性、切换恢复、路由、64/256QAM 前导、导出及 PWM 用户帧。合成数据和本机 GUI 测试不等于真实射频验收。

远端更新已经记录的 PWM 真实采样边界歧义、相位/周期不一致及脉宽偏差仍需单独修复，见 [原始分析记录](../reports/pwm_mt_test_analysis/RESULT.md)。本次未用已知净荷反向修正判决，也不将“已判决”解释为参考比特正确。

运行整合专项：`PYTHONPATH=host:tests MPLBACKEND=Agg python -B -m unittest test_pwm_qam_integration test_qam256_120k test_qam256_120k_legacy test_qam_pwm test_pwm_pulse test_pwm_frames test_pwm_protocol`。

## 2026-11-09 最终验证记录

- 暂停后重新核对工作区，与暂停前的整合结果一致；未覆盖其他本地 QAM 修改。
- 最终适用回归共287项：286通过，1项原有失败，无导入错误。覆盖全部PWM与QAM功能测试、12项布局测试和6项整合专项。
- 唯一失败是 `test_config5p8.Config5p8Tests.test_generated_script_fidelity`：已有生成头文件中的两处源SHA256摘要与当前源文件不一致。寄存器操作表相同；整合前已复现，本次未改动射频配置。
- `test_rx706.py`、`test_rx706_gui.py` 须由专用C库运行器启动，不适用于普通unittest发现。仓库的 `tests/run_network_capture.sh` 在本机macOS因缺少Linux的 `MAP_FIXED_NOREPLACE` 无法编译，须在Linux补跑；不能记为通过。
- `git diff --check` 和host/tests共85个Python文件语法检查通过。没有连接板卡或执行真实射频采集；新版PWM原报告的问题仍未修复。
- 结果保留在当前本地工作区，未暂存、提交或推送。Git HEAD仍为9a825cb，远端跟踪为9659f9e；代码内容已整合，但未创建提交。

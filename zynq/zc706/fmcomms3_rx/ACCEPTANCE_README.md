# RIS 验收收端

此分支整理三功能接收界面、板卡采集桥接、PWM 自动搜索与 CRC12、256QAM 自动捕获/相位恢复，以及对应配置和软件回归。保留原板卡控制入口。

## 环境与启动

需要 Python 3.11 或更新版本、Tk、NumPy 和 Matplotlib。依赖版本见 `host/requirements-fft.txt`。Linux GUI 需要桌面会话或 Xvfb；macOS 可使用带 Tk 的 Python。仅历史中央窗冻结回放额外需要 SciPy，普通三功能接收流程不依赖它。

在仓库根目录运行：

```sh
python3 -m pip install -r zynq/zc706/fmcomms3_rx/host/requirements-fft.txt
python3 zynq/zc706/fmcomms3_rx/host/rx706_receiver_gui.py
```

启动时不自动连接板卡、采集或解调。可通过界面打开自己的 BIN，再加载与实际发送端一致的配置。新整理的配置在 `host/frame_profiles/acceptance/`；原始大 BIN、采集元数据、实验报告、备份和缓存不在本次提交中。

- 三功能操作与结果边界：[RECEIVER_WORKFLOWS_README.md](host/RECEIVER_WORKFLOWS_README.md)。
- 板卡连接、采集、校验与载入：[RECEIVER_CAPTURE_README.md](host/RECEIVER_CAPTURE_README.md)。
- 配置列表：[frame_profiles/acceptance/README.md](host/frame_profiles/acceptance/README.md)。

## 软件回归

在仓库根目录运行接收机主机软件回归：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 zynq/zc706/fmcomms3_rx/tests/run_receiver_acceptance.py
```

测试包含合成 IQ、真实 Tk 控件和内存模拟板卡，不连接真实硬件。旧 C/ctypes 网络测试、独立采集包测试及 FPGA 寄存器配置测试有各自运行入口，不包含在这套主机软件回归中。QAM 测试的发端合成辅助函数和 JSON 夹具已放入仓库，不再依赖原机器仓库外的文件。

`tests/verify_pwm_json_gui.py` 与 `tests/verify_pwm_json_mismatch_gui.py` 是需显式指定采样文件和配置的回放检查。`tests/verify_pwm_symbols78_iq.py` 是合成 IQ 的完整搜索检查，单独运行会产生临时合成数据。它们不属于上述验收入口的默认测试范围。

## 验收边界

软件回归通过不等于真实射频验收通过。真实板卡连接、现场采集和未知净荷 BER 仍需匹配的硬件及发端记录。QAM EVM 不能替代 BER，PWM 参考 BER 不包含擦除。请勿将默认配置、合成净荷或某次捕获绑定的回放参数冒充新的实测真值。

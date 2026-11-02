# 5.8 GHz 配置到 AD9361

已将用户提供的 `C:\Users\19342\Desktop\config_5p8` 原样保存为 `config/config5p8`，通过现有脚本执行器生成初始化表，并构建 `export/key_ctrlll_5p8.elf`。2568 条操作全部保留；FMCOMMS3 晶体/DCXO、PS SPI、掩码写和进入 FDD 的板级适配沿用现有逻辑。

## 配置结果

- RX/TX LO：5800 MHz（从脚本寄存器独立解码验证，固件启动时再次读回检查）。
- RX1A/RX2A，双通道 LVDS DDR，Fs=15.36 MS/s，原 I1/Q1/I2/Q2 格式。
- MGC，索引 76；TX 衰减 20 dB，正常接收时 TX BIST 关闭。
- 源码默认配置改为 CONFIG5P8；原 915/2.4 GHz 脚本与独立 ELF 保留。
- LO 预期值、读回报告和采集 JSON 支持 64 位频率，避免 5800000000 被截成 1505032704。
- 本次未重建 PL，复用已有 `export/zc706_top.bit`。构建脚本不连接板卡，也不写 QSPI。

## JTAG 临时运行

1. 停止当前采集/下载并断开上位机连接。ZC706 上电，连接 JTAG，打开 115200 8N1 串口查看启动日志。
2. 确认 Vivado Hardware Manager 的 hw_server 已启动于 localhost:3121；避免其他调试会话同时控制 CPU。
3. 在 PowerShell 运行：

```powershell
cd D:\TDCM\ad9361\zynq\zc706\fmcomms3_rx
& "D:\software\vivado\xilinx\SDK\2018.3\bin\xsct.bat" scripts/program_config5p8.tcl run
```

该脚本会复位 PS、下载现有 BIT、执行匹配的 ps7_init，再下载并运行新 ELF；当前板上未保存的采样会丢失。本步骤不写 QSPI，断电后不会保留该程序。命令省略 `run` 则下载后 CPU 停在入口，初始化尚未执行。

也可在 SDK/Vitis 的调试下载配置中使用已有 BIT 与 `export/key_ctrlll_5p8.elf`，执行相同 PS 初始化后运行 Cortex-A9 #0。不要只把原始 config 文件拷到前端；它由板上应用在启动时通过 SPI 执行。

## 判断是否配置成功

串口应出现类似：

```text
CONFIG5P8: GUI 2.1.4 LTE10, REF=40 MHz, RX/TX LO=5800 MHz
RF SCRIPT LO readback: RX=5800000000 TX=5800000000 Hz; ...
CONFIG5P8 PASS: operations=2568/2568 ...
CONFIG5P8 ACTIVE: RX/TX=5800 MHz, Fs~15.36 MSPS, TX BIST OFF ...
```

寄存器计算值允许小范围整数舍入差异。还需检查 PLL 锁定、PN 扫描和恢复后的接口验证通过；未通过时，不能只根据 ACTIVE 文本认定可用。初始化完成后重新连接原前端，板卡 IP/端口和下载方式不变。

新下载文件 JSON 应包含：

```json
{"rx_lo_config_hz":5800000000,"rf_profile":"CONFIG5P8","sample_rate_hz":15360000}
```

JSON 是配置标签，实际寄存器 LO 以启动读回为准。旧文件保留原来的 2.4 GHz 元数据，不能通过修改 JSON 把它们变成 5.8 GHz 采样。

## 射频验证与后续解调

先暂时固定超表面一个状态，矢网点频设为 5801 或 5802 MHz。相对 5800 MHz LO，目标应分别出现在约 +1 或 +2 MHz（符号按实际 IQ 约定确认）。先检查 RF 关/开对比，再检查频率改变 1 MHz 时目标谱峰是否跟随移动约 1 MHz。保证接收无削顶。

不建议首次验证让矢网和接收 LO 都恰为 5800 MHz：目标将落在直流附近，AD9361 的直流校正及前端软件去直流均可能影响该信号。若超表面实验必须严格在 5800 MHz，建议另导出接收 LO=5798 或 5799 MHz 的脚本，保留约 2 或 1 MHz 数字中频；对应配置常量、读回检查和 JSON 也必须同时适配，不可只改频率标签。

确认接收正确后，再采固定全 0、固定全 1、10 Hz 全面切换及 half 图案切换；原始 15.36 MS/s IQ 保留。先数字移频、低通，再按目标带宽降采样。当前 10 Hz 试验的窄带参数不能直接用于最终 3 MHz 切换。

关于近直流 CW 与校正的关系：[ADI 官方说明](https://ez.analog.com/rf/wide-band-rf-transceivers/design-support/a/documents/DO26515/ad936x_dcoffset_issue)。

## 重新构建和回退

当前 Windows 环境可复现构建：

```powershell
python3.14 scripts/build_windows.py --profile 5800
```

脚本使用本机 SDK 2018.3 ARM GCC 7.3.1，从保留的 BSP/lwIP 源码构建静态库，输出日志、链接 map 和哈希清单到 `reports/config5p8_20261028`，发布独立 `export/key_ctrlll_5p8.elf`，不覆盖旧通用 ELF 或 2.4 GHz ELF。若本机工具安装位置不同，可传 `--toolchain`、`--make`。

原 Linux/Vitis 构建入口 `build_rx_network.sh` 也已增加 config5p8 表生成，依照源码默认构建。发布脚本按所选默认配置命名独立 ELF，避免把 5.8 GHz 结果误覆盖为 `_2p4.elf`。

回到 2.4 GHz 时可下载原 `export/key_ctrlll_2p4.elf`；重新编译则在 rf_profile.h 中选择 RX_RF_PROFILE_2P4，或用 Windows 构建命令 `--profile 2400`。不要用新 5.8 GHz ELF 覆盖旧固化启动镜像，直到实际上板接收验证完成。

## 已完成与待验证

已完成：全应用与 BSP/lwIP ARM 编译链接；三项 5.8 GHz 脚本/寄存器/时钟接口测试；五项原 2.4 GHz 回归；真实 C STATUS 序列化在 915/2400/5800 三配置下运行通过；15 项 IQ 数值测试通过（含 5.8 GHz 元数据）。有两条既有 lwIP 源码的可能未初始化变量编译警告，详见对应构建日志，应用链接无错误。

2026-10-28 已完成实际 JTAG 下载运行：RX/TX LO 读回均为 5800000000 Hz、初始化 PASS、PN 1248/1248 PASS、接口检查 20/20 PASS，TCP STATUS 确认 CONFIG5P8。见 reports/onboard_5p8_20261028/RESULT.md 及原始日志。未进行 QSPI 固化或外部 5.8 GHz 射频采样，实际无线链路与解调仍需验证。

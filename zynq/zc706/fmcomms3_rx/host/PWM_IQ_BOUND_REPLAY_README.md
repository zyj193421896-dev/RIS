# PWM 原始IQ绑定JSON一键复放

此功能只读本地采集，不连接板卡、不改变射频或采样设置。QAM/TDCM原入口保持不变。

## 界面

打开接收信号分析窗口，在“分析参数 → 信号解调模式”中：

- **载入PWM IQ绑定JSON并分析…**：后台校验完整IQ的字节数、SHA-256及同名采集元数据，随后自动设置“PWM（占空比）/脉冲宽度与时延”，恢复并显示频谱、星座、波形及诊断。
- **保存本次PWM IQ绑定JSON…**：保存上次完成的分析快照，后续编辑的控件不会改变该快照。哈希在后台计算；不会覆盖已有JSON。普通手工分析导出时，未核验净荷参考默认关闭；再次保存已加载的绑定JSON时保留其确认状态。
- “PWM 符号表 / 比特流…”继续提供CSV、比特流及逐块JSON导出。逐块JSON包含不依赖参考的净荷擦除计数及假设提示。

校验或解调失败时不保留旧结果。重复载入在工作期间被阻止；“停止”可取消哈希及后续处理，关闭窗口不会再更新界面。开始分析和完成复核均验证输入；文件或同名元数据在过程中改变时丢弃结果。

完成分析不代表解码可用。无参考时显示“BER未测”；总净荷擦除独立统计，不能把参考覆盖范围内的“擦除0位”解释为接收没有擦除。

## 离线复放

```sh
python -B host/rx706_pwm_replay.py /absolute/path/capture_replay.json
python -B host/rx706_pwm_replay.py /absolute/path/capture_replay.json --output /absolute/path/new_diagnostics.json
```

输出文件使用排他创建，不覆盖已有文件。CLI和GUI均调用同一个`run_recipe`；锁定失败仍可输出诊断，但输入绑定失败、侧车Fs冲突或数据变更会停止。诊断JSON含净荷总数、已判决数、擦除数、同步状态计数、逐块指标及前导校准，不包含17万行逐行参考数组。

## API与JSON

`make_recipe(path, pulse_options, *, sample_rate_hz, symbol_hz, channel=1, first_tuple=0, duration_s=1., center_hint_hz=None, decimation=1, state_one='auto', width_tolerance=.3, hysteresis=.10, glitch_tolerance_chips=.15, nfft=65536, averages=16, remove_dc=True, assumptions=None, cancel_event=None)`生成并校验配置。

- `read_recipe(path)`读取至多512KiB JSON。
- `validate_recipe(dict)`校验并规范化，不读取IQ或外部参考文件。
- `verify_input(recipe, cancel_event=None)`验证完整输入及同名元数据，返回路径。
- `run_recipe(recipe_or_path, cancel_event=None, progress=None)`返回`(spectrum, recovery)`。
- `result_summary(spectrum, recovery)`返回可JSON序列化的诊断。

顶层字段：

```json
{
  "version": 1,
  "method": "iq_bound_pwm_replay",
  "input_path": "/absolute/path/capture.bin",
  "input_fingerprint": {"file_bytes": 122880000, "sha256": "完整文件的小写64位SHA-256"},
  "input_format": {"dtype": "int16", "byte_order": "little", "order": ["I1", "Q1", "I2", "Q2"]},
  "spectrum": {"sample_rate_hz": 15360000, "channel": 1, "first_tuple": 0, "nfft": 65536, "averages": 16, "remove_dc": true},
  "recovery": {
    "duration_s": 1,
    "symbol_hz": 169963.9,
    "center_hint_hz": null,
    "decimation": 1,
    "state_one": "auto",
    "width_tolerance": 0.3,
    "hysteresis": 0.1,
    "glitch_tolerance_chips": 0.15,
    "pulse_options": {"说明": "使用validate_pulse_options参数，必须包含完整frame_options/users和实际码表"}
  },
  "assumptions": {"preamble_status": "hypothesis", "reference_status": "unverified", "notes": ["前导、帧长、码表须与发端核验"]},
  "caveat": "净荷参考不参与恢复；假设不是确认真值"
}
```

以上是字段示意，不是可直接运行的配置。`pulse_options`保存步长、最小宽度、低电平预留、完整码表、联合时延开关、容差、最近码字策略，及内嵌`frame_options`。每个用户保存名称、身份序列、净荷长度及可选参考。恢复时长支持1～1000ms；范围不能超出绑定文件，处理后每片至少3采样点。

前导/帧/码表未核验时必须使用`preamble_status="hypothesis"`。净荷参考默认`reference_enabled=false`；只有明确标注`reference_status="confirmed"`才允许启用计分。`@外部参考文件`不被接受，保证复放自包含。即使参考已确认，运行器也先清空所有净荷参考，再进行恢复，最后才调用独立评分器。参考不参与同步、时钟、校准、极性、码字选择或擦除回填。

## 验证

```sh
PYTHONPATH=host:tests python -B -m unittest test_pwm_iq_replay test_qam64_controls test_pwm_frames_gui test_pwm_gui test_iq_bound_replay
```

新增测试包含完整SHA、缺失/变更输入、侧车Fs冲突、非法配置、外部参考拒绝、取消、恢复期间数据变更、参考前后判决完全一致、GUI后台加载和重复操作、失败清空、快照导出及禁止覆盖。真实Tk布局测试另外使用`test_gui_layout`等既有测试。

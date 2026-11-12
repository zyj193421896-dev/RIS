# PWM 离线参数候选搜索

新增 `rx706_pwm_autotune.py`，复用现有恢复接口，不改变原解码器默认行为或原始 frame.json。GUI 增加可操作的自动接收调试闭环。

## GUI：无需代理的实验流程

1. 选择本次完整 bin，载入本次发送帧 JSON。前导、码表及帧组合由用户固定，本程序不搜索前导。
2. 点击“自动接收调试 → 全文件码率 / BER”。后台显示每个候选的覆盖和身份统计；可用停止按钮取消，重复点击不会再启动线程。
3. 程序自动应用最佳极性、名义码率和容差，并对完整文件正常解码。原 JSON 的参考只在候选排序冻结后参与 BER / BLER 统计，不参与选择。
4. 统计页显示本文件跟踪实测码元率、码表信息容量、扣除身份/校准后的净荷容量上限、完整净荷判决擦除和参考覆盖 BER / BLER。无参考时 BER 未测，但独立判决擦除仍显示，不能误报零擦除。输出标签位不等于独立信息位。
5. 点击“保存本文件固定接收方案…”，保存 JSON 及同名 `.search.json`。固定 JSON 包含极性、码率、容差、输入hash、原参考和完整接收设置；重新载入可复现，无需再次由代理调参。输入/接收参数变化后旧搜索结果失效，不能应用过期线程结果。

在 `zynq/zc706/fmcomms3_rx` 目录运行：

```sh
python host/rx706_pwm_autotune.py capture.bin \
  --profile actual_frame.json --output reports/my_pwm_search
```

输出目录必须不存在。生成 `search.json`（参数、排名、逐候选结果和复现实验清单），有可用且极性不冲突的候选时生成 `selected_frame.json`。CLI 的该副本禁用参考统计，并绑定当前输入 SHA256，原 profile 不改；同时持久化 `state_one` 极性，完整候选也在 `selection.candidate` 中。

CLI 在候选排序冻结后对已固定判决做独立 `post_selection_evaluation`，给出实测码元率、信息容量、完整净荷判决擦除和按原 JSON 参考比对的 BER / BLER及分母。参考未独立认证，不能称为认证BER；没有参考时 BER 为 null。后评分不更改判决，也不反馈排名。GUI 固定方案则保留用户原参考，便于正常载入后重复后评分。

## 输入边界

- 必须提供完整 `.bin.json` 采集元数据，支持 little-endian I1/Q1/I2/Q2 int16、12-bit、双通道。校验完整标志、长度、可用的 chunk CRC32，并计算 SHA256。采样率从元数据读，禁止猜测。
- 帧 profile 必须明确实际 identity、各用户 payload_symbols、codebook、symbol_khz。可用当前 GUI 导出的 profile，但应确认是本次发送帧。明确提供 profile 是用户的帧假设，不证明发送端真的如此。
- 只给 bin 无法推断任意新前导、码表或标签。没有 profile 时直接报告需要实际帧，不套用历史配置。若采集 sidecar 中有 `pwm_search_profile: {path, source_sha256, profile_sha256}`，可自动使用经双哈希校验的绑定 profile；这证明绑定一致，不证明发送内容真实性。
- 本版本对不超过 2 秒的完整文件做最终回放。更长文件明确拒绝，不静默截断。失败、超时不发布最终配置。

## 候选与选择

默认 12 个以内候选：A/B 极性 × profile 名义率及 ±100ppm × 原边沿容差与 0.49chip。只搜索接近实际设计速率的小范围，不声称能发现任意速率或全局最优；码表、前导、保留校准位置不搜索。先扫描前 30ms，再用文件前最多 1 秒比较前 2 名和最佳相反极性（若能获取），最多 3 次候选恢复。超过 1 秒时冻结胜出参数，再单独回放整个文件；最终 BER、擦除和净荷速率按完整分析区间统计，不把前 1 秒冒充全文件。`selection_record` / `selection_interval` 记录选参区间，`full_record` 仅记录覆盖全文件的回放，`analysis_interval` 明确全文件时长。1 秒以内保持原候选与全文件回放路径。可用 `--scan-seconds`、`--finalists`、`--max-candidates`、`--max-seconds` 控制。时限在候选之间检查，不是强制打断单次恢复；Python API 支持 cancel_event，CLI 可 Ctrl-C。

所有用户 payload reference 在验证配置和解码前均移除。排名依次考虑可用性（至少 2 完整块、50%有效覆盖和身份覆盖）、按总时长归一化的有效净荷覆盖、身份覆盖、较少竞争前导、较小码字距离；并列时依次较小容差、更接近名义率、频率和极性。全擦除不会获胜，迟获取会被时长分母惩罚。两极性覆盖相差不足 2 个百分点报告 ambiguous，不导出固定配置。指标不包含 BER。

当前 `current_pwm/frame.json` 的四个 `0` 是用户身份前导；`cycle_calibration.payload_indices=[3,4,5,6]` 是原始净荷的四个零基位置，发送符号由 `symbols=[3,4,5,6]` 明确指定。它们是额外保留校准符号，不能计入测试净荷，不能从测试参考序列反推。`average_blocks=64` 是居中窗口，会使用相邻后续块，属于离线校准，不代表实时因果解码。换前导时修改 identity；不用循环校准时删除 cycle_calibration；新帧仍要真实存在所配置的预留码。

重复身份码仍可能存在初始帧相位歧义，结果保留 timing_note。高覆盖不证明初始帧唯一，也不证明 payload 正确。本工具仅同一文件调参和整文件回放，不是独立验证集。验收还需独立捕获、经发送端确认的帧/码表以及未参与训练的参考净荷。

## 固定回放接口

```python
import json
from rx706_pwm_autotune import search_options, sha256
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2
r = json.load(open('reports/my_pwm_search/search.json'))
p = json.load(open('reports/my_pwm_search/selected_frame.json'))
assert sha256(r['source']) == r['provenance']['source_sha256']
c = r['selection']['candidate']
result = recover_pwm2(analyze_bin(r['source'], channel=r['channel'], nfft=4096, averages=4),
    pulse_options=search_options(p), symbol_hz=c['symbol_hz'], state_one=c['state_one'],
    duration_s=r['provenance']['duration_s'], decimation=1, allow_unlocked=True)
```

回放前还应核对记录的 host Python 文件 SHA256、commit、Python/NumPy/SciPy 版本。报告明确原始帧 hash、输入 hash、候选集合、评分、分母、排除项和命令。不同机器的浮点运算可能产生微小差异。

测试（tests 目录）：`python -m unittest test_pwm_autotune test_pwm_pulse test_pwm_frames`。

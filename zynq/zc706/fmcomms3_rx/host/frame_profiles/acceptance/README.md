# 验收帧配置

这些 JSON 从已有实验配置逐字节复制，不包含原始 BIN、采集元数据、标定结果或设备凭据。
载入后仍须核对本次实际发送端、符号率、前导、码表与净荷，不自动代表硬件验收通过。

- `frame_PWM10_76_BER_0to75.json`：76 符号 PWM，约 170 kHz。仅在发送端确实发送 0～75 参考序列时启用其 BER 统计。
- `pwm_frame_three_users_crc12_10khz.json`：10 kHz、三个顺序用户块、空中 CRC12；不是三个射频通道。
- `rx_256qam_256slice_10khz_10_4_256_config.json`：10 kHz，10 个独立 PWM 前导、4 个四角训练、256 个净荷。
- `rx_256qam_256slice_20khz_10_4_256_config.json`：上述结构的 20 kHz 配置。
- `rx_256qam_256slice_10khz_4_4_256_config.json`：10 kHz，4 个 PWM 前导、4 个四角训练、256 个净荷。
- `rx_256qam_120k_64slice_3_4_16_no_crc_config.json`：历史 120 kHz 参数配置，保留用于兼容性回归。

QAM 未知净荷没有真实参考时 BER 未测；EVM 不等于 BER 或锁定证明。
BIN 需由使用者单独提供，并保留匹配的 `.bin.json` 元数据。

ZC706 / FMCOMMS3 独立采集与二进制下载包（Linux / Python 3）

用途
  在有网口的Linux主机上控制板卡、采集、下载；再把原始数据拷回笔记本分析。
  只使用Python标准库，不需要pip、NumPy、Matplotlib或Vivado，不需要互联网。
  建议Python 3.8及以上。图形窗口可选，需要系统已有Tk及图形桌面；没有时自动进入终端菜单。
  板卡须运行当前支持TCP采集/下载的固件。本工具不烧写BIT/ELF，不修改射频参数或数据格式。

1. 解压并启动
  tar -xzf rx706_capture_linux_20261109.tar.gz
  cd rx706_capture_linux_20261109
  bash start.sh

  有Tk桌面时打开采集下载窗口；没有时打开终端菜单。
  强制使用终端菜单：bash start.sh --cli
  指定板卡地址：bash start.sh --cli --host 192.168.1.10
  zip解压后也可使用bash start.sh，不要求执行权限。

2. 连接板卡
  Linux主机的有线网口与ZC706 RJ45相连。
  将该有线网卡的IPv4设为192.168.1.20，掩码255.255.255.0（/24）。
  直连时网关和DNS留空；不需要改Wi-Fi网卡。
  板卡默认192.168.1.10，控制/数据TCP5001，控制台TCP5000。
  可先运行ping 192.168.1.10确认连通，等待板卡启动完成。
  这里的地址须与板上实际运行固件一致，界面改IP不会修改板卡IP。

3. 图形窗口操作
  点击连接，确认状态；默认采集长度1秒，勾选“采集完成后自动下载”。
  选择.bin保存路径，点击开始采集。
  如果板卡已经有ready冻结数据，直接点“下载当前数据”，不要先释放。
  下载暂停后保留.bin.part与.bin.part.json，按同一输出路径续传。
  本独立包只提供采集下载；信号分析入口已隐藏。

4. 无桌面/无Tk时的终端菜单
  bash start.sh --cli
  1 查询状态；2 采集并下载（时长回车默认1秒）；3 下载当前冻结数据；4 续传。
  5 打开板卡日志/按键控制台；6 查询诊断；7 显式释放；8 显式终止采集；0 退出。
  释放与终止需明确输入确认，不会自动丢弃板上数据。

5. 直接命令行（不使用菜单）
  bash cli.sh status
  bash cli.sh console
  bash cli.sh capture --mib 117.1875 -o captures/rx_1s.bin
  bash cli.sh download -o captures/rx_saved.bin
  bash cli.sh download -o captures/rx_saved.bin --resume
  bash cli.sh --host 192.168.1.10 --port 5001 status

  117.1875 MiB恰好对应当前采样格式下的1秒，3750个32KiB包。
  最大2044 MiB，约17.442133秒；按32KiB完整包对齐。
  刚采集好的数据冻结在DDR；换新文件名可以再次下载。
  下一次采集前必须显式释放旧数据，命令为：bash cli.sh release <capture_id>
  不要在尚未下载完成时释放或重启板卡。CLI capture不填--mib会使用原工具的2044MiB默认值。

6. 拷回笔记本（两个文件一起拷）
  captures/rx_xxx.bin
  captures/rx_xxx.bin.json

  .bin为原始IQ，.bin.json含采样率、采集ID、射频记录、完整性与SHA256。
  不要重命名成不匹配的文件名。未完成的.part不是正式分析输入。
  下载工具保持原有逐块CRC和完整文件SHA256，不会改变样本、裁剪或解调数据。
  原始排列为小端int16 I1,Q1,I2,Q2，双通道，15.36 MS/s。

  搬运前或搬运后可检查：
    python3 -I verify_capture.py captures/rx_xxx.bin
  Windows笔记本也可使用已有Python运行同一个verify_capture.py；保持app/rx706.py一起在包内。
  校验工具流式读取，不会把整GiB文件一次加载到内存。

  回到笔记本，在现有rx706_gui.py的“IQ信号分析”中打开拷回的.bin即可，离线分析不需要连板卡。
  PWM码表、用户前导和已知净荷仍在笔记本原分析GUI配置；此下载包不依赖这些参数。

内容
  start.sh             图形/终端自动入口
  cli.sh               原命令行协议入口
  capture_gui.py       只采集下载的GUI入口
  capture_menu.py      无桌面交互菜单
  verify_capture.py    文件及元数据搬运校验
  app/rx706.py         原协议/CRC/续传/SHA256实现
  app/rx706_gui.py     原采集控制器和界面（保留原样）
  MANIFEST.sha256      包内工具文件校验清单

验证说明
  本包通过本机回环TCP下载、CRC/SHA256、断线续传、拷贝损坏检测及Tk窗口检查。
  未在目标Linux主机或实际板卡网口上实测；实际链路与板卡启动状态须在现场确认。

"""Capture/download terminal menu. Python standard library only, no display."""
import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent/'app'))
import rx706 as protocol

ROOT = Path(__file__).resolve().parent


def capture_bytes(seconds):
    try:
        amount = Decimal(str(seconds))
        if not amount.is_finite() or amount <= 0:
            raise ValueError
        requested = amount*15360000*8
        if requested > protocol.MAX_BYTES:
            raise ValueError
        packets = (requested/protocol.PACKET_BYTES).to_integral_value(rounding=ROUND_CEILING)
        return int(packets)*protocol.PACKET_BYTES
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError('采集时长须大于0，且不超过%.6f秒' % (protocol.MAX_BYTES/(15360000*8))) from None


def output_path(resume=False):
    default = ROOT/'captures'/('rx_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bin')
    value = input('原下载文件路径（.bin，不是.part）：' if resume else '保存路径（回车使用%s）：' % default).strip()
    if resume and not value:
        raise ValueError('续传须填写首次下载使用的.bin路径')
    path = Path(value).expanduser() if value else default
    if path.name.endswith('.part'):
        path = path.with_name(path.name[:-5])
    return path


def capture_and_download(client, seconds, output, note=''):
    count = capture_bytes(seconds)
    output = Path(output)
    for path in (output, output.with_suffix(output.suffix+'.json'),
                 output.with_name(output.name+'.part'), output.with_name(output.name+'.part.json')):
        if path.exists():
            raise FileExistsError('输出或部分文件已存在，请换新路径或选择断点续传：%s' % path)
    state = client.request('STATUS')
    if state.get('state') == 'ready':
        raise ValueError('板卡已有冻结数据。请先下载；确认不再需要后显式释放，再开始新采集。')
    client.request('CAPTURE %d' % count)
    deadline = time.monotonic()+90
    while True:
        state = client.request('STATUS')
        if state.get('state') != 'capturing':
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('采集等待超时。数据仍在板卡，先查询状态，不要重复启动采集。')
        time.sleep(.2)
    protocol.validate_capture(state)
    return protocol.download(client, output, note=note)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.1.10')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--console-port', type=int, default=5000)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    print('ZC706独立采集下载；板卡%s:%d；无需pip、NumPy或图形桌面。' % (args.host, args.port))
    while True:
        print('\n1 查询状态  2 采集并下载  3 下载当前冻结数据  4 断点续传\n'
              '5 板卡控制台  6 下载诊断信息  7 释放已下载的数据  8 终止采集  0 退出')
        try:
            choice = input('选择：').strip()
            if choice in ('0', 'q', 'exit'):
                return 0
            if choice == '5':
                protocol.console(args.host, args.console_port, args.timeout)
                continue
            if choice not in ('1', '2', '3', '4', '6', '7', '8'):
                print('请输入菜单数字。')
                continue
            with protocol.Client(args.host, args.port, args.timeout) as client:
                if choice in ('1', '6'):
                    import json
                    print(json.dumps(client.request('STATUS' if choice == '1' else 'DEBUG'), ensure_ascii=False, indent=2))
                elif choice in ('2', '3', '4'):
                    seconds = None
                    if choice == '2':
                        seconds = input('采集时长，单位秒（回车=1）：').strip() or '1'
                    if seconds is not None:
                        count = capture_bytes(seconds)
                        print('实际采集%d字节，约%.6f秒。' % (count, count/(15360000*8)))
                    output = output_path(resume=choice == '4')
                    saved = (capture_and_download(client, seconds, output) if choice == '2' else
                             protocol.download(client, output, resume=choice == '4'))
                    print('下载完成：%s\n同时保留：%s.json\nSHA256：%s' % (output, output, saved['sha256']))
                elif choice == '7':
                    state = client.request('STATUS')
                    capture_id = state.get('capture_id', '')
                    answer = input('释放会丢弃板上冻结数据。若已保存完毕，输入采集ID %s 确认：' % capture_id).strip()
                    if capture_id and answer == capture_id:
                        print(client.request('RELEASE '+capture_id))
                    else:
                        print('未释放。')
                elif choice == '8':
                    if input('终止后本次未完成采集不可下载。输入ABORT确认：').strip() == 'ABORT':
                        print(client.request('ABORT'))
        except EOFError:
            return 0
        except KeyboardInterrupt:
            print('\n操作中断；不会自动释放板上数据。重新查询状态，可按原路径续传。')
        except (OSError, ValueError, KeyError, protocol.ProtocolError) as exc:
            print('操作失败：%s' % exc)


if __name__ == '__main__':
    raise SystemExit(main())

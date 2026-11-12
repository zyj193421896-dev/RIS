"""Order-dependent square-QAM profiles; no assumed transmitter frame length."""
import math


def four_corner_labels(order):
    """Lower-left, upper-left, upper-right, lower-right in I-then-Q Gray."""
    if order not in (16,64,256):raise ValueError('QAM阶数须为16、64或256')
    side=math.isqrt(order);positive=(side-1)^((side-1)>>1)
    return [0,positive,positive*side+positive,positive*side]


def four_corner_hex(order):
    return ' '.join(f'{v:02X}' for v in four_corner_labels(order))


def four_corner_payload(order,total_windows,previous_training=0,explicit_payload=''):
    """64QAM has no established over-the-air frame: require an explicit count."""
    four_corner_labels(order)
    if str(explicit_payload).strip():
        try:payload=int(explicit_payload)
        except (ValueError,TypeError):raise ValueError('四角训练净荷数须为整数') from None
    elif order==256:
        payload=max(1,int(total_windows)-int(previous_training))
    else:
        raise ValueError(f'{order}QAM四角训练须填写实际每帧净荷数；阶数不能决定帧长')
    if not 1<=payload<=65532:raise ValueError('四角训练每帧净荷数须为1～65532')
    return payload


def mapping_note(order):
    if order not in (16,64,256):raise ValueError('QAM阶数须为16、64或256')
    width=int(math.log2(order));axis=width//2
    return (f'{order}QAM：每符号{width}个有效bit（'+ 'I'*axis+'Q'*axis+
            f'，I后Q、各轴MSB优先、Gray标签）；HEX 00～{order-1:02X}为单符号存储值。'+
            ('64QAM高2位必须为0；两位HEX不表示每符号8个净荷bit，未进行跨符号比特打包。' if order==64 else ''))


def four_corner_training_hex(order,repetitions=1):
    if repetitions not in (1,2):raise ValueError('四角训练遍数须为1或2')
    return ' '.join(f'{v:02X}' for v in four_corner_labels(order)*repetitions)


def corner_repetitions(labels,order):
    values=list(labels)
    return next((n for n in (1,2) if values==four_corner_labels(order)*n),0)

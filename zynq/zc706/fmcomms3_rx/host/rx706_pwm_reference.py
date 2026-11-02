"""Compare PWM decisions to a known cycle without correcting receive data.

Only the initial synchronization window selects the cycle offset. Its decisions
are excluded from BER. Erasures retain their positions; no later realignment.
"""
from dataclasses import dataclass
import re
import numpy as np


def parse_reference(text, code_chips=8):
    """Accept '7642' for 8 symbols, or separated decimal symbol values."""
    text = str(text).strip()
    if not text:
        raise ValueError('请填写已知循环符号，例如 7 6 4 2')
    if not re.fullmatch(r'[0-9,，;；\s]+', text):
        raise ValueError('参考序列只能包含十进制符号值及空格/逗号分隔符')
    tokens = [v for v in re.split(r'[,，;；\s]+', text) if v]
    if len(tokens) == 1 and code_chips <= 10:
        tokens = list(tokens[0])
    if not 1 <= len(tokens) <= 256:
        raise ValueError('参考循环长度须为 1～256 个符号')
    sequence = tuple(int(v) for v in tokens)
    if any(v < 0 or v >= code_chips for v in sequence):
        raise ValueError(f'参考符号值必须在 0～{code_chips-1} 内；它不是时间片或比特串')
    for length in range(1, len(sequence)+1):
        if len(sequence) % length == 0 and sequence[:length]*(len(sequence)//length) == sequence:
            return sequence[:length]
    return sequence


@dataclass
class ReferenceCheck:
    sequence: tuple
    sync_symbols: int
    sync_valid: int
    sync_matches: int
    offset: object
    reason: str
    expected: object
    row_status: object
    row_bit_errors: object
    evaluation_symbols: int = 0
    compared_symbols: int = 0
    symbol_errors: int = 0
    bit_errors: int = 0
    compared_bits: int = 0
    erasures: int = 0
    warned_symbols: int = 0
    warned_mismatches: int = 0
    bitwords: object = None
    training_symbols: int = 0

    @property
    def locked(self):
        return self.offset is not None

    @property
    def ber(self):
        return self.bit_errors/self.compared_bits if self.compared_bits else None

    def summary(self):
        sequence = '→'.join(map(str, self.sequence))
        if not self.locked:
            return f'参考循环 {sequence} 未同步：{self.reason}；不报告 BER。'
        ber = f'{self.ber:.6g}' if self.ber is not None else '不可计算（无已判决比特）'
        return (f'参考循环 {sequence}；首个接收符号对应 {self.expected[0]}；'
                f'前 {self.sync_symbols} 个符号仅用于同步（{self.sync_matches}/{self.sync_valid} 匹配），不计入 BER。'
                f'\n统计 {self.evaluation_symbols} 个符号；已判决 {self.compared_symbols}，'
                f'错符号 {self.symbol_errors}；错比特 {self.bit_errors}/{self.compared_bits}，'
                f'BER（仅已判决比特）={ber}；'
                f'擦除 {self.erasures}/{self.evaluation_symbols}（{self.erasures/max(1,self.evaluation_symbols):.3%}，单列且不计入 BER 分母）。'
                f'\n全片段可判决毛刺告警 {self.warned_symbols} 个，其中参考不匹配 {self.warned_mismatches} 个。'
                '接收符号和 ? 保持原判决；参考序列不参与纠错。'
                + (f' 另排除20%训练符号{self.training_symbols}个，不计擦除和BER。' if self.training_symbols else '')
                + (f' PWM version2误码按中间{len(self.bitwords[0])}片码型计算，不代表同等数量的独立净荷比特。' if self.bitwords is not None else ''))


def check_reference(frames, sequence, bits_per_symbol=3, sync_symbols=64, symbol_bitwords=None):
    """Align once, compare valid symbols, and count erasures separately."""
    if not isinstance(sync_symbols, (int, np.integer)) or not 8 <= sync_symbols <= 4096:
        raise ValueError('同步窗须为 8～4096 个符号')
    limit = 254 if symbol_bitwords is not None else 8
    if not isinstance(bits_per_symbol, (int, np.integer)) or not 1 <= bits_per_symbol <= limit:
        raise ValueError(f'每符号码型长度须为 1～{limit}')
    if symbol_bitwords is not None:
        words = tuple(symbol_bitwords)
        if not words or len(words) > 256 or len(set(words)) != len(words) or any(
            len(w) != bits_per_symbol or set(w)-{'0', '1'} for w in words):
            raise ValueError('参考码型必须互异且为指定长度的0/1字符串')
        alphabet = len(words)
        encoded = [int(w, 2) for w in words]
    else:
        words = None
        alphabet = 1 << bits_per_symbol
        encoded = np.arange(alphabet)
    sequence = parse_reference(' '.join(map(str, sequence)), alphabet)
    n = len(frames)
    if n and np.any(np.diff(frames['index']) != 1):
        raise ValueError('参考校验要求保留连续符号位置，包括擦除')
    result = ReferenceCheck(sequence, sync_symbols, 0, 0, None, '',
                            np.full(n, -1, dtype=np.int16),
                            np.full(n, '未同步', dtype='U16'),
                            np.full(n, -1, dtype=np.int16))
    result.bitwords = words
    training = frames['training'] if 'training' in frames.dtype.names else np.zeros(n, dtype=bool)
    result.training_symbols = int(training.sum())
    positions = np.cumsum(~training)-1
    sync_window = ~training & (positions < sync_symbols)
    if np.count_nonzero(~training) <= sync_symbols:
        result.reason = '片段不足以同时提供同步窗和独立统计窗，请增加恢复时长'
        return result
    valid = frames['valid'] & (frames['symbol'] >= 0) & (frames['symbol'] < alphabet)
    valid &= ~training
    result.sync_valid = int(valid[sync_window].sum())
    if result.sync_valid < max(8, 2*len(sequence)):
        result.reason = '同步窗内可判决符号不足，请增加同步窗或改善接收质量'
        return result
    pattern = np.asarray(sequence)
    observed = frames['symbol']
    sync_compared = valid & sync_window
    scores = np.array([np.mean(observed[sync_compared] ==
                               pattern[(positions+phase) % len(pattern)][sync_compared])
                       for phase in range(len(pattern))])
    phase = int(np.argmax(scores))
    best = float(scores[phase])
    runner = float(np.partition(scores, -2)[-2]) if len(scores) > 1 else 0.
    result.sync_matches = int(round(best*result.sync_valid))
    if best < .75 or best-runner < .20:
        result.reason = f'循环相位不可靠（最佳匹配 {best:.1%}，与次优差 {best-runner:.1%}）'
        return result
    result.offset = phase
    result.expected[~training] = pattern[(positions[~training]+phase) % len(pattern)]
    evaluation = ~training & (positions >= sync_symbols)
    compared = evaluation & valid
    mismatch = valid & (frames['symbol'] != result.expected)
    result.row_status[:] = np.where(valid, np.where(mismatch, '不匹配', '匹配'), '擦除')
    result.row_status[sync_window] = np.char.add('同步/', result.row_status[sync_window])
    result.row_status[training] = '训练/不统计'
    if words is not None:
        # Alphabet-sized distance matrix avoids an exponential 2**word_length
        # lookup and supports codewords longer than a machine integer.
        distances = np.array([[(a ^ b).bit_count() for b in encoded] for a in encoded], dtype=np.int16)
        result.row_bit_errors[valid] = distances[frames['symbol'][valid], result.expected[valid]]
    else:
        lookup = np.array([i.bit_count() for i in range(1 << bits_per_symbol)], dtype=np.int16)
        result.row_bit_errors[valid] = lookup[encoded[frames['symbol'][valid]] ^ encoded[result.expected[valid]]]
    result.evaluation_symbols = int(evaluation.sum())
    result.compared_symbols = int(compared.sum())
    result.symbol_errors = int((compared & mismatch).sum())
    result.bit_errors = int(result.row_bit_errors[compared].sum())
    result.compared_bits = result.compared_symbols*bits_per_symbol
    result.erasures = int((evaluation & ~valid).sum())
    warned = valid & (frames['glitch_count'] > 0)
    result.warned_symbols = int(warned.sum())
    result.warned_mismatches = int((warned & mismatch).sum())
    return result

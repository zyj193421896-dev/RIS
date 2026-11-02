"""Causal width-offset calibration from three equal 20%-duty pilot symbols."""
import math
import numpy as np
from rx706_spectrum import _cancel


TRAINING_FIELDS = [('raw_reason', 'U64'), ('structure_ok', '?'), ('training', '?'),
                   ('calibration_id', 'i4'), ('width_correction_chips', 'f8'),
                   ('corrected_high_chips', 'f8'), ('corrected_one_us', 'f8'),
                   ('corrected_zero_us', 'f8')]


def validate_training(total, prefix, code, options):
    target = total*.2
    if abs(target-round(target)) > 1e-8 or not prefix <= round(target) <= prefix+code:
        raise ValueError('20%训练宽度必须为整数片，且位于当前version2可用高片数范围内')
    tolerance = float(options.get('same_tolerance_chips', .5))
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('三连同宽的极差上限须大于0且小于1片')
    return int(round(target)), tolerance


def apply_training(frames, total, prefix, code, guard, width_tolerance, options, cancel_event=None):
    """Use only past/current pilot samples; payload values never fit an offset.

    Every stable triple is a pilot by protocol. Repeated equal payload triples
    are consequently ambiguous and must be avoided by the transmitter.
    """
    target, same_tolerance = validate_training(total, prefix, code, options)
    idle = frames['symbol'] == -2
    frames['raw_reason'] = frames['reason']
    frames['corrected_high_chips'] = np.nan
    frames['corrected_one_us'] = np.nan
    frames['corrected_zero_us'] = np.nan
    frames['width_correction_chips'] = np.nan
    frames['valid'] = False
    frames['symbol'] = -1
    frames['bits'] = '?'*code
    raw = frames['high_chips'].copy()
    usable = frames['structure_ok'] & np.isfinite(raw) & (raw > 0) & (raw < total)
    correction, identifier, armed, last_pilot = None, 0, True, None
    pilots = []

    def corrected(at, value, calibration_id):
        row = frames[at]
        period_us = float(row['one_us']+row['zero_us'])
        high = float(raw[at]+value)
        row['calibration_id'] = calibration_id
        row['width_correction_chips'] = value
        row['corrected_high_chips'] = high
        row['corrected_one_us'] = high/total*period_us
        row['corrected_zero_us'] = period_us-row['corrected_one_us']
        return high

    for i in range(len(frames)):
        if i % 2048 == 0:
            _cancel(cancel_event)
        if not armed and (not usable[i] or abs(raw[i]-last_pilot) > same_tolerance):
            armed = True
        if idle[i]:
            frames['symbol'][i], frames['bits'][i] = -2, '-'
            frames['reason'][i] = '空闲：不输出数据比特'
            continue
        is_pilot = (armed and i >= 2 and np.all(usable[i-2:i+1]) and
                    np.ptp(raw[i-2:i+1]) <= same_tolerance)
        if is_pilot:
            observed = float(np.median(raw[i-2:i+1]))
            correction = target-observed
            identifier += 1
            last_pilot, armed = observed, False
            pilots.append(dict(id=identifier, first_row=i-2, last_row=i,
                               start_s=float(frames['start_s'][i-2]),
                               measured_high_chips=observed,
                               correction_chips=correction))
            for j in range(i-2, i+1):
                corrected(j, correction, identifier)
                frames['training'][j] = True
                frames['valid'][j] = False
                frames['symbol'][j] = -3
                frames['bits'][j] = '-'
                frames['reason'][j] = '20%三连训练：仅校准，不输出净荷'
            continue
        row = frames[i]
        if correction is None:
            row['reason'] = '尚未检测到20%三连训练段'
            continue
        high = corrected(i, correction, identifier)
        reason = ''
        if not usable[i]:
            reason = '原始边沿/跳变不完整，不能用脉宽补偿修复'
        elif not 0 < high < total:
            reason = '补偿后脉宽超出符号周期'
        else:
            number = int(math.floor(high+.5))
            if not prefix <= number <= prefix+code:
                reason = '补偿后高片数超出编码范围'
            elif abs(high-number) > width_tolerance:
                reason = '补偿后脉宽仍偏离整数时间片'
            elif total-high < guard-width_tolerance:
                reason = '补偿后末尾低电平保护不足'
            if not reason:
                k = number-prefix
                row['valid'], row['symbol'] = True, k
                row['bits'] = '1'*k+'0'*(code-k)
        row['reason'] = reason or ('有效（训练补偿，毛刺告警）' if row['glitch_count'] else '有效（训练补偿）')
    return pilots

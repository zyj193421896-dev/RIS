"""Explicit 10 independent PWM pulses + four QAM corners + 256 payload slots.

The PWM preamble is not mapped through the QAM codebook. Its timing domain is
explicitly separate from the QAM codebook's 256 slices.
"""
import copy
import math

MT_PERCENT = [[25,50],[65,0],[40,30],[35,0],[45,20],
              [75,0],[25,10],[55,0],[25,40],[50,15]]
CORNERS_HEX = ['00','08','88','80']


def validate_transmitter(info, config):
    """Reject metadata which contradicts the actual receiver timing contract."""
    fields = {'qam_slices','qam_cycles_per_symbol','preamble_timing',
              'preamble_mt_percent','training_hex','payload_symbols','frame_symbols'}
    if not isinstance(info,dict) or set(info) != fields:
        raise ValueError('发端时序说明字段不完整或含未知字段')
    if (type(info['qam_slices']) is not int or not 16 <= info['qam_slices'] <= 16384 or
            type(info['qam_cycles_per_symbol']) is not int or info['qam_cycles_per_symbol'] < 1):
        raise ValueError('QAM slice数及每符号周期数须为有效整数')
    if info['preamble_timing'] not in ('independent_exact_percent','uniform_slice_nearest_integer'):
        raise ValueError('未知独立前导时序模式')
    pwm=config['pwm_preamble_options'];td=config['tdcm_options']
    mt=info['preamble_mt_percent']
    if not isinstance(mt,list) or len(mt)!=len(pwm['symbols'].split()):
        raise ValueError('前导MT时序数与独立PWM模板长度不一致')
    try:
        if any(len(row)!=2 or not all(math.isfinite(v) for v in row) for row in mt): raise ValueError()
        pulse=[[d*pwm['total_chips']/100,w*pwm['total_chips']/100] for w,d in mt]
    except (TypeError,ValueError):
        raise ValueError('MT百分比须为[高电平%,延迟%]数值列表') from None
    if info['preamble_timing']=='uniform_slice_nearest_integer':
        if pwm['total_chips']!=info['qam_slices']: raise ValueError('统一slice时前导/QAM片数须一致')
        pulse=[[math.floor(v+.5) for v in row] for row in pulse]
    if pulse!=pwm.get('pulse_chips'):
        raise ValueError('前导MT说明与实际接收脉宽/延迟模板不一致')
    from rx706_qam_preamble import parse_preamble
    import numpy as np
    expected=parse_preamble(' '.join(info['training_hex']),'hex',config['order'])
    pre=config['preamble_options']
    observed=parse_preamble(pre['symbols'],pre.get('representation','labels'),config['order'])
    if not np.array_equal(expected,observed): raise ValueError('训练HEX说明与接收训练不一致')
    count=info['payload_symbols'];frame=info['frame_symbols']
    if (type(count) is not int or count!=td['payload_data_symbols'] or type(frame) is not int or
            frame!=len(mt)+len(expected)+count+td['payload_tail_symbols']):
        raise ValueError('发端帧长度说明与前导/训练/净荷窗口不一致')
    if td['base_hz']!=config['symbol_hz']*info['qam_cycles_per_symbol']:
        raise ValueError('QAM调制基频/符号率与每符号周期数不一致')
    return copy.deepcopy(info)


def build_qam10_config(*, quantized_preamble=False, symbol_hz=10000.):
    """Nominal protocol only. Does not claim a physical transmitter was changed."""
    if isinstance(symbol_hz, bool) or not isinstance(symbol_hz, (int, float)) or not math.isfinite(symbol_hz) or symbol_hz <= 0:
        raise ValueError("符号率须为有限正数")
    symbol_hz = float(symbol_hz)
    total=256 if quantized_preamble else 20
    pulses=[[d*total/100,w*total/100] for w,d in MT_PERCENT]
    if quantized_preamble: pulses=[[math.floor(v+.5) for v in row] for row in pulses]
    cfg=dict(version=1,method='tdcm_parameter_config',
        name=f'256QAM 256slice {symbol_hz/1000:g}kHz 独立10前导+4四角+256净荷',
        description=('逻辑发送/接收参数，不绑定IQ或标定。前导是独立PWM序列，不映射到256QAM。'
            f'当前假定前导周期与QAM符号周期均为{1e6/symbol_hz:g}us、无间隔、QAM每符号1个调制周期。'
            '精确百分比前导与QAM的256slice时钟独立；发端须确实实现该时序。'
            '载波频率、ADC采样率、通道从本次采集设置读取；未知净荷不作参考，BER未测。'),
        order=256,symbol_hz=symbol_hz,
        tdcm_options=dict(acquisition_mode='pwm',base_hz=symbol_hz,harmonic=1,
            normalization='blind',amplitude_scale=1.,phase_deg=0.,payload_symbols=0,
            payload_repeat=1,payload_data_symbols=256,payload_tail_symbols=0,
            crc_debug=dict(enabled=False),compensation=dict(enabled=True,
                clock_mode='off',phase_mode='off',frame_mode='all',frame_symbols=260,
                max_frames=2048,calibration_frames=20,period_ppm=0,fit_timing=False)),
        pwm_preamble_options=dict(symbols=' '.join(map(str,range(10))),total_chips=total,
            prefix_chips=1,guard_chips=total-1,pulse_chips=pulses,symbol_hz=symbol_hz,
            gap_s=0.,search_s=1.,max_error_percent=25.,width_offset_chips=0.,
            max_frequency_hz=1000.,qam_search_symbols=0.),
        preamble_options=dict(symbols=' '.join(CORNERS_HEX),representation='hex',
            max_frequency_hz=1000.,max_evm_percent=10.,search_s=1.,orientation='auto',fit_dc_offset=True,estimate_frequency=False),
        transmitter=dict(qam_slices=256,qam_cycles_per_symbol=1,
            preamble_timing='uniform_slice_nearest_integer' if quantized_preamble else 'independent_exact_percent',
            preamble_mt_percent=copy.deepcopy(MT_PERCENT),training_hex=list(CORNERS_HEX),
            payload_symbols=256,frame_symbols=270))
    if quantized_preamble:
        cfg['name']+='（前导按256slice最近整数量化备选）'
        cfg['description']='未确认的统一256slice备选：前导高宽和延迟分别取最近整数slice，实际百分比并非原精确5%网格。'+cfg['description'].replace('精确百分比前导与QAM的256slice时钟独立；','前导与QAM共用256slice；')
    from rx706_qam_config import validate_parameter_config
    return validate_parameter_config(cfg)

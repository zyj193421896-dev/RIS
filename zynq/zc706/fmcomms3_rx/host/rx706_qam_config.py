"""Portable TDCM parameters only: never a capture fingerprint or calibration."""
import copy
import json
import math
from pathlib import Path

METHOD='tdcm_parameter_config'


def validate_parameter_config(value):
    if not isinstance(value,dict) or value.get('version')!=1 or value.get('method')!=METHOD:
        raise ValueError('不支持的通用TDCM参数JSON格式')
    allowed={'version','method','name','description','order','symbol_hz','tdcm_options',
             'pwm_preamble_options','preamble_options','transmitter'}
    extra=set(value)-allowed
    if extra:raise ValueError('通用参数JSON不允许未知字段或采集绑定字段：'+', '.join(sorted(extra)))
    cfg=copy.deepcopy(value)
    order=cfg.get('order');rate=cfg.get('symbol_hz')
    if type(order) is not int or order not in (16,64,256):raise ValueError('QAM阶数须为16、64或256')
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:
        raise ValueError('符号率须为有限正数')
    for key in ('tdcm_options','pwm_preamble_options','preamble_options'):
        if not isinstance(cfg.get(key),dict):raise ValueError(key+'须为参数对象')
    td=cfg['tdcm_options'];comp=td.get('compensation')
    if td.get('acquisition_mode')!='pwm' or not isinstance(comp,dict):
        raise ValueError('通用参数JSON目前用于有PWM前导TDCM及显式补偿参数')
    td_allowed={'acquisition_mode','base_hz','harmonic','normalization','amplitude_scale','phase_deg',
                'payload_symbols','payload_repeat','payload_data_symbols','payload_tail_symbols','crc_debug','compensation'}
    if set(td)-td_allowed:raise ValueError('TDCM包含未知字段、自动搜索或冻结复算；请用原方案格式')
    from rx706_tdcm import validate_tdcm
    from rx706_tdcm_debug import validate_compensation
    from rx706_qam_pwm import validate_pwm_header
    from rx706_qam_preamble import validate_preamble
    # Infinity only disables Fs-dependent checks during parameter loading.
    # No Fs is written into the config; analyze() validates the user's actual Fs.
    checked=validate_tdcm(math.inf,rate,td);checked.pop('cycles_per_symbol')
    comp=validate_compensation(comp)
    if not comp['enabled']:raise ValueError('该参数格式要求启用TDCM补偿与显式帧窗口')
    pwm_allowed={'symbols','total_chips','prefix_chips','guard_chips','symbol_hz','gap_s',
        'search_s','max_error_percent','width_offset_chips','max_frequency_hz','qam_search_symbols','pulse_chips','observation_average_chips'}
    if set(cfg['pwm_preamble_options'])-pwm_allowed:
        raise ValueError('PWM前导包含未知字段，拒绝忽略时序参数')
    pwm=validate_pwm_header(cfg['pwm_preamble_options'],math.inf)
    pwm['symbols']=' '.join(map(str,pwm['symbols']))
    pre=cfg['preamble_options'];points,_,_,_=validate_preamble(pre,order,rate)
    data=checked['payload_data_symbols'];tail=checked['payload_tail_symbols']
    if data<1 or comp['frame_symbols']!=len(points)+data+tail:
        raise ValueError('每帧QAM窗口数须等于训练数＋物理净荷数＋额外尾数')
    if checked['payload_symbols'] not in (0,comp['frame_symbols']):
        raise ValueError('提取符号数须为0或等于每帧QAM窗口数')
    cfg.update(order=order,symbol_hz=float(rate),tdcm_options=dict(checked,acquisition_mode='pwm',compensation=comp),
               pwm_preamble_options=pwm)
    if 'transmitter' in cfg:
        from rx706_qam10_profile import validate_transmitter
        cfg['transmitter']=validate_transmitter(cfg['transmitter'],cfg)
    return cfg


def read_parameter_config(path):
    path=Path(path)
    if path.stat().st_size>512*1024:raise ValueError('参数JSON超过512KiB')
    return validate_parameter_config(json.loads(path.read_text(encoding='utf-8')))


def rescale_parameter_rate(config, symbol_hz):
    """Explicit nominal-rate edit; preserve cycle and PWM/QAM timing ratios.

    No file/capture/receiver settings are changed. Historical configs remain
    unchanged unless the user explicitly invokes this operation.
    """
    cfg = validate_parameter_config(config)
    if isinstance(symbol_hz, bool) or not isinstance(symbol_hz, (int, float)) or not math.isfinite(symbol_hz) or symbol_hz <= 0:
        raise ValueError('符号率须为有限正数')
    factor = float(symbol_hz) / cfg['symbol_hz']
    cfg['symbol_hz'] = float(symbol_hz)
    cfg['tdcm_options']['base_hz'] *= factor
    cfg['pwm_preamble_options']['symbol_hz'] *= factor
    cfg['name'] = f"{cfg['order']}QAM {symbol_hz/1000:g} kBaud（速率已调整）"
    cfg['description'] = ('显式调整名义速率，保留QAM每符号周期数和PWM/QAM速率比。'
        '间隔秒数、门限、训练及净荷保持不变；须确认与实际发端一致。未知净荷BER未测。')
    return validate_parameter_config(cfg)

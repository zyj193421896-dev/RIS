"""One capture, one complete, checksum-locked offline TDCM replay."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from rx706_qam_config import validate_parameter_config

METHOD = 'iq_bound_tdcm_replay'


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('IQ文件在计算指纹期间发生变化')
    return {'file_bytes': before.st_size, 'sha256': digest.hexdigest()}


def make_recipe(path, parameter_config, *, sample_rate_hz, channel, first_tuple, duration_s, center_hint_hz, nfft=65536, averages=16, remove_dc=True):
    path = Path(path).expanduser().resolve(strict=True)
    params = validate_parameter_config(parameter_config)
    recipe = dict(version=1, method=METHOD, input_path=str(path), input_fingerprint=fingerprint(path),
        spectrum=dict(sample_rate_hz=sample_rate_hz, channel=channel, first_tuple=first_tuple,
                      nfft=nfft, averages=averages, remove_dc=remove_dc),
        recovery=dict(duration_s=duration_s, center_hint_hz=center_hint_hz,
                      order=params['order'], symbol_hz=params['symbol_hz'],
                      tdcm_options=params['tdcm_options'], pwm_preamble_options=params['pwm_preamble_options'],
                      preamble_options=params['preamble_options']),
        caveat='训练标签是用户提供或配置中的假设，除非发端确认；最近邻EVM并非真实参考EVM或BER。')
    return validate_recipe(recipe)


def validate_recipe(value):
    if not isinstance(value, dict) or value.get('version') != 1 or value.get('method') != METHOD:
        raise ValueError('不支持的IQ绑定方案格式')
    v=copy.deepcopy(value); s=v.get('spectrum'); r=v.get('recovery'); fp=v.get('input_fingerprint')
    if not isinstance(v.get('input_path'),str) or not v['input_path'] or not isinstance(s,dict) or not isinstance(r,dict) or not isinstance(fp,dict):
        raise ValueError('IQ绑定方案缺少文件、采样或恢复参数')
    if type(fp.get('file_bytes')) is not int or fp['file_bytes'] <= 0 or fp['file_bytes'] % 8 or not isinstance(fp.get('sha256'),str) or len(fp['sha256']) != 64 or any(c not in '0123456789abcdef' for c in fp['sha256']):
        raise ValueError('IQ文件指纹无效')
    def number(x): return not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x)
    if not number(s.get('sample_rate_hz')) or s['sample_rate_hz']<=0 or s.get('channel') not in (1,2) or type(s.get('channel')) is not int or type(s.get('first_tuple')) is not int or s['first_tuple']<0:
        raise ValueError('采样率、通道或起点无效')
    if type(s.get('nfft')) is not int or not 256<=s['nfft']<=1048576 or s['nfft']&(s['nfft']-1) or type(s.get('averages')) is not int or not 1<=s['averages']<=64 or type(s.get('remove_dc')) is not bool:
        raise ValueError('频谱参数无效')
    if not number(r.get('duration_s')) or not 0<r['duration_s']<=1 or not number(r.get('center_hint_hz')) or abs(r['center_hint_hz'])>=s['sample_rate_hz']/2:
        raise ValueError('时长或载波无效')
    if s['first_tuple']+math.ceil(r['duration_s']*s['sample_rate_hz'])>fp['file_bytes']//8:
        raise ValueError('恢复范围超出IQ文件')
    p=validate_parameter_config(dict(version=1,method='tdcm_parameter_config',order=r.get('order'),symbol_hz=r.get('symbol_hz'),tdcm_options=r.get('tdcm_options'),pwm_preamble_options=r.get('pwm_preamble_options'),preamble_options=r.get('preamble_options')))
    for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options'): r[k]=p[k]
    from rx706_tdcm import validate_tdcm
    from rx706_qam_pwm import validate_pwm_header
    validate_tdcm(s['sample_rate_hz'],r['symbol_hz'],r['tdcm_options'])
    validate_pwm_header(r['pwm_preamble_options'],s['sample_rate_hz'])
    return v


def read_recipe(path):
    path=Path(path)
    if path.stat().st_size>512*1024: raise ValueError('方案JSON超过512KiB')
    return validate_recipe(json.loads(path.read_text(encoding='utf-8')))


def verify_input(recipe):
    p=Path(recipe['input_path']).expanduser()
    if not p.is_file(): raise ValueError('绑定的IQ文件不存在：'+str(p))
    if fingerprint(p)!=recipe['input_fingerprint']: raise ValueError('IQ文件大小或SHA-256不符；已停止复算')
    return p


def main():
    parser=argparse.ArgumentParser(description='把一份已确认的通用参数绑定到一个IQ文件')
    parser.add_argument('iq'); parser.add_argument('parameters'); parser.add_argument('output')
    parser.add_argument('--fs',type=float,required=True);parser.add_argument('--channel',type=int,choices=(1,2),required=True)
    parser.add_argument('--first-tuple',type=int,default=0);parser.add_argument('--duration',type=float,required=True)
    parser.add_argument('--carrier',type=float,required=True,help='原载波相对LO，Hz')
    a=parser.parse_args(); from rx706_qam_config import read_parameter_config
    result=make_recipe(a.iq,read_parameter_config(a.parameters),sample_rate_hz=a.fs,channel=a.channel,first_tuple=a.first_tuple,duration_s=a.duration,center_hint_hz=a.carrier)
    output=Path(a.output)
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(output)
if __name__=='__main__':main()

"""Non-destructive repeated-payload/tail diagnostics; CRC is deliberately unverified."""
import math
import numpy as np


def validate_payload_options(options):
    def integer(key,default,lo,hi):
        value=options.get(key,default)
        try:n=float(value)
        except (ValueError,TypeError):raise ValueError(key+'须为整数') from None
        if isinstance(value,bool) or not math.isfinite(n) or not n.is_integer() or not lo<=n<=hi:
            raise ValueError(f'{key}须为{lo}～{hi}的整数')
        return int(n)
    repeat=integer('payload_repeat',1,1,2)
    data=integer('payload_data_symbols',0,0,65536);tail=integer('payload_tail_symbols',0,0,4096)
    crc=options.get('crc_debug',{'enabled':False})
    if not isinstance(crc,dict) or not isinstance(crc.get('enabled',False),bool):raise ValueError('CRC调试开关须为布尔值')
    if tail and not data:raise ValueError('查看额外帧尾须明确物理净荷数，不能猜测分界')
    return dict(payload_repeat=repeat,payload_data_symbols=data,payload_tail_symbols=tail,
                crc_debug=dict(enabled=crc.get('enabled',False)))


def attach_payload_debug(result):
    """Keep every physical sample/decision; never vote, average or remove a tail."""
    cfg=getattr(result,'tdcm',{}) or {};o=validate_payload_options(cfg)
    if o['payload_repeat']==1 and not o['payload_tail_symbols'] and not o['crc_debug']['enabled']:return result
    crc=dict(enabled=o['crc_debug']['enabled'],status='unverified' if o['crc_debug']['enabled'] else 'disabled',
        note='未提供CRC算法参数、范围、比特打包或字节序；不计算、不判通过/失败，额外符号只保留原始候选。')
    summary=dict(options=o,crc=crc,status='alignment_unverified',physical_symbols=len(result.labels),
        pair_positions=0,agreeing_pairs=0,disagreeing_pairs=0,tail_symbols=0,unpaired_symbols=0,
        policy='保留全部物理符号；逐帧从训练后第一个净荷开始配对；不平均、不投票、不去重改写。相同判决仅作逻辑候选，不同则未知。')
    result.payload_debug=summary;result.payload_pairs=[];result.tail_records=[]
    ids=getattr(result,'frame_ids',None);frames=getattr(result,'frames',None)
    if ids is None or not frames or getattr(result,'preamble',None) is None or getattr(result,'pwm_preamble',None) is None:
        summary['note']='没有已匹配训练的逐帧净荷边界，不能可靠配对或定位CRC尾部；原始符号仍保留。'
        return result
    rows={row['index']:row for row in frames};all_valid=True
    for frame in np.unique(ids):
        indices=np.flatnonzero(ids==frame);ndata=o['payload_data_symbols'] or len(indices)-o['payload_tail_symbols']
        valid=len(indices)==ndata+o['payload_tail_symbols'] and bool(rows[int(frame)]['valid'])
        all_valid &= valid
        data=indices[:max(0,ndata)]
        for k,index in enumerate(indices[max(0,ndata):]):
            z=result.scatter[index]
            result.tail_records.append(dict(frame=int(frame),tail_index=k,physical_index=int(index),
                time_s=float(result.time_s[index]),candidate_label=int(result.labels[index]),
                candidate_hex=f'{int(result.labels[index]):02X}',candidate_bits=''.join(map(str,result.bits[index])),
                normalized_i=float(z.real),normalized_q=float(z.imag),classification='未分类额外尾符号，CRC未验证'))
        if o['payload_repeat']==2:
            summary['unpaired_symbols']+=len(data)%2
            for k in range(len(data)//2):
                left,right=map(int,data[2*k:2*k+2]);a,b=map(int,result.labels[[left,right]])
                equal=a==b;logical=a if equal and valid else None
                result.payload_pairs.append(dict(frame=int(frame),logical_index=k,left_physical_index=left,
                    right_physical_index=right,left_time_s=float(result.time_s[left]),right_time_s=float(result.time_s[right]),
                    left_hex=f'{a:02X}',right_hex=f'{b:02X}',agree=bool(equal),alignment_valid=bool(valid),
                    logical_label=logical,logical_hex=None if logical is None else f'{logical:02X}',
                    normalized_distance=float(abs(result.scatter[left]-result.scatter[right]))))
    summary.update(status='checked' if all_valid else 'frame_length_mismatch',pair_positions=len(result.payload_pairs),
        agreeing_pairs=sum(row['agree'] for row in result.payload_pairs),
        disagreeing_pairs=sum(not row['agree'] for row in result.payload_pairs),tail_symbols=len(result.tail_records))
    return result

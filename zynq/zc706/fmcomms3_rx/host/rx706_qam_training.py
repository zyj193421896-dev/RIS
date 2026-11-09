"""Fit only explicitly known QAM training; payload decisions never train it."""
import math
import numpy as np
from rx706_spectrum import _cancel


def validate_orientation(options):
    mode=options.get('orientation','normal')
    if mode not in ('normal','conjugate','auto'):
        raise ValueError('QAM训练镜像方式须为normal、conjugate或auto')
    offset=options.get('fit_dc_offset',False)
    if not isinstance(offset,bool):raise ValueError('训练零点校正须为布尔值')
    return mode,offset


def normalize_training(samples,fit):
    z=(np.asarray(samples)-fit['offset'])/fit['gain']
    return z.conj() if fit['conjugated'] else z


def fit_training(samples,points,rate,bound,tolerance,*,orientation='normal',
                 fit_dc_offset=False,estimate_frequency=False,cancel_event=None):
    c=np.asarray(samples,dtype=complex);points=np.asarray(points,dtype=complex)
    if c.shape!=points.shape or len(c)<3:raise ValueError('QAM训练至少需要3个已知复符号')
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(points)):raise ValueError('训练含无效复数')
    if orientation not in ('normal','conjugate','auto'):raise ValueError('未知训练镜像方式')
    design=np.column_stack([points.real,points.imag])
    if orientation=='auto' and np.linalg.matrix_rank(design)<2:
        raise ValueError('已知训练不能唯一辨识镜像；请提供非共线复符号或指定镜像方式')
    t=np.arange(len(c))/rate;power=max(float(np.mean(abs(points)**2)),1e-30)
    results=[]
    for mirrored in ((False,True) if orientation=='auto' else (orientation=='conjugate',)):
        expected=points.conj() if mirrored else points
        matrix=np.column_stack([expected,np.ones(len(expected))]) if fit_dc_offset else expected[:,None]
        if np.linalg.matrix_rank(matrix)<matrix.shape[1]:raise ValueError('已知训练不足以同时标定复增益和零点')
        def fit(f):
            _cancel(cancel_event)
            rotated=c*np.exp(-2j*np.pi*f*t)
            params=np.linalg.lstsq(matrix,rotated,rcond=None)[0]
            gain=params[0];offset=params[1] if fit_dc_offset else 0j
            if abs(gain)<1e-12:return float('inf'),gain,offset
            error=float(100*np.sqrt(np.mean(abs(rotated-matrix@params)**2)/(abs(gain)**2*power)))
            return error,gain,offset
        frequency=0.
        if estimate_frequency:
            grid=np.linspace(-bound,bound,65);scores=[fit(f)[0] for f in grid];at=int(np.argmin(scores))
            if at in (0,len(grid)-1):continue
            lo,hi=grid[at-1],grid[at+1];ratio=(math.sqrt(5)-1)/2
            for _ in range(22):
                a,b=hi-ratio*(hi-lo),lo+ratio*(hi-lo)
                if fit(a)[0]<fit(b)[0]:hi=b
                else:lo=a
            frequency=float((lo+hi)/2)
        error,gain,offset=fit(frequency)
        results.append(dict(evm_percent=error,gain=gain,offset=offset,frequency_hz=frequency,conjugated=mirrored))
    accepted=sorted((v for v in results if v['evm_percent']<=tolerance),key=lambda v:v['evm_percent'])
    if not accepted:raise ValueError('已知QAM训练未通过幅相/镜像匹配；请核对顺序、时序、码本及噪声')
    if len(accepted)>1 and accepted[1]['evm_percent']-accepted[0]['evm_percent']<.5:
        raise ValueError('QAM训练的正常/镜像模型接近，不能唯一确定映射')
    best=accepted[0];best['alternatives']=[dict(conjugated=v['conjugated'],evm_percent=v['evm_percent']) for v in results]
    return best

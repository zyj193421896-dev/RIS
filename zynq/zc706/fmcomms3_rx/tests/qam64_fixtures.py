"""Independent Gray labels and sampled TX pulses; no receiver mapping in synthesis."""
from pathlib import Path
import numpy as np
from rx706_spectrum import analyze_bin


def reference_points(labels,order=64):
    side=int(np.sqrt(order));gray=np.array([v^(v>>1) for v in range(side)])
    inverse=np.argsort(gray);labels=np.asarray(labels,dtype=int)
    return ((2*inverse[labels//side]-side+1)+1j*(2*inverse[labels%side]-side+1))/np.sqrt(2*(order-1)/3)


def save_capture(directory,name,x,fs):
    raw=np.zeros((len(x),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
    p=Path(directory)/name;raw.tofile(p)
    return analyze_bin(p,nfft=1024,averages=1,sample_rate_hz=fs)


def direct_capture(directory,*,pulse='Rectangular',pwm=False,known=False,cfo=0):
    fs=3200000.;rate=100000.;sps=32;carrier=40000.;rng=np.random.default_rng(6402)
    pilots=np.array([0,4,36,32,9,17,63,28]*2) if known else np.empty(0,int)
    labels=rng.integers(0,64,1000);points=reference_points(np.r_[pilots,labels])
    if pulse=='RRC':
        size=1<<(len(points)*sps+1023).bit_length();imp=np.zeros(size,complex);imp[512:512+sps*len(points):sps]=points
        f=abs(np.fft.fftfreq(size,1/fs));raised=np.zeros(size);raised[f<=rate*.65/2]=1
        edge=(f>rate*.65/2)&(f<rate*1.35/2);raised[edge]=.5*(1+np.cos(np.pi*(f[edge]-rate*.65/2)/(rate*.35)))
        signal=np.fft.ifft(np.fft.fft(imp)*np.sqrt(raised))*sps
    else:signal=np.repeat(points,sps)
    lead=1000 if known or pwm else 0
    pwm_rate=25000.;pwm_sps=fs/pwm_rate
    header=np.where((np.arange(int(3*pwm_sps))/pwm_sps)%1<.2,1.,-1.) if pwm else np.empty(0)
    gap=64 if pwm else 0
    x=650*np.r_[np.zeros(lead),header,np.zeros(gap),signal]
    x=x.astype(complex)*np.exp(1j*(.7+2*np.pi*(carrier+cfo)*np.arange(len(x))/fs))
    s=save_capture(directory,'direct.bin',x,fs)
    o=dict(duration_s=(lead+len(header)+gap+512+len(points)*sps-256)/fs if pulse=='RRC' else .2,order=64,symbol_hz=rate,center_hint_hz=carrier,pulse=pulse)
    if pwm:o['pwm_preamble_options']=dict(symbol_hz=pwm_rate,gap_s=gap/fs,qam_search_symbols=64)
    if known:o['preamble_options']=dict(symbols=' '.join(map(str,pilots)),max_frequency_hz=1000,max_evm_percent=6)
    return s,o,labels


def tdcm_capture(directory,*,order=64,ndata=64,pilots=(),frames=32,header=True,repeat=True,
                 ppm=45,carrier=120173.4,ripple=5,mirror=False,noise=.7,bad_training=False,
                 harmonic=1,rate=19531.25,cycles=1,dc_offset=False,tail_change=False,pwm_rate=None,paired_payload=False,pair_disagreement=False,tail_labels=()):
    fs=6144000.;period=fs/rate*(1+ppm*1e-6);rng=np.random.default_rng(6411)
    labels=rng.permutation(np.resize(np.arange(order),ndata))
    payload=np.tile(labels,(frames,1)) if repeat else rng.integers(0,order,(frames,ndata))
    if tail_change:payload[int(.65*frames):]=rng.integers(0,order,(frames-int(.65*frames),ndata))
    if paired_payload:
        if ndata%2:raise ValueError('paired fixture requires even physical length')
        logical=rng.permutation(np.resize(np.arange(order),ndata//2))
        payload=np.tile(np.repeat(logical,2),(frames,1)) if repeat else np.repeat(rng.integers(0,order,(frames,ndata//2)),2,axis=1)
        if pair_disagreement:payload[:,9]=payload[:,8]^1
    sent=np.column_stack([np.tile(np.array(pilots,dtype=int),(frames,1)),payload,np.tile(np.asarray(tail_labels,dtype=int),(frames,1))]);points=reference_points(sent,order)
    if bad_training:points[:,:len(pilots)]=points[:,np.array([0,2,1,3])]
    if mirror:points=points.conj()
    side=int(np.sqrt(order));radius=np.sqrt(2)*(side-1)/np.sqrt(2*(order-1)/3)
    duty=np.arcsin(np.clip(abs(points)/radius,0,1))/np.pi;delay=(-np.angle(points)/(2*np.pi*harmonic)-duty/2)%1
    nh=3 if header else 0;pwm_rate=pwm_rate or rate;header_symbols=nh*rate/pwm_rate
    nf=header_symbols+sent.shape[1];lead=1000;size=int(np.ceil(lead+frames*nf*period+40))
    n=np.arange(size);c=(n-lead)/period;fi=np.floor(c/nf).astype(int);local=c-fi*nf
    active=(fi>=0)&(fi<frames);high=np.zeros(size,bool);h=active&(local<header_symbols)
    high[h]=(local[h]*pwm_rate/rate)%1<.2
    d=active&(local>=header_symbols);f=fi[d];data=local[d]-header_symbols;j=np.floor(data).astype(int);within=(data*cycles)%1
    high[d]=((within-delay[f,j])%1)<duty[f,j]
    t=n/fs;phase=2*np.pi*carrier*t+.4+ripple*np.sin(2*np.pi*2*t)
    x=np.where(active,700*(high-.5),0)*np.exp(1j*phase)
    if dc_offset:x+=np.where(active,7*np.exp(1j*(phase+2*np.pi*rate*cycles*t+.2)),0)
    x+=noise*(rng.normal(size=size)+1j*rng.normal(size=size))
    s=save_capture(directory,'tdcm.bin',x,fs)
    o=dict(duration_s=1,center_hint_hz=carrier-173.4,symbol_hz=rate,order=order,
        tdcm_options=dict(base_hz=rate*cycles,harmonic=harmonic,normalization='blind'),
        pwm_preamble_options=dict(symbol_hz=pwm_rate,qam_search_symbols=0))
    if pilots:o['preamble_options']=dict(symbols=' '.join(f'{v:02X}' for v in pilots),representation='hex',orientation='auto',fit_dc_offset=True,max_evm_percent=8)
    return s,o,payload,period


def compensation(o,*,ndata=64,pilots=0,clock='repeat',phase='square',ppm=45,fit=False):
    o['tdcm_options']['compensation']=dict(enabled=True,frame_mode='all',clock_mode=clock,phase_mode=phase,
        period_ppm=ppm,clock_search_ppm=1000,frame_symbols=ndata+pilots,calibration_frames=8,fit_timing=fit)
    return o

"""Synthetic-only complete IQ search acceptance; no hardware or user BIN touched."""
import argparse, copy, json, math, sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'))
from rx706_pwm_autotune import search, search_options, file_evaluation
from rx706_pwm_symbols import plus_fields, PLUS_SHA256
from rx706_pwm_frame_builder import build_profile
from rx706_pwm import recover_pwm2
from rx706_spectrum import analyze_bin

def write_iq(path,profile,duration,fs,rate,polarity=1):
    book=np.asarray(profile['codebook'])
    cycle=np.r_[profile['users'][0]['identity'],np.arange(78)]
    count=int(duration*fs);origin=.37/rate
    with path.open('xb') as stream:
        for start in range(0,count,500000):
            t=np.arange(start,min(start+500000,count))/fs
            index=np.floor((t-origin)*rate).astype(int)
            symbol=cycle[index%len(cycle)]
            phase=(t-origin)*rate-index
            high=(index>=0)&(phase>=book[symbol,0]/20)&(phase<(book[symbol,0]+book[symbol,1]+.5)/20)
            x=polarity*np.where(high,-500,500)*np.exp(1j*(2*np.pi*713456.7*t+.43))
            raw=np.zeros((len(t),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
            raw.tofile(stream)
    n=path.stat().st_size
    Path(str(path)+'.json').write_text(json.dumps(dict(complete=True,file_bytes=n,capture=dict(bytes=n,tuples=n//8,
        order=['I1','Q1','I2','Q2'],dtype='int16',byte_order='little',valid_bits=12,channels=2,sample_rate_hz=fs))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=2.)
    args=parser.parse_args();assert not args.output.exists()
    fields=plus_fields();fields.update(payload_symbols='78',reference_enabled=True,
        reference=' '.join(map(str,range(78))))
    profile=build_profile(fields);nominal=profile['symbol_khz']*1000;actual=nominal*(1+15e-6)
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='pwm78_synthetic_') as folder:
        folder=Path(folder);capture=folder/'SYNTHETIC_pwm78.bin';path=folder/'SYNTHETIC_profile.json'
        path.write_text(json.dumps(profile));write_iq(capture,profile,args.seconds,15.36e6,actual)
        def progress(item):
            print(item['stage'],item['candidate']['state_one'],item['duration_s'],
                item.get('error',item.get('metrics',{}).get('usable_fraction')),flush=True)
        report=search(capture,path,finalists=1,max_seconds=1200,progress=progress)
        e=report['post_selection_evaluation']
        assert report['status']=='candidate_selected',report['status']
        assert report['analysis_interval']['completed'] and e['analyzed_duration_s']==args.seconds
        assert e['ber']==0 and e['erased_payload_symbols']==0,e
        assert e['codebook_size']==78 and e['label_bits_per_symbol']==7
        assert abs(e['physical_symbol_rate_hz']/actual-1)<5e-6
        assert e['actual_goodput_bps'] is None
        chosen=report['selection']['candidate']
        # Short opposite physical polarity, independent of search reference; full decoder.
        short=folder/'SYNTHETIC_inverted.bin';write_iq(short,profile,.008,15.36e6,actual,polarity=-1)
        spec=analyze_bin(short,nfft=4096,averages=4)
        recovered=recover_pwm2(spec,pulse_options=search_options(profile),symbol_hz=nominal,
            duration_s=.008,decimation=1,allow_unlocked=True)
        inverse=file_evaluation(recovered,profile)
        assert inverse['ber']==0 and inverse['erased_payload_symbols']==0,inverse
        no_reference=file_evaluation(recovered,dict(profile,reference_enabled=False))
        assert no_reference['ber'] is None
        result=dict(type='SYNTHETIC ONLY: generated int16 IQ through production search and decoder',
            hardware_or_new_real_capture_tested=False,source_codebook_sha256=PLUS_SHA256,
            synthetic_profile=profile,actual_symbol_hz=actual,full_file_seconds=args.seconds,
            evaluation=e,selection=chosen,analysis_interval=report['analysis_interval'],
            selection_interval=report['selection_interval'],runtime=report['runtime'],
            inverted_short_evaluation=inverse,no_reference_ber=no_reference['ber'],
            elapsed_s=time.monotonic()-started)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status='PASS',seconds=args.seconds,compared_bits=e['reference_overall']['compared_bits'],
            ber=e['ber'],erased=e['erased_payload_symbols'],measured=e['physical_symbol_rate_hz'],elapsed_s=result['elapsed_s'])),flush=True)

if __name__=='__main__':main()

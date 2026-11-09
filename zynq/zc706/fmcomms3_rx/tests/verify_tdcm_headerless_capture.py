"""Opt-in read-only headerless TDCM capture acceptance and report export."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_tdcm import recover_tdcm
from rx706_qam_debug_gui import export_report_html,export_recipe_json,export_symbols_csv


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-rate',type=float,required=True);p.add_argument('--symbol-rate',type=float,required=True)
    p.add_argument('--carrier-center',type=float,required=True);p.add_argument('--base-rate',type=float)
    p.add_argument('--channel',type=int,choices=[1,2],default=1);p.add_argument('--harmonic',type=int,default=1)
    p.add_argument('--order',type=int,choices=[16,64,256],default=256)
    p.add_argument('--mode',choices=['repeat','fixed'],default='repeat');p.add_argument('--period-ppm',type=float,default=0)
    p.add_argument('--carrier-span',type=float,default=500000);p.add_argument('--block-symbols',type=int,default=256)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);before=digest(a.capture)
    spectrum=analyze_bin(a.capture,channel=a.channel,nfft=262144,averages=8,sample_rate_hz=a.sample_rate)
    start=time.monotonic()
    r=recover_tdcm(spectrum,duration_s=1,center_hint_hz=a.carrier_center,symbol_hz=a.symbol_rate,order=a.order,
        tdcm_options=dict(acquisition_mode='no_header_'+a.mode,base_hz=a.base_rate or a.symbol_rate,harmonic=a.harmonic,
            normalization='blind',auto_search=dict(enabled=True,carrier_span_hz=a.carrier_span),
            compensation=dict(enabled=True,phase_mode='square',clock_mode='repeat' if a.mode=='repeat' else 'manual',
                period_ppm=a.period_ppm,frame_symbols=a.block_symbols)))
    elapsed=time.monotonic()-start
    export_report_html(r,spectrum,a.output/'report.html');export_recipe_json(r,a.output/'config.json')
    export_symbols_csv(r,a.output/'candidate_symbols.csv')
    result=dict(input=str(a.capture.resolve()),source_module=sys.modules['rx706_tdcm_headerless'].__file__,
        sha256_before=before,sha256_after=digest(a.capture),symbols=len(r.scatter),blocks=len(r.frames),
        elapsed_s=elapsed,nearest_evm_percent=r.evm_percent,acquisition=r.diagnostics['acquisition'],ber=None)
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['sha256_before']==result['sha256_after'],'Input file changed'
    print(json.dumps({k:v for k,v in result.items() if k!='acquisition'},ensure_ascii=False,indent=2))
    print('Structure status:',r.diagnostics['acquisition']['status'],'; unknown payload BER not measured')
    if not r.diagnostics['structural_validation']:raise SystemExit(2)

if __name__=='__main__':main()

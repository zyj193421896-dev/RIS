"""Opt-in current-capture acceptance and ablation; offline, no GUI or hardware."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_tdcm import recover_tdcm
from rx706_qam_debug_gui import export_report_html,export_symbols_csv,scatter_indices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-rate',type=float,default=15360000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sha=lambda:hashlib.sha256(args.capture.read_bytes()).hexdigest()
    before=sha();s=analyze_bin(args.capture,nfft=262144,averages=8,sample_rate_hz=args.sample_rate)
    common=dict(enabled=True,clock_mode='repeat',phase_mode='square',frame_mode='all',fit_timing=True,
                frame_symbols=256,calibration_frames=20,max_frames=128)
    def run(config):
        return recover_tdcm(s,duration_s=1,center_hint_hz=-200000,symbol_hz=19531.25,
            tdcm_options=dict(normalization='blind',compensation=config),
            pwm_preamble_options=dict(symbol_hz=19531.25,search_s=.1))
    started=time.monotonic();r=run(common);elapsed=time.monotonic()-started
    d=r.diagnostics
    export_report_html(r,s,args.output/'report.html');export_symbols_csv(r,args.output/'candidate_symbols.csv')
    names=['quality','period_samples','period_ppm','measured_symbol_hz','timing_offset_symbols',
           'nearest_evm_percent','heldout_template_evm_percent','occupied_labels','phase_coherence','blind_fit']
    result={k:d[k] for k in names}
    result.update(elapsed_recovery_s=elapsed,frames=len(r.frames),valid_frames=sum(bool(v['valid']) for v in r.frames),
                  symbols=len(r.scatter),sample_rate_source=s.rate_source,assumed_sample_rate_hz=s.sample_rate_hz)
    results=[dict(name='both',**result)]
    ablations=[('phase_only',dict(common,clock_mode='off',fit_timing=False)),
               ('clock_only',dict(common,clock_mode='manual',period_ppm=d['period_ppm'],phase_mode='off',fit_timing=False)),
               ('both_off',dict(common,clock_mode='off',phase_mode='off',fit_timing=False))]
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(12,10),layout='constrained');FigureCanvasAgg(figure);axes=figure.subplots(2,2)
    def plot(ax,z,title):
        v=z[scatter_indices(len(z))];ax.scatter(v.real,v.imag,s=2,alpha=.25);ax.set_aspect('equal');ax.set(title=title,xlabel='I',ylabel='Q');ax.grid(alpha=.2)
    plot(axes[0,0],r.scatter,f"Both / NN EVM {r.evm_percent:.2f}%")
    for ax,(name,config) in zip(axes.ravel()[1:],ablations):
        try:
            a=run(config);results.append(dict(name=name,nn_evm_percent=a.evm_percent,symbols=len(a.scatter),
                structural_validation=a.diagnostics['structural_validation'],frames=len(a.frames),valid_frames=sum(bool(v['valid']) for v in a.frames)))
            plot(ax,a.scatter,f'{name} / NN EVM {a.evm_percent:.2f}%')
        except ValueError as exc:
            results.append(dict(name=name,error=str(exc)));ax.set_title(name+': rejected')
    figure.savefig(args.output/'ablation.png',dpi=130)
    result=dict(input=str(args.capture.resolve()),sha256_before=before,sha256_after=sha(),runs=results,
                note='Blind grid / repeat template, not known payload EVM or BER. Each ablation recalibrates using its configured pipeline.')
    (args.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:float(x)))
    assert before==result['sha256_after'],'Input changed'
    assert d['heldout_template_evm_percent'] is not None and d['heldout_template_evm_percent']<5,'Holdout structure regression'
    assert d['occupied_labels']==256 and len(r.frames)>=70,'Frame/grid regression'
    print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:float(x)))

if __name__=='__main__':main()

"""Opt-in bounded auto-search acceptance, frozen recipe replay and HTML output."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_tdcm import recover_tdcm
from rx706_tdcm_auto import replay_recipe
from rx706_qam_debug_gui import export_report_html,export_recipe_json,export_symbols_csv


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-rate',type=float,default=15360000)
    p.add_argument('--symbol-rate',type=float,default=19531.25)
    p.add_argument('--carrier-center',type=float,default=-200000)
    p.add_argument('--carrier-span',type=float,default=500000)
    p.add_argument('--frame-candidates',default='64 128 256 512')
    p.add_argument('--order',type=int,choices=[16,64,256],default=256)
    p.add_argument('--max-trials',type=int,default=12)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);before=digest(a.capture)
    spectrum=analyze_bin(a.capture,nfft=262144,averages=8,sample_rate_hz=a.sample_rate)
    start=time.monotonic()
    r=recover_tdcm(spectrum,duration_s=1,center_hint_hz=a.carrier_center,symbol_hz=a.symbol_rate,order=a.order,
        tdcm_options=dict(base_hz=a.symbol_rate,auto_search=dict(enabled=True,carrier_span_hz=a.carrier_span,
            frame_candidates=a.frame_candidates,max_trials=a.max_trials),compensation=dict(enabled=True)),
        pwm_preamble_options=dict(symbol_hz=a.symbol_rate))
    search=r.diagnostics['auto_search'];elapsed=time.monotonic()-start
    export_report_html(r,spectrum,a.output/'report.html');export_recipe_json(r,a.output/'selected_recipe.json')
    export_symbols_csv(r,a.output/'candidate_symbols.csv')
    recipe=json.loads((a.output/'selected_recipe.json').read_text());replay=replay_recipe(spectrum,recipe)
    difference=float(np.max(abs(replay.scatter-r.scatter)))
    result=dict(input=str(a.capture.resolve()),source_module=sys.modules['rx706_tdcm_auto'].__file__,
        sha256_before=before,sha256_after=digest(a.capture),status=search['status'],adopted=search['adopted'],
        selection_metrics=search['selection_metrics'],final_test_metrics=search['final_test_metrics'],
        final_test_coverage=search['final_test_coverage'],partitions=search['partitions'],
        ranges=search['ranges'],carrier_bounds=search['carrier_bounds'],candidates=search['candidates'],
        evaluated_trials=search['evaluated_trials'],proposed_combinations=search['proposed_combinations'],
        budget_stopped=search['budget_stopped'],elapsed_search_s=elapsed,replay_max_abs_difference=difference,
        replay_labels_identical=bool(np.array_equal(replay.labels,r.labels)),frames=len(r.frames),symbols=len(r.scatter),
        nearest_evm_percent=r.evm_percent,ber=r.diagnostics['ber'],note=search['note'])
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['sha256_before']==result['sha256_after']
    assert search['status']=='selected','Auto search did not pass final model verification'
    assert difference<1e-10 and result['replay_labels_identical'],'Frozen replay mismatch'
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

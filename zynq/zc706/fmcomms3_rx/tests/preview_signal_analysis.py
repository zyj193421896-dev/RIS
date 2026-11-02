"""Render the actual analysis plotting path against existing captures (no board)."""
import json
from pathlib import Path
from test_signal_analysis_view import headless_window

root=Path(__file__).resolve().parents[1]
out=root/'reports/signal_analysis_20261028'
out.mkdir(exist_ok=True)
results={}
for name in ['rx_10hz_5s.bin','rx_20261027_154445_998.bin','rx_20261027_160840_971.bin']:
    view=headless_window()
    view._analyze_worker(root/'captures'/name,dict(nfft=65536,averages=16))
    view.poll()
    assert view.result is not None, view.status.get()
    view.figure.savefig(out/(name+'.png'),dpi=120)
    results[name]=dict(status=view.status.get(),scope=view.summary.get())
(out/'sample_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=True,indent=2))

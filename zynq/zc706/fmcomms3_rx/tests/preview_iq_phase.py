"""Offline IQ/phase previews of existing captures using the frontend plotting path."""
import json
from pathlib import Path
import numpy as np
from test_signal_analysis_view import headless_window
from rx706_spectrum import phase_trace

root=Path(__file__).resolve().parents[1]
out=root/'reports/iq_phase_20261028'
out.mkdir(exist_ok=True)
records=[]
for path in sorted((root/'captures').glob('*.bin')):
    for channel in (1,2):
        view=headless_window()
        view.figure.set_size_inches(14,11)
        view._analyze_worker(path,dict(nfft=65536,averages=16,channel=channel))
        view.poll()
        assert view.result is not None, view.status.get()
        result=view.result
        trace=phase_trace(result)
        shifted=phase_trace(result,shifted=True)
        prefix=f'{path.stem}_RX{channel}'
        view.figure.savefig(out/(prefix+'.png'),dpi=120)
        view.constellation_mode.set('主峰移频 + 去直流')
        view.draw_constellation()
        view.figure.savefig(out/(prefix+'_shifted.png'),dpi=120)
        np.savetxt(out/(prefix+'.csv'),np.column_stack([
            trace['time_s'],trace['iq'].real,trace['iq'].imag,abs(trace['iq']),
            trace['wrapped_deg'],trace['unwrapped_deg'],shifted['wrapped_deg'],trace['relative_deg']]),
            delimiter=',',header='time_s,I_counts,Q_counts,magnitude_counts,raw_phase_deg,raw_unwrapped_deg,peak_shifted_phase_deg,raw_selected_minus_other_phase_deg',comments='')
        records.append(dict(file=path.name,channel=channel,sample_rate_hz=result.sample_rate_hz,
            rate_source=result.rate_source,total_duration_s=result.total_tuples/result.sample_rate_hz,
            spectrum_duration_s=result.samples/result.sample_rate_hz,
            waveform_duration_s=len(trace['iq'])/result.sample_rate_hz,
            waveform_samples=len(trace['iq']),phase_floor_dbfs=-45,
            waveform_phase_visible_fraction=float(np.mean(trace['valid'])),
            waveform_relative_phase_visible_fraction=float(np.mean(np.isfinite(trace['relative_deg']))),
            peak_hz=result.peak_hz,power_dbfs=result.power_dbfs))
(out/'summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Rendered {len(records)} channel previews with IQ/phase CSVs to {out}')

"""Read-only residual/erasure audit of the existing four-pilot calibration."""
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2

folder=PROJECT/'host/gui_data/current_pwm'
profile=json.loads((folder/'frame.json').read_text(encoding='utf-8'))
options=dict(profile,step_percent=profile['duty_step'],symbol_count=profile['duty_count'],
             frame_options=dict(profile,reference_format='symbols'))
result=recover_pwm2(analyze_bin(folder/'rx_test.bin',channel=1,nfft=4096,averages=4),
                    pulse_options=options,symbol_hz=float(profile['symbol_khz'])*1000,
                    duration_s=1.,state_one='B',allow_unlocked=True)
rows=result.frames
active=(rows['frame_part']=='净荷') & rows['block_complete']
selected=rows[active]
book=np.asarray(result.pulse_options['codebook'])
expected=book[selected['payload_index']]
report=dict(reference=result.frame_check.overall,by_width=[],short_width_by_delay=[],
            reference_used_for_diagnostics_only=True)
for width in range(5,16):
    match=expected[:,1]==width
    subset=selected[match]
    measured=subset[np.isfinite(subset['corrected_high_chips']) & subset['identity_ok']]
    report['by_width'].append(dict(expected_width=width,slots=len(subset),
        erased=int(np.count_nonzero(~subset['valid'])),measured=len(measured),
        raw_median=float(np.median(measured['high_chips'])),
        corrected_median=float(np.median(measured['corrected_high_chips'])),
        corrected_q10_q90=np.quantile(measured['corrected_high_chips'],[.1,.9]).tolist(),
        below_minimum_guard=int(np.count_nonzero(measured['corrected_high_chips']<4.6))))
for delay in range(11):
    mask=(expected[:,1]==5) & (expected[:,0]==delay)
    subset=selected[mask]
    measured=subset[np.isfinite(subset['corrected_high_chips']) & subset['identity_ok']]
    report['short_width_by_delay'].append(dict(delay=delay,slots=len(subset),
        erased=int(np.count_nonzero(~subset['valid'])),
        raw_median=float(np.median(measured['high_chips'])),
        corrected_median=float(np.median(measured['corrected_high_chips']))))
report['combined_error_or_erasure_rate']=(report['reference']['bit_errors']+report['reference']['erased_bits'])/(report['reference']['compared_bits']+report['reference']['erased_bits'])
(Path(__file__).parent/'short_width_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

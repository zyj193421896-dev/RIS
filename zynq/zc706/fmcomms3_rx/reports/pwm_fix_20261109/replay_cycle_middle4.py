"""Explicit four-known-slot calibration on the unchanged 70-symbol capture."""
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2

folder = PROJECT/'host/gui_data/current_pwm'
profile = json.loads((folder/'frame.json').read_text(encoding='utf-8'))
report = dict(profile='frame.json', calibration_slots=[3,4,5,6],
              known_symbols=[3,4,5,6], pilot_width_percent=[40,45,50,55],
              cycle_symbols=70, reference_fitted=False, experimental_known_slot_calibration=True, modes={})
spectrum = analyze_bin(folder/'rx_test.bin', channel=1, nfft=4096, averages=4)
for mode in ('nearest','floor_width'):
    options = dict(profile, step_percent=profile['duty_step'], symbol_count=profile['duty_count'],
                   width_quantization=mode, frame_options=dict(profile,reference_format='symbols'))
    result = recover_pwm2(spectrum, pulse_options=options, symbol_hz=float(profile['symbol_khz'])*1000,
                          duration_s=1., state_one='B', allow_unlocked=True)
    rows = result.frames
    active = (rows['frame_part']=='净荷') & rows['valid']
    item = dict(result.frame_check.overall, decided_symbols=int(active.sum()),
                symbol_errors=int(np.count_nonzero(rows['symbol'][active] != rows['payload_index'][active])),
                gain_range=[min(c['width_gain'] for c in result.pulse_calibrations),max(c['width_gain'] for c in result.pulse_calibrations)],
                offset_us_range=[min(c['width_offset_us'] for c in result.pulse_calibrations),max(c['width_offset_us'] for c in result.pulse_calibrations)],
                first_rows=[{k:row[k].item() for k in ('index','frame_part','payload_index','symbol','bits','delay_chips',
                            'corrected_high_chips','width_scale','valid','reason')} for row in rows[:22]])
    report['modes'][mode]=item
    print(mode,json.dumps({k:v for k,v in item.items() if k!='first_rows'},ensure_ascii=False),flush=True)
    (Path(__file__).parent/'cycle_middle4_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

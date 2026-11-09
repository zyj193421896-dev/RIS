"""Offline regression of rx_30.bin: 30% pilots and 30--79% payload."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2
from rx706_pwm_protocol import duty_protocol


def main():
    project = Path(__file__).resolve().parents[1]
    source = project/'captures/rx_30.bin'
    output = project/'reports/pwm_30_20261104'
    output.mkdir(parents=True,exist_ok=True)
    spectrum = analyze_bin(source,nfft=4096,averages=4,sample_rate_hz=15.36e6)
    options,_ = duty_protocol(30,7,8)
    result = recover_pwm2(spectrum,symbol_hz=169971.671,duration_s=1.,
        allow_unlocked=True,**options,training_options=dict(preamble_widths=(30,30,30),
            payload_symbols=8,clock_width=True,auto_scale=True,same_tolerance_chips=.5))
    rows = result.frames
    assert result.decode_locked
    assert np.all(rows['training'][8:11])
    assert not rows['training'][11] and rows['symbol'][11]==0
    assert result.training_calibration['spacing_symbols'] > 100000
    # The known cycle is only inspected after recovery; no reference enters it.
    index = np.arange(len(rows))
    phase = index%11
    stable = (index>1024)&(phase<8)&rows['valid']
    errors = int(np.count_nonzero(rows['symbol'][stable]!=phase[stable]))
    assert errors/int(stable.sum()) < .01
    summary = dict(source=str(source),locked=bool(result.decode_locked),rows=len(rows),
        valid=int(rows['valid'].sum()),training=int(rows['training'].sum()),
        invalid=int((~rows['valid']&~rows['training']).sum()),
        first_preamble_rows=rows['index'][8:11].tolist(),first_payload_row=12,
        spacing_rows=result.training_calibration['spacing_symbols'],
        checked_after_warmup=int(stable.sum()),mismatches_after_warmup=errors,
        reference_used_for_recovery=False,configuration=result.calibration_description)
    with (output/'symbols.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('index','time_us','raw_duty','clock_width','corrected_duty','symbol',
            'bits','thermometer','training','structure_ok','calibration_source','reason','raw_reason'))
        for row in rows:
            writer.writerow((int(row['index']),float(row['start_s']*1e6),float(row['duty_percent']),
                float(row['clock_high_chips']),float(row['corrected_high_chips']),int(row['symbol']),
                str(row['bits']),result.duty_thermometer(row),bool(row['training']),
                bool(row['structure_ok']),str(row['calibration_source']),str(row['reason']),str(row['raw_reason'])))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()

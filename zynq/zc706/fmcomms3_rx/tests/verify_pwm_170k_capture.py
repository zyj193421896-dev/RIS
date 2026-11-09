"""Offline audit of the user's 169.971671-kHz capture; no board access."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum import analyze_bin
from rx706_pwm import recover_pwm2
from rx706_pwm_protocol import duty_protocol


def verify_calibration_start(project, output):
    """Reproduce the 24%-target screenshot and audit the corrected settings."""
    spectrum = analyze_bin(project/'captures/rx_160k.bin', nfft=4096, averages=4,
        sample_rate_hz=15.36e6, first_tuple=round(.154535*15.36e6))
    options, _ = duty_protocol(20,7,8)
    audit = {}
    for label, target, count, auto in (('old_defaults',24,0,False),('corrected_defaults',20,8,True)):
        result = recover_pwm2(spectrum, symbol_hz=169971.671,duration_s=.02,
            allow_unlocked=True, **options,training_options=dict(
                preamble_widths=(target,)*3,payload_symbols=count,clock_width=True,
                same_tolerance_chips=.5,auto_scale=auto))
        rows = result.frames
        assert np.all(rows['training'][2:5])
        assert abs(np.median(rows['corrected_high_chips'][2:5])-target) < .01
        audit[label] = dict(description=result.calibration_description, first_rows=[])
        for row in rows[:16]:
            audit[label]['first_rows'].append({k:row[k].item() for k in
                ('index','start_s','duty_percent','corrected_high_chips','training','symbol','reason')})
        if auto:
            assert np.all(rows['training'][13:16])
            assert np.any(np.char.find(rows['reason'], '档位标定预热') >= 0)
            phase = int(np.flatnonzero(rows['training'])[0])
            position = (np.arange(len(rows))-phase)%11
            data = (position>=3) & (np.arange(len(rows))>256) & rows['valid']
            checked = int(data.sum())
            errors = int(np.count_nonzero(rows['symbol'][data] != position[data]-3))
            assert checked > 1500
            assert errors/checked < .01, (checked,errors)
            audit[label].update(checked_after_warmup=checked,mismatches_after_warmup=errors)
            with (output/'calibration_start_154552us.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(('index','time_us','raw_duty','corrected_duty','symbol','training','source','reason'))
                for row in rows[:500]:
                    writer.writerow((int(row['index']),float(row['start_s']*1e6),float(row['duty_percent']),
                        float(row['corrected_high_chips']),int(row['symbol']),bool(row['training']),
                        str(row['calibration_source']),str(row['reason'])))
    (output/'calibration_start_154552us.json').write_text(
        json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Screenshot calibration audit:', json.dumps({k:v.get('mismatches_after_warmup') for k,v in audit.items()}))


def main():
    project = Path(__file__).resolve().parents[1]
    source = project/'captures/rx_160k.bin'
    output = project/'reports/pwm_170k_20261104'
    output.mkdir(parents=True, exist_ok=True)
    spectrum = analyze_bin(source, nfft=4096, averages=4, sample_rate_hz=15.36e6)
    options, _ = duty_protocol(20, 7, 8)
    result = recover_pwm2(spectrum, symbol_hz=169971.671, duration_s=1.,
        allow_unlocked=True, **options,
        training_options=dict(preamble_widths=(20,20,20), payload_symbols=8,
            clock_width=True, same_tolerance_chips=.5, auto_scale=True))
    rows = result.frames
    # The capture begins with a complete three-pilot preamble; the fourth
    # equal-width pulse is payload 0 and must not be consumed as training.
    assert np.all(rows['training'][:3])
    assert not rows['training'][3] and rows['symbol'][3] == 0
    assert np.all(rows['training'][11:14])
    assert not rows['training'][14] and rows['symbol'][14] == 0
    # Reference is ONLY a post-decode check, never an input to recovery.
    indices = np.arange(len(rows))
    training_indices = indices[rows['training']]
    phase = int(training_indices[0] % 11)
    position = (indices-phase) % 11
    expected = position-3
    payload = position >= 3
    valid = rows['valid'] & payload
    warm = indices > 256
    summary = dict(source=str(source), symbol_hz=169971.671,
        locked=bool(result.decode_locked), rows=len(rows),
        training=int(rows['training'].sum()), valid_payload=int(valid.sum()),
        invalid_payload=int((payload & ~rows['valid']).sum()),
        nearest_outside_tolerance=int(np.count_nonzero(np.char.find(rows['reason'], '超容差') >= 0)),
        checked_after_warmup=int((valid & warm).sum()),
        mismatches_after_warmup=int((valid & warm & (rows['symbol'] != expected)).sum()),
        invalid_payload_after_warmup=int((payload & warm & ~rows['valid']).sum()),
        scale_range=result.training_calibration['scale_range'],
        spacing_rows=result.training_calibration['spacing_symbols'],
        reference_used_for_recovery=False)
    summary['first_preamble_rows'] = rows['index'][:3].tolist()
    summary['first_payload_row'] = int(rows['index'][3])
    summary['preamble_sampling_allowance_chips'] = result.training_calibration['measurement_uncertainty_chips']
    with (output/'symbols.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('index','start_s','raw_duty_percent','corrected_duty_percent',
            'symbol','binary_bits','thermometer_level_mapping','valid','training','reason'))
        for row in rows:
            writer.writerow((int(row['index']),float(row['start_s']),float(row['duty_percent']),
                float(row['corrected_high_chips']),int(row['symbol']),str(row['bits']),
                result.duty_thermometer(row),bool(row['valid']),bool(row['training']),str(row['reason'])))
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    verify_calibration_start(project, output)

if __name__ == '__main__':
    main()

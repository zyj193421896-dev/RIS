"""Exercise the actual fixed-directory button and production GUI worker."""
import json
from pathlib import Path
import sys
import tkinter as tk

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'host'))
from rx706_spectrum_gui import SpectrumWindow, PWM_DATA_DIR

root = tk.Tk()
root.withdraw()
view = SpectrumWindow(root)
view.window.withdraw()
errors = []
root.report_callback_exception = lambda *args: errors.append(str(args[1]))
try:
    view.fixed_pwm_button.invoke()
    assert Path(view.filename.get()) == PWM_DATA_DIR/'rx_test.bin'
    options = view.pwm2_protocol_options()
    options.update(symbol_hz=float(view.pwm_symbol_khz.get())*1000,
                   duration_s=.02, state_one='B', decimation=1)
    view._analyze_worker(view.filename.get(), dict(channel=1, nfft=4096, averages=4,
                         sample_rate_hz=15.36e6), options)
    view.poll()
    assert view.recovery is not None, view.status.get()
    result = view.recovery
    assert abs(result.pulse_options['user_information_rates'][0]-1027375.76865) < 1
    assert '前导计入' in view.summary.get()
    dialog = view.show_pwm_symbols()
    assert dialog is not None
    assert not errors, errors
    report = dict(filename=view.filename.get(), data_dir=str(PWM_DATA_DIR),
                  first_payload=int(result.frames['symbol'][~result.frames['training']][0]),
                  coded_information_rate=result.pulse_options['coded_information_rate'],
                  summary=view.summary.get(), callbacks=errors)
    (Path(__file__).parent/'gui_load.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'summary'}, ensure_ascii=False, indent=2))
finally:
    view.close()
    root.update_idletasks()
    root.destroy()

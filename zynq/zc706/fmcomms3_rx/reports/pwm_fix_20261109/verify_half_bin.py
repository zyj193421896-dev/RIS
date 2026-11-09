"""Real GUI replay and assertions for the two short pulses in the screenshot."""
import json
from pathlib import Path
import sys
import tkinter as tk

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'host'))
from rx706_spectrum_gui import SpectrumWindow, PWM_DATA_DIR

root=tk.Tk()
root.withdraw()
view=SpectrumWindow(root)
view.window.withdraw()
errors=[]
root.report_callback_exception=lambda *args:errors.append(str(args[1]))
try:
    view.fixed_pwm_button.invoke()
    assert view.load_frame_config(PWM_DATA_DIR/'frame.json')
    options=view.pwm2_protocol_options()
    options.update(symbol_hz=float(view.pwm_symbol_khz.get())*1000,duration_s=1.,state_one='B',decimation=1)
    view._analyze_worker(view.filename.get(),dict(channel=1,nfft=4096,averages=4,sample_rate_hz=15.36e6),options)
    view.poll()
    result=view.recovery
    assert result is not None,view.status.get()
    for index,symbol in ((56,51),(61,56)):
        row=result.frames[index-1]
        assert row['index']==index and row['valid'] and row['symbol']==symbol, str(row)
        assert row['decision_estimated']
    assert view.show_pwm_symbols() is not None
    assert not errors,errors
    report=dict(callback_errors=errors,minimum_width_boundary=4.,reference=result.frame_check.overall,
        rows=[{key:result.frames[index-1][key].item() for key in
               ('index','delay_chips','corrected_high_chips','symbol','bits','valid','decision_estimated','reason')}
              for index in (56,61)])
    (Path(__file__).parent/'current_gui.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
finally:
    view.close()
    root.update_idletasks()
    root.destroy()

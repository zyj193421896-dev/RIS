"""Load the experimental JSON through Tk and decode the existing IQ file."""
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

PROJECT = Path(__file__).resolve().parents[2]
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
    options.update(symbol_hz=float(view.pwm_symbol_khz.get())*1000,duration_s=.02,state_one='B',decimation=1)
    view._analyze_worker(view.filename.get(),dict(channel=1,nfft=4096,averages=4,sample_rate_hz=15.36e6),options)
    view.poll()
    result=view.recovery
    assert result is not None,view.status.get()
    assert result.pulse_options['width_quantization']=='nearest'
    assert result.pulse_options['cycle_calibration']['symbols']==[3,4,5,6]
    assert '循环已知码校准实验' in view.summary.get()
    pilots=result.frames['frame_part']=='校准前导'
    assert pilots.sum()>0
    assert all(result.frame_check.row_status[pilots]=='校准前导/不统计')
    dialog=view.show_pwm_symbols()
    assert dialog is not None
    assert not errors,errors
    report=dict(callback_errors=errors,reference=result.frame_check.overall,
                calibration_pilot_rows=int(pilots.sum()),summary=view.summary.get())
    (Path(__file__).parent/'cycle_gui.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='summary'},ensure_ascii=False,indent=2))
finally:
    view.close()
    root.update_idletasks()
    root.destroy()

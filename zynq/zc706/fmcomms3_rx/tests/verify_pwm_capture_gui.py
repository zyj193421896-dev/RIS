"""Exercise the actual 100 kHz capture using the GUI's default 200 ms settings."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=Path, default=ROOT/'captures/5p802_switch100k_m50dbm.bin')
args = parser.parse_args()
root = tk.Tk(); root.withdraw()
errors, notices = [], []
root.report_callback_exception = lambda *values: errors.append(str(values))
view = None
try:
    view = SpectrumWindow(root, args.capture)
    view.window.withdraw()
    assert view.pwm_symbol_khz.get() == '100'
    assert view.recovery_ms.get() == '200'
    assert view.pwm_polarity.get() == '自动（PWM约束）'
    with patch('rx706_spectrum_gui.messagebox.showwarning', side_effect=lambda *a, **kw: notices.append(str(a))):
        view.pwm_symbols_button.invoke()
        deadline = time.monotonic()+45
        while view.busy and time.monotonic() < deadline:
            root.update(); time.sleep(.01)
        root.update()
    assert not view.busy, 'Capture analysis timed out'
    assert not errors, errors
    assert not notices, notices
    assert view.recovery is not None, view.status.get()
    r = view.recovery
    assert r.kind == 'pwm'
    assert view._pwm_dialog is not None, 'Symbol table was not opened'
    view._pwm_dialog.withdraw()
    assert r.input_samples == round(.2*15360000)
    assert len(r.frames) > 19500
    assert int(r.frames['valid'].sum()) > 19500
    assert np.all(r.frames['symbol'][r.frames['valid']] == 4)
    assert abs(np.nanmedian(r.frames['one_us'])-5) < .1
    assert abs(np.nanmedian(r.frames['zero_us'])-5) < .1
    assert '真实控制 0/1 未标定' in r.mapping_note
    out = ROOT/'reports/pwm_capture_fix_20261029'
    out.mkdir(parents=True, exist_ok=True)
    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)
    buttons = {w.cget('text'): w for w in children(view._pwm_dialog) if w.winfo_class() == 'TButton'}
    with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/'capture_symbols.csv')), patch('tkinter.messagebox.showinfo'):
        buttons['导出全部 CSV…'].invoke()
    with (out/'capture_symbols.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = sum(1 for _ in csv.reader(stream))-1
    assert csv_rows == len(r.frames)
    view.figure.savefig(out/'capture_overview.png', dpi=130)
    view.set_plot_view('PWM 包络'); root.update()
    view.figure.savefig(out/'capture_envelope.png', dpi=130)
    result = dict(capture=str(args.capture), recovery_ms=200, code_chips=8, guard_chips=2,
        configured_symbol_khz=view.pwm_symbol_khz.get(), polarity='auto',
        carrier_hz=r.carrier_hz, measured_symbol_rate_hz=r.measured_symbol_rate_hz,
        frames=len(r.frames), valid=int(r.frames['valid'].sum()),
        invalid=int(np.count_nonzero(~r.frames['valid'])),
        median_T1_us=float(np.nanmedian(r.frames['one_us'])),
        median_T0_us=float(np.nanmedian(r.frames['zero_us'])),
        mapping_note=r.mapping_note, valid_symbols=np.unique(r.frames['symbol'][r.frames['valid']]).tolist(),
        csv_rows=csv_rows, callback_errors=errors, warning_dialogs=notices,
        symbol_table_opened=True, board_access=False)
    (out/'capture_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    if view is not None:
        view.close()
    root.destroy()

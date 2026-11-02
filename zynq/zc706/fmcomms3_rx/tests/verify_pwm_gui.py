"""Hidden real-Tk smoke test of PWM controls, worker, table, and plot export."""
from pathlib import Path
import csv
import json
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
sys.path.insert(0, str(ROOT/'tests'))
from rx706_spectrum_gui import SpectrumWindow, PWM_MODE
from test_rx706_pwm import PWMTests

fixture = PWMTests()
fixture.setUp()
root = tk.Tk()
root.withdraw()
errors = []
root.report_callback_exception = lambda *args: errors.append(str(args))
view = None
try:
    fixture.capture()
    view = SpectrumWindow(root, fixture.path)
    view.window.withdraw()
    view.rate.set(str(fixture.fs/1e6))
    view.pwm_symbol_khz.set(str(fixture.rate/1000))
    view.recovery_ms.set('10')
    view.nfft.set('4096')
    view.averages.set('1')
    assert view.pwm_guard_chips.get() == '2'
    # Reproduce the user's action directly, without first selecting PWM mode.
    assert view.constellation_mode.get() == '原始 IQ'
    view.pwm_symbols_button.invoke()
    assert view.constellation_mode.get() == PWM_MODE
    deadline = time.monotonic()+30
    while view.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    root.update()
    assert not errors, errors
    assert view.recovery is not None, view.status.get()
    assert view.recovery.kind == 'pwm'
    dialog = view._pwm_dialog
    assert dialog is not None, 'Button did not automatically open the PWM table'
    assert str(dialog.transient()) == str(view.window)
    view.pwm_symbols_button.invoke()
    assert view._pwm_dialog is dialog, 'Repeat click created a second window'
    for child in view.window.winfo_children():
        if isinstance(child, tk.Toplevel):
            child.withdraw()
    root.update()
    assert not errors, errors
    for name in ('PWM 包络', 'PWM 脉宽', '星座', '总览'):
        view.set_plot_view(name)
        root.update()
    out = ROOT/'reports/pwm_button_fix_20261029'
    out.mkdir(parents=True, exist_ok=True)
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    buttons = {widget.cget('text'): widget for widget in descendants(dialog)
               if widget.winfo_class() == 'TButton'}
    # A later edit to the file entry must not relabel an existing table/export.
    view.filename.set('different_file.bin')
    for label, name in [('导出全部 CSV…', 'pwm_symbols.csv'), ('导出分组比特流…', 'pwm_bits.txt')]:
        with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/name)) as save_dialog, patch('tkinter.messagebox.showinfo'):
            buttons[label].invoke()
            assert save_dialog.call_args.kwargs['initialfile'].startswith(fixture.path.stem+'_')
    with (out/'pwm_symbols.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows)-1 == len(view.recovery.frames)
    assert (out/'pwm_bits.txt').read_text(encoding='utf-8').count('|') == len(view.recovery.frames)-1
    assert not errors, errors
    view.figure.savefig(out/'pwm_overview.png', dpi=130)
    (out/'verification.json').write_text(json.dumps(dict(
        hidden_tk=True, board_access=False, callback_errors=errors,
        code_chips=view.recovery.code_chips, guard_chips=view.recovery.guard_chips,
        symbol_rate_hz=view.recovery.measured_symbol_rate_hz,
        sample_rate_hz=view.recovery.output_rate_hz,
        valid_symbols=int(view.recovery.frames['valid'].sum()),
        csv_rows=len(rows)-1, bitstream_export=True,
        parameter_controls=True, symbol_table=True, plot_navigation=True,
        direct_button_click=True, automatic_mode_switch=True,
        transient_popup=True, singleton_popup=True, pinned_export_name=True), indent=2), encoding='utf-8')
    print(view.status.get())
    print(view.summary.get())
    print('Tk PWM parameters, result table, view switching, and plot export: PASS')
finally:
    if view is not None:
        view.close()
    root.destroy()
    fixture.tearDown()

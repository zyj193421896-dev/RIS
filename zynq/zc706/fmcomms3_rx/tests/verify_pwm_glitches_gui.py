"""Replay the user's random PWM capture and check row 52 through real Tk/CSV."""
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
from rx706_pwm import recover_pwm

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=Path, default=ROOT/'captures/5p802_switch100k_m50dbm_random.bin')
args = parser.parse_args()
root = tk.Tk(); root.withdraw()
errors, notices = [], []
root.report_callback_exception = lambda *values: errors.append(str(values))
view = None
try:
    view = SpectrumWindow(root, args.capture)
    view.window.withdraw()
    assert view.pwm_glitch_chips.get() == '0.15'
    with patch('rx706_spectrum_gui.messagebox.showwarning', side_effect=lambda *a, **kw: notices.append(str(a))):
        view.pwm_symbols_button.invoke()
        deadline = time.monotonic()+45
        while view.busy and time.monotonic() < deadline:
            root.update(); time.sleep(.01)
        root.update()
    assert not view.busy and not errors and not notices, (errors, notices)
    assert view.recovery is not None, view.status.get()
    r = view.recovery
    dialog = view._pwm_dialog
    assert dialog is not None
    dialog.withdraw()
    row = r.frames[51]
    assert row['valid'] and row['symbol'] == 7 and row['bits'] == '111', row
    assert row['pattern'] == '1111111100'
    assert row['glitch_count'] >= 1 and row['warning'] != ''
    assert 10 < row['glitch_max_us']*1000 < 100
    strict = recover_pwm(view.result, symbol_hz=100000, duration_s=.2,
                         state_one='AB'[r.state_one], glitch_tolerance_chips=0)
    assert len(r.frames) == len(strict.frames)
    np.testing.assert_array_equal(r.frames['pattern'], strict.frames['pattern'])
    assert not strict.frames[51]['valid'] and strict.frames[51]['bits'] == '???'
    rescued = r.frames['valid'] & ~strict.frames['valid']
    assert np.count_nonzero(rescued) > 0
    assert np.all(r.frames['glitch_count'][rescued] > 0)
    for value in r.frames[rescued]:
        high = int(value['symbol'])+1
        assert value['pattern'] == '1'*high+'0'*(10-high)
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)
    widgets = list(descendants(dialog))
    tree = next(w for w in widgets if w.winfo_class() == 'Treeview')
    assert 'warning' in tree.item(tree.get_children()[51], 'tags')
    export = next(w for w in widgets if w.winfo_class() == 'TButton' and w.cget('text') == '导出全部 CSV…')
    out = ROOT/'reports/pwm_glitch_20261029'
    out.mkdir(parents=True, exist_ok=True)
    with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/'random_symbols.csv')), patch('tkinter.messagebox.showinfo'):
        export.invoke()
    with (out/'random_symbols.csv').open(encoding='utf-8-sig', newline='') as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == len(r.frames)
    assert exported[51]['信息比特'] == '111' and int(exported[51]['忽略毛刺数']) >= 1
    view.figure.savefig(out/'random_overview.png', dpi=130)
    result = dict(capture=str(args.capture), selected_duration_ms=200,
        configured_glitch_slots=r.glitch_tolerance_chips, per_symbol_budget_slots=.30,
        frames=len(r.frames), decodable=int(r.frames['valid'].sum()),
        strict_decodable=int(strict.frames['valid'].sum()),
        glitch_warnings=int(np.count_nonzero(r.frames['valid'] & (r.frames['glitch_count'] > 0))),
        rescued_with_warning=int(np.count_nonzero(rescued)),
        invalid=int(np.count_nonzero((~r.frames['valid']) & (r.frames['symbol'] != -2))),
        row52=dict(symbol=int(row['symbol']), bits=str(row['bits']), pattern=str(row['pattern']),
                   one_us=float(row['one_us']), zero_us=float(row['zero_us']),
                   glitch_count=int(row['glitch_count']), glitch_ns=float(row['glitch_us']*1000),
                   reason=str(row['reason']), warning=str(row['warning'])),
        original_slot_patterns_unchanged=True, amber_table_warning=True,
        csv_rows=len(exported), callback_errors=errors, warning_dialogs=notices,
        known_transmit_bits=False, measured_ber=None, board_access=False)
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
finally:
    if view is not None:
        view.close()
    root.destroy()

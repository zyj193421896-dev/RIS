"""Replay the known 7642 capture through real Tk controls; no board traffic."""
from pathlib import Path
import csv
import json
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow, PWM_MODE

capture = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path('D:/TDCM/ad9361/zynq/zc706/fmcomms3_rx/captures/5p802_switch100k_m50dbm_random.bin'))
out = ROOT/'reports/pwm_reference_20261029'
out.mkdir(parents=True, exist_ok=True)
root = tk.Tk()
root.withdraw()
errors = []
root.report_callback_exception = lambda *args: errors.append(str(args))
view = None


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


try:
    view = SpectrumWindow(root, capture)
    view.window.withdraw()
    assert not view.pwm_reference_enabled.get(), 'Other captures must not silently assume 7642'
    assert view.pwm_reference.get() == '7 6 4 2'
    view.pwm_reference_enabled.set(True)
    view.pwm_reference.set('7642')
    view.pwm_sync_symbols.set('64')
    view.pwm_symbols_button.invoke()
    deadline = time.monotonic()+45
    while view.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    root.update()
    assert not view.busy, 'Worker timeout'
    assert not errors, errors
    assert view.recovery is not None, view.status.get()
    assert view.constellation_mode.get() == PWM_MODE
    r = view.recovery
    c = r.reference_check
    assert c is not None and c.locked, view.summary.get()
    assert c.offset == 1
    assert c.sync_symbols == 64 and c.sync_matches == 64
    assert c.compared_symbols == 19760
    assert c.compared_bits == 59280 and c.bit_errors == 0 and c.symbol_errors == 0
    assert c.erasures == 174 and c.evaluation_symbols == 19934
    assert c.warned_symbols == 130 and c.warned_mismatches == 0
    assert 'BER（仅已判决比特）=0' in view.summary.get()
    dialog = view._pwm_dialog
    assert dialog is not None
    dialog.withdraw()
    tree = next(widget for widget in descendants(dialog) if widget.winfo_class() == 'Treeview')
    assert 'reference_status' in tree.cget('columns')
    buttons = {widget.cget('text'): widget for widget in descendants(dialog) if widget.winfo_class() == 'TButton'}
    for label, name in [('导出全部 CSV…', 'pwm_symbols.csv'), ('导出分组比特流…', 'pwm_bits.txt')]:
        with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/name)), patch('tkinter.messagebox.showinfo'):
            buttons[label].invoke()
    with (out/'pwm_symbols.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(r.frames)
    assert sum(row['序列校验'] == '擦除' for row in rows) == c.erasures
    assert sum(int(row['错比特数（同步不计入BER）'] or 0) for row in rows[64:]) == c.bit_errors
    assert sum(row['序列校验'].startswith('同步/') for row in rows) == 64
    row52 = next(row for row in rows if row['符号序号'] == '52')
    assert row52['符号值'] == row52['参考符号'] == '7'
    assert row52['信息比特'] == row52['参考比特'] == '111'
    assert row52['序列校验'] == '同步/匹配'
    assert row52['忽略毛刺数'] == '1'
    assert 'warning' in tree.item(tree.get_children()[51], 'tags')
    assert (out/'pwm_bits.txt').read_text(encoding='utf-8').count('|') == len(r.frames)-1
    view.figure.savefig(out/'pwm_reference_overview.png', dpi=120)
    report = dict(capture=str(capture), hidden_tk=True, board_access=False, callback_errors=errors,
                  frames=len(r.frames), sync_symbols=c.sync_symbols, phase=c.offset,
                  evaluation_symbols=c.evaluation_symbols, compared_symbols=c.compared_symbols,
                  compared_bits=c.compared_bits, symbol_errors=c.symbol_errors,
                  bit_errors=c.bit_errors, ber_decided=c.ber, erasures=c.erasures,
                  erasure_rate=c.erasures/c.evaluation_symbols,
                  warned_symbols=c.warned_symbols, warned_mismatches=c.warned_mismatches,
                  csv_rows=len(rows), csv_columns=len(rows[0]), row52=row52)
    (out/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(c.summary())
    print('Known-cycle controls, worker, table, CSV, raw bitstream and actual capture: PASS')
finally:
    if view is not None:
        view.close()
    root.destroy()

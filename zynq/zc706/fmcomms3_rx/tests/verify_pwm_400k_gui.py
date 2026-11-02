"""Real Tk replay of the 400k capture, retaining states when PWM code fails."""
from pathlib import Path
import csv
import json
import sys
import time
import tkinter as tk
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow

capture = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    'D:/TDCM/ad9361/zynq/zc706/fmcomms3_rx/captures/5p802_switch400k_m50dbm_3bit_random.bin')
out = ROOT/'reports/pwm_400k_20261029'
out.mkdir(parents=True, exist_ok=True)
root = tk.Tk()
root.withdraw()
errors = []
notices = []
root.report_callback_exception = lambda *args: errors.append(str(args))
view = None


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


try:
    view = SpectrumWindow(root, capture)
    view.window.withdraw()
    view.pwm_symbol_khz.set('400')
    assert view.pwm_decimation.get() == '1'
    assert not view.pwm_reference_enabled.get()
    with patch('tkinter.messagebox.showwarning', side_effect=lambda *args, **kw: notices.append(args)):
        view.pwm_symbols_button.invoke()
        deadline = time.monotonic()+45
        while view.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        root.update()
    assert not view.busy, 'Worker timed out'
    assert not errors and not notices, (errors, notices)
    r = view.recovery
    assert r is not None, view.status.get()
    assert r.output_rate_hz == 15.36e6
    assert '3.840' in r.sampling_note
    assert r.calibration_start_s == .001
    assert r.separation >= 3.5
    assert abs(r.carrier_hz-799700) < 100
    assert abs(r.measured_symbol_rate_hz-400000) < 40
    assert not r.decode_locked, 'This capture does not pass the original 8+2 PWM constraints'
    assert not np.any(r.frames['valid'])
    assert np.all(r.frames['bits'] == '???')
    assert np.count_nonzero(r.frames['candidate_symbol'] >= 0) > 50000
    assert r.reference_check is None, 'Do not assume the old 7642 ground truth'
    assert 'PWM编码未锁定' in view.status.get()
    assert 'NOT locked' in view.iq_axes.get_title()
    assert 'Relative level' == view.wave_axes.get_ylabel()
    assert '不据此计算BER' in view.summary.get()
    dialog = view._pwm_dialog
    assert dialog is not None
    dialog.withdraw()
    tree = next(widget for widget in descendants(dialog) if widget.winfo_class() == 'Treeview')
    assert 'candidate_symbol' in tree.cget('columns') and 'candidate_bits' in tree.cget('columns')
    buttons = {widget.cget('text'): widget for widget in descendants(dialog) if widget.winfo_class() == 'TButton'}
    with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/'pwm_widths.csv')), patch('tkinter.messagebox.showinfo'):
        buttons['导出全部 CSV…'].invoke()
    with (out/'pwm_widths.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(r.frames)
    assert all(row['信息比特'] == '???' and row['序列校验'] == '未启用' for row in rows)
    assert sum(row['候选符号（允许错误）'].isdigit() for row in rows) > 50000
    assert any(row['候选比特'] == '111' for row in rows)
    with patch('tkinter.filedialog.asksaveasfilename', return_value=str(out/'pwm_candidate_bits.txt')), patch('tkinter.messagebox.showinfo'):
        buttons['导出候选比特流…'].invoke()
    exported = (out/'pwm_candidate_bits.txt').read_text(encoding='utf-8')
    assert exported.count('|') == len(r.frames)-1
    assert '不用于BER' in exported and '111' in exported
    assert not errors, errors
    view.set_plot_view('总览')
    view.figure.savefig(out/'pwm_400k_overview.png', dpi=140)
    view.set_plot_view('星座')
    view.figure.savefig(out/'pwm_400k_constellation.png', dpi=140)
    report = dict(capture=str(capture), board_access=False, callback_errors=errors,
                  blocking_notices=notices, duration_s=r.input_samples/15.36e6,
                  sample_rate_hz=r.output_rate_hz, points_per_slot=r.output_rate_hz/(400000*10),
                  decimation=r.decimation, calibration_start_s=r.calibration_start_s,
                  carrier_hz=r.carrier_hz, separation=r.separation,
                  phase_difference_deg=r.phase_difference_deg,
                  candidate_symbol_rate_hz=r.measured_symbol_rate_hz,
                  frames=len(r.frames), decode_locked=r.decode_locked,
                  decode_note=r.decode_note, ber_enabled=False, csv_rows=len(rows))
    report.update(candidate_symbols=int(np.count_nonzero(r.frames['candidate_symbol'] >= 0)),
                  candidate_preview=r.frames['candidate_symbol'][:64].tolist(),
                  state_fit_note=r.state_fit_note, candidate_export=True)
    (out/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('400k real-Tk state display, candidate symbols/bits, CSV/TXT and BER disabled: PASS')
finally:
    if view is not None:
        view.close()
    root.destroy()

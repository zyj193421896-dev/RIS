"""Opt-in real Tk acceptance: selected capture + arbitrary pulse frame JSON.

Run with --capture BIN --profile JSON --output DIR. No board traffic.
Uses actual file/config buttons, analysis thread/poll, table and export buttons.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import faulthandler
import json
import sys
import time
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import SpectrumWindow


def children(widget):
    for child in widget.winfo_children():
        yield child
        yield from children(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--autotune', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gui_source_sha256 = hashlib.sha256((Path(__file__).resolve().parents[1]/'host/rx706_spectrum_gui.py').read_bytes()).hexdigest()
    faulthandler.dump_traceback_later(90, repeat=True)
    root = tk.Tk(); root.withdraw()
    errors = []
    root.report_callback_exception = lambda *values: errors.append(str(values))
    view = SpectrumWindow(root); view.window.withdraw()
    try:
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(args.capture)):
            view.browse()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(args.profile)), \
             patch('rx706_spectrum_gui.messagebox.showerror') as error:
            button = next(w for w in children(view.window) if isinstance(w, ttk.Button) and w.cget('text') == '载入帧配置…')
            button.invoke()
            error.assert_not_called()
        profile = json.loads(args.profile.read_text())
        assert view.pulse_config()['tolerance_chips'] == float(profile.get('tolerance_chips', .4))
        assert view.pwm_polarity.get() == '自动（PWM约束）'
        started = time.monotonic()
        if args.autotune:
            view.center_hint.set('100')  # Stale manually selected center must not leak into the fixed decode.
            view.autotune_button.invoke()
            first_thread = view.thread
            view.start_pwm_autotune()
            assert view.thread is first_thread, 'Duplicate click started another search'
        else:
            view.analyze_button.invoke()
        assert view.busy
        deadline = time.monotonic()+600
        def wait_for_result():
            if not view.busy or time.monotonic() >= deadline:
                root.quit()
            else:
                root.after(50, wait_for_result)
        root.after(50, wait_for_result)
        root.mainloop()
        assert not view.busy, 'GUI timed out'
        assert not errors, errors
        print('mainloop_returned', time.monotonic()-started, flush=True)
        r = view.recovery
        assert r is not None, view.status.get()
        assert str(view.analyze_button['state']) == 'normal'
        assert str(view.stop_button['state']) == 'disabled'
        print('analysis_complete', view.status.get(), flush=True)
        dialog = view.show_pwm_symbols(); dialog.withdraw()
        print('table_complete', flush=True)
        widgets = list(children(dialog))
        tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
        shown = tree.get_children()
        assert len(shown) == min(2000, len(r.frames))
        assert all(len(tree.item(item, 'values')) == len(tree['columns']) for item in shown)
        for label, name in [('导出全部 CSV…', 'symbols.csv'), ('导出分组比特流…', 'bits.txt'), ('导出逐块统计…', 'blocks.json')]:
            button = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == label)
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(args.output/name)), \
                 patch('rx706_spectrum_gui.messagebox.showinfo'), patch('rx706_spectrum_gui.messagebox.showerror') as error:
                button.invoke(); error.assert_not_called()
        with (args.output/'symbols.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.reader(stream); columns = len(next(rows)); count = 0
            for row in rows:
                assert len(row) == columns
                count += 1
        assert count == len(r.frames)
        exported = json.loads((args.output/'blocks.json').read_text())
        assert exported['overall'] == r.frame_check.overall
        assert exported['pulse_config']['tolerance_chips'] == view.pulse_config()['tolerance_chips']
        view.figure.savefig(args.output/'overview.png', dpi=110)
        result = dict(gui_source_sha256=gui_source_sha256, elapsed_s=time.monotonic()-started, source=str(args.capture), source_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
            profile=str(args.profile), profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
            summary=view.summary.get(), status=view.status.get(), overall=r.frame_check.overall,
            csv_rows=count, table_rows=len(shown), tolerance_chips=r.pulse_options['tolerance_chips'],
            interpretation='Conditional configured-reference experiment, not independently certified link BER.')
        if args.autotune:
            assert view._autotune_report is not None
            assert view.center_hint.get() == '自动'
            fixed = args.output/'fixed_profile.json'
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(fixed)):
                view.freeze_pwm_button.invoke()
            assert fixed.is_file() and fixed.with_suffix('.search.json').is_file()
            report = json.loads(fixed.with_suffix('.search.json').read_text())
            assert report['gui_post_selection_evaluation']['reference_overall'] == r.frame_check.overall
            polarity = view.pwm_polarity.get()
            tolerance = view.pulse_config()['tolerance_chips']
            assert view.load_frame_config(fixed)
            assert view.pwm_polarity.get() == polarity
            assert view.center_hint.get() == '自动'
            assert view.pulse_config()['tolerance_chips'] == tolerance
            result['autotune_selection'] = report['selection']['candidate']
        # Reloading a profile must retire old exports and clear omitted calibration.
        without = dict(profile); without.pop('cycle_calibration', None); without.pop('tolerance_chips', None)
        reset_profile = args.output/'reset_profile.json'; reset_profile.write_text(json.dumps(without))
        assert view.load_frame_config(reset_profile)
        assert view.recovery is None and view.result is None
        assert view._pwm_dialog is None
        assert view.pwm_cycle_calibration == {}
        assert view.pulse_config()['tolerance_chips'] == .4
        # Saved JSON must retain the candidate tolerance for a future GUI session.
        view.load_frame_config(args.profile)
        saved = args.output/'saved_profile.json'
        with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(saved)):
            view.save_frame_config()
        assert json.loads(saved.read_text())['tolerance_chips'] == float(profile.get('tolerance_chips', .4))
        view.pwm_polarity.set('B=1'); view.filename.set('next_capture.bin')
        assert view.pwm_polarity.get() == '自动（PWM约束）'
        assert not errors, errors
        result['state_reset_passed'] = True
        (args.output/'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))
    finally:
        faulthandler.cancel_dump_traceback_later()
        view.close(); root.destroy()


if __name__ == '__main__':
    main()

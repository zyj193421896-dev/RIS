"""Opt-in real Tk negative test: valid but deliberately mismatched frame JSON."""
from pathlib import Path
import argparse
import json
import sys
import time
import tkinter as tk
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import SpectrumWindow

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--profile', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = tk.Tk(); root.withdraw()
errors = []; root.report_callback_exception = lambda *a: errors.append(str(a))
view = SpectrumWindow(root); view.window.withdraw()
try:
    view.filename.set(str(args.capture))
    assert view.load_frame_config(args.profile)
    view.recovery_ms.set('30')
    view.result = view.recovery = object()  # Deliberately seed an obsolete result.
    view.analyze_button.invoke()
    assert view.recovery is None
    deadline = time.monotonic()+60
    def complete():
        if not view.busy or time.monotonic() > deadline:
            root.quit()
        else:
            root.after(25, complete)
    root.after(25, complete); root.mainloop()
    assert not view.busy and not errors, errors
    assert view.result is not None, view.status.get()
    r = view.recovery
    assert r is None or not r.decode_locked or not r.frame_check.locked
    if r is not None and r.frame_check is not None:
        assert r.frame_check.overall['ber'] is None
    result = dict(capture=str(args.capture), deliberately_mismatched_profile=str(args.profile),
        duration_ms=30, status=view.status.get(), summary=view.summary.get(),
        recovery_present=r is not None, callback_errors=errors, stale_result_cleared=True,
        interpretation='Negative test only: this new-transmitter profile does not apply to the old capture.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
finally:
    view.close(); root.destroy()

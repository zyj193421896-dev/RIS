"""Run the portable receiver host regression suite, without real hardware.

Legacy network C/ctypes tests and FPGA register/header tests have separate
entry points and are intentionally outside this host acceptance suite.
"""
from pathlib import Path
import sys
import unittest

TESTS = Path(__file__).resolve().parent
PATTERNS = (
    "test_four_corner_training.py", "test_gui_layout.py", "test_plot_navigation.py",
    "test_pwm*.py", "test_qam*.py", "test_receiver*.py",
    "test_rx706_pwm.py", "test_rx706_qam.py", "test_rx706_recovery.py",
    "test_rx706_spectrum.py", "test_signal_analysis_view.py",
    "test_symbol_table_workflow.py", "test_tdcm*.py",
)


def main():
    sys.path.insert(0, str(TESTS))
    names = sorted({p.stem for pattern in PATTERNS for p in TESTS.glob(pattern)})
    print(f"Receiver host acceptance: {len(names)} modules; synthetic IQ and Tk, no real board.", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

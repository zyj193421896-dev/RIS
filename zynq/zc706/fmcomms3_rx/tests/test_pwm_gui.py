"""PWM worker delivery and actual plotted control/width data, without hardware."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from test_signal_analysis_view import headless_window
import test_rx706_pwm as fixtures
from rx706_spectrum_gui import PWM_MODE


class PWMViewTests(unittest.TestCase):
    def test_worker_checks_known_cycle_without_losing_recovery_on_bad_reference(self):
        import numpy as np
        fixture = fixtures.PWMTests()
        fixture.setUp()
        try:
            fixture.capture(widths=np.tile([8, 7, 5, 3], 70))
            for sequence, locked in [((7, 6, 4, 2), True), ((0, 1, 3, 5), False)]:
                view = headless_window()
                view.constellation_mode.set(PWM_MODE)
                view._analyze_worker(fixture.path,
                    dict(nfft=4096, averages=1, sample_rate_hz=fixture.fs),
                    dict(kind='pwm', symbol_hz=fixture.rate, duration_s=.01,
                         reference_options=dict(sequence=sequence, sync_symbols=64)))
                view.poll()
                self.assertIsNotNone(view.recovery, view.status.get())
                self.assertEqual(view.recovery.reference_check.locked, locked)
                if locked:
                    self.assertEqual(view.recovery.reference_check.bit_errors, 0)
                    self.assertIn('BER（仅已判决比特）=0', view.summary.get())
                else:
                    self.assertIn('不报告 BER', view.summary.get())
        finally:
            fixture.tearDown()

    def test_button_starts_pwm_from_raw_or_periodic_view(self):
        for recovery in (None, SimpleNamespace(kind='periodic')):
            with self.subTest(recovery=recovery):
                view = headless_window()
                view.busy = False
                view.recovery = recovery
                view.filename.set('capture.bin')
                view.analyze = Mock(side_effect=lambda: setattr(view, 'busy', True))
                view.show_pwm_symbols()
                self.assertEqual(view.constellation_mode.get(), PWM_MODE)
                self.assertTrue(view._open_pwm_after_analysis)
                view.analyze.assert_called_once()

    def test_busy_click_and_invalid_parameters_have_visible_feedback(self):
        view = headless_window()
        view.constellation_mode.set(PWM_MODE)
        with patch('rx706_spectrum_gui.messagebox.showinfo') as notice:
            view.show_pwm_symbols()
        notice.assert_called_once()
        self.assertTrue(view._open_pwm_after_analysis)
        view.busy = False
        view.filename.set('capture.bin')
        view.status.set('参数错误：符号率无效')
        view.analyze = Mock()
        with patch('rx706_spectrum_gui.messagebox.showwarning') as warning:
            view.show_pwm_symbols()
        warning.assert_called_once()
        self.assertIn('符号率', warning.call_args.args[1])
        self.assertFalse(view._open_pwm_after_analysis)

    def test_failed_background_analysis_opens_explanation(self):
        view = headless_window()
        view._open_pwm_after_analysis = True
        view.events.put(('error', '符号率与数据不匹配'))
        with patch('rx706_spectrum_gui.messagebox.showwarning') as warning:
            view.poll()
        warning.assert_called_once()
        self.assertIn('符号率与数据不匹配', warning.call_args.args[1])
        self.assertFalse(view._open_pwm_after_analysis)

    def test_worker_renders_levels_widths_and_keeps_erasure_summary(self):
        fixture = fixtures.PWMTests()
        fixture.setUp()
        try:
            fixture.capture()
            view = headless_window()
            view.constellation_mode.set(PWM_MODE)
            view._open_pwm_after_analysis = True
            view.show_pwm_symbols = Mock()
            view._analyze_worker(fixture.path,
                dict(nfft=4096, averages=1, sample_rate_hz=fixture.fs),
                dict(kind='pwm', symbol_hz=fixture.rate, guard_chips=2, duration_s=.01))
            view.poll()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertEqual(view.recovery.kind, 'pwm')
            view.show_pwm_symbols.assert_called_once()
            self.assertFalse(view._open_pwm_after_analysis)
            self.assertIn('8+2片', view.status.get())
            self.assertIn('分组比特流预览', view.summary.get())
            self.assertIn('Control level', view.wave_axes.get_ylabel())
            self.assertFalse(hasattr(view, 'phase_axes'))
            self.assertEqual(set(view.wave_axes.lines[1].get_ydata()), {0, 1})
            self.assertLessEqual(len(view.recovery.scatter), 8192)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()

"""Regression: general frame JSON options and result ownership in actual Tk."""
from pathlib import Path
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import test_pwm_frames_gui as fixtures


class GeneralFrameJSONTests(unittest.TestCase):
    setUp = fixtures.FrameGUITests.setUp
    tearDown = fixtures.FrameGUITests.tearDown

    def test_tolerance_saved_loaded_and_not_inherited(self):
        view = self.view
        profile = Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json'
        config = json.loads(profile.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp)/'selected.json'
            config['tolerance_chips'] = .49
            selected.write_text(json.dumps(config))
            self.assertTrue(view.load_frame_config(selected))
            self.assertEqual(view.pulse_config()['tolerance_chips'], .49)
            saved = Path(tmp)/'saved.json'
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(saved)):
                view.save_frame_config()
            self.assertEqual(json.loads(saved.read_text())['tolerance_chips'], .49)
            self.assertTrue(view.load_frame_config(profile))
            self.assertEqual(view.pulse_config()['tolerance_chips'], .4)

    def test_file_and_profile_change_clear_results(self):
        view = self.view
        view.result = object(); view.recovery = object()
        view.pwm_polarity.set('B=1')
        view.filename.set('another.bin')
        self.assertIsNone(view.result)
        self.assertIsNone(view.recovery)
        self.assertEqual(view.pwm_polarity.get(), '自动（PWM约束）')
        view.result = object(); view.recovery = object()
        profile = Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json'
        self.assertTrue(view.load_frame_config(profile))
        self.assertIsNone(view.result)
        self.assertIsNone(view.recovery)

    def test_new_preamble_profile_clears_cycle_and_preserves_layout(self):
        host = Path(__file__).resolve().parents[1]/'host'
        view = self.view
        view.load_frame_config(host/'gui_data/current_pwm/frame.json')
        self.assertTrue(view.pwm_cycle_calibration)
        profile = host/'frame_profiles/new_capture_preamble12_20261110/frame_new_capture_preamble12.json'
        self.assertTrue(view.load_frame_config(profile))
        actual = view.pulse_config()
        self.assertEqual(actual['cycle_calibration'], {})
        self.assertEqual(actual['frame_options']['users'][0]['identity'], (0,65,8,54,2,42,10,21,6,60,4,35))
        self.assertEqual(actual['frame_options']['users'][0]['payload_symbols'], 66)
        self.assertFalse(actual['frame_options']['reference_enabled'])
        self.assertEqual(actual['symbol_count'], 66)

    def test_duplicate_analysis_and_cancel_controls(self):
        view = self.view
        view.filename.set('capture.bin')
        view.show_advanced.set(True); view.refresh_parameter_visibility()
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            view.workflows.expert_button.invoke()
            self.assertTrue(view.busy)
            view.analyze_button.invoke()
            view.analyze()
            self.assertEqual(thread.call_count, 1)
            self.assertEqual(str(view.analyze_button['state']), 'disabled')
            view.stop_button.invoke()
            self.assertTrue(view.cancel.is_set())
        view.busy = False

    def _pump_until_idle(self):
        deadline = time.monotonic()+5
        def check():
            if not self.view.busy or time.monotonic() > deadline:
                self.root.quit()
            else:
                self.root.after(10, check)
        self.root.after(10, check)
        self.root.mainloop()
        self.assertFalse(self.view.busy)

    def test_autotune_cancel_duplicate_and_stale_input(self):
        from rx706_spectrum import AnalysisCancelled
        view = self.view
        profile = Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json'
        view.filename.set('capture.bin'); view.load_frame_config(profile)
        def cancel_search(*args, **kwargs):
            self.assertTrue(kwargs['cancel_event'].wait(3))
            raise AnalysisCancelled('cancelled')
        with patch('rx706_pwm_autotune.search', side_effect=cancel_search) as search:
            view.autotune_button.invoke()
            first = view.thread
            view.start_pwm_autotune()
            self.assertIs(view.thread, first)
            view.stop_button.invoke()
            self._pump_until_idle()
            self.assertEqual(search.call_count, 1)
            self.assertIsNone(view._autotune_report)
            self.assertIn('取消', view.status.get())
        with patch('rx706_pwm_autotune.search', side_effect=cancel_search):
            view.start_pwm_autotune()
            view.filename.set('new_capture.bin')
            self._pump_until_idle()
            self.assertIsNone(view._autotune_report)
            self.assertIsNone(view.recovery)
            self.assertIn('过期', view.status.get())

    def test_reference_toggle_invalidates_fixed_search_report(self):
        view = self.view
        view._autotune_report = {'old': True}
        view.result = view.recovery = object()
        view.pwm_frame_reference_enabled.set(not view.pwm_frame_reference_enabled.get())
        self.assertIsNone(view._autotune_report)
        self.assertIsNone(view.recovery)
        self.assertIsNone(view.result)

    def test_selected_profile_restores_center_and_reference_on_save_reload(self):
        view = self.view
        profile = Path(__file__).resolve().parents[1]/'host/gui_data/current_pwm/frame.json'
        original = json.loads(profile.read_text())
        view.filename.set('capture.bin'); view.load_frame_config(profile)
        view.center_hint.set('100')
        report = dict(original_profile=original, selected_profile={}, channel=1,
            provenance=dict(source_sha256='0'*64, duration_s=1),
            selection=dict(candidate=dict(symbol_hz=169971.671, tolerance_chips=.49, state_one='A'), metrics={}))
        token = view._autotune_token
        with patch.object(view, 'analyze') as analyze, patch('rx706_pwm_autotune.sha256', return_value='0'*64):
            view._handle_autotune_event('autotune_done', (token, report, None))
            analyze.assert_called_once()
        self.assertEqual(view.center_hint.get(), '自动')
        self.assertEqual(view.pwm_polarity.get(), 'A=1')
        self.assertTrue(view.pwm_frame_reference_enabled.get())
        self.assertIs(view._autotune_report, report)
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp)/'fixed.json'
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(saved)):
                view.save_pwm_autotune()
            self.assertTrue(saved.with_suffix('.search.json').is_file())
            view.center_hint.set('100')
            with patch('rx706_pwm_autotune.sha256', return_value='0'*64):
                self.assertTrue(view.load_frame_config(saved))
            self.assertEqual(view.center_hint.get(), '自动')
            self.assertEqual(view.pwm_polarity.get(), 'A=1')
            self.assertEqual(view.pulse_config()['tolerance_chips'], .49)
            self.assertTrue(view.pwm_frame_reference_enabled.get())

    def test_no_reference_summary_still_reports_decision_erasures(self):
        from types import SimpleNamespace
        import math
        import numpy as np
        from test_pwm_pulse import config, synthesize
        from rx706_pwm_pulse import decode_pulses
        from rx706_pwm_frames import check_frame_reference
        options, sent, rises, falls, times, states, origin = synthesize(config(payload=32), rounds=4)
        edges = np.sort(np.r_[rises, falls]); after = np.isin(edges, rises)
        rows, rate, calibration, normalized = decode_pulses(states, edges, after, times,
            15.36e6, 169971.671, options)
        payload = rows['block_complete'] & (rows['frame_part'] == '净荷')
        rows['valid'][np.flatnonzero(payload)[:5]] = False
        widths = tuple(w for d, w in normalized['codebook'])
        bits = math.ceil(math.log2(len(widths)))
        bitwords = [f'{i:0{bits}b}' for i in range(len(widths))]
        check = check_frame_reference(rows, normalized['frame_options'], widths, bitwords)
        result = SimpleNamespace(frames=rows, pulse_options=normalized, measured_symbol_rate_hz=rate,
            frame_check=check, symbol_widths=widths, symbol_bitwords=bitwords, bits_per_symbol=bits)
        summary = self.view._pwm_file_summary(result)
        self.assertIn('未测', summary)
        self.assertIn('擦除5', summary)
        self.assertIn('参考覆盖擦除0', summary)

    def test_nonobject_json_reports_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'array.json'; path.write_text('[]')
            with patch('rx706_spectrum_gui.messagebox.showerror') as error:
                self.assertFalse(self.view.load_frame_config(path))
                error.assert_called_once()

    def test_invalid_json_does_not_destroy_previous_configuration(self):
        view = self.view
        old = view.pulse_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.json'; path.write_text('{bad')
            with patch('rx706_spectrum_gui.messagebox.showerror') as error:
                self.assertFalse(view.load_frame_config(path))
                error.assert_called_once()
        self.assertEqual(view.pulse_config(), old)


if __name__ == '__main__':
    unittest.main()

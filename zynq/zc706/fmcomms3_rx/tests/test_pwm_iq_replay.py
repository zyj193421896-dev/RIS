"""PWM bound replay, metadata safety and GUI dispatch without hardware or Tk."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
from rx706_pwm_replay import (make_recipe, validate_recipe, read_recipe, verify_input,
                               run_recipe, result_summary, fingerprint)
from rx706_spectrum import AnalysisCancelled
from test_qam64_controls import controls
from test_signal_analysis_view import Value
from test_pwm_pulse import config, synthesize


def window():
    view = controls()
    view.pwm_duty_summary = Value('')
    view.pwm2_layout_summary = Value('')
    view.show_pwm_symbols = Mock()
    return view


class PWMReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name).resolve() / 'capture.bin'
        c, _, _, _, t, high, _ = synthesize(config(payload=32), rounds=6, tail_us=.2)
        z = (np.where(high, 600 + 250j, -450 - 200j) * np.exp(2j * np.pi * 700000 * t))
        np.rint(np.column_stack([z.real, z.imag, z.real, z.imag])).astype('<i2').tofile(self.path)
        self.recipe = make_recipe(self.path, c, sample_rate_hz=15.36e6, symbol_hz=169971.671,
                                  duration_s=len(t) / 15.36e6, center_hint_hz=700000,
                                  nfft=256, averages=2)
        self.json_path = Path(self.tmp.name) / 'replay.json'
        self.json_path.write_text(json.dumps(self.recipe), encoding='utf-8')

    def metadata(self, rate=15.36e6):
        size = self.path.stat().st_size
        value = dict(complete=True, file_bytes=size, capture=dict(bytes=size, tuples=size // 8,
                     order=['I1', 'Q1', 'I2', 'Q2'], dtype='int16', byte_order='little',
                     channels=2, valid_bits=12, sample_rate_hz=rate))
        Path(str(self.path) + '.json').write_text(json.dumps(value))

    def test_round_trip_and_defaults(self):
        r = read_recipe(self.json_path)
        self.assertEqual(r, self.recipe)
        self.assertEqual(verify_input(r), self.path)
        self.assertFalse(r['recovery']['pulse_options']['frame_options']['reference_enabled'])
        self.assertEqual(r['assumptions']['preamble_status'], 'hypothesis')

    def test_bad_binding_missing_and_changed(self):
        original = self.path.read_bytes()
        self.path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            verify_input(self.recipe)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, '不存在'):
            verify_input(self.recipe)

    def test_sidecar_rate_mismatch_stops_before_recovery(self):
        self.metadata(20e6)
        with patch('rx706_pwm.recover_pwm') as recover:
            with self.assertRaisesRegex(ValueError, '采样率'):
                run_recipe(self.recipe)
            recover.assert_not_called()

    def test_invalid_sidecar_is_not_ignored(self):
        Path(str(self.path) + '.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, '同名 JSON 无效'):
            run_recipe(self.recipe)

    def test_reject_invalid_ranges_types_and_unverified_reference(self):
        mutations = [lambda r: r['spectrum'].update(channel=True),
                     lambda r: r['recovery'].update(duration_s=2),
                     lambda r: r['recovery'].update(center_hint_hz=float('nan')),
                     lambda r: r['recovery'].update(symbol_hz=1e9),
                     lambda r: r['recovery'].update(decimation=True),
                     lambda r: r['spectrum'].update(first_tuple=999999),
                     lambda r: r['recovery']['pulse_options']['frame_options'].update(reference_enabled=True)]
        for mutate in mutations:
            value = copy.deepcopy(self.recipe)
            mutate(value)
            with self.assertRaises(ValueError):
                validate_recipe(value)

    def test_external_reference_never_read(self):
        value = copy.deepcopy(self.recipe)
        value['recovery']['pulse_options']['frame_options']['users'][0]['reference'] = '@/missing.txt'
        with patch('pathlib.Path.read_text', side_effect=AssertionError('external reference access')):
            with self.assertRaisesRegex(ValueError, '外部@'):
                validate_recipe(value)

    def test_cancel_hash_and_no_worker_recovery(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(AnalysisCancelled):
            fingerprint(self.path, cancel)
        with patch('rx706_pwm.recover_pwm') as recover:
            with self.assertRaises(AnalysisCancelled):
                run_recipe(self.recipe, cancel)
            recover.assert_not_called()

    def test_real_pipeline_and_reference_independent_decisions(self):
        spectrum, a = run_recipe(self.recipe)
        report = result_summary(spectrum, a)
        json.dumps(report, allow_nan=False)
        self.assertGreater(len(a.frames), 0)
        self.assertIsNone(report['frame_check']['overall']['ber'])
        value = copy.deepcopy(self.recipe)
        value['assumptions']['reference_status'] = 'confirmed'
        frames = value['recovery']['pulse_options']['frame_options']
        frames['reference_enabled'] = True
        frames['users'][0]['reference'] = '0 1 2 3'
        from rx706_pwm import recover_pwm
        with patch('rx706_pwm.recover_pwm', wraps=recover_pwm) as recover:
            _, b = run_recipe(value)
            passed = recover.call_args.kwargs['pulse_options']['frame_options']
            self.assertFalse(passed['reference_enabled'])
            self.assertEqual(passed['users'][0]['reference'], '')
            self.assertTrue(recover.call_args.kwargs['allow_unlocked'])
        np.testing.assert_array_equal(a.frames['symbol'], b.frames['symbol'])
        np.testing.assert_array_equal(a.frames['valid'], b.frames['valid'])
        self.assertEqual(a.measured_symbol_rate_hz, b.measured_symbol_rate_hz)
        self.assertEqual(a.pulse_calibrations, b.pulse_calibrations)
        self.assertTrue(b.frame_options['reference_enabled'])

    def test_change_during_recovery_discards_result(self):
        def mutate(*args, **kwargs):
            self.path.write_bytes(bytes(self.path.stat().st_size))
            return SimpleNamespace()
        with patch('rx706_pwm.recover_pwm', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, '变化'):
                run_recipe(self.recipe)

    def test_added_sidecar_during_recovery_discards_result(self):
        def mutate(*args, **kwargs):
            self.metadata()
            return SimpleNamespace()
        with patch('rx706_pwm.recover_pwm', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, '变化'):
                run_recipe(self.recipe)

    def test_gui_async_load_mode_dispatch_repeat_failure_and_cancel(self):
        v = window()
        v.constellation_mode.set('TDCM 谐波QAM')
        old_filename = v.filename.get()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(self.json_path)), \
             patch('rx706_spectrum_gui.threading.Thread') as thread, \
             patch('rx706_pwm_replay.fingerprint', side_effect=AssertionError('hash on Tk thread')):
            v.load_pwm_recipe()
            self.assertEqual(v.filename.get(), old_filename)
            self.assertTrue(v.busy)
            v.load_pwm_recipe()
            thread.assert_called_once()
            recipe = thread.call_args.kwargs['args'][0]
        v._pwm_recipe_worker(recipe)
        while not v.events.empty():
            v.poll()
        self.assertEqual(v.constellation_mode.get(), 'PWM（占空比）')
        self.assertEqual(v.filename.get(), str(self.path))
        self.assertIsNotNone(v.recovery, v.status.get())
        self.assertIn('未经发端确认', v.summary.get())
        self.assertFalse(v.pwm_frame_reference_enabled.get())
        self.assertTrue(len(v.iq_axes.collections) > 0)
        self.path.unlink()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(self.json_path)), \
             patch('rx706_spectrum_gui.threading.Thread'):
            v.load_pwm_recipe()
        self.assertIsNone(v.result)
        self.assertIsNone(v.recovery)
        v._pwm_recipe_worker(recipe)
        while not v.events.empty():
            v.poll()
        self.assertIn('不存在', v.status.get())
        self.assertFalse(v.busy)
        v.cancel.set()
        # A closed window ignores all queued worker notifications.
        v.closed = True
        v.events.put(('error', 'should not render'))
        old = v.status.get()
        v.poll()
        self.assertEqual(v.status.get(), old)

    def test_replay_preserves_visible_table_until_success_and_refreshes_it(self):
        v = window()
        old = Mock()
        old.winfo_exists.return_value = True
        v._pwm_dialog = old
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(self.json_path)), \
             patch('rx706_spectrum_gui.threading.Thread'):
            v.load_pwm_recipe()
        old.destroy.assert_not_called()
        self.assertIn('上次', old.title.call_args.args[0])
        self.assertTrue(v._open_pwm_after_analysis)
        v._pwm_recipe_worker(read_recipe(self.json_path))
        while not v.events.empty():
            v.poll()
        v.show_pwm_symbols.assert_called_once()
        old.destroy.assert_not_called()  # Mock refresh; real method replaces only now.

    def test_failed_replay_keeps_old_table_and_reports_failure(self):
        v = window()
        old = Mock()
        old.winfo_exists.return_value = True
        v._pwm_dialog = old
        with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(self.json_path)), \
             patch('rx706_spectrum_gui.threading.Thread'):
            v.load_pwm_recipe()
        self.path.unlink()
        v._pwm_recipe_worker(read_recipe(self.json_path))
        with patch('rx706_spectrum_gui.messagebox.showwarning') as warning:
            while not v.events.empty():
                v.poll()
        old.destroy.assert_not_called()
        v.show_pwm_symbols.assert_not_called()
        warning.assert_called_once()
        self.assertFalse(v.busy)
        self.assertIn('不存在', v.status.get())

    def test_export_snapshot_ignores_edits_and_preserves_previous_result(self):
        v = window()
        v.result, v.recovery = run_recipe(self.recipe)
        v._last_pwm_recipe = self.recipe
        v._last_iq_request = (str(self.path), self.recipe['spectrum'], dict(kind='pwm', **self.recipe['recovery']))
        v.pwm_symbol_khz.set('999')
        output = Path(self.tmp.name) / 'saved.json'
        with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(output)), \
             patch('rx706_spectrum_gui.threading.Thread') as worker, \
             patch('rx706_pwm_replay.fingerprint', side_effect=AssertionError('hash on Tk thread')):
            v.save_pwm_recipe()
            args = worker.call_args.kwargs['args']
        old_result = v.result
        v._save_pwm_recipe_worker(*args)
        v.poll()
        self.assertEqual(read_recipe(output), self.recipe)
        self.assertIs(v.result, old_result)
        self.assertFalse(v.busy)
        # A repeated export must not overwrite the already saved recipe.
        v._save_pwm_recipe_worker(*args)
        v.poll()
        self.assertIn('保存未完成', v.status.get())
        self.assertEqual(read_recipe(output), self.recipe)

    def test_export_changed_source_refused(self):
        v = window()
        sp, _ = run_recipe(self.recipe)
        self.path.write_bytes(bytes(self.path.stat().st_size))
        output = Path(self.tmp.name) / 'saved.json'
        request = (str(self.path), self.recipe['spectrum'], self.recipe['recovery'])
        v._save_pwm_recipe_worker(request, sp._source_stat, self.recipe, str(output))
        kind, error = v.events.get_nowait()
        self.assertEqual(kind, 'pwm_recipe_save_error')
        self.assertIn('变化', error)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

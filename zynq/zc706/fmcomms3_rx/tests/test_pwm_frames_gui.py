"""Real Tk configuration, worker, per-user filtering and complete exports."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import SpectrumWindow, PWM2_MODE
from test_pwm_frames import configuration, sent_symbols


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


class FrameGUITests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(args)
        self.view = SpectrumWindow(self.root)
        self.view.window.withdraw()

    def tearDown(self):
        if hasattr(self, 'view'):
            self.view.close()
            self.root.update_idletasks()
            self.root.destroy()
            self.assertEqual(self.callback_errors, [])

    def test_fixed_data_button_loads_matching_configuration_without_file_dialog(self):
        view = self.view
        host = Path(__file__).resolve().parents[1]/'host'
        profile = json.loads((host/'gui_data/current_pwm/frame.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'frame.json').write_text(json.dumps(profile, ensure_ascii=False), encoding='utf-8')
            (folder/'rx_test.bin').write_bytes(b'\0'*8)
            latest = folder/'rx_20261109_235959_999.bin'
            latest.write_bytes(b'\0'*8)
            import os
            os.utime(folder/'rx_test.bin', (100, 100))
            os.utime(latest, (200, 200))
            with patch('rx706_spectrum_gui.PWM_DATA_DIR', folder), \
                 patch('rx706_spectrum_gui.filedialog.askopenfilename') as picker, \
                 patch('rx706_spectrum_gui.messagebox.showerror') as error:
                view.fixed_pwm_button.invoke()
                error.assert_not_called()
                picker.assert_not_called()
            self.assertEqual(Path(view.filename.get()), latest)
            self.assertEqual(view.pwm_polarity.get(), 'B=1')
            self.assertEqual(view.first_tuple.get(), '0')
            self.assertEqual(view.recovery_ms.get(), '1000')
            c = view.pulse_config()
            self.assertEqual(c['symbol_count'], 66)
            self.assertEqual(c['frame_options']['users'][0]['identity'], (0, 0, 0, 0))
            self.assertAlmostEqual(c['user_information_rates'][0], 169971.671*np.log2(66))
            self.assertIn('前导计入', view.pwm_duty_summary.get())

    def test_fixed_data_missing_pair_does_not_change_current_file(self):
        view = self.view
        view.filename.set('keep.bin')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('rx706_spectrum_gui.PWM_DATA_DIR', Path(tmp)), \
             patch('rx706_spectrum_gui.messagebox.showerror') as error:
            view.load_fixed_pwm_data()
            error.assert_called_once()
        self.assertEqual(view.filename.get(), 'keep.bin')

    def test_cycle_calibration_and_quantizer_survive_gui_save_load_and_worker_options(self):
        host = Path(__file__).resolve().parents[1]/'host'
        view = self.view
        view.load_frame_config(host/'gui_data/current_pwm/frame.json')
        self.assertIn('循环已知码校准实验',view.pwm_duty_summary.get())
        self.assertEqual(view.pwm2_protocol_options()['pulse_options']['cycle_calibration']['payload_indices'],[3,4,5,6])
        view.pwm_width_quantization.set('脉宽向下取整')
        self.assertEqual(view.pulse_config()['width_quantization'],'floor_width')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'cycle.json'
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename',return_value=str(target)):
                view.save_frame_config()
            saved = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(saved['width_quantization'],'floor_width')
            self.assertEqual(saved['cycle_calibration']['symbols'],[3,4,5,6])
            view.load_frame_config(target)
            self.assertEqual(view.pulse_config()['width_quantization'],'floor_width')
        view.load_frame_config(host/'gui_data/current_pwm/frame_original.json')
        self.assertEqual(view.pwm_cycle_calibration,{})
        self.assertEqual(view.pulse_config()['width_quantization'],'nearest')

    def test_pulse_rates_steps_and_removed_common_prefix_controls(self):
        import math
        view = self.view
        c = view.pulse_config()
        self.assertEqual(c['total_chips'], 20)
        self.assertEqual(c['symbol_count'], 105)
        self.assertIn('6.714246', view.pwm_duty_summary.get())
        self.assertNotIn('无帧开销', view.pwm_duty_summary.get())
        self.assertNotIn('symbol_information_rate', c)
        labels = [w.cget('text') for w in descendants(view.window) if isinstance(w, ttk.Label)]
        self.assertNotIn('起始占空比 (%)', labels)
        self.assertNotIn('启动缓冲长度（周期）', labels)
        self.assertNotIn('公共训练长度（周期）', labels)
        self.assertNotIn('下降沿拖尾初值 (µs)', labels)
        self.assertEqual(str(view._pwm_count_widget['state']), 'readonly')
        view.pwm_frame_user_count.set('3')
        c = view.pulse_config()
        self.assertAlmostEqual(c['user_information_rates'][0], 169971.671*math.log2(105)/3)
        view.pwm_duty_step.set('4')
        c = view.pulse_config()
        self.assertEqual(c['total_chips'], 25)
        self.assertEqual(c['symbol_count'], 190)
        view.pwm_duty_step.set('7')
        self.assertIn('等分', view.pwm_duty_summary.get())
        view.pwm_duty_step.set('5')

    def test_pulse_rf_worker_reference_and_diagnostic_exports(self):
        from test_pwm_pulse import config, synthesize
        view = self.view
        fs, rate = 15.36e6, 169971.671
        options, sent, _, _, t, high, _ = synthesize(config(users=3), tail_us=1.79)
        view.pwm_low_guard_chips.set('7')
        users = [dict(u) for u in options['frame_options']['users']]
        # The transmitter used a different payload each round; references are
        # concatenated per user, with no data-assisted alignment.
        for i, user in enumerate(users):
            payloads = [sent[(rnd*3+i)*144+16:(rnd*3+i+1)*144] for rnd in range(3)]
            user['reference'] = ' '.join(map(str, np.concatenate(payloads)))
        view.pwm_frame_users = users
        view.pwm_frame_user_count.set('3')
        view.pwm_frame_reference_enabled.set(True)
        view.pwm_frame_reference_reset.set(False)
        view.pwm_frame_reference_repeat.set(False)
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pulse.bin'
            raw.tofile(path)
            view.filename.set(str(path))
            recovery = view.pwm2_protocol_options()
            recovery.update(symbol_hz=rate, duration_s=.1)
            view._analyze_worker(path, dict(nfft=4096, averages=1, sample_rate_hz=fs), recovery)
            view.poll()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertEqual(view.recovery.frame_check.overall['evaluated_blocks'], 9)
            self.assertEqual(view.recovery.frame_check.overall['bad_blocks'], 0)
            self.assertIn('log₂(M)', view.summary.get())
            dialog = view.show_pwm_symbols()
            widgets = list(descendants(dialog))
            tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
            self.assertIn('delay_chips', tree['displaycolumns'])
            self.assertEqual(len(tree.item(tree.get_children()[0], 'values')), len(tree['columns']))
            for label, name in (('导出全部 CSV…', 'pulse.csv'), ('导出逐块统计…', 'pulse.json')):
                button = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == label)
                target = Path(tmp)/name
                with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(target)), \
                     patch('rx706_spectrum_gui.messagebox.showinfo'):
                    button.invoke()
                if name.endswith('.json'):
                    report = json.loads(target.read_text(encoding='utf-8'))
                    self.assertEqual(report['pulse_config']['symbol_count'], 36)
                    self.assertEqual(len(report['calibrations']), 9)
                else:
                    with target.open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.reader(stream))
                    self.assertTrue(all(len(row) == len(rows[0]) for row in rows))

    def test_nearest_estimates_are_visible_colored_and_exported_as_decisions(self):
        from test_pwm_pulse import screenshot_case
        view = self.view
        c, _, _, _, t, high, _ = screenshot_case()
        view.pwm_low_guard_chips.set('6')
        view.pwm_pulse_codebook.set('\n'.join(f'{int(d)} {int(w)}' for d, w in c['codebook']))
        view.pwm_frame_users = c['frame_options']['users']
        view.pwm_frame_reference_enabled.set(True)
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'nearest.bin'
            raw.tofile(path)
            view.filename.set(str(path))
            options = view.pwm2_protocol_options()
            options.update(symbol_hz=169971.671, duration_s=.01)
            view._analyze_worker(path, dict(nfft=4096, averages=1, sample_rate_hz=15.36e6), options)
            view.poll()
            result = view.recovery
            self.assertIsNotNone(result, view.status.get())
            estimated = result.frames['decision_estimated'] & result.frames['valid']
            self.assertGreater(int(estimated.sum()), 0)
            self.assertEqual(result.frame_check.overall['erased_bits'], 0)
            self.assertEqual(result.frame_check.overall['compared_bits'], 3*23*6)
            self.assertIn('最近码字估计', view.summary.get())
            dialog = view.show_pwm_symbols()
            widgets = list(descendants(dialog))
            tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
            self.assertIn('decision_estimated', tree['displaycolumns'])
            columns = list(tree['columns'])
            shown = [tree.item(item) for item in tree.get_children()]
            approximate = [item for item in shown if item['values'][columns.index('decision_estimated')] == '是']
            self.assertGreater(len(approximate), 0)
            for item in approximate:
                self.assertNotEqual(item['values'][columns.index('symbol')], '无效')
                self.assertNotIn('?', str(item['values'][columns.index('bits')]))
                self.assertTrue(set(item['tags']) & {'warning', 'mismatch'})
            target = Path(tmp)/'nearest.csv'
            button = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == '导出全部 CSV…')
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(target)), \
                 patch('rx706_spectrum_gui.messagebox.showinfo'):
                button.invoke()
            with target.open(encoding='utf-8-sig', newline='') as stream:
                exported = list(csv.DictReader(stream))
            self.assertTrue(all(len(row) == len(columns) for row in exported))
            approximated = [row for row in exported if row['估计判决'] == 'True']
            self.assertGreater(len(approximated), 0)
            self.assertTrue(all('?' not in row['信息比特'] for row in approximated))

    def test_default_single_user_and_all_frame_configuration_reaches_worker(self):
        view = self.view
        self.assertEqual(view.constellation_mode.get(), PWM2_MODE)
        self.assertEqual(view.frame_config()['user_count'], 1)
        view.pwm_frame_user_count.set('3')
        view.pwm_frame_startup_count.set('17')
        view.pwm_frame_startup_pattern.set('0 7')
        view.pwm_frame_training_count.set('80')
        view.pwm_frame_training_pattern.set('7 6 5 4 3 2 1 0')
        view.pwm_frame_users[1]['identity'] += ' 0 7'
        view.pwm_frame_users[1]['payload_symbols'] = 91
        view.pwm_frame_users[0]['reference'] = '0 1 2'
        view.pwm_frame_reference_enabled.set(True)
        view.filename.set('local.bin')
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            view.analyze()
            options = thread.call_args.kwargs['args'][2]
        config = options['pulse_options']['frame_options']
        self.assertEqual(config['user_count'], 3)
        self.assertEqual(config['startup_symbols'], 0)
        self.assertEqual(config['training_symbols'], 0)
        self.assertEqual(len(config['users'][1]['identity']), 18)
        self.assertEqual(config['users'][1]['payload_symbols'], 91)
        self.assertEqual(config['users'][0]['reference'], (0, 1, 2))
        self.assertNotIn('reference_options', options)
        view.busy = False

    def test_user_dialog_and_configuration_round_trip(self):
        view = self.view
        view.pwm_frame_user_count.set('2')
        view.configure_frame_users()
        dialog = next(w for w in view.window.winfo_children() if isinstance(w, tk.Toplevel))
        entries = [w for w in descendants(dialog) if isinstance(w, ttk.Entry) and not isinstance(w, ttk.Combobox)]
        self.assertEqual(len(entries), 3)
        entries[2].delete(0, 'end')
        entries[2].insert(0, '256')
        button = next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget('text') == '保存所有用户')
        with patch('rx706_spectrum_gui.messagebox.showerror') as error:
            button.invoke()
            error.assert_not_called()
        self.assertEqual(view.pwm_frame_users[0]['payload_symbols'], 256)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'protocol.json'
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(path)):
                view.save_frame_config()
            view.pwm_frame_user_count.set('1')
            view.pwm_frame_startup_count.set('3')
            with patch('rx706_spectrum_gui.filedialog.askopenfilename', return_value=str(path)):
                view.load_frame_config()
        self.assertEqual(view.frame_config()['user_count'], 2)
        self.assertEqual(view.frame_config()['startup_symbols'], 0)
        self.assertEqual(view.frame_config()['users'][0]['payload_symbols'], 256)

    def test_worker_table_user_filter_bits_and_block_statistics_exports(self):
        view = self.view
        config = configuration(3, 24)
        sent = sent_symbols(config)
        fs, rate = 15.36e6, 169971.671
        origin = .23/rate
        t = np.arange(int((origin+(len(sent)+.8)/rate)*fs))/fs
        index = np.floor((t-origin)*rate).astype(int)
        phase = (t-origin)*rate-index
        duties = (20+7*sent)/100
        high = (index >= 0) & (index < len(sent)) & (phase < duties[np.clip(index, 0, len(sent)-1)])
        x = np.where(high, -300, 300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
        raw = np.zeros((len(t), 4), dtype='<i2')
        raw[:, 0], raw[:, 1] = np.rint(x.real), np.rint(x.imag)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'burst.bin'
            raw.tofile(path)
            view.filename.set(str(path))
            view.pwm2_input_mode.set('占空比范围')
            view.pwm_duty_start.set('20')
            view.pwm_duty_step.set('7')
            view.pwm_duty_count.set('8')
            view.pwm_frame_user_count.set('3')
            view.pwm_frame_users = config['users']
            view.pwm_frame_reference_enabled.set(True)
            options = view.pwm2_protocol_options()
            options.update(symbol_hz=rate, duration_s=.01)
            view._analyze_worker(path, dict(nfft=4096, averages=1, sample_rate_hz=fs), options)
            with patch('rx706_spectrum_gui.messagebox.showwarning') as warning:
                view.poll()
                warning.assert_not_called()
            self.assertIsNotNone(view.recovery, view.status.get())
            self.assertEqual(view.recovery.frame_check.overall['bad_blocks'], 0)
            self.assertIn('误块率0/6', view.summary.get())
            dialog = view.show_pwm_symbols()
            widgets = list(descendants(dialog))
            tree = next(w for w in widgets if isinstance(w, ttk.Treeview))
            self.assertIn('user_id', tree['columns'])
            self.assertEqual(len(tree.item(tree.get_children()[0], 'values')), len(tree['columns']))
            chooser = next(w for w in widgets if isinstance(w, ttk.Combobox))
            chooser.current(2)
            chooser.event_generate('<<ComboboxSelected>>')
            user_column = list(tree['columns']).index('user_id')
            self.assertTrue(all(tree.item(row, 'values')[user_column] == '2' for row in tree.get_children()))
            for label, name in (('导出全部 CSV…', 'symbols.csv'), ('导出分组比特流…', 'bits.txt'),
                                 ('导出逐块统计…', 'blocks.json')):
                button = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == label)
                target = Path(tmp)/name
                with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(target)), \
                     patch('rx706_spectrum_gui.messagebox.showinfo'), \
                     patch('rx706_spectrum_gui.messagebox.showerror') as error:
                    button.invoke()
                    error.assert_not_called()
                if name.endswith('.csv'):
                    with target.open(encoding='utf-8-sig', newline='') as stream:
                        rows = list(csv.reader(stream))
                    self.assertTrue(all(len(row) == len(rows[0]) for row in rows))
                    self.assertEqual(len(rows)-1, 2*(16+24))
                elif name.endswith('.txt'):
                    bits = ''.join(line for line in target.read_text(encoding='utf-8').splitlines() if not line.startswith('#'))
                    self.assertEqual(len(bits.replace(' | ', '')), 2*24*3)
                else:
                    report = json.loads(target.read_text(encoding='utf-8'))
                    self.assertEqual(report['overall']['evaluated_blocks'], 6)
                    self.assertEqual(len(report['per_user']), 3)


if __name__ == '__main__':
    unittest.main()

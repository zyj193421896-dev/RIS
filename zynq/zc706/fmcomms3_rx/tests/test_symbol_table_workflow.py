"""Actual Tk symbol entry routing, QAM table/export lifecycle and PWM isolation."""
from pathlib import Path
import csv
import sys
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow, TDCM_MODE, PWM2_MODE
from rx706_workflows import AREAS, qam_training_note, auto_qam_summary
from rx706_qam import decide
from rx706_qam_preamble import points_from_labels


class SymbolTableWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk(); self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        self.patches = [patch.object(SpectrumWindow, 'maximize'),
                        patch.object(tk.Toplevel, 'deiconify'),
                        patch.object(tk.Toplevel, 'lift'),
                        patch.object(tk.Toplevel, 'focus_set')]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
        self.view = SpectrumWindow(self.root); self.view.window.withdraw()
        self.view.workflows.switch(AREAS[1])

    def tearDown(self):
        self.view.close(); self.root.update_idletasks(); self.root.destroy()
        self.assertEqual(self.errors, [])

    def install_qam_result(self):
        z = points_from_labels(np.tile(np.arange(256), 3), 256)
        ideal, labels, bits = decide(z, 256)
        r = SimpleNamespace(kind='qam', order=256, scatter=z, ideal=ideal,
            labels=labels, bits=bits, time_s=np.arange(len(z))/20000,
            evm_percent=0., input_samples=100000)
        spec = SimpleNamespace(path=Path('synthetic_table_only.bin'), channel=1,
            first_tuple=0, sample_rate_hz=15360000., rate_source='synthetic test',
            frequency_hz=np.linspace(-100000,100000,128), dbfs=np.full(128,-80.))
        self.view.recovery=r; self.view.result=spec
        return r

    def test_qam_click_with_no_result_never_reinterprets_as_pwm(self):
        v=self.view
        self.assertTrue(v.pwm_frame_enabled.get())
        self.assertEqual(v.pwm2_input_mode.get(), '时间片协议')
        before=v.workflows.snapshot()
        with patch.object(v,'analyze') as analyze, patch('rx706_spectrum_gui.messagebox.showinfo') as notice:
            v.pwm_symbols_button.invoke()
        self.assertEqual(v.workflows.snapshot(),before)
        self.assertEqual(v.constellation_mode.get(),TDCM_MODE)
        analyze.assert_not_called(); notice.assert_called_once()
        self.assertIn('尚无QAM',notice.call_args.args[0])
        self.assertIn('QAM 符号表',v.pwm_symbols_button.cget('text'))

    def test_raw_iq_in_qam_area_still_does_not_change_mode(self):
        v=self.view; v.constellation_mode.set('原始 IQ'); before=v.workflows.snapshot()
        with patch.object(v,'analyze') as analyze, patch('rx706_spectrum_gui.messagebox.showinfo'):
            v.show_pwm_symbols()
        analyze.assert_not_called(); self.assertEqual(v.workflows.snapshot(),before)

    def test_busy_qam_click_is_read_only(self):
        v=self.view; v.busy=True; before=v.workflows.snapshot()
        try:
            with patch('rx706_spectrum_gui.messagebox.showinfo') as notice:
                v.pwm_symbols_button.invoke()
            self.assertEqual(v.workflows.snapshot(),before)
            self.assertFalse(getattr(v,'_open_pwm_after_analysis',False))
            self.assertIn('QAM',notice.call_args.args[1])
        finally: v.busy=False

    def test_real_qam_table_pagination_reuse_and_complete_csv(self):
        v=self.view; r=self.install_qam_result(); before=v.workflows.snapshot()
        window=v.show_pwm_symbols(); window.withdraw(); report=v._qam_report
        self.assertEqual(report.notebook.tab(report.notebook.select(),'text'),'候选符号 / 比特')
        self.assertEqual(len(report.symbol_tree.get_children()),500)
        self.assertEqual(v.workflows.snapshot(),before)
        report.set_page(1); self.assertEqual(len(report.symbol_tree.get_children()),268)
        self.assertIs(v.show_pwm_symbols(),window)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qam.csv'
            with patch('rx706_qam_debug_gui.filedialog.asksaveasfilename',return_value=str(path)):
                report.export('csv'); report.export_thread.join(10)
            self.assertFalse(report.export_thread.is_alive())
            with path.open(encoding='utf-8-sig',newline='') as stream: rows=list(csv.reader(stream))
            self.assertEqual(len(rows),len(r.scatter)+2)
            self.assertIn('未测BER',rows[0][0])
            self.assertEqual(rows[2][-1],'00'); self.assertEqual(rows[-1][-1],'FF')
            self.assertEqual(rows[2][8],'00000000')

    def test_actual_auto_qam_recovery_opens_training_and_payload_tables(self):
        from test_qam10_frame import capture
        from rx706_qam_acquisition import recover_auto_qam
        with tempfile.TemporaryDirectory() as directory:
            spec, config, truth, _ = capture(directory, frames=8)
            result = recover_auto_qam(spec, duration_s=1, center_hint_hz=-800000,
                **{key:config[key] for key in ('order','symbol_hz','tdcm_options',
                                               'pwm_preamble_options','preamble_options')})
            v=self.view; v.result=spec; v.recovery=result
            v.draw_qam(); v.workflows.results()
            self.assertIn('payload (compensated)',v.iq_axes.get_title())
            self.assertIn('净荷最近邻EVM',v.workflows.metrics.get())
            self.assertIn('四角训练EVM',v.workflows.metrics.get())
            window=v.show_pwm_symbols(); window.withdraw(); report=v._qam_report
            self.assertEqual(len(report.training_tree.get_children()),32)
            self.assertEqual(len(result.labels),2048)
            np.testing.assert_array_equal(result.labels,truth.ravel())
            self.assertEqual(report.notebook.tab(report.notebook.select(),'text'),'候选符号 / 比特')
            self.assertIsNone(result.diagnostics['ber'])

    def test_closed_table_can_reopen_and_new_inputs_close_old_report(self):
        v=self.view; self.install_qam_result(); v.show_pwm_symbols(); first=v._qam_report
        first.close(); v.show_pwm_symbols(); second=v._qam_report
        self.assertIsNot(first,second)
        v.filename.set('different_capture.bin')
        self.assertTrue(second.closed); self.assertIsNone(v._qam_report)
        self.assertIsNone(v.recovery)

    def test_area_switch_closes_qam_table_and_restores_pwm_entry(self):
        v=self.view; self.install_qam_result(); v.show_pwm_symbols(); report=v._qam_report
        self.assertTrue(v.workflows.switch(AREAS[0]))
        self.assertTrue(report.closed)
        self.assertEqual(v.constellation_mode.get(),PWM2_MODE)
        self.assertIn('PWM 符号表',v.pwm_symbols_button.cget('text'))

    def test_pwm_time_slice_and_simple_frame_rules_are_not_silently_changed(self):
        v=self.view; v.workflows.switch(AREAS[0])
        v.pwm2_input_mode.set('时间片协议'); v.pwm_frame_enabled.set(True)
        with self.assertRaisesRegex(ValueError,'最简帧使用占空比范围'):v.pwm2_protocol_options()
        self.assertTrue(v.pwm_frame_enabled.get())
        v.pwm_frame_enabled.set(False); v.pwm2_training_enabled.set(False)
        self.assertEqual(v.pwm2_protocol_options()['kind'],'pwm2')
        v.pwm2_input_mode.set('占空比范围'); v.pwm_frame_enabled.set(True)
        v.pwm_duty_start.set('20'); v.pwm_duty_step.set('5'); v.pwm_duty_count.set('16')
        from test_pwm_frames import configuration
        config=configuration(count=1)
        v.pwm_frame_users=config['users']; v.pwm_frame_user_count.set('1')
        self.assertTrue(v.pwm2_protocol_options()['output_binary'])


class QAMMetricLabelTests(unittest.TestCase):
    def test_training_median_excludes_failed_frames_and_does_not_replace_payload(self):
        r=SimpleNamespace(order=256, evm_percent=5.2,
            frames=[dict(valid=True,pilot_evm_percent=1.),dict(valid=True,pilot_evm_percent=3.),
                    dict(valid=False,pilot_evm_percent=99.),dict(valid=True,pilot_evm_percent=float('inf'))],
            training_records=[dict(expected_hex=x) for x in ('00','08','88','80')])
        self.assertEqual(qam_training_note(r),'四角训练EVM 2.00%（有效帧中位数）')
        self.assertEqual(r.evm_percent,5.2)
        r.training_records=[]; self.assertIn('已知训练EVM',qam_training_note(r))

    def test_static_training_and_no_valid_frame_are_distinguished(self):
        r=SimpleNamespace(preamble={'evm_percent':1.2})
        self.assertEqual(qam_training_note(r),'已知训练EVM 1.20%（当前匹配）')
        r.frames=[dict(valid=False,pilot_evm_percent=99.)]
        self.assertEqual(qam_training_note(r),'已知训练EVM 无有效帧')

    def test_auto_summary_describes_actual_clock_frame_and_phase_settings(self):
        r=SimpleNamespace(order=256,evm_percent=7.,scatter=[0],bits=[[0]*8],frames=[],
            diagnostics={'auto_acquisition':dict(carrier_hz=797021.,period_ppm=45.,
                effective_recovery={'tdcm_options':{'compensation':{'frame_mode':'all','phase_mode':'off'}}})})
        text=auto_qam_summary(r)
        self.assertIn('净荷最近邻EVM 7.00%',text)
        self.assertIn('已逐帧提取净荷',text); self.assertIn('非连续时钟漂移跟踪',text)
        self.assertIn('未启用连续公共相位跟踪',text); self.assertIn('BER未测',text)
        self.assertNotIn('后续重复前导不自动剔除',text)

if __name__=='__main__':unittest.main()

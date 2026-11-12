"""Real Tk checks for resizing symbol columns and the three-panel analysis UI."""
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_table_layout import configure_symbol_table
from rx706_spectrum_gui import SpectrumWindow, PLOT_NAMES, RECOVERY_MODES
from rx706_gui import Rx706App, capture_length


class LayoutTests(unittest.TestCase):
    def test_pwm_binary_symbol_table_and_complete_csv_remain_aligned(self):
        import csv
        from test_pwm2 import PWM2Tests
        from rx706_spectrum_gui import PWM2_MODE
        fixture = PWM2Tests()
        fixture.setUp()
        view = SpectrumWindow(self.root)
        try:
            fixture.capture(total=25, prefix=6, guard=7)
            view.window.withdraw()
            self.assertEqual(view.constellation_mode.get(), PWM2_MODE)
            self.assertEqual(view.pwm_symbol_khz.get(), '169.971671')
            self.assertEqual(view.pwm2_preset.get(), '自定义')
            self.assertFalse(view.show_advanced.get())
            view.constellation_mode.set(PWM2_MODE)
            view._analyze_worker(fixture.path, dict(nfft=4096, averages=1, sample_rate_hz=fixture.fs),
                dict(kind='pwm2', total_chips=25, prefix_chips=6, guard_chips=7,
                     symbol_hz=fixture.rate, symbol_widths=tuple(range(6, 19)), output_binary=True))
            view.poll()
            self.assertIsNotNone(view.recovery)
            dialog = view.show_pwm_symbols()
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            tree = next(w for w in descendants(dialog) if isinstance(w, ttk.Treeview))
            self.assertEqual(tuple(tree['displaycolumns']), ('index', 'start_s', 'duty_percent',
                'decision_duty_percent', 'decoded_duty_percent', 'symbol', 'duty_thermometer', 'bits', 'reason'))
            self.assertEqual(len(tree.item(tree.get_children()[0], 'values')), len(tree['columns']))
            export_button = next(w for w in descendants(dialog) if isinstance(w, ttk.Button)
                                 and w.cget('text') == '导出全部 CSV…')
            target = fixture.path.with_suffix('.csv')
            with patch('rx706_spectrum_gui.filedialog.asksaveasfilename', return_value=str(target)), \
                 patch('rx706_spectrum_gui.messagebox.showinfo'):
                export_button.invoke()
            with target.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), len(view.recovery.frames)+1)
            self.assertTrue(all(len(row) == len(rows[0]) for row in rows))
        finally:
            view.close()
            fixture.tearDown()

    def test_pwm_simple_four_fields_and_configurable_advanced_protocol(self):
        from rx706_spectrum_gui import PWM2_MODE, PWM150_PRESET
        view = SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.constellation_mode.set(PWM2_MODE)
            view.refresh_parameter_visibility()
            visible = [view._parameter_titles[box] for box, _ in view._parameter_groups if box.winfo_manager()]
            self.assertEqual(visible, [])
            self.assertTrue(view.workflows.panel.winfo_manager())
            view.show_advanced.set(True)
            view.refresh_parameter_visibility()
            self.assertTrue(view.pwm_frame_enabled.get())
            pulse = view.pwm2_protocol_options()['pulse_options']
            config = pulse['frame_options']
            self.assertEqual(config['user_count'], 1)
            self.assertEqual(config['training_symbols'], 0)
            self.assertEqual(pulse['symbol_count'], 105)
            view.pwm2_input_mode.set('占空比范围')
            view.pwm_duty_start.set('20')
            view.pwm_duty_step.set('7')
            view.pwm_duty_count.set('8')
            view.pwm_frame_enabled.set(False)
            pwm_box = next(box for box, _ in view._parameter_groups
                           if view._parameter_titles[box] == 'PWM 脉宽与信息解调')
            entries = [w for w in pwm_box.winfo_children()
                       if isinstance(w, ttk.Entry) and w.winfo_manager()]
            self.assertEqual(len(entries), 8)  # expert fields now explicitly enabled
            self.assertTrue(view.pwm2_training_enabled.get())
            self.assertTrue(view.pwm2_clock_width.get())
            view.pwm_duty_start.set('20')
            view.pwm_duty_step.set('7')
            view.pwm_duty_count.set('8')
            self.assertIn('结束占空比 69%', view.pwm_duty_summary.get())
            options = view.pwm2_protocol_options()
            self.assertEqual(options['symbol_widths'], (20, 27, 34, 41, 48, 55, 62, 69))
            self.assertTrue(options['output_binary'])
            self.assertEqual(options['training_options']['preamble_widths'], (20, 20, 20))
            self.assertEqual(options['training_options']['payload_symbols'], 8)
            self.assertTrue(options['training_options']['auto_scale'])
            self.assertIn('每帧净荷 8', view.pwm_duty_summary.get())
            self.assertTrue(options['training_options']['clock_width'])
            view.pwm2_preset.set(PWM150_PRESET)
            view.apply_pwm2_preset()
            options = view.pwm2_protocol_options()
            self.assertEqual(options['symbol_widths'], tuple(range(6, 19)))
            self.assertEqual(options['training_options']['preamble_widths'], (6, 6, 6))
            self.assertEqual(options['training_options']['payload_symbols'], 13)
            view.show_advanced.set(True)
            view.pwm2_input_mode.set('时间片协议')
            view.pwm2_mapping.set('自定义高片数表')
            view.pwm2_width_table.set('18 17 16 15 14 13 12 11 10 9 8 7 6')
            view.pwm2_preamble_start.set('20')
            view.pwm2_preamble_step.set('7')
            view.pwm2_preamble_count.set('3')
            options = view.pwm2_protocol_options()
            self.assertEqual(options['training_options']['preamble_widths'], (5, 6.75, 8.5))
            self.assertEqual(options['symbol_widths'][0], 18)
            view.filename.set('local.bin')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                delivered = thread.call_args.kwargs['args'][2]
                self.assertEqual(delivered['training_options']['preamble_widths'], (5, 6.75, 8.5))
                self.assertEqual(delivered['symbol_widths'], tuple(range(18, 5, -1)))
                self.assertTrue(delivered['training_options']['clock_width'])
                self.assertTrue(delivered['training_options']['auto_scale'])
                self.assertEqual(delivered['training_options']['payload_symbols'], 13)
            view.busy = False
            view.pwm2_training_enabled.set(False)
            self.assertNotIn('training_options', view.pwm2_protocol_options())
            for preset in ('20片：8+10+2', '50片：20+25+5', '50片：8+40+2', PWM150_PRESET):
                view.pwm2_training_enabled.set(False)
                view.pwm2_clock_width.set(False)
                view.pwm2_preset.set(preset)
                view.apply_pwm2_preset()
                self.assertTrue(view.pwm2_training_enabled.get())
                self.assertTrue(view.pwm2_clock_width.get())
                self.assertIn('training_options', view.pwm2_protocol_options())
        finally:
            view.close()

    def test_170k_preset_delivers_calibration_and_thermometer_settings(self):
        from rx706_spectrum_gui import PWM170_PRESET
        view = SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.pwm2_preset.set(PWM170_PRESET)
            view.apply_pwm2_preset()
            self.assertEqual(view.pwm_symbol_khz.get(), '169.971671')
            view.filename.set('local.bin')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                delivered = thread.call_args.kwargs['args'][2]
            self.assertEqual(delivered['symbol_widths'], (20,27,34,41,48,55,62,69))
            training = delivered['training_options']
            self.assertEqual(training['preamble_widths'], (20,20,20))
            self.assertEqual(training['payload_symbols'], 8)
            self.assertTrue(training['auto_scale'])
            self.assertTrue(training['clock_width'])
            self.assertNotIn('reference_options', delivered)
            view.busy = False
            view.pwm2_preamble_start.set('30')
            view.pwm_duty_start.set('30')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                changed = thread.call_args.kwargs['args'][2]
            self.assertEqual(changed['symbol_widths'], (30,37,44,51,58,65,72,79))
            self.assertEqual(changed['training_options']['preamble_widths'], (30,30,30))
            self.assertEqual(changed['training_options']['payload_symbols'], 8)
            self.assertTrue(changed['training_options']['auto_scale'])
        finally:
            view.close()

    def test_payload_edits_preserve_independent_preamble_and_explicit_frame_length(self):
        from rx706_spectrum_gui import PWM150_PRESET, PWM170_PRESET
        view = SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.pwm2_preset.set(PWM150_PRESET)
            view.apply_pwm2_preset()
            view.pwm_duty_start.set('20')
            view.pwm_duty_step.set('7')
            view.pwm_duty_count.set('8')
            # A valid independent 24% preamble must not be silently overwritten.
            self.assertEqual(view.pwm2_preamble_start.get(), '24')
            self.assertIn('前导：24%', view.pwm_duty_summary.get())
            view.pwm2_preset.set(PWM170_PRESET)
            view.apply_pwm2_preset()
            self.assertEqual(view.pwm2_protocol_options()['training_options']['preamble_widths'], (20,20,20))
            view.pwm2_payload_count.set('17')
            view.pwm_duty_count.set('7')
            self.assertEqual(view.pwm2_protocol_options()['training_options']['payload_symbols'], 17)
            view.pwm2_auto_scale.set(False)
            self.assertIn('同宽前导只校准偏移', view.pwm_duty_summary.get())
        finally:
            view.close()

    def setUp(self):
        try:
            self.root=tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk display/runtime required: {exc}')
        self.root.withdraw()
        self.errors=[]
        self.root.report_callback_exception=lambda *args:self.errors.append(args)

    def pump_events(self):
        # macOS Tk can continuously enqueue native redraws inside update().
        # Use the application's actual event loop with a bounded timer instead.
        self.root.after(100, self.root.quit)
        self.root.mainloop()

    def tearDown(self):
        try:self.root.destroy()
        except tk.TclError:pass
        self.assertFalse(self.errors,self.errors)

    def test_table_resize_scroll_and_separator_clamp_at_multiple_scales(self):
        for scale in (1.,1.5,2.):
            with self.subTest(scale=scale):
                self.root.tk.call('tk','scaling',scale)
                window=tk.Toplevel(self.root)
                window.geometry('900x300+20+20')
                cols=('index','start_s','pattern','candidate_symbol','candidate_source')
                headers=('符号序号','起始时间(μs)','原始时间片码型','候选符号','候选依据')
                tree=ttk.Treeview(window,columns=cols,show='headings')
                tree.pack(fill='both',expand=True)
                reset=configure_symbol_table(tree,cols,headers,50,40)
                item=tree.insert('','end',values=(32,'318.6603','1'*20+'0'*30,6,'按码型判决'))
                self.pump_events()
                widths={key:int(tree.column(key,'width')) for key in cols}
                window.geometry('460x300+20+20')
                self.pump_events()
                self.assertEqual(widths,{key:int(tree.column(key,'width')) for key in cols})
                self.assertLess(tree.xview()[1],1.)
                for key in cols:
                    self.assertFalse(tree.column(key,'stretch'))
                    tree.column(key,width=4)
                tree.event_generate('<ButtonRelease-1>',x=20,y=10)
                self.pump_events()
                for key in cols:
                    self.assertGreaterEqual(tree.column(key,'width'),tree.column(key,'minwidth'))
                reset()
                tree.xview_moveto(.25)
                self.pump_events()
                # Header hit-testing and row cells retain a common x origin.
                for n,key in enumerate(cols,1):
                    bounds=tree.bbox(item,key)
                    if bounds:
                        x,y,w,h=bounds
                        if 2 <= x+w/2 <= tree.winfo_width()-2:
                            self.assertEqual(tree.identify_column(int(x+w/2)),f'#{n}')
                window.destroy()

    def test_analysis_has_only_three_panels_and_no_retired_modes(self):
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            self.pump_events()
            self.assertEqual(len(view.figure.axes),3)
            self.assertFalse(hasattr(view,'power_axes'))
            self.assertFalse(hasattr(view,'phase_axes'))
            self.assertNotIn('功率',PLOT_NAMES)
            self.assertNotIn('PWM 脉宽',PLOT_NAMES)
            self.assertNotIn('单音载波恢复',RECOVERY_MODES)
            self.assertEqual(view.recovery_ms.get(),'1000')
            for name in ('星座','IQ 波形','频谱','总览'):
                view.set_plot_view(name)
                view.canvas.draw()
                self.assertEqual(sum(ax.get_visible() for ax in view.figure.axes),3 if name=='总览' else 1)
        finally:view.close()

    def test_capture_defaults_to_exactly_one_second(self):
        app=Rx706App(self.root)
        try:
            self.assertEqual(app.length_unit.get(),'秒')
            self.assertEqual(app.length_value.get(),'1')
            self.assertEqual(capture_length(app.length_value.get(),app.length_unit.get()),122880000)
        finally:
            app.controller.close()
            app.controller.thread.join(timeout=2)
            app.destroy()

    def test_retained_tdcm_configurable_preamble_is_delivered_to_worker(self):
        from rx706_spectrum_gui import TDCM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.filename.set('local.bin')
            view.constellation_mode.set(TDCM_MODE)
            self.assertFalse(view.qam_preamble_enabled.get())
            view.qam_preamble_enabled.set(True)
            view.qam_preamble_symbols.set('0x00 0b00010001 255')
            view.qam_preamble_frequency_khz.set('0.7')
            view.qam_preamble_search_ms.set('750')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                options=thread.call_args.kwargs['args'][2]
                self.assertEqual(options['kind'],'qam')
                pilot=options['preamble_options']
                self.assertEqual(pilot['symbols'],'0x00 0b00010001 255')
                self.assertEqual(pilot['max_frequency_hz'],700)
                self.assertEqual(pilot['search_s'],.75)
                self.assertEqual(pilot['representation'],'labels')
                thread.return_value.start.assert_called_once()
        finally:view.close()

    def test_retired_modes_are_removed_and_raw_pwm_tdcm_remain(self):
        from rx706_spectrum_gui import QAM_PWM_MODE, QAM_MODE, RECOVERY_MODE, PWM_MODE, PWM2_MODE, TDCM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            self.assertEqual(RECOVERY_MODES, (PWM_MODE, PWM2_MODE, TDCM_MODE))
            for mode in (QAM_PWM_MODE, QAM_MODE, RECOVERY_MODE):
                self.assertNotIn(mode, RECOVERY_MODES)
            view.refresh_parameter_visibility()
            self.assertEqual(view.constellation_mode.get(), PWM2_MODE)
        finally:view.close()

    def test_tdcm_parameters_are_independent_of_ordinary_pwm_and_qam(self):
        from rx706_spectrum_gui import TDCM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.filename.set('tdcm.bin')
            view.constellation_mode.set(TDCM_MODE)
            self.assertEqual(view.pwm_symbol_khz.get(),'169.971671')
            self.assertEqual(view.qam_symbol_khz.get(),'100')
            self.assertEqual(view.tdcm_symbol_khz.get(),'19.53125')
            self.assertEqual(view.tdcm_pwm_khz.get(),'19.53125')
            self.assertEqual(view.tdcm_harmonic.get(),'1')
            self.assertEqual(view.tdcm_carrier_mhz.get(),'-0.2')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                options=thread.call_args.kwargs['args'][2]
                self.assertEqual(options['kind'],'qam')
                self.assertEqual(options['symbol_hz'],19531.25)
                self.assertEqual(options['center_hint_hz'],-200000)
                self.assertEqual(options['pwm_preamble_options']['symbol_hz'],19531.25)
                self.assertEqual(options['tdcm_options']['harmonic'],1)
                self.assertEqual(options['tdcm_options']['normalization'],'pwm')
                self.assertEqual(options['tdcm_options']['base_hz'],19531.25)
                self.assertNotIn('preamble_options',options)
                thread.return_value.start.assert_called_once()
        finally:view.close()


if __name__=='__main__':unittest.main()

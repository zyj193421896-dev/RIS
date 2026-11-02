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
    def setUp(self):
        try:
            self.root=tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f'Tk display/runtime required: {exc}')
        self.root.withdraw()
        self.errors=[]
        self.root.report_callback_exception=lambda *args:self.errors.append(args)

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
                self.root.update()
                widths={key:int(tree.column(key,'width')) for key in cols}
                window.geometry('460x300+20+20')
                self.root.update()
                self.assertEqual(widths,{key:int(tree.column(key,'width')) for key in cols})
                self.assertLess(tree.xview()[1],1.)
                for key in cols:
                    self.assertFalse(tree.column(key,'stretch'))
                    tree.column(key,width=4)
                tree.event_generate('<ButtonRelease-1>',x=20,y=10)
                self.root.update()
                for key in cols:
                    self.assertGreaterEqual(tree.column(key,'width'),tree.column(key,'minwidth'))
                reset()
                tree.xview_moveto(.25)
                self.root.update()
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
            self.root.update()
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

    def test_qam_configurable_preamble_is_delivered_to_worker(self):
        from rx706_spectrum_gui import QAM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.filename.set('local.bin')
            view.constellation_mode.set(QAM_MODE)
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

    def test_mixed_pwm_qam_mode_defaults_and_worker_options(self):
        from rx706_spectrum_gui import QAM_PWM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.filename.set('mixed.bin')
            view.constellation_mode.set(QAM_PWM_MODE)
            self.assertIn(QAM_PWM_MODE, RECOVERY_MODES)
            self.assertEqual([view.pwm2_total_chips.get(),view.pwm2_prefix_chips.get(),view.pwm2_guard_chips.get()],
                             ['25','5','8'])
            self.assertEqual(view.qam_pwm_symbols.get(),'0 0 0')
            view.qam_pwm_gap_us.set('12.5')
            view.qam_pwm_width_offset.set('-0.25')
            view.qam_pwm_qam_window.set('4')
            with patch('rx706_spectrum_gui.threading.Thread') as thread:
                view.analyze()
                options=thread.call_args.kwargs['args'][2]
                self.assertEqual(options['kind'],'qam')
                self.assertNotIn('preamble_options',options)
                pwm=options['pwm_preamble_options']
                self.assertEqual((pwm['total_chips'],pwm['prefix_chips'],pwm['guard_chips']),(25,5,8))
                self.assertEqual(pwm['symbols'],'0 0 0')
                self.assertAlmostEqual(pwm['gap_s'],.0000125)
                self.assertEqual(pwm['width_offset_chips'],-.25)
                self.assertEqual(pwm['qam_search_symbols'],4)
                thread.return_value.start.assert_called_once()
        finally:view.close()

    def test_tdcm_parameters_are_independent_of_ordinary_pwm_and_qam(self):
        from rx706_spectrum_gui import TDCM_MODE
        view=SpectrumWindow(self.root)
        try:
            view.window.withdraw()
            view.filename.set('tdcm.bin')
            view.constellation_mode.set(TDCM_MODE)
            self.assertEqual(view.pwm_symbol_khz.get(),'100')
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

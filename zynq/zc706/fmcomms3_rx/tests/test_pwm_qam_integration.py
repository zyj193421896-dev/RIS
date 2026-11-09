"""Combined PWM/QAM routing and isolated real-Tk regression coverage."""
from pathlib import Path
import sys
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_spectrum_gui import (SpectrumWindow, QAM_MODES, TDCM_MODES,
    RECOVERY_MODES, PWM2_MODE, QAM_MODE, QAM_PWM_MODE, TDCM_MODE)


class EngineBoundaryTests(unittest.TestCase):
    def test_known_header_does_not_relax_standalone_pwm(self):
        from rx706_qam_pwm import validate_pwm_header
        from rx706_pwm import pwm2_layout
        self.assertEqual(validate_pwm_header(dict(total_chips=64,prefix_chips=16,
            guard_chips=48,symbol_hz=120000,symbols='0 0 0'),15360000)['symbols'],[0,0,0])
        with self.assertRaises(ValueError):pwm2_layout(64,16,48)
        self.assertTrue(set(QAM_MODES).issubset(RECOVERY_MODES))


class CombinedGUI(unittest.TestCase):
    def setUp(self):
        try:self.root=tk.Tk()
        except tk.TclError as exc:self.skipTest(str(exc))
        self.root.withdraw();self.errors=[]
        self.root.report_callback_exception=lambda *args:self.errors.append(args)
        original=tk.Toplevel
        class Hidden(original):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);self.withdraw()
        self.hidden=patch('tkinter.Toplevel',Hidden);self.hidden.start()
        self.maximize=patch.object(SpectrumWindow,'maximize',lambda self:None);self.maximize.start()
        self.view=SpectrumWindow(self.root)
    def tearDown(self):
        if not hasattr(self,'view'):return
        self.view.busy=False;self.view.close();self.root.destroy()
        self.maximize.stop();self.hidden.stop()
        self.assertFalse(self.errors,self.errors)
    def geometry(self):
        v=self.view
        return tuple(x.get() for x in (v.pwm2_total_chips,v.pwm2_prefix_chips,v.pwm2_guard_chips))
    def test_all_qam_groups_and_header_geometry_visible(self):
        v=self.view
        self.assertEqual(v.pwm2_input_mode.get(),'脉冲宽度与时延')
        for mode in QAM_MODES:
            v.constellation_mode.set(mode);v.refresh_parameter_visibility()
            visible={v._parameter_titles[b] for b,_ in v._parameter_groups if b.winfo_manager()}
            self.assertIn('QAM 解调设置',visible)
            if mode != QAM_MODE:
                self.assertIn('PWM version2：协议与映射',visible)
                for variable in (v.pwm2_total_chips,v.pwm2_prefix_chips,v.pwm2_guard_chips):
                    fields=[w for w in v._time_protocol_fields if 'textvariable' in w.keys() and str(w.cget('textvariable'))==str(variable)]
                    self.assertEqual(len(fields),1)
                    self.assertEqual(fields[0].winfo_manager(),'grid')
            if mode in TDCM_MODES:self.assertIn('TDCM：按谐波复系数解调',visible)
    def test_profiles_preserve_pwm_and_latest_recipe_geometry(self):
        v=self.view
        v.pwm2_total_chips.set('50');v.pwm2_prefix_chips.set('8');v.pwm2_guard_chips.set('2')
        v.rate.set('15.36');v.apply_qam256_120k_frame_profile()
        self.assertEqual(self.geometry(),('64','16','48'))
        self.assertIn('每片2.000',v.pwm2_layout_summary.get())
        v.constellation_mode.set(PWM2_MODE)
        self.assertEqual(self.geometry(),('50','8','2'))
        v.apply_qam_recipe_parameters(dict(method='tdcm_parameter_config',
            pwm_preamble_options=dict(total_chips=25,prefix_chips=5,guard_chips=8)))
        v.constellation_mode.set(TDCM_MODE)
        self.assertEqual(self.geometry(),('25','5','8'))
        v.constellation_mode.set(PWM2_MODE)
        self.assertEqual(self.geometry(),('50','8','2'))
        v.constellation_mode.set(TDCM_MODE)
        self.assertEqual(self.geometry(),('25','5','8'))
    def request(self):
        v=self.view;v.busy=False;v.filename.set('/tmp/routing-only.bin')
        with patch('rx706_spectrum_gui.threading.Thread') as thread:
            v.analyze()
            self.assertTrue(thread.called,v.status.get())
            result=thread.call_args.kwargs['args'][2]
        v.busy=False
        return result
    def test_pwm_rate_edit_does_not_replace_qam_header_summary(self):
        v=self.view
        v.constellation_mode.set(QAM_PWM_MODE)
        v.pwm_symbol_khz.set('120')
        self.assertIn('每片',v.pwm2_layout_summary.get())
        self.assertNotIn('单用户净荷信息率',v.pwm2_layout_summary.get())
        v.apply_qam256_120k_frame_profile()
        v.pwm_symbol_khz.set('150')
        self.assertIn('每片2.000',v.pwm2_layout_summary.get())

    def test_engine_options_remain_separate(self):
        v=self.view
        pwm=self.request()
        self.assertEqual(pwm['kind'],'pwm2');self.assertIn('pulse_options',pwm)
        v.rate.set('15.36');v.apply_qam256_120k_frame_profile()
        qam=self.request()
        self.assertEqual(qam['kind'],'qam');self.assertNotIn('pulse_options',qam)
        self.assertEqual(qam['pwm_preamble_options']['total_chips'],64)
        self.assertEqual(qam['tdcm_options']['payload_data_symbols'],16)
        v.constellation_mode.set(PWM2_MODE)
        self.assertEqual(self.request()['pulse_options'],pwm['pulse_options'])
    def test_qam_auto_analysis_and_pwm_manual_analysis(self):
        v=self.view
        with patch.object(v,'analyze') as analyze:
            v.constellation_mode.set(QAM_MODE);v.on_mode_changed();analyze.assert_called_once()
            analyze.reset_mock()
            v.constellation_mode.set(PWM2_MODE);v.on_mode_changed();analyze.assert_not_called()

if __name__=='__main__':unittest.main()

"""10 kHz defaults, explicit legacy rates, and real Tk workflow transitions."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam10_profile import build_qam10_config
from rx706_qam_config import rescale_parameter_rate, validate_parameter_config

class RateTests(unittest.TestCase):
    def test_default_and_legacy_roundtrip(self):
        for rate in (10000, 20000, 5000, 12500):
            c = build_qam10_config(symbol_hz=rate)
            self.assertEqual(c, validate_parameter_config(c))
            for value in (c['symbol_hz'], c['tdcm_options']['base_hz'], c['pwm_preamble_options']['symbol_hz']):
                self.assertEqual(value, rate)
        self.assertEqual(build_qam10_config()['symbol_hz'], 10000)
    def test_scale_preserves_original_and_ratio(self):
        c=build_qam10_config(symbol_hz=20000); original=copy.deepcopy(c)
        d=rescale_parameter_rate(c,10000)
        self.assertEqual(c,original)
        self.assertEqual(d['pwm_preamble_options']['symbol_hz'],10000)
        self.assertEqual(d['tdcm_options']['base_hz'],10000)
        self.assertEqual(d['pwm_preamble_options']['max_error_percent'],25)
        self.assertEqual(d['preamble_options'],c['preamble_options'])
        c['transmitter']['qam_cycles_per_symbol']=2;c['tdcm_options']['base_hz']=40000
        c['pwm_preamble_options']['symbol_hz']=5000
        d=rescale_parameter_rate(c,10000)
        self.assertEqual(d['tdcm_options']['base_hz'],20000)
        self.assertEqual(d['pwm_preamble_options']['symbol_hz'],2500)
    def test_independent_payload_at_multiple_rates(self):
        import tempfile
        import numpy as np
        from test_qam10_frame import capture
        from rx706_qam_acquisition import recover_auto_qam
        for rate in (10000, 20000, 5000):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                spectrum,c,truth,lead=capture(directory,frames=12,rate=rate)
                r=recover_auto_qam(spectrum,duration_s=1,**{k:c[k] for k in
                    ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')})
                self.assertEqual(len(r.labels),12*256)
                self.assertIsNone(r.diagnostics['ber'])
                self.assertFalse(r.diagnostics['auto_acquisition']['payload_used'])
                for frame in np.unique(r.frame_ids):
                    k=int(round((r.frames[int(frame)]['header_time_s']*spectrum.sample_rate_hz-lead)/(270*spectrum.sample_rate_hz/rate)))
                    np.testing.assert_array_equal(r.labels[r.frame_ids==frame],truth[k])

    def test_invalid_rate(self):
        for rate in (0,-1,float('inf'),float('nan'),True,'10'):
            with self.assertRaises(ValueError):build_qam10_config(symbol_hz=rate)
            with self.assertRaises(ValueError):rescale_parameter_rate(build_qam10_config(),rate)

class GuiRateTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from rx706_spectrum_gui import SpectrumWindow
        self.root=tk.Tk();self.root.withdraw()
        with patch.object(SpectrumWindow,'maximize'):self.v=SpectrumWindow(self.root)
        self.v.window.withdraw();self.w=self.v.workflows
    def tearDown(self):
        self.v.close();self.root.destroy()
    def test_default_edit_restore_cancel_and_legacy(self):
        from rx706_workflows import AREAS
        self.w.switch(AREAS[1])
        self.assertEqual(self.w.portable['symbol_hz'],10000)
        self.assertIsNone(self.w.confirmed)
        self.assertEqual(self.w.qam_rate_button.winfo_manager(),'pack')
        for field in ('tdcm_symbol_khz','tdcm_base_khz','tdcm_pwm_khz'):
            self.assertEqual(float(getattr(self.v,field).get()),10)
        before=self.w.snapshot()
        with patch('rx706_workflows.simpledialog.askfloat',return_value=None):
            self.assertFalse(self.w.qam_rate_button.invoke())
        self.assertEqual(before,self.w.snapshot())
        self.assertTrue(self.w.set_qam_rate(20));self.assertIsNone(self.w.confirmed)
        self.assertEqual(self.w.validate()['symbol_hz'],20000)
        self.w.switch(AREAS[0]);self.w.switch(AREAS[1])
        self.assertEqual(self.w.portable['symbol_hz'],20000)
        for field in ('tdcm_symbol_khz','tdcm_base_khz','tdcm_pwm_khz'):
            self.assertEqual(float(getattr(self.v,field).get()),20)
        before=self.w.snapshot();self.v.busy=True
        self.assertFalse(self.w.set_qam_rate(5));self.v.busy=False
        self.assertEqual(before,self.w.snapshot())
    def test_real_old_json_unchanged(self):
        from rx706_workflows import AREAS
        p=Path(__file__).resolve().parents[1]/'host/frame_profiles/acceptance/rx_256qam_256slice_20khz_10_4_256_config.json'
        old=p.read_bytes();self.w.switch(AREAS[1]);self.assertTrue(self.w.load(p))
        self.assertEqual(self.w.portable['symbol_hz'],20000)
        self.assertEqual(float(self.v.tdcm_pwm_khz.get()),20)
        self.assertTrue(self.w.set_qam_rate(10));self.assertEqual(p.read_bytes(),old)

if __name__=='__main__':unittest.main()

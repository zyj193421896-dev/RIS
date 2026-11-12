"""Withdrawn Tk workflow checks; no hardware or foreground windows."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_spectrum_gui import SpectrumWindow
from rx706_workflows import AREAS, validate_qam_profile

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk(); self.root.withdraw()
        with patch.object(SpectrumWindow, 'maximize'):
            self.view = SpectrumWindow(self.root)
        self.view.window.withdraw()
        self.flow = self.view.workflows
        self.temp = tempfile.TemporaryDirectory()
        self.bin = Path(self.temp.name)/'capture.bin'; self.bin.write_bytes(b'\0'*8192)
    def tearDown(self):
        self.view.close(); self.root.destroy(); self.temp.cleanup()
    def load_pwm(self):
        return self.view.load_frame_config(filename=ROOT/'host/gui_data/current_pwm/frame.json')
    def test_three_areas_compact_default(self):
        self.assertEqual(tuple(b.cget('text') for b in self.flow.area_buttons), AREAS)
        self.assertEqual([b for b,_ in self.view._parameter_groups if b.winfo_manager()], [])
        self.assertEqual(self.view.analyze_button.cget('text'), '运行接收')
    def test_new_protocol_clears_old_capture_and_reference(self):
        self.load_pwm(); self.view.filename.set(str(self.bin))
        self.view._fixed_source_sha256 = 'old-binding'
        self.view.result = object()
        self.view._autotune_report = {'old': True}
        self.assertTrue(self.flow.new_protocol_button.invoke())
        self.assertEqual(self.view.filename.get(), '')
        self.assertIsNone(self.view.result); self.assertIsNone(self.view.recovery)
        self.assertIsNone(self.view._fixed_source_sha256)
        self.assertIsNone(self.view._autotune_report)
        self.assertIsNone(self.flow.confirmed)
        self.assertFalse(self.view.pwm_frame_reference_enabled.get())
        pulse = self.view.pulse_config()
        self.assertEqual(len(pulse['frame_options']['users'][0]['identity']), 10)
        self.assertEqual(pulse['frame_options']['users'][0]['payload_symbols'], 66)
        self.assertFalse(pulse['cycle_calibration'])
        self.assertIn('新采样', self.view.status.get())
        with patch.object(self.view, 'start_pwm_autotune') as start:
            self.assertFalse(self.flow.run()); start.assert_not_called()

    def test_new_protocol_busy_guard_and_pwm_only(self):
        self.assertEqual(self.flow.new_protocol_button.winfo_manager(), 'pack')
        self.view.filename.set(str(self.bin)); before = self.flow.snapshot()
        self.view.busy = True
        self.assertFalse(self.flow.load_new_pwm_protocol())
        self.assertEqual(self.flow.snapshot(), before)
        self.view.busy = False
        for area in AREAS[1:]:
            self.flow.switch(area)
            self.assertEqual(self.flow.new_protocol_button.winfo_manager(), '')
            self.assertFalse(self.flow.load_new_pwm_protocol())
        self.flow.switch(AREAS[0])
        self.assertEqual(self.flow.new_protocol_button.winfo_manager(), 'pack')

    def test_new_protocol_invalid_or_missing_preserves_state(self):
        self.load_pwm(); self.view.filename.set(str(self.bin))
        sentinel = object(); self.view.result = sentinel
        before = self.flow.snapshot()
        bad = Path(self.temp.name)/'invalid_new.json'
        bad.write_text('{"reference_enabled":true}')
        for path in (bad, bad.with_name('missing.json')):
            with patch('rx706_workflows.FIXED_PREAMBLE10_PROFILE', path), patch('rx706_workflows.messagebox.showerror') as error:
                self.assertFalse(self.flow.load_new_pwm_protocol())
                error.assert_called_once()
            self.assertEqual(self.flow.snapshot(), before)
            self.assertIs(self.view.result, sentinel)

    def test_missing_config_does_not_start(self):
        self.view.filename.set(str(self.bin))
        with patch.object(self.view,'start_pwm_autotune') as start:
            self.assertFalse(self.flow.run()); start.assert_not_called()
        self.assertIn('配置', self.view.status.get())
    def test_load_pwm_runs_autotune(self):
        self.view.filename.set(str(self.bin)); self.assertTrue(self.load_pwm())
        with patch.object(self.view,'start_pwm_autotune') as start:
            self.assertTrue(self.flow.run()); start.assert_called_once()
    def test_switch_preserves_each_bin_and_configuration(self):
        self.view.filename.set(str(self.bin)); self.load_pwm(); before=self.view._pwm_profile_snapshot()
        self.assertTrue(self.flow.switch(AREAS[1])); self.view.tdcm_harmonic.set('-2')
        self.view.filename.set('second.bin'); self.flow.switch(AREAS[0])
        self.assertEqual(self.view.filename.get(),str(self.bin)); self.assertEqual(before,self.view._pwm_profile_snapshot())
        self.flow.switch(AREAS[1]); self.assertEqual(self.view.filename.get(),'second.bin'); self.assertEqual(self.view.tdcm_harmonic.get(),'-2')
    def test_busy_blocks_switch_run_and_template(self):
        self.view.busy=True
        with patch.object(self.view,'start_pwm_autotune') as start, patch.object(self.view,'configure_frame_users') as edit:
            self.assertFalse(self.flow.switch(AREAS[2])); self.assertFalse(self.flow.run()); self.flow.template()
            start.assert_not_called(); edit.assert_not_called()
        self.assertEqual(self.flow.area,AREAS[0]); self.view.busy=False
    def test_invalid_qam_profile_retires_results(self):
        self.flow.switch(AREAS[1]); bad=Path(self.temp.name)/'bad.json'; bad.write_text('{"schema":"unknown"}')
        self.view.result=object(); self.view.recovery=object()
        with patch('rx706_workflows.messagebox.showerror'):
            self.assertFalse(self.flow.load(bad))
        self.assertIsNone(self.view.result); self.assertIsNone(self.view.recovery)
    def test_qam_roundtrip_and_extra_field_rejection(self):
        self.flow.switch(AREAS[1]); profile=self.flow.qam_profile()
        self.assertEqual(validate_qam_profile(profile),profile['settings'])
        filename=Path(self.temp.name)/'qam.json'; filename.write_text(json.dumps(profile))
        self.assertTrue(self.flow.load(filename)); self.assertEqual(self.flow.confirmed,self.flow.signature())
        profile['settings']['clock_tracking']=True
        with self.assertRaises(ValueError): validate_qam_profile(profile)
    def test_sample_rate_edit_retires_results_and_confirmation(self):
        self.load_pwm(); self.view.result=object(); self.view.recovery=object()
        self.view.rate.set('20')
        self.assertIsNone(self.view.result); self.assertIsNone(self.view.recovery)
        self.assertNotEqual(self.flow.confirmed,self.flow.signature()); self.assertEqual(self.flow.label.get(),'发送配置待确认')
    def test_three_user_template_is_not_actual_configuration(self):
        self.flow.switch(AREAS[2]); self.view.filename.set(str(self.bin))
        with patch.object(self.view,'configure_frame_users'): self.flow.template()
        self.assertEqual(self.view.pwm_frame_user_count.get(),'3'); self.assertIsNone(self.flow.confirmed)
        self.assertFalse(self.flow.run()); self.assertTrue(self.flow.confirm())
        with patch.object(self.view,'start_pwm_autotune') as start:
            self.assertTrue(self.flow.run()); start.assert_called_once()
        self.assertTrue(self.view.pwm_pulse_codebook.get()); self.assertFalse(self.view.pwm_frame_reference_enabled.get())
    def test_invalid_reference_rejected_before_start(self):
        self.load_pwm(); self.view.filename.set(str(self.bin))
        self.view.pwm_frame_users[0]['reference']='999999'
        self.view.pwm_frame_reference_enabled.set(True)
        self.assertFalse(self.flow.confirm())
        with patch.object(self.view,'start_pwm_autotune') as start:
            self.assertFalse(self.flow.run()); start.assert_not_called()
    def test_bound_recipe_worker_not_reassembled(self):
        self.flow.switch(AREAS[1]); self.flow.replay={'example':'all options'}
        result,recovery=object(),object()
        class Immediate:
            def __init__(self,target,**kw): self.target=target
            def start(self): self.target()
        with patch('threading.Thread',Immediate), patch('rx706_qam_replay_adapter.run_replay',return_value=(result,recovery)) as run, patch('rx706_spectrum.display_trace',return_value=('freq','power')):
            self.flow.run_replay()
        self.assertEqual(run.call_args.args[0],{'example':'all options'})
        self.assertEqual(self.view.events.get_nowait(),('result',(result,('freq','power'),recovery,None)))
        self.view.busy=False; self.view.thread=None
    def test_no_reference_three_user_physical_erasure(self):
        import numpy as np
        from types import SimpleNamespace
        self.flow.switch(AREAS[2])
        rows=np.zeros(6,dtype=[('block_complete','?'),('frame_part','U8'),('user_id','i4'),('valid','?'),('identity_ok','?')])
        rows['block_complete']=True; rows['frame_part']='净荷'; rows['user_id']=[1,1,2,2,3,3]
        rows['valid']=True; rows['identity_ok']=True; rows['valid'][2]=False
        users=[dict(user_id=i,name=f'U{i}',ber=None,bler=None,bad_blocks=0,evaluated_blocks=0,erased_bits=0) for i in (1,2,3)]
        self.view.recovery=SimpleNamespace(kind='pwm2',pulse_options={'explicit':True},frames=rows,frame_check=SimpleNamespace(per_user=users))
        with patch('rx706_pwm_autotune.file_evaluation',return_value=dict(measured_symbol_rate_hz=170000,reference_overall=dict(ber=None,bler=None),erased_payload_symbols=1)):
            self.flow.results()
        self.assertIn('U2   BER 未测   BLER 未测   净荷擦除 1/2符号',self.flow.metrics.get())
        self.assertIn('U1   BER 未测   BLER 未测   净荷擦除 0/2符号',self.flow.metrics.get())
        self.view.recovery=None

    def test_advanced_legacy_analysis_is_explicit(self):
        self.view.constellation_mode.set('PWM v1（旧协议）')
        self.view.show_advanced.set(True); self.view.refresh_parameter_visibility()
        self.assertEqual(self.flow.expert_button.winfo_manager(),'pack')
        with patch.object(self.view,'analyze') as analyze:
            self.assertTrue(self.flow.expert_run()); analyze.assert_called_once()
        self.view.show_advanced.set(False); self.view.refresh_parameter_visibility()
        self.assertFalse(self.flow.expert_button.winfo_manager())

    def test_editor_blocks_navigation(self):
        editor=tk.Toplevel(self.root); editor.withdraw(); self.view._workflow_editor=editor
        self.assertFalse(self.flow.switch(AREAS[1])); self.assertFalse(self.flow.run()); editor.destroy()

if __name__=='__main__': unittest.main()

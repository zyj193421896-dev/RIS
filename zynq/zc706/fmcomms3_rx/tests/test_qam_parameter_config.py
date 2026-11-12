"""Portable backend parameters and redesigned strict GUI loading, without Tk."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_config import METHOD,read_parameter_config,validate_parameter_config
from rx706_qam import recover_qam
from rx706_workflows import ReceiverWorkflows,AREAS
from test_four_corner_training import capture,PAYLOAD

CONFIG=Path(__file__).resolve().parents[1]/'host/frame_profiles/acceptance/rx_256qam_120k_64slice_3_4_16_no_crc_config.json'

class ParameterConfigTests(unittest.TestCase):
    def test_delivered_json_is_portable_valid_and_idempotent(self):
        raw=json.loads(CONFIG.read_text());cfg=read_parameter_config(CONFIG)
        self.assertEqual(cfg,validate_parameter_config(cfg));self.assertEqual(cfg['method'],METHOD)
        for k in ('input_fingerprint','input_path','sample_rate_hz','center_hint_hz','channel','first_tuple'):
            self.assertNotIn(k,raw)
        self.assertEqual(cfg['symbol_hz'],120000);self.assertEqual(cfg['tdcm_options']['compensation']['frame_symbols'],20)
        self.assertEqual(cfg['tdcm_options']['payload_data_symbols'],16)
        self.assertFalse(cfg['tdcm_options']['crc_debug']['enabled']);self.assertEqual(cfg['tdcm_options']['payload_tail_symbols'],0)

    def test_invalid_parameters_rejected_without_mutation(self):
        for edit in (lambda c:c.update(input_fingerprint={}),lambda c:c.update(center_hint_hz=800000),
                     lambda c:c.update(symbol_hz=float('nan')),
                     lambda c:c['tdcm_options']['compensation'].update(frame_symbols=136),
                     lambda c:c['tdcm_options'].update(replay_recipe={})):
            cfg=json.loads(CONFIG.read_text());edit(cfg);before=json.dumps(cfg,sort_keys=True)
            with self.assertRaises(ValueError):validate_parameter_config(cfg)
            self.assertEqual(before,json.dumps(cfg,sort_keys=True))

    def test_same_parameters_run_two_independent_iq_files(self):
        shared=None
        for case in (dict(phase=.3,scale=.7),dict(phase=-1.7,scale=1.3,mirrored=True)):
            with tempfile.TemporaryDirectory() as directory:
                spectrum,options=capture(directory,**case)
                if shared is None:
                    td=copy.deepcopy(options['tdcm_options']);td.update(acquisition_mode='pwm',payload_data_symbols=12)
                    shared=validate_parameter_config(dict(version=1,method=METHOD,order=256,symbol_hz=options['symbol_hz'],
                        tdcm_options=td,pwm_preamble_options=options['pwm_preamble_options'],preamble_options=options['preamble_options']))
                saved=json.dumps(shared,sort_keys=True)
                kwargs={k:copy.deepcopy(shared[k]) for k in ('order','symbol_hz','tdcm_options','pwm_preamble_options','preamble_options')}
                result=recover_qam(spectrum,duration_s=1,center_hint_hz=120000,**kwargs)
                for frame in np.unique(result.frame_ids)[2:-2]:np.testing.assert_array_equal(result.labels[result.frame_ids==frame],PAYLOAD)
                self.assertGreaterEqual(len(result.training_records),28*4)
                self.assertIsNone(result.diagnostics['ber']);self.assertEqual(saved,json.dumps(shared,sort_keys=True))

    def workflow(self):
        v=SimpleNamespace(busy=False,window=None,filename=Mock(),status=Mock(),analyze=Mock(),
                          _receiver_changed=Mock(),_invalidate_analysis=Mock())
        for name in ('first_tuple','nfft','averages','channel','rate','remove_dc','recovery_ms','qam_order'):
            setattr(v,name,Mock())
        v.filename.get.return_value='/tmp/current.bin'
        return SimpleNamespace(view=v,area=AREAS[1],replay=None,confirmed=None,accepted=Mock(),refresh=Mock())

    def test_bound_dialog_loads_whole_recipe_without_analysis_or_parameter_rewrite(self):
        w=self.workflow();recipe=json.loads((Path(__file__).resolve().parent/'fixtures/qam_replay_synthetic.json').read_text());recipe['input_path']='/tmp/current.bin'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'recipe.json';p.write_text(json.dumps({'method':'iq_bound_tdcm_replay'}))
            with patch('rx706_qam_replay_adapter.load_replay',return_value=recipe) as loader, patch('rx706_workflows.messagebox.showerror') as error:
                self.assertTrue(ReceiverWorkflows.load(w,str(p)),error.call_args)
            loader.assert_called_once_with(str(p),input_path='/tmp/current.bin')
        self.assertEqual(w.replay,recipe);w.view.analyze.assert_not_called();w.accepted.assert_called_once()

    def test_unsupported_legacy_format_fails_without_mutating_capture(self):
        for config in (dict(version=1,method='bounded_repeated_tdcm'),):
            w=self.workflow()
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'old.json';p.write_text(json.dumps(config))
                with patch('rx706_workflows.messagebox.showerror') as error:
                    self.assertFalse(ReceiverWorkflows.load(w,str(p)));error.assert_called_once()
            w.view.filename.set.assert_not_called();w.view.analyze.assert_not_called();self.assertIsNone(w.replay)

if __name__=='__main__':unittest.main()

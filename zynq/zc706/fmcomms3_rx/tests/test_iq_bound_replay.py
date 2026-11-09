"""File identity and GUI dispatch regression without Tk or hardware."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from rx706_iq_replay import read_recipe,verify_input,make_recipe
from test_qam64_controls import controls

BASE=Path('/Users/carrot29/work/code/ris/数据')
REPLAY=BASE/'rx_256qam_64slice_100khz_comp_4pliot_replay.json'
PARAM=BASE/'rx_256qam_64slice_100khz_comp_4pliot_candidate_config.json'

@unittest.skipUnless(REPLAY.is_file() and PARAM.is_file(), "需要本次真实采集及配置文件")
class BoundReplayTests(unittest.TestCase):
    def test_real_capture_binding(self):
        r=read_recipe(REPLAY);self.assertEqual(verify_input(r),Path(r['input_path']))
        self.assertEqual(r['spectrum']['sample_rate_hz'],15360000)
        self.assertEqual(r['recovery']['center_hint_hz'],798105)
        self.assertEqual(r['recovery']['tdcm_options']['compensation']['period_ppm'],45.375)

    def test_gui_load_restores_and_schedules(self):
        v=controls();v.analyze=Mock()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(REPLAY)),patch('rx706_spectrum_gui.messagebox.showerror') as error:
            v.load_qam_recipe();error.assert_not_called()
        v.analyze.assert_called_once();self.assertEqual(v.filename.get(),str(BASE/'rx_256qam_64slice_100khz_comp_4pliot.bin'))
        self.assertEqual(v.rate.get(),'15.36');self.assertEqual(v.tdcm_carrier_mhz.get(),'0.798105')
        self.assertEqual(v.first_tuple.get(),'0');self.assertTrue(v.qam_compensation.auto_open.get())
        self.assertEqual(v._next_iq_bound_recipe['recovery']['tdcm_options']['compensation']['frame_symbols'],260)

    def test_headless_gui_load_worker_and_plot(self):
        v=controls();v.show_qam_report=Mock()
        with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(REPLAY)),patch('rx706_spectrum_gui.messagebox.showerror') as error,patch('rx706_spectrum_gui.threading.Thread') as worker:
            v.load_qam_recipe();error.assert_not_called();self.assertTrue(worker.called)
            name,sp,rec=worker.call_args.kwargs['args']
        v._analyze_worker(name,sp,rec);v.poll()
        self.assertIsNotNone(v.recovery,v.status.get())
        self.assertEqual(sum(bool(f.get('valid')) for f in v.recovery.frames),376)
        self.assertAlmostEqual(v.recovery.evm_percent,10.3449736631,places=5)
        self.assertTrue(v.show_qam_report.called)
        self.assertTrue(len(v.iq_axes.collections)>0)

    def test_missing_or_changed_file_stops(self):
        r=read_recipe(REPLAY)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'sample.bin';p.write_bytes(bytes(2048))
            r['input_path']=str(p)
            with self.assertRaisesRegex(ValueError,'SHA-256'):verify_input(r)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'不存在'):verify_input(r)

    def test_gui_missing_file_does_not_launch_or_change_controls(self):
        r=read_recipe(REPLAY);r['input_path']='/missing/iq.bin';v=controls();v.analyze=Mock();old=v.filename.get()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'recipe.json';p.write_text(json.dumps(r))
            with patch('rx706_spectrum_gui.filedialog.askopenfilename',return_value=str(p)),patch('rx706_spectrum_gui.messagebox.showerror') as error:
                v.load_qam_recipe();error.assert_called_once()
        v.analyze.assert_not_called();self.assertEqual(v.filename.get(),old)

    def test_new_file_generation_and_hash_change(self):
        cfg=json.loads(PARAM.read_text())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.bin';p.write_bytes(bytes(8192))
            r=make_recipe(p,cfg,sample_rate_hz=15360000,channel=1,first_tuple=0,duration_s=1024/15360000,center_hint_hz=0)
            self.assertEqual(verify_input(r),p.resolve())
            p.write_bytes(b'X'+bytes(8191))
            with self.assertRaisesRegex(ValueError,'SHA-256'):verify_input(r)

if __name__=='__main__':unittest.main()

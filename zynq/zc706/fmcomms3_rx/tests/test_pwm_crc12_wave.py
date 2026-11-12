import sys,json,unittest,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'host'))
from rx706_pwm_crc12 import test_frames
from rx706_pwm_pulse import decode_pulses
class Wave(unittest.TestCase):
 def run_case(self,missing=False,iq=False,flip=None):
  p=json.loads((Path(__file__).resolve().parents[1]/'host/pwm_frame_three_users_crc12.json').read_text())
  o=dict(p,step_percent=p['duty_step'],symbol_count=p['duty_count'],frame_options=dict(p,reference_format='symbols'))
  rate=169971.671;fs=15.36e6;s=np.tile(np.array(test_frames()).ravel(),4);origin=.37/rate
  if flip is not None:s[60+flip]^=1
  rises=origin+np.arange(len(s))/rate;falls=rises+(5+s)/20/rate+.2e-6
  if missing:rises=np.delete(rises,np.arange(60,120));falls=np.delete(falls,np.arange(60,120))
  t=np.arange(int((origin+(len(s)+.2)/rate)*fs))/fs
  k=np.searchsorted(rises,t,side='right')-1;states=(k>=0)&(t<falls[np.maximum(k,0)])
  edges=np.r_[rises,falls];after=np.r_[np.ones(len(rises),bool),np.zeros(len(falls),bool)];order=np.argsort(edges)
  if iq:
   from rx706_spectrum import analyze_bin
   from rx706_pwm import recover_pwm2
   x=np.where(states,-300,300)*np.exp(1j*(2*np.pi*713456.7*t+.43))
   raw=np.zeros((len(t),4),dtype='<i2');raw[:,0]=np.rint(x.real);raw[:,1]=np.rint(x.imag)
   with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'crc.bin';raw.tofile(path)
    spectrum=analyze_bin(path,nfft=4096,averages=1,sample_rate_hz=fs)
    result=recover_pwm2(spectrum,symbol_hz=rate,pulse_options=o)
   self.assertTrue(result.decode_locked,result.decode_note)
   from rx706_pwm_autotune import metrics
   from rx706_pwm_pulse import validate_pulse_options
   stat=metrics(result,len(t)/fs,validate_pulse_options(o,rate,fs),rate)
   self.assertGreater(stat['usable_fraction'],.9)
   return result.pulse_options['air_crc12_report']['overall']
  rows,_,_,c=decode_pulses(states,edges[order],after[order],t,fs,rate,o)
  return c['air_crc12_report']['overall']
 def test_corruptions(self):
  for pos in (0,5,12,18):
   with self.subTest(pos=pos):self.assertEqual(self.run_case(flip=pos)["failed_slots"],1)
 def test_iq(self):
  s=self.run_case(iq=True);self.assertGreaterEqual(s["expected_complete_slots"],10);self.assertEqual(s["failed_slots"],0)
 def test_ideal(self):
  s=self.run_case();self.assertEqual(s['expected_complete_slots'],12);self.assertEqual(s['failed_slots'],0)
 def test_silent_round(self):self.assertEqual(self.run_case(True)['failed_slots'],3)
if __name__=='__main__':unittest.main()

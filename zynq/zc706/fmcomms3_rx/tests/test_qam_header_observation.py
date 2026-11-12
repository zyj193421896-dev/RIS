"""Noise robustness uses known header geometry, never payload labels."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host'))
from rx706_qam_pwm import acquire_pwm_header, validate_pwm_header, header_edges, smooth_pwm_observation
from rx706_qam10_profile import build_qam10_config

class HeaderObservationTests(unittest.TestCase):
    def test_fixed_filter_preserves_length_and_timestamp(self):
        o=build_qam10_config(symbol_hz=20000)['pwm_preamble_options'];o['observation_average_chips']=1.
        x=np.zeros(1000,complex);x[500]=1;before=x.copy()
        y=smooth_pwm_observation(x,15360000,o)
        self.assertEqual(len(x),len(y));self.assertAlmostEqual(np.sum(y).real,1.)
        self.assertAlmostEqual(np.sum(np.arange(len(y))*y).real,500.)
        np.testing.assert_array_equal(x,before)
    def test_low_snr_known_header_and_wrong_sequence(self):
        fs=15360000.;o=build_qam10_config(symbol_hz=20000)['pwm_preamble_options'];o['search_s']=.001
        pwm=validate_pwm_header(o,fs);size=7680;lead=1200
        h=np.zeros(size,bool)
        for a,b in zip(*header_edges(pwm,768)):h[a:b]=True
        rng=np.random.default_rng(731)
        x= .5*(rng.normal(size=17000)+1j*rng.normal(size=17000))
        x[lead:lead+size]+=h-.5
        with self.assertRaises(ValueError):acquire_pwm_header(x,fs,o)
        filtered=copy.deepcopy(o);filtered['observation_average_chips']=1.
        result=acquire_pwm_header(x,fs,filtered)
        self.assertAlmostEqual(result['index'],lead,delta=8)
        self.assertLess(result['error_percent'],o['max_error_percent'])
        wrong=copy.deepcopy(filtered);wrong['pulse_chips']=wrong['pulse_chips'][::-1]
        with self.assertRaises(ValueError):acquire_pwm_header(x,fs,wrong)
    def test_short_input_rejects_before_smoothing_allocation(self):
        o=build_qam10_config(symbol_hz=20000)['pwm_preamble_options']
        o.update(symbol_hz=1e-9,max_frequency_hz=1e-10,observation_average_chips=1.)
        with self.assertRaisesRegex(ValueError,'不足以容纳完整PWM前导'):
            acquire_pwm_header(np.zeros(100,complex),15360000.,o)

    def test_filter_validation_and_legacy_default(self):
        o=build_qam10_config(symbol_hz=20000)['pwm_preamble_options']
        self.assertNotIn('observation_average_chips',validate_pwm_header(o,15360000))
        from rx706_qam_config import validate_parameter_config
        cfg=build_qam10_config(symbol_hz=20000);cfg['pwm_preamble_options']['observation_average_chips']=0.
        self.assertEqual(validate_parameter_config(cfg)['pwm_preamble_options']['observation_average_chips'],0.)
        for v in (True,-1,1.01,float('nan'),'1'):
            with self.assertRaises(ValueError):validate_pwm_header(dict(o,observation_average_chips=v),15360000)

if __name__=='__main__':unittest.main()

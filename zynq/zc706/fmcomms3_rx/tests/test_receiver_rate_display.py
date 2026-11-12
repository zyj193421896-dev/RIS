"""Captured-time payload rates and theme-aware hint contrast."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
from rx706_pwm_autotune import file_evaluation, payload_rates
from rx706_ui_contrast import contrast


def recovery(seconds=1, decimation=1, first=0):
    rows = np.zeros(8, dtype=[('block_complete', '?'), ('frame_part', 'U8'),
                             ('valid', '?'), ('identity_ok', '?'), ('block_id', 'i4')])
    rows['block_complete'][:6] = True
    rows['frame_part'] = ['身份', '净荷', '净荷', '净荷', '净荷', '净荷', '净荷', '净荷']
    rows['valid'] = True; rows['valid'][2] = False; rows['identity_ok'] = True
    return SimpleNamespace(frames=rows, input_samples=round(15360000*seconds),
        output_rate_hz=15360000/decimation, decimation=decimation, first_tuple=first,
        measured_symbol_rate_hz=170000, bits_per_symbol=7,
        pulse_options=dict(clock_source='多个身份前导的长间隔拟合', total_chips=20, codebook=list(range(66)),
            frame_options=dict(users=[dict(identity=[0]*4, payload_symbols=66)]),
            cycle_calibration=dict(payload_indices=[0, 1, 2, 3])),
        frame_check=SimpleNamespace(overall=dict(ber=None, bler=None)))


class RateDisplayTests(unittest.TestCase):
    def test_one_second_nonpower_two_capacity_and_no_reference(self):
        e = file_evaluation(recovery())
        self.assertEqual(e['analyzed_duration_s'], 1)
        self.assertEqual(e['complete_payload_symbols'], 5)
        self.assertEqual(e['decided_payload_symbol_rate_hz'], 4)
        self.assertEqual(e['erased_payload_symbols'], 1)
        self.assertEqual(e['physical_chip_rate_hz'], 3400000)
        self.assertAlmostEqual(e['decided_payload_information_ceiling_bps'], 4*math.log2(66))
        self.assertNotEqual(e['information_bits_per_symbol'], 7)
        self.assertAlmostEqual(e['payload_information_ceiling_bps'], 170000*math.log2(66)*62/70)
        self.assertIsNone(e['ber']); self.assertIsNone(e['actual_goodput_bps'])

    def test_non_one_second_decimation_and_offset(self):
        e = file_evaluation(recovery(.25, 4, 123456))
        self.assertEqual(e['analyzed_duration_s'], .25)
        self.assertEqual(e['analyzed_first_tuple'], 123456)
        self.assertEqual(e['decided_payload_symbol_rate_hz'], 16)

    def test_empty_partial_erased_and_gapped_counts(self):
        for mode in ('empty', 'partial', 'erased'):
            r = recovery()
            if mode == 'empty': r.frames = r.frames[:0]
            elif mode == 'partial': r.frames['block_complete'] = False
            else: r.frames['valid'] = False
            e = file_evaluation(r)
            self.assertEqual(e['decided_payload_symbol_rate_hz'], 0)
            self.assertEqual(e['analyzed_duration_s'], 1)
        r = recovery(); r.frames = r.frames[3:4]
        self.assertEqual(file_evaluation(r)['decided_payload_symbol_rate_hz'], 1)

    def test_missing_or_invalid_duration_is_unknown(self):
        for value in (0, -1, float('nan'), None):
            r = recovery(); r.output_rate_hz = value
            self.assertIsNone(payload_rates(r, 4, 5)['decided_payload_symbol_rate_hz'])
        r = recovery(); del r.input_samples
        self.assertIsNone(payload_rates(r, 4, 5)['analyzed_duration_s'])

    def test_physical_equivalent_includes_all_slots_and_ignores_erasures(self):
        r = recovery()
        expected = 170000 * math.log2(66)
        self.assertAlmostEqual(file_evaluation(r)['physical_equivalent_ceiling_bps'], expected)
        r.pulse_options['frame_options']['users'][0]['identity'] = [0]*10
        self.assertAlmostEqual(file_evaluation(r)['physical_equivalent_ceiling_bps'], expected)
        r.frames['valid'] = False
        e = file_evaluation(r)
        self.assertAlmostEqual(e['physical_equivalent_ceiling_bps'], expected)
        self.assertEqual(e['decided_payload_equivalent_ceiling_bps'], 0)
        self.assertIsNone(e['actual_transport_bps'])
        r.pulse_options['codebook'] = list(range(64))
        self.assertEqual(file_evaluation(r)['physical_equivalent_ceiling_bps'], 170000*6)

    def test_no_clock_fit_does_not_claim_measured_physical_bps(self):
        for source in ('', '配置符号率（仅一个完整身份前导）'):
            r = recovery(); r.pulse_options['clock_source'] = source
            e = file_evaluation(r)
            self.assertFalse(e['physical_rate_measured'])
            self.assertIsNone(e['physical_equivalent_ceiling_bps'])
            self.assertIsNone(e['physical_chip_rate_hz'])
            self.assertIsNone(e['physical_symbol_rate_hz'])
            self.assertEqual(e['decided_payload_symbol_rate_hz'], 4)

    def test_headline_reset_and_qam_unchanged(self):
        from rx706_workflows import ReceiverWorkflows, AREAS
        from unittest.mock import Mock
        r = recovery(); r.kind = 'pwm2'
        flow = SimpleNamespace(view=SimpleNamespace(recovery=r), metrics=Mock(), area=AREAS[0])
        ReceiverWorkflows.results(flow)
        text = flow.metrics.set.call_args.args[0]
        self.assertIn('物理码率等效上限 1.027547 Mbit/s（含前导/训练）', text)
        self.assertIn('已判决净荷等效上限', text)
        self.assertIn('BER 未测', text)
        r.pulse_options['clock_source'] = ''
        ReceiverWorkflows.results(flow)
        self.assertIn('物理码率等效上限 未测', flow.metrics.set.call_args.args[0])
        r.pulse_options['clock_source'] = '多个身份前导的长间隔拟合'
        r.decode_locked = False
        self.assertIsNone(file_evaluation(r)['physical_equivalent_ceiling_bps'])
        flow.view.recovery = None
        ReceiverWorkflows.results(flow)
        self.assertNotIn('1.027', flow.metrics.set.call_args.args[0])
        flow.view.recovery = SimpleNamespace(kind='qam', order=256, scatter=[0], evm_percent=2)
        ReceiverWorkflows.results(flow)
        self.assertEqual(flow.metrics.set.call_args.args[0], 'TDCM 256QAM   1 净荷符号   净荷最近邻EVM 2.00%   BER 未测\n已知训练EVM 未测')

    def test_amber_contrast_light_and_dark(self):
        rgb = lambda s: tuple(int(s[i:i+2], 16)*257 for i in (1, 3, 5))
        self.assertGreaterEqual(contrast(rgb('#ffe08a'), rgb('#242424')), 4.5)
        self.assertGreaterEqual(contrast(rgb('#704800'), rgb('#fafafa')), 4.5)


if __name__ == '__main__': unittest.main()

"""MT percent import, waveform-preserving index migration and legacy compatibility."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'host'))
from rx706_pwm_symbols import (PLUS_SOURCE, PLUS_SHA256, LEGACY_PROFILE, LEGACY_SHA256,
    plus_fields, parse_mt_percent, remap_symbols, import_mt_fields)
from rx706_pwm_frame_builder import initial_fields, build_profile, save_new_profile, preview_profile
from rx706_pwm_pulse import validate_pulse_options, decode_pulses
from rx706_pwm_frames import check_frame_reference
from rx706_pwm_autotune import search_options


def profile78(reference=False):
    fields = plus_fields()
    fields.update(payload_symbols='78', symbol_khz='169.971671', reference_enabled=reference,
                  reference=' '.join(map(str, range(78))))
    return build_profile(fields)


def edge_case(tail_chips=.5, rounds=6, corrupt=False):
    rate, fs = 169971.671, 15.36e6
    profile = profile78(True)
    options = search_options(profile)
    book = np.asarray(profile['codebook'])
    cycle = np.r_[profile['users'][0]['identity'], np.arange(78)]
    sent = np.tile(cycle, rounds)
    origin = .37/rate
    rises = origin+(np.arange(len(sent))+book[sent, 0]/20)/rate
    falls = rises+(book[sent, 1]+tail_chips)/20/rate
    t = np.arange(int((origin+(len(sent)+.3)/rate)*fs))/fs
    at = np.searchsorted(rises,t,side='right')-1
    high = (at>=0) & (t<falls[np.maximum(at,0)])
    # Independent full high-state union, including merged adjacent pulses.
    if tail_chips>=4:
        high = np.searchsorted(rises,t,side='right')>np.searchsorted(falls,t,side='right')
        changes = np.flatnonzero(high[1:] != high[:-1])+1
        times, after = (t[changes]+t[changes-1])/2, high[changes]
    else:
        times = np.r_[rises, falls]
        after = np.r_[np.ones(len(sent),bool),np.zeros(len(sent),bool)]
        order = np.argsort(times);times,after=times[order],after[order]
    rows, measured, calibration, normalized = decode_pulses(high,times,after,t,fs,rate,options)
    return profile,sent,rows,measured,calibration,normalized


class Symbols78Tests(unittest.TestCase):
    def test_source_exact_order_and_original_unchanged(self):
        self.assertEqual(hashlib.sha256(PLUS_SOURCE.read_bytes()).hexdigest(),PLUS_SHA256)
        self.assertEqual(hashlib.sha256(LEGACY_PROFILE.read_bytes()).hexdigest(),LEGACY_SHA256)
        book=parse_mt_percent(PLUS_SOURCE.read_text())
        expected=tuple((d,w) for d in range(12) for w in range(5,17-d))
        self.assertEqual(book,expected);self.assertEqual(len(book),78)
        old=json.loads(LEGACY_PROFILE.read_text())
        legacy=initial_fields(old)['codebook']
        mapped=remap_symbols(' '.join(map(str,range(66))),legacy,book)
        self.assertEqual(len(set(mapped)),66)
        self.assertEqual([i for i,pair in enumerate(book) if sum(pair)==16], [11,22,32,41,49,56,62,67,71,74,76,77])

    def test_header_mapping_payload_not_inferred_and_no_stale_reference(self):
        f=plus_fields()
        self.assertEqual(f['identity'],'75 8 60 2 46 10 23 6 68 38')
        self.assertEqual(f['payload_symbols'],'')
        self.assertEqual(f['low_guard_chips'],'4')
        with self.assertRaises(ValueError):build_profile(f)
        for length in (66,78,109):
            p=build_profile(dict(f,payload_symbols=str(length)))
            self.assertEqual(p['users'][0]['payload_symbols'],length)
            self.assertEqual(p['users'][0]['reference'],'')
            self.assertFalse(p['reference_enabled']);self.assertEqual(p['cycle_calibration'],{})

    def test_percent_format_bom_and_rejections(self):
        self.assertEqual(parse_mt_percent('\ufeff (25,0);\n\n (30 , 5)\n'),((0,5),(1,6)))
        for invalid in ('', '25 0\n30 0', '(0,25);\n(5,30);', '(25,0);\n(25,0);',
            '(25,0);\n(nan,5);', '(25,0);\n(30,inf);', '(25,0);\n(26,5);',
            '(25,0);\n(30,-5);', '(25,0);\n(80,20);', '(25,0);junk\n(30,5);'):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):parse_mt_percent(invalid)
        for step in (0,-5,float('nan'),7):
            with self.assertRaises(ValueError):parse_mt_percent('(25,0);\n(30,5);',step)

    def test_import_atomic_missing_waveform_and_clears_old_lengths(self):
        fields=initial_fields(json.loads(LEGACY_PROFILE.read_text()))
        before=copy.deepcopy(fields)
        with self.assertRaisesRegex(ValueError,'缺少'):
            import_mt_fields(fields,'(25,0);\n(30,0);')
        self.assertEqual(fields,before)
        imported=import_mt_fields(fields,PLUS_SOURCE.read_text())
        self.assertEqual(fields,before);self.assertEqual(imported['payload_symbols'],'')
        self.assertEqual(imported['reference'],'');self.assertEqual(imported['calibration_symbols'],'')

    def test_guard5_and_width_only_rejected(self):
        p=profile78()
        for change in ({'low_guard_chips':5},{'joint_delay':False}):
            opt=search_options(p);opt.update(change)
            with self.assertRaises(ValueError):validate_pulse_options(opt,169971.671)
        opt=validate_pulse_options(search_options(p),169971.671)
        self.assertEqual(opt['symbol_count'],78)
        self.assertAlmostEqual(opt['information_bits'],math.log2(78))
        self.assertEqual(opt['integer_payload_bits'],6)

    def test_save_roundtrip_no_overwrite_and_preview(self):
        p=profile78()
        with tempfile.TemporaryDirectory() as directory:
            path=save_new_profile(Path(directory)/'new.json',p)
            self.assertEqual(json.loads(path.read_text()),p)
            with self.assertRaises(FileExistsError):save_new_profile(path,p)
        text=preview_profile(p)
        self.assertIn('符号77',text);self.assertIn('BER未测',text)

    def test_all78_symbols_and_index_ber_independent_of_reference(self):
        profile,sent,rows,rate,cals,cfg=edge_case()
        payload=rows['frame_part']=='净荷'
        np.testing.assert_array_equal(rows['symbol'][payload],sent[payload])
        self.assertTrue(rows['valid'][payload].all())
        self.assertAlmostEqual(rate,169971.671,places=5)
        words=tuple(format(k,'07b') for k in range(78))
        frame=dict(cfg['frame_options'],users=profile['users'],reference_enabled=True,reference_format='symbols')
        check=check_frame_reference(rows,frame,tuple(w for d,w in profile['codebook']),words)
        self.assertEqual(check.overall['compared_bits'],6*78*7)
        self.assertEqual(check.overall['ber'],0)
        disabled=check_frame_reference(rows,dict(frame,reference_enabled=False),tuple(w for d,w in profile['codebook']),words)
        self.assertIsNone(disabled.overall['ber'])
        # Post-decision scoring must detect an incorrect label and a separate erasure.
        damaged=rows.copy();damaged['symbol'][10]=1;damaged['bits'][10]='0000001'
        damaged['valid'][11]=False;damaged['bits'][11]='???????'
        score=check_frame_reference(damaged,frame,tuple(w for d,w in profile['codebook']),words)
        self.assertEqual(score.overall['bit_errors'],1)
        self.assertEqual(score.overall['erased_bits'],7)
        self.assertEqual(score.overall['compared_bits'],6*78*7-7)

    def test_tail_limit_is_not_a_reliability_guarantee(self):
        _,sent,rows,_,_,_=edge_case(tail_chips=3.9)
        payload=rows['frame_part']=='净荷'
        np.testing.assert_array_equal(rows['symbol'][payload],sent[payload])
        try:
            _,sent,rows,_,_,_=edge_case(tail_chips=6)
        except ValueError:
            return  # Failure to acquire is an honest failure, not zero BER.
        payload=rows['frame_part']=='净荷'
        positions=np.rint((rows['start_s']-.37/169971.671)*169971.671).astype(int)
        self.assertTrue(np.any(~rows['valid'][payload]) or not np.array_equal(rows['symbol'][payload],sent[positions[payload]]))

if __name__=='__main__':unittest.main()

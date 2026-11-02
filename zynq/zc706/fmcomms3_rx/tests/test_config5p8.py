"""Decode the exported RF register program independently of firmware execution."""
import hashlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from generate_ad9361_script import parse_script,render_header


class Config5p8Tests(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'config/config5p8').read_bytes()
        self.operations=parse_script(self.source)
        self.regs={r:v for r,_,_,op,v,_ in self.operations if op==1}

    def test_generated_script_fidelity(self):
        self.assertEqual(len(self.operations),2568)
        self.assertEqual((ROOT/'vitis_code/key_ctrlll/src/ad_config/config5p8_data.h').read_text(),
                         render_header(self.source,'config5p8'))

    def test_both_lo_registers_decode_to_5800mhz(self):
        r=self.regs
        for base,shift in [(0x231,0),(0x271,4)]:
            integer=r[base]|((r[base+1]&7)<<8)
            fraction=r[base+2]|(r[base+3]<<8)|((r[base+4]&127)<<16)
            hz=80_000_000*(integer+fraction/8_388_593)/2**(1+((r[5]>>shift)&15))
            self.assertAlmostEqual(hz,5_800_000_000,delta=100)
            self.assertGreater(hz,2**32)

    def test_clocks_interface_fir_and_calibration_compatibility(self):
        old=parse_script((ROOT/'config/config2p4').read_bytes())
        # Exact ordered sequences for timing, clocks, ports, FIR coefficients and waits.
        protected=set(range(0x040,0x060))|set(range(0x060,0x066))|set(range(0x0f0,0x0f6))|{2,3,4,6,0xa,0x10,0x11,0x12,0x3d,0x3e}
        def select(ops): return [(r,t,op,v,m) for r,t,_,op,v,m in ops if (op==1 and r in protected) or op in (3,4)]
        self.assertEqual(select(self.operations),select(old))
        r=self.regs
        frac=(r[0x041]<<16)|(r[0x042]<<8)|r[0x043]
        bbpll=40_000_000*(r[0x044]+frac/2_088_960)
        adc=bbpll/2**(r[0x00a]&7)
        self.assertAlmostEqual(adc/16,15_360_000,delta=1)
        self.assertEqual(r[0x010],0xc8)


if __name__=='__main__': unittest.main()

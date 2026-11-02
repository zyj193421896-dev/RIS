"""Run the actual C STATUS serializer for all profiles, including >32-bit LO."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('gcc'),'native GCC required')
class ProfileJsonTests(unittest.TestCase):
    def test_all_profile_frequencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile,name in [(915,'CONFIG915'),(2400,'CONFIG2P4'),(5800,'CONFIG5P8')]:
                with self.subTest(profile=profile):
                    exe=Path(tmp)/f'profile{profile}.exe'
                    subprocess.run(['gcc','-std=c11','-O2','-flto','-fwhole-program',
                        f'-DRX_RF_PROFILE={profile}',f'-I{ROOT/"tests/network_stubs"}',
                        f'-I{ROOT/"vitis_code/key_ctrlll/src/drivers"}',
                        str(ROOT/'tests/test_profile_json.c'),
                        str(ROOT/'vitis_code/key_ctrlll/src/drivers/capture_store.c'),'-o',str(exe)],
                        check=True,capture_output=True,text=True)
                    data=json.loads(subprocess.check_output([str(exe)],text=True))
                    self.assertEqual(data['rx_lo_config_hz'],profile*1_000_000)
                    self.assertEqual(data['rf_profile'],name)
                    self.assertEqual(data['sample_rate_hz'],15_360_000)


if __name__=='__main__': unittest.main()

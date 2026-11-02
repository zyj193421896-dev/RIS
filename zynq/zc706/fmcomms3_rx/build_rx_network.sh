#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="/home/yyz/vivado2020.2/main/Vitis/2020.2/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/home/yyz/vivado2020.2/main/Vitis/2020.2/bin:$PATH"
python3 "$project_root/scripts/generate_ad9361_script.py" \
    "$project_root/config/config2p4" \
    "$project_root/vitis_code/key_ctrlll/src/ad_config/config2p4_data.h"
python3 "$project_root/scripts/generate_ad9361_script.py" \
    "$project_root/config/config5p8" \
    "$project_root/vitis_code/key_ctrlll/src/ad_config/config5p8_data.h" --symbol config5p8
if [[ ! -f "$project_root/vitis_code/network_bsp/ps7_cortexa9_0/lib/liblwip4.a" ]]; then
    xsct "$project_root/scripts/build_network_bsp.tcl"
fi
python3 "$project_root/scripts/stage_network_bsp.py"
make -C "$project_root/vitis_code/key_ctrlll/Debug" -j2 all
python3 "$project_root/scripts/stage_zc706_artifacts.py"

"""Rebuild the application and generated BSP with the installed Windows ARM GCC.

Outputs are isolated in reports; hardware and existing published ELFs are untouched.
No JTAG connection, reset, flash write or board command is issued by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from generate_ad9361_script import render_header

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=int,choices=[915,2400,5800],default=5800)
    parser.add_argument('--toolchain',type=Path,default=Path('D:/software/vivado/xilinx/SDK/2018.3/gnu/aarch32/nt/gcc-arm-none-eabi/bin'))
    parser.add_argument('--make',type=Path,default=Path('D:/software/vivado2020/Vivado/2020.2/gnuwin/bin/make.exe'))
    args=parser.parse_args()
    suffix={915:'915',2400:'2p4',5800:'5p8'}[args.profile]
    report=ROOT/f'reports/config{suffix}_20261028'
    report.mkdir(parents=True,exist_ok=True)
    source=ROOT/'vitis_code/key_ctrlll/src'
    for symbol in ['config2p4','config5p8']:
        (source/f'ad_config/{symbol}_data.h').write_text(render_header((ROOT/f'config/{symbol}').read_bytes(),symbol))
    bsp=ROOT/'vitis_code/network_bsp/ps7_cortexa9_0'
    gcc=args.toolchain/'arm-none-eabi-gcc.exe'; ar=args.toolchain/'arm-none-eabi-ar.exe'
    for path in [gcc,ar,args.make,bsp/'include/xparameters.h']:
        if not path.is_file(): raise FileNotFoundError(path)
    common=['-mcpu=cortex-a9','-mfpu=vfpv3','-mfloat-abi=hard','-Wall','-g3',f'-I{(bsp/"include").as_posix()}']
    commands=[]

    def compile_one(job):
        path,group,extra=job
        relative=path.relative_to(ROOT)
        obj=report/'obj'/group/relative.with_suffix('.o')
        obj.parent.mkdir(parents=True,exist_ok=True)
        opt='-O2' if group!='app' or path.stem in ['capture_format','capture_debug','dma_diag','capture_store','rx_network','rx_control','mig_ddr'] else '-O0'
        command=[str(gcc),*common,opt,*extra,'-c',str(path),'-o',str(obj)]
        result=subprocess.run(command,capture_output=True,text=True)
        commands.append(command)
        (obj.with_suffix('.log')).write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(f'{path}\n{result.stderr}')
        return obj

    xil_sources=[]
    for library in sorted((bsp/'libsrc').iterdir()):
        if library.is_dir() and not library.name.startswith('lwip'):
            xil_sources.extend(sorted((library/'src').glob('*.c')))
            xil_sources.extend(sorted((library/'src').glob('*.S')))
    lwip=next((bsp/'libsrc').glob('lwip*/src'))
    list_make=report/'list_lwip.mk'
    list_make.write_text('print_sources:\n\t@echo $(LWIP_SRCS) $(ADAPTER_SRCS)\n')
    output=subprocess.check_output([str(args.make),'-s','-f','Makefile','-f',str(list_make),'print_sources'],cwd=lwip,text=True)
    lwip_sources=[]
    for item in output.split():
        path=lwip/item
        if not path.exists():
            # The BSP makefile uses basename VPATH resolution for historical entries.
            matches=list(lwip.rglob(path.name))
            if len(matches)!=1: raise ValueError(f'Ambiguous lwIP source: {item}')
            path=matches[0]
        if path not in lwip_sources: lwip_sources.append(path)
    app_sources=sorted(source.rglob('*.c'))
    jobs=([(p,'xil',[]) for p in xil_sources]+[(p,'lwip',['-DPROCESSOR_LITTLE_ENDIAN']) for p in lwip_sources]+
          [(p,'app',[f'-DRX_RF_PROFILE={args.profile}']) for p in app_sources])
    print(f'Compiling BSP={len(xil_sources)} lwIP={len(lwip_sources)} application={len(app_sources)}',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects=list(pool.map(compile_one,jobs))
    lib=report/'lib'; lib.mkdir(exist_ok=True)
    nx=len(xil_sources); nl=len(lwip_sources)
    for name,items in [('libxil.a',objects[:nx]),('liblwip4.a',objects[nx:nx+nl])]:
        target=lib/name
        # A fresh archive avoids carrying stale members from a previous source list.
        staging=target.with_suffix('.new.a')
        if staging.exists(): staging.unlink()
        subprocess.run([str(ar),'rcs',str(staging),*[str(p) for p in items]],check=True)
        staging.replace(target)
    elf=report/f'key_ctrlll_{suffix}.elf'
    command=[str(gcc),*common[:3],'-Wl,-build-id=none',f'-specs={(source/"Xilinx.spec").as_posix()}',
        f'-Wl,-T,{(source/"lscript.ld").as_posix()}',f'-L{lib.as_posix()}',f'-Wl,-Map,{(report/"application.map").as_posix()}',
        '-o',str(elf),*[str(p) for p in objects[nx+nl:]],'-Wl,--start-group','-lxil','-llwip4','-lgcc','-lc','-Wl,--end-group']
    result=subprocess.run(command,capture_output=True,text=True)
    (report/'link.log').write_text(result.stdout+result.stderr)
    commands.append(command)
    (report/'commands.json').write_text(json.dumps(commands,indent=2))
    if result.returncode: raise RuntimeError(result.stderr)
    subprocess.run([str(args.toolchain/'arm-none-eabi-size.exe'),str(elf)],check=True)
    published=ROOT/f'export/key_ctrlll_{suffix}.elf'
    shutil.copy2(elf,published)
    manifest={'profile':args.profile,'configured_lo_hz':args.profile*1000000,
              'compiler':subprocess.check_output([str(gcc),'--version'],text=True).splitlines()[0],
              'validation':'Compiled and linked only; not executed on hardware.',
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [elf,published,ROOT/f'config/config{suffix}',ROOT/'export/zc706_top.bit']}}
    (report/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'BUILT {published}',flush=True)


if __name__=='__main__': main()

"""Assemble a standard-library-only Linux capture package; no receiver edits."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(output, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('package name must be a single directory name')
    output = Path(output).resolve()
    stage = output/name
    stage.mkdir(parents=True, exist_ok=True)
    source = ROOT/'tools'/'capture_portable'
    entries = {}
    for filename in ('start.sh', 'cli.sh', 'capture_gui.py', 'capture_menu.py', 'verify_capture.py', 'README.txt'):
        entries[filename] = (source/filename).read_bytes()
    for filename in ('rx706.py', 'rx706_gui.py'):
        entries['app/'+filename] = (ROOT/'host'/filename).read_bytes()
    entries['captures/README.txt'] = '完成下载后，将.bin和同名.bin.json一起拷回笔记本。\n'.encode('utf-8')
    for filename, data in entries.items():
        target = stage/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Explicit file allowlist prevents accidental inclusion of captures, local
    # settings or third-party packages on repeated builds.
    manifest = ''.join(hashlib.sha256(data).hexdigest()+'  '+filename+'\n'
                       for filename, data in sorted(entries.items()))
    entries['MANIFEST.sha256'] = manifest.encode('utf-8')
    (stage/'MANIFEST.sha256').write_bytes(entries['MANIFEST.sha256'])
    archives = [output/(name+'.tar.gz'), output/(name+'.zip')]
    with tarfile.open(archives[0], 'w:gz') as archive:
        for filename in sorted(entries):
            target = stage/filename
            info = archive.gettarinfo(str(target), arcname=name+'/'+filename)
            info.mode = 0o755 if filename.endswith('.sh') else 0o644
            with target.open('rb') as stream:
                archive.addfile(info, stream)
    with zipfile.ZipFile(archives[1], 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(entries):
            info = zipfile.ZipInfo(name+'/'+filename)
            info.external_attr = ((0o100755 if filename.endswith('.sh') else 0o100644) << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entries[filename])
    for archive in archives:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n', encoding='ascii')
        print('%s (%d bytes)' % (archive, archive.stat().st_size))
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist')
    parser.add_argument('--name', default='rx706_capture_linux_20261109')
    args = parser.parse_args()
    build(args.output, args.name)


if __name__ == '__main__':
    main()

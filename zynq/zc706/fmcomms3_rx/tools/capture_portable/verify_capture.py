"""Verify copied raw IQ against its .bin.json metadata (streaming SHA256)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'app'))
import rx706 as protocol


def verify(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf-8-sig'))
    expected = metadata.get('sha256', '')
    size = protocol.validate_capture(metadata['capture'])
    if metadata.get('complete') is not True or metadata.get('file_bytes') != size or path.stat().st_size != size:
        raise ValueError('数据未完成，或文件长度与采集记录不一致')
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('元数据缺少有效SHA256')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('SHA256不匹配，文件搬运后内容损坏或元数据不匹配')
    return dict(path=str(path), bytes=size, sha256=expected, capture_id=metadata['capture']['capture_id'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path, help='completed .bin files; keep matching .bin.json beside each')
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        try:
            result = verify(path)
            print('校验通过：%s；%d字节；SHA256 %s' % (path, result['bytes'], result['sha256']))
        except (OSError, ValueError, KeyError, protocol.ProtocolError) as exc:
            failed = True
            print('校验失败：%s：%s' % (path, exc), file=sys.stderr)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())

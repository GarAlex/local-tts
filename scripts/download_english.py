"""Download pinned English assets and verify known SHA-256 checksums."""
import argparse
import shutil
import tempfile
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'voices/kokoro-en'

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def download(out, url, name, expected):
    target = out / name
    if target.exists() and digest(target) == expected:
        print(f'Already verified: {name}', flush=True)
        return
    print(f'Downloading {name}', flush=True)
    with tempfile.NamedTemporaryFile(dir=out, prefix=name + '.', suffix='.part', delete=False) as stream:
        temp = Path(stream.name)
    try:
        cached = BASE / name
        if cached != target and cached.is_file() and digest(cached) == expected:
            shutil.copyfile(cached, temp)
        else:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if digest(temp) != expected:
            raise ValueError(f'Checksum mismatch for {name}; existing file was not replaced')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', nargs='?', choices=['f32', 'q8', 'q4', 'all'], default='f32')
    args = parser.parse_args()
    variants = json.loads((ROOT / 'assets/kokoro-variants.json').read_text())
    for variant in variants if args.variant == 'all' else [args.variant]:
        info = variants[variant]
        out = ROOT / 'voices' / info['folder']
        out.mkdir(parents=True, exist_ok=True)
        records = json.loads((ROOT / 'assets/english-assets.json').read_text())
        records['model.onnx'] = dict(url=info['url'], sha256=info['sha256'])
        for name, record in records.items():
            download(out, record['url'], name, record['sha256'])
        (out / 'provenance.json').write_text(json.dumps(dict(variant=variant, assets=records), indent=2))
        print(f'Ready: {info["label"]} at {out}', flush=True)

if __name__ == '__main__': main()

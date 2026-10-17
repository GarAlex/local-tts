#!/usr/bin/env python3
"""Download pinned native RU/UK inference assets; Python is acquisition-only."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def fetch(url, target, expected):
    if target.exists() and digest(target) == expected:
        print(f'Already verified: {target.name}', flush=True)
        return
    print(f'Downloading {target.name}', flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        partial = Path(temp.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if digest(partial) != expected:
            raise ValueError(f'Checksum mismatch for {target.name}; existing file preserved')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['ru', 'uk', 'all'])
    args = parser.parse_args()
    for language in (['ru','uk'] if args.language == 'all' else [args.language]):
        name = 'russian' if language == 'ru' else 'ukrainian'
        records = json.loads((ROOT / f'assets/{name}-assets.json').read_text())
        out = ROOT / ('voices/vosk-ru' if language == 'ru' else 'voices/piper-uk')
        out.mkdir(parents=True, exist_ok=True)
        if language == 'ru':
            archive = records['archive']
            cache = ROOT / 'target/pretrained-evaluation/vosk-ru-0.7.zip'
            cache.parent.mkdir(parents=True, exist_ok=True)
            fetch(archive['url'], cache, archive['sha256'])
            with zipfile.ZipFile(cache) as package:
                for filename, record in records['files'].items():
                    if 'archive_path' not in record:
                        fetch(record['url'], out/filename, record['sha256'])
                        continue
                    if (out/filename).exists() and digest(out/filename) == record['sha256']:
                        continue
                    with tempfile.NamedTemporaryFile(dir=out, delete=False) as temp:
                        partial = Path(temp.name)
                        try:
                            with package.open(record['archive_path']) as stream:
                                while chunk := stream.read(1024*1024): temp.write(chunk)
                            temp.flush()
                            if digest(partial) != record['sha256']:
                                raise ValueError(f'Archive checksum mismatch for {filename}')
                            partial.replace(out/filename)
                        finally:
                            partial.unlink(missing_ok=True)
        else:
            for filename, record in records.items():
                fetch(record['url'], out/filename, record['sha256'])
        (out/'provenance.json').write_text(json.dumps(records, indent=2)+'\n')
        print(f'Ready: {out}', flush=True)

if __name__ == '__main__': main()

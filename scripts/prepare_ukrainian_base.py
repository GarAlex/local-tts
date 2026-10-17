"""Acquire pinned Ukrainian training weights and import a fixed source speaker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(65536):h.update(b)
    return h.hexdigest()

def download(source, name, record):
    target=source/name
    if target.exists() and target.stat().st_size==record['size'] and digest(target)==record['sha256']:
        print('Already verified: '+name,flush=True);return
    print('Downloading '+name,flush=True)
    with tempfile.NamedTemporaryFile(dir=source,delete=False) as f:temporary=Path(f.name)
    try:
        h=hashlib.sha256();size=0
        with urllib.request.urlopen(record['url'],timeout=120) as response,temporary.open('wb') as f:
            while b:=response.read(1024*1024):
                size+=len(b)
                if size>record['size']:raise ValueError('download exceeds pinned size: '+name)
                h.update(b);f.write(b)
        if size!=record['size'] or h.hexdigest()!=record['sha256']:raise ValueError('checksum/size mismatch; existing file preserved: '+name)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speaker',choices=['lada','mykyta','tetiana'],default='lada')
    parser.add_argument('--source',default=str(ROOT/'voices/vits-piper-uk-base/source'))
    parser.add_argument('--output',default=str(ROOT/'voices/vits-piper-uk-base/base.pt'))
    args=parser.parse_args()
    if Path(args.output).exists():raise ValueError('base output exists; choose a new destination')
    source=Path(args.source);source.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'assets/vits-piper-uk-assets.json').read_text())
    for name,record in manifest['files'].items():download(source,name,record)
    subprocess.run([sys.executable,str(ROOT/'training/import_piper_uk.py'),'--source',str(source),'--output',args.output,'--speaker',args.speaker],check=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Acquire pinned Qwen3 VoiceDesign INT4 assets; inference remains native."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, tempfile, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output', type=pathlib.Path, default=ROOT/'voices/qwen-voice-design');args=parser.parse_args()
    pins=json.loads((ROOT/'assets/qwen-design-assets.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
    def acquire(item):
        name, record=item;path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file() and path.stat().st_size==record['bytes'] and digest(path)==record['sha256']: return
        fd,stage=tempfile.mkstemp(prefix='.download-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as out, urllib.request.urlopen(record['url'],timeout=120) as incoming:
                h=hashlib.sha256();size=0
                while block:=incoming.read(1024*1024):
                    size+=len(block)
                    if size>record['bytes']: raise ValueError('unexpected download size: '+name)
                    out.write(block);h.update(block)
            if size!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError('checksum mismatch: '+name)
            os.replace(stage,path);print('Verified '+name,flush=True)
        finally:
            if os.path.exists(stage):os.unlink(stage)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(acquire,pins['files'].items()))
    print('Pinned Qwen VoiceDesign assets ready: '+str(args.output),flush=True)
if __name__=='__main__':main()

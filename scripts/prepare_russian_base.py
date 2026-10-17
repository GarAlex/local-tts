"""Acquire and import the pinned Russian Vosk training generator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from prepare_ukrainian_base import download
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speaker', choices=['F01','F02','F03','M01','M02'], default='F01')
    parser.add_argument('--source', default=str(ROOT/'voices/vits-vosk-ru-base/source'))
    parser.add_argument('--output', default=str(ROOT/'voices/vits-vosk-ru-base/base.pt'))
    parser.add_argument('--dictionary', default=str(ROOT/'voices/vosk-ru/dictionary'))
    args=parser.parse_args()
    if Path(args.output).exists(): raise ValueError('base output exists; choose a new destination')
    dictionary=Path(args.dictionary)
    if not dictionary.is_file():
        raise ValueError('Install the Russian inference package first using scripts/download_cyrillic.py ru, or pass --dictionary')
    source=Path(args.source);source.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'assets/vits-vosk-ru-assets.json').read_text())
    for name,record in manifest['files'].items():
        if record['url'].startswith('embedded:'):
            data=(ROOT/'training/vendor/vosk/STFT-LICENSE.txt').read_bytes()
            if len(data)!=record['size'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('local STFT notice checksum mismatch')
            (source/name).write_bytes(data)
        else: download(source,name,record)
    subprocess.run([sys.executable,str(ROOT/'training/import_vosk_ru.py'),'--source',str(source),
                    '--output',args.output,'--speaker',args.speaker,'--dictionary',str(dictionary)],check=True)

if __name__=='__main__': main()

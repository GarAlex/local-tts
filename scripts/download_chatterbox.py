"""Acquire pinned MIT Chatterbox multilingual ONNX assets (about 1.56 GB)."""
import argparse
import json
from pathlib import Path
from download_english import download

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'voices/chatterbox-multilingual')
    args = parser.parse_args()
    pins = json.loads((ROOT/'assets/chatterbox-assets.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, record in pins['files'].items():
        path = args.output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        download(path.parent, record['url'], path.name, record['sha256'])
    (args.output/'provenance.json').write_text(json.dumps(pins, indent=2)+'\n')
    print(f'Ready: {args.output}', flush=True)

if __name__ == '__main__': main()

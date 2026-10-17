"""Acquire and import a pinned French, Italian, Portuguese, Polish or Dutch base."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from download_english import download

ROOT = Path(__file__).resolve().parents[1]


def main():
    pins = json.loads((ROOT / 'assets/vits-neon-european.json').read_text())['candidates']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=list(pins))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--test-reference', action='store_true', help='also fetch publisher code for opt-in parity tests only')
    args = parser.parse_args()
    output = args.output or ROOT / f'voices/vits-neon-{args.language}-base/base.pt'
    if output.exists():
        raise ValueError('base output exists; choose a new destination')
    record = pins[args.language]
    source = ROOT / 'voices/neon-candidates' / args.language
    source.mkdir(parents=True, exist_ok=True)
    for name, asset in record['files'].items():
        download(source, asset['url'], name, asset['sha256'])
    if args.test_reference:
        asset = record['test_reference']
        download(source, asset['url'], 'model.pt', asset['sha256'])
    subprocess.run([sys.executable, str(ROOT / 'training/import_neon.py'), args.language,
                    '--source', str(source), '--output', str(output)], check=True)


if __name__ == '__main__':
    main()

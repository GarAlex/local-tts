"""Acquire and import a pinned German or Spanish CSS10 fine-tuning base."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f'voices/vits-css10-{args.language}-base/base.pt'
    if output.exists():
        raise ValueError('base output exists; choose a new destination')
    subprocess.run([sys.executable, str(ROOT / 'scripts/download_css10_candidates.py'), args.language], check=True)
    subprocess.run([sys.executable, str(ROOT / 'training/import_css10.py'), args.language, '--output', str(output)], check=True)


if __name__ == '__main__':
    main()

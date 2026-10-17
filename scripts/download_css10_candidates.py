"""Download pinned German/Spanish VITS source assets for project import."""
import argparse
import json
from pathlib import Path

from download_english import download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es', 'all'])
    parser.add_argument('--output', type=Path, help='parent folder; each language gets its own subfolder')
    parser.add_argument('--test-reference', action='store_true', help='also download publisher code package for opt-in conversion tests; not needed for import')
    args = parser.parse_args()
    pins = json.loads((ROOT / 'assets/vits-css10-candidates.json').read_text())
    parent = args.output or ROOT / 'voices/css10-candidates'
    for language in ['de', 'es'] if args.language == 'all' else [args.language]:
        record = pins['candidates'][language]
        out = parent / language
        out.mkdir(parents=True, exist_ok=True)
        for name, asset in record['files'].items():
            download(out, asset['url'], name, asset['sha256'])
        if args.test_reference:
            asset = record['test_reference']
            download(out, asset['url'], 'model.pt', asset['sha256'])
        (out / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
        print(f'Acquired {language} source: {out}. Convert with training/import_css10.py before fine-tuning.', flush=True)


if __name__ == '__main__':
    main()

"""Acquire pinned LJSpeech VITS weights/config/notices for native import."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from download_english import download

ROOT = Path(__file__).resolve().parents[1]
def main():
    out = ROOT / 'voices/vits-ljs-base/source'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'assets/vits-ljs-assets.json').read_text())
    for name, record in manifest['files'].items():
        download(out, record['url'], name, record['sha256'])
    (out / 'provenance.json').write_text(json.dumps(manifest, indent=2))
    print(f'Acquired source: {out}', flush=True)
if __name__ == '__main__': main()

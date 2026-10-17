"""Import one pinned public VITS generator without executing downloaded Python/pickle."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from train import MODEL, SynthesizerTrn
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(65536): result.update(chunk)
    return result.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT / 'voices/vits-ljs-base/source'))
    parser.add_argument('--output', default=str(ROOT / 'voices/vits-ljs-base/base.pt'))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists(): raise ValueError('base output exists; choose a new destination')
    manifest = json.loads((ROOT / 'assets/vits-ljs-assets.json').read_text())
    contents = {}
    for name, record in manifest['files'].items():
        contents[name] = (source / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != record['sha256']: raise ValueError('source checksum mismatch: ' + name)
    config = json.loads(contents['ljs-config.json'])
    cfg = dict(config['model'], use_sdp=True)
    if cfg != dict(MODEL, use_sdp=True) or config['data']['sampling_rate'] != 22050 or not config['data']['add_blank']:
        raise ValueError('unsupported source architecture/token configuration')
    # Parse only literal inventory assignments; never exec the source file.
    assignments = {}
    for node in ast.parse(contents['symbols.py.txt'].decode()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name): assignments[target.id] = ast.literal_eval(node.value)
    symbols = ['\0'] + list(assignments['_punctuation'] + assignments['_letters'] + assignments['_letters_ipa'])
    if len(symbols) != 178: raise ValueError('unexpected upstream inventory')
    torch.set_num_threads(2)
    public = torch.load(io.BytesIO(contents['pretrained_ljs.pth']), map_location='cpu', weights_only=True)
    net = SynthesizerTrn(len(symbols), 513, 32, **cfg)
    net.load_state_dict(public['model'], strict=True)
    # Independently hash-check the dictionary and its actual BSD notice.
    records = json.loads((ROOT / 'assets/english-assets.json').read_text())
    sys.path.insert(0, str(ROOT / 'scripts'))
    from download_english import download
    dictionary = output.parent / 'frontend'
    dictionary.mkdir(parents=True, exist_ok=True)
    for name in ['cmudict.dict', 'CMUDICT-LICENSE.txt']:
        rec = records[name]; download(dictionary, rec['url'], name, rec['sha256'])
    shutil.copyfile(dictionary / 'CMUDICT-LICENSE.txt', dictionary / 'CMU-LICENSE.txt')
    rights = dict(audio='LJSpeech publisher declares public-domain recordings (US); DATASET.html retained',
                  transcripts='LJSpeech publisher declares public-domain texts (US)',
                  speaker_consent='Published LJSpeech single-speaker corpus; no additional consent inferred')
    notices = {name: contents[name].decode() for name in ['MODEL_CARD.md', 'SOURCE-LICENSE.txt', 'WEIGHTS-LICENSE.txt', 'DATASET.html']}
    notices['PROVENANCE.json'] = json.dumps(manifest, indent=2)
    origin = dict(mode='public-import', repository=manifest['repository'], revision=manifest['revision'],
                  weights_sha256=manifest['files']['pretrained_ljs.pth']['sha256'],
                  license=manifest['license'], discriminator='absent; initialized fresh during adaptation',
                  frontend_note='Native CMU IPA adapter differs from upstream english_cleaners2/eSpeak; preserves all upstream indices',
                  license_limit='Apache-2.0 declared by mirror; linked earlier weight mirror has no separate license record',
                  notices=notices)
    training = dict(schema_version=1, language='en', frontend='cmu-ipa-v1',
                    dictionary=dict(path=str((dictionary / 'cmudict.dict').resolve()), sha256=sha(dictionary / 'cmudict.dict')),
                    sample_rate=22050, symbols=symbols, speaker='ljspeech', utterances=[], rights=rights)
    payload = dict(generator=public['model'], discriminator=None, model_config=cfg, model_profile='ljs-sdp-v1',
                   training_config=training, step=public['iteration'], dataset_digest='public-base-no-local-dataset',
                   source_revision='vits:' + manifest['source_revision'], initialization=origin)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.tmp', delete=False) as stream: temp = Path(stream.name)
    try:
        torch.save(payload, temp)
        # Atomic, exclusive install: a concurrently-created checkpoint is never overwritten.
        import os
        os.link(temp, output)
    finally: temp.unlink(missing_ok=True)
    print(json.dumps(dict(event='imported', checkpoint=str(output.resolve()), generator_tensors=len(public['model']),
                          symbols=len(symbols), discriminator='fresh on adaptation', model_profile='ljs-sdp-v1')), flush=True)

if __name__ == '__main__': main()

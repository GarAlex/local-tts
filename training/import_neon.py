"""Convert a pinned European Neon generator to the project fine-tuning contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from neon_adapter import MODEL, PROFILE, FRONTEND, LANGUAGES, convert_state, symbols
from inspect_css10 import ROOT, inspect_candidate, load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=LANGUAGES)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('base output exists; choose a new destination')
    source = args.source or ROOT / 'voices/neon-candidates' / args.language
    torch.set_num_threads(2)
    record = json.loads((ROOT / 'assets/vits-neon-european.json').read_text())['candidates'][args.language]
    report = inspect_candidate(args.language, source, record)
    if symbols(args.language) != ['\0', '\ue001', '\ue002', '\ue003'] + report['frontend']['symbols'][4:]:
        raise ValueError('native inventory differs from source token indices')
    public = load_weights(source / record['weights'], record['files'][record['weights']])
    generator = convert_state(public['model'], args.language)
    notices = {}
    for name, asset in record['files'].items():
        if name == record['weights']:
            continue
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise ValueError('source metadata changed: ' + name)
        notices[name] = data.decode()
    notices['PROVENANCE.json'] = json.dumps(record, indent=2)
    rights = dict(audio=record['dataset_description'],
                  transcripts='Source text provenance unresolved; pinned model configuration retained',
                  speaker_consent='Published source speaker; no additional adaptation consent inferred')
    origin = dict(mode='public-import', repository=record['repository'], revision=record['revision'],
                  weights_sha256=report['weights_sha256'], license=record['declared_license'],
                  license_limit='; '.join(record['provenance_limits']), notices=notices,
                  discriminator='absent; initialized fresh during adaptation',
                  frontend_note='Native Neon blank=3 and original character indices; NFC canonical spelling; explicit removal of inverted punctuation; other unknown characters rejected rather than silently dropped',
                  source_speaker_aliases=report['conditioning']['speakers'], conditioning_speaker=0,
                  source_epoch=public['epoch'], source_global_step=public['step'])
    config = dict(schema_version=1, language=args.language, frontend=FRONTEND, sample_rate=22050,
                  symbols=symbols(args.language), speaker='neon', conditioning_speaker=0, utterances=[], rights=rights)
    payload = dict(generator=generator, discriminator=None, model_config=MODEL, model_profile=PROFILE,
                   training_config=config, step=public['step'], dataset_digest='public-base-no-local-dataset',
                   source_revision=record['repository'] + ':' + record['revision'], initialization=origin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(event='imported', language=args.language, checkpoint=str(args.output.resolve()),
                          generator_tensors=len(generator), model_profile=PROFILE, commercial_clearance_established=False)), flush=True)


if __name__ == '__main__':
    main()

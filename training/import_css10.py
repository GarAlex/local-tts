"""Convert a pinned DE/ES CSS10 generator to the project fine-tuning contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from css10_adapter import MODEL, PROFILE, FRONTEND, convert_state, symbols
from inspect_css10 import ROOT, inspect_candidate, load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('base output exists; choose a new destination')
    source = args.source or ROOT / 'voices/css10-candidates' / args.language
    torch.set_num_threads(2)
    report = inspect_candidate(args.language, source)
    if symbols() != ['\0', '\ue001', '\ue002', '\ue003'] + report['frontend']['symbols'][4:]:
        raise ValueError('native inventory differs from source token indices')
    record = json.loads((ROOT / 'assets/vits-css10-candidates.json').read_text())['candidates'][args.language]
    public = load_weights(source / record['weights'], record['files'][record['weights']])
    generator = convert_state(public['model'])
    notices = {}
    for name, asset in record['files'].items():
        if name == record['weights']:
            continue
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset['sha256']:
            raise ValueError('source metadata changed: ' + name)
        notices[name] = data.decode()
    notices['PROVENANCE.json'] = json.dumps(record, indent=2)
    rights = dict(audio='CSS10 LibriVox recordings; source titles/readers retained; territorial rights not independently verified',
                  transcripts='CSS10 aligned audiobook text; source records retained',
                  speaker_consent='Published source speaker; no additional adaptation consent inferred')
    origin = dict(mode='public-import', repository=record['repository'], revision=record['revision'],
                  weights_sha256=report['weights_sha256'], license=record['declared_license'],
                  license_limit='; '.join(record['provenance_limits']), notices=notices,
                  discriminator='absent; initialized fresh during adaptation',
                  frontend_note='Native CSS10 blank=3 and original character indices; NFC canonical spelling; explicit removal of inverted punctuation; other unknown characters rejected rather than silently dropped',
                  source_speaker_aliases=report['conditioning']['speakers'], conditioning_speaker=0,
                  source_epoch=public['epoch'], source_global_step=public['step'])
    config = dict(schema_version=1, language=args.language, frontend=FRONTEND, sample_rate=22050,
                  symbols=symbols(), speaker='css10', conditioning_speaker=0, utterances=[], rights=rights)
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

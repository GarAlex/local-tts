"""Cache trained phoneme teacher alignments for training clips only."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from checkpoint_tools import load, build
from train import Acoustics, load_batch, conditioning_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('output already exists')
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    saved, digest = load(args.teacher)
    if saved.get('model_profile') != 'ljs-sdp-v1': raise ValueError('requires trained LJS phoneme teacher')
    raw = Path(args.prepared).read_bytes()
    config = json.loads(raw)
    source = saved['training_config']
    for field in ('language', 'frontend', 'sample_rate', 'symbols'):
        if config[field] != source[field]: raise ValueError(f'teacher {field} differs')
    if config['frontend'] != 'cmu-ipa-v1': raise ValueError('requires phoneme data')
    net = build(saved).eval()
    acoustics = Acoustics('cpu')
    records = []
    with torch.no_grad():
        for record in config['utterances']:
            if record['split'] != 'train': continue
            if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('teacher audio changed')
            x, xl, spec, sl, _ = load_batch([record], 'cpu', acoustics)
            _, _, attn, _, _, _, _ = net(x, xl, spec, sl, sid=conditioning_ids(config, x))
            durations = attn[0, 0].sum(0).long().tolist()
            if len(durations) != len(record['tokens']) or sum(durations) != int(sl[0]) or min(durations) < 1:
                raise ValueError('invalid teacher alignment')
            records.append(dict(audio=record['audio'], audio_sha256=record['sha256'],
                                tokens_sha256=hashlib.sha256(json.dumps(record['tokens']).encode()).hexdigest(),
                                durations=durations))
            print(f'aligned {len(records)} training clips', flush=True)
    report = dict(schema_version=1, prepared_sha256=hashlib.sha256(raw).hexdigest(),
                  teacher_sha256=digest, teacher_profile=saved['model_profile'],
                  teacher_initialization=saved.get('initialization'), records=records,
                  note='Trained teacher MAS on target audio; automatic labels are not manually verified. Validation/test clips are excluded.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(report, stream, indent=2)


if __name__ == '__main__': main()

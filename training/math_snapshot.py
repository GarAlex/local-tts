"""Preserve arithmetic generator weights for evaluation, excluding training state."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import torch
from checkpoint_tools import load, build

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists(): parser.error('output already exists')
saved, source_hash = load(args.checkpoint)
if saved.get('model_profile') not in ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1'):
    raise ValueError('only the arithmetic student profile is supported')
torch.set_num_threads(2)
build(saved)
fields = ['generator', 'training_config', 'model_config', 'model_profile',
          'initialization', 'dataset_digest', 'step', 'source_revision']
payload = {field: saved[field] for field in fields}
if 'math_prior_mel_weight' in saved:
    payload['math_prior_mel_weight'] = saved['math_prior_mel_weight']
if saved.get('teacher_alignment'):
    payload['teacher_alignment'] = saved['teacher_alignment']
for field in ('training_objective', 'acoustic_fit_steps', 'prior_fit_steps', 'preferred_latent_noise_scale'):
    if field in saved:
        payload[field] = saved[field]
payload['evaluation_snapshot'] = dict(source_sha256=source_hash,
                                      source=str(Path(args.checkpoint).resolve()),
                                      resumable=False)
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.math-snapshot-', delete=False) as stream:
    temporary = Path(stream.name)
try:
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
    # Atomic creation; refuses a racing overwrite rather than replacing a snapshot.
    os.link(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps(dict(step=saved['step'], output=str(output), bytes=output.stat().st_size,
                      source_sha256=source_hash, resumable=False)))

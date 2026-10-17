"""Validated training-only teacher duration labels for arithmetic distillation."""
import hashlib
import json
from pathlib import Path
import torch
import commons


def load_alignment(path, prepared_raw, config):
    raw = Path(path).read_bytes()
    saved = json.loads(raw)
    if saved.get('schema_version') != 1 or saved.get('teacher_profile') != 'ljs-sdp-v1':
        raise ValueError('unsupported teacher alignment schema/profile')
    if saved.get('prepared_sha256') != hashlib.sha256(prepared_raw).hexdigest():
        raise ValueError('teacher alignment prepared dataset differs')
    expected = {r['audio']: r for r in config['utterances'] if r['split'] == 'train'}
    labels = {}
    for row in saved['records']:
        audio = row['audio']
        if audio not in expected or audio in labels:
            raise ValueError('teacher alignment contains nontraining/duplicate recording')
        target = expected[audio]
        if row['audio_sha256'] != target['sha256'] or row['tokens_sha256'] != hashlib.sha256(json.dumps(target['tokens']).encode()).hexdigest():
            raise ValueError('teacher alignment recording/tokens differ')
        durations = row['durations']
        if len(durations) != len(target['tokens']) or not durations or any(type(d) is not int or d < 1 for d in durations):
            raise ValueError('invalid teacher durations')
        labels[audio] = durations
    if labels.keys() != expected.keys(): raise ValueError('teacher alignment missing training recordings')
    return labels, dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                        teacher_sha256=saved['teacher_sha256'])


def alignment_batch(records, labels, x_lengths, y_lengths):
    device = x_lengths.device
    duration = torch.zeros(len(records), 1, int(x_lengths.max()), device=device)
    for i, record in enumerate(records):
        values = labels[record['audio']]
        if len(values) != int(x_lengths[i]) or sum(values) != int(y_lengths[i]):
            raise ValueError('teacher duration count/frame sum differs from batch')
        duration[i, 0, :len(values)] = torch.tensor(values, device=device)
    x_mask = commons.sequence_mask(x_lengths).float()
    y_mask = commons.sequence_mask(y_lengths).float()
    mask = y_mask[:, None, :, None] * x_mask[:, None, None, :]
    return commons.generate_path(duration, mask).detach()

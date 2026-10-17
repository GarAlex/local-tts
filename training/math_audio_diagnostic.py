"""Separate arithmetic student decoder reconstruction from text-only inference.

Writes raw WAVs for listening. Numerical measurements are not quality gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from checkpoint_tools import load, build
from train import Acoustics, load_batch, conditioning_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('output already exists')
    torch.set_num_threads(2)
    saved, digest = load(args.checkpoint)
    if saved.get('model_profile') not in ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1'):
        raise ValueError('requires arithmetic student checkpoint')
    raw = Path(args.prepared).read_bytes()
    if hashlib.sha256(raw).hexdigest() != saved['dataset_digest']:
        raise ValueError('prepared dataset differs from checkpoint')
    config = json.loads(raw)
    record = next(r for r in config['utterances'] if r['split'] == 'validation')
    if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('validation audio changed')
    net = build(saved).eval()
    x, xl, spec, sl, audio = load_batch([record], 'cpu', Acoustics('cpu'))
    sid = conditioning_ids(config, x)
    waves = {'teacher': audio[0, 0].numpy()}
    with torch.no_grad():
        torch.manual_seed(1234)
        g = None if sid is None else net.emb_g(sid).unsqueeze(-1)
        z, _, _, mask = net.enc_q(spec, sl, g=g)
        waves['posterior-reconstruction'] = net.dec(z * mask, g=g)[0, 0].numpy()
        torch.manual_seed(1234)
        _, _, attn, _, _, y_mask, latent = net(x, xl, spec, sl, sid=sid)
        _, _, m_p, _, _, _ = latent
        prior_z = net.flow(m_p, y_mask, g=g, reverse=True)
        waves['aligned-text-prior'] = net.dec(prior_z * y_mask, g=g)[0, 0].numpy()
        durations = attn[0, 0].sum(0).numpy()
        for scale in (0.0, 0.333, 0.667, 1.0):
            torch.manual_seed(1234)
            waves[f'text-noise-{scale}'] = net.infer(
                x, xl, sid=sid, noise_scale=scale, noise_scale_w=0.0)[0][0, 0].numpy()
    if any(not np.isfinite(w).all() for w in waves.values()):
        raise ValueError('non-finite audio')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    acoustics = Acoustics('cpu')
    teacher_mel = acoustics.to_mel(acoustics.spec(torch.from_numpy(waves['teacher'])[None]))
    for name, wave in waves.items():
        wavfile.write(output / f'{name}.wav', 22050,
                      np.clip(wave * 32767, -32768, 32767).astype(np.int16))
        row = dict(name=name, seconds=len(wave) / 22050,
                   rms=float(np.sqrt(np.mean(wave ** 2))),
                   peak=float(np.max(np.abs(wave))))
        if name in ('posterior-reconstruction', 'aligned-text-prior'):
            predicted_mel = acoustics.to_mel(acoustics.spec(torch.from_numpy(wave)[None]))
            frames = min(teacher_mel.shape[-1], predicted_mel.shape[-1])
            row['aligned_mel_l1'] = float(torch.nn.functional.l1_loss(
                predicted_mel[:, :, :frames], teacher_mel[:, :, :frames]))
        rows.append(row)
    report = dict(checkpoint_sha256=digest, step=saved['step'],
                  teacher_audio=record['audio'], recordings=rows,
                  alignment=dict(token_frames=durations.tolist(),
                                 max_token_fraction=float(durations.max() / durations.sum())),
                  note='Posterior uses ground-truth audio; text-only samples do not. Listen before judging quality. Noise variation is diagnostic, not a production fix.')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Overfit one training clip to isolate decoder capacity from text/alignment.

This is reconstruction from cached ground-truth latents, never a TTS candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
import torch
from torch.nn import functional as F

from checkpoint_tools import load, build
from train import Acoustics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists(): parser.error('output already exists')
    if not 1 <= args.steps <= 10000: parser.error('steps must be 1..10000')
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    saved, checkpoint_hash = load(args.checkpoint)
    if saved.get('model_profile') not in ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1'):
        raise ValueError('requires compact arithmetic student')
    dataset = Path(args.dataset).resolve()
    data = json.loads(dataset.read_text())
    record = next(r for r in data['utterances'] if r['split'] == 'train')
    audio_path = (dataset.parent / record['audio']).resolve()
    if not audio_path.is_relative_to(dataset.parent): raise ValueError('escaping audio path')
    rate, raw = wavfile.read(audio_path)
    if rate != 22050 or raw.ndim != 1 or raw.dtype != np.int16:
        raise ValueError('requires mono 22050 Hz PCM16')
    audio = torch.from_numpy(raw.astype(np.float32) / 32768)[None, None]
    if not 8192 <= audio.shape[-1] <= 22050 * 20: raise ValueError('clip length invalid')
    net = build(saved).eval()
    if net.n_speakers: raise ValueError('probe requires unconditioned student')
    acoustics = Acoustics('cpu')
    with torch.no_grad():
        spec = acoustics.spec(audio[:, 0])
        latent, _, _, mask = net.enc_q(spec, torch.tensor([spec.shape[-1]]))
        latent = (latent * mask).detach()
        before = net.dec(latent)[0, 0].numpy()
    for parameter in net.parameters(): parameter.requires_grad_(False)
    for parameter in net.dec.parameters(): parameter.requires_grad_(True)
    optim = torch.optim.AdamW(net.dec.parameters(), lr=0.0002, betas=(0.8, 0.99))
    root.mkdir(parents=True, exist_ok=False)
    history = []
    for step in range(1, args.steps + 1):
        frame = int(torch.randint(latent.shape[-1] - 31, (1,)))
        predicted = net.dec(latent[:, :, frame:frame + 32])[:, 0]
        target = audio[:, 0, frame * 256:frame * 256 + 8192]
        mel = F.l1_loss(acoustics.to_mel(acoustics.spec(predicted)),
                        acoustics.to_mel(acoustics.spec(target)))
        spectral = 0
        for size, hop in [(512, 128), (1024, 256), (2048, 512)]:
            window = torch.hann_window(size)
            p = torch.stft(predicted, size, hop_length=hop, window=window, return_complex=True).abs().clamp_min(1e-5)
            t = torch.stft(target, size, hop_length=hop, window=window, return_complex=True).abs().clamp_min(1e-5)
            spectral = spectral + (p-t).norm() / t.norm().clamp_min(1e-5) + F.l1_loss(p.log(), t.log())
        loss = 45 * mel + 5 * spectral / 3
        if not torch.isfinite(loss): raise ValueError('non-finite probe loss')
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.dec.parameters(), 1000)
        optim.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = dict(step=step, mel=float(mel.detach()), spectral=float(spectral.detach()) / 3)
            history.append(row); print(json.dumps(row), flush=True)
    net.eval()
    with torch.no_grad(): after = net.dec(latent)[0, 0].numpy()
    for name, samples in [('teacher', audio[0, 0].numpy()), ('before', before), ('after', after)]:
        if not np.isfinite(samples).all(): raise ValueError('invalid probe audio')
        wavfile.write(root / f'{name}.wav', 22050, np.clip(samples * 32767, -32768, 32767).astype(np.int16))
    torch.save(net.dec.state_dict(), root / 'decoder-only.pt')
    report = dict(checkpoint_sha256=checkpoint_hash, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  teacher_audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                  transcript=record['text'], split=record['split'], history=history,
                  decoder_parameters=sum(p.numel() for p in net.dec.parameters()),
                  note='Overfits one training clip using cached ground-truth posterior latents; does not test text inference or held-out generalization.')
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print('Completed diagnostic reconstruction; not a TTS voice', flush=True)


if __name__ == '__main__': main()

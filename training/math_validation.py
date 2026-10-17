"""Posterior/aligned-prior and text-only diagnostics, not pronunciation ratings."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from checkpoint_tools import load, build
from train import Acoustics, load_batch, conditioning_ids, math_prior_mel_loss
import commons
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--prepared', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists(): parser.error('output already exists')
torch.set_num_threads(2)
saved, digest = load(args.checkpoint)
if saved.get('model_profile') not in ('math-student-v1', 'math-phoneme-v1', 'math-phoneme-wide-v1'): raise ValueError('requires math student')
raw = Path(args.prepared).read_bytes()
if hashlib.sha256(raw).hexdigest() != saved['dataset_digest']:
    raise ValueError('prepared dataset differs from this checkpoint')
config = json.loads(raw)
records = [r for r in config['utterances'] if r['split'] == 'validation']
if not records: raise ValueError('no validation clips')
net = build(saved).eval()
acoustics = Acoustics('cpu')
rows = []
with torch.no_grad():
    for record in records:
        if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('validation audio changed')
        torch.manual_seed(1234)  # Same random segment/noise for repeated evaluations.
        x, xl, spec, sl, audio = load_batch([record], 'cpu', acoustics)
        generated, duration_loss, attn, slices, _, mask, latent = net(x, xl, spec, sl, sid=conditioning_ids(config,x))
        target = commons.slice_segments(acoustics.to_mel(spec), slices, 32)
        predicted = acoustics.to_mel(acoustics.spec(generated.squeeze(1)))
        mel = float(F.l1_loss(target,predicted))
        duration = float(duration_loss.sum())
        prior_mel = float(math_prior_mel_loss(net, latent[2], mask, slices, target, acoustics))
        torch.manual_seed(1234)
        text_audio = net.infer(x, xl, sid=conditioning_ids(config,x), noise_scale=0.0, noise_scale_w=0.0)[0]
        if not torch.isfinite(text_audio).all(): raise ValueError('non-finite text audio')
        rms = float(text_audio.square().mean().sqrt())
        teacher_rms = float(audio.square().mean().sqrt())
        token_frames = attn[0, 0].sum(0)
        if not torch.isfinite(torch.tensor([mel,duration,prior_mel,rms])).all(): raise ValueError('non-finite validation diagnostics')
        rows.append(dict(audio=record['audio'], audio_sha256=record['sha256'], mel_l1=mel, duration_loss=duration,
                         aligned_prior_mel_l1=prior_mel,
                         text_duration_ratio=text_audio.shape[-1] / audio.shape[-1],
                         text_rms=rms, teacher_rms=teacher_rms,
                         alignment_max_token_fraction=float(token_frames.max() / token_frames.sum()),
                         alignment_one_frame_fraction=float((token_frames == 1).float().mean())))
report = dict(checkpoint_sha256=digest, step=saved['step'], clips=rows,
              mean_mel_l1=sum(r['mel_l1'] for r in rows)/len(rows),
              mean_duration_loss=sum(r['duration_loss'] for r in rows)/len(rows),
              mean_aligned_prior_mel_l1=sum(r['aligned_prior_mel_l1'] for r in rows)/len(rows),
              mean_text_duration_ratio=sum(r['text_duration_ratio'] for r in rows)/len(rows),
              mean_text_rms=sum(r['text_rms'] for r in rows)/len(rows),
              note='Posterior reconstruction and aligned-prior loss use ground-truth audio/alignment. Text duration/RMS use text-only inference. None measures intelligibility or naturalness.')
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x') as stream: json.dump(report,stream,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='clips'}))

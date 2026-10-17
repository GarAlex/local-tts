"""Fit arithmetic decoder reconstruction before learning the text prior.

Ground-truth posterior means are cached; no validation audio is optimized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from scipy.io import wavfile
import numpy as np
from checkpoint_tools import load, build
from train import Acoustics, load_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', choices=['cpu','mps'], default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('output already exists')
    if not 1 <= args.steps <= 20000: parser.error('steps must be 1..20000')
    torch.set_num_threads(2); torch.manual_seed(1234); random.seed(1234)
    saved, digest = load(args.checkpoint)
    if saved.get('model_profile') != 'math-phoneme-wide-v1': raise ValueError('requires wide arithmetic phoneme profile')
    raw = Path(args.prepared).read_bytes()
    if hashlib.sha256(raw).hexdigest() != saved['dataset_digest']: raise ValueError('prepared dataset differs')
    config = json.loads(raw)
    net = build(saved).eval()
    acoustics = Acoustics('cpu')
    cached = []
    with torch.no_grad():
        for record in config['utterances']:
            if record['split'] not in ('train','validation'): continue
            if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('audio changed')
            _, _, spec, sl, audio = load_batch([record], 'cpu', acoustics)
            _, mean, _, mask = net.enc_q(spec, sl)
            cached.append((record, (mean * mask).detach(), audio))
    training = [r for r in cached if r[0]['split'] == 'train']
    validation = [r for r in cached if r[0]['split'] == 'validation']
    if not training or not validation: raise ValueError('training/validation required')
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if args.device == 'mps':
        if not torch.backends.mps.is_available(): raise ValueError('MPS unavailable')
        torch.mps.manual_seed(1234)
    net.to(device)
    for parameter in net.parameters(): parameter.requires_grad_(False)
    for parameter in net.dec.parameters(): parameter.requires_grad_(True)
    training = [(r,z.to(device),a.to(device)) for r,z,a in training]
    acoustics = Acoustics(device)
    windows = [(size,hop,torch.hann_window(size,device=device)) for size,hop in [(512,128),(1024,256),(2048,512)]]
    optim = torch.optim.AdamW(net.dec.parameters(),lr=0.0002,betas=(0.8,0.99))
    history = []
    for step in range(1,args.steps+1):
        _, latent, audio = random.choice(training)
        frame = random.randrange(latent.shape[-1]-31)
        predicted = net.dec(latent[:,:,frame:frame+32])[:,0]
        target = audio[:,0,frame*256:frame*256+8192]
        mel = F.l1_loss(acoustics.to_mel(acoustics.spec(predicted)),acoustics.to_mel(acoustics.spec(target)))
        spectral = 0
        for size,hop,window in windows:
            p = torch.stft(predicted,size,hop_length=hop,window=window,return_complex=True).abs().clamp_min(1e-5)
            t = torch.stft(target,size,hop_length=hop,window=window,return_complex=True).abs().clamp_min(1e-5)
            spectral = spectral + (p-t).norm()/t.norm().clamp_min(1e-5)+F.l1_loss(p.log(),t.log())
        loss = 45*mel+5*spectral/3
        if not torch.isfinite(loss): raise ValueError('non-finite acoustic loss')
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.dec.parameters(),1000);optim.step()
        if step==1 or step%100==0 or step==args.steps:
            row=dict(step=step,mel=float(mel.detach()),spectral=float(spectral.detach())/3)
            history.append(row);print(json.dumps(row),flush=True)
    net.eval()
    with torch.no_grad():
        for i,(record,latent,audio) in enumerate(validation):
            predicted=net.dec(latent.to(device))[0,0].cpu().numpy()
            if not np.isfinite(predicted).all(): raise ValueError('invalid acoustic validation audio')
            wavfile.write(output/f'{i:02}-reconstruction.wav',22050,np.clip(predicted*32767,-32768,32767).astype(np.int16))
    saved.update(generator={k:v.cpu() for k,v in net.state_dict().items()},
                 initialization=dict(mode='acoustic-reconstruction-fit',base_sha256=digest,
                                     base_initialization=saved.get('initialization'),
                                     steps=args.steps,posterior='mean',trained_split='train'),
                 evaluation_snapshot=dict(source_sha256=digest,resumable=False),
                 training_objective='fixed-posterior-mean multiresolution acoustic fit',
                 acoustic_fit_steps=saved.get('acoustic_fit_steps',0)+args.steps)
    for key in ('optimizer','optimizer_d','rng','python_rng','mps_rng','cuda_rng'): saved.pop(key,None)
    torch.save(saved,output/'acoustic.pt')
    (output/'report.json').write_text(json.dumps(dict(base_sha256=digest,steps=args.steps,
        device=args.device,history=history,validation=[dict(audio=r['audio'],sha256=r['sha256']) for r,_,_ in validation],
        note='Ground-truth posterior reconstruction, not text-only TTS or test-split evaluation.'),indent=2))
    print('Completed acoustic fit; text inference must be trained separately',flush=True)


if __name__=='__main__': main()

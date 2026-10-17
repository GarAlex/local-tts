"""Fit phoneme means/durations to a frozen, reconstructed arithmetic voice.

Deterministic posterior-mean decoding is intentional for this staged model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import torch
from torch.nn import functional as F
from scipy.io import wavfile
import numpy as np
from checkpoint_tools import load, build
from train import Acoustics, load_batch
from math_alignment import load_alignment, alignment_batch


def prior_loss(net, x, xl, path, mask, target_latent, durations):
    hidden, means, _, x_mask = net.enc_p(x, xl)
    log_duration = net.dp(hidden, x_mask)
    duration_loss = ((log_duration - durations.clamp_min(1).log()).square() * x_mask).sum() / x_mask.sum()
    expanded = torch.matmul(path[:,0], means.transpose(1,2)).transpose(1,2)
    latent_loss = ((expanded-target_latent).square()*mask).sum()/mask.sum()/expanded.shape[1]
    posterior_mean = net.flow(expanded, mask, reverse=True)*mask
    return posterior_mean, latent_loss, duration_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--alignment',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():parser.error('output already exists')
    if not 1<=args.steps<=20000:parser.error('steps must be 1..20000')
    torch.set_num_threads(2);torch.manual_seed(1234);random.seed(1234)
    saved,digest=load(args.checkpoint)
    if saved.get('model_profile')!='math-phoneme-wide-v1' or saved.get('training_objective') not in (
            'fixed-posterior-mean multiresolution acoustic fit','fixed-acoustic-mean-prior-fit'):
        raise ValueError('requires acoustic-fit or prior-fit wide phoneme student')
    raw=Path(args.prepared).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=saved['dataset_digest']:raise ValueError('prepared dataset differs')
    config=json.loads(raw)
    labels,alignment=load_alignment(args.alignment,raw,config)
    if alignment['sha256']!=(saved.get('teacher_alignment') or {}).get('sha256'):
        raise ValueError('alignment differs from acoustic source')
    net=build(saved).eval()
    acoustics=Acoustics('cpu')
    cached=[]
    with torch.no_grad():
        for record in config['utterances']:
            if record['split'] not in ('train','validation'):continue
            if hashlib.sha256(Path(record['audio']).read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('audio changed')
            if record['split']!='train':continue
            x,xl,spec,sl,audio=load_batch([record],'cpu',acoustics)
            path=alignment_batch([record],labels,xl,sl)
            _,mean,_,mask=net.enc_q(spec,sl)
            latent=net.flow(mean*mask,mask)
            durations=path.sum(2)
            cached.append((x,xl,path,mask,latent,durations,audio))
    if not cached:raise ValueError('no training records')
    device=torch.device(args.device)
    if device.type=='mps':
        if not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
        torch.mps.manual_seed(1234)
    net.to(device)
    for p in net.parameters():p.requires_grad_(False)
    parameters=list(net.enc_p.parameters())+list(net.dp.parameters())
    for p in parameters:p.requires_grad_(True)
    cached=[tuple(t.to(device) for t in row) for row in cached]
    acoustics=Acoustics(device)
    optim=torch.optim.AdamW(parameters,lr=0.0002,betas=(0.8,0.99))
    output.mkdir(parents=True,exist_ok=False)
    history=[];stopping=False;step=0
    def stop(_signal,_frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while step<args.steps and not stopping:
        x,xl,path,mask,latent,durations,audio=random.choice(cached)
        mean,latent_loss,duration_loss=prior_loss(net,x,xl,path,mask,latent,durations)
        frame=random.randrange(mean.shape[-1]-31)
        predicted=net.dec(mean[:,:,frame:frame+32])[:,0]
        target=audio[:,0,frame*256:frame*256+8192]
        mel=F.l1_loss(acoustics.to_mel(acoustics.spec(predicted)),acoustics.to_mel(acoustics.spec(target)))
        loss=20*latent_loss+45*mel+duration_loss
        if not torch.isfinite(loss):raise ValueError('non-finite prior loss')
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1000);optim.step();step+=1
        if step==1 or step%100==0 or step==args.steps:
            row=dict(step=step,mel=float(mel.detach()),latent=float(latent_loss.detach()),duration=float(duration_loss.detach()))
            history.append(row);print(json.dumps(row),flush=True)
    validation=[r for r in config['utterances'] if r['split']=='validation']
    net.eval()
    with torch.no_grad():
        for i,record in enumerate(validation):
            x=torch.tensor([record['tokens']],device=device)
            audio=net.infer(x,torch.tensor([x.shape[1]],device=device),noise_scale=0,noise_scale_w=0)[0][0,0].cpu().numpy()
            if not np.isfinite(audio).all():raise ValueError('invalid text validation audio')
            wavfile.write(output/f'{i:02}-text.wav',22050,np.clip(audio*32767,-32768,32767).astype(np.int16))
    saved.update(generator={k:v.cpu() for k,v in net.state_dict().items()},
        initialization=dict(mode='fixed-acoustic-prior-fit',base_sha256=digest,base_initialization=saved.get('initialization'),steps=step),
        evaluation_snapshot=dict(source_sha256=digest,resumable=False),
        training_objective='fixed-acoustic-mean-prior-fit',preferred_latent_noise_scale=0.0,
        prior_fit_steps=saved.get('prior_fit_steps',0)+step)
    for key in ('optimizer','optimizer_d','rng','python_rng','mps_rng','cuda_rng'):saved.pop(key,None)
    torch.save(saved,output/'prior.pt')
    (output/'report.json').write_text(json.dumps(dict(base_sha256=digest,steps=step,device=args.device,history=history,
        validation=[dict(audio=r['audio'],sha256=r['sha256']) for r in validation],
        note='Text-only validation synthesis. Acoustic stack frozen; prior means and durations fitted on training clips. Human listening is required.'),indent=2))
    print('Stopped' if stopping else 'Completed frozen-acoustic prior fit',flush=True)


if __name__=='__main__':main()

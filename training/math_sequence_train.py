"""Train full-utterance teacher-latent/duration students. Not a voice exporter."""
import math
import argparse
import hashlib
import json
from pathlib import Path
import random
import unicodedata
import torch
from checkpoint_tools import load, build
from math_sequence_student import SequenceStudent, supervised_loss


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def spoken_hash(text):return hashlib.sha256(" ".join(unicodedata.normalize("NFC",text).split()).encode()).hexdigest()
def save(payload,path):
    temp=path.with_suffix('.partial');torch.save(payload,temp);temp.replace(path)


def prepare(args):
    if args.output.exists():raise ValueError('output already exists')
    saved,sha=load(args.checkpoint);config=saved['training_config'];language=config['language']
    if (language,saved['model_profile']) not in [('ru','vosk-ru-multi-v1'),('uk','piper-uk-medium-v1')]:raise ValueError('unsupported teacher profile')
    corpus=json.loads(args.corpus.read_text())
    if corpus.get('schema_version')!=1 or corpus['language']!=language:raise ValueError('corpus contract differs')
    records=corpus['records'];seen=set();splits={}
    for record in records:
        if record['spoken'] in seen or record['split'] not in ('train','validation','test'):raise ValueError('duplicate spoken targets or invalid split')
        seen.add(record['spoken']);splits[record['split']]=splits.get(record['split'],0)+1
        ids=record['tokens']
        if not ids or len(ids)>1024 or any(type(t)!=int or not 0<=t<len(config['symbols']) for t in ids):raise ValueError('invalid native teacher tokens')
    if any(splits.get(s,0)==0 for s in ('train','validation','test')):raise ValueError('requires all three splits')
    net=build(saved).eval();net.requires_grad_(False);russian=language=='ru'
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.no_grad():
        for i,record in enumerate(records):
            ids=torch.tensor([record['tokens']]);sid=torch.tensor([config['conditioning_speaker']])
            result=net.infer(ids,torch.tensor([ids.shape[1]]),sid=sid,noise_scale=0,noise_scale_w=0)
            attention=result[2] if russian else result[1];mask=result[3] if russian else result[2]
            latent=((result[4] if russian else result[3])[0]*mask)[0].cpu().contiguous()
            raw=attention[0,0].sum(0);duration=raw.round().long().cpu()
            if not torch.allclose(raw,duration.float()) or duration.sum()!=latent.shape[1] or not torch.isfinite(latent).all() or not 0<latent.shape[1]<=7740:raise ValueError('invalid teacher alignment')
            file=f'{i:05}.pt';save(dict(tokens=ids[0],durations=duration,latent=latent),args.output/file)
            rows.append(dict(record,file=file,sha256=digest(args.output/file),frames=latent.shape[1]))
            if (i+1)%32==0:print(json.dumps(dict(event='prepared',utterances=i+1)),flush=True)
    manifest=dict(schema_version=1,language=language,vocabulary=len(config['symbols']),channels=192,
                  teacher_checkpoint_sha256=sha,corpus_sha256=digest(args.corpus),source_origin=saved['initialization'],rights=config['rights'],records=rows,splits=splits,
                  design='full-utterance native teacher decoder-input latents; no word/phrase stitching',teacher_noise_scale=0)
    (args.output/'dataset.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(event='ready',splits=splits)),flush=True)


def frozen_waveform_loss(predicted,expected):
    losses=[]
    for fft in (256,512,1024):
        window=torch.hann_window(fft,device=predicted.device,dtype=predicted.dtype)
        def spectrum(value):return torch.stft(value.reshape(-1),fft,fft//4,window=window,return_complex=True).abs().clamp_min(1e-5)
        a,b=spectrum(predicted),spectrum(expected)
        losses.append((a.log()-b.log()).abs().mean()+0.5*torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-5))
    return torch.stack(losses).mean()


def fit(args):
    if args.output.exists():raise ValueError('output already exists')
    initialize=getattr(args,'initialize',None)
    if args.resume and initialize:raise ValueError('resume and initialize are mutually exclusive')
    if not 1<=args.steps<=100000 or not 1<=args.batch<=8:raise ValueError('invalid training steps/batch')
    path=args.dataset/'dataset.json'
    if path.stat().st_size>10_000_000:raise ValueError('dataset manifest too large')
    manifest=json.loads(path.read_text());data=[];seen=set()
    if manifest.get('schema_version')!=1 or manifest.get('channels')!=192 or manifest.get('language') not in ('ru','uk') or type(manifest.get('vocabulary'))!=int or not 4<=manifest['vocabulary']<=1024:raise ValueError('invalid sequence dataset contract')
    for row in manifest['records']:
        if row['split'] not in ('train','validation','test') or row['spoken'] in seen:raise ValueError('duplicate spoken targets or invalid split')
        seen.add(row['spoken'])
        file=(args.dataset/row['file']).resolve()
        if not file.is_relative_to(args.dataset.resolve()) or digest(file)!=row['sha256']:raise ValueError('teacher tensor checksum differs')
        if file.stat().st_size>100_000_000:raise ValueError('teacher tensor too large')
        value=torch.load(file,map_location='cpu',weights_only=True)
        tokens=value['tokens'];durations=value['durations'];latent=value['latent']
        if tokens.dtype!=torch.long or durations.dtype!=torch.long or tokens.ndim!=1 or durations.shape!=tokens.shape or not 0<len(tokens)<=1024 or (tokens<0).any() or (tokens>=manifest['vocabulary']).any() or (durations<0).any() or not 0<int(durations.sum())<=7740 or latent.shape!=(192,int(durations.sum())) or not torch.isfinite(latent).all():raise ValueError('invalid cached alignment')
        data.append((row,value))
    training=[v for r,v in data if r['split']=='train'];validation=[v for r,v in data if r['split']=='validation']
    if not training or not validation:raise ValueError('missing training/validation split')
    torch.manual_seed(1234);model=SequenceStudent(manifest['vocabulary']);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    # Channel normalization uses training frames only, never validation/test targets.
    combined=torch.cat([v['latent'] for v in training],dim=1)
    mean=combined.mean(1,keepdim=True);scale=combined.std(1,keepdim=True,correction=0).clamp_min(1e-3);del combined
    start=0;dataset_sha=digest(path);initialization=None
    trained_spoken={spoken_hash(r['spoken']) for r,_ in data if r['split']=='train'}
    if args.resume:
        previous=torch.load(args.resume,map_location='cpu',weights_only=True)
        if previous['dataset_sha256']!=dataset_sha:raise ValueError('resume dataset differs')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);start=previous['step']
        mean=previous['mean'];scale=previous['scale']
        initialization=previous.get('initialization')
        trained_spoken.update(previous.get('training_spoken_sha256',[]))
    if initialize:
        initial_dataset=getattr(args,'initial_dataset',None)
        if not initial_dataset:raise ValueError('initialization requires original dataset for split audit')
        previous=torch.load(initialize,map_location='cpu',weights_only=True)
        original_path=initial_dataset/'dataset.json'
        if digest(original_path)!=previous['dataset_sha256']:raise ValueError('initial dataset checksum differs')
        original=json.loads(original_path.read_text())
        if previous['teacher_checkpoint_sha256']!=manifest['teacher_checkpoint_sha256'] or original['language']!=manifest['language'] or previous['vocabulary']!=manifest['vocabulary']:raise ValueError('initial student teacher/language/vocabulary differs')
        previous_train={spoken_hash(r['spoken']) for r in original['records'] if r['split']=='train'}
        previous_train.update(previous.get('training_spoken_sha256',[]))
        heldout={spoken_hash(r['spoken']) for r,_ in data if r['split'] in ('validation','test')}
        if previous_train & heldout:raise ValueError('prior training overlaps new validation/test utterances')
        model.load_state_dict(previous['model'])
        mean=previous['mean'];scale=previous['scale']
        if mean.shape!=(192,1) or scale.shape!=(192,1) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('invalid initial normalization')
        trained_spoken.update(previous_train)
        initialization=dict(mode='new-corpus student continuation',checkpoint_sha256=digest(initialize),source_step=previous['step'],source_dataset_sha256=previous['dataset_sha256'],optimizer_reset=True,prior_training_overlap_with_holdout=0,previous=previous.get('initialization'))
    waveform_weight=getattr(args,'waveform_weight',None)
    if waveform_weight is None:waveform_weight=previous.get('waveform_weight',0.0) if args.resume else 0.0
    if not 0<=waveform_weight<=1:raise ValueError('waveform weight must be 0..1')
    duration_weight=getattr(args,'duration_weight',None)
    if duration_weight is None:duration_weight=previous.get('duration_weight',0.2) if args.resume else 0.2
    if not math.isfinite(duration_weight) or not 0<=duration_weight<=10:raise ValueError('duration weight must be finite and 0..10')
    decoder=None
    if waveform_weight:
        if not getattr(args,'teacher',None):raise ValueError('waveform fitting requires source teacher checkpoint')
        saved,teacher_sha=load(args.teacher)
        if teacher_sha!=manifest['teacher_checkpoint_sha256']:raise ValueError('waveform teacher differs')
        teacher=build(saved).eval();teacher.requires_grad_(False)
        if manifest['language']=='ru':
            from vosk_adapter import prepare_export
            prepare_export(teacher)
        from math_cyrillic_export import Decoder
        decoder=Decoder(teacher,saved['training_config']['conditioning_speaker'],manifest['language']=='ru').eval();decoder.requires_grad_(False)
    def batch(values,audio=False):
        lengths=torch.tensor([len(v['tokens']) for v in values]);tokens=torch.zeros(len(values),int(lengths.max()),dtype=torch.long);durations=torch.zeros_like(tokens)
        frames=max(v['latent'].shape[1] for v in values);target=torch.zeros(len(values),192,frames)
        for i,v in enumerate(values):
            n=len(v['tokens']);f=v['latent'].shape[1];tokens[i,:n]=v['tokens'];durations[i,:n]=v['durations'];target[i,:,:f]=(v['latent']-mean)/scale
        output,log_duration,mask=model(tokens,lengths,durations)
        token_mask=torch.arange(tokens.shape[1])[None,:]<lengths[:,None]
        losses=supervised_loss(output,target,log_duration,durations,token_mask,mask)
        return (losses,output,target) if audio else losses
    def evaluate():
        model.eval()
        with torch.no_grad():values=[tuple(float(x) for x in batch([v])) for v in validation]
        model.train();return dict(latent=sum(v[0] for v in values)/len(values),duration=sum(v[1] for v in values)/len(values))
    args.output.mkdir(parents=True,exist_ok=False);initial=evaluate();history=[]
    for step in range(start+1,start+args.steps+1):
        rng=random.Random(1234+step);selected=rng.sample(training,min(args.batch,len(training)))
        optimizer.zero_grad()
        if decoder is not None:
            (latent,duration),prediction,target=batch(selected,True)
            window=min(64,selected[0]['latent'].shape[1]);start_frame=rng.randrange(selected[0]['latent'].shape[1]-window+1)
            section=slice(start_frame,start_frame+window)
            predicted_wave=decoder(prediction[:1,:,section]*scale[None]+mean[None])
            with torch.no_grad():target_wave=decoder(target[:1,:,section]*scale[None]+mean[None])
            waveform=frozen_waveform_loss(predicted_wave,target_wave)
        else:
            latent,duration=batch(selected);waveform=latent.new_zeros(())
        loss=latent+duration_weight*duration+waveform_weight*waveform
        if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
        if decoder is not None and any(p.grad is not None for p in decoder.parameters()):raise ValueError('frozen decoder acquired parameter gradients')
        if step%25==0 or step==start+args.steps:
            record=dict(step=step,train_latent=float(latent.detach()),train_duration=float(duration.detach()),train_waveform=float(waveform.detach()),validation=evaluate());history.append(record);print(json.dumps(record),flush=True)
            save(dict(schema_version=1,architecture='arithmetic-sequence-prototype-v1',model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,initialization=initialization,training_spoken_sha256=sorted(trained_spoken),waveform_weight=waveform_weight,duration_weight=duration_weight,mean=mean,scale=scale,vocabulary=manifest['vocabulary'],dataset_sha256=dataset_sha,teacher_checkpoint_sha256=manifest['teacher_checkpoint_sha256'],source_origin=manifest['source_origin'],rights=manifest['rights'],status='research checkpoint; Python diagnostic inference only; not a native voice package'),args.output/'student.pt')
    (args.output/'report.json').write_text(json.dumps(dict(initial_validation=initial,final_validation=evaluate(),history=history,initialization=initialization,training_spoken_count=len(trained_spoken),waveform_weight=waveform_weight,duration_weight=duration_weight,test_split_used=False,decoder_updated=False,status='Training only; use math_sequence_evaluate for predicted timing; no native export or speech acceptance'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--checkpoint',required=True);a.add_argument('--corpus',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('fit');a.add_argument('--dataset',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--steps',type=int,default=200);a.add_argument('--batch',type=int,default=2);a.add_argument('--resume',type=Path);a.add_argument('--initialize',type=Path);a.add_argument('--initial-dataset',type=Path);a.add_argument('--teacher');a.add_argument('--waveform-weight',type=float);a.add_argument('--duration-weight',type=float)
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(1234)
    (prepare if args.command=='prepare' else fit)(args)
if __name__=='__main__':main()

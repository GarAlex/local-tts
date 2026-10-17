"""Render sequence-student diagnostics through a frozen full-precision decoder.

Compare teacher latents, student teacher-timed latents and predicted-duration
student output. Validation diagnostics only; not native export or quality acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import torch
from checkpoint_tools import load,build
from math_sequence_student import SequenceStudent
from math_cyrillic_export import Decoder


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def spectral_error(actual,expected):
    window=torch.hann_window(512)
    a=torch.stft(actual,512,128,window=window,return_complex=True).abs().clamp_min(1e-5)
    b=torch.stft(expected,512,128,window=window,return_complex=True).abs().clamp_min(1e-5)
    return float((a.log()-b.log()).abs().mean())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student',type=Path,required=True);p.add_argument('--teacher',required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=4)
    args=p.parse_args()
    if args.output.exists():p.error('output already exists')
    if not 1<=args.count<=16:p.error('count must be 1..16')
    torch.set_num_threads(2)
    checkpoint=torch.load(args.student,map_location='cpu',weights_only=True)
    manifest_path=args.dataset/'dataset.json';manifest=json.loads(manifest_path.read_text())
    if checkpoint.get('architecture')!='arithmetic-sequence-prototype-v1' or checkpoint['dataset_sha256']!=sha(manifest_path):raise ValueError('student/dataset contract differs')
    saved,teacher_sha=load(args.teacher)
    if teacher_sha!=checkpoint['teacher_checkpoint_sha256'] or teacher_sha!=manifest['teacher_checkpoint_sha256']:raise ValueError('teacher checkpoint differs')
    net=build(saved).eval();net.requires_grad_(False)
    language=manifest['language']
    if language=='ru':
        from vosk_adapter import prepare_export
        prepare_export(net)
    elif language!='uk':raise ValueError('unsupported sequence teacher language')
    decoder=Decoder(net,saved['training_config']['conditioning_speaker'],language=='ru').eval();decoder.requires_grad_(False)
    model=SequenceStudent(checkpoint['vocabulary']).eval();model.load_state_dict(checkpoint['model'])
    mean,scale=checkpoint['mean'],checkpoint['scale']
    if mean.shape!=(192,1) or scale.shape!=(192,1) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale>0).all():raise ValueError('invalid normalization')
    rows=[r for r in manifest['records'] if r['split']=='validation'][:args.count]
    if len(rows)!=args.count:raise ValueError('insufficient validation utterances')
    inputs=[]
    for row in rows:
        path=(args.dataset/row['file']).resolve()
        if not path.is_relative_to(args.dataset.resolve()) or sha(path)!=row['sha256']:raise ValueError('teacher tensor differs')
        inputs.append(torch.load(path,weights_only=True,map_location='cpu'))
    args.output.mkdir(parents=True,exist_ok=False);cases=[]
    with torch.no_grad():
        for i,(row,data) in enumerate(zip(rows,inputs)):
            tokens=data['tokens'][None];lengths=torch.tensor([tokens.shape[1]]);true_duration=data['durations'][None]
            fitted,_,_=model(tokens,lengths,true_duration)
            predicted,duration,_=model.infer(tokens,lengths)
            truth=decoder(data['latent'][None])[0,0].cpu()
            forced=decoder(fitted*scale[None]+mean[None])[0,0].cpu()
            free=decoder(predicted*scale[None]+mean[None])[0,0].cpu()
            if forced.shape!=truth.shape:raise ValueError('teacher-timed waveform geometry differs')
            recordings=[]
            for label,audio in [('teacher',truth),('teacher-timed-student',forced),('predicted-timing-student',free)]:
                if not torch.isfinite(audio).all() or not audio.numel():raise ValueError('invalid decoded audio')
                file=args.output/f'{i:02}-{label}.wav'
                # PCM16 output for local comparison; clipping is recorded, not hidden.
                peak=float(audio.abs().max());samples=np.clip(np.rint(audio.numpy()*32767),-32768,32767).astype('<i2')
                wavfile.write(file,22050,samples)
                recordings.append(dict(kind=label,file=file.name,sha256=sha(file),duration_seconds=len(audio)/22050,peak=peak,clipped_samples=int((audio.abs()>1).sum())))
            cases.append(dict(expression=row['expression'],spoken=row['spoken'],recordings=recordings,
                              teacher_frames=int(true_duration.sum()),predicted_frames=int(duration.sum()),
                              duration_frame_mae=float((duration-true_duration).abs().float().mean()),
                              teacher_timed_log_spectral_error=spectral_error(forced,truth)))
            print(json.dumps(dict(event='rendered',case=i+1,language=language)),flush=True)
    (args.output/'report.json').write_text(json.dumps(dict(language=language,student_sha256=sha(args.student),teacher_sha256=teacher_sha,dataset_sha256=sha(manifest_path),student_step=checkpoint['step'],split='validation',test_split_used=False,decoder_updated=False,decoder_precision='float32; not packed/compressed',cases=cases,status='research diagnostic; not native export or quality acceptance'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

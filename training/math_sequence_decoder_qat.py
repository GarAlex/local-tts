"""Experimental decoder quantization-aware adaptation on training-only latents."""
import argparse,copy,hashlib,json,random,shutil,zipfile
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.func import functional_call
from checkpoint_tools import load,build
from math_cyrillic_export import Decoder
from math_pretrained_compress import compress
from math_sequence_decoder_check import validate_package
from math_sequence_export import features
from math_sequence_frontend import encode_words
from math_sequence_train import frozen_waveform_loss


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_quantize(weight,block=64,bits=4):
    if bits not in (4,8):raise ValueError('QAT bits must be 4 or 8')
    if weight.ndim<2 or weight.numel()<1024:return weight
    if bits==8:
        axes=tuple(range(1,weight.ndim))
        scale=(weight.detach().abs().amax(axes,keepdim=True)/127).clamp_min(torch.finfo(weight.dtype).tiny)
        rounded=(weight/scale).round().clamp(-127,127)*scale
        return weight+(rounded-weight).detach()
    flat=weight.flatten();padded=torch.nn.functional.pad(flat,(0,(-flat.numel())%block)).reshape(-1,block)
    scale=(padded.detach().abs().amax(1,keepdim=True)/7).clamp_min(torch.finfo(weight.dtype).tiny)
    rounded=(padded/scale).round().clamp(-7,7)*scale
    value=padded+(rounded-padded).detach()
    return value.flatten()[:flat.numel()].reshape_as(weight)


def quantized_forward(decoder,z,bits=4):
    return functional_call(decoder,{name:fake_quantize(value,bits=bits) for name,value in decoder.named_parameters()},(z,))


def read_latents(dataset,manifest):
    training=[];validation=[];seen=set()
    for row in manifest['records']:
        if row['spoken'] in seen or row['split'] not in ('train','validation','test'):raise ValueError('duplicate spoken data or invalid split')
        seen.add(row['spoken'])
        if row['split']=='test':continue
        path=(dataset/row['file']).resolve()
        if not path.is_relative_to(dataset.resolve()) or sha(path)!=row['sha256']:raise ValueError('target integrity differs')
        data=torch.load(path,weights_only=True,map_location='cpu');z=data['latent']
        if z.ndim!=2 or z.shape[0]!=192 or not 1<=z.shape[1]<=7740 or not torch.isfinite(z).all():raise ValueError('invalid training latent')
        (training if row['split']=='train' else validation).append(z)
    if not training or not validation:raise ValueError('missing training/validation data')
    return training,validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','teacher','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bits',type=int,choices=[4,8],default=4)
    p.add_argument('--student-latents',action='store_true',help='calibrate on native predicted student latents instead of teacher latents')
    p.add_argument('--steps',type=int,default=200);p.add_argument('--lr',type=float,default=1e-5)
    a=p.parse_args()
    if a.output.exists():p.error('output exists')
    if not 1<=a.steps<=10000 or not np.isfinite(a.lr) or not 0<a.lr<=1e-3:p.error('invalid steps/lr')
    torch.set_num_threads(2);torch.manual_seed(1234)
    metadata=validate_package(a.source);manifest=json.loads((a.dataset/'dataset.json').read_text())
    saved,digest=load(str(a.teacher))
    language=metadata['language']
    if language not in ('ru','uk') or digest!=metadata['source_checkpoint_sha256'] or digest!=manifest['teacher_checkpoint_sha256'] or manifest['language']!=language:raise ValueError('QAT requires matching RU/UK lineage')
    net=build(saved).eval()
    from vosk_adapter import prepare_export
    if language=='ru':prepare_export(net)
    net.dec.remove_weight_norm()
    decoder=Decoder(net,saved['training_config']['conditioning_speaker'],language=='ru').eval()
    teacher=copy.deepcopy(decoder).eval();teacher.requires_grad_(False)
    decoder.requires_grad_(True)
    training,validation=read_latents(a.dataset,manifest)
    if a.student_latents:
        options=ort.SessionOptions();options.intra_op_num_threads=2
        encoder=ort.InferenceSession(str(a.source/'encoder.onnx'),options,providers=['CPUExecutionProvider'])
        acoustic=ort.InferenceSession(str(a.source/'latent.onnx'),options,providers=['CPUExecutionProvider'])
        training=[];validation=[]
        for row in manifest['records']:
            if row['split']=='test':continue
            ids=np.asarray([row['tokens']],dtype=np.int64)
            fs=features(*encoder.run(None,{'tokens':ids}),1.0)
            z=acoustic.run(None,{'features':fs})[0][0]
            if z.shape[0]!=192 or not np.isfinite(z).all():raise ValueError('invalid student calibration latent')
            (training if row['split']=='train' else validation).append(torch.from_numpy(z.copy()))
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=a.lr)
    # Fixed held-out crops measure reconstruction, never contribute gradients.
    controls=[]
    for z in validation[:8]:
        length=min(64,z.shape[-1]);start=(z.shape[-1]-length)//2
        controls.append(z[None,:,start:start+length])
    with torch.no_grad():targets=[teacher(z) for z in controls]
    def evaluate():
        with torch.no_grad():return sum(float(frozen_waveform_loss(quantized_forward(decoder,z,a.bits),y)) for z,y in zip(controls,targets))/len(controls)
    a.output.mkdir(parents=True,exist_ok=False);initial=evaluate();history=[]
    for step in range(1,a.steps+1):
        rng=random.Random(1234+step);value=training[rng.randrange(len(training))]
        length=min(64,value.shape[-1]);start=rng.randrange(value.shape[-1]-length+1);z=value[None,:,start:start+length]
        with torch.no_grad():target=teacher(z)
        optimizer.zero_grad();predicted=quantized_forward(decoder,z,a.bits)
        loss=frozen_waveform_loss(predicted,target)
        if not torch.isfinite(loss):raise ValueError('nonfinite QAT loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if any(param.grad is not None for param in teacher.parameters()):raise ValueError('frozen teacher acquired gradients')
        if step%25==0 or step==a.steps:
            row=dict(step=step,train_loss=float(loss.detach()),validation=evaluate());history.append(row);print(json.dumps(row),flush=True)
    info=dict(steps=a.steps,lr=a.lr,initial_validation=initial,final_validation=evaluate(),history=history,
        dataset_sha256=sha(a.dataset/'dataset.json'),source_checksums_sha256=sha(a.source/'checksums.json'),teacher_sha256=digest,
        bits=a.bits,language=language,training_count=len(training),validation_count=len(validation),test_split_used=False,decoder_updated=True,calibration_latents="native-student-predicted" if a.student_latents else "teacher",
        objective=f'training-only latent random 64-frame crops; frozen float decoder spectral target; straight-through {a.bits}-bit weights',
        validation_scope='fixed center 64-frame crops from first eight validation utterances; no training gradients',
        tool_sha256=sha(Path(__file__)),status='experimental decoder adaptation; no listening acceptance')
    torch.save(dict(decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),report=info,rights=saved['training_config']['rights'],source_origin=saved['initialization']),a.output/'decoder.pt')
    full=a.output/'decoder-adapted-float.onnx'
    with torch.no_grad():torch.onnx.export(decoder,torch.zeros(1,192,32),full,input_names=['latent'],output_names=['audio'],dynamic_axes={'latent':{2:'frames'},'audio':{2:'samples'}},opset_version=17,dynamo=False)
    out=a.output/'voice';shutil.copytree(a.source,out)
    model=onnx.load(full);compress(model,a.bits);onnx.save(model,out/'decoder.onnx')
    metadata.update(decoder_bits=a.bits,decoder_block_size=64 if a.bits==4 else None,decoder_int8_weights=[],decoder_adaptation=info)
    (out/'sequence.json').write_text(json.dumps(metadata,indent=2));(out/'decoder-adaptation.json').write_text(json.dumps(info,indent=2))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions={name:ort.InferenceSession(str(out/(name+'.onnx')),options,providers=['CPUExecutionProvider']) for name in ['encoder','latent','decoder']}
    words=json.loads((out/'words.json').read_text())['words'];refs=[];bank=bytearray()
    for row in json.loads((a.source/'reference.json').read_text()):
        spoken=[('двена́дцать' if language=='ru' else 'двана́дцять')] if row['expression']=='12' else ['два','плюс','три'] if row['expression']=='2 + 3' else None
        if spoken is None:raise ValueError('unsupported reference expression')
        ids=np.asarray([encode_words(spoken,words,language)],dtype=np.int64)
        fs=features(*sessions['encoder'].run(None,{'tokens':ids}),row['duration_scale']);z=sessions['latent'].run(None,{'features':fs})[0]
        audio=sessions['decoder'].run(None,{'latent':z})[0].reshape(-1)
        if not np.isfinite(audio).all() or len(audio)!=z.shape[-1]*256:raise ValueError('invalid adapted output')
        with torch.no_grad():expected=quantized_forward(decoder,torch.from_numpy(z),a.bits).numpy().reshape(-1)
        if np.max(np.abs(audio-expected))>1e-3:raise ValueError('fake-quantized decoder/export parity failed')
        refs.append(dict(expression=row['expression'],duration_scale=row['duration_scale'],offset=len(bank),count=len(audio)));bank.extend(audio.astype('<f4').tobytes())
    (out/'reference.json').write_text(json.dumps(refs));(out/'reference.bin').write_bytes(bank)
    hashes={f.name:sha(f) for f in out.iterdir() if f.name!='checksums.json'};(out/'checksums.json').write_text(json.dumps(hashes,indent=2))
    archive=a.output/'voice.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):z.write(f,'voice/'+f.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        for name,d in hashes.items():
            if hashlib.sha256(z.read('voice/'+name)).hexdigest()!=d:raise ValueError('archive checksum failure')
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  voice.zip\n')
    info.update(package_bytes=sum(f.stat().st_size for f in out.iterdir()),download_bytes=archive.stat().st_size,download_under_10mb=archive.stat().st_size<10_000_000)
    (a.output/'report.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

if __name__=='__main__':main()

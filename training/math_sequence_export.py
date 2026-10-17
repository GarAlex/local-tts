"""Export experimental continuous numeric students to three native ONNX graphs."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F
from math_sequence_student import SequenceStudent
from math_sequence_frontend import encode_words

class Encoder(nn.Module):
    def __init__(self,student):super().__init__();self.student=student
    def forward(self,tokens):
        m=self.student;embedded=m.embedding(tokens)
        position=torch.arange(tokens.shape[1],device=tokens.device,dtype=embedded.dtype)[:,None]
        frequency=torch.exp(torch.arange(0,96,2,device=tokens.device,dtype=embedded.dtype)*(-math.log(10000.0)/96))
        phase=position*frequency[None,:];positional=torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(1)
        hidden=embedded+positional[None,:,:]
        # Explicit attention keeps sequence length dynamic in legacy ONNX export.
        for layer in m.context.layers:
            q,k,v=F.linear(hidden,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,dim=-1)
            q=q.reshape(1,-1,4,24).transpose(1,2);k=k.reshape(1,-1,4,24).transpose(1,2);v=v.reshape(1,-1,4,24).transpose(1,2)
            attention=torch.softmax((q@k.transpose(-1,-2))/math.sqrt(24),dim=-1)@v
            attention=attention.transpose(1,2).reshape(1,-1,96)
            hidden=layer.norm1(hidden+layer.self_attn.out_proj(attention))
            hidden=layer.norm2(hidden+layer.linear2(layer.activation(layer.linear1(hidden))))
        duration=m.duration(hidden).squeeze(-1).clamp(0,math.log1p(128)).exp()-1
        return hidden.transpose(1,2),duration

class Latent(nn.Module):
    def __init__(self,student,mean,scale):
        super().__init__();self.student=student;self.register_buffer('mean',mean[None]);self.register_buffer('scale',scale[None])
    def forward(self,features):
        m=self.student;frames=F.gelu(m.frame_input(features))
        for layer in m.refine:frames=frames+F.gelu(layer(frames))
        return m.output(frames)*self.scale+self.mean


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def features(hidden,duration,speed):
    counts=np.clip(np.rint(duration[0]*np.float32(speed)),1,256).astype(np.int64)
    total=int(counts.sum())
    if not 0<total<=7740:raise ValueError('invalid predicted frame count')
    out=np.empty((1,98,total),dtype=np.float32);offset=0
    for i,count in enumerate(counts):
        out[0,:96,offset:offset+count]=hidden[0,:,i,None]
        phase=np.arange(count,dtype=np.float32)/max(int(count)-1,1)
        out[0,96,offset:offset+count]=phase;out[0,97,offset:offset+count]=np.sin(phase*np.float32(math.pi));offset+=count
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['student','dataset','phonemes','decoder','output']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    checkpoint=torch.load(a.student,map_location='cpu',weights_only=True);data=json.loads((a.dataset/'dataset.json').read_text());phonemes=json.loads(a.phonemes.read_text())
    if checkpoint['dataset_sha256']!=sha(a.dataset/'dataset.json') or checkpoint['teacher_checkpoint_sha256']!=data['teacher_checkpoint_sha256']:raise ValueError('student dataset lineage differs')
    if phonemes['language']!=data['language'] or phonemes['corpus_sha256']!=data['corpus_sha256']:raise ValueError('native phoneme corpus differs')
    for row in data['records']:
        if encode_words(row['spoken'].split(),phonemes['words'],data['language'])!=row['tokens']:raise ValueError('native full-utterance phonemes differ')
    decoder_metadata=json.loads((a.decoder/'arithmetic-neural.json').read_text())
    if decoder_metadata['source_checkpoint_sha256']!=data['teacher_checkpoint_sha256'] or decoder_metadata['language']!=data['language']:raise ValueError('decoder source differs')
    for name,digest in json.loads((a.decoder/'checksums.json').read_text()).items():
        path=(a.decoder/name).resolve()
        if not path.is_relative_to(a.decoder.resolve()) or sha(path)!=digest:raise ValueError('decoder package integrity differs')
    model=SequenceStudent(checkpoint['vocabulary']).eval();model.load_state_dict(checkpoint['model'])
    encoder=Encoder(model).eval();latent=Latent(model,checkpoint['mean'],checkpoint['scale']).eval()
    a.output.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        torch.onnx.export(encoder,torch.tensor([data['records'][0]['tokens']]),a.output/'encoder.onnx',input_names=['tokens'],output_names=['hidden','duration'],dynamic_axes={'tokens':{1:'tokens'},'hidden':{2:'tokens'},'duration':{1:'tokens'}},opset_version=17,dynamo=False)
        torch.onnx.export(latent,torch.zeros(1,98,32),a.output/'latent.onnx',input_names=['features'],output_names=['latent'],dynamic_axes={'features':{2:'frames'},'latent':{2:'frames'}},opset_version=17,dynamo=False)
    shutil.copy2(a.decoder/'model.onnx',a.output/'decoder.onnx')
    (a.output/'words.json').write_text(json.dumps(dict(words=phonemes['words']),ensure_ascii=False))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions={name:ort.InferenceSession(str(a.output/f'{name}.onnx'),options,providers=['CPUExecutionProvider']) for name in ['encoder','latent','decoder']}
    errors=[]
    with torch.no_grad():
        for row in [r for r in data['records'] if r['split']=='validation'][:3]:
            ids=torch.tensor([row['tokens']]);expected=encoder(ids)
            original,log_duration,_=model._encode(ids,torch.tensor([ids.shape[1]]))
            if float((expected[0]-original.transpose(1,2)).abs().max())>1e-4:raise ValueError('explicit attention differs from trained model')
            actual=sessions['encoder'].run(None,{'tokens':ids.numpy()})
            err=max(float(np.max(np.abs(x.numpy()-y))) for x,y in zip(expected,actual))
            if err>1e-4:raise ValueError('encoder ONNX parity failed')
            fs=features(*actual,1.0);native=latent(torch.from_numpy(fs)).numpy();exported=sessions['latent'].run(None,{'features':fs})[0]
            latent_error=float(np.max(np.abs(native-exported)))
            if latent_error>1e-4:raise ValueError('latent ONNX parity failed')
            errors.append(dict(tokens=len(row['tokens']),encoder_max_error=err,latent_max_error=latent_error))
    # Small portable native fixtures; every reference uses continuous sequence inference.
    refs=[];bank=bytearray()
    for expression,speed in [('12',1.0),('2 + 3',0.7)]:
        # Derive target-language words from already-native corpus readings for integers/operators.
        if data['language']=='ru':spoken=['двена́дцать'] if expression=='12' else ['два','плюс','три']
        else:spoken=['двана́дцять'] if expression=='12' else ['два','плюс','три']
        ids=np.array([encode_words(spoken,phonemes['words'],data['language'])],dtype=np.int64)
        fs=features(*sessions['encoder'].run(None,{'tokens':ids}),speed)
        z=sessions['latent'].run(None,{'features':fs})[0];audio=sessions['decoder'].run(None,{'latent':z})[0].reshape(-1)
        if len(audio)!=fs.shape[-1]*256 or not np.isfinite(audio).all():raise ValueError('invalid decoder output')
        refs.append(dict(expression=expression,duration_scale=speed,offset=len(bank),count=len(audio)));bank.extend(audio.astype('<f4').tobytes())
    (a.output/'reference.json').write_text(json.dumps(refs));(a.output/'reference.bin').write_bytes(bank)
    manifest=dict(schema_version=1,architecture='arithmetic-sequence-v1',language=data['language'],sample_rate=22050,channels=192,hidden=96,hop=256,vocabulary=checkpoint['vocabulary'],source_checkpoint_sha256=data['teacher_checkpoint_sha256'],student_sha256=sha(a.student),student_step=checkpoint['step'],decoder_bits=decoder_metadata['decoder_bits'],quality='experimental research voice; numeric accuracy and listening acceptance pending')
    (a.output/'sequence.json').write_text(json.dumps(manifest,indent=2))
    (a.output/'training-origin.json').write_text(json.dumps(checkpoint['source_origin'],ensure_ascii=False,indent=2));(a.output/'rights.json').write_text(json.dumps(checkpoint['rights'],ensure_ascii=False,indent=2))
    for i,notice in enumerate(checkpoint['source_origin'].get('notices',{}).values()):(a.output/f'BASE-NOTICE-{i:02}.txt').write_text(notice)
    (a.output/'export-report.json').write_text(json.dumps(dict(parity=errors,student_initialization=checkpoint.get('initialization'),design='continuous whole-utterance student; phoneme lookup only; no cached acoustic fragments'),indent=2))
    hashes={path.name:sha(path) for path in a.output.iterdir() if path.is_file()};(a.output/'checksums.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(bytes=sum(path.stat().st_size for path in a.output.iterdir()),parity=errors)),flush=True)
if __name__=='__main__':main()

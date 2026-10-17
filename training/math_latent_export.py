"""Export a numeric-only decoder plus cached word conditioning (not distillation)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort
from checkpoint_tools import load, build
from math_pretrained_compress import compress


def numeric_context(word):
    if word in ('hundred','thousand','million'):
        return ['one',word,'plus','one'],1
    if word in ('plus','minus','times','equals'):
        return ['one',word,'one'],1
    if word in ('divided','by'):
        return ['one','divided','by','one'],1 if word=='divided' else 2
    if word in ('negative','positive'):
        return [word,'one','plus','one'],0
    if word=='point':return ['one','point','one'],1
    if word in ('open','close','parenthesis'):
        return ['open','parenthesis','one','plus','one','close','parenthesis'],{'open':0,'close':5,'parenthesis':1}[word]
    if word in ('twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'):
        return [word,'one','plus','one'],0
    return [word,'plus','one'],0


def phrase_tokens(phrase,index,inventory,space):
    tokens=[0];start=end=None
    for i,word in enumerate(phrase):
        if i:tokens.extend([space,0])
        if i==index:start=len(tokens)
        tokens.extend(inventory[word][1:])
        if i==index:end=len(tokens)
    return tokens,start,end


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--source-voice',required=True)
    parser.add_argument('--tokens',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--contextual',action='store_true',help='extract each word from a one/word/one phrase')
    parser.add_argument('--numeric-contexts',action='store_true',help='extract words from grammatical arithmetic phrases')
    parser.add_argument('--compounds',action='store_true',help='include tens/units phrases with compact binary assets')
    parser.add_argument('--hundreds',action='store_true',help='also include one hundred through nine hundred')
    parser.add_argument('--reuse-core',help='reuse validated grammatical conditioning from a schema-1 or schema-2 package')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():parser.error('output already exists')
    torch.set_num_threads(2);torch.manual_seed(1234)
    saved,digest=load(args.checkpoint)
    if saved.get('model_profile')!='ljs-sdp-v1':raise ValueError('requires pretrained LJS voice')
    source=Path(args.source_voice)
    origin=json.loads((source/'training-origin.json').read_text())
    if origin!=saved['initialization']:raise ValueError('source voice does not match checkpoint lineage')
    for name,expected in json.loads((source/'checksums.json').read_text()).items():
        path=(source/name).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('source package checksum mismatch')
    words=json.loads(Path(args.tokens).read_text())
    if args.hundreds and not args.compounds:raise ValueError('hundreds require compounds')
    count=124 if args.hundreds else 115 if args.compounds else 43
    if len(words)!=count or len({w['word'] for w in words})!=count:raise ValueError(f'requires {count} unique arithmetic entries')
    reusable={};reuse_bank=None;reuse_digest=None
    if args.reuse_core:
        if not args.numeric_contexts:raise ValueError('reused core requires numeric contexts')
        reuse=Path(args.reuse_core)
        for name,expected in json.loads((reuse/'checksums.json').read_text()).items():
            path=(reuse/name).resolve()
            if not path.is_relative_to(reuse.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('reused package checksum mismatch')
        metadata=json.loads((reuse/'arithmetic-neural.json').read_text())
        if metadata['schema_version'] not in (1,2) or metadata['source_checkpoint_sha256']!=digest or metadata['conditioning_context']!='grammatical arithmetic phrases':
            raise ValueError('incompatible reused conditioning')
        reusable=metadata['entries'];reuse_bank=(reuse/'conditioning.bin').read_bytes()
        if metadata['schema_version']==2:
            raw_scales=(reuse/'conditioning-scales.bin').read_bytes()
            reusable={word:dict(entry,scales=np.frombuffer(raw_scales[entry['scales_offset']:entry['scales_offset']+192*4],dtype='<f4').tolist()) for word,entry in reusable.items()}
        reuse_digest=hashlib.sha256(reuse_bank).hexdigest()
    net=build(saved).eval();net.dec.remove_weight_norm()
    with torch.no_grad():
        for case in json.loads((source/'reference.json').read_text()):
            ids=torch.tensor([case['tokens']])
            audio=net.infer(ids,torch.tensor([ids.shape[1]]),noise_scale=0,noise_scale_w=0,
                            length_scale=case['duration_scale'])[0][0,0].numpy()
            expected=np.array(case['samples'],dtype=np.float32)
            if audio.shape!=expected.shape or np.max(np.abs(audio-expected))>1e-4:
                raise ValueError('source checkpoint does not reproduce voice fixtures')
    output.mkdir(parents=True,exist_ok=False)
    bank=bytearray();scale_bank=bytearray();entries={};fixtures=[];scales_by_word={}
    with torch.no_grad():
        for record in words:
            word=record['word']
            if word in reusable:
                entry=reusable[word];frames=entry['frames'];scale=np.asarray(entry['scales'],dtype=np.float32)
                raw_quantized=reuse_bank[entry['offset']:entry['offset']+frames*192]
                if len(raw_quantized)!=frames*192 or scale.shape!=(192,) or not np.isfinite(scale).all():raise ValueError('invalid reused conditioning')
                q=np.frombuffer(raw_quantized,dtype=np.int8).reshape(192,frames)
            else:
                q,scale=generate_conditioning(net,record,words,saved,args)
                frames=q.shape[1]
            entries[word]=dict(offset=len(bank),frames=frames)
            scales_by_word[word]=scale
            if args.compounds:
                entries[word]['scales_offset']=len(scale_bank)
                scale_bank.extend(scale.astype('<f4').tobytes())
            else:entries[word]['scales']=scale.tolist()
            bank.extend(q.tobytes())
    torch.onnx.export(net.dec,torch.zeros(1,192,32),output/'model.onnx',
                      input_names=['latent'],output_names=['audio'],
                      dynamic_axes={'latent':{2:'frames'},'audio':{2:'samples'}},
                      opset_version=17,dynamo=False)
    model=onnx.load(output/'model.onnx');count=compress(model,4);onnx.save(model,output/'model.onnx')
    (output/'conditioning.bin').write_bytes(bank)
    if args.compounds:(output/'conditioning-scales.bin').write_bytes(scale_bank)
    manifest=dict(schema_version=2 if args.compounds else 1,architecture='arithmetic-latent-v1',language='en',sample_rate=22050,
                  channels=192,hop=256,model='model.onnx',conditioning='conditioning.bin',entries=entries,
                  source_checkpoint_sha256=digest,decoder_bits=4,
                  source_tokens_sha256=hashlib.sha256(Path(args.tokens).read_bytes()).hexdigest(),
                  reused_core_bank_sha256=reuse_digest,
                  conditioning_context='grammatical arithmetic phrases' if args.numeric_contexts else 'one/word/one medial extraction' if args.contextual else 'isolated words',
                  design='cached acoustic conditioning with shared neural waveform generator; not end-to-end distillation',
                  quality='experimental; joins and prosody require listening')
    if args.compounds:manifest['conditioning_scales']='conditioning-scales.bin'
    if args.compounds:manifest['repeat_digit_pause_ms']=45
    (output/'arithmetic-neural.json').write_text(json.dumps(manifest,indent=2))
    for path in source.iterdir():
        if path.name.startswith('BASE-') or path.name in ('VITS-LICENSE.txt','CMU-LICENSE.txt','rights.json','training-origin.json'):
            shutil.copy2(path,output/path.name)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(output/'model.onnx'),opts,providers=['CPUExecutionProvider'])
    reference_bank=bytearray()
    for word in ('twelve','plus','hundred') + (('seventy four','twenty nine') if args.compounds else ()):
        e=entries[word];n=192*e['frames']
        q=np.frombuffer(bank[e['offset']:e['offset']+n],dtype=np.int8).reshape(192,-1)
        z=(q*scales_by_word[word][:,None])[None]
        audio=session.run(None,{'latent':z})[0].reshape(-1)
        if args.compounds:
            fixtures.append(dict(word=word,offset=len(reference_bank),count=len(audio)))
            reference_bank.extend(audio.astype('<f4').tobytes())
        else:fixtures.append(dict(word=word,samples=audio.tolist()))
    (output/'latent-reference.json').write_text(json.dumps(fixtures))
    if args.compounds:(output/'latent-reference.bin').write_bytes(reference_bank)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'checksums.json').write_text(json.dumps(hashes,indent=2))
    report=dict(graph_bytes=(output/'model.onnx').stat().st_size,conditioning_bytes=len(bank),
                scale_bytes=len(scale_bank),entries=len(entries),
                total_package_bytes=sum(p.stat().st_size for p in output.iterdir()),quantized_tensors=count)
    print(json.dumps(report),flush=True)


def generate_conditioning(net,record,words,saved,args):
    tokens=record['tokens'];start_token=0;end_token=len(tokens)
    if args.numeric_contexts:
        phrase,index=numeric_context(record['word'])
        inventory={w['word']:w['tokens'] for w in words}
        tokens,start_token,end_token=phrase_tokens(phrase,index,inventory,saved['training_config']['symbols'].index(' '))
    elif args.contextual:
        anchor=next(w['tokens'] for w in words if w['word']=='one')
        space=saved['training_config']['symbols'].index(' ')
        start_token=len(anchor)+2
        tokens=anchor+[space,0]+tokens[1:]+[space,0]+anchor[1:]
        end_token=start_token+len(record['tokens'])-1
    x=torch.tensor([tokens])
    audio,attention,mask,latents=net.infer(x,torch.tensor([x.shape[1]]),noise_scale=0,noise_scale_w=0)
    raw=audio[0,0].numpy();z=(latents[0]*mask)[0].numpy()
    if args.contextual or args.numeric_contexts:
        durations=attention[0,0].sum(0).numpy().astype(np.int64)
        a=int(durations[:start_token].sum());b=int(durations[:end_token].sum())
        z=z[:,a:b];raw=raw[a*256:b*256]
    rms=np.sqrt(np.mean(raw[:z.shape[1]*256].reshape(-1,256)**2,axis=1))
    active=np.flatnonzero(rms>max(float(rms.max())*.025,1e-4))
    if not len(active):raise ValueError('teacher word has no acoustic activity')
    start=max(0,int(active[0])-3);end=min(z.shape[1],int(active[-1])+4)
    z=z[:,start:end]
    scale=np.maximum(np.max(np.abs(z),axis=1)/127,np.finfo(np.float32).tiny)
    q=np.clip(np.rint(z/scale[:,None]),-127,127).astype(np.int8)
    return q,scale

if __name__=='__main__':main()

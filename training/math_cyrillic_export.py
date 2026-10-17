"""Numeric-only pretrained European/Cyrillic conditioning exporter; no end-to-end distillation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from checkpoint_tools import load, build
from math_pretrained_compress import compress

class Decoder(torch.nn.Module):
    def __init__(self, net, speaker, russian):
        super().__init__()
        self.decoder = net.dec
        self.register_buffer('speaker', net.emb_g(torch.tensor([speaker])).detach().unsqueeze(-1))
        self.russian = russian
    def forward(self, latent):
        value = self.decoder(latent, g=self.speaker)
        return value[0] if self.russian else value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--tokens',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--contextual',action='store_true',help='use validated native arithmetic phrase token spans')
    p.add_argument('--decode-per-entry',action='store_true',help='experimental independent word decoding with waveform joins')
    p.add_argument('--double-zero',action='store_true',help='schema-3 native cached double-zero phrase; retain isolated core')
    p.add_argument('--bits',type=int,choices=[4,8],default=4)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():p.error('output already exists')
    torch.set_num_threads(2);torch.manual_seed(1234)
    saved,digest=load(args.checkpoint)
    config=saved['training_config'];language=config['language']
    supported = {
        'vosk-ru-multi-v1': {'ru'}, 'piper-uk-medium-v1': {'uk'},
        'css10-de-es-v1': {'de', 'es'},
        'neon-european-v1': {'fr', 'it', 'pt', 'pl', 'nl'},
    }
    if language not in supported.get(saved['model_profile'], set()):
        raise ValueError('requires a supported pinned European or Cyrillic checkpoint')
    if config['sample_rate'] != 22050:
        raise ValueError('requires the 22050 Hz decoder contract')
    records=json.loads(Path(args.tokens).read_text())
    if not 40 <= len(records) <= 100 or len({r['word'] for r in records}) != len(records):
        raise ValueError('invalid arithmetic inventory')
    for r in records:
        if not r['word'] or not r['tokens'] or len(r['tokens'])>1024 or any(type(t)!=int or not 0<=t<len(config['symbols']) for t in r['tokens']):
            raise ValueError('invalid token entry')
    if args.contextual or args.double_zero:
        if language not in ('ru','uk'):raise ValueError('native contextual spans currently require ru or uk')
        for record in records:
            if not args.contextual and 'context_tokens' not in record:continue
            ids=record.get('context_tokens',[])
            start,end=record.get('context_start'),record.get('context_end')
            if (not ids or len(ids)>1024 or any(type(t)!=int or not 0<=t<len(config['symbols']) for t in ids)
                    or type(start)!=int or type(end)!=int or not 0<=start<end<=len(ids)):
                raise ValueError('invalid native contextual token span')
            suffix=2 if language=='ru' else 1
            if ids[start:end]!=record['tokens'][2:-suffix]:
                raise ValueError('context span does not reproduce isolated word tokens')
    if args.double_zero:
        zero={'ru':'ноль','uk':'нуль'}[language]
        pairs=[r for r in records if 'context_tokens' in r]
        if args.contextual or len(records)!=56 or len(pairs)!=1 or pairs[0]['word']!=f'{zero} {zero}':
            raise ValueError('double-zero export requires isolated core plus exactly one native zero pair')
    net=build(saved).eval();speaker=config['conditioning_speaker'];sid=torch.tensor([speaker])
    russian=language=='ru'
    if russian:
        from vosk_adapter import prepare_export
        prepare_export(net)
    net.dec.remove_weight_norm()
    decoder=Decoder(net,speaker,russian).eval()
    entries={};bank=bytearray();scales=bytearray();reference_latents=[]
    with torch.no_grad():
        for record in records:
            use_context=args.contextual or (args.double_zero and 'context_tokens' in record)
            x=torch.tensor([record['context_tokens'] if use_context else record['tokens']])
            result=net.infer(x,torch.tensor([x.shape[1]]),sid=sid,noise_scale=0,noise_scale_w=0)
            raw=result[0][0,0].numpy()
            mask=result[3] if russian else result[2]
            z=((result[4] if russian else result[3])[0]*mask)[0].numpy()
            if len(raw)!=z.shape[1]*256:raise ValueError('unexpected decoder hop')
            if use_context:
                attention=result[2] if russian else result[1]
                durations=attention[0,0].sum(0).numpy().astype(np.int64)
                a=int(durations[:record['context_start']].sum())
                b=int(durations[:record['context_end']].sum())
                if not 0<=a<b<=z.shape[1]:raise ValueError('empty or invalid contextual frame span')
                z=z[:,a:b];raw=raw[a*256:b*256]
            rms=np.sqrt(np.mean(raw.reshape(-1,256)**2,axis=1))
            active=np.flatnonzero(rms>max(float(rms.max())*.025,1e-4))
            if not len(active):raise ValueError('silent teacher entry')
            z=z[:,max(0,int(active[0])-3):min(z.shape[1],int(active[-1])+4)]
            if z.shape[0]!=192 or not 1<=z.shape[1]<=1024:raise ValueError('invalid conditioning shape')
            scale=np.maximum(np.max(np.abs(z),axis=1)/127,np.finfo(np.float32).tiny)
            q=np.clip(np.rint(z/scale[:,None]),-127,127).astype(np.int8)
            entries[record['word']]=dict(offset=len(bank),frames=z.shape[1],scales_offset=len(scales))
            bank.extend(q.tobytes());scales.extend(scale.astype('<f4').tobytes())
            if len(reference_latents)<3:reference_latents.append((record['word'],(q*scale[:,None])[None]))
    out.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        torch.onnx.export(decoder,torch.zeros(1,192,32),out/'model.onnx',input_names=['latent'],output_names=['audio'],
                          dynamic_axes={'latent':{2:'frames'},'audio':{2:'samples'}},opset_version=17,dynamo=False)
    model=onnx.load(out/'model.onnx');compress(model,args.bits);onnx.save(model,out/'model.onnx')
    (out/'conditioning.bin').write_bytes(bank);(out/'conditioning-scales.bin').write_bytes(scales)
    manifest=dict(schema_version=3 if args.double_zero else 2,architecture='arithmetic-latent-v1',language=language,sample_rate=22050,channels=192,hop=256,
                  model='model.onnx',conditioning='conditioning.bin',conditioning_scales='conditioning-scales.bin',
                  entries=entries,repeat_digit_pause_ms=45,decode_per_entry=args.decode_per_entry,source_checkpoint_sha256=digest,
                  source_tokens_sha256=hashlib.sha256(Path(args.tokens).read_bytes()).hexdigest(),decoder_bits=args.bits,
                  conditioning_context='isolated core with native double-zero phrase' if args.double_zero else 'native arithmetic phrase spans' if args.contextual else 'isolated stressed words',source_speaker=config['speaker'],
                  design='cached acoustic conditioning with shared neural waveform generator; not end-to-end distillation',
                  quality='experimental; joins and numeric accuracy require listening')
    (out/'arithmetic-neural.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'training-origin.json').write_text(json.dumps(saved['initialization'],ensure_ascii=False,indent=2))
    (out/'rights.json').write_text(json.dumps(config['rights'],ensure_ascii=False,indent=2))
    for i,(name,notice) in enumerate(saved['initialization'].get('notices',{}).items()):
        # Use only generated safe filenames; original names remain in provenance.
        (out/f'BASE-NOTICE-{i:02}.txt').write_text(notice)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(out/'model.onnx'),opts,providers=['CPUExecutionProvider'])
    refs=[];reference=bytearray();errors=[]
    with torch.no_grad():
        for word,z in reference_latents:
            audio=session.run(None,{'latent':z})[0].reshape(-1)
            native=decoder(torch.from_numpy(z))[0,0].numpy()
            if audio.shape != native.shape or not np.isfinite(audio).all():raise ValueError('invalid exported waveform')
            errors.append(float(np.max(np.abs(audio-native))))
            refs.append(dict(word=word,offset=len(reference),count=len(audio)))
            reference.extend(audio.astype('<f4').tobytes())
    (out/'latent-reference.json').write_text(json.dumps(refs,ensure_ascii=False))
    (out/'latent-reference.bin').write_bytes(reference)
    (out/'export-report.json').write_text(json.dumps(dict(quantization_max_errors=errors,reference='compressed ONNX fixtures; errors compare float decoder on identical conditioning'),indent=2))
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
    (out/'checksums.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(language=language,entries=len(entries),graph_bytes=(out/'model.onnx').stat().st_size,
                         total_package_bytes=sum(f.stat().st_size for f in out.iterdir()))),flush=True)
if __name__=='__main__':main()

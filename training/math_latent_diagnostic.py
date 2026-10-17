"""Isolate cached-word joins from decoder quantization on development examples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from scipy.io import wavfile
from checkpoint_tools import load,build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--voice',required=True)
    parser.add_argument('--comparison',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():parser.error('output already exists')
    root=Path(args.voice)
    for name,expected in json.loads((root/'checksums.json').read_text()).items():
        file=(root/name).resolve()
        if not file.is_relative_to(root.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('voice checksum mismatch')
    manifest=json.loads((root/'arithmetic-neural.json').read_text())
    saved,digest=load(args.checkpoint)
    if digest!=manifest['source_checkpoint_sha256']:raise ValueError('decoder source mismatch')
    comparison=json.loads(Path(args.comparison).read_text())
    if comparison.get('split')!='validation':raise ValueError('development diagnostics require validation')
    torch.set_num_threads(2)
    net=build(saved).eval();net.dec.remove_weight_norm()
    options=ort.SessionOptions();options.intra_op_num_threads=2
    quant=ort.InferenceSession(str(root/'model.onnx'),options,providers=['CPUExecutionProvider'])
    bank=(root/'conditioning.bin').read_bytes()
    scales_bank=(root/'conditioning-scales.bin').read_bytes() if manifest['schema_version']==2 else None
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    with torch.no_grad():
        for i in (1,8,10,11,13):
            example=comparison['examples'][i]
            conditioning=[]
            words=example['spoken'].split();selected=[];index=0
            while index<len(words):
                compound=' '.join(words[index:index+2])
                if index+1<len(words) and compound in manifest['entries']:
                    selected.append(compound);index+=2
                else:selected.append(words[index]);index+=1
            for word in selected:
                entry=manifest['entries'][word];n=entry['frames']*192
                q=np.frombuffer(bank[entry['offset']:entry['offset']+n],dtype=np.int8).reshape(192,-1)
                scale=np.frombuffer(scales_bank[entry['scales_offset']:entry['scales_offset']+192*4],dtype='<f4') if scales_bank is not None else np.asarray(entry['scales'],dtype=np.float32)
                conditioning.append(q*scale[:,None])
            z=np.concatenate(conditioning,axis=1)[None]
            original=net.dec(torch.from_numpy(z))[0,0].numpy()
            packed=quant.run(None,{'latent':z})[0].reshape(-1)
            for name,audio in [('float',original),('packed',packed)]:
                if not np.isfinite(audio).all():raise ValueError('non-finite audio')
                wavfile.write(output/f'{i:02}-{name}.wav',22050,np.clip(audio*32767,-32768,32767).astype(np.int16))
            records.append(dict(index=i,spoken=example['spoken'],expression=example['expression'],
                                waveform_rmse=float(np.sqrt(np.mean((original-packed)**2)))))
    (output/'report.json').write_text(json.dumps(dict(source_sha256=digest,examples=records,
        interpretation='Identical greedily selected cached conditioning with original float and packed four-bit decoders. Digit pauses excluded to isolate decoder error; no quality acceptance implied.'),indent=2))
    print(f'Created {len(records)} paired diagnostics',flush=True)


if __name__=='__main__':main()

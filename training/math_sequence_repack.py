"""Repack a checked sequence voice from its original frozen decoder weights.

Preserves encoder, acoustic predictor and phoneme lookup byte-for-byte. Regenerates
native fixtures, checksums and ZIP. Download size is measured, not assumed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import onnx
import onnxruntime as ort
import torch
from checkpoint_tools import load, build
from math_cyrillic_export import Decoder
from math_pretrained_compress import compress
from math_sequence_decoder_check import validate_package
from math_sequence_export import features
from math_sequence_frontend import encode_words


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_weights(model, tail):
    from onnx import numpy_helper
    eligible={t.name for t in model.graph.initializer if t.data_type==onnx.TensorProto.FLOAT and len(t.dims)>=2 and numpy_helper.to_array(t).size>=1024}
    indices=[i for i,node in enumerate(model.graph.node) if node.op_type=='ConvTranspose' and len(node.input)>1 and node.input[1] in eligible]
    if not indices:raise ValueError('no learned upsampling stage')
    stage=model.graph.node[max(indices):] if tail else model.graph.node[:min(indices)+1]
    consumed={name for node in stage for name in node.input}
    result=sorted(eligible & consumed)
    if not result:raise ValueError('no learned stage weights selected')
    return result


def middle_weights(model):
    excluded=set(stage_weights(model,True)+stage_weights(model,False))
    groups={}
    for tensor in model.graph.initializer:
        if tensor.name not in excluded and tensor.data_type==onnx.TensorProto.FLOAT and len(tensor.dims)==3:
            groups.setdefault(tensor.dims[-1],[]).append(tensor.name)
    if not groups:raise ValueError('no intermediate residual groups')
    return {kernel:sorted(names) for kernel,names in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','teacher','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--block-sizes',type=int,nargs='+',default=[32,16])
    p.add_argument('--include-int5',action='store_true')
    p.add_argument('--include-int8',action='store_true')
    p.add_argument('--int8-middle',action='store_true',help='test each intermediate residual kernel group, alone and with int8 tail')
    p.add_argument('--int8-head',action='store_true',help='also test int8 input and first learned upsampling stage')
    p.add_argument('--int8-tail',action='store_true',help='also test int8 weights from the final upsampling stage onward')
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    if len(set(a.block_sizes))!=len(a.block_sizes) or any(v<2 or v>1024 or v%2 for v in a.block_sizes):p.error('block sizes must be distinct even integers in 2..1024')
    torch.set_num_threads(2)
    manifest=validate_package(a.source)
    saved,digest=load(str(a.teacher))
    if digest!=manifest['source_checkpoint_sha256'] or saved['training_config']['language']!=manifest['language']:raise ValueError('teacher lineage differs')
    net=build(saved).eval();net.requires_grad_(False)
    russian=manifest['language']=='ru'
    if russian:
        from vosk_adapter import prepare_export
        prepare_export(net)
    elif manifest['language']!='uk':raise ValueError('requires ru or uk')
    decoder=Decoder(net,saved['training_config']['conditioning_speaker'],russian).eval();decoder.requires_grad_(False)
    a.output.mkdir(parents=True,exist_ok=False)
    full=a.output/'decoder-float.onnx'
    with torch.no_grad():
        torch.onnx.export(decoder,torch.zeros(1,192,32),full,input_names=['latent'],output_names=['audio'],dynamic_axes={'latent':{2:'frames'},'audio':{2:'samples'}},opset_version=17,dynamo=False)
    original=onnx.load(full)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    encoder=ort.InferenceSession(str(a.source/'encoder.onnx'),options,providers=['CPUExecutionProvider'])
    latent=ort.InferenceSession(str(a.source/'latent.onnx'),options,providers=['CPUExecutionProvider'])
    words=json.loads((a.source/'words.json').read_text())['words']
    reference=json.loads((a.source/'reference.json').read_text())
    refs=[]
    for row in reference:
        spoken=['двена́дцать' if russian else 'двана́дцять'] if row['expression']=='12' else ['два','плюс','три'] if row['expression']=='2 + 3' else None
        if spoken is None:raise ValueError('unsupported source reference expression')
        ids=np.asarray([encode_words(spoken,words,manifest['language'])],dtype=np.int64)
        fs=features(*encoder.run(None,{'tokens':ids}),row['duration_scale'])
        refs.append((row,latent.run(None,{'features':fs})[0]))
    rows=[]
    profiles=[(f'q4-block{v}',4,v,[]) for v in a.block_sizes]+([('q8',8,64,[])] if a.include_int8 else [])
    if a.include_int5:profiles.append(('q5-block64',5,64,[]))
    if a.int8_tail:
        tail=stage_weights(original,True)
        profiles.append(('q4-tail-int8',4,64,tail))
    if a.int8_head:
        head=stage_weights(original,False)
        profiles.append(('q4-head-int8',4,64,head))
        if a.int8_tail:profiles.append(('q4-head-tail-int8',4,64,sorted(set(head+tail))))
    if a.int8_middle:
        groups=middle_weights(original)
        for kernel,names in sorted(groups.items()):
            profiles.append((f'q4-mid-k{kernel}-int8',4,64,sorted(names)))
            profiles.append((f'q4-mid-k{kernel}-tail-int8',4,64,sorted(set(names+stage_weights(original,True)))))
    for name,bits,block,int8_names in profiles:
        out=a.output/name;shutil.copytree(a.source,out)
        model=copy.deepcopy(original);count=compress(model,bits,block,int8_names);onnx.save(model,out/'decoder.onnx')
        session=ort.InferenceSession(str(out/'decoder.onnx'),options,providers=['CPUExecutionProvider'])
        fixtures=[];bank=bytearray()
        for row,z in refs:
            audio=session.run(None,{'latent':z})[0].reshape(-1)
            if audio.size!=z.shape[-1]*256 or not np.isfinite(audio).all():raise ValueError('invalid repacked output')
            fixtures.append(dict(expression=row['expression'],duration_scale=row['duration_scale'],offset=len(bank),count=len(audio)))
            bank.extend(audio.astype('<f4').tobytes())
        (out/'reference.json').write_text(json.dumps(fixtures));(out/'reference.bin').write_bytes(bank)
        metadata=dict(manifest,decoder_bits=bits,decoder_block_size=block if bits in (4,5) else None,decoder_int8_weights=int8_names)
        (out/'sequence.json').write_text(json.dumps(metadata,indent=2))
        comparison=dict(source_package_checksums_sha256=sha(a.source/'checksums.json'),teacher_sha256=digest,
            full_decoder_sha256=sha(full),compressed_tensors=count,bits=bits,block_size=block if bits in (4,5) else None,int8_weights=int8_names,
            preserved_graphs={n:sha(out/n) for n in ['encoder.onnx','latent.onnx','words.json']},
            status='experimental repack; quality acceptance pending; float32 execution')
        (out/'repack.json').write_text(json.dumps(comparison,indent=2))
        hashes={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='checksums.json'}
        (out/'checksums.json').write_text(json.dumps(hashes,indent=2))
        archive=a.output/(name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.iterdir()):zf.write(f,name+'/'+f.name)
        with zipfile.ZipFile(archive) as zf:
            if zf.testzip() is not None:raise ValueError('archive CRC failed')
            for file,digest_file in hashes.items():
                if hashlib.sha256(zf.read(name+'/'+file)).hexdigest()!=digest_file:raise ValueError('archive checksum differs')
        archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
        rows.append(dict(name=name,package_bytes=sum(f.stat().st_size for f in out.iterdir()),download_bytes=archive.stat().st_size,download_under_10mb=archive.stat().st_size<10_000_000))
        print(json.dumps(rows[-1]),flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(source_checksums_sha256=sha(a.source/'checksums.json'),tool_sha256=sha(Path(__file__)),candidates=rows),indent=2))

if __name__=='__main__':main()

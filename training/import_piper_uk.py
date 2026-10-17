"""Restricted import of the pinned original Ukrainian Piper medium training checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PosixPath
import tempfile
import torch
from train import PIPER_MODEL, SynthesizerTrn, MultiPeriodDiscriminator
ROOT = Path(__file__).resolve().parents[1]

def symbols():
    config = json.loads((ROOT/'assets/piper-uk-config.json').read_text())
    inventory = [chr(0xf0000+i) for i in range(256)]
    for char, ids in config['phoneme_id_map'].items():
        if len(char)!=1 or len(ids)!=1: raise ValueError('unsupported Piper character mapping')
        inventory[ids[0]] = '\0' if ids[0]==0 else char
    return inventory

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(65536):digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'voices/vits-piper-uk-base/source'))
    parser.add_argument('--output',default=str(ROOT/'voices/vits-piper-uk-base/base.pt'))
    parser.add_argument('--speaker',choices=['lada','mykyta','tetiana'],default='lada')
    args=parser.parse_args();source=Path(args.source);output=Path(args.output)
    if output.exists():raise ValueError('base output exists; choose a new destination')
    manifest=json.loads((ROOT/'assets/vits-piper-uk-assets.json').read_text())
    contents={}
    for name,record in manifest['files'].items():
        if name.endswith('.ckpt'):continue
        contents[name]=(source/name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest()!=record['sha256']:raise ValueError('source checksum mismatch: '+name)
    config=json.loads(contents['config.json'])
    pinned=json.loads((ROOT/'assets/piper-uk-config.json').read_text())
    for key in ['phoneme_type','phoneme_id_map','speaker_id_map','num_symbols','num_speakers']:
        if config[key]!=pinned[key]:raise ValueError('source token/speaker configuration mismatch')
    if config['audio']['sample_rate']!=22050:raise ValueError('unsupported source sample rate')
    path=source/'epoch=2090-step=1166778.ckpt'
    # Only this known inert standard-library metadata type is additionally allowed.
    # No dynamic globals, model code, unsafe pickle fallback or source paths are executed.
    torch.set_num_threads(2)
    with path.open('rb') as stream:
        def stream_hash():
            stream.seek(0);h=hashlib.sha256()
            while chunk:=stream.read(65536):h.update(chunk)
            return h.hexdigest()
        expected=manifest['files'][path.name]['sha256']
        if stream_hash()!=expected:raise ValueError('source checksum mismatch: '+path.name)
        stream.seek(0)
        with torch.serialization.safe_globals([PosixPath]):
            public=torch.load(stream,map_location='cpu',weights_only=True)
        if stream_hash()!=expected:raise ValueError('source checkpoint changed while loading')
    hyper=public['hyper_parameters']
    cfg=json.loads(json.dumps({key:hyper['num_speakers' if key=='n_speakers' else key] for key in PIPER_MODEL}))
    if cfg!=PIPER_MODEL or hyper['filter_length']!=1024 or hyper['hop_length']!=256 or hyper['sample_rate']!=22050 or hyper['segment_size']!=8192 or hyper['num_symbols']!=256:
        raise ValueError('unsupported source architecture/acoustics')
    states=public['state_dict']
    if any(not key.startswith(('model_g.','model_d.')) for key in states):raise ValueError('unexpected source state prefix')
    generator={key.removeprefix('model_g.'):value for key,value in states.items() if key.startswith('model_g.')}
    discriminator={key.removeprefix('model_d.'):value for key,value in states.items() if key.startswith('model_d.')}
    net=SynthesizerTrn(256,513,32,**cfg);net.load_state_dict(generator,strict=True)
    disc=MultiPeriodDiscriminator(False);disc.load_state_dict(discriminator,strict=True)
    if any(not torch.isfinite(value).all() for value in list(generator.values())+list(discriminator.values())):
        raise ValueError('non-finite source weights')
    notices={name:contents[name].decode() for name in manifest['files'] if name.endswith(('.md','.txt')) or name=='MODEL_CARD'}
    notices['PROVENANCE.json']=json.dumps(manifest,indent=2)
    rights=dict(audio='Published Ukrainian Piper checkpoint; supplied card CC0 claim has attribution gap; dataset repository Apache-2.0',
                transcripts='Published Ukrainian TTS dataset record retained; target transcripts require separate rights',
                speaker_consent='Published three-speaker corpus; complete consent lineage is not supplied')
    origin=dict(mode='public-import',repository=manifest['repository'],revision=manifest['revision'],
                weights_sha256=manifest['files'][path.name]['sha256'],license=manifest['license'],
                license_limit='; '.join(manifest['provenance_limits']),
                frontend_note='Native Piper Ukrainian text/NFD/BOS/pad/EOS indices preserved; numbers and explicit stress use the same project frontend in preparation/inference',
                discriminator='included in original checkpoint',conditioning_speaker=manifest['speakers'][args.speaker],
                source_speaker=args.speaker,source_epoch=public['epoch'],source_global_step=public['global_step'],notices=notices)
    training=dict(schema_version=1,language='uk',frontend='piper-uk-text-v1',sample_rate=22050,
                  symbols=symbols(),conditioning_speaker=manifest['speakers'][args.speaker],
                  speaker=args.speaker,utterances=[],rights=rights)
    payload=dict(generator=generator,discriminator=discriminator,model_config=cfg,model_profile='piper-uk-medium-v1',
                 training_config=training,step=public['global_step'],dataset_digest='public-base-no-local-dataset',
                 source_revision='piper:'+manifest['source_revision'],initialization=origin)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.tmp',delete=False) as f:temporary=Path(f.name)
    try:
        torch.save(payload,temporary);os.link(temporary,output)
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps(dict(event='imported',checkpoint=str(output.resolve()),generator_tensors=len(generator),discriminator_tensors=len(discriminator),symbols=256,speaker=args.speaker,model_profile='piper-uk-medium-v1')),flush=True)

if __name__=='__main__':main()

"""Strict import of the pinned original Russian Vosk 0.7 generator checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from vosk_adapter import ROOT, MODEL, PROFILE, FRONTEND, build_model, symbols

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT/'voices/vits-vosk-ru-base/source'))
    parser.add_argument('--output', default=str(ROOT/'voices/vits-vosk-ru-base/base.pt'))
    parser.add_argument('--dictionary', default=str(ROOT/'voices/vosk-ru/dictionary'))
    parser.add_argument('--speaker', choices=['F01','F02','F03','M01','M02'], default='F01')
    args=parser.parse_args(); source=Path(args.source); output=Path(args.output)
    if output.exists(): raise ValueError('base output exists; choose a new destination')
    manifest=json.loads((ROOT/'assets/vits-vosk-ru-assets.json').read_text())
    contents={}
    for name,record in manifest['files'].items():
        if name=='G_1000.pth': continue
        contents[name]=(source/name).read_bytes()
        if len(contents[name])!=record['size'] or hashlib.sha256(contents[name]).hexdigest()!=record['sha256']:
            raise ValueError('source checksum mismatch: '+name)
    source_config=json.loads(contents['config.json'])
    if dict(source_config['model'],n_speakers=source_config['data']['n_speakers'])!=MODEL:
        raise ValueError('unsupported source architecture')
    acoustics=source_config['data']
    if any(acoustics[k]!=v for k,v in dict(sampling_rate=22050,filter_length=1024,hop_length=256,win_length=1024,n_mel_channels=80,use_mel_posterior_encoder=True,add_blank=True).items()) or source_config['train']['segment_size']!=8192:
        raise ValueError('unsupported source acoustics/tokens')
    dictionary=Path(args.dictionary).resolve()
    from checkpoint_tools import sha
    expected=json.loads((ROOT/'assets/russian-assets.json').read_text())['files']['dictionary']['sha256']
    if sha(dictionary)!=expected: raise ValueError('Russian dictionary checksum mismatch')
    torch.set_num_threads(2)
    with (source/'G_1000.pth').open('rb') as stream:
        def digest():
            stream.seek(0); h=hashlib.sha256()
            while chunk:=stream.read(65536): h.update(chunk)
            return h.hexdigest()
        pin=manifest['files']['G_1000.pth']
        if os.fstat(stream.fileno()).st_size!=pin['size'] or digest()!=pin['sha256']:
            raise ValueError('source checkpoint checksum mismatch')
        stream.seek(0); public=torch.load(stream,map_location='cpu',weights_only=True)
        if digest()!=pin['sha256']: raise ValueError('source checkpoint changed while loading')
    generator=public['model']; net=build_model(); net.load_state_dict(generator,strict=True)
    if len(generator)!=818 or any(not torch.isfinite(v).all() for v in generator.values()):
        raise ValueError('invalid source generator tensors')
    notices={name:data.decode() for name,data in contents.items() if name.endswith(('.md','.txt')) or name=='SOURCE-LICENSE'}
    notices['PROVENANCE.json']=json.dumps(manifest,indent=2)
    sid=manifest['speakers'][args.speaker]
    rights=dict(audio='Publisher Apache-2.0 model declaration; source recording/derived voice rights unverified',
                transcripts='Native pinned Vosk pronunciation dictionary; full dictionary rights chain unverified',
                speaker_consent='Five published source voices; complete speaker/derived voice lineage not supplied')
    origin=dict(mode='public-import',repository=manifest['repository'],revision=manifest['revision'],
                weights_sha256=pin['sha256'],license=manifest['license'],license_limit='; '.join(manifest['provenance_limits']),
                frontend_note='Original 62 phoneme indices, dictionary/manual stress and fixed source conditioning retained; corrected normalized inverse-STFT export matches PyTorch',
                discriminator='absent; fresh waveform and duration discriminators on adaptation',
                conditioning_speaker=sid,source_speaker=args.speaker,source_iteration=public['iteration'],notices=notices)
    config=dict(schema_version=1,language='ru',frontend=FRONTEND,sample_rate=22050,symbols=symbols(),
                dictionary=dict(path=str(dictionary),sha256=expected),conditioning_speaker=sid,
                speaker=args.speaker,utterances=[],rights=rights)
    payload=dict(generator=generator,discriminator=None,model_config=MODEL,model_profile=PROFILE,
                 training_config=config,step=0,dataset_digest='public-base-no-local-dataset',
                 source_revision='vosk:'+manifest['source_revision'],initialization=origin)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,suffix='.tmp',delete=False) as f: temporary=Path(f.name)
    try: torch.save(payload,temporary);os.link(temporary,output)
    finally: temporary.unlink(missing_ok=True)
    print(json.dumps(dict(event='imported',checkpoint=str(output.resolve()),generator_tensors=818,symbols=62,speaker=args.speaker,model_profile=PROFILE)),flush=True)

if __name__=='__main__': main()

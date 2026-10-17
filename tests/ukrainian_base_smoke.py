"""Real Ukrainian checkpoint adaptation mechanics; synthetic recordings are not speech."""
import hashlib
import json
from pathlib import Path, PosixPath
import subprocess
import sys
import tempfile
import numpy as np
from scipy.io import wavfile
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'target/public-uk-base-check';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'voices/vits-piper-uk-base/source';BASE=ROOT/'voices/vits-piper-uk-base/base.pt';CLI=ROOT/'target/debug/local-tts'
sys.path.insert(0,str(ROOT/'training'))
from checkpoint_tools import sha, build
from export import Export

def run(*command):subprocess.run(list(map(str,command)),cwd=ROOT,check=True)
def reject(*command):
    p=subprocess.run(list(map(str,command)),cwd=ROOT,capture_output=True,text=True)
    assert p.returncode, p.stdout
    return p.stderr

# Downloads are size/hash bounded and retain previous files on any rejected response.
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_ukrainian_base import download
with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp);out=folder/'cache';out.mkdir();remote=folder/'remote'
    remote.write_bytes(b'new-data');target=out/'weights';target.write_bytes(b'old')
    record=dict(url=remote.as_uri(),sha256=hashlib.sha256(b'new-data').hexdigest(),size=8)
    for bad in [dict(record,size=3),dict(record,sha256='0'*64)]:
        try:download(out,'weights',bad);raise AssertionError('invalid response accepted')
        except ValueError:pass
        assert target.read_bytes()==b'old' and list(out.iterdir())==[target]
    download(out,'weights',record);assert target.read_bytes()==b'new-data'
    remote.unlink();download(out,'weights',record)  # Verified cache does not need network.
torch.set_num_threads(2)
original=sha(BASE)
with torch.serialization.safe_globals([PosixPath]):public=torch.load(SOURCE/'epoch=2090-step=1166778.ckpt',map_location='cpu',weights_only=True,mmap=True)
base=torch.load(BASE,map_location='cpu',weights_only=True)
for group,prefix in [('generator','model_g.'),('discriminator','model_d.')]:
    source={k.removeprefix(prefix):v for k,v in public['state_dict'].items() if k.startswith(prefix)}
    assert set(source)==set(base[group]) and all(torch.equal(v,base[group][k]) for k,v in source.items())
assert base['training_config']['conditioning_speaker']==0 and len(base['training_config']['symbols'])==256
assert base['model_config']['use_sdp'] and base['model_config']['gin_channels']==512
assert 'base output exists' in reject(sys.executable,ROOT/'training/import_piper_uk.py')
# A bad source notice fails before the checkpoint can be loaded or output created.
with tempfile.TemporaryDirectory() as folder:
    bad=Path(folder);(bad/'config.json').write_text('{}')
    assert 'source checksum mismatch' in reject(sys.executable,ROOT/'training/import_piper_uk.py','--source',bad,'--output',bad/'base.pt')
    assert not (bad/'base.pt').exists()
# Select another original conditioning row; preserve all tensors, not an embedding transplant.
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'mykyta.pt'
    run(sys.executable,ROOT/'training/import_piper_uk.py','--source',SOURCE,'--output',path,'--speaker','mykyta')
    other=torch.load(path,map_location='cpu',weights_only=True)
    assert other['training_config']['conditioning_speaker']==1 and other['training_config']['speaker']=='mykyta'
    assert all(torch.equal(v,other['generator'][k]) for k,v in base['generator'].items())
    del other
for name,freq in [('train',220),('validation',330)]:
    wavfile.write(OUT/(name+'.wav'),22050,(np.sin(np.arange(22050)*freq*2*np.pi/22050)*8000).astype(np.int16))
rights=dict(audio='synthetic tones',transcripts='engineering strings',speaker_consent='no human speaker')
data=dict(schema_version=1,language='uk',sample_rate=22050,rights=rights,utterances=[
    dict(id='train',audio='train.wav',text='Прив+іт, сві́те!',speaker='fixture',split='train'),
    dict(id='validation',audio='validation.wav',text='Це украї́нська мо́ва.',speaker='fixture',split='validation')])
(OUT/'dataset.json').write_text(json.dumps(data,ensure_ascii=False))
run_dir=Path(tempfile.mkdtemp(prefix='adapt-',dir=OUT))
command=[sys.executable,ROOT/'training/launch.py','--dataset',OUT/'dataset.json','--output',run_dir,'--batch-size','1']
run(*command,'--init-checkpoint',BASE,'--steps','1')
adapted=torch.load(run_dir/'latest.pt',map_location='cpu',weights_only=True)
assert adapted['step']==1 and adapted['learning_rate']==5e-5 and adapted['model_profile']=='piper-uk-medium-v1'
assert adapted['initialization']['discriminator']=='pretrained' and adapted['initialization']['base_sha256']==original
assert adapted['training_config']['symbols']==base['training_config']['symbols']
assert adapted['training_config']['conditioning_speaker']==0
assert any(not torch.equal(v,adapted['generator'][k]) for k,v in base['generator'].items() if k.startswith('dp.'))
assert any(not torch.equal(v,adapted['generator'][k]) for k,v in base['generator'].items() if k.startswith('dec.'))
assert all(int(s['step'])==1 for s in adapted['optimizer']['state'].values())
# Unused vocabulary rows survive within optimizer weight-decay tolerance.
assert torch.allclose(adapted['generator']['enc_p.emb.weight'][200],base['generator']['enc_p.emb.weight'][200],atol=1e-5)
del adapted,public,source
run(*command,'--resume','--steps','2')
adapted=torch.load(run_dir/'latest.pt',map_location='cpu',weights_only=True)
assert adapted['step']==2 and sha(BASE)==original and adapted['training_config']['conditioning_speaker']==0
run(sys.executable,ROOT/'training/export_and_verify.py','--checkpoint',run_dir/'latest.pt','--output',run_dir/'voice','--new-output')
run(CLI,'vits-synthesize',run_dir/'voice',run_dir/'native.wav','Прив+іт, сві́те! Це украї́нська мо́ва. У мене 12 кни́г.')
rate,raw=wavfile.read(run_dir/'native.wav');assert rate==22050 and raw.size>0
hashes=json.loads((run_dir/'voice/checksums.json').read_text())
assert all(name in hashes for name in ['BASE-SOURCE-LICENSE.txt','BASE-MODEL_CARD','BASE-PROVENANCE.json','BASE-DATASET-LICENSE.txt','training-origin.json'])
# Native text preparation and inference must produce the identical token sequence.
base_voice=OUT/'base-voice'
run(sys.executable,ROOT/'training/export.py','--checkpoint',BASE,'--output',base_voice)
prepared=json.loads((run_dir/'prepared.json').read_text());net=build(base);net.dec.remove_weight_norm();wrapper=Export(net,0).eval()
refs=json.loads((base_voice/'reference.json').read_text())
with torch.no_grad():
    for u in prepared['utterances']:
        ids=u['tokens'];expected=wrapper(torch.tensor([ids]),torch.tensor([len(ids)]),torch.tensor([1.0])).numpy().reshape(-1)
        refs.append(dict(tokens=ids,duration_scale=1.0,samples=expected.tolist()))
(base_voice/'reference.json').write_text(json.dumps(refs))
checks=json.loads((base_voice/'checksums.json').read_text());checks['reference.json']=sha(base_voice/'reference.json');(base_voice/'checksums.json').write_text(json.dumps(checks))
run(CLI,'vits-verify',base_voice)
for u,record in zip(prepared['utterances'],data['utterances']):
    expected=next(r['samples'] for r in refs if r['tokens']==u['tokens'] and r['duration_scale']==1.0)
    path=run_dir/(record['id']+'-base.wav');run(CLI,'vits-synthesize',base_voice,path,record['text'])
    _,actual=wavfile.read(path);expected=np.clip(np.array(expected)*32767,-32768,32767).astype(np.int16)
    assert len(actual)==len(expected) and np.abs(actual.astype(np.int32)-expected.astype(np.int32)).max()<=int(np.ceil(0.001*32767))+1
run(sys.executable,ROOT/'scripts/test_cpp.py',run_dir/'voice','2','fixture','Прив+іт, сві́те!')
run('cargo','test','-p','local-tts-core','--test','piper_base','--','--ignored','--nocapture')
run(sys.executable,ROOT/'training/preflight.py','--checkpoint',BASE,'--dataset',OUT/'dataset.json')
comp=subprocess.run([str(sys.executable),str(ROOT/'training/compare.py'),'--base',str(BASE),'--checkpoint',str(run_dir/'latest.pt'),'--dataset',str(OUT/'dataset.json'),'--output',str(run_dir/'evaluations')],cwd=ROOT,capture_output=True,text=True,check=True)
result=json.loads(comp.stdout.splitlines()[-1]);assert result['report']['frontend']=='piper-uk-text-v1' and len(result['report']['samples'])==1
# A changed conditioning speaker cannot be used with an otherwise matching token inventory.
changed=dict(base,training_config=dict(base['training_config'],conditioning_speaker=2));bad=run_dir/'wrong-conditioning.pt';torch.save(changed,bad)
assert 'conditioning_speaker differs' in reject(sys.executable,ROOT/'training/compare.py','--base',bad,'--checkpoint',run_dir/'latest.pt','--dataset',OUT/'dataset.json','--output',run_dir/'bad-evaluation')
assert not (run_dir/'bad-evaluation').exists()
result_summary=dict(passed=True,base_sha256=original,run=str(run_dir),comparison=result['directory'],source_generator_tensors=len(base['generator']),source_discriminator_tensors=len(base['discriminator']),conditioning_speaker=0,notes='Synthetic engineering recordings do not establish personal voice quality; source rights gaps remain.')
(OUT/'result.json').write_text(json.dumps(result_summary,indent=2))
print(json.dumps(result_summary),flush=True)

"""Original Russian generator adaptation mechanics; synthetic tones are not speech quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.io import wavfile
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'target/public-ru-base-check';OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'voices/vits-vosk-ru-base/base.pt';SOURCE=BASE.parent/'source';CLI=ROOT/'target/debug/local-tts'
sys.path.insert(0,str(ROOT/'training'))
from checkpoint_tools import sha, build
from export import Export
from vosk_adapter import prepare_export
from vendor.vosk.stft import OnnxSTFT

def run(*command): subprocess.run(list(map(str,command)),cwd=ROOT,check=True)
def reject(*command):
    p=subprocess.run(list(map(str,command)),cwd=ROOT,capture_output=True,text=True)
    assert p.returncode,p.stdout
    return p.stderr

torch.set_num_threads(2)
original=sha(BASE)
base=torch.load(BASE,map_location='cpu',weights_only=True)
public=torch.load(SOURCE/'G_1000.pth',map_location='cpu',weights_only=True)
assert len(base['generator'])==818 and set(base['generator'])==set(public['model'])
assert all(torch.equal(v,base['generator'][k]) for k,v in public['model'].items())
assert base['discriminator'] is None and base['training_config']['conditioning_speaker']==0
assert len(base['training_config']['symbols'])==62 and base['generator']['emb_g.weight'].shape==(5,256)
assert 'base output exists' in reject(sys.executable,ROOT/'training/import_vosk_ru.py')
with tempfile.TemporaryDirectory() as folder:
    bad=Path(folder);(bad/'config.json').write_text('{}')
    assert 'source checksum mismatch' in reject(sys.executable,ROOT/'training/import_vosk_ru.py','--source',bad,'--output',bad/'base.pt')
    assert not (bad/'base.pt').exists()
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'M02.pt'
    run(sys.executable,ROOT/'training/import_vosk_ru.py','--output',path,'--speaker','M02')
    other=torch.load(path,map_location='cpu',weights_only=True)
    assert other['training_config']['conditioning_speaker']==4
    assert all(torch.equal(v,other['generator'][k]) for k,v in base['generator'].items())
    del other
del public
for name,freq in [('train',220),('validation',330)]:
    wavfile.write(OUT/(name+'.wav'),22050,(np.sin(np.arange(22050)*freq*2*np.pi/22050)*8000).astype(np.int16))
data=dict(schema_version=1,language='ru',sample_rate=22050,
          rights=dict(audio='synthetic tones',transcripts='engineering strings',speaker_consent='no human speaker'),
          utterances=[dict(id='train',audio='train.wav',text='Прив+ет, м+ир!',speaker='fixture',split='train'),
                      dict(id='validation',audio='validation.wav',text='Это р+усский яз+ык.',speaker='fixture',split='validation')])
(OUT/'dataset.json').write_text(json.dumps(data,ensure_ascii=False))
run_dir=Path(tempfile.mkdtemp(prefix='adapt-',dir=OUT))
command=[sys.executable,ROOT/'training/launch.py','--dataset',OUT/'dataset.json','--output',run_dir,'--batch-size','1']
run(*command,'--init-checkpoint',BASE,'--steps','1')
adapted=torch.load(run_dir/'latest.pt',map_location='cpu',weights_only=True)
assert adapted['step']==1 and adapted['model_profile']=='vosk-ru-multi-v1' and adapted['learning_rate']==5e-5
assert adapted['training_config']['conditioning_speaker']==0 and adapted['training_config']['symbols']==base['training_config']['symbols']
assert adapted['initialization']['base_sha256']==original and adapted['initialization']['discriminator']=='fresh'
assert adapted['duration_discriminator'] and adapted['duration_optimizer']['state']
assert all(int(v['step'])==1 for v in adapted['duration_optimizer']['state'].values())
assert all(int(v['step'])==1 for v in adapted['optimizer']['state'].values())
for prefix in ['enc_p.','flow.','dp.','dec.']:
    assert any(not torch.equal(v,adapted['generator'][k]) for k,v in base['generator'].items() if k.startswith(prefix)),prefix
assert torch.allclose(adapted['generator']['emb_g.weight'][4],base['generator']['emb_g.weight'][4],atol=1e-5)
del adapted
run(*command,'--resume','--steps','2')
adapted=torch.load(run_dir/'latest.pt',map_location='cpu',weights_only=True)
assert adapted['step']==2 and sha(BASE)==original
assert all(int(v['step'])==2 for v in adapted['duration_optimizer']['state'].values())
assert adapted['source_revision'].startswith('vosk:')
run(sys.executable,ROOT/'training/export_and_verify.py','--checkpoint',run_dir/'latest.pt','--output',run_dir/'voice','--new-output')
run(CLI,'vits-synthesize',run_dir/'voice',run_dir/'native.wav','Прив+ет, м+ир! Это р+усский яз+ык. У меня 12 кни́г.')
rate,pcm=wavfile.read(run_dir/'native.wav');assert rate==22050 and pcm.size>0
hashes=json.loads((run_dir/'voice/checksums.json').read_text())
assert all(name in hashes for name in ['dictionary','BASE-SOURCE-LICENSE','BASE-SOURCE-MB-VITS-LICENSE.txt','BASE-SOURCE-PQMF-LICENSE.txt','BASE-SOURCE-STFT-LICENSE.txt','BASE-PROVENANCE.json'])
# Compare the complex training inverse against the real-valued native export,
# including short/long text and all three duration scales at the existing tolerance.
base_voice=OUT/'base-voice'
run(sys.executable,ROOT/'training/export.py','--checkpoint',BASE,'--output',base_voice)
prepared=json.loads((run_dir/'prepared.json').read_text())
net=build(base);net.dec.remove_weight_norm();wrapper=Export(net,0).eval()
refs=json.loads((base_voice/'reference.json').read_text());source_refs=[]
conversion_max=0.0
with torch.no_grad():
    for record in prepared['utterances']:
        ids=record['tokens'];t=torch.tensor([ids]);length=torch.tensor([len(ids)])
        for scale in [0.8,1.0,1.3]:
            # Expected samples use original torch.istft, not the export operation.
            expected=wrapper(t,length,torch.tensor([scale])).numpy().reshape(-1)
            refs.append(dict(tokens=ids,duration_scale=scale,samples=expected.tolist()))
            original_stft=net.dec.stft;prepare_export(net)
            converted=wrapper(t,length,torch.tensor([scale])).numpy().reshape(-1)
            assert len(converted)==len(expected)
            conversion_max=max(conversion_max,float(np.max(np.abs(converted-expected))))
            assert conversion_max < 1e-5
            net.dec.stft=OnnxSTFT(16,4,16)
            publisher=wrapper(t,length,torch.tensor([scale])).numpy().reshape(-1)
            source_refs.append(dict(tokens=ids,duration_scale=scale,samples=publisher.tolist()))
            net.dec.stft=original_stft
(base_voice/'reference.json').write_text(json.dumps(refs))
checks=json.loads((base_voice/'checksums.json').read_text());checks['reference.json']=sha(base_voice/'reference.json');(base_voice/'checksums.json').write_text(json.dumps(checks))
(OUT/'publisher-reference.json').write_text(json.dumps(source_refs))
run(CLI,'vits-verify',base_voice)
for record,u in zip(data['utterances'],prepared['utterances']):
    expected=next(r['samples'] for r in refs if r['tokens']==u['tokens'] and r['duration_scale']==1.0)
    path=run_dir/(record['id']+'-base.wav');run(CLI,'vits-synthesize',base_voice,path,record['text'])
    _,actual=wavfile.read(path);expected=np.clip(np.array(expected)*32767,-32768,32767).astype(np.int16)
    assert len(actual)==len(expected) and np.abs(actual.astype(np.int32)-expected.astype(np.int32)).max()<=int(np.ceil(.001*32767))+1
run('cargo','test','-p','local-tts-core','--test','vosk_base','--','--ignored','--nocapture')
run(sys.executable,ROOT/'scripts/test_cpp.py',run_dir/'voice','1','fixture','Прив+ет, м+ир!')
run(sys.executable,ROOT/'training/preflight.py','--checkpoint',BASE,'--dataset',OUT/'dataset.json')
comp=subprocess.run(list(map(str,[sys.executable,ROOT/'training/compare.py','--base',BASE,'--checkpoint',run_dir/'latest.pt','--dataset',OUT/'dataset.json','--output',run_dir/'evaluations'])),cwd=ROOT,capture_output=True,text=True,check=True)
result=json.loads(comp.stdout.splitlines()[-1]);assert result['report']['frontend']=='vosk-ru-phones-v1' and len(result['report']['samples'])==1
changed=dict(base,training_config=dict(base['training_config'],conditioning_speaker=4));bad=run_dir/'wrong-conditioning.pt';torch.save(changed,bad)
assert 'conditioning_speaker differs' in reject(sys.executable,ROOT/'training/compare.py','--base',bad,'--checkpoint',run_dir/'latest.pt','--dataset',OUT/'dataset.json','--output',run_dir/'bad-evaluation')
assert not (run_dir/'bad-evaluation').exists()
# Dictionary damage is rejected even if package integrity is resealed locally.
with tempfile.TemporaryDirectory() as folder:
    wrong=Path(folder)/'dictionary';wrong.write_text('bad')
    config=dict(base['training_config'],dictionary=dict(path=str(wrong),sha256=sha(wrong)))
    try: build(dict(base,training_config=config));raise AssertionError('unrecognized dictionary accepted')
    except ValueError as e: assert 'dictionary checksum mismatch' in str(e)
summary=dict(passed=True,base_sha256=original,run=str(run_dir),comparison=result['directory'],
             source_generator_tensors=818,conditioning_speaker=0,inverse_conversion_max_error=conversion_max,
             notes='Synthetic engineering recordings do not establish voice quality. Dictionary/data/consent gaps remain. Publisher ONNX has an unnormalized inverse; project export matches PyTorch.')
(OUT/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

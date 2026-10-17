"""Opt-in real public weights test. Synthetic clips check plumbing, not voice quality."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.io import wavfile
import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'target/public-base-check'
OUT.mkdir(parents=True, exist_ok=True)
base = ROOT / 'voices/vits-ljs-base/base.pt'
source = ROOT / 'voices/vits-ljs-base/source'
cli = ROOT / 'target/debug/local-tts'

def run(*args): subprocess.run(list(map(str, args)), cwd=ROOT, check=True)
def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(65536): result.update(chunk)
    return result.hexdigest()
original = sha(base)
public = torch.load(source / 'pretrained_ljs.pth', map_location='cpu', weights_only=True)
imported = torch.load(base, map_location='cpu', weights_only=True)
assert set(public['model']) == set(imported['generator'])
assert all(torch.equal(v, imported['generator'][k]) for k, v in public['model'].items())
assert imported['model_config']['use_sdp'] and imported['discriminator'] is None
assert len(imported['training_config']['symbols']) == 178
assert imported['training_config']['symbols'].count("'") == 2
# Exclusive import cannot replace a checkpoint.
rejected = subprocess.run([sys.executable, 'training/import_ljs.py'], cwd=ROOT, capture_output=True, text=True)
assert rejected.returncode and 'base output exists' in rejected.stderr and sha(base) == original
# Changed downloaded source fails before deserialization.
import tempfile
with tempfile.TemporaryDirectory() as temp:
    bad = Path(temp)
    (bad / 'ljs-config.json').write_text('{}')
    rejected = subprocess.run([sys.executable, 'training/import_ljs.py', '--source', str(bad), '--output', str(bad / 'base.pt')], cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode and 'source checksum mismatch' in rejected.stderr
    assert not (bad / 'base.pt').exists()
for name, freq in [('train', 220), ('validation', 330)]:
    raw = (8000 * np.sin(np.arange(22050) * freq * 2 * np.pi / 22050)).astype(np.int16)
    wavfile.write(OUT / (name + '.wav'), 22050, raw)
rights = dict(audio='synthetic tones', transcripts='engineering test', speaker_consent='no human speaker')
data = dict(schema_version=1, language='en', sample_rate=22050, rights=rights,
            utterances=[dict(id=name, audio=name+'.wav', text='Hello world.' if name=='train' else 'The cat sat.', speaker='fixture', split=name) for name in ['train', 'validation']])
(OUT / 'dataset.json').write_text(json.dumps(data))
# Fresh directory allows repeated runs while keeping prior evidence.
import tempfile
run_dir = Path(tempfile.mkdtemp(prefix='adapt-', dir=OUT))
command = [sys.executable, ROOT / 'training/launch.py', '--dataset', OUT / 'dataset.json', '--output', run_dir, '--batch-size', '1']
run(*command, '--init-checkpoint', base, '--steps', '1')
adapted = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=False)
assert adapted['step'] == 1 and adapted['learning_rate'] == 5e-5
assert adapted['model_profile'] == 'ljs-sdp-v1'
assert adapted['initialization']['discriminator'] == 'fresh'
assert adapted['initialization']['base_sha256'] == original
assert adapted['training_config']['symbols'] == imported['training_config']['symbols']
# Unused alias embedding rows are kept; active generator/duration weights are updated.
assert torch.allclose(adapted['generator']['enc_p.emb.weight'][174], imported['generator']['enc_p.emb.weight'][174], atol=1e-5)
assert any(not torch.equal(v, adapted['generator'][k]) for k,v in imported['generator'].items() if k.startswith('dp.'))
assert all(int(s['step']) == 1 for s in adapted['optimizer']['state'].values())
del adapted, public, imported
run(*command, '--resume', '--steps', '2')
adapted = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=False)
assert adapted['step'] == 2 and adapted['model_config']['use_sdp']
assert adapted['initialization']['base_sha256'] == original and sha(base) == original
assert adapted['training_config']['dictionary']['sha256'] == sha(Path(adapted['training_config']['dictionary']['path']))
del adapted
voice = run_dir / 'voice'
run(sys.executable, ROOT / 'training/export_and_verify.py', '--checkpoint', run_dir/'latest.pt', '--output', voice)
run(cli, 'vits-synthesize', voice, run_dir / 'native.wav', 'Hello world. The cat sat. I have 12 apples.')
rate, raw = wavfile.read(run_dir/'native.wav')
assert rate == 22050 and len(raw) > 0
hashes = json.loads((voice/'checksums.json').read_text())
assert 'cmudict.dict' in hashes and 'CMU-LICENSE.txt' in hashes and 'BASE-MODEL_CARD.md' in hashes
assert json.loads((voice/'training-origin.json').read_text())['base_initialization']['mode'] == 'public-import'
dictionary = voice/'cmudict.dict'; original_dict = dictionary.read_bytes()
dictionary.write_bytes(original_dict+b'\nchanged')
rejected = subprocess.run([cli, 'vits-verify', voice], capture_output=True, text=True)
assert rejected.returncode and 'checksum mismatch for cmudict.dict' in rejected.stderr
dictionary.write_bytes(original_dict)
# Assert native text tokens exactly equal prepared training tokens against unadapted base.
config = json.loads((run_dir/'prepared.json').read_text())
run(sys.executable, ROOT/'training/export.py', '--checkpoint', base, '--output', OUT/'base-voice')
# Append the exact prepared sequences to PyTorch fixtures and verify in Rust.
sys.path.insert(0,str(ROOT/'training'))
from export import Export, SynthesizerTrn
saved = torch.load(base, map_location='cpu', weights_only=True)
net = SynthesizerTrn(178,513,32,**saved['model_config']);net.load_state_dict(saved['generator']);net.eval();net.dec.remove_weight_norm()
wrapper = Export(net).eval()
refs = json.loads((OUT/'base-voice/reference.json').read_text())
with torch.no_grad():
    for rec in config['utterances']:
        ids=rec['tokens']; scale=1.0
        raw=wrapper(torch.tensor([ids]),torch.tensor([len(ids)]),torch.tensor([scale])).numpy().reshape(-1)
        refs.append(dict(tokens=ids,duration_scale=scale,samples=raw.tolist()))
(OUT/'base-voice/reference.json').write_text(json.dumps(refs))
hashes=json.loads((OUT/'base-voice/checksums.json').read_text());hashes['reference.json']=sha(OUT/'base-voice/reference.json')
(OUT/'base-voice/checksums.json').write_text(json.dumps(hashes))
run(cli,'vits-verify',OUT/'base-voice')
run(cli,'vits-synthesize',OUT/'base-voice',OUT/'base-native.wav','Hello world.')
_,native=wavfile.read(OUT/'base-native.wav')
reference=np.array(refs[-2]['samples'])
expected=np.clip(reference*32767,-32768,32767).astype(np.int16)
assert len(native)==len(expected) and np.max(np.abs(native.astype(int)-expected.astype(int))) < 16
(OUT/'result.json').write_text(json.dumps(dict(passed=True,run=str(run_dir),base_sha256=original,native_max_pcm_error=int(np.max(np.abs(native.astype(int)-expected.astype(int))))),indent=2))
print('Public base: all generator tensors preserved; strict import, fine-tuning, SDP updates, resume, native parity, frontend agreement, notices and tamper rejection passed')

"""Engineering-only training fixture: sine waves are not speech or a usable voice."""
import json
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile

root = Path(__file__).resolve().parents[1]
import sys
python = Path(sys.executable)
parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['en', 'ru', 'uk', 'fr', 'de', 'es'], default='en')
language = parser.parse_args().language
prompts = {'en': ['a', 'b'], 'ru': ['а́', 'ё'], 'uk': ['ґ', 'ї́'], 'fr': ['é', 'œ'], 'de': ['ä', 'ß'], 'es': ['ñ', 'ü']}[language]
with tempfile.TemporaryDirectory(prefix='tts-train-smoke-') as temp:
    folder = Path(temp)
    for name, freq in [('train', 220), ('validation', 330)]:
        audio = (np.sin(np.arange(22050) * 2 * np.pi * freq / 22050) * 8000).astype(np.int16)
        wavfile.write(folder / f'{name}.wav', 22050, audio)
    data = dict(schema_version=1, language=language, sample_rate=22050,
                rights=dict(audio='synthetic engineering fixture', transcripts='test strings', speaker_consent='no human speaker'),
                utterances=[dict(id=name, audio=f'{name}.wav', text=prompts[0] if name == 'train' else prompts[1], speaker='fixture', split=name) for name in ['train', 'validation']])
    (folder / 'dataset.json').write_text(json.dumps(data))
    subprocess.run([str(root / 'target/debug/local-tts'), 'training-prepare', str(folder / 'dataset.json'), str(folder / 'prepared.json')], check=True)
    common = [str(python), str(root / 'training/train.py'), '--prepared', str(folder / 'prepared.json'), '--output', str(folder / 'run'), '--batch-size', '1', '--tiny']
    subprocess.run(common + ['--steps', '1', '--from-scratch'], check=True)
    subprocess.run(common + ['--steps', '2', '--resume'], check=True)
    import torch
    saved = torch.load(folder / 'run/latest.pt', map_location='cpu', weights_only=False)
    assert saved['step'] == 2
    # Stop is graceful: wait for a real update, terminate, then verify progress is saved.
    process = subprocess.Popen(common + ['--steps', '100', '--resume'], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        if '"event": "step"' in line:
            process.terminate()
            break
    process.communicate(timeout=60)
    assert process.returncode == 0
    stopped = torch.load(folder / 'run/latest.pt', map_location='cpu', weights_only=False)
    assert 2 < stopped['step'] < 100
    original_audio = (folder / 'train.wav').read_bytes()
    (folder / 'train.wav').write_bytes(original_audio + b'changed')
    rejected = subprocess.run(common + ['--steps', '100', '--resume'], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'recording changed after preparation' in rejected.stderr
    (folder / 'train.wav').write_bytes(original_audio)
    assert (folder / 'run/preview.wav').is_file()
    subprocess.run([str(python), str(root / 'training/export.py'), '--checkpoint', str(folder / 'run/latest.pt'), '--output', str(folder / 'voice')], check=True)
    subprocess.run([str(root / 'target/debug/local-tts'), 'vits-verify', str(folder / 'voice')], check=True)
    subprocess.run([str(root / 'target/debug/local-tts'), 'vits-synthesize', str(folder / 'voice'), str(folder / 'native.wav'), ''.join(prompts)], check=True)
    rate, audio = wavfile.read(folder / 'native.wav')
    assert rate == 22050 and len(audio) > 0
    if language in ['ru', 'uk']:
        import re
        alternate = re.sub(r'(.)\u0301', r'+\1', ''.join(prompts))
        subprocess.run([str(root / 'target/debug/local-tts'), 'vits-synthesize', str(folder / 'voice'), str(folder / 'alternate.wav'), alternate], check=True)
        assert (folder / 'alternate.wav').read_bytes() == (folder / 'native.wav').read_bytes()
    # Adapt a different voice/dataset, preserving even base symbols absent in its text.
    target = dict(data)
    target['utterances'] = [dict(item, text=prompts[1], speaker='new-voice') for item in data['utterances']]
    (folder / 'target.json').write_text(json.dumps(target))
    adapt = [str(python), str(root / 'training/launch.py'), '--dataset', str(folder / 'target.json'),
             '--output', str(folder / 'adapt'), '--batch-size', '1', '--tiny']
    base_path = folder / 'run/latest.pt'
    original_base = base_path.read_bytes()
    subprocess.run(adapt + ['--steps', '1', '--init-checkpoint', str(base_path)], check=True)
    adapted = torch.load(folder / 'adapt/latest.pt', map_location='cpu', weights_only=False)
    assert adapted['step'] == 1 and adapted['learning_rate'] == 5e-5
    assert adapted['training_config']['symbols'] == stopped['training_config']['symbols']
    assert adapted['training_config']['speaker'] == 'new-voice'
    assert adapted['dataset_digest'] != stopped['dataset_digest']
    assert adapted['initialization']['mode'] == 'fine-tune'
    import hashlib
    assert adapted['initialization']['base_sha256'] == hashlib.sha256(original_base).hexdigest()
    # The unused base embedding row is inherited, not randomly reinitialized.
    unused = stopped['training_config']['symbols'].index(prompts[0][0])
    assert torch.allclose(adapted['generator']['enc_p.emb.weight'][unused],
                          stopped['generator']['enc_p.emb.weight'][unused], atol=1e-5, rtol=1e-5)
    assert base_path.read_bytes() == original_base
    assert all(int(state['step']) == 1 for state in adapted['optimizer']['state'].values())
    subprocess.run(adapt + ['--steps', '2', '--resume'], check=True)
    adapted = torch.load(folder / 'adapt/latest.pt', map_location='cpu', weights_only=False)
    assert adapted['step'] == 2 and adapted['initialization']['base_sha256'] == hashlib.sha256(original_base).hexdigest()
    assert adapted['learning_rate'] == 5e-5
    subprocess.run([str(python), str(root / 'training/export.py'), '--checkpoint', str(folder / 'adapt/latest.pt'), '--output', str(folder / 'adapt/voice')], check=True)
    subprocess.run([str(root / 'target/debug/local-tts'), 'vits-verify', str(folder / 'adapt/voice')], check=True)
    subprocess.run([str(root / 'target/debug/local-tts'), 'vits-synthesize', str(folder / 'adapt/voice'), str(folder / 'adapt/native.wav'), ''.join(prompts)], check=True)
    origin = json.loads((folder / 'adapt/voice/training-origin.json').read_text())
    assert origin['base_sha256'] == hashlib.sha256(original_base).hexdigest()
    assert 'fine-tune' in json.loads((folder / 'adapt/voice/voice.json').read_text())['provenance']
    # Preflight is read-only; comparisons synthesize only held-out clips natively.
    preflight = subprocess.run([str(python), str(root/'training/preflight.py'), '--checkpoint', str(base_path), '--dataset', str(folder/'target.json')], capture_output=True, text=True, check=True)
    inspection = json.loads(preflight.stdout.splitlines()[-1])
    assert inspection['language'] == language and inspection['target_speaker'] == 'new-voice'
    assert inspection['clips'] == dict(train=1,validation=1,test=0)
    assert inspection['sha256'] == hashlib.sha256(original_base).hexdigest()
    assert base_path.read_bytes() == original_base
    comparison = subprocess.run([str(python), str(root/'training/compare.py'), '--base', str(base_path), '--checkpoint', str(folder/'adapt/latest.pt'), '--dataset', str(folder/'target.json'), '--output', str(folder/'evaluations'), '--limit', '3'], capture_output=True, text=True, check=True)
    result = json.loads(comparison.stdout.splitlines()[-1]); samples = Path(result['directory'])
    assert result['report']['split'] == 'validation' and len(result['report']['samples']) == 1
    sample = result['report']['samples'][0]
    assert sample['id'] == 'validation' and sample['text'] == prompts[1]
    assert (samples/sample['reference']['file']).read_bytes() == (folder/'validation.wav').read_bytes()
    for role in ['reference','before','after']:
        file=sample[role]
        assert hashlib.sha256((samples/file['file']).read_bytes()).hexdigest() == file['sha256']
        assert file['seconds'] > 0 and np.isfinite(file['rms'])
    assert not any(samples.parent.glob('.work-*'))
    metadata = json.loads((samples.parent/'comparison.json').read_text())
    assert metadata['base']['sha256'] == hashlib.sha256(original_base).hexdigest()
    assert metadata['target']['sha256'] == hashlib.sha256((folder/'adapt/latest.pt').read_bytes()).hexdigest()
    assert metadata['report_sha256'] == hashlib.sha256((samples/'report.json').read_bytes()).hexdigest()
    # Export the evaluated snapshot, not whichever latest file happens to exist.
    evaluated_hash = metadata['target']['sha256']
    evaluated_voice = samples.parent/'exported-voice'
    export_command = [str(python),str(root/'training/export_and_verify.py'),'--checkpoint',str(folder/'adapt/latest.pt'),'--output',str(evaluated_voice),'--expected-sha256',evaluated_hash,'--new-output']
    rejected = subprocess.run(export_command[:-3]+['--expected-sha256','0'*64,'--new-output'],capture_output=True,text=True)
    assert rejected.returncode and 'changed since evaluation' in rejected.stderr
    assert not evaluated_voice.exists()
    exported = subprocess.run(export_command,capture_output=True,text=True,check=True)
    event = json.loads(exported.stdout.splitlines()[-1])
    assert event == dict(event='evaluated_export',voice=str(evaluated_voice.resolve()),checkpoint_sha256=evaluated_hash)
    identity = json.loads((evaluated_voice/'exported-checkpoint.json').read_text())
    assert identity['sha256'] == evaluated_hash and identity['step'] == 2
    checksums = json.loads((evaluated_voice/'checksums.json').read_text())
    assert checksums['exported-checkpoint.json'] == hashlib.sha256((evaluated_voice/'exported-checkpoint.json').read_bytes()).hexdigest()
    preserved = (evaluated_voice/'model.onnx').read_bytes()
    rejected = subprocess.run(export_command,capture_output=True,text=True)
    assert rejected.returncode and 'File exists' in rejected.stderr
    assert (evaluated_voice/'model.onnx').read_bytes() == preserved
    # Atomic replacement by a resumed training run invalidates the old evaluation.
    subprocess.run(adapt+['--steps','3','--resume'],check=True)
    stale_command = export_command.copy()
    stale_command[stale_command.index('--output')+1] = str(folder/'stale-export')
    rejected = subprocess.run(stale_command,capture_output=True,text=True)
    assert rejected.returncode and 'changed since evaluation' in rejected.stderr
    assert not (folder/'stale-export').exists()
    print(f'{language}: evaluated export hash binding, native parity, provenance and exclusive destination passed')
    # Repeat the native model path and require deterministic audio.
    again=folder/'comparison-again'
    subprocess.run([str(root/'target/debug/local-tts'),'vits-compare',str(folder/'voice'),str(folder/'adapt/voice'),str(folder/'target.json'),str(again),'validation','1'],check=True)
    for role in ['before','after']:
        assert (again/sample[role]['file']).read_bytes() == (samples/sample[role]['file']).read_bytes()
    rejected=subprocess.run([str(root/'target/debug/local-tts'),'vits-compare',str(folder/'voice'),str(folder/'adapt/voice'),str(folder/'target.json'),str(again),'train','1'],capture_output=True,text=True)
    assert rejected.returncode and 'validation or test' in rejected.stderr
    # Evaluation cannot replace prior samples or relabel a changed dataset.
    rejected=subprocess.run([str(root/'target/debug/local-tts'),'vits-compare',str(folder/'voice'),str(folder/'adapt/voice'),str(folder/'target.json'),str(again),'validation','1'],capture_output=True,text=True)
    assert rejected.returncode and 'new directory' in rejected.stderr
    changed=dict(target,rights=dict(target['rights'],transcripts='changed metadata'))
    (folder/'changed.json').write_text(json.dumps(changed))
    rejected=subprocess.run([str(python),str(root/'training/compare.py'),'--base',str(base_path),'--checkpoint',str(folder/'adapt/latest.pt'),'--dataset',str(folder/'changed.json'),'--output',str(folder/'wrong-evaluation')],capture_output=True,text=True)
    assert rejected.returncode and 'evaluation dataset differs' in rejected.stderr
    assert not (folder/'wrong-evaluation').exists()
    if language=='en':
        # Project preflight rejects pickle instructions rather than executing them.
        sentinel=folder/'should-never-exist'
        class Unsafe:
            def __reduce__(self):
                return (eval,("__import__('pathlib').Path("+repr(str(sentinel))+").write_text('executed')",))
        torch.save(Unsafe(),folder/'unsafe.pt')
        rejected=subprocess.run([str(python),str(root/'training/preflight.py'),'--checkpoint',str(folder/'unsafe.pt'),'--dataset',str(folder/'target.json')],capture_output=True,text=True)
        assert rejected.returncode and not sentinel.exists()
    print(f'{language}: preflight, held-out native comparison, deterministic audio, metadata binding, exclusive output and restricted loading passed')
    # New symbols/language/architecture are rejected rather than partially loading.
    invalid = dict(target)
    invalid['utterances'] = [dict(item, text='z' if language == 'en' else 'я') for item in target['utterances']]
    (folder / 'invalid.json').write_text(json.dumps(invalid))
    rejected = subprocess.run(adapt[:2] + ['--dataset', str(folder / 'invalid.json'), '--output', str(folder / 'invalid'), '--init-checkpoint', str(base_path), '--steps', '1', '--tiny'], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'absent from base token inventory' in rejected.stderr
    assert not (folder / 'invalid/latest.pt').exists()
    prepared_target = json.loads((folder / 'adapt/prepared.json').read_text())
    prepared_target['language'] = 'uk' if language != 'uk' else 'ru'
    (folder / 'wrong-language.json').write_text(json.dumps(prepared_target))
    rejected = subprocess.run([str(python), str(root / 'training/train.py'), '--prepared', str(folder / 'wrong-language.json'), '--output', str(folder / 'wrong'), '--steps', '1', '--tiny', '--init-checkpoint', str(base_path)], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'base language differs' in rejected.stderr
    rejected = subprocess.run([str(python), str(root / 'training/train.py'), '--prepared', str(folder / 'adapt/prepared.json'), '--output', str(folder / 'architecture'), '--steps', '1', '--init-checkpoint', str(base_path)], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'architecture differs' in rejected.stderr
    print(f'{language}: fine-tuning, fixed inventory, fresh optimizer, resume, base retention, lineage and native export passed')
    # Loader must reject changed token inventories before attempting inference.
    token_path = folder / 'voice/tokens.json'
    original_tokens = token_path.read_bytes()
    token_path.write_bytes(original_tokens + b' ')
    rejected = subprocess.run([str(root / 'target/debug/local-tts'), 'vits-verify', str(folder / 'voice')], capture_output=True, text=True)
    assert rejected.returncode != 0 and 'checksum mismatch for tokens.json' in rejected.stderr
    token_path.write_bytes(original_tokens)
    print(f'{language}: training, resume, preview, export, native text inference, parity and tamper detection passed')

"""Fine-tune real CSS10 weights on synthetic fixtures; no voice-quality claim."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from checkpoint_tools import build, sha
from css10_adapter import PROFILE, FRONTEND


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def reject(*args):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode, result.stdout
    return result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es', 'fr', 'it', 'pt', 'pl', 'nl'])
    args = parser.parse_args(); language = args.language
    torch.set_num_threads(2)
    neon = language not in ['de', 'es']
    family = 'neon' if neon else 'css10'
    profile = 'neon-european-v1' if neon else PROFILE
    frontend = 'neon-graphemes-v1' if neon else FRONTEND
    base_path = ROOT / f'voices/vits-{family}-{language}-base/base.pt'
    original_hash = sha(base_path)
    base = torch.load(base_path, weights_only=True, map_location='cpu')
    assert base['model_profile'] == profile and len(base['generator']) == 696
    assert base['discriminator'] is None
    build(base)
    assert 'base output exists' in reject(sys.executable, ROOT / f'training/import_{family}.py', language, '--output', base_path)
    for field, value in [('language', 'en'), ('conditioning_speaker', 1), ('frontend', 'grapheme-v2')]:
        bad = copy.deepcopy(base); bad['training_config'][field] = value
        try:
            build(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid {field}')
    bad = copy.deepcopy(base)
    bad['training_config']['symbols'][3], bad['training_config']['symbols'][4] = bad['training_config']['symbols'][4], bad['training_config']['symbols'][3]
    try:
        build(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted changed source token IDs')
    del bad
    out = Path(tempfile.mkdtemp(prefix=f'{family}-{language}-adapt-', dir=ROOT / 'target'))
    phrases = {'de': ['guten morgen.', 'die straße führt über eine brücke.'], 'es': ['buenos días.', 'el niño está cerca del río.'], 'fr': ['bonjour.', 'le garçon est près du café.'], 'it': ['buongiorno.', 'la città è già sveglia.'], 'pt': ['bom dia.', 'o coração está tranquilo.'], 'pl': ['dzień dobry.', 'łódź płynie przez rzekę.'], 'nl': ['goedemorgen.', 'het meisje fietst naar het café.']}[language]
    for name, frequency in [('train', 220), ('validation', 330)]:
        samples = (np.sin(np.arange(22050 * 3) * frequency * 2 * np.pi / 22050) * 8000).astype(np.int16)
        wavfile.write(out / (name + '.wav'), 22050, samples)
    dataset = dict(schema_version=1, language=language, sample_rate=22050,
        rights=dict(audio='Synthetic tones', transcripts='Original engineering fixtures', speaker_consent='No human speaker'),
        utterances=[dict(id=split, audio=split + '.wav', text=text, speaker='fixture', split=split)
                    for split, text in zip(['train', 'validation'], phrases)])
    manifest = out / 'dataset.json'; manifest.write_text(json.dumps(dataset, ensure_ascii=False))
    cli = ROOT / 'target/debug/local-tts'
    run(sys.executable, ROOT / 'training/preflight.py', '--checkpoint', base_path, '--dataset', manifest)
    command = [sys.executable, ROOT / 'training/launch.py', '--dataset', manifest, '--output', out / 'adapt', '--batch-size', '1']
    run(*command, '--init-checkpoint', base_path, '--steps', '1')
    adapted_path = out / 'adapt/latest.pt'
    adapted = torch.load(adapted_path, weights_only=True, map_location='cpu')
    assert adapted['step'] == 1 and adapted['learning_rate'] == 5e-5
    assert adapted['initialization']['base_sha256'] == original_hash
    assert adapted['initialization']['discriminator'] == 'fresh'
    assert adapted['training_config']['symbols'] == base['training_config']['symbols']
    assert adapted['training_config']['frontend'] == frontend
    assert all(int(v['step']) == 1 for v in adapted['optimizer']['state'].values())
    for prefix in ['dp.', 'enc_p.language.', 'dec.']:
        assert any(not torch.equal(v, adapted['generator'][k]) for k, v in base['generator'].items() if k.startswith(prefix)), prefix
    del adapted
    run(*command, '--resume', '--steps', '2')
    adapted = torch.load(adapted_path, weights_only=True, map_location='cpu')
    assert adapted['step'] == 2 and sha(base_path) == original_hash
    voice = out / 'voice'
    run(sys.executable, ROOT / 'training/export_and_verify.py', '--checkpoint', adapted_path, '--output', voice, '--new-output')
    run(cli, 'vits-synthesize', voice, out / 'native.wav', phrases[1])
    rate, pcm = wavfile.read(out / 'native.wav')
    assert rate == 22050 and pcm.size > 1000 and np.max(np.abs(pcm.astype(np.float64))) > 32
    # The native text path and prepared-token path must agree at PCM precision.
    from export import Export
    model = build(adapted); model.dec.remove_weight_norm(); wrapper = Export(model, 0).eval()
    prepared = json.loads((out / 'adapt/prepared.json').read_text())
    ids = prepared['utterances'][1]['tokens']
    assert ids[0] == ids[-1] == 3 and all(i == 3 for i in ids[::2]) and 0 not in ids
    with torch.no_grad():
        expected = wrapper(torch.tensor([ids]), torch.tensor([len(ids)]), torch.tensor([1.0])).numpy().reshape(-1)
    expected = np.clip(expected * 32767, -32768, 32767).astype(np.int16)
    assert pcm.shape == expected.shape
    error = np.abs(pcm.astype(np.int32) - expected.astype(np.int32)).max()
    assert error <= 34, error
    hashes = json.loads((voice / 'checksums.json').read_text())
    assert 'BASE-PROVENANCE.json' in hashes and 'BASE-README.md' in hashes
    run(sys.executable, ROOT / 'scripts/test_cpp.py', voice, str({'de':3, 'fr':4, 'es':5, 'it':6, 'pt':7, 'pl':8, 'nl':9}[language]), 'fixture', phrases[1])
    comparison = subprocess.run(list(map(str, [sys.executable, ROOT / 'training/compare.py', '--base', base_path,
        '--checkpoint', adapted_path, '--dataset', manifest, '--output', out / 'evaluations', '--limit', '1'])),
        cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(comparison.stdout.splitlines()[-1])
    assert result['report']['frontend'] == frontend and len(result['report']['samples']) == 1
    summary = dict(passed=True, language=language, base_sha256=original_hash, run=str(out),
                   comparison=result['directory'], native_pcm_max_error=int(error),
                   synthetic_fixtures=True, native_speaker_quality_evaluated=False)
    (ROOT / f'target/{family}-{language}-training.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

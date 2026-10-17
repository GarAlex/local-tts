"""Engineering-only numeric text/fine-tuning/export parity, using synthetic tones."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'target/debug/local-tts'
sys.path.insert(0, str(ROOT / 'training'))
from checkpoint_tools import load, build
from export import Export
torch.set_num_threads(2)

def run(*arguments):
    subprocess.run([str(argument) for argument in arguments], cwd=ROOT, check=True)

metrics = []
for language in ['en', 'ru', 'uk']:
    with tempfile.TemporaryDirectory(prefix=f'language-{language}-', dir=ROOT / 'target') as temporary:
        folder = Path(temporary)
        prompts = ['2026-10-03', '2.5 kg' if language == 'en' else '2,5 кг']
        records = []
        for index, split in enumerate(['train', 'validation']):
            audio = (np.sin(np.arange(22050 * 4) * 2 * np.pi * (220 + index * 110) / 22050) * 8000).astype(np.int16)
            wavfile.write(folder / f'{split}.wav', 22050, audio)
            records.append(dict(id=split, audio=f'{split}.wav', text=prompts[index], speaker='fixture', split=split))
        dataset = dict(schema_version=1, language=language, sample_rate=22050,
                       rights=dict(audio='Synthetic engineering tones', transcripts='Original test notation', speaker_consent='No human speaker'), utterances=records)
        manifest = folder / 'dataset.json'
        manifest.write_text(json.dumps(dataset, ensure_ascii=False))
        prepared = folder / 'prepared.json'
        run(CLI, 'training-prepare', manifest, prepared)
        config = json.loads(prepared.read_text())
        assert config['normalization'] == 'text-v1'
        run(sys.executable, ROOT / 'training/train.py', '--prepared', prepared, '--output', folder / 'base', '--from-scratch', '--tiny', '--steps', '1', '--batch-size', '1')
        for record in dataset['utterances']: record['speaker'] = 'adapted-fixture'
        manifest.write_text(json.dumps(dataset, ensure_ascii=False))
        run(sys.executable, ROOT / 'training/launch.py', '--dataset', manifest, '--output', folder / 'adapt', '--init-checkpoint', folder / 'base/latest.pt', '--tiny', '--steps', '1', '--batch-size', '1')
        checkpoint, _ = load(folder / 'adapt/latest.pt')
        target = checkpoint['training_config']
        assert target['normalization'] == 'text-v1'
        assert checkpoint['initialization']['mode'] == 'fine-tune'
        voice = folder / 'voice'
        run(sys.executable, ROOT / 'training/export.py', '--checkpoint', folder / 'adapt/latest.pt', '--output', voice)
        assert json.loads((voice / 'voice.json').read_text())['normalization'] == 'text-v1'
        run(CLI, 'vits-verify', voice)
        wrapper = Export(build(checkpoint), target.get('conditioning_speaker')).eval()
        for index, prompt in enumerate(prompts):
            tokens = target['utterances'][index]['tokens']
            output = folder / f'native-{index}.wav'
            run(CLI, 'vits-synthesize', voice, output, prompt)
            with torch.no_grad():
                reference = wrapper(torch.tensor([tokens]), torch.tensor([len(tokens)]), torch.tensor([1.0])).numpy().reshape(-1)
            rate, native = wavfile.read(output)
            expected = (np.clip(reference, -1, 1) * 32767).astype(np.int16)
            assert rate == 22050 and native.shape == expected.shape
            max_pcm_error = int(np.max(np.abs(native.astype(np.int32) - expected.astype(np.int32))))
            assert max_pcm_error <= 33, (language, prompt, max_pcm_error)
            metrics.append(dict(language=language, text=prompt, samples=len(native), max_pcm_error=max_pcm_error, normalization='text-v1', mode='fine-tune'))
        # Version metadata is validated before model use in both runtimes.
        invalid = dict(checkpoint, training_config=dict(target, normalization='text-v999'))
        try: build(invalid)
        except ValueError as error: assert 'normalization' in str(error)
        else: raise AssertionError('unknown normalization version accepted')
report = dict(passed=True, note='Synthetic tone fixtures verify token/training/export contracts, not speech quality.', cases=metrics)
(ROOT / 'target/language-training-metrics.json').write_text(json.dumps(report, indent=2))
print('Numeric EN/RU/UK fine-tuning, export metadata and native/PyTorch text-token parity passed', flush=True)

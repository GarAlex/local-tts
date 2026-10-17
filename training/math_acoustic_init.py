"""Warm a phoneme student from compatible acoustic weights and a decoder probe."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from checkpoint_tools import load, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phoneme-base', required=True)
    parser.add_argument('--acoustic-base', required=True)
    parser.add_argument('--decoder-probe', required=True)
    parser.add_argument('--probe-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): parser.error('output already exists')
    torch.set_num_threads(2)
    phoneme, phoneme_sha = load(args.phoneme_base)
    acoustic, acoustic_sha = load(args.acoustic_base)
    if phoneme.get('model_profile') != 'math-phoneme-v1' or acoustic.get('model_profile') != 'math-student-v1':
        raise ValueError('requires compatible phoneme and grapheme arithmetic profiles')
    if phoneme['model_config'] != acoustic['model_config']:
        raise ValueError('acoustic architectures differ')
    probe = Path(args.decoder_probe)
    report = json.loads((probe / 'report.json').read_text())
    probe_source, _ = load(args.probe_source, report.get('checkpoint_sha256'))
    if probe_source['generator'].keys() != acoustic['generator'].keys() or any(
            not torch.equal(value, acoustic['generator'][key])
            for key, value in probe_source['generator'].items()):
        raise ValueError('probe acoustic weights differ from transfer source')
    if report.get('split') != 'train' or report.get('teacher_audio_sha256') not in {
            r['sha256'] for r in phoneme['training_config']['utterances'] if r['split'] == 'train'}:
        raise ValueError('invalid reconstruction probe provenance')
    decoder_path = probe / 'decoder-only.pt'
    decoder = torch.load(decoder_path, map_location='cpu', weights_only=True)
    model = build(phoneme)
    base = build(acoustic)
    model.enc_q.load_state_dict(base.enc_q.state_dict(), strict=True)
    model.flow.load_state_dict(base.flow.state_dict(), strict=True)
    model.dec.load_state_dict(decoder, strict=True)
    phoneme['generator'] = model.state_dict()
    # Validate the assembled finite weights before writing anything.
    build(phoneme)
    if 'discriminator' in acoustic:
        phoneme['discriminator'] = acoustic['discriminator']
    for key in ('optimizer', 'optimizer_d', 'rng', 'python_rng', 'cuda_rng'):
        phoneme.pop(key, None)
    phoneme['initialization'] = dict(mode='component-distillation',
        phoneme_sha256=phoneme_sha, acoustic_sha256=acoustic_sha,
        acoustic_initialization=acoustic.get('initialization'),
        decoder_sha256=hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        decoder_probe=report, note='Phoneme text encoder/duration network retained; acoustic encoder/flow transferred; decoder warm-started from one training-clip reconstruction.')
    phoneme['evaluation_snapshot'] = dict(resumable=False, source='component-distillation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream: torch.save(phoneme, stream)
    print(f'Created acoustic warm-start {output}; initialize a new run, not resume')


if __name__ == '__main__': main()

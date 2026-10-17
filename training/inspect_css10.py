"""Inspect pinned CSS10 base candidates without importing or executing model code."""
import argparse
import hashlib
import json
from pathlib import Path
import stat

import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_FIELDS = (
    'num_chars', 'out_channels', 'spec_segment_size', 'hidden_channels',
    'hidden_channels_ffn_text_encoder', 'num_heads_text_encoder',
    'num_layers_text_encoder', 'kernel_size_text_encoder', 'dropout_p_text_encoder',
    'dropout_p_duration_predictor', 'kernel_size_posterior_encoder',
    'dilation_rate_posterior_encoder', 'num_layers_posterior_encoder',
    'kernel_size_flow', 'dilation_rate_flow', 'num_layers_flow',
    'resblock_type_decoder', 'resblock_kernel_sizes_decoder',
    'resblock_dilation_sizes_decoder', 'upsample_rates_decoder',
    'upsample_initial_channel_decoder', 'upsample_kernel_sizes_decoder',
    'use_sdp', 'use_speaker_embedding', 'speaker_embedding_channels',
    'use_d_vector_file', 'use_language_embedding', 'embedded_language_dim',
    'num_languages', 'condition_dp_on_speaker',
)


def hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def verify_stream(stream, record, name):
    import os
    metadata = os.fstat(stream.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != record['bytes']:
        raise ValueError('source size/type mismatch: ' + name)
    if hash_stream(stream) != record['sha256']:
        raise ValueError('source checksum mismatch: ' + name)
    stream.seek(0)


def load_weights(path, record):
    # Check and deserialize the same open file. No safe_globals expansion or
    # unsafe pickle fallback; publisher Python and TorchScript are never loaded.
    # Unbuffered reads ensure the second hash observes in-place writes instead
    # of reusing bytes cached by Python's BufferedReader during deserialization.
    with path.open('rb', buffering=0) as stream:
        verify_stream(stream, record, path.name)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        verify_stream(stream, record, path.name)
    return checkpoint


def token_inventory(config, expected_size=129):
    chars = config['characters']
    if (config['use_phonemes'] or config['text_cleaner'] != 'multilingual_cleaners'
            or not config['add_blank'] or config['enable_eos_bos_chars']
            or chars['characters_class'] != 'TTS.tts.utils.text.characters.Graphemes'
            or not chars['is_unique'] or not chars['is_sorted']
            or [chars[k] for k in ['pad', 'eos', 'bos', 'blank']] != ['<PAD>', '<EOS>', '<BOS>', '<BLNK>']):
        raise ValueError('unexpected CSS10 text contract')
    symbols = ['<PAD>', '<EOS>', '<BOS>', '<BLNK>'] + sorted(set(chars['characters'])) + list(chars['punctuations'])
    if len(symbols) != expected_size or len(set(symbols)) != expected_size:
        raise ValueError('unexpected CSS10 token inventory')
    return symbols


def inspect_candidate(language, source, record=None):
    if record is None:
        pins = json.loads((ROOT / 'assets/vits-css10-candidates.json').read_text())
        record = pins['candidates'][language]
    vocabulary_size = record.get('vocabulary_size', 129)
    contents = {}
    for name, asset in record['files'].items():
        if name == record['weights']:
            continue
        with (source / name).open('rb') as stream:
            verify_stream(stream, asset, name)
            contents[name] = stream.read()
        if hashlib.sha256(contents[name]).hexdigest() != asset['sha256']:
            raise ValueError('source metadata changed while reading: ' + name)
    config = json.loads(contents['config.json'])
    inventory = token_inventory(config, vocabulary_size)
    languages = json.loads(contents['language_ids.json'])
    speakers = json.loads(contents['speaker_ids.json'])
    if languages != {language: 0} or not speakers or set(speakers.values()) != {0}:
        raise ValueError('unexpected source conditioning IDs')
    registry = json.loads(contents['UPSTREAM-MODELS.json'])
    if record['registry_model'] is not None:
        entry = registry['tts_models'][language][record.get('registry_dataset', 'css10')][record['registry_model']]
        if entry['license'] != 'bsd-3-clause':
            raise ValueError('upstream registry license differs from reviewed candidate')
    if b'license: bsd-3-clause' not in contents['README.md']:
        raise ValueError('source license declarations differ from reviewed candidate')
    checkpoint = load_weights(source / record['weights'], record['files'][record['weights']])
    state = checkpoint['model']
    if not isinstance(state, dict) or len(state) != record['tensor_count']:
        raise ValueError('unexpected generator tensor count')
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('source generator must contain finite float32 tensors')
    shapes = {key: list(value.shape) for key, value in state.items()}
    schema = hashlib.sha256(json.dumps(shapes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if schema != record['tensor_schema_sha256']:
        raise ValueError('unexpected generator architecture')
    embedded = checkpoint['config']
    if (any(embedded['model_args'][k] != config['model_args'][k] for k in ARCHITECTURE_FIELDS)
            or token_inventory(embedded, vocabulary_size) != inventory
            or embedded['audio'] != config['audio']
            or embedded['num_speakers'] != 1
            or embedded['datasets'][0]['language'] != language):
        raise ValueError('embedded checkpoint configuration disagrees with source contract')
    return dict(
        schema_version=1, language=language, repository=record['repository'], revision=record['revision'],
        source=str(source.resolve()), source_verified=True, training_ready=False,
        weights_sha256=record['files'][record['weights']]['sha256'],
        tensor_count=len(state), tensor_schema_sha256=schema,
        parameter_count=sum(v.numel() for v in state.values()),
        source_step=checkpoint['step'], source_epoch=checkpoint['epoch'],
        sample_rate=config['audio']['sample_rate'],
        frontend=dict(use_phonemes=False, symbols=inventory, pad_id=0, eos_id=1, bos_id=2,
                      blank_id=3, add_boundary_blanks=True, add_bos_eos=False,
                      cleaner=config['text_cleaner'],
                      unsupported_punctuation=[c for c in '¡¿' if c not in inventory]),
        conditioning=dict(speakers=speakers, languages=languages,
                          speaker_embedding_shape=shapes['emb_g.weight'],
                          language_embedding_shape=shapes['emb_l.weight']),
        declared_license=record['declared_license'], commercial_clearance_established=False,
        provenance_limits=record['provenance_limits'],
        conversion_required=[
            f'Preserve the {vocabulary_size} source token rows and use blank ID 3; project VITS inserts ID 0.',
            'Match the source text cleaner; decide explicitly how unsupported characters are handled.',
            'Preserve 192 character plus 4 language channels in the 196-channel text encoder.',
            'Preserve speaker conditioning and the additional language projection in the duration predictor.',
            'Map flow indices, affine parameter names and decoder normalization without dropping weights.',
            'Use a fresh discriminator and target optimizer; source checkpoints contain generator weights only.',
            'Verify original-versus-converted inference, adaptation/resume and native ONNX parity before Studio integration.',
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('language', choices=['de', 'es'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--report', type=Path, help='new JSON report path; existing files are preserved')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = inspect_candidate(args.language, args.source or ROOT / 'voices/css10-candidates' / args.language)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.report:
        with args.report.open('x') as output:
            output.write(rendered)
        print(f'Inspection passed: {args.report}. Training conversion remains required.')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()

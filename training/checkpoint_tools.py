"""Restricted project checkpoint inspection shared by preflight and evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from runtime import cli
from train import MATH_MODEL, MATH_WIDE_MODEL, MATH_PROFILES, MODEL, PIPER_MODEL, SynthesizerTrn, MultiPeriodDiscriminator

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(MODEL, inter_channels=16, hidden_channels=16, filter_channels=32, n_layers=2,
            upsample_initial_channel=32, resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3, 5]])

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(65536): digest.update(chunk)
    return digest.hexdigest()

def load(path, expected_sha256=None):
    if expected_sha256 is not None and (len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    # Hash and deserialize the same file handle, without arbitrary pickle globals.
    with Path(path).open('rb') as stream:
        import os, stat
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 2_000_000_000:
            raise ValueError('checkpoint must be a regular project file under two gigabytes')
        digest = hashlib.sha256()
        while chunk := stream.read(65536): digest.update(chunk)
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError('checkpoint changed since evaluation (SHA-256 mismatch)')
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        # Atomic checkpoint replacement leaves this open handle stable. Also detect
        # in-place changes while deserializing rather than mislabel the loaded model.
        stream.seek(0)
        after = hashlib.sha256()
        while chunk := stream.read(65536): after.update(chunk)
        if after.hexdigest() != actual: raise ValueError('checkpoint changed while loading')
    if not isinstance(saved, dict): raise ValueError('invalid checkpoint')
    return saved, digest.hexdigest()

def build(saved, discriminator=False):
    profile = saved.get('model_profile', 'project-v1')
    config = saved['training_config']
    if config.get('normalization') not in [None, 'text-v1']:
        raise ValueError('unsupported text normalization version')
    if config.get('normalization') is not None and config.get('frontend', 'grapheme-v1') == 'grapheme-v1':
        raise ValueError('legacy grapheme frontend cannot use text-v1 normalization')
    if profile == 'project-v1': allowed = [MODEL, TINY]
    elif profile in MATH_PROFILES:
        if config['language'] != 'en' or config.get('frontend') != ('cmu-ipa-v1' if profile != 'math-student-v1' else 'grapheme-v2'): raise ValueError('invalid math student frontend')
        allowed = [MATH_WIDE_MODEL if profile == 'math-phoneme-wide-v1' else MATH_MODEL]
    elif profile == 'ljs-sdp-v1': allowed = [dict(MODEL, use_sdp=True)]
    elif profile == 'piper-uk-medium-v1': allowed = [PIPER_MODEL]
    elif profile in ['css10-de-es-v1', 'neon-european-v1']:
        from css10_adapter import MODEL as CSS10_MODEL
        allowed = [CSS10_MODEL]
    elif profile == 'vosk-ru-multi-v1':
        from vosk_adapter import MODEL as VOSK_MODEL
        allowed = [VOSK_MODEL]
    else: raise ValueError('unsupported architecture profile')
    if saved['model_config'] not in allowed: raise ValueError('unsupported checkpoint architecture')
    if (config['schema_version'] != 1 or config['sample_rate'] != 22050
            or config['language'] not in ['en', 'ru', 'uk', 'de', 'fr', 'es', 'it', 'pt', 'pl', 'nl']
            or config.get('frontend', 'grapheme-v1') not in ['grapheme-v1', 'grapheme-v2', 'cmu-ipa-v1', 'piper-uk-text-v1', 'vosk-ru-phones-v1', 'css10-graphemes-v1', 'neon-graphemes-v1']):
        raise ValueError('unsupported checkpoint language/frontend/sample rate')
    symbols = config['symbols']
    if (not 2 <= len(symbols) <= 4096 or symbols[0] != '\0'
            or any(not isinstance(c, str) or len(c) != 1 for c in symbols)):
        raise ValueError('invalid checkpoint token inventory')
    for c in set(symbols):
        if symbols.count(c) > 1 and not (config.get('frontend') == 'cmu-ipa-v1' and c == "'" and symbols.count(c) == 2):
            raise ValueError('duplicate checkpoint token')
    if config.get('frontend') == 'cmu-ipa-v1':
        if config['language'] != 'en': raise ValueError('CMU IPA requires English')
        dictionary = config['dictionary']
        if sha(dictionary['path']) != dictionary['sha256']: raise ValueError('frontend dictionary checksum mismatch')
    if profile == 'ljs-sdp-v1' and (config.get('frontend') != 'cmu-ipa-v1' or len(symbols) != 178):
        raise ValueError('LJSpeech profile requires its IPA inventory')
    if config.get('frontend') == 'piper-uk-text-v1':
        from import_piper_uk import symbols as piper_symbols
        if config['language'] != 'uk' or symbols != piper_symbols() or (type(config.get('conditioning_speaker')) is not int or config['conditioning_speaker'] not in [0,1,2]):
            raise ValueError('invalid Ukrainian inventory/language/conditioning speaker')
    if profile == 'piper-uk-medium-v1' and config.get('frontend') != 'piper-uk-text-v1':
        raise ValueError('Piper profile requires its text frontend')
    if profile == 'vosk-ru-multi-v1':
        from vosk_adapter import FRONTEND, symbols as vosk_symbols
        if config.get('frontend') != FRONTEND or config['language'] != 'ru' or symbols != vosk_symbols() or type(config.get('conditioning_speaker')) is not int or not 0 <= config['conditioning_speaker'] < 5:
            raise ValueError('invalid Russian frontend/inventory/conditioning speaker')
        pin = json.loads((ROOT/'assets/russian-assets.json').read_text())['files']['dictionary']['sha256']
        dictionary = config['dictionary']
        if dictionary['sha256'] != pin or sha(dictionary['path']) != pin: raise ValueError('Russian dictionary checksum mismatch')
    elif config.get('frontend') == 'vosk-ru-phones-v1': raise ValueError('Russian phoneme frontend requires Vosk profile')
    if profile == 'neon-european-v1':
        from neon_adapter import validate_config
        validate_config(config)
    elif config.get('frontend') == 'neon-graphemes-v1':
        raise ValueError('Neon frontend requires its architecture profile')
    if profile == 'css10-de-es-v1':
        from css10_adapter import validate_config
        validate_config(config)
    elif config.get('frontend') == 'css10-graphemes-v1':
        raise ValueError('CSS10 frontend requires its architecture profile')
    if profile not in ['piper-uk-medium-v1', 'vosk-ru-multi-v1', 'css10-de-es-v1', 'neon-european-v1'] and config.get('conditioning_speaker') is not None:
        raise ValueError('conditioning speaker requires the Piper profile')
    if profile == 'vosk-ru-multi-v1':
        from vosk_adapter import build_model
        net = build_model()
    elif profile == 'neon-european-v1':
        from neon_adapter import build_model
        net = build_model(config['language'])
    elif profile == 'css10-de-es-v1':
        from css10_adapter import build_model
        net = build_model()
    else: net = SynthesizerTrn(len(symbols), 513, 32, **saved['model_config'])
    net.load_state_dict(saved['generator'], strict=True)
    states = [saved['generator']]
    if discriminator and saved.get('discriminator') is not None:
        disc = MultiPeriodDiscriminator(False)
        disc.load_state_dict(saved['discriminator'], strict=True)
        states.append(saved['discriminator'])
        del disc
    for state in states:
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('non-finite checkpoint weights')
    if discriminator and profile == 'vosk-ru-multi-v1' and saved.get('duration_discriminator') is not None:
        from vosk_adapter import duration_discriminator
        duration = duration_discriminator(); duration.load_state_dict(saved['duration_discriminator'], strict=True)
        if any(not torch.isfinite(v).all() for v in saved['duration_discriminator'].values()): raise ValueError('non-finite duration discriminator')
    net.eval()
    return net

def prepare(dataset, config, directory, current=False):
    directory = Path(directory)
    base = directory / 'base-config.json'
    result = directory / 'prepared.json'
    base.write_text(json.dumps(config, ensure_ascii=False))
    arguments = ['training-prepare', str(Path(dataset).resolve()), str(result), str(base)]
    if current and config.get('frontend', 'grapheme-v1') != 'grapheme-v1':
        arguments.append('--text-v1')
    subprocess.run(cli() + arguments, cwd=ROOT, check=True)
    return json.loads(result.read_text()), sha(result)

def summary(saved, digest, path):
    config = saved['training_config']
    notes = ['Engineering tiny architecture; not a production voice.'] if saved['model_config']==TINY else []
    if config.get('conditioning_speaker') is not None:
        notes.append(f"Fixed source conditioning row {config['conditioning_speaker']}; the exported target has one named speaker")
    origin = saved.get('initialization', {})
    for _ in range(64):
        if not origin: break
        if origin.get('license_limit'): notes.append(origin['license_limit'])
        if origin.get('frontend_note'): notes.append(origin['frontend_note'])
        origin = origin.get('base_initialization', {})
    return dict(checkpoint=str(Path(path).resolve()), sha256=digest, step=saved['step'],
                language=config['language'], frontend=config.get('frontend', 'grapheme-v1'),
                normalization=config.get('normalization'),
                sample_rate=config['sample_rate'], speaker=config['speaker'], symbols=len(config['symbols']),
                model_profile=saved.get('model_profile', 'project-v1'),
                discriminator='included in checkpoint' if saved.get('discriminator') is not None else 'absent; fresh on adaptation',
                notes=notes)

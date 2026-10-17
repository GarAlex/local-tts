"""Pinned European Neon VITS profiles, including the distinct Italian inventory."""
import json
from pathlib import Path
from css10_adapter import MODEL, build_model as build_css10, convert_state as convert_css10

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ('fr', 'it', 'pt', 'pl', 'nl')
PROFILE = 'neon-european-v1'
FRONTEND = 'neon-graphemes-v1'


def symbols(language):
    if language not in LANGUAGES:
        raise ValueError('unsupported Neon European language')
    name = 'neon-it-symbols.json' if language == 'it' else 'css10-symbols.json'
    return json.loads((ROOT / 'assets' / name).read_text())


def validate_config(config):
    language = config['language']
    if (language not in LANGUAGES or config.get('frontend') != FRONTEND
            or config['symbols'] != symbols(language)
            or type(config.get('conditioning_speaker')) is not int
            or config['conditioning_speaker'] != 0):
        raise ValueError('invalid Neon language/frontend/inventory/conditioning speaker')


def build_model(language):
    return build_css10(len(symbols(language)))


def convert_state(source, language):
    return convert_css10(source, len(symbols(language)))

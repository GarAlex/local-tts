#!/usr/bin/env python3
"""Repeatable local verification; --full includes installed training/model assets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='also run alignment, training/export and C++ inference')
parser.add_argument('--effects', action='store_true', help='also test effects C ABI on Kokoro FP32/Q8/Q4 and WAV processing CLI')
parser.add_argument('--cyrillic', action='store_true', help='also exercise installed Russian/Ukrainian pretrained voices and C++ consumers')
parser.add_argument('--kokoro', action='store_true', help='also test pinned FP32/Q8/Q4 Kokoro models and C++ consumers')
parser.add_argument('--language', action='store_true', help='test expanded language forms on installed EN/RU/UK models and numeric fine-tuning/export parity')
parser.add_argument('--public-base', action='store_true', help='also test installed public English base import, adaptation, resume and native export')
parser.add_argument('--russian-base', action='store_true', help='also test installed original Russian Vosk checkpoint adaptation and normalized native export')
parser.add_argument('--ukrainian-base', action='store_true', help='also test installed original Ukrainian Piper checkpoint import/adaptation/native parity')
parser.add_argument('--french', action='store_true', help='test installed French Chatterbox inference/cancellation and project VITS adaptation')
parser.add_argument('--european', action='store_true', help='test added-language inference plus French/German/Spanish VITS adaptation and C++ consumption')
parser.add_argument('--css10-candidates', action='store_true', help='inspect installed pinned German/Spanish public base candidates; does not imply import support')
parser.add_argument('--neon-bases', action='store_true', help='verify installed FR/IT/PT/PL/NL conversion, adaptation/resume and native export; requires optional model.pt references')
parser.add_argument('--css10-bases', action='store_true', help='test installed DE/ES source conversion, adaptation/resume and native export; requires optional model.pt references')
args = parser.parse_args()
commands = [
    ('format', ['cargo', 'fmt', '--all', '--check']),
    ('rust-tests', ['cargo', 'test', '--workspace']),
    ('clippy', ['cargo', 'clippy', '--workspace', '--all-targets', '--', '-D', 'warnings']),
    ('build', ['cargo', 'build', '--workspace']),
]
if args.full:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.extend([
        ('alignment', [str(python), 'tests/alignment.py']),
        ('training-export-parity-en', [str(python), 'tests/training_smoke.py', '--language', 'en']),
        ('training-export-parity-ru', [str(python), 'tests/training_smoke.py', '--language', 'ru']),
        ('training-export-parity-uk', [str(python), 'tests/training_smoke.py', '--language', 'uk']),
        ('cpp-inference-cancellation', [sys.executable, 'scripts/test_cpp.py']),
    ])
if args.french or args.european:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.extend([
        ('multilingual-tokenizer', ['cargo', 'test', '--release', '-p', 'local-tts-core', 'multilingual_tokenizer', '--', '--ignored']),
        ('french-native-inference', ['cargo', 'test', '--release', '-p', 'local-tts-core', '--test', 'european', '--', '--ignored']),
        ('french-training-export', [str(python), 'tests/training_smoke.py', '--language', 'fr']),
    ])
if args.european:
    for language in ['de', 'es']:
        commands.append((f'{language}-training-export', [str(python), 'tests/training_smoke.py', '--language', language]))
    commands.append(('european-cpp', [sys.executable, 'scripts/test_european_cpp.py']))
if args.css10_candidates:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.append(('css10-candidate-inspection', [str(python), 'tests/css10_candidates.py', '--installed']))
if args.neon_bases:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    for language in ['fr', 'it', 'pt', 'pl', 'nl']:
        commands.extend([
            (f'neon-{language}-conversion', [str(python), 'tests/css10_conversion.py', language,
                '--reference', str(ROOT / 'voices/neon-candidates' / language / 'model.pt')]),
            (f'neon-{language}-adaptation', [str(python), 'tests/css10_base_smoke.py', language]),
        ])
if args.css10_bases:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    for language in ['de', 'es']:
        commands.extend([
            (f'css10-{language}-conversion', [str(python), 'tests/css10_conversion.py', language,
                '--reference', str(ROOT / 'voices/css10-candidates' / language / 'model.pt')]),
            (f'css10-{language}-adaptation', [str(python), 'tests/css10_base_smoke.py', language]),
        ])
if args.effects:
    commands.extend([
        ('cpp-effects-kokoro', [sys.executable, 'scripts/test_cpp.py', '--kokoro']),
        ('native-shift-sanitizers', [sys.executable, 'scripts/test_native_shift.py']),
        ('effects-cli', [sys.executable, 'tests/effects_cli.py']),
    ])
if args.language:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.extend([
        ('language-real-models', ['cargo', 'test', '-p', 'local-tts-core', '--test', 'language', '--', '--ignored']),
        ('language-fine-tuning-parity', [str(python), 'tests/language_training.py']),
    ])
    if not args.full:
        commands.append(('cpp-text-and-inference', [sys.executable, 'scripts/test_cpp.py']))
if args.cyrillic:
    commands.extend([
        ('pretrained-ru-uk', ['cargo', 'test', '-p', 'local-tts-core', '--test', 'pretrained', '--', '--ignored']),
        ('cpp-cyrillic', [sys.executable, 'scripts/test_cpp.py', '--cyrillic']),
    ])
if args.kokoro:
    commands.extend([
        ('kokoro-variants', ['cargo', 'test', '-p', 'local-tts-core', '--test', 'kokoro_variants', '--', '--ignored', '--nocapture']),
        ('cpp-kokoro-variants', [sys.executable, 'scripts/test_cpp.py', '--kokoro']),
    ])
if args.public_base:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.append(('public-base-import-adaptation', [str(python), 'tests/public_base_smoke.py']))
if args.ukrainian_base:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.append(('ukrainian-base-import-adaptation', [str(python), 'tests/ukrainian_base_smoke.py']))
if args.russian_base:
    python = ROOT / ('.venv/Scripts/python.exe' if sys.platform == 'win32' else '.venv/bin/python')
    commands.append(('russian-base-import-adaptation', [str(python), 'tests/russian_base_smoke.py']))
results = []
for name, command in commands:
    print(f'Verifying {name}', flush=True)
    start = time.monotonic()
    try:
        process = subprocess.run(command, cwd=ROOT, timeout=600)
        code = process.returncode
        failure = None
    except (OSError, subprocess.TimeoutExpired) as error:
        code, failure = -1, str(error)
    results.append(dict(check=name, command=command, exit_code=code,
                        seconds=round(time.monotonic()-start, 2), error=failure))
report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), platform=sys.platform,
              effects=args.effects, language=args.language, french=args.french, european=args.european, css10_candidates=args.css10_candidates, css10_bases=args.css10_bases, neon_bases=args.neon_bases,
              full=args.full, cyrillic=args.cyrillic, kokoro=args.kokoro, public_base=args.public_base, ukrainian_base=args.ukrainian_base, russian_base=args.russian_base, passed=all(r['exit_code'] == 0 for r in results), checks=results)
output = ROOT / 'target/verification.json'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, indent=2))
print(f'Verification {"passed" if report["passed"] else "failed"}: {output}', flush=True)
raise SystemExit(0 if report['passed'] else 1)

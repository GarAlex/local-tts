"""Verify relocated macOS SDK against installed European-base engineering fixtures."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    parser.add_argument('--report', type=Path, default=ROOT / 'target/neon-release-verification.json')
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('this relocation harness currently targets the macOS SDK')
    candidate = args.candidate.resolve()
    if not (candidate / 'release-manifest.json').is_file():
        raise ValueError('candidate has no release manifest')
    kits = list(candidate.glob('local-tts-sdk-*/bin/local-tts'))
    if len(kits) != 1:
        raise ValueError('expected exactly one extracted SDK in candidate')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DYLD_', 'ORT_', 'LOCAL_TTS_', 'PYTHON'))}
    env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Preserve a virtualenv interpreter symlink; resolving it escapes its environment.
    python = args.python.absolute()
    with tempfile.TemporaryDirectory(prefix='European SDK relocation ') as temporary:
        work = Path(temporary)
        sdk = work / 'Moved SDK'
        shutil.copytree(kits[0].parents[1], sdk)

        def run(*command):
            result = subprocess.run(list(map(str, command)), cwd=work, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(f"{command[0]} failed ({result.returncode}):\n{result.stdout}")
            return result.stdout

        cpp = sdk / 'bin/neon-speak'
        run('/usr/bin/clang++', '-std=c++17', '-I', sdk / 'include', sdk / 'examples/cpp/speak.cpp',
            '-L', sdk / 'lib', '-llocal_tts', '-Wl,-rpath,@loader_path/../lib', '-o', cpp)
        reports = []
        for language, abi in [('fr', 4), ('it', 6), ('pt', 7), ('pl', 8), ('nl', 9)]:
            prior = json.loads((ROOT / f'target/neon-{language}-training.json').read_text())
            if not prior['passed']:
                raise ValueError('training fixture did not pass: ' + language)
            fixture = Path(prior['run'])
            base = ROOT / f'voices/vits-neon-{language}-base/base.pt'
            result = run(python, sdk / 'training/preflight.py', '--checkpoint', base, '--dataset', fixture / 'dataset.json')
            check = json.loads(result.splitlines()[-1])
            assert check['model_profile'] == 'neon-european-v1'
            dataset = json.loads((fixture / 'dataset.json').read_text())
            phrase = dataset['utterances'][1]['text']
            for name, command in [
                ('cli', [sdk / 'bin/local-tts', 'vits-synthesize', fixture / 'voice', work / f'{language}-cli.wav', phrase]),
                ('cpp', [cpp, fixture / 'voice', work / f'{language}-cpp.wav', str(abi), 'fixture', phrase, 'vits']),
            ]:
                run(*command)
                with wave.open(str(work / f'{language}-{name}.wav')) as wav:
                    assert wav.getframerate() == 22050 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    pcm = struct.unpack('<' + 'h' * wav.getnframes(), wav.readframes(wav.getnframes()))
                    assert len(pcm) > 1000 and max(map(abs, pcm)) > 32
            reports.append(dict(language=language, packaged_preflight=True, relocated_cli=True, relocated_cpp=True))
            print(language + ': packaged preflight and relocated Rust/C++ passed', flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(candidate=str(candidate), passed=True,
            checks=reports, synthetic_fixtures=True), indent=2) + '\n')
        print(args.report)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Compile and exercise actual C++ consumers against the development shared library."""
import pathlib
import subprocess
import sys
import tempfile
import os
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
if sys.platform not in ("darwin", "linux"):
    raise SystemExit("This harness supports macOS/Linux; Windows builds remain unverified.")
subprocess.run(["cargo", "build", "-p", "local-tts-ffi"], cwd=ROOT, check=True)
out = ROOT / "target" / "cpp"
out.mkdir(exist_ok=True)
for name in ("speak", "abi_test", "pronunciation_test", "text_test"):
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-I", str(ROOT / "include"), str(ROOT / "examples" / "cpp" / f"{name}.cpp"),
                    "-L", str(ROOT / "target" / "debug"), "-llocal_tts",
                    f"-Wl,-rpath,{ROOT / 'target' / 'debug'}", "-o", str(out / name)], check=True)
subprocess.run([str(out / "text_test")], check=True)
subprocess.run([str(out / "abi_test"), str(ROOT / "voices/kokoro-en")], check=True)
subprocess.run([str(out / "speak"), str(ROOT / "voices/kokoro-en"), str(out / "english.wav")], check=True)
if len(sys.argv) == 5:
    subprocess.run([str(out / "speak"), sys.argv[1], str(out / "own-voice.wav"),
                    sys.argv[2], sys.argv[3], sys.argv[4], "vits"], check=True)

if sys.argv[1:] == ["--cyrillic"]:
    subprocess.run([str(out / "speak"), str(ROOT / "voices/vosk-ru"), str(out / "russian.wav"),
                    "1", "F03", "Привет! 21 и 3,05.", "vosk-ru"], check=True)
    subprocess.run([str(out / "speak"), str(ROOT / "voices/piper-uk"), str(out / "ukrainian.wav"),
                    "2", "lada", "Приві́т! 21 і 3,05.", "piper-uk"], check=True)

    for language, folder, files in [('ru', 'vosk-ru', ['model.onnx', 'config.json', 'dictionary']),
                                    ('uk', 'piper-uk', ['model.onnx', 'config.json'])]:
        with tempfile.TemporaryDirectory(prefix='pronunciation-', dir=out) as temporary:
            fixture = pathlib.Path(temporary)
            for name in files:
                source = ROOT / 'voices' / folder / name
                try:
                    os.link(source, fixture / name)
                except OSError:
                    shutil.copyfile(source, fixture / name)
            subprocess.run([str(out / 'pronunciation_test'), str(fixture), language], check=True)

if sys.argv[1:] == ["--kokoro"]:
    for variant in ['q8', 'q4']:
        folder = ROOT / 'voices' / f'kokoro-en-{variant}'
        subprocess.run([str(out / 'abi_test'), str(folder)], check=True)
        subprocess.run([str(out / 'speak'), str(folder), str(out / f'english-{variant}.wav')], check=True)
        print(f'Kokoro {variant}: C++ inference, cancellation, ownership and concurrency passed', flush=True)

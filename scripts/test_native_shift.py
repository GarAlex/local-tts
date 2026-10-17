"""Check the vendored C++ bridge with address/undefined-behavior sanitizers."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if sys.platform not in ('darwin', 'linux'):
    raise SystemExit('Native sanitizer harness supports macOS/Linux; Windows remains unverified.')
output = ROOT / 'target/cpp/native-shift-sanitized'
output.parent.mkdir(exist_ok=True)
native = ROOT / 'crates/tts-core/native'
subprocess.run(['c++', '-std=c++11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-I', str(native / 'vendor/signalsmith-stretch'),
                '-I', str(native / 'vendor/linear/include'),
                str(native / 'voice_shift.cpp'), str(ROOT / 'tests/native_shift.cpp'),
                '-o', str(output)], check=True, cwd=ROOT)
subprocess.run([str(output)], check=True, timeout=120, cwd=ROOT)

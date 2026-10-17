"""Build a development macOS app bundle (workspace must remain available)."""
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
subprocess.run(['cargo', 'build', '--workspace'], cwd=root, check=True)
bundle = root / 'target/Local TTS Studio.app'
contents = bundle / 'Contents'
(contents / 'MacOS').mkdir(parents=True, exist_ok=True)
notices = contents / 'Resources' / 'licenses'
notices.mkdir(parents=True, exist_ok=True)
vendor = root / 'crates/tts-core/native/vendor'
for name in ('signalsmith-stretch', 'linear'):
    shutil.copy2(vendor / name / 'LICENSE.txt', notices / f'{name}-MIT.txt')
shutil.copy2(vendor / 'UPSTREAM.json', notices / 'signalsmith-UPSTREAM.json')
# Install a new inode so an already-running app retains its mapped executable.
with tempfile.NamedTemporaryFile(dir=contents / 'MacOS', prefix='.tts-studio-', delete=False) as stream:
    temporary = Path(stream.name)
try:
    shutil.copy2(root / 'target/debug/tts-studio', temporary)
    temporary.replace(contents / 'MacOS/tts-studio')
finally:
    temporary.unlink(missing_ok=True)
with (contents / 'Info.plist').open('wb') as f:
    plistlib.dump(dict(CFBundleName='Local TTS Studio', CFBundleDisplayName='Local TTS Studio',
                      CFBundleIdentifier='local.tts.studio', CFBundleExecutable='tts-studio',
                      CFBundlePackageType='APPL', CFBundleVersion='0.1.0', NSHighResolutionCapable=True,
                      NSMicrophoneUsageDescription='Record voice training clips locally when you press Record microphone.'), f)
print(bundle)

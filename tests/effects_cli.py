"""Exercise local WAV processing and CLI failures; no Python DSP dependencies."""
from pathlib import Path
import hashlib
import math
import struct
import subprocess
import tempfile
import wave
import json

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'target/debug/local-tts'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def invoke(source, output, bass, brightness, compression, success=True, pitch=None, formant=None):
    extra = [] if pitch is None and formant is None else [str(0 if pitch is None else pitch)]
    if formant is not None:
        extra.append(str(formant))
    result = subprocess.run([str(CLI), 'audio-process', str(source), str(output),
                             str(bass), str(brightness), str(compression), *extra],
                            text=True, capture_output=True, timeout=30)
    assert (result.returncode == 0) == success, result.stderr
    if success:
        with wave.open(str(source)) as before, wave.open(str(output)) as after:
            assert after.getnchannels() == 1 and after.getsampwidth() == 2
            assert before.getframerate() == after.getframerate()
            assert before.getnframes() == after.getnframes()
            pcm = struct.unpack('<' + 'h' * after.getnframes(), after.readframes(after.getnframes()))
            assert max(abs(x) for x in pcm) <= math.ceil(.98 * 32767)
        assert 'protection gain' in result.stdout
    else:
        assert not output.exists(), 'invalid settings wrote output'

def frequency(path):
    with wave.open(str(path)) as clip:
        values = struct.unpack('<' + 'h' * clip.getnframes(), clip.readframes(clip.getnframes()))
        start, end = len(values) // 4, len(values) * 3 // 4
        count = sum(values[i] < 0 <= values[i+1] for i in range(start, end-1))
        return count * clip.getframerate() / (end-start)

# Vendored implementation and license bytes must match their recorded pins.
vendor = ROOT / 'crates/tts-core/native/vendor'
for record in json.loads((vendor / 'UPSTREAM.json').read_text()):
    repo = record['repository'].rsplit('/', 1)[-1]
    for name, expected in record['files'].items():
        assert digest(vendor / repo / name) == expected, 'vendored source hash mismatch'

with tempfile.TemporaryDirectory(prefix='effects-cli-', dir=ROOT / 'target') as temporary:
    folder = Path(temporary)
    source = folder / 'source.wav'
    samples = [int(20000 * math.sin(2 * math.pi * 100 * i / 24000)) for i in range(24000)]
    with wave.open(str(source), 'wb') as stream:
        stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        stream.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    original_hash = digest(source)
    invoke(source, folder / 'processed.wav', 12, -6, .8)
    invoke(source, folder / 'neutral.wav', 0, 0, 0)
    with wave.open(str(folder / 'neutral.wav')) as stream:
        neutral = struct.unpack('<' + 'h' * len(samples), stream.readframes(len(samples)))
        assert max(abs(a - b) for a, b in zip(samples, neutral)) <= 1
    for index, settings in enumerate([('NaN', 0, 0), (0, 'inf', 0), (13, 0, 0), (0, 0, -1), (0, 0, 1.1), ('invalid', 0, 0)]):
        invoke(source, folder / f'invalid-{index}.wav', *settings, success=False)
    for index, (pitch, formant) in enumerate([('NaN', 0), ('inf', 0), (12.1, 0), (-12.1, 0), (0, 'NaN'), (0, 6.1), (0, -6.1)]):
        invoke(source, folder / f'invalid-shift-{index}.wav', 0, 0, 0, success=False, pitch=pitch, formant=formant)
    for pitch, expected in [(-12, 50), (12, 200)]:
        output = folder / f'pitch-{pitch}.wav'
        invoke(source, output, 0, 0, 0, pitch=pitch)
        assert abs(frequency(output) - expected) <= 4, 'pitch did not shift measured frequency'
    formants = folder / 'formants.wav'
    invoke(source, formants, 0, 0, 0, formant=3)
    assert abs(frequency(formants) - 100) <= 4, 'formant processing changed pitch'
    assert original_hash == digest(source)

# Retain reviewable processing samples from actual native Kokoro inference.
output_folder = ROOT / 'target/voice-effects'
output_folder.mkdir(exist_ok=True)
names = ['english', 'english-q8', 'english-q4']
names += [name for name in ['russian', 'ukrainian'] if (ROOT / 'target/cpp' / f'{name}.wav').exists()]
for name in names:
    source = ROOT / 'target/cpp' / f'{name}.wav'
    original_hash = digest(source)
    invoke(source, output_folder / f'{name}-warm.wav', 6, -2, .5)
    shift_folder = ROOT / 'target/voice-effects-pitch'
    shift_folder.mkdir(exist_ok=True)
    for pitch, formant in [(-4, -2), (4, 2)]:
        invoke(source, shift_folder / f'{name}-pitch-{pitch}-formants-{formant}.wav', 0, 0, 0,
               pitch=pitch, formant=formant)
    assert digest(source) == original_hash
assert 'audio-process' in subprocess.check_output([str(CLI), '--help'], text=True)
print('Effects CLI: WAV roundtrip, rejected settings, source preservation passed; speech clips: ' + ', '.join(names))

# Native voice effects

Bass/warmth EQ, brightness EQ, gentle compression, duration-preserving pitch
and independent formant shifts are implemented in the Rust library, additive
C ABI v1 functions, WAV CLI and Studio. They process generated PCM and need no
retraining. Pitch/formants use pinned MIT-licensed Signalsmith Stretch and Linear
headers compiled locally; no service, Python runtime or extra model is involved.

Studio: Generate a clip, adjust **Voice effects**, then **Apply effects**.
Choose **Original** or **Processed** before Play or Save WAV. Processing runs
on a worker; every pass uses the retained original. Changing controls requires
another Apply. Reset controls restores neutral settings; Apply updates the
processed clip. New generation/model selection clears the previous comparison.
Dataset recordings and training comparisons are played as recorded/exported.

```rust
use local_tts_core::effects::{VoiceEffects, process};
let (processed, stats) = process(&original, VoiceEffects {
    bass_db: 6.0,
    brightness_db: -2.0,
    compression: 0.5,
    pitch_semitones: -3.0,
    formant_semitones: -1.0,
    ..Default::default()
})?;
```

```cpp
TtsEffectsV2 settings = tts_effects_v2_default();
settings.bass_db = 6;
settings.brightness_db = -2;
settings.compression = .5f;
settings.pitch_semitones = -3;
settings.formant_semitones = -1;
TtsAudio *processed = nullptr;
char *error = nullptr;
int status = tts_audio_process_v2(original, &settings, &processed, &error);
// Check status; processed has independent ownership. Original remains intact.
tts_audio_free(processed);
tts_error_free(error);
```

```sh
target/debug/local-tts audio-process input.wav output.wav 6 -2 0.5 -3 -1
python3 scripts/verify.py --effects
```

The [native Rust example](../crates/tts-core/examples/voice_effects.rs) processes
the synthesis PCM directly and saves both versions, avoiding input quantization:

```sh
cargo run -p local-tts-core --example voice_effects -- \
  voices/kokoro-en target/my-pitch-comparison "Hello world." -4 -2
```

Use a new output folder; the example refuses to overwrite an existing folder.

The library accepts nonempty finite mono PCM at 8–96 kHz, up to ten minutes.
WAV input must already be mono. Sample rate and sample count stay unchanged.
Both shelves allow -12 to +12 dB. Bass uses 180 Hz and brightness uses 3000 Hz;
the implementation follows the mathematical [RBJ shelf equations](https://www.w3.org/TR/audio-eq-cookbook/)
with shelf slope S=1. Compression amount 0–1 moves from disabled to 4:1, with
a -18 dBFS threshold, 6 dB soft knee and 5 ms attack/120 ms release peak envelope.
There is no makeup gain. These are fixed tuning parameters for this first version.

Pitch accepts -12 to +12 semitones, and formants -6 to +6. Pitch compensates the
spectral envelope so formants approximately stay in place; the separate formant
control moves vocal resonances while retaining pitch. Fundamental frequency is
estimated internally. Processing order is pitch/formants, EQ, compression, then
peak protection. Both spectral controls at zero bypass that stage exactly.
Spectral windows are padded and drained, and their latency is removed before
returning exactly the original number of samples. Processing uses a fixed seed
for reproducibility. Formant compensation is approximate; large shifts can sound
artificial and transients may smear. These controls do not establish an age or
mood transformation. Small shifts are available for listening comparisons.

Peak protection applies one global attenuation factor if processed peaks exceed
the ceiling (default 0.98, allowed 0.1–1). It preserves relative dynamics without
hard clipping. Quiet clips are not amplified. Neutral controls preserve samples
exactly below the ceiling; above it, protection still attenuates. This is offline
whole-clip processing, not a streaming limiter or loudness normalization. f64
filter intermediates protect large finite f32 values; source PCM is immutable.
Memory and processing cost grow with clip length. C callers initialize the
20-byte settings struct and functions remain unchanged. The additive 28-byte
`TtsEffectsV2` uses `tts_effects_v2_default` and `tts_audio_process_v2` for pitch
and formants; unsupported sizes fail before reading the full struct. ABI version
stays 1. Rust's settings struct adds two fields; use `..Default::default()` for
forward-compatible source construction.

Builds now require a C++11 compiler alongside Rust. The `cc` build helper compiles
the vendored bridge at optimization level 2 even in debug builds. Deployed code
includes the native implementation and uses the platform C++ standard runtime.
There are no optional accelerated FFT dependencies. Source hashes and MIT notices
are under `crates/tts-core/native/vendor`; include both notices with distributions.
The development macOS bundle includes them in `Contents/Resources/licenses`.

Verification covers measured frequency gains/corners in both directions,
compression attack/release and loud/quiet passages, neutral identity, relative
peak preservation, silence/short clips, extreme settings/rates/PCM, malformed
input, independent ownership and concurrent C++ calls sharing a source after
engine disposal. CLI checks WAV roundtrips, invalid arguments and actual native
Kokoro FP32/Q8/Q4 speech. This establishes numerical behavior, not a listening
assessment of naturalness or every playback device.

This increment passed the workspace Rust suite, strict Clippy and builds on
macOS ARM; actual C++ shared-library consumers passed for all three Kokoro
variants. Existing Russian/Ukrainian native WAVs also passed processing at
22.05 kHz (the CLI harness includes those fixtures when available). Studio was
checked with English and Russian generation, effects application, model-change
reset and WAV saving. The English neutral WAV is byte-identical to its retained
original; the processed WAV differs with the same 79,200 samples at 24 kHz.
Reviewable clips/checks are under `target/voice-effects`. Playback-device and
subjective listening quality remain unverified.

Pitch verification adds measured frequency at five sample rates, source/duration
preservation, exact neutral bypass, deterministic output, silence/short/extreme
inputs and start/end latency checks. A synthetic harmonic vowel checks envelope
movement in both directions, pitch compensation and retained periodicity. C++
checks retain the legacy struct interface and concurrently exercise v2 shifts.
CLI checks reject invalid shifts and measure octave changes in WAVs; reviewable
native speech variants are generated under `target/voice-effects-pitch`.
The private C++ bridge also passes address/undefined-behavior sanitizers for
short clips, sample-rate extremes, guard-buffer preservation and malformed
arguments. The live Studio pitch/formant check passed after unlocking the Mac:
set -4/-2 semitones, apply, select Original/Processed and save each version.
Reset controls plus Apply recovered a byte-identical original WAV. Both the
original and processed Studio WAVs match direct native Rust synthesis/processing
byte-for-byte, retaining 79,200 samples at 24 kHz. Results and hashes are recorded
in `target/voice-effects-pitch/studio-check.json` and `verification.json`.
Reprocessing a saved 16-bit WAV changes the input PCM through quantization and
can change spectral phase decisions, so it is not an exact parity reference for
processing the original synthesis PCM. The native example provides that reference.

Age presets and listening evaluation remain planned. Selectable learned moods
require explicit style conditioning and labeled recordings; these audio effects
do not implement learned moods or a validated age transformation.

# Small English arithmetic voice

The first implementation is a model-free word-bank baseline, not a distilled
neural model. Rust `math_speech::MathVoice` and the ordinary C ABI (backend
`TTS_MATH_EN`, 8) read an expression using supplied voice data. It needs no
Kokoro model, pronunciation dictionary, Python or network at inference time.
The existing shared library still includes other neural backends; its executable
and ONNX Runtime size are separate from the small arithmetic voice data.

```sh
cargo run --release -p local-tts-core --example math_pack -- \
  voices/kokoro-en voices/new-math-bank target/example.wav
cargo build --release -p local-tts-speak
./target/release/tts-speak --backend math --model voices/math-en-v1 \
  --text '12 + 225 = 237' --output arithmetic.wav
```

`math_speech::words` exposes the deterministic spoken-word parser, and
`MathVoice::expression` returns mono 16 kHz audio. `MathVoice::build` accepts a
teacher implementing the project's Synthesizer trait. Existing output folders
are protected. Copy the source model's licenses/notices/provenance alongside
any generated bank; the builder records a supplied provenance description but
does not automatically certify rights or copy notices.

Supported: integers 0–999999999, signed values, decimal fractional digits,
addition/subtraction/multiplication/division, equality and parentheses up to
16 levels. Operators include ASCII and Unicode forms. It reads literally and
does not calculate, correct equations, or infer implicit multiplication.
Fractions expressed with `/` read as division. Scientific notation, powers,
units, arbitrary text and localized decimal commas are rejected. Integer leading
zeros are read as a cardinal; fractional zeros are retained. Use default speaker
and duration scale 1. Expression length is bounded at 4096 bytes and output at
120 seconds.

The bank stores trimmed, faded, windowed-sinc downsampled PCM16 fragments with
word offsets and a SHA-256 checksum. Short gaps separate fragments, with longer
pauses before operators. This is concatenation rather than learned sentence
prosody: audible joins and isolated-word stress are expected. Native-speaker
listening evaluation remains pending. A sub-1 MB bank demonstrates small data,
not superior naturalness to neural TTS.

English v1 was generated using the installed pinned Kokoro af_heart voice. Core
runtime data (words.pcm + math-voice.json) totals 764314 bytes. The complete folder, including retained teacher license/model card/provenance,
is 788244 bytes (0.788 MB).

## Studio and comparison samples

Select **English — Arithmetic (small word bank)** in Studio. Its default folder
is voices/math-en-v1. Enter an expression, inspect Arithmetic pronunciation
preview, then Generate/Play/Save WAV. Duration is fixed for this baseline; the
normal optional post-processing effects remain available.

Generate repeatable fragment/teacher pairs with the same parser readings:

```sh
cargo run --release -p local-tts-core --example math_compare -- \
  voices/math-en-v1 voices/kokoro-en target/new-math-comparison
```

The command refuses to overwrite an output folder. comparison.json records
expressions, spoken words, audio filenames, durations and synthesis latency
excluding model load and disk writes. Listening status remains pending; latency
is a machine-specific measurement, not a speech-quality score. The generated
v1 set contains ten pairs; it is not training data for the student.

## Remaining comparison work

1. Evaluate unseen expressions and fragment boundaries by listening; use the
   shared parser and a held-out arithmetic corpus for teacher/student comparisons.
2. Add context-conditioned fragments if they improve joins within the size budget.
3. Continue the [experimental neural student](MATH_STUDENT.md): teacher generation,
   a smaller training profile and a 100-step verified export now work. More data,
   sustained training, quantization and perceptual evaluation remain.
4. Compare complete package sizes, exact number/operator pronunciation,
   naturalness and latency. Keep the fragment baseline if the neural student
   does not justify its size. Russian is a later phase.

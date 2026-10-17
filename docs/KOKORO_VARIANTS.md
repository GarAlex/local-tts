# Kokoro quantized inference

FP32, Q8 and Q4 packages are supported by the same native Rust engine, CLI,
C/C++ ABI and Studio. Downloads retain the existing voice, tokenizer, dictionary
and notices; the model files come from the same pinned ONNX repository revision
`1939ad2a8e416c0acfeecc08a694d14ef25f2231`. SHA-256 hashes are compiled into the
loader through `assets/kokoro-variants.json`. Unknown or damaged models fail before
ONNX loading; editable pronunciation dictionaries remain outside those pins.

| Choice | Upstream file | Local folder | Model bytes |
|---|---|---|---:|
| FP32 | `onnx/model.onnx` | `voices/kokoro-en` | 325,532,232 |
| Q8 | `onnx/model_quantized.onnx` | `voices/kokoro-en-q8` | 92,361,116 |
| Q4 (mixed FP32) | `onnx/model_q4.onnx` | `voices/kokoro-en-q4` | 305,215,966 |

[Published files](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx)
and the [model card](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/blob/1939ad2a8e416c0acfeecc08a694d14ef25f2231/README.md)
declare the repository's Apache-2.0 license. Existing Apache/BSD notices apply.
Quantization does not remove the base-model terms.

Q4 is not an all-tensors four-bit model. Inspection of the actual downloaded graph
finds 76 `com.microsoft::MatMulNBits` nodes with `bits=4`, while roughly 301 MB of
initializer payload remains FP32. Q8 quantizes many more operations, including
integer matrix products, convolution and dynamic LSTM. This explains why the Q4
file is much larger than Q8. File size is not a measurement of process memory,
speed or speech quality. FP16 hybrid variants are not included in this selection.
The local graph report is `target/kokoro-graph-report.json`.

## Use

```sh
python3 scripts/download_english.py all
# Or acquire just q8 / q4; no argument keeps the existing FP32 default.
cargo run -p local-tts-cli -- synthesize voices/kokoro-en-q8 q8.wav 'Hello world.'
cargo run -p local-tts-cli -- synthesize voices/kokoro-en-q4 q4.wav 'Hello world.'
python3 scripts/verify.py --kokoro
```

In Studio choose English/Kokoro, then **Kokoro variant**. The folder changes to the
selected package; Download English model acquires that variant. Generate logs the
actual detected variant. Manually browsing another trusted package also works;
the actual pinned model hash determines the loaded precision.

Rust consumers call `Kokoro::load(folder)` and can inspect `engine.variant()`.
C/C++ consumers call the existing `tts_engine_load(folder, TTS_KOKORO, ...)`.
No ABI/backend IDs changed. Voice `af_heart`, 24 kHz mono float PCM, number
expansion, duration controls, paragraph chunking and cooperative cancellation
are shared across variants. Host applications need neither Python nor a server.

## Verification and listening

The optional native regression loads every actual model, verifies identical
frontend phonemes, finite non-silent PCM, three sentence cases including numbers,
paragraph generation, duration changes, cancellation and invalid inputs. It emits
same-text WAVs and local timings to `target/kokoro-comparison`. Timings include
native text processing but exclude model loading; each model is warmed once and
each sentence is generated three times. No timing threshold is used. The C++
consumer checks actual quantized inference, ownership, errors, cancellation,
engine reuse and concurrent calls.

These are engineering checks. Waveforms are not expected to equal FP32, and
listening quality still needs evaluation. The Q8/Q4 ONNX models are inference
assets; they are not fine-tuning inputs to the current own-voice trainer. The
required pretrained trainable-base import remains in PHASES.md.

### Local results — Apple M4 Pro, macOS ARM

Two ONNX CPU threads, Rust debug host, one warm-up and nine timed sentence runs per
variant. The models were timed sequentially without concurrent synthesis tests.
Loading includes package hashing, dictionary parsing and session creation; these
are development-build measurements, not release startup promises.

| Variant | Model MB (decimal) | Median warm RTF | Load seconds |
|---|---:|---:|---:|
| f32 | 325.53 | 0.302 | 12.38 |
| q8 | 92.36 | 0.337 | 4.17 |
| q4 | 305.22 | 0.308 | 11.92 |

RTF is generation seconds divided by audio seconds; lower is faster. Q8 reduced
the model file by about 72% but was slightly slower than FP32 on these local
samples. Q4 reduced file size by about 6% and was also slightly slower. All timed
outputs had finite, non-silent PCM and zero samples beyond +/-1. This is not a
speech-quality score or a performance promise for other hardware.

Listen to the same first sentence in `target/kokoro-comparison/f32-0.wav`,
`q8-0.wav` and `q4-0.wav`; numeric and paragraph cases are also available there.

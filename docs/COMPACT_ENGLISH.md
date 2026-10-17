# Compact English candidates

Two experimental native backends are included. Neither is established as better
than Apple's default voice: compare identical passages by listening before choosing.

| Candidate | Neural ONNX weights | Installed runtime assets |
| --- | ---: | ---: |
| Inflect Nano v2 FP32 | 16.21 MB | 19.84 MB |
| Kitten Nano 0.8 INT8 | 24.37 MB | 31.28 MB |

Decimal MB. Installed totals include the BSD pronunciation dictionary, voice
assets where applicable, configuration and notices. The executable/ONNX runtime
are additional. Inflect fits under 20 MB even with these supporting assets; Kitten
does not. Inflect is one fixed male voice. Kitten offers Bella, Jasper, Luna, Bruno,
Rosie, Hugo, Kiki and Leo. English only; these models do not clone voices.

## Acquisition and use

```sh
python3 scripts/download_compact_english.py inflect
python3 scripts/download_compact_english.py kitten
cargo build --release -p local-tts-speak
./target/release/tts-speak --backend inflect --model voices/inflect-nano-en --output small.wav --text 'Hello world.'
./target/release/tts-speak --backend kitten --model voices/kitten-nano-en --speaker Bruno --output kitten.wav --text 'Hello world.'
```

Studio lists both as experimental English models and provides download buttons;
Kitten adds a voice selector. Rust uses `compact_english::CompactEnglish` with
`CompactKind::Inflect` or `CompactKind::Kitten`. C/C++ loads backend IDs
`TTS_INFLECT_NANO` (6) or `TTS_KITTEN_NANO` (7), then uses ordinary synthesis.
Use English, speaker default for Inflect, and default or a named Kitten voice.
The shared English frontend expands ISO dates, times, prices, quantities, units,
fractions, percentages, ordinals and supported abbreviations. It resolves
contextual homographs across the full passage before model chunking. Studio
provides Pronunciation preview for the expanded words. Ambiguous numeric syntax
is rejected rather than guessed.
Duration scale is 0.5–2; greater values slow speech. Paragraphs are chunked with
120 ms joins. Cancellation is checked between graph calls/chunks; model loading
is synchronous. Unknown dictionary words fail explicitly; add a CMU-format
pronunciation only by creating a separately supported package, because these
publisher candidate packages are hash-pinned.

## Frontend and license boundary

Both publishers use eSpeak-ng in their reference frontends. This project instead
uses its existing native BSD CMU dictionary and contextual English normalization.
This avoids bundling eSpeak-ng but is **not frontend parity** with the publishers.
Names, stress, punctuation and prosody may differ. No upstream quality benchmark
is claimed for this alternative pipeline. Listening, intelligibility and Apple
comparisons remain pending even when inference/PCM checks pass.

Model/export declarations are Apache-2.0. Publisher notices are retained with
models; vocabulary attribution and licenses are also copied into release notices.
`assets/compact-english-assets.json` pins exact revisions/files, and
`assets/compact-notices/UPSTREAM.json` records vocabulary/reference sources.
Kitten style arrays are extracted deterministically from its verified NPZ using
Python standard-library acquisition tools; Rust inference uses raw f32 arrays.
No Python or network access occurs during inference.

- https://huggingface.co/owensong/Inflect-Nano-v2-ONNX
- https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8
- https://github.com/KittenML/KittenTTS

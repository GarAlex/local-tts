# German and Spanish

German and Spanish use the local Chatterbox multilingual backend in Rust, the
C ABI, CLI and Studio. They share the downloaded model with French. Switching
languages in Studio reuses the loaded engine.

## Try speech

Download once with `python3 scripts/download_chatterbox.py`. In Studio choose
**European languages — Chatterbox**, choose **German** or **Spanish**, select the
model folder and press **Generate**. The default folder in this checkout is
`voices/chatterbox-multilingual`. A packaged Studio installation can browse to
that same folder. No server or API key is needed.

```sh
cargo run --release -p local-tts-cli -- chatterbox-synthesize \
  voices/chatterbox-multilingual target/german.wav de \
  'Grüße aus Köln. Die Straße führt über eine Brücke.'
cargo run --release -p local-tts-cli -- chatterbox-synthesize \
  voices/chatterbox-multilingual target/spanish.wav es \
  '¡Buenos días! El niño está cerca del río.'
```

Rust uses `Language::De` and `Language::Es`. C/C++ uses `TTS_CHATTERBOX` with
`TTS_GERMAN` (3) or `TTS_SPANISH` (5), speaker `default`, duration scale `1.0`.
`examples/cpp/european_test.cpp` demonstrates loading one engine and generating
French, German and Spanish audio. Output is mono at 24 kHz. There is no regional
accent selector; accent consistency needs listening assessment.

Write numbers, dates, units and abbreviations as spoken words. Accents, umlauts,
ß, ñ and inverted punctuation can be entered normally. The model tokenizer uses
NFKD decomposition; dataset and project VITS text use NFC normalization.

## Make a voice dataset and fine-tune

In **Build dataset**, choose German or Spanish before creating the dataset.
Follow the built-in six-step guide to document recording permission, choose
sentences, record and review clips, and reserve held-out evaluation clips.
Each language includes twelve original starter prompts. These introduce the
recording workflow; they are not enough to train a usable voice.

Fine-tuning needs a same-language **project VITS checkpoint** with every character
in the transcripts already present in its token inventory. Include German
umlauts/ß or Spanish accents/ñ/ü in the base inventory. Preflight rejects language
and inventory mismatches. Studio now offers German and Spanish CSS10 base
import/select controls. The downloaded Chatterbox ONNX files are for inference and cannot
be selected as a VITS fine-tuning checkpoint.

CSS10 import preserves source tokens and speaker/language conditioning. Full
weight notices and recording/consent provenance remain unresolved; see
[public-base setup and verification](PUBLIC_DE_ES_BASES.md).

## Verification

`python3 scripts/verify.py --european` runs the normal Rust checks, all seven added
languages through the installed model, tokenizer/cancellation checks, French,
German and Spanish project VITS adaptation/resume/export tests, and the C++
consumer. It needs the downloaded model and the configured `.venv` training
environment. Training fixtures use synthetic tones to verify mechanics and native
export parity; they do not demonstrate trained-speaker quality.

The release process accepts `--multilingual-voice voices/chatterbox-multilingual`
to test French CLI and French/German/Spanish C++ synthesis after relocating the
SDK into a path containing spaces. Models remain external to release archives.
See [model provenance and shared limitations](FRENCH.md) and
[release packaging](RELEASE.md).

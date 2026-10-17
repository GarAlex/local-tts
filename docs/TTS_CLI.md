# Inference-only command-line app

`tts-speak` is a separate Rust application built on local-tts-core. It contains
no training commands, downloads, GUI, dataset tools or Python subprocesses.
Use installed, hash-pinned model folders. The existing `local-tts` developer CLI
remains available for training/export workflows.

```sh
cargo build --release -p local-tts-speak
./target/release/tts-speak --model voices/kokoro-en-q8 --output hello.wav --text 'Hello world.'
printf 'Hello world.' | ./target/release/tts-speak --model voices/kokoro-en-q8 --output piped.wav
./target/release/tts-speak --model voices/kokoro-en-q8 --output book.wav --input chapter.txt
```

Input files/stdin must contain UTF-8 text, at most 1 MiB. Backend-specific text
limits still apply. Output is mono 16-bit WAV, atomically committed only after
successful synthesis. Existing files are preserved unless `--force` is specified.
Diagnostics go to stderr; success exits 0 and errors exit 1. Parent folders must
already exist. `--help` and `--version` do not load a model.

Other supported native backends:

```sh
./target/release/tts-speak --backend vosk --language ru --model voices/vosk-ru --output russian.wav --text 'Привет.'
./target/release/tts-speak --backend piper --language uk --model voices/piper-uk --output ukrainian.wav --text 'Привіт.'
./target/release/tts-speak --backend chatterbox --language en --model voices/chatterbox-multilingual --reference reference.wav --output cloned.wav --text 'Welcome.'
./target/release/tts-speak --backend qwen-design --language en --model voices/qwen-voice-design --description 'A calm adult male voice.' --output designed.wav --text 'Welcome.'
```

Supply your actual model paths. Default speakers are af_heart (Kokoro), F01
(Vosk), lada (Piper), and default (VITS/Chatterbox). Use `--speaker` to select
another supported speaker. Exported fine-tuned VITS voices use `--backend vits`.
Chatterbox accepts `--profile FOLDER` instead of `--reference`; language selection
is explicit and defaults to English. Qwen requires a description. Unsupported
backend/option combinations fail instead of silently ignoring controls.

`--duration-scale` accepts 0.5–2 for Kokoro/Vosk/Piper/VITS. Model loading and
synthesis are synchronous in this standalone app. It writes files rather than
playing through an audio device. No app windows or Dock icons are created.

Compact experimental English backends `inflect` and `kitten` are also available.
See [COMPACT_ENGLISH.md](COMPACT_ENGLISH.md) for model sizes, acquisition, voice
names and the alternative pronunciation frontend's limitations.

## Use from other tools

Brand names and other words missing from the CMU dictionary fail by design.
Supply them in a CMU-format lexicon (English `kokoro`, `inflect`, `kitten` and
CMU IPA `vits` voices); lexicon entries win over the bundled dictionary:

```text
;;; brand words — WORD then ARPAbet phones with stress digits
PromoShot  P R OW1 M OW0 SH AA2 T
macOS      M AE1 K OW0 EH2 S
```

```sh
./target/release/tts-speak --model voices/kokoro-en-q8 --lexicon brand.dict --json --output promo.wav --text 'Meet PromoShot.'
```

Words may contain only ASCII letters and apostrophes; symbols such as `&`, `…`
or `/` and words with digits (`MP4`) still need spelling out in the text.

`--json` prints one line on stdout after the WAV is committed:
`{"output","format":"wav","seconds","sampleRate","backend","language","speaker"}`.
Errors stay on stderr with exit 1.

`--list-voices [--backend NAME] [--model FOLDER]` prints the speakers a backend
accepts, its default, its languages and whether it takes `--lexicon`, without
synthesizing. Piper and VITS read their package, so they need `--model`; the
other backends answer without one. Exported VITS voices default to their first
speaker.


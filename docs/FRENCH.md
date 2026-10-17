# French and European language support

French is the first added European language being validated end to end. The
language catalogue now also includes German, Spanish, Italian, Portuguese,
Polish and Dutch, alongside English, Russian and Ukrainian. Language IDs are
stable: en=0, ru=1, uk=2, de=3, fr=4, es=5, it=6, pt=7, pl=8, nl=9.

## Ready-to-use French voice

The native Rust Chatterbox backend runs the pinned multilingual ONNX release
locally on CPU. It uses a Q4 language model with FP32 acoustic components,
the publisher's default reference voice and nucleus sampling. An optional
consenting-speaker WAV can replace that reference at load time for zero-shot
voice cloning. Download size is about 1.56 GB. Python is used only to acquire files; inference stays in Rust and
ONNX Runtime, in process, without a server or network access.

```sh
python3 scripts/download_chatterbox.py
cargo run --release -p local-tts-cli -- chatterbox-synthesize \
  voices/chatterbox-multilingual target/french.wav fr \
  'Bonjour, le soleil entre par la fenêtre.'
```

In Studio select **Multilingual / voice cloning — Chatterbox**, keep **French** selected,
then download the model or browse to its folder and press **Generate**.
For cloning, choose an optional speaker reference; **Prepare / trim speaker reference**
imports a mono/stereo WAV (up to five minutes), converts it directly to mono
24 kHz, and lets you choose a 0.1–30 second segment. Preview the selection,
then **Save and use reference…** to create and select a separate file. Use
5–15 seconds of clean speech for better reference quality. The same
backend offers en/ru/de/fr/es/it/pt/pl/nl. Other existing backends retain their controls.
Portuguese does not expose a Portugal/Brazil dialect selector. Native-speaker
assessment, accent consistency and speech quality remain to be evaluated.

Rust callers use `chatterbox::Chatterbox` with `Language::Fr`. C/C++ callers use
`TTS_CHATTERBOX` and `TTS_FRENCH`, speaker `default`, duration scale `1.0`.
Existing ABI IDs and ownership rules are unchanged. The C++ example accepts:

```sh
./speak voices/chatterbox-multilingual target/french-cpp.wav 4 default \
  'Bonjour, le soleil entre par la fenêtre.' chatterbox
```

Chatterbox applies the upstream model's NFKD Unicode decomposition before its
BPE tokenizer; project VITS datasets retain their NFC grapheme contract. Token
IDs outside the pinned embedding vocabulary are rejected before ONNX inference.
Text is bounded and split into short chunks. Cancellation is checked between
model calls, including autoregressive token steps. A token-limit failure is an
error, rather than silently returning truncated speech. The default remains
fixed for reproducible tests. Custom-reference cloning is available through
`Chatterbox::load_with_reference` and the additive C ABI
`tts_engine_load_with_reference`; references must be non-silent mono 24 kHz WAV files between
0.1 and 30 seconds (5–15 seconds of clean speech is recommended).
Studio invalidates cached conditioning when reference contents change.
**Expressiveness** exposes the model’s exaggeration input from 0 to 1 (default
0.5); higher values can speed speech up. Rust callers use `set_exaggeration`
and C/C++ callers use `tts_engine_set_exaggeration`. Invalid values leave the
previous value unchanged. This adjusts delivery intensity, without selecting
a mood or designing a speaker. Chatterbox fine-tuning and duration controls
are not exposed. Obtain permission before cloning a real speaker.

## Datasets and own-voice training

The dataset language selector includes all ten languages with twelve original
starter prompts each. French preserves accents, ligatures and normalized Unicode
spelling; accents are letters, not the Russian/Ukrainian stress annotation
syntax. The prompts are a recording introduction, not sufficient training data.
French spellings such as `é`, `è`, `ç` and `œ` must be represented in a base
checkpoint's token inventory before adaptation.

Fine-tuning remains the recommended workflow. Select a **same-language project
VITS checkpoint** whose inventory covers the transcripts. Preparation, preflight,
fine-tuning, resume, held-out comparison, evaluated export and native inference
accept the new language codes. Language/inventory mismatches fail explicitly.
A pinned French Neon CSS10 base can now be imported with Studio or
`scripts/prepare_european_base.py fr`. Italian, Portuguese, Polish and Dutch
have the same workflow with their own checkpoints. See
[European training bases](EUROPEAN_TRAINING_BASES.md) for contracts, commands,
verification and unresolved source rights.

The new languages currently accept text written as spoken words. Spell out
numbers, dates, units and abbreviations. Structured normalization and contextual
pronunciation rules remain implemented only for EN/RU/UK. New languages do not
borrow English or Cyrillic number rules. French training smoke tests use
synthetic tones and validate mechanics only, not a usable trained speaker.

## Provenance and commercial distribution

The [ONNX publisher](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX)
declares MIT; [upstream Chatterbox](https://github.com/resemble-ai/chatterbox)
also supplies MIT terms. `assets/chatterbox-assets.json` pins the ONNX revision,
per-file SHA-256, default reference, tokenizer, model card and upstream license.
The loader checks every pinned asset before use. Retain these notices with the
model. Repository license declarations do not establish every training recording
or reference speaker's rights; the default reference's consent lineage is not
independently established here. Binary release archives still exclude models.

The Rust tokenizer uses Apache-2.0 with HTTP features disabled. No eSpeak/GPL
frontend was added. Updated dependency notice collection is required for new
releases; previous release manifests describe their own frozen dependency trees.

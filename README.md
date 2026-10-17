# Local TTS

Offline Rust TTS library with a simple egui desktop application and own-voice training tools. The application calls the same library that other projects can embed. No inference server or Python runtime is required for synthesis.

## Run the desktop app

```sh
./run.sh
```

The launcher builds Studio and opens it, and avoids launching another copy if Studio is already running. On macOS it uses a development app bundle with microphone recording permission support.

To create that bundle manually:

```sh
python3 scripts/bundle_macos.py
```

Open `target/Local TTS Studio.app`. This development bundle still uses the workspace for training scripts and artifacts; it is not a standalone release installer.

### Test speech

1. Select English/Kokoro, Russian/Vosk, Ukrainian/Piper or your exported VITS voice. All three pretrained packages are installed in this workspace; download buttons reacquire pinned assets.
2. Enter a sentence, then Generate, Play, or Save WAV.
3. Adjust the duration multiplier to change speaking speed.
4. Cancel synthesis stops at the next model chunk boundary and discards partial output. The current model invocation must finish first; the loaded model stays available for the next request.
5. Adjust bass/warmth, brightness, compression, pitch and formants in **Voice effects**, then Apply effects. Pitch retains duration; formants adjust vocal resonance independently. Select Original or Processed for playback/WAV output. Each pass uses the original; no retraining is needed. See [native effects and embedding](docs/VOICE_EFFECTS.md).

The initial pretrained backend is Kokoro English / `af_heart`. Studio supports **Kokoro variant** selection for FP32, Q8 and Q4 packages. See [quantized models and checks](docs/KOKORO_VARIANTS.md); Q4 retains substantial FP32 weights and is larger than Q8. A native BSD-licensed CMU dictionary produces Kokoro phonemes. The shared text-v1 frontend expands integers below one billion, decimals, supported dates/times, grouped numbers, currencies, measurements, percentages, ordinals, signs, fractions, ranges and selected abbreviations. Leading zeroes in bare numbers are spoken digit by digit. See [supported forms and limits](docs/LANGUAGE.md); unsupported or ambiguous notation needs an explicit spoken spelling. Unknown words fail explicitly: add a CMU-format dictionary pronunciation rather than silently generating an incorrect word. Pronunciation uses a dictionary with limited adjacent-word homograph cues; broad contextual disambiguation remains unfinished. Text synthesis now chunks paragraphs at sentence/word boundaries and joins them with 120 ms silence. Requests and encoded text are capped at 10,000 characters; total output is capped at ten minutes. Direct phoneme input remains limited to 510 characters. Chunk boundaries may affect prosody; this is blocking generation, not streaming.

Russian provides five speakers and dictionary pronunciation with manual stress
overrides. Ukrainian provides three speakers with a native text frontend. Neither
requires Python or eSpeak at inference time. Put `+` before a stressed vowel (or
use an acute accent); stress markers are optional and are not spoken. Plain
Cyrillic integers/decimals now expand automatically with stress on generated
number words. Studio offers a Pronunciation preview and **Saved word stress**:
save `Прив+іт` once, then type plain `Привіт` in later sentences. Corrections
persist beside the voice and can be reloaded by Rust/C++ hosts without retraining. Supported dates and units expand automatically; see [language forms](docs/LANGUAGE.md). See [Cyrillic voices and license
records](docs/CYRILLIC.md) for commands, limitations and the corrected Ukrainian
dataset provenance record.

### Train your own voice

1. In Build dataset, follow **First dataset: step-by-step guide** for a beginner walkthrough. Create a folder and choose English, Russian or Ukrainian. Document rights and speaker consent. Import mono/stereo WAV files or record the default microphone, enter exact transcripts, trim and preview each take, then Add clip. Originals remain untouched; saved clips are mono 22050 Hz.
   Guided recording prompts offers starter lists and custom text-list import, with progress matched against saved clips. Assign held-out splits, Save edits, Check saved dataset, then Use for training. Alternatively, prepare a [dataset manifest](docs/DATASET.md) yourself.
2. In Train a voice, keep **Fine-tune (recommended)** selected, choose a trusted compatible project VITS `.pt` base, or **Download / import English base** / **Download / import Russian base** / **Download / import Ukrainian base**, then select that base. Choose the manifest and a new checkpoint folder. The [LJSpeech import](docs/PUBLIC_BASE.md) is implemented; its originating weight-license record and speech quality still need evaluation. The [original Ukrainian Piper checkpoint](docs/PUBLIC_UK_BASE.md) also supports fine-tuning; its data/consent lineage remains under audit. [Original Russian Vosk weights](docs/PUBLIC_RU_BASE.md) now support fine-tuning with shared native phonemes and normalized vocoder export; dictionary/data/consent rights remain under audit. Installed ONNX voices remain inference-only. **Train from scratch** remains an explicit fallback.
3. Set up training environment, then Use project environment. Python 3.10+ is required only for training/export. The environment is already installed here.
4. Check dataset and **Check base and target compatibility**, select total steps/batch size, and Start training. CPU and CUDA are supported; MPS is not currently supported. CPU is suitable for engineering checks; full training is substantially more expensive.
5. Stop job saves at a step boundary on macOS/Linux. Resume continues to the specified **total** step count, not that many additional steps. Windows stop terminates immediately and retains the last periodic checkpoint.
6. Listen to the held-out preview, then **Evaluate before / after…** for recording/base/adapted listening samples. Choose a checkpoint and **Keep evaluated checkpoint** to preserve it. **Export evaluated checkpoint**, then **Test evaluated voice** to use exactly that checked snapshot. The training tab's general export still uses latest. See [evaluation tools](docs/EVALUATION.md).

The initial trainer uses single-speaker VITS, a deterministic duration predictor for project scratch models, and preserves the stochastic predictor for imported LJSpeech weights. Versioned native grapheme and CMU IPA frontends are shared with inference. New preparations use NFC/lowercase/whitespace normalization and accept manual stress annotations for Russian/Ukrainian. It accepts English, Russian and Ukrainian datasets; Russian/Ukrainian quality has not been evaluated; limited phrase stress rules are implemented, while general automatic stress prediction remains unfinished. No usable custom voice can be produced without suitable speech data and enough training. Fine-tuning starts a separate target run from compatible project VITS weights, retaining the base inventory and resetting optimizer/steps. It does not fine-tune Kokoro. See [training modes and limitations](training/README.md).

Checkpoints save every 100 steps and on normal completion/graceful stop. Training token configuration and audio hashes prevent resuming with changed inputs. Checkpoints are local trusted PyTorch files: do not load untrusted downloaded `.pt` files.

## Library and CLI

`local-tts-core` exposes `Synthesizer`, `SynthesisRequest`, `Audio`, `Kokoro`, `Vits`, `vosk::Vosk`, `piper::Piper`, dataset validation and training preparation. Load a model once and reuse it. Playback belongs to the host application. See [Rust consumer example](crates/tts-core/examples/speak.rs).

EN/RU/UK [text normalization](docs/LANGUAGE.md) supports dates, measurements,
prices, percentages, ordinals, fractions, explicit ranges and whitelisted
abbreviations, with limited phrase-based pronunciation/stress cues. Studio shows
the expanded text in Pronunciation preview. The same model-free helper is exposed
to Rust and C/C++; new trained voices record its version while legacy voices and
resume preparation keep their original contract.

```sh
cargo run -p local-tts-cli -- --help
cargo run -p local-tts-cli -- synthesize voices/kokoro-en speech.wav 'Hello world.'
cargo run -p local-tts-cli -- vosk-synthesize voices/vosk-ru russian.wav F03 'Привет, мир!'
cargo run -p local-tts-cli -- piper-synthesize voices/piper-uk ukrainian.wav lada 'Прив+іт, св+іте!'
cargo run -p local-tts-cli -- dataset-check /path/to/dataset.json
cargo run -p local-tts-cli -- training-prepare /path/to/dataset.json /path/to/prepared.json
cargo run -p local-tts-cli -- vits-verify /path/to/exported/voice
cargo run -p local-tts-cli -- vits-synthesize /path/to/exported/voice speech.wav 'Text in your voice language.'
```

Voice metadata checking verifies paths and schema, not legal clearance or graph compatibility. VITS numerical verification compares native ONNX output to PyTorch at several text lengths and duration scales.

## Verification

Run `python3 scripts/verify.py --full` for the complete local engineering checks
and a JSON report in `target/verification.json`. It requires the training
environment and downloaded English assets. See [implementation audit](docs/IMPLEMENTATION_AUDIT.md)
for verified features and remaining limitations.

```sh
python3 scripts/verify.py --cyrillic # requires downloaded RU/UK packages
python3 scripts/verify.py --russian-base # original Russian checkpoint adaptation/native parity
python3 scripts/verify.py --ukrainian-base # original Ukrainian checkpoint adaptation/parity
python3 scripts/verify.py --kokoro   # requires downloaded FP32/Q8/Q4 packages
python3 scripts/verify.py --public-base # requires downloaded/imported LJSpeech base
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
.venv/bin/python tests/training_smoke.py
.venv/bin/python tests/alignment.py
```

The training smoke test uses synthetic tones to exercise training, resume, preview, export and native numerical parity. It is not a speech-quality benchmark.

## Release packages

Run `python3 scripts/release.py` on macOS to build a relocatable Studio app, C/C++ SDK and source archives with checksums, notice inventory and relocation checks. Add `--voice voices/kokoro-en --training-python .venv/bin/python` for inference and installed training/export validation. Outputs go to `dist/`. Use `--package sdk` for an independent library/CLI release. The SDK and Studio dependency notice inventories are complete; dispatch 0.2.0 is replaced by the project’s own MIT adapter in `crates/dispatch-compat`. Public signing/platform checks remain. See [release process](docs/RELEASE.md) and [repository review](docs/RELEASE_REVIEW.md).

## Commercial distribution

Project code is MIT. VITS source is vendored under its MIT license; Kokoro weights are Apache-2.0; the CMU dictionary is BSD; Noto font is SIL OFL-1.1. Required notices are retained. No eSpeak or Piper implementation is linked. Vosk weights/source declare Apache-2.0; the Piper repository declares MIT. Per-package provenance gaps are recorded in [Cyrillic voices](docs/CYRILLIC.md). Fine-tuned weights retain base-model terms; target data rights and speaker permissions also apply. Base lineage is recorded in exports. Scratch training remains available. See [third-party records](docs/THIRD_PARTY.md).

The C ABI now builds shared/static libraries with a tested C++ consumer. See [embedding instructions](docs/EMBEDDING.md). Native macOS candidate packaging is implemented; public release gates, other platforms, complete language frontends and production performance work remain in the [project phases](docs/PHASES.md).

### French and European languages

Select **Multilingual / voice cloning — Chatterbox** in Studio for French (default), German,
Spanish, Italian, Portuguese, Polish or Dutch. Download the pinned model with
`python3 scripts/download_chatterbox.py`. Inference is native Rust/ONNX, including
the C ABI; datasets and same-language project VITS fine-tuning accept these
languages. Studio and the C ABI also accept an optional permitted mono 24 kHz
reference WAV (up to 30 seconds) for zero-shot voice cloning. New-language text
currently needs numbers written as words. Chatterbox is inference-only; a
compatible own-voice training checkpoint is separate.
See [French setup and limitations](docs/FRENCH.md) and
[German and Spanish setup, training and checks](docs/GERMAN_SPANISH.md).

French, Italian, Portuguese, Polish and Dutch now have pinned public-base import
and same-language fine-tuning controls. See [European training bases](docs/EUROPEAN_TRAINING_BASES.md)
for setup and source-rights limitations.


Saved reference voice profiles are available in Studio, Rust, CLI and the C/C++
ABI. A profile copies its reference and stores language/expressiveness settings
without bundling model weights. See [profile workflow](docs/VOICE_PROFILES.md).

For an inference-only application, build `local-tts-speak` and run `tts-speak`.
It accepts argument/file/stdin text and writes WAV using the native library.
See [the dedicated CLI guide](docs/TTS_CLI.md).
Other tools can call it as a subprocess: `--json` prints the result, `--list-voices`
lists speakers, and `--lexicon` adds brand-name pronunciations. See the
[PromoShot integration plan](docs/PROMOSHOT_INTEGRATION.md) for embedding it in apps.

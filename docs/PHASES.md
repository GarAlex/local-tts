# Project phases

## Current progress

- Phase 1: Rust contracts, schemas, CLI and WAV validation implemented. Training preparation also detects identical file contents and records audio hashes. Validation reports include speaker/split clip counts, validated duration and shortest/longest clips.
- Phase 2: Native Kokoro English FP32/Q8/Q4 inference with Studio package selection, BSD dictionary frontend, audio playback/WAV export and egui controls implemented. Bounded paragraph chunking is implemented for both backends. Versioned text-v1 expands dates, times, measurements, prices, percentages, ordinals, fractions, explicit ranges and selected abbreviations. Limited English homograph cues are implemented; broad contextual pronunciation, unsupported formats and release audit remain.
- Phase 3: Dataset editor with WAV import, microphone recording, waveform trimming, transcript review and deterministic grouped splits implemented. A six-page in-app beginner guide covers dataset setup through fine-tuning. Guided recording supports original EN/RU/UK starter prompts, custom lists and resumable transcript matching. Native library saves training-ready clips with external-edit protection. Initial single-speaker VITS trainer, native grapheme token preparation, stop/resume, held-out preview, ONNX export and native parity checks implemented. Checkpoint-based fine-tuning is now the default workflow, with base token inventory preservation, fresh target optimizer/step count and exported initialization lineage; pinned public English LJSpeech import, preserved stochastic predictor, shared native CMU IPA and Studio acquisition/selection are implemented. Original Ukrainian Piper and Russian Vosk training checkpoint import, fixed source conditioning, native shared text tokens and adaptation/export are implemented. English originating weight terms, Ukrainian data/consent lineage, Russian dictionary/data/consent lineage and real speech evaluation remain required. Engineering smoke tests use synthetic tones; real speech training and quality evaluation remain.
- Phase 4: Pretrained Vosk 0.7 Russian (five speakers) and Piper Ukrainian text voice (three speakers) now run natively through Rust, C/C++ and Studio, with pinned acquisition and manual stress. Native RU/UK plain cardinal/decimal expansion now includes lexical stress, and Studio offers a pronunciation preview and clear optional-stress help. Editable per-voice saved word stress is implemented in the library, CLI, C ABI and Studio, with reload without reloading weights. Structured dates/quantities/abbreviations and limited phrase stress rules are implemented with explicit/saved precedence. Arbitrary grammatical case and general automatic stress prediction remain unfinished. All eight speakers passed real-model integration checks; provenance gaps are recorded in CYRILLIC.md. Russian/Ukrainian dataset identifiers and Unicode grapheme training supported, with synthetic training/export/native text inference checks. Versioned grapheme-v2 adds manual stress overrides shared by preparation and inference, with legacy v1 compatibility. Original Ukrainian Piper and Russian Vosk checkpoint imports add separately versioned shared native frontends and tested fine-tuning/export paths. Broader contextual stress and real speech quality remain unvalidated. Russian phonemes share the original native frontend; Ukrainian retains its text model.
- Phase 5: egui consumes the reusable Rust library. C ABI v1 shared/static builds and a C++ RAII example implemented; macOS shared-library inference, errors, PCM ownership and concurrent calls tested. Cooperative cancellation implemented for Rust, C/C++ and egui at model chunk boundaries. Model-package SHA-256 verification and VITS export manifests implemented. Native macOS candidate packaging now includes relocatable app/SDK/source archives, notice inventory, hashes, linkage/relocation checks and a Developer ID/notarization path. Public notice gaps and signing/clean-machine validation remain; Linux/Windows packaging remains pending. See RELEASE.md and RELEASE_REVIEW.md.

## Requirements

- Offline, in-process Rust library, followed by a stable C ABI for embedding.
- English first, Russian and Ukrainian next; extensible language-specific frontends.
- Commercial distribution without royalties, noncommercial restrictions or copyleft requirements in the shipped dependency graph. Standard permissive notices remain.
- Fine-tuning a compatible pretrained checkpoint is the primary own-voice training workflow and a required milestone. Scratch training remains an explicit fallback for creating base models and research. Base weights, data provenance and redistribution terms must fit the commercial requirements.
- Implement project tools here. Python/PyTorch is provisionally limited to development/training; deployed applications require neither Python nor a server.
- Own or suitably licensed recordings/transcripts and documented speaker consent. Rights metadata is a record, not automated legal clearance.

## Phase 1 — Contracts and dataset validation (started)

Deliver Rust workspace, library contracts, CLI, dataset schema, voice package schema and validation tests.

Acceptance: valid WAV datasets pass; missing/corrupt/empty/silent audio, duplicate files, incorrect channel/sample-rate settings and unsafe paths fail. Reports include duration and held-out split warnings. No fake synthesis backend.

Repeatable local verification is available through scripts/verify.py. Remaining: hosted CI and complete dependency license auditing. Legacy VITS uses grapheme-v1; new preparation/export uses grapheme-v2 with shared normalization and explicit stress annotations.

## Phase 2 — Working English inference

Select and pin permissive pretrained English assets; FP32/Q8/Q4 Kokoro packages are included for comparison; begin with Kokoro as the candidate and compare Kitten for size/quality. Record checksums, license texts and provenance for every asset. Implement a compatible native English frontend without eSpeak or other copyleft dependencies. Audit dictionary/data licenses separately from source licenses.

Integrate ONNX Runtime behind the library's synthesis interface, add WAV output and an end-to-end CLI. Verify model tensor contracts, token mappings, voice vectors, punctuation and unknown words. Benchmark on the target machine and listen to a fixed evaluation set before selecting the backend.

Acceptance: understandable English speech produced offline; library consumer example; no Python at runtime; reproducible asset acquisition; measured latency, memory and real-time factor. No performance promise before measurement.

## Phase 3 — Own-voice training toolkit

Implement recording prompt lists and import/preparation tools; resampling, segmentation and transcript review; deterministic train/validation/test splits; training configuration, resume, checkpoint selection and evaluation; fine-tuning from compatible pretrained VITS checkpoints as the main path; ONNX export matching a documented native backend. Retain explicit from-scratch training for producing compatible base models.

Use identical versioned normalization and tokenization in training and inference. Prevent model/data dependencies with incompatible terms. Record training seed, data revision, config and source revision with the exported model.

Acceptance: a small real-data fine-tuning run adapts a compatible pretrained model to a new speaker, resumes correctly and exports; native versus training-runtime output comparison passes agreed tolerances; held-out quality is evaluated. A full usable voice requires a suitable speech dataset and training hardware.

### Required fine-tuning milestones (priority)

1. **Implemented, engineering validation:** new-voice initialization from a trusted
   project VITS `.pt` checkpoint, preserving its complete token inventory and
   language/frontend/sample rate; strict architecture loading; fresh optimizers
   and target step counter; lower adaptation learning rate; resume of the target
   run; base hash/rights/lineage carried into hashed ONNX exports. Studio defaults
   to Fine-tune and asks for a local base checkpoint. CLI requires a deliberate
   initialization/resume/scratch mode. Synthetic fixtures do not establish voice
   quality or constitute a useful pretrained speech model.
2. **English import implemented, engineering validation:** pinned public LJSpeech
   VITS generator import with restricted external deserialization, strict loading
   of all 838 generator tensors and all 178 token indices, retained stochastic
   duration predictor, explicit fresh discriminator, native CMU IPA, adaptation/
   resume/export and Rust/C++ inference. Model/data records and notices survive
   export. See [public base](PUBLIC_BASE.md): native frontend differs from the
   source's eSpeak frontend; the mirror declares Apache-2.0 but originating weight
   terms remain an audit gap. Commercial provenance confirmation remains required. Russian and Ukrainian import are described
   below. ONNX packages are inference-only.
3. **Implemented, engineering validation:** Studio English base acquisition and
   selection; explicit read-only weight/config/dataset preflight; held-out native
   recording/base/adapted comparisons at fixed duration/noise; checkpoint picker,
   saved evaluated snapshots, hash-bound selected export/test and hashed reports. Comparison binds base lineage and
   the exact prepared target dataset; changed inputs and training split selection
   fail. See [evaluation](EVALUATION.md). Additional public-base importers and real-data quality assessment remain. Training still retains latest
   and explicitly kept checkpoints, not every periodic update automatically.
4. **Ukrainian import implemented, engineering validation:** pinned original Piper
   medium generator/discriminator import with restricted loading; all 256 token
   rows and fixed source conditioning retained; shared native text/NFD/BOS/pad/EOS
   frontend, fresh-target adaptation, resume and Rust/C++ export. Numerical
   comparison with the original Piper ONNX graph passes. Source MIT declaration
   is recorded separately from unresolved model-card/data/consent lineage. See
   [Ukrainian base](PUBLIC_UK_BASE.md). Russian Vosk import/adaptation now uses its separately supported architecture and corrected normalized vocoder export. See [Russian base](PUBLIC_RU_BASE.md).
5. **Russian import implemented, engineering validation:** pinned original Vosk
   generator with all 818 tensors, 62 phone indices and source conditioning; shared
   Rust dictionary/manual-stress preparation; mel posterior, transformer flows,
   duration adversarial/subband spectral adaptation, resume and normalized native
   Rust/C++ export. Studio import/select and evaluation controls use the same
   profile. Publisher ONNX gain mismatch is recorded and separately tested;
   dictionary/data/consent lineage remains unresolved. See [Russian base](PUBLIC_RU_BASE.md).
6. **Required, pending real speech:** adapt a new consenting speaker, measure
   held-out intelligibility/similarity, evaluate native-speaker pronunciation and
   retained language coverage, and verify exported Rust/C++ inference. Record
   actual recording/compute needs from those runs rather than promise a minimum.

Fine-tuning creates a new run on new recordings; Resume continues that run with
its optimizer, step count and unchanged prepared dataset. A target dataset may
use fewer symbols, but the full base inventory remains available in the export.
Unseen symbols and language/frontend/architecture mismatches fail explicitly.
Cross-language transfer requires a separately implemented and evaluated adapter.

## Phase 4 — Russian and Ukrainian

Implement separate normalization, phoneme inventories, pronunciation exceptions and stress handling. Per-word saved stress corrections and limited context phrase rules are implemented for pretrained RU/UK; general prediction remains. Versioned text-v1 handles structured dates, units, prices, ordinals, fractions and ranges; legacy trained packages and resume contracts are preserved. Cover numbers, dates, abbreviations, Cyrillic/Latin mixtures, Russian ё and Ukrainian ґ/є/і/ї. Ukrainian Piper base adaptation is implemented and verified on engineering recordings. Fine-tune a compatible base into separate voices first; Russian Vosk architecture import/adaptation is implemented with engineering fixtures. Next is real consenting-speaker adaptation and held-out listening for all three languages. Do not treat Ukrainian as a Russian configuration.

Acceptance: native-speaker evaluation of held-out sentences, explicit stress overrides, reproducible pretrained-base fine-tuning/export for each language.

## Phase 5 — Embedding and distribution

Deliver C ABI, ownership/error conventions, C++ example, Rust example, shared/static libraries, platform builds, cancellation and documented concurrency. Chunked generation is distinguished from true model streaming. Add model checksums and resource limits; ship third-party notices.

Acceptance: clean consuming applications run offline on selected macOS/Linux/Windows targets. Compatibility matrix and CPU benchmarks are published locally with release artifacts.

## Phase 6 — Quality and expansion

Improve prosody, pronunciation, long text, additional voices/languages and optional hardware acceleration. Add multilingual architecture only after separate language models meet quality requirements.

### Voice effects and learned style controls (in progress)

Implemented: reusable bass/warmth and brightness EQ, gentle compression,
whole-clip peak protection, duration-preserving pitch and independent formant
shifts in Rust, C ABI, CLI and egui. Original/processed
comparison is retained; each pass starts from original PCM. Numeric and C++
consumer checks cover bounded settings, finite PCM, frequency response, dynamics,
ownership, concurrency, transposition, envelope movement and boundary latency.
See [voice effects](VOICE_EFFECTS.md). Listening quality assessment remains.
Pitch/formants use pinned MIT native headers and the existing comparison controls.
Next evaluate moderate voice-character settings before offering age presets.
These audio effects need no model retraining.

Age presets can approximate vocal character by combining pitch, formants and
texture; they must not promise accurate age transformation. Explore Kokoro voice
selection/blending separately, with pinned asset provenance and compatible terms.

Later extend own-model training with explicit style conditioning and labeled
expressive recordings for selectable moods. Use fine-tuning as the primary path; keep scratch training for new base architectures. Mood controls must affect prosody and vocal delivery, with held-out
listening evaluation; speed/pitch adjustments alone do not satisfy this milestone.
This work follows the existing English, training, RU/UK and embedding phases.

## Decisions and inputs still needed

- Target deployment hardware/platforms and latency/quality preferences.
- Recordings and transcript sources for own voices, with rights documentation.
- Available training accelerator; do not assume CUDA is available on this Mac.
- Whether Python in the training environment is acceptable (the current design assumes it is).

These inputs do not block Phase 1; recordings and compute block completing a trained voice.

See [implementation audit](IMPLEMENTATION_AUDIT.md) for checked features, corrected issues and verification limits.

## European expansion — French first (2026-10-04)

- Add stable language IDs and dataset prompts for French, German, Spanish,
  Italian, Portuguese, Polish and Dutch; preserve EN/RU/UK model contracts.
- Integrate pinned MIT Chatterbox ONNX inference in Rust, C ABI, CLI and Studio,
  with French selected by default and explicit backend capability limits.
- Validate French synthesis and C++ embedding; run French project-checkpoint
  fine-tuning/resume/export/parity checks with accented text.
- Keep same-language fine-tuning as the primary own-voice workflow. Acquire and
  validate a suitable French training base separately; do not treat inference
  ONNX weights as a training checkpoint.
- Follow with native-speaker quality checks and the remaining languages' audio
  evaluations. Numeric/contextual expansion for new languages remains pending.

See [French and European language guide](FRENCH.md) for current capabilities,
model provenance, commands and limitations.

### German and Spanish follow-up

German and Spanish are available in Studio speech/dataset selectors, the Rust
library, CLI and C ABI. Their accented-text project VITS checks cover fine-tuning,
resume, held-out comparison, evaluated export and native parity. A shared C++
consumer and relocated release smoke test exercise French, German and Spanish
with one loaded engine. See [German and Spanish](GERMAN_SPANISH.md).

German/Spanish public CSS10 import is now implemented (below). Remaining: collect real
consenting-speaker recordings and evaluate held-out speech with native speakers;
implement new-language numeric/contextual normalization. Synthetic training
fixtures validate the pipeline but do not satisfy voice-quality acceptance.

### Public German/Spanish bases — import and engineering validation

Implemented pinned Neon CSS10 candidate acquisition and restricted inspection of
both real generator checkpoints. All 696 tensors, source/embedded configuration,
129 token rows and speaker/language conditioning are checked. The report retains
`training_ready=false` for raw source files. The importer now converts them to
`css10-de-es-v1` with native `css10-graphemes-v1` text preparation. All 696 tensors
are retained, including shared language and speaker conditioning. Original-network
inference/duration-objective parity checks passed. Studio has import/select
controls; synthetic adaptation tests cover fresh optimizers/discriminator,
resume, held-out comparison and Rust/C++ export. Real speech evaluation remains. Publisher
BSD-3-Clause declarations are pinned; full weight copyright notices and complete
recording/consent provenance remain unresolved. See [candidate contracts and
verification details](PUBLIC_DE_ES_BASES.md).

### Remaining European training bases (2026-10-05, engineering verified)

Implemented pinned same-language base import and fine-tuning for French, Italian,
Portuguese, Polish and Dutch, with Studio import/select controls and reusable
Rust/C++ inference. All five passed original-network comparison, adaptation,
resume, evaluated ONNX export, held-out comparison and C++ consumption. Italian's
140 token rows are preserved separately from the other bases' 129 rows. Existing
German/Spanish contracts and original-network comparisons still pass.

The comparison now checks bounded FP32 error and near-exact FP64 inference;
Dutch exposed accumulated FP32 rounding that disappears in FP64. These checks
establish engineering compatibility. Real consenting-speaker recordings,
native-speaker listening, contextual/numeric language improvements and unresolved
model/data notices remain acceptance items. See
[European training bases](EUROPEAN_TRAINING_BASES.md).


### Zero-shot cloning and voice design (2026-10-05)

- Implemented optional Chatterbox speaker references in Rust, C/C++, CLI and
  Studio. English, Russian and the seven European extension languages can use
  the cloning engine; Ukrainian is unsupported by the pinned model.
- Validate reference headers before model loading, bound recording duration,
  reject silence/non-finite audio, and clear stale Studio playback
  when reference selection changes. Fine-tuning remains the main training path.
- Passed explicit publisher speech reference conditioning for English/Russian
  and independent C++ reference loading for French/German/Spanish.
  Engineering inference success does not establish speaker similarity.
- Implemented native reference preparation (stereo downmix and direct 24 kHz
  conversion) with a Studio save-and-select workflow. Reference content changes
  invalidate Studio’s engine cache. Trimming, preview and save-and-select are now
  implemented and verified in the live Studio app. Portable saved profiles are
  now implemented in Studio, Rust, CLI and C/C++ (2026-10-06).
- Exposed Chatterbox expressiveness (0–1, default 0.5) in Rust, C/C++ and
  Studio. English/Russian and C++ French/German/Spanish synthesis passed with
  non-default intensity. This control adjusts delivery, not speaker identity.
- Native descriptive voice design is implemented using pinned Qwen3 VoiceDesign
  INT4 through Rust, CLI, C/C++ and Studio. English/Russian real synthesis passed.
  Supports EN/RU/DE/FR/ES/IT/PT; UK/PL/NL are rejected. See VOICE_DESIGN.md.
  Existing audio effects remain available for pitch, formants, bass and brightness.
- Remaining acceptance: consenting-speaker similarity/intelligibility evaluation,
  Ukrainian cloning candidate, descriptive style adherence/listening evaluation,
  and public-release readiness checks. The local macOS archives now include the new backend.


### Portable reference profiles (2026-10-06)

Implemented versioned reference packages with copied audio, SHA-256, model revision,
suggested language and expressiveness. Studio save/open restores the reference
and settings; Rust/CLI/C ABI provide the same storage and inference workflow.
Package relocation/source removal, invalid metadata and altered-reference checks
passed, alongside CLI and independent C++ real speech. Live Studio opening,
name editing and saving passed. See [voice profiles](VOICE_PROFILES.md).
Remaining: similarity evaluation with real consenting speakers, Ukrainian cloning,
descriptive style quality evaluation and public-release readiness checks.
Local macOS archives were rebuilt and verified on 2026-10-06.

### macOS dependency notice cleanup — 2026-10-06

Replaced the single narrow legacy dispatch use with an original MIT compatibility
adapter backed by dispatch2. The enabled Studio notice inventory is now complete;
public release still requires signing/notarization and manual platform/model checks.

### Voice-design loading robustness — 2026-10-07

Qwen model verification/loading supports cancellation in Rust and Studio, tested
against the real package and live packaged GUI. Reusing a designed voice through
WAV/reference/profile is now explained in Studio. Refreshed macOS candidate:
local-tts-0.1.0-aarch64-apple-darwin-0f14857080db. Style/listening evaluation,
Ukrainian cloning and public-release manual validation remain.

### Compact English inference — 2026-10-07

Added pinned Inflect Nano v2 and Kitten Nano 0.8 INT8 to the native Rust library,
C ABI/C++ consumer, inference-only CLI and Studio, with verified acquisition.
Inflect runtime assets total 19.84 MB; Kitten totals 31.28 MB (decimal, excluding
executables/runtime). Kitten exposes eight voices; Inflect has one male voice.
Apache-2.0 model declarations/notices and BSD dictionary attribution are retained.
The native dictionary frontend differs from publisher eSpeak pipelines, so
listening/intelligibility and Apple comparison remain pending; both are experimental.

Real-model checks passed for both and all eight Kitten voices, invalid requests
and cancellation. Fresh downloads, independent C++ synthesis, CLI synthesis,
workspace checks, strict Clippy, package relocation, signatures and archive hashes
passed. Live packaged Studio generated both models after fixing a speaker-default
reset discovered during UI testing. Studio was closed after testing.

Updated candidate: dist/local-tts-0.1.0-aarch64-apple-darwin-18bf2a100423.
Standalone CLI: dist/tts-speak-0.1.0-macos-arm64-compact.
These are local ad-hoc signed candidates; public notarization remains pending.
See [compact model guide](COMPACT_ENGLISH.md). This audit entry postdates the
release source snapshot.

### Compact English normalization follow-up — 2026-10-07

Moved normalization/contextual phonemization before compact model chunking to
preserve numeric-unit and abbreviation context across raw length boundaries.
Added Studio pronunciation preview and numeric support guidance. Real Inflect
and Kitten tests passed with dates, times, currency, decimals, units, percentages,
ordinals, fractions and a long numeric-unit passage. Numeric and written-out
inputs produce identical frontend phonemes; graph audio is not assumed bit-exact.
Strict workspace Clippy passed; development Studio was rebuilt and reopened.
Older dist candidates predate this follow-up.

### English arithmetic specialist — fragment baseline

Added bounded English arithmetic parser and PCM16 word-bank builder/loader to
the library, C ABI backend 8 and inference CLI backend math. Generated Kokoro
af_heart baseline: 764314 bytes of runtime data, with source notices/provenance
retained separately. Parser, request validation and integrity tests passed.
Neural student distillation/training and listening comparisons remain planned;
the fragment voice is not a neural distilled model. See MATH_SPEECH.md.

Arithmetic baseline full folder including teacher notices: 788244 bytes.
Real CLI and independent C++ backend-8 synthesis passed; strict workspace
Clippy passed. Sample: target/math-12-plus-225.wav. Studio integration and
neural student training remain outstanding.

Arithmetic baseline added to Studio with expression preview, fixed-duration
generation and normal playback/export/effects. Generated ten comparison pairs
against Kokoro from identical spoken readings (target/math-comparison-v1).
Strict workspace Clippy and development bundle build passed. Manual arithmetic
Studio generation is pending: the user is running Qwen synthesis in the app,
which was left undisturbed. Neural student training remains outstanding.

### English neural arithmetic student experiment

Generated 64 Kokoro utterances, added math-student-v1 train/resume/restricted
checkpoint/export support, ran 16 steps and resumed to 100. ONNX is 2326689
bytes; native PyTorch parity passed (max error 2.8871e-8). Rust MathStudent,
CLI math-neural and C ABI backend 9 consume the shared arithmetic parser.
Real model request tests and CLI/C++ inference passed, with strict Clippy.
This is a mechanical distillation experiment, not a usable trained voice or
verified sub-1 MB neural model. More corpus/data split auditing, training,
quantization and listening work remain. See MATH_STUDENT.md.

Arithmetic curriculum v2 now enforces unique spoken forms and distributes all
16 structure families across train/validation/test. Generated 128 clips (96/16/16),
with no test overlap against v1. Resumed v1 to 1000 steps, then separately
fine-tuned v2 for 200 target steps. Export/native parity passed (max 2.4447e-8),
plus C++ synthesis and corpus/profile/lineage tests. Exported 16 reserved test
teacher/fragment/student triples with hashes and duration/RMS/latency reports.
Model remains 2.33 MB; full v2 export 2.61 MB. One MB is a flexible goal; quality
comes first. Student is still quiet/undertrained; no usable-voice, accuracy or
naturalness claim. Quantization deferred until quality can be evaluated.

### Arithmetic student extended training and evaluation

Resumed v2 to 2000 total fine-tuning steps. Preserved generator-only 800/2000
evaluation snapshots (~3.14 MB), with source hashes and explicit no-resume
contract. Exported step2000 and validated native parity plus C++ inference.
Added validation split comparison/HTML audio players, teacher-forced diagnostics,
and experimental Studio student selector using the shared cancellable parser.
Observed live Studio student generation/playback; no extra app copies launched.
Validation duration ratio improved 0.795 to 0.961; mel L1 0.8886 to 0.8499
(800 to 2000). Student remains quiet; no intelligibility/naturalness assertion.
Clippy, arithmetic/cancellation tests, snapshot immutability/resume rejection,
lineage audit and HTML audio-path checks passed. Model 2.33 MB; full export
2.65 MB. See MATH_STUDENT.md for limitations and next quality experiments.


### Arithmetic student recovery — 2026-10-08

User listening rejected the rebuilt step-2000 student as mechanical noise.
Recognizable speech is now the first acceptance requirement. Added full posterior,
teacher-aligned mean-prior and seeded text-only audio diagnostics. Added an opt-in
math-only direct prior mel objective, objective-preserving resume checks and
snapshot/export provenance. Gradient test verifies text encoder, inverse flow and
decoder receive finite gradients. A separate 300-update recovery run completed;
aligned mel L1 improved 1.47458 → 1.14664 on one validation diagnostic, but
free-running audio is still weak and short. It is not an accepted speech model.
Export parity and 16 lineage-audited validation triples passed. Next: predicted
alignment and free-running convergence; retain fragment baseline for actual use.
See docs/MATH_STUDENT.md for artifacts and reproduction commands.


### Arithmetic convergence investigation — 2026-10-08

Expanded diagnostics across all 16 validation clips, including teacher-aligned
prior audio, predicted durations/RMS and alignment concentration. Exhaustive MAS
checks pass; there is no obvious single-token collapse in these diagnostics.
Completed a separate 600-update recovery run retaining the original trained
discriminator. Aligned-prior mel improved, text RMS increased, but duration ratio
is still 0.71445 and intelligibility is unaccepted. Native parity passed at
4.5634806e-8 maximum sample error. Added tested offline shuffled listening ratings
with local persistence/export and hashed provenance; 16 comparison groups saved
in target/math-listening-retained600.html. No student promoted to usable status.
Next planned: distinct CMU-phoneme student profile and short-utterance curriculum;
accept recognizable text-only speech before longer equations or size reduction.


### Phoneme arithmetic curriculum — 2026-10-08

User rejected retained-discriminator step600 as no voice. Mark it failed.
Implemented distinct math-phoneme-v1 profile with explicit CMU IPA/stress,
877-byte arithmetic lexicon and shared native math parser/CLI/C ABI support.
Generated 128 short teacher phrases (96/16/16); padding meets existing segment
minimum. Curriculum and profile integrity tests, strict Clippy and native export
checks pass. Sixteen scratch updates validate plumbing only; no new voice
candidate or Studio default is promoted. Actual phoneme training convergence and
listening acceptance remain required before long equations or quantization.


### Staged arithmetic recovery — 2026-10-08

Investigated failed student: native inference matches Python; sampled export alone
did not restore speech. Posterior reconstruction also failed, exposing an acoustic
training/capacity problem. Added teacher duration labeling, component-provenance
transfer, function-preserving 64-to-256 decoder widening, frozen-posterior acoustic
fitting and frozen-acoustic phoneme-prior fitting. Trained 3,000 acoustic and
6,000 prior updates on the 96-clip short curriculum; validation/test stay excluded.
The 5,507,081-byte graph passes native parity (1.49e-7 maximum error). Local
unprompted ASR semantically reads four of 16 validation phrases correctly; other
phrases still fail. This is not a usable arithmetic voice or listening acceptance.
Studio experimental default points to the new export with a short example and
accuracy warning. Full lineage-audited comparisons and offline listening page
are saved. Staged gradient/freezing, metadata, alignment, widening, export and
curriculum checks pass; strict workspace Clippy passes.

Remaining: diagnose phoneme latent/duration errors, expand arithmetic vocabulary
and full equations, obtain listening acceptance and reserved-test accuracy, then
consider compression. Details: docs/MATH_STUDENT_RECOVERY.md.


User listening rejected prior-fit6000 as very robotic: FAILED. Recovery now
prioritizes preserving pretrained speech quality with weight-storage compression
(and larger model size), before further architecture reduction. ASR recognition
alone does not constitute voice-quality acceptance.


### Pretrained compression candidate — 2026-10-08

The 5.5 MB student failed user listening and is replaced in Studio's experimental
default by stored-int8 pretrained LJSpeech VITS. Preserved architecture/notices;
172 tensors compressed, graph 114,015,221 -> 29,563,997 bytes; full development
package 34,462,971 bytes. Operations remain float32; runtime RAM/speed reduction
is not established. This is compression, not completed tiny distillation.
Native CLI full-equation sample is transcribed correctly by local ASR. Compared
16 longer validation phrases against the original: mostly matching transcripts,
but both have errors/ASR ambiguities; compressed decimal 0.009 was read as 0.09.
Human quality and numeric acceptance remain pending. Native compressed-graph
fixtures pass, convolution error-bound tests pass, strict Clippy and Studio
build pass. No additional Studio process launched. Listening comparison:
target/math-listening-pretrained-int8.html.


### Numeric-only under-10-MB candidate — 2026-10-08

User accepted the 29.6 MB reference sample quality but requested <10 MB. Implemented
a numeric-only cached acoustic-conditioning architecture: retains pretrained neural
decoder, removes runtime text encoder/duration/flow networks, uses 43 word latents
joined before neural decoding. This is not end-to-end distillation. Contextual
word extraction (one/WORD/one) avoids isolated sentence-final prosody. Shared
Rust MathStudent, CLI math-neural and C ABI backend9 load either format.

Complete contextual package 9,438,483 bytes; decoder graph 8,320,688; conditioning
268,608. Source and dictionary notices retained, source checkpoint/voice fixtures
and hashes validated. Native parity is exact on three decoder fixtures. Main
12+225=237 sample is ASR-correct. All 16 long development clips were checked;
several still have errors/ASR ambiguities (e.g. 0.009 -> 0.09, 274 -> 174,
million-number repetition). Size goal achieved for a candidate; naturalness and
reliable arithmetic speech remain unaccepted. Do not transfer user acceptance
of the larger voice to this design. Studio offers compact/baseline buttons and
keeps the accepted larger voice as initial default.

Native parity, package size limit, synthesis/cancellation, corruption and resealed
invalid geometry tests pass; 4/8-bit rounding tests and strict Clippy pass.
See docs/MATH_LATENT.md and target/math-listening-under10mb.html.


### Numeric conditioning and compounds — 2026-10-08

Continued the under-10-MB numeric-only candidate. Float/packed decoder comparison
showed cached-conditioning failures with both decoders. Added valid arithmetic
contexts, 72 tens/units and nine hundreds readings, schema-2 binary scales and
native fixtures, greedy compound selection, and short repeated-digit gaps.
Final v4 folder: 9,959,101 bytes including notices/fixtures; architecture remains
cached conditioning plus neural decoding, not completed end-to-end distillation.

Native parity, package budget, compound/fraction contracts, cancellation, checksum
and resealed geometry/nonfinite-scale tests pass (five tests on actual v4).
Strict Clippy, Python context/packing/listening tests and Studio build pass.
All 16 native development expressions retained; stronger ASR checks difficult
cases against accepted baseline. 274/74 improves in both ASR models; base.en
reads 418 and 0.009. Repeated 670.70 equation remains a failure/regression.
Naturalness and reliable numeric acceptance remain pending. Studio compact
button selects v4; initial voice remains the accepted larger baseline.
Checked local ZIP plus SHA saved; details and reproducible commands in
docs/MATH_LATENT.md. No extra app instance launched.

### Compact arithmetic: Russian/Ukrainian extension (2026-10-09)

- Implemented stressed literal arithmetic rendering, gender/magnitude forms and
  bounded vocabularies in the Rust library for RU/UK.
- Added reproducible fixed-speaker numeric decoder export from existing pinned
  Vosk/Piper checkpoints; retained source notices and rights qualifications.
- Built Russian 8.84 MB and Ukrainian 2.80 MB experimental packages, CLI WAVs,
  verified ZIPs, public-API/package tests and Studio candidate selectors.
- Pending: listener acceptance, contextual conditioning/compound entries,
  broader Cyrillic numeric accuracy checks. Russian 4-bit compression must be
  compared with its retained 14.77 MB 8-bit baseline. Other European numeric
  candidates follow Russian/Ukrainian quality work.

### Other European compact arithmetic preparation (2026-10-09)

- Extended the numeric exporter profile allowlist to existing German/Spanish
  CSS10 and French/Italian/Portuguese/Polish/Dutch Neon checkpoints, retaining
  fixed source speakers, source notices, and the 22050 Hz contract.
- Loaded all seven checkpoints and checked their decoder output on a 32-frame
  latent input: finite output, shape 1×1×8192. Each decoder has 1,728,768 learned
  parameters. Preflight results and existing rights records are saved in
  `target/math-european-decoder-preflight.json`.
- This is architecture preparation only. Numeric rules, conditioning inventories,
  exported packages, runtime vocabulary contracts and listening checks for these
  seven languages are still incomplete. An automated content filter blocked the
  number-rule implementation during this turn; no partial rule file was created.

### European decoder exports (2026-10-10)

- Added `training/math_decoder_preflight.py`, a reproducible export/check tool
  for all seven existing CSS10/Neon European checkpoints. It validates profiles,
  language/rate, and source checkpoint hashes before exporting fixed-speaker
  8-bit decoders. Source notices and rights records accompany each component.
- Actual decoder graphs are 1,770,045 bytes each. Python CPU ONNX execution is
  checked against the float decoder at three dynamic lengths; these are random
  latent probes, not a speech-quality evaluation. Results live under
  `target/math-european-decoders-q8-v1`.
- Added the Rust `math_decoder_probe` example for checksum, dynamic length and
  finite-output checks on these components. Components deliberately have no
  arithmetic voice manifest and cannot be selected as finished numeric voices.
- Numeric rules and conditioning inventories remain unfinished under the earlier
  content-filter restriction. No European arithmetic quality or final package
  size is claimed.

### European full-text playback check (2026-10-10)

- Generated seven listening WAVs using the existing two-step adaptation smoke-test
  exports and the Rust CLI's VITS backend, with the first original starter prompt
  for each language. Every invocation completed and produced mono 22050 Hz WAVs.
- Comparison page: `target/european-fulltext-listening-v1/index.html`.
  `report.json` records exact text, audio SHA, source export/checkpoint provenance,
  rate and duration. These are existing full-text voices, not new compact numeric
  packages; listener quality acceptance remains pending.
- European frontend regression tests: three passed; two model-dependent
  Chatterbox tests were ignored. No Studio process was launched.

### Cyrillic paired listening checks (2026-10-11)

- Added `math_numeric_compare`, a Rust example that checks full/cached checkpoint
  lineage before generating the same literal arithmetic through both engines.
  The full source receives the exact stressed spoken wording from the shared
  parser; the compact candidate receives the corresponding expression.
- Generated four full/compact pairs per language: ordinary equation, bracketed
  multiplication/division, leading fractional zeroes and repeated decimals.
  All 16 WAVs generated successfully; SHA, timings, duration, rate and source
  model/checkpoint hashes are saved in each report. Clippy passed for the tool.
- Listening page: `target/math-numeric-cyrillic-paired-v1.html`. This is diagnostic
  comparison, not a new model or independent held-out quality result. Listener
  acceptance remains pending; European arithmetic implementation remains blocked.

### Cyrillic contextual conditioning (2026-10-11)

- Implemented opt-in native arithmetic carrier token spans and duration-aligned
  conditioning extraction for the existing RU/UK numeric vocabulary.
- Exported Russian 8.51 MB and Ukrainian 2.54 MB contextual candidates; original
  packages and source rights/notices retained. Both native reference checks pass
  with zero error; eight new CLI WAVs and verified ZIPs produced.
- Studio has separate contextual candidate selectors. Source full/isolated/new
  comparison page is `target/math-cyrillic-context-listening-v2/index.html`.
- Quality acceptance remains pending. Contextual pacing changes are measured,
  not treated as intelligibility evidence. Other European arithmetic work remains
  incomplete under the earlier content-filter restriction.

### Cyrillic ASR diagnostic findings (2026-10-11)

- Extended optional local ASR to explicit RU/UK and multilingual tiny/base;
  prevented accidental English-only recognition. Language guard tests cover
  four incompatible combinations without downloads or output files.
- Generated unprompted multilingual-base transcripts for all 24 full/isolated/
  contextual audio controls; hashes and diagnostics saved and added to the
  listening page. Evaluation checkpoint is development-only, outside TTS packs.
- Contextual v2 is not accepted: Russian equality/operator recognition regresses
  versus its controls. Ukrainian controls themselves are misrecognized, limiting
  conclusions. Studio now labels contextual candidates as diagnostic with
  recognition regressions; previous candidates retained. Listening remains pending.

### Decoder versus conditioning investigation (2026-10-12)

- Verified a byte-identical Russian contextual bank with an 8-bit decoder;
  equality/operator recognition failures persist. Higher decoder precision alone
  does not resolve the contextual regression. Larger model retained as diagnostic.
- Added opt-in independent cached-entry decoding plus waveform edge fades to the
  reusable Rust runtime and exporter. Separate RU/UK diagnostics stay under 10 MB;
  existing default joined-latent behavior is preserved.
- Tested exact independent-waveform joins at three speeds, repeat-zero gaps,
  reference parity, finite output and cancellation in both languages; English
  and original Cyrillic package regression checks passed. Added Studio selectors.
- Eight new CLI samples and local ASR results show retained Russian equation/
  operator readings, but no established improvement over isolated-word baseline;
  decimal accuracy and listening acceptance remain open. Comparison pages are
  linked from MATH_LATENT.md. Other European arithmetic remains unfinished.

### Cached double-zero phrase probe (2026-10-12)

- Added schema-3 support for precisely the cached zero/zero phrase; retained
  strict vocabulary validation and schema-1/2 compatibility. Native frontend
  spans/export preserve original isolated conditioning and decoder weights.
- New RU 8.845 MB / UK 2.808 MB diagnostic packages; source core byte equality
  verified. Runtime tests cover pair grouping/zero counts and frame geometry.
- Russian targeted `0.009 + 1 = 1.009` recognition now succeeds. Extra fractional
  probes still show missing zeroes/operators; Ukrainian ASR regresses. No broad
  accuracy acceptance or default promotion. Russian main-equation WAV unchanged.
- Added labeled Studio diagnostics and a verified 7.38 MB Russian archive.
  Comparison/transcripts: `target/math-double-zero-listening-v4.html`.

### Source-controlled decimal evaluation (2026-10-13)

- Added bounded custom-expression lists to the native full/compact comparison
  tool, including five invalid-input guards and probe-file provenance.
- Generated 16 new source/compact decimal control WAVs and unprompted local ASR
  transcripts. Source models also show some ambiguous zero-count recognition;
  Russian longer fractions still expose compact-specific recognition losses.
  Russian simple tenths transcribe correctly in both; Ukrainian controls remain
  unreliable. No new voice/default promoted.
- Listening page: `target/math-decimal-controls-v1.html`. Human review remains
  requested; audio quality is not established by these recognition transcripts.

### Listening rejection and sequence-student direction (2026-10-13)

**User verdict:** the latest Russian double-zero sample is “Still robotic or
unclear.” The v4 Russian package is rejected for listening quality, regardless
of its successful targeted ASR result. Studio and its listening page now show
that verdict; the original artifact/checksums are preserved for comparisons.
Cached-word/phrase additions are no longer the main quality-improvement route.

New work prioritizes continuous, full-utterance acoustic prediction through the
existing pretrained waveform decoder:

1. Prototype added: `training/math_sequence_student.py`, a 322,273-parameter
   contextual token/duration/frame-latent predictor with positional features.
   Teacher duration expansion uses full-utterance token alignments; padding is
   masked out of attention, acoustic refinement and supervision. It predicts
   decoder-input latents, not stitched cached word waveforms.
2. Contract tests pass for padding invariance, masked supervision/backpropagation,
   invalid duration rejection and parameter budget. The prototype has 1,289,092
   float32 parameter bytes (644,546 at float16), excluding serialization/runtime
   overhead. This is a feasibility estimate, not a measured shipping package.
3. **Not implemented yet:** full-utterance teacher-target dataset, training loop
   with pretrained decoder frozen, free-running duration inference, ONNX/native
   library integration, and held-out speech evaluation. No trained student or
   playable voice has been produced from this module.
4. Fit utterance-level teacher latents/durations first; retain the pretrained
   decoder. Validate continuous output with teacher-forced and predicted durations
   separately, then test compressed-decoder effects. Train/evaluate on disjoint
   spoken utterances; do not train against ASR transcripts or tune only to demos.
5. Human listening acceptance and faithful numeric readings are required before
   default promotion. Under 10 MB remains a size target, subordinate to reaching
   acceptable speech. Preserve source notices and existing rights qualifications.

### Whole-utterance student training pipeline (2026-10-13)

- Added native Russian/Ukrainian sequence-corpus generation with spoken split
  overlap rejection, complete pretrained duration/latent target preparation,
  checksummed datasets, training-only channel normalization, checkpoint/resume,
  validation and preserved source/data rights metadata.
- Prepared 128 full utterances per language (96 train / 16 validation / 16 test).
  Trained Russian 200 steps plus 200 resumed, Ukrainian 200 steps. No source
  decoder weights updated. Reserved test data stayed out of optimization and
  validation; pipeline tests verify held-out isolation and resume/integrity guards.
- Validation normalized latent error: RU 0.8533→0.2774; UK 0.8398→0.4211. These
  teacher-forced losses establish learning, not acceptable speech. Prototypes are
  training checkpoints, not native/playable voice exports.
- Next: predicted durations, waveform reconstruction through frozen decoder,
  held-out listening, native export/frontend integration and measured package
  budget. Full details/reproduction: MATH_SEQUENCE.md. Cached v4 stays rejected.

### Continuous inference and frozen-decoder waveform fitting (2026-10-14)

- Added bounded predicted-duration inference, teacher/predicted-timing diagnostic
  rendering through the frozen full-precision pretrained decoder, waveform
  peak/geometry checks, source/data hashes and spectral/timing reports.
- Generated 24 RU/UK validation diagnostic WAVs. No cached word waveform joins.
  First-case local ASR still makes numeric errors; teacher controls also show
  recognition limits. No listening-quality acceptance or native voice promotion.
- Added optional waveform loss through frozen decoder crops, student-only
  optimization and objective-preserving checkpoint resume. Gradient/input guards,
  padding/duration limits and trainer regression tests pass.
- Russian 100-step waveform pilot (400→500) did not improve matched four-case
  mean spectral error: 0.71786→0.73444. Teacher WAVs are byte-identical. Checkpoint
  retained as diagnostic only. More data/training and matched objective controls
  are needed before native packaging. Details: MATH_SEQUENCE.md.

### Matched 1000-step student controls (2026-10-14)

- Completed two 600-step Russian continuations from identical step-400 state:
  latent-only versus waveform weight 0.03, same dataset/draws. Audited source,
  tool, dataset/checkpoint hashes and identical initial validation metrics.
- Rendered all 16 validation utterances per branch with fixed full-precision
  decoder; teacher WAVs are byte-identical. Spectral error 0.64680 vs 0.63918,
  waveform branch improves 13/16 cases; no convincing numeric/voice acceptance.
  Four representative local ASR probes still contain numeric/operator errors.
- Added tested paired report/audio-integrity comparison tool. No candidate
  promoted. User listening verdict requested on the full-size Russian teacher
  itself before further quality-target-dependent training. No response assumed.
- Comparison: `target/math-sequence-matched-v4.html`; details: MATH_SEQUENCE.md.

### Accepted teacher and expanded sequence training (2026-10-14)

- User accepts the full-size Russian reference voice as the quality target.
  Recorded source/audio hashes; acceptance does not extend to cached/student
  candidates. Earlier cached v4 rejection remains in effect.
- Expanded whole-utterance targets to 512 (384/64/64). Added tested new-corpus
  initialization with source/normalization checks, optimizer reset, and direct/
  cumulative training-versus-held-out overlap rejection. Strict resume remains.
- Trained 1000 further steps from the latent-only branch. New validation latent
  error 0.18166→0.13380. Shared 16-case spectral error 0.64680→0.57734 improves
  all cases with identical teacher audio; numeric recognition still not reliable.
- Verified compact numeric phoneme tables against native full utterances (512 RU,
  128 UK). These are token lookup only, not stitched speech. Native integration
  remains pending. Teacher/student data rights records are preserved.
- Comparison: `target/math-sequence-expanded-v5.html`; details: MATH_SEQUENCE.md.
  No compact student accepted or shipped yet.

### Continuous numeric student: native Russian v6 (2026-10-14)

- Completed 3,000 more training steps on the 512-utterance corpus; validation
  latent L1 0.091455, duration MSE 0.052164, frozen decoder and unused test split.
- Added dynamic ONNX export, phoneme-only lookup and continuous Rust library/CLI
  inference, exposed through existing MathStudent/FFI backend and experimental
  Studio selection. Rust reference waveform error 2.98e-7.
- Packaged candidate: 9,506,053 bytes extracted, 7,958,864-byte ZIP with preserved
  source notices/checksums. Archive CRC and contained SHA-256 verified.
- Numeric contract, integrity/corruption, cancellation and parser regressions
  pass; workspace Clippy clean. No duplicate Studio instance launched.
- First native equation ASR is correct, but this is not user quality acceptance.
  Next: broader numeric fidelity/listening comparison against accepted teacher,
  improve remaining errors, then extend continuous Ukrainian training/export.
  Source-rights qualifications remain; no blanket commercial clearance claimed.

### Numeric timing and decoder controls (2026-10-15)

- Added reproducible identical-latent float/4-bit decoder controls and
  teacher-duration controls. v6 multiplication failure persists with float
  decoding; teacher timing repairs its automated transcription.
- Added persisted configurable duration-loss weight with resume/validation tests.
  Matched 500-step continuations reduce 16-case duration MAE by 4.6% (13/16
  improve), with slightly worse acoustic reconstruction. Float multiplication
  transcription improves; this is not a blanket quality improvement.
- Exported experimental timing-focused v7: 9.50 MB installed, 7.96 MB ZIP,
  native parity 5.83e-7, three native tests and workspace Clippy pass.
- Identical v7 latents through packed decoding garble the multiplication
  operator that float decoding transcribes correctly. Compression remains an
  issue; next test compression alternatives, keeping the download under 10 MB.
- Listening controls: target/math-sequence-timing-decoder-v7.html. No student
  acceptance or default promotion; source-rights qualifications preserved.

### Decoder storage precision sweep v8 (2026-10-15)

- Implemented configurable 4-bit blocks, portable 5-bit storage, selective int8
  weights and reproducible sequence repacking with verified native fixtures and
  archives. Student/phoneme/encoder/acoustic graphs remain unchanged.
- Tested 4-bit blocks 64/32/16, 5-bit, int8 and mixed input/final decoder stages.
  Best reconstruction candidate under the download budget: mixed 4/8-bit,
  9.942 MB ZIP, 11.824 MB installed. Decoder log-spectral difference falls 51.2%
  on four validation clips, but targeted multiplication ASR remains incorrect.
- Int8 transcribes the target correctly but downloads at 13.107 MB. Smaller
  blocks and 5-bit storage did not resolve the operator within the budget.
- Three native integration tests pass; mixed native parity 4.40e-7. Compression,
  stage-selection and package-integrity unit tests and workspace Clippy pass.
- Listening page: target/math-sequence-compression-v8.html. Remains experimental;
  no default/acceptance change, no training or test-set use, no new app instance.

### Decoder fine-tuning infrastructure and negative controls v9 (2026-10-15)

- Tested intermediate residual kernel groups in int8, alone/with final stage;
  no operator repair in six targeted ASR probes.
- Added training-only decoder QAT with straight-through 4-bit weights, a frozen
  original float target, native package export and CRC/SHA-verified archives.
  Student/encoder/acoustic/phoneme components remain unchanged.
- Two matched 200-step trials calibrate on teacher vs native-predicted student
  latents. Validation crop loss decreases, but neither repairs the operator;
  teacher calibration makes the number less clear. No quality/default promotion.
- Both packages download at 7.96 MB and install at 9.51 MB. Native tests pass;
  student-calibrated waveform parity 4.02e-7. Tests cover packed-rounding parity,
  gradient/frozen-teacher behavior, reserved-test isolation and layer selection.
- Updated diagnostics to identify decoder adaptation explicitly. Source rights
  and notices retained. Listening: target/math-sequence-qat-v9.html.
- Next investigate speech-sensitive loss/training stability; do not extend
  training solely because a reconstruction metric improves.

### Ukrainian continuous sequence rollout v10 (2026-10-16)

- Expanded to 512 native numeric utterances, 384/64/64 disjoint train/val/test;
  checked compact phonemes against every complete native utterance.
- Continued Ukrainian prototype for 4,000 steps. Validation latent L1
  0.422294→0.220958; duration MSE 0.075758→0.056785. Frozen decoder/test isolation.
- Exported native int8 candidate: 2.931 MB ZIP / 3.421 MB installed. CRC/SHA
  checks, native fixture parity and three integration tests pass. Experimental
  Studio selector added; shared CLI/library/FFI backend works without Python.
- Generalized decoder QAT for RU/UK and int8 storage. Ukrainian 200-step
  student-latent trial worsens validation loss (0.158987→0.178714), so it does
  not replace the unadapted candidate. Both packages pass native tests.
- Numeric/ASR errors remain; no Ukrainian voice acceptance/default promotion.
  Paired listening: target/math-sequence-uk-v10.html. Source notices preserved.
- Verified all seven existing European decoder component packages. Full numeric
  sequence frontends/datasets/students remain pending for FR/DE/ES/IT/PT/PL/NL.
  Rollout order and explicit gaps: docs/MATH_LANGUAGE_ROLLOUT.md.

### Ukrainian matched continuation v11 (2026-10-16)

- Matched 1,000-step baseline and duration-heavy continuations from v10 show
  the heavier duration objective worsens reconstruction and has no promotion
  case. Baseline improves 14/16 paired reconstruction controls and becomes the
  latest experimental Ukrainian package.
- v11 native package: 2.930 MB ZIP / 3.424 MB installed; Rust native tests,
  archive CRC and checksum verification pass. Numeric ASR errors remain and no
  listening acceptance/default promotion is made.

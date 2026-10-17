# Continuous numeric sequence student

Current state, 2026-10-14: trained research prototypes with Python full-utterance
diagnostic inference for Russian and Ukrainian, **not accepted/native TTS packages**. The user rejected the latest Russian
cached double-zero voice as robotic or unclear. This pipeline replaces cached
acoustic word joins with full-utterance duration and latent prediction; it has
not yet demonstrated acceptable listening quality.

The native Rust `math_sequence_corpus` example renders the project's deterministic
arithmetic curriculum through existing stressed cardinal rules and the pinned
Vosk/Piper frontend. It rejects duplicate spoken readings across splits. Each
128-utterance pilot has 96 training, 16 validation and 16 reserved test utterances.
The preparation command runs the complete pretrained teacher utterance, storing
its exact token durations and 192-channel decoder-input latents. Source checkpoint
and corpus hashes, tensor checksums, source notices/provenance and rights records
are retained. The decoder is frozen during teacher preparation and is not loaded
or updated by latent fitting.

`SequenceStudent` has contextual token representations, positional features,
log-duration prediction, duration expansion, within-token progress features and
frame refinement. Training uses masked, channel-normalized latent L1 and
log-duration MSE. Normalization comes only from training frames. Checkpoints
include optimizer state and can resume against exactly the same hashed dataset.
Validation excludes training and test examples; the reserved test split has not
been evaluated. Merely loading cached test tensors does not use them for loss,
normalization, parameter updates or validation.

Completed pilot fits:

| Language | Steps | Validation normalized latent L1 | Validation log-duration MSE |
|---|---:|---:|---:|
| Russian | 400 | 0.8533 → 0.2774 | 1.8720 → 0.06675 |
| Ukrainian | 200 | 0.8398 → 0.4211 | 2.4214 → 0.08009 |

Russian checkpoints: `checkpoints/math-sequence-ru-prototype-v1` (200 steps) and
`math-sequence-ru-prototype-v2` (400, resumed). Ukrainian:
`checkpoints/math-sequence-uk-prototype-v1`. Their `student.pt` files are training
checkpoints with optimizer state, not shipping models. The Russian float32
student has 322,273 parameters (1,289,092 raw parameter bytes). Serialization,
frontend, decoder, notices and runtime overhead are additional; no final package
size is claimed. Existing compact candidates/defaults have not been replaced.

## Reproduce

Use new paths; corpus, target and checkpoint output folders are never overwritten.

```sh
cargo run --locked -p local-tts-core --example math_sequence_corpus -- \
  ru voices/vosk-ru/dictionary 128 target/new-ru-sequence-corpus.json
.venv/bin/python training/math_sequence_train.py prepare \
  --checkpoint voices/vits-vosk-ru-base/base.pt \
  --corpus target/new-ru-sequence-corpus.json --output target/new-ru-sequence-targets
.venv/bin/python training/math_sequence_train.py fit \
  --dataset target/new-ru-sequence-targets --output checkpoints/new-ru-sequence-run \
  --steps 200 --batch 2
.venv/bin/python training/math_sequence_train.py fit \
  --dataset target/new-ru-sequence-targets --output checkpoints/new-ru-sequence-resume \
  --steps 200 --batch 2 --resume checkpoints/new-ru-sequence-run/student.pt
```

Ukrainian corpus preparation uses `uk unused`; its teacher checkpoint is
`voices/vits-piper-uk-base/base.pt`. The same tools support matching fine-tuned
checkpoints, retaining their source/data rights metadata. This does not establish
new commercial rights or erase the original teacher's provenance qualifications.
All preparation/fitting runs locally; no inference API or new runtime dependency
was introduced into the Rust application.

## Verification and remaining gates

Tests cover padding invariance, masked loss/backpropagation, invalid alignment
bounds and parameter budget. Pipeline tests confirm that altering reserved test
targets does not alter trained parameters; resume increments steps and rejects
another dataset; modified teacher tensors are rejected before output creation.
The Rust corpus example passes Clippy.

Before a voice is offered for listening or selected in Studio:

1. Predicted-duration inference is implemented and tested; validate its speech
   and numeric fidelity separately from teacher-forced timings.
2. Decode continuous predicted latents with the pretrained decoder and evaluate
   waveform/spectral reconstruction, not latent loss alone. Add waveform training
   through frozen decoder weights if latent matching is insufficient.
3. Produce held-out full-utterance comparisons and verify literal numeric fidelity.
   Human listening acceptance is required; ASR is a secondary diagnostic only.
4. Export and integrate native Rust inference. The numeric phoneme frontend must
   fit the shipping budget without including a large general Russian dictionary.
5. Measure the complete package and compare compressed versus full decoder
   output before attempting the under-10-MB target. No random student vocoder.

Python predicted-duration inference and optional frozen-decoder waveform fitting
now exist, but no ONNX/native sequence package, egui sequence voice selector or
listening acceptance exists. Do not confuse diagnostic WAV generation with a
finished TTS model or claim training loss solves the robotic-speech complaint.

## Predicted timing and waveform diagnostics (2026-10-14)

`SequenceStudent.infer` now predicts durations without teacher labels. Inference
requires eval mode and validates token/length shapes, finite durations, scale
0.5..2 and maximum 7740 acoustic frames. Real input tokens receive at least one
frame, padded tokens zero; nonfinite latents and excessive duration are rejected.
Tests verify predicted/explicit-timing equivalence, padding isolation, scaling,
nonfinite rejection and the frame cap. Training checkpoints remain compatible.

`training/math_sequence_evaluate.py` renders three full-utterance paths using
exactly the same frozen **float32** pretrained decoder: original teacher latents,
student latents with teacher timings, and student latents with predicted timings.
It verifies dataset/checkpoint/tensor hashes and records timing errors, WAV
hashes, peaks/clipping and teacher-timed log-spectral differences. Evaluation
uses validation only; the reserved test split remains unused. No audio gain
normalization hides bad/clipped predictions.

Four cases per language produced 24 diagnostic WAVs. The first Russian addition
has 380 predicted frames versus 386 teacher frames, but first-case local ASR
still misreads numeric output. Ukrainian recognition also fails and remains
unreliable even on its teacher controls. These are research samples, not a new
accepted voice. Comparison: `target/math-sequence-validation-v1.html`.

The trainer now supports `--teacher CHECKPOINT --waveform-weight 0.1`. It decodes
64-frame training crops through the frozen teacher decoder and adds multi-scale
log-spectral/convergence loss to duration and latent losses. Optimizer parameters
remain student-only, and any decoder parameter gradient fails training. Resume
inherits the checkpoint waveform weight unless explicitly overridden, and a
positive weight requires the matching source teacher. Tests verify nonzero
input gradients, no decoder gradients/weight changes, and resume objective guards.

A first Russian run resumed step 400 to 500 at
`checkpoints/math-sequence-ru-waveform-v3`. On the same four validation utterances,
with **byte-identical teacher audio**, mean teacher-timed log-spectral error
changed from 0.71786 to 0.73444. Thus this short run did not improve aggregate
reconstruction; recognition errors also remain. It is a diagnostic checkpoint,
not a recommended successor. Report:
`target/math-sequence-waveform-ablation-v3.json`.

```sh
.venv/bin/python training/math_sequence_evaluate.py \
  --student checkpoints/math-sequence-ru-prototype-v2/student.pt \
  --teacher voices/vits-vosk-ru-base/base.pt --dataset target/math-sequence-ru-targets-v1 \
  --output target/new-ru-sequence-render --count 4
.venv/bin/python training/math_sequence_train.py fit \
  --dataset target/math-sequence-ru-targets-v1 --output checkpoints/new-ru-waveform-run \
  --resume checkpoints/math-sequence-ru-prototype-v2/student.pt --steps 100 --batch 2 \
  --teacher voices/vits-vosk-ru-base/base.pt --waveform-weight 0.1
```

Next work should compare longer/more representative training and waveform-loss
weighting against a matched latent-only baseline, then verify free-running
numeric fidelity and human listening. Native packaging must not be mistaken for
solving a reconstruction failure. Source rights/notices remain unchanged.

## Matched longer-training comparison (2026-10-14)

Two Russian branches resumed the same step-400 checkpoint, optimizer and hashed
128-utterance dataset for 600 more steps (batch 2). One used latent/duration loss
only; the other used waveform weight 0.03. Step-seeded training draws are the
same in both, and the initial validation metrics match exactly. Both reached
step 1000. Source and tool hashes and final checkpoint contracts are recorded in
`target/math-sequence-matched-v4-experiment.json`. The reserved test split stayed
out of optimization and evaluation; all comparisons use the 16 validation items.

Full-precision decoder parameters stayed fixed. All 16 teacher WAVs are
byte-identical across the two diagnostic evaluations. Results:

| Validation metric | Latent-only | Waveform weight 0.03 |
|---|---:|---:|
| Normalized latent L1 | 0.18560 | 0.19625 |
| Log-duration MSE | 0.06468 | 0.06533 |
| Mean teacher-timed log-spectral error | 0.64680 | 0.63918 |
| Mean duration frame MAE | 0.67833 | 0.67928 |

The waveform branch improves teacher-timed spectral error on 13/16 cases, but the
mean difference is small (~1.18%). This is reconstruction evidence only, not a
voice-quality claim. Unprompted local ASR on four representative predicted-timing
renders still misreads numbers/operators. In both branches the first addition
is transcribed as 785 plus 17 equals **822**, while the intended result is **802**.
The source transcript/target data are unchanged and the arithmetic parser never
evaluates or repairs equations. Other probe readings are also imperfect.
Neither checkpoint is accepted or promoted into Studio/native release packaging.

Checkpoints: `checkpoints/math-sequence-ru-matched-latent-v4` and
`checkpoints/math-sequence-ru-matched-wave-v4`. Comparison page:
`target/math-sequence-matched-v4.html`; report:
`target/math-sequence-matched-v4-comparison.json`.

`training/math_sequence_compare.py` now provides reusable paired-report checks:
it requires matching source/dataset/language/precision, matching utterance text,
identical actual teacher-audio SHA, valid metric values, and unmodified bounded
relative audio paths. Tests reject mismatched phrases, nonfinite metrics and
modified audio. The tool does not accept speech quality on the user's behalf.

```sh
.venv/bin/python training/math_sequence_compare.py \
  --before target/math-sequence-ru-matched-latent-render-v4/report.json \
  --after target/math-sequence-ru-matched-wave-render-v4/report.json \
  --output target/new-matched-comparison.json
```

The full-size Russian teacher listening check was requested after this comparison
and accepted by the user (see below). That acceptance applies to the teacher
voice only; the cached v4 sample remains rejected and no student is accepted.

## Accepted teacher and expanded corpus (2026-10-14)

The user answered **“Yes, use this voice”** for the full-size Russian reference.
`target/math-russian-teacher-acceptance-2026-10-14.json` binds that verdict to its
reference WAV/hash, teacher model/hash and source checkpoint. The source hash
matches the expanded targets. It is not acceptance of any compact student and
makes no new statement about model/data licensing or source rights.

Prepared 512 complete utterances (384 train / 64 validation / 64 test) using the
same teacher. `--initialize STUDENT --initial-dataset TARGET_FOLDER` now supports
continuation onto a new corpus, distinct from exact-dataset `--resume`. It checks
source teacher/language/vocabulary, verifies the original dataset hash, rejects
previously trained spoken utterances in new validation/test, preserves the old
training-only normalization, and resets optimizer/stage-step state. Checkpoints
carry cumulative trained-spoken hashes so an earlier corpus cannot silently
leak into a later held-out split. Tests cover successful transfer, normalization
preservation and both direct/historical overlap rejection before creating output.

Russian continuation:

- Dataset: `target/math-sequence-ru-targets-v2`.
- Initialized from `checkpoints/math-sequence-ru-matched-latent-v4/student.pt`.
- Output: `checkpoints/math-sequence-ru-expanded-v5`.
- 1000 additional latent/duration steps, batch 2; accepted decoder unchanged.
- New 64-item validation latent L1: 0.18166 → 0.13380; duration MSE:
  0.06166 → 0.05719. Reserved test data remains unused.
- On the 16 validation utterances shared with the earlier corpus, teacher audio
  is byte-identical. Mean teacher-timed log-spectral error improves
  0.64680 → 0.57734 (16/16 cases); duration frame MAE 0.67833 → 0.64712.

This combines additional training, a larger corpus and an optimizer reset; it is
not an isolated causal experiment about data quantity. The comparison tool's
explicit `--shared-cases` option permits different dataset hashes only when
paired utterances/teacher audio/precision/source match exactly. Default behavior
still rejects different datasets. Original and new dataset hashes are retained.

Four local ASR probes now correctly recognize the first addition
`785 + 17 = 802` and leading-zero multiplication `0.009 × 164`. Multiplication
wording and another decimal probe still fail. These narrow diagnostic gains do
not establish faithful speech across numbers or human listening acceptance.
No student/native package/default is promoted.

`training/math_sequence_frontend.py` verifies tiny **phoneme token lookup** tables
against native whole-utterance tokenization: all 512 Russian utterances match
(10,612-byte table), as do 128 Ukrainian utterances (12,063 bytes). Tables contain
no audio or acoustic latents and introduce no word-waveform joins. They are
native-integration preparation, not a new runtime frontend yet. This avoids
planning to ship the large general-purpose Russian dictionary. Tests cover
boundaries/pads, unknown words, invalid boundaries and the 1024-token limit.

```sh
.venv/bin/python training/math_sequence_train.py fit \
  --dataset target/math-sequence-ru-targets-v2 --output checkpoints/new-expanded-run \
  --steps 1000 --batch 2 --waveform-weight 0 \
  --initialize checkpoints/math-sequence-ru-matched-latent-v4/student.pt \
  --initial-dataset target/math-sequence-ru-targets-v1
```

Comparison page: `target/math-sequence-expanded-v5.html`; paired metric report:
`target/math-sequence-ru-expanded-comparison-v5.json`. Full native export,
compression validation, final package measurement and student listening
acceptance remain unfinished. Continue using the accepted teacher as the target.

## Native continuous Russian candidate v6 — 2026-10-14

Expanded-corpus training continued for 3,000 additional steps from v5, reaching
stage step 4,000 (5,000 cumulative steps). Validation normalized latent L1 is
0.091455 and log-duration MSE is 0.052164. Reserved test targets remain unused.
The decoder remains frozen. These training metrics do not establish speech
accuracy or listening acceptance.

`training/math_sequence_export.py` now produces a self-contained native package:
`checkpoints/math-sequence-ru-native-v6` (9,506,053 bytes including notices,
checksums and fixtures), ZIP `target/math-sequence-ru-native-v6.zip`
(7,958,864 bytes). The compressed decoder stores 4-bit weights with float
operations; it is distinct from the accepted full-size teacher reference.
Source recording/initialization/consent qualifications remain in the package.
Compression and distillation do not remove source license obligations.

`MathSequence` in the Rust library executes encoder, duration expansion, acoustic
prediction and one continuous waveform decoder pass. Numeric word lookup contains
phoneme tokens only; it ships no word audio or dictionary and does not stitch
waveforms. `MathStudent`, the existing CLI `math-neural` backend and C ABI
backend 9 recognize `sequence.json` automatically. Studio has an explicitly
experimental v6 selector and recognizes its Russian pronunciation preview.
The accepted/default English voice remains unchanged.

Export checks all 512 corpus phoneme sequences, validates source lineage and
checksums, and compares dynamic encoder/latent graph outputs with PyTorch.
Attention is exported explicitly to preserve dynamic text lengths. Rust waveform
parity across two fixture lengths/speeds has maximum amplitude error
0.000000298. Model-dependent tests exercise multiple lengths/speeds, finite
samples, parser/language rejection, cancellation, package size, checksum damage
and invalid resealed lexicons. Shared parser regressions and workspace Clippy
also pass. No extra Studio process was started.

```sh
.venv/bin/python training/math_sequence_export.py \
  --student checkpoints/math-sequence-ru-expanded-v6/student.pt \
  --dataset target/math-sequence-ru-targets-v2 \
  --phonemes target/math-sequence-ru-phonemes-v1.json \
  --decoder checkpoints/math-latent-ru-q4-v1 \
  --output checkpoints/new-native-sequence
cargo test --locked -p local-tts-core --test math_sequence -- --ignored
cargo run --locked -p local-tts-speak -- \
  --backend math-neural --model checkpoints/math-sequence-ru-native-v6 \
  --language ru --text '12 + 225 = 237' --output target/equation.wav
```

Unprompted local Whisper base transcribed the first native equation as
`12 плюс 225 равно 237`. This is a diagnostic only. Listening acceptance, broader
numeric validation and comparison against the accepted teacher remain required;
the student is experimental. Ukrainian native architecture is supported by the
export/runtime contract but has not received this extended training/export run.

Four additional native probes are recorded in
`target/math-sequence-native-v6-probes-asr.json`. Leading-zero multiplication
`0.009 * 164` is transcribed correctly. Multiplication `14 * 7` remains unclear;
`785 + 17 = 802` is transcribed with an incorrect result (`702`), and decimal
addition is ambiguously segmented. Thus this run is not a uniformly improved
numeric voice. Paired listening page: `target/math-sequence-native-v6.html`.
It compares accepted full-size teacher controls with the native candidate;
longer training and decoder compression both changed, so it does not isolate
quantization from training effects. Next investigate those effects separately
before another quality claim or default promotion.

## Controlled timing and decoder diagnosis v7 — 2026-10-15

Added `training/math_sequence_decoder_check.py`. It validates package and target
hashes/lineage, selects four validation utterances, and decodes identical teacher
and identical native-predicted latents with the frozen float and packed decoders.
A student teacher-timing control isolates duration effects. Reports record latent
and audio hashes, frame counts, clipping, spectral errors and source/tool hashes.
No training or reserved-test evaluation occurs in this tool. Integrity tests
reject altered, incomplete and escaping package components.

v6's `14 × 7` remains unclear with either decoder, but the student with teacher
durations transcribes correctly. ASR reads the teacher's `802` as `702` too;
that probe cannot establish a student-specific number error. Mean absolute
log-spectral differences between packed and float decoding are substantial, but
that metric alone cannot determine intelligibility.

Added configurable `--duration-weight` to fitting. The old default remains 0.2;
resume preserves the saved weight unless explicitly overridden. The checkpoint
and training report store it. Tests cover old-checkpoint default, objective
change, resume preservation, and rejection of NaN/infinite/out-of-range values.

Matched runs `checkpoints/math-sequence-ru-baseline-v7` and
`checkpoints/math-sequence-ru-duration-v7` each continue v6 for 500 identical
sampled batches, using duration weights 0.2 and 1.0. Both reach stage 4500
(5500 cumulative steps), with unchanged 384/64/64 splits. Validation duration
MSE is 0.051403 vs 0.051001; latent L1 is 0.090625 vs 0.091698.
Across 16 paired validation cases, duration frame MAE improves from 0.615691 to
0.587437 (13/16 cases), while teacher-timed spectral error worsens from 0.499214
to 0.502063 (only 3/16 improve). This is a timing tradeoff, not uniform quality
improvement. The float-decoder `14 × 7` now transcribes correctly. Other ASR
numeric ambiguities remain, including ones present in teacher controls.

Native timing-focused package `checkpoints/math-sequence-ru-native-v7` is
9,501,959 bytes; its verified ZIP is 7,955,281 bytes. Rust waveform parity is
5.83e-7; all three model-dependent native tests and workspace Clippy pass.
Studio exposes an explicitly experimental selector. The same native latent
through float versus packed decoding transcribes as `14 умножить на 7` versus
`14 множеств на 7`, so compression is a remaining contributor on this sample.
The package is not promoted or accepted. Source notices/rights qualifications
are preserved. No new Studio instance was launched.

Listening page: `target/math-sequence-timing-decoder-v7.html`.
Experiment record: `target/math-sequence-duration-experiment-v7.json`.
Matched metric report: `target/math-sequence-ru-duration-comparison-v7.json`.
Controlled decoder reports: `target/math-sequence-decoder-check-v6/report.json`
and `target/math-sequence-decoder-check-v7/report.json`.
Next: test less destructive decoder compression within the download budget,
then broaden numeric/listening validation before extending Ukrainian training.

## Decoder precision sweep v8 — 2026-10-15

`training/math_sequence_repack.py` rebuilds the original frozen decoder from the
pinned teacher and tests stored-weight precision alternatives. The student,
encoder, acoustic predictor and phoneme lookup stay byte-identical to v7.
No training or reserved-test evaluation occurs. Source notices and rights
qualifications are copied. Native fixtures/checksums are regenerated; each ZIP
is checked for CRC, contained SHA-256 and actual download bytes.

| Candidate | Download MB | Installed MB |
|---|---:|---:|
| 4-bit, block 64 | 7.950 | 9.462 |
| 4-bit, block 32 | 9.030 | 10.318 |
| 4-bit, block 16 | 10.851 | 12.030 |
| 5-bit, block 64 | 10.642 | 11.411 |
| 8-bit | 13.107 | 15.429 |
| 4-bit + int8 final learned stage | 8.905 | 10.604 |
| 4-bit + int8 input/first learned upsampling | 8.986 | 10.682 |
| 4-bit + int8 input and final learned stage | 9.942 | 11.824 |

The mixed input/final-stage candidate meets the user's **download** budget;
installed bytes exceed 10 MB. It reduces mean student decoder log-spectral
error against float decoding from 0.647692 to 0.316081 (51.2%) on the four
validation controls. However, the targeted multiplication operator still
transcribes incorrectly. The 8-bit decoder transcribes it correctly but exceeds
the download limit. Smaller blocks and 5-bit storage do not repair that probe.
These are diagnostic results, not a claim of listening quality or model acceptance.
No default is changed and no extra Studio process is started.

Added portable ONNX five-bit packing (eight signed values into five bytes),
configurable four-bit block sizes and per-weight int8 overrides. Arithmetic
unpacking stays in float32 with intermediates small enough for exact integer
representation. Unit tests cover dynamic convolutions, zero channels, non-block
aligned shapes, per-block rounding bounds, mixed overrides and invalid settings.
Learned-stage selection explicitly excludes later fixed STFT transpose nodes;
a regression test covers this distinction. Native mixed-candidate tests pass,
with waveform fixture error 4.40e-7; workspace Clippy passes. Native tests accept
an explicit download archive for download-budget candidates, retaining the old
installed-size check for earlier compact packages.

```sh
.venv/bin/python training/math_sequence_repack.py \
  --source checkpoints/math-sequence-ru-native-v7 \
  --teacher voices/vits-vosk-ru-base/base.pt \
  --output target/new-precision-sweep --block-sizes 64 \
  --int8-head --int8-tail
# Library/CLI can load the result without Python at inference:
target/debug/tts-speak --backend math-neural \
  --model target/math-sequence-decoder-mixed-v8c/q4-head-tail-int8 \
  --language ru --text '14 * 7' --output target/mixed-equation.wav
```

Listening comparison: `target/math-sequence-compression-v8.html`.
Hashed experiment record: `target/math-sequence-compression-experiment-v8.json`.
Candidate ZIP: `target/math-sequence-decoder-mixed-v8c/q4-head-tail-int8.zip`.
An early `mixed-v8` exploratory output selected no learned tail weights; it is
excluded from the results. Selection was corrected and tested before `v8b/v8c`.
Next investigate sensitivity of intermediate decoder layers or a quantization-aware
student/decoder adaptation; do not infer acceptance from lower reconstruction error.

## Layer sensitivity and quantization-aware decoder adaptation v9 — 2026-10-15

Extended repacking with `--int8-middle`: intermediate residual kernels 3/7/11
are tested separately and with the final learned stage in int8. Learned-stage
selection and intermediate grouping have regression tests excluding input/final
weights and fixed STFT operations. None of the six variants repaired the
multiplication operator in the unprompted ASR probe. Five fit the download limit;
the kernel-11-plus-tail variant downloads at 10.52 MB and exceeds it.

Added `training/math_sequence_decoder_qat.py`. Unlike earlier frozen-decoder
student training, this phase **intentionally adapts decoder weights** using
straight-through 4-bit block64 rounding. The original float decoder remains a
frozen target. The native sequence student, encoder, acoustic predictor and
phoneme lookup remain unchanged. The tool validates source/teacher lineage and
cache hashes, excludes reserved test tensors entirely, and keeps validation
under no-grad. Its 64-frame random training crops optimize the existing
multi-resolution spectral objective. Only learned decoder parameters are optimized.

Two reproducible 200-step runs use the same 384 training utterances and learning
rate 1e-5, with fixed center crops from the first eight validation utterances for
monitoring (64 validation utterances loaded; no validation gradients):

- Teacher-latent calibration (`target/math-sequence-qat-final-v9`): crop loss
  0.749216 → 0.622065; 9,505,916 bytes installed, 7,957,442-byte download.
- Native student-latent calibration (`target/math-sequence-qat-student-v9`):
  crop loss 0.750137 → 0.658101; 9,506,008 bytes installed,
  7,957,612-byte download. Training uses native-predicted latents from train
  utterances only; validation latents are prediction/evaluation only.

Neither resolves the operator transcription. Teacher-latent calibration also
makes the first number unclear in that probe. These are negative intelligibility
results, despite lower reconstruction losses; neither is promoted. They do not
uniformly outperform the 9.94 MB mixed-precision v8 candidate. Reserved test
utterances remain unused. The earlier `target/math-sequence-qat-v9` trial used
leading validation crops; the final teacher-latent run above repeats its training
with center-crop monitoring. Crop metrics across those scopes must not be pooled.

The exporter removes weight normalization before adapting effective weights,
checks fake-quantized PyTorch versus packed ONNX outputs, regenerates native
fixtures, verifies CRC/contained SHA-256 and records source/rights/optimizer/
training provenance. Decoder diagnostics now explicitly mark adapted decoders
as updated. The QAT unit tests compare fake rounding with native unpacking on
non-aligned shapes, check straight-through gradients and frozen teacher weights,
and prove unreadable reserved test tensors are never loaded. Both adapted
packages pass the three Rust native integration tests. Student-calibrated native
waveform parity is 4.02e-7. No Studio default or process changes.

```sh
.venv/bin/python training/math_sequence_decoder_qat.py \
  --source checkpoints/math-sequence-ru-native-v7 \
  --teacher voices/vits-vosk-ru-base/base.pt \
  --dataset target/math-sequence-ru-targets-v2 \
  --output target/new-decoder-adaptation --steps 200 --student-latents
```

Listening: `target/math-sequence-qat-v9.html`.
Hashed experiment: `target/math-sequence-qat-experiment-v9.json`.
Next: investigate a speech-sensitive objective and training stability before
longer QAT runs or broader promotion; lower spectral loss is insufficient evidence.

## Ukrainian continuous numeric voice v10 — 2026-10-16

Applied the same student/export/decoder-control pipeline to Ukrainian. Prepared
512 native whole-utterance targets (384 train, 64 validation, 64 reserved test),
with disjoint spoken readings and teacher hashes. All 512 phoneme sequences
match the compact lookup. Continued the previous Ukrainian prototype for 4,000
steps on the expanded corpus, using duration weight 1.0 and frozen decoder.
Validation normalized latent L1 decreases from 0.422294 to 0.220958; log-duration
MSE decreases from 0.075758 to 0.056785. The original prototype contributed 200
steps before this new-corpus continuation. Reserved test targets are not used in
optimization or listening/ASR evaluation.

`checkpoints/math-sequence-uk-native-v10` is 3,421,386 bytes installed;
`target/math-sequence-uk-native-v10.zip` is 2,930,924 bytes downloaded. The
existing int8 Ukrainian decoder fits comfortably under budget, so no 4-bit
compression is needed. Archive CRC/contained hashes are verified and the SHA-256
sidecar is written. Native waveform fixture parity is zero on this host. All
three native model tests pass for Ukrainian, including rejection of English and
Russian request languages. Tests now use the loaded voice's actual language.
Studio has an explicitly experimental Ukrainian v10 selector and uses the shared
Ukrainian arithmetic pronunciation preview. The library/CLI/FFI use the existing
MathStudent backend, without Python inference. No extra Studio instance launched.

Generalized decoder QAT to RU/UK and configurable 4-bit/int8 simulation. Int8
fake quantization matches the exporter's per-axis symmetric storage, with a new
native rounding/gradient test. The Ukrainian 200-step trial calibrates on native
predicted training latents. Its validation center-crop loss worsens from 0.158987
to 0.178714; `target/math-sequence-uk-qat-v10/voice` is therefore not selected.
That package is 3,425,491 bytes installed and 2,931,889 bytes downloaded, with
source notices preserved; its native tests and waveform parity also pass.

Unprompted Ukrainian ASR reads the native equation's numbers as 12/225/237, but
its equality wording is imperfect. Multiplication and decimal probes remain
unclear. Teacher controls also have number errors in ASR, so listening is needed.
Neither the Ukrainian student nor this Ukrainian teacher has received user voice
acceptance. The candidate remains experimental. Rights/source qualifications
are unchanged and copied into the artifacts.

```sh
.venv/bin/python training/math_sequence_decoder_qat.py \
  --source checkpoints/math-sequence-uk-native-v10 \
  --teacher voices/vits-piper-uk-base/base.pt \
  --dataset target/math-sequence-uk-targets-v2 \
  --output target/new-uk-decoder-trial --steps 200 --student-latents --bits 8
cargo run --locked -p local-tts-speak -- \
  --backend math-neural --model checkpoints/math-sequence-uk-native-v10 \
  --language uk --text '12 + 225 = 237' --output target/uk-equation.wav
```

Listening: `target/math-sequence-uk-v10.html`.
Hashed experiment: `target/math-sequence-uk-experiment-v10.json`.
Seven other European decoder exports were integrity-checked; their numeric
sequence frontends/corpora/students remain pending. Exact status and order are
in `docs/MATH_LANGUAGE_ROLLOUT.md`; decoder availability is not full voice support.

## Ukrainian matched continuation v11 — 2026-10-16

Two 1,000-step continuations began from the same v10 checkpoint and sampled the
same training examples. The baseline retained duration weight 1.0; the other
used 2.0. The stronger duration objective has no promotion case: duration frame
MAE improved only from 0.579811 to 0.572389 on 8/16 paired validation controls,
while teacher-timed spectral error worsened from 0.875975 to 0.878068 and only
3/16 cases improved. It is retained as a diagnostic checkpoint only.

The normal continuation improves teacher-timed spectral error from 0.889650 to
0.875975 on 14/16 controls. Its duration MAE moves from 0.584801 to 0.579811,
with 7/16 controls improving. It therefore becomes the latest Ukrainian
experimental candidate, not an accepted/default voice. Native package
`checkpoints/math-sequence-uk-native-v11` is 3,424,453 bytes; verified ZIP
`target/math-sequence-uk-native-v11.zip` is 2,930,202 bytes. Dynamic ONNX
parity remains below 8e-6 and all Rust native tests pass.

ASR continues to misread the multiplication and decimal controls. This is
diagnostic evidence only and does not overrule listening. The experimental v11
Studio selector leaves v10 available for comparison. Listening control:
`target/math-sequence-uk-v11.html`.

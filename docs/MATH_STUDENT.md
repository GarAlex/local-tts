# Experimental English arithmetic student

Numeric-only under-10-MB candidate: [MATH_LATENT.md](MATH_LATENT.md). Uses cached conditioning and a neural decoder; listening/numeric acceptance pending.


Latest failure investigation and staged recovery: [MATH_STUDENT_RECOVERY.md](MATH_STUDENT_RECOVERY.md). The 5.5 MB student FAILED user listening. Studio now selects a 29.6 MB stored-int8 pretrained compression candidate, whose listening acceptance remains pending. This is compression, not tiny-model distillation.


This is an output-based distillation experiment: Kokoro generates whole arithmetic
utterances; a separate, smaller VITS student trains on those WAVs and transcripts.
Teacher weights are not copied into the student. It is not feature/logit
matching. The new architecture requires scratch initialization; subsequent
adaptation supports the ordinary fine-tuning/resume workflow. General own-voice
training still uses fine-tuning as its primary route.

## First experiment

- Dataset: datasets/math-teacher-en-v1/dataset.json, 64 mono 22050 Hz synthetic
  clips, 58 train / 6 validation. Teacher is the installed pinned Kokoro af_heart.
- Profile: math-student-v1, 32 latent/hidden channels, 64 filter/upsample channels,
  two text layers, one residual-kernel family and deterministic duration predictor.
- About 525000 inference parameters (the exact count depends on token inventory);
  training includes a larger discriminator which is not shipped at inference.
- Completed 16 steps, resumed to 100, exported to
  checkpoints/math-student-en-v1/voice-step100.
- ONNX model: 2326689 bytes (2.33 decimal MB). Full export is about 2.64 MB,
  including a numerical reference fixture and retained teacher notices.
- Native ONNX/PyTorch verification passed at maximum sample error
  0.000000028871. Rust request/parser tests, CLI and independent C++ backend-9
  inference passed. This is numerical parity, not intelligibility or voice quality.

**This is not a usable trained voice yet.** One hundred scratch-training steps
on this small corpus is a pipeline experiment. There is no verified sub-1 MB
neural voice, and no evidence it outperforms the 788 KB fragment baseline.
The student is deliberately not offered as a ready voice in Studio.

## Reproduce

```sh
cargo run --release -p local-tts-core --example math_dataset -- \
  voices/kokoro-en datasets/new-math-teacher 128
LOCAL_TTS_CLI="$PWD/target/release/local-tts" .venv/bin/python training/launch.py \
  --dataset datasets/new-math-teacher/dataset.json --output checkpoints/new-math-student \
  --from-scratch --math-student --steps 100 --batch-size 1 --device cpu
LOCAL_TTS_CLI="$PWD/target/release/local-tts" .venv/bin/python training/export_and_verify.py \
  --checkpoint checkpoints/new-math-student/latest.pt \
  --output checkpoints/new-math-student/voice --new-output
./target/release/tts-speak --backend math-neural \
  --model checkpoints/new-math-student/voice --text '807 + 694 = 1501' --output student.wav
```

Resume with --resume in place of --from-scratch and increase the total --steps.
Do not regenerate or alter the prepared dataset while resuming. Dataset generation
refuses an existing folder and retains teacher license/card/provenance. Copy
those records into exports intended for distribution, as done in the experiment.
The training/export dependencies are the existing pinned project environment.
Inference uses Rust MathStudent (shared expression parser), C ABI
TTS_MATH_STUDENT_EN (9), or tts-speak --backend math-neural. Default speaker is
required. A student export can read arithmetic only after it has learned it;
the parser does not guarantee neural pronunciation accuracy.

## Corpus and training follow-up

The 1 MB figure is a flexible design goal. Prefer intelligibility and naturalness
over shrinking weights prematurely; quantization is deferred until a usable
voice provides a meaningful quality baseline.

The current generator uses arithmetic-v2, minimum 128 clips. It rejects duplicate
spoken forms and covers 16 structures in each split. The generated v2 corpus has
96 train, 16 validation and 16 test clips; none of its test readings overlap the
original v1 corpus. The old 64-clip data remains as a historical pilot.

The v1 student resumed to step 1000 and exported to voice-step1000. A separate
v2 run fine-tuned those weights for 200 target steps on the revised corpus,
retaining initialization lineage. Export: checkpoints/math-student-en-v2/voice-step200.
Model size is unchanged at 2326689 bytes; full export is 2612085 bytes, including
reference fixture and teacher notices. Native parity maximum error was
0.000000024447218; C++ inference and request checks passed.

Generate three-way comparisons using only reserved test examples:

```sh
cargo run --release -p local-tts-core --example math_student_compare -- \
  datasets/math-teacher-en-v2 voices/math-en-v1 \
  checkpoints/math-student-en-v2/voice-step200 target/new-heldout-comparison \
  datasets/math-teacher-en-v1 datasets/math-teacher-en-v2
```

The optional final arguments audit training/validation transcripts of each
ancestor dataset against test readings, before loading models or writing audio.
The tool also rejects duplicate spoken forms within the evaluation dataset.
Reports record audio/model/dataset hashes and teacher/fragment/student files.
A deliberate training-lineage overlap test passed.

Outputs are in target/math-heldout-step1000 and target/math-heldout-v2-step200.
The v2 median student/teacher duration ratio is 0.79 and median RMS is 0.0061
versus teacher 0.0691. These measurements show a quiet, early prototype, and do
not establish correct pronunciation. Compared with the earlier 1000-step model,
duration matching worsened while signal level rose; no quality winner is claimed.
Listening evaluation and substantially more training remain necessary.


## Longer run and Studio testing

Resumed v2 to 2000 total fine-tuning steps (after 1000 v1 scratch steps).
Preserved generator-only evaluation snapshots at steps 800 and 2000. These are
about 3.14 MB and retain source checkpoint SHA-256; they exclude the discriminator,
optimizers and RNG state. They can be exported or used as fine-tuning bases, but
cannot resume the original training run. Resume from the full latest.pt instead.
Snapshot creation refuses existing outputs, and resume rejection was tested.

Final export: checkpoints/math-student-en-v2/voice-step2000; ONNX remains
2326689 bytes, full exported folder is 2651336 bytes including teacher notices
and numerical reference fixture. Teacher notices are included in checksums.
Studio now offers **English — Arithmetic neural student (experimental)**, using
this export by default. Live Studio generation/playback was observed with
12 + 225 = 237. The shared library, Studio and C ABI also support cancellation
at the underlying VITS chunk boundaries.

Validation comparisons are in target/math-validation-step200,
target/math-validation-step800 and target/math-validation-step2000. Each folder
includes comparison.html with local teacher/fragment/student audio players and
comparison.json with hashes/diagnostics. Add --validation to math_student_compare
for these comparisons; test is still the default. Validation overlap auditing
excludes training texts; test auditing excludes both training and validation texts.

| Fine-tuning step | Median student/teacher duration | Median student RMS |
| --- | ---: | ---: |
| 200 | 0.795 | 0.00608 |
| 800 | 0.811 | 0.00430 |
| 2000 | 0.961 | 0.00478 |

Teacher median RMS is 0.06917. These numbers do not establish intelligibility.
Teacher-forced validation mel L1 decreased from 0.8886 (800) to 0.8499 (2000),
and duration loss from 0.6232 to 0.5654. This diagnostic uses ground-truth audio
in the posterior encoder; it does not measure free-running pronunciation.
The student remains quiet and experimental, with no verified quality winner.

```sh
.venv/bin/python training/math_snapshot.py \
  --checkpoint checkpoints/math-student-en-v2/latest.pt \
  --output checkpoints/math-student-en-v2/new-evaluation.pt
.venv/bin/python training/math_validation.py \
  --checkpoint checkpoints/math-student-en-v2/evaluation-step2000.pt \
  --prepared checkpoints/math-student-en-v2/prepared.json \
  --output target/new-validation-diagnostics.json
```

The next quality experiment should improve the small arithmetic speech corpus
and consider a compact, fixed-vocabulary phoneme frontend if grapheme learning
limits accuracy. More training alone is not assumed to be sufficient. Quantization
remains deferred; 1 MB is a flexible size goal, not a reason to sacrifice speech.

## Next experiments

### Step-2000 listening failure (2026-10-08)

The user listened to a freshly generated `12 + 225 = 237` in the rebuilt
development app and reported mechanical noise rather than a voice. This
checkpoint fails the perceptual acceptance criterion. Duration matching and
ONNX parity did not establish speech quality; it must not be presented as a
usable distilled voice.

`training/math_audio_diagnostic.py` now separates ground-truth posterior
reconstruction from text-only inference and exports seeded noise-scale probes.
The step-2000 report is `target/math-audio-diagnostic-step2000/report.json`.
For the first validation clip, teacher RMS is 0.07130, posterior reconstruction
RMS 0.03363, and deterministic text inference RMS 0.00484. Increasing latent
noise to 0.667 gives RMS 0.01088, but this is not a quality fix. Python inference
also produces weak output, and previous native/Python parity excludes a large
export discrepancy. These results suggest inadequate text-prior/flow learning;
posterior reconstruction still requires listening and does not prove the
decoder is good. The precise cause is not yet established.

The next quality experiment must inspect reconstruction and text inference
separately, improve the pronunciation frontend/data and training convergence,
and pass listening checks before becoming a usable voice. The fragment bank
remains the working compact arithmetic implementation.

### Direct text-prior supervision experiment

The diagnostic now also decodes the mean text prior with ground-truth monotonic
alignment. On the failed checkpoint its RMS is 0.00539, close to text-only
inference and much lower than posterior reconstruction. This localizes a large
gap upstream of duration prediction, without proving the precise cause.

`--math-prior-mel-weight` adds an optional mel reconstruction loss on this
aligned mean prior. The default is zero for all existing workflows; nonzero
values are restricted to the arithmetic profile, recorded in checkpoints and
exports, and cannot be changed during resume. The objective backpropagates
through the text encoder, inverse flow and waveform decoder, using the existing
detached alignment. It is experimental and does not eliminate the need to learn
durations or pass free-running listening checks.

Controlled run: `checkpoints/math-student-en-prior-v1`, initialized from the
step-2000 generator snapshot, learning rate 0.0002, prior weight 15, batch size
1, 300 additional updates. Its discriminator starts fresh because the snapshot
contains only generator weights. This changes both the training objective and
learning rate, so it is a recovery experiment rather than an isolated ablation.

Reproduce in a new output directory:

```sh
.venv/bin/python training/train.py \
  --prepared checkpoints/math-student-en-v2/prepared.json \
  --output checkpoints/new-math-prior \
  --init-checkpoint checkpoints/math-student-en-v2/evaluation-step2000.pt \
  --math-student --math-prior-mel-weight 15 --learning-rate 0.0002 \
  --steps 300 --batch-size 1 --device cpu
```

Acceptance requires recognizable speech first, then correct numbers and
operators on reserved expressions, followed by comparison of naturalness and
complete asset size. Neither RMS nor mel loss can satisfy listening acceptance.

The 300-update run completed. On the same validation diagnostic, aligned-prior
mel L1 decreased from 1.47458 to 1.14664; posterior mel L1 from 0.78389 to
0.76262. Free-running output remains weak (RMS 0.00608 versus teacher 0.07130)
and shorter (3.36 versus 4.25 seconds). This is not accepted as speech.
Export: `checkpoints/math-student-en-prior-v1/voice-step300`. Native/Python
maximum waveform difference is 0.000000025262125. The arithmetic example is
`target/math-prior-step300-12-plus-225.wav`, and 16 validation triples with
training-lineage auditing are in `target/math-prior-validation-step300`.
Teacher notices are retained. Studio's current default student is not replaced
by this unaccepted experiment. Next work must address predicted alignment and
free-running convergence, not only improve posterior/teacher-aligned loss.

### Validation-wide alignment and retained-discriminator experiment

`training/math_validation.py` now reports mean aligned-prior mel L1, text-only
duration ratio and RMS, plus per-clip alignment concentration. It uses all 16
validation clips, hashes the prepared dataset and each teacher recording, and
keeps the test split reserved. These metrics are diagnostics, not intelligibility.

Expanded original results (`target/math-validation-original-expanded.json`):
mean aligned-prior mel 1.52358, mean duration ratio 0.95937, text RMS 0.00474.
The first recovery run (`target/math-validation-prior300-expanded.json`) gives
1.18349, 0.71705 and 0.00575 respectively. Thus improved teacher-aligned loss
coexists with worse predicted durations. The original alignment's median maximum
token-frame share is 0.05796; it does not show a single-token collapse. Portable
maximum-path tests pass exhaustive path comparisons, which checks algorithm
correctness but does not establish linguistically correct alignment.

The separate `checkpoints/math-student-en-prior-retained-v1` run initializes
from the full original step-2000 checkpoint, retaining its trained discriminator,
with prior weight 15 and learning rate 0.0002. At 300 updates, validation mean
aligned-prior mel is 1.26462, duration ratio 0.83284 and RMS 0.00392. This is
not evidence of recovered speech.

`training/math_listening.py` makes an offline shuffled listening comparison
from two matching validation reports with audited training lineage. It includes
previous/new students, teacher and word bank, records recognizable speech,
heard words and arithmetic errors, and exports ratings together with report
and audio hashes. Browser-local storage is optional; export preserves ratings.
The page is a listening aid, not a statistically blinded study. Tests cover
escaping, audio paths, refusal to overwrite and matching audited validation sets.

The retained-discriminator run completed 600 updates. All-16-clip validation:
posterior mel L1 0.80421, aligned-prior mel L1 1.19039, duration loss 0.51384,
text duration ratio 0.71445, text RMS 0.01817. The stronger waveform is not an
intelligibility result, and shortened output remains a concern. Native export
parity maximum error is 0.000000045634806. The model remains 2,326,689 bytes;
teacher notices and checksums are retained. No Studio default is replaced.

Artifacts:

- `checkpoints/math-student-en-prior-retained-v1/voice-step600`
- `target/math-validation-retained600-expanded.json`
- `target/math-retained-diagnostic-step600` (posterior/aligned/noise probes)
- `target/math-listening-retained600.html` (16 shuffled listening comparisons)
- `target/math-retained-step600-12-plus-225.wav` (native arithmetic sample)

Next experiment: separate pronunciation learning from acoustic learning using
the existing CMU English phoneme frontend, with a distinct student profile and
short-utterance curriculum before long arithmetic chains. Preserve existing
grapheme exports, compare text-only speech first, and retain the fixed validation
and test boundaries. The frontend/curriculum plumbing is now implemented below;
speech convergence remains unfinished.

### Phoneme curriculum after a second listening rejection

The user also rejected the retained-discriminator step-600 sample as no voice.
It is a failed listening candidate. Stronger RMS did not imply recognizable
speech. Stop extending these grapheme runs as though they are nearing acceptance.

The separate `math-phoneme-v1` architecture profile uses the same compact network
with CMU IPA/stress tokens. Native `MathStudent` now accepts English grapheme-v2
or CMU IPA exports. Both use the same arithmetic parser, preserving old exports
and the CLI/C ABI contract. Checkpoint validation rejects frontend/profile
mismatches and modified dictionaries. Export retains the dictionary and BSD
notice. The curated dictionary covers all 43 math vocabulary words in 877 bytes;
it does not require distributing the full English dictionary.

`short_corpus()` and `math_dataset --short` provide deterministic single numbers
and short addition/subtraction/multiplication phrases (128–384 unique readings).
They reserve complete four-structure blocks for validation/test. Very short
teacher recordings receive trailing silence to meet the existing 8192-sample
training segment minimum. The first 128 clips are in
`datasets/math-teacher-short-en-v3`, with 96 train, 16 validation and 16 test.
This first stage does not teach all supported operators or long number chains.

Prepared configuration: `checkpoints/math-phoneme-en-v1/prepared.json`.
An explicit scratch run of 16 updates verifies the new profile end to end;
`voice-smoke16` is an untrained engineering fixture, not a voice candidate.
Native export parity maximum error is 0.00000006146729. Teacher and CMU notices
are retained. Unit tests cover curriculum uniqueness/reproducibility/split
structures and phoneme profile/dictionary rejection. Strict workspace Clippy and
existing math parser, bank and real legacy-student request tests pass.

Resume this new profile for actual convergence (example target, not a promise of
speech quality):

```sh
.venv/bin/python training/train.py \
  --prepared checkpoints/math-phoneme-en-v1/prepared.json \
  --output checkpoints/math-phoneme-en-v1 --resume --math-student \
  --steps 10000 --batch-size 1 --device cpu
```

The failed grapheme weights cannot fine-tune this different token inventory.
Subsequent phoneme runs can resume or fine-tune compatible phoneme checkpoints.
Keep size reduction deferred and require listening acceptance before expanding
to long equations. A native pretrained-voice control is available at
`target/math-pretrained-phoneme-control.wav`; it uses the existing full LJSpeech
voice through the arithmetic interface, not newly distilled weights.

Grow and audit the teacher corpus, ensuring unique spoken forms and separate
held-out number combinations, operator structures and word transitions. The
small comparison-v1 set includes an expression also present in this initial
corpus; it must not be treated as a fully held-out test set for this student.
Run longer training with validation checkpoints, then manually evaluate number
accuracy and naturalness. Quantization and smaller-capacity profiles are future
experiments; compare complete inference assets rather than checkpoint files.
Student perceptual quality and neural quantization remain unfinished. Studio
student testing controls are now available. The 788 KB fragment baseline remains available.

# Arithmetic neural student: failure investigation and recovery

The 2.33 MB student did not produce a usable voice. Both user listening tests
rejected it, and unprompted local ASR failed to recover its arithmetic. Native
ONNX parity showed that the app faithfully reproduced the Python model; that
check did not establish speech quality.

## What the experiments established

- **Inference mismatch:** old exports always used latent noise scale zero,
  although the stochastic VITS decoder trained on sampled latents. The
  [upstream inference example](https://github.com/jaywalnut310/vits/blob/main/inference.ipynb)
  uses 0.667. A sampled export of the same failed checkpoint remained an ASR
  failure, so sampling alone was not the cure.
- **Premature capacity reduction:** the original decoder had 64,608 parameters,
  versus 14,337,024 in the pretrained reference. It was trained from random
  weights on a small corpus. The
  [upstream configuration](https://github.com/jaywalnut310/vits/blob/main/configs/ljs_base.json)
  uses batch size 64; our batch-one short runs were not comparable training.
- **Acoustic failure preceded text failure:** the 500-step phoneme model still
  failed ASR when reconstructing held-out audio with ground-truth posterior
  latents. Optimizing duration matching alone could not repair this.
- **The architecture can learn an acoustic phrase:** fitting the narrow decoder
  on one training phrase with a multiresolution spectral loss reduced full-clip
  mel L1 from 0.5723 to 0.2799. Local ASR recovered “78” afterward. This was
  overfitted reconstruction, not TTS or held-out performance.
- **More capacity and staged training helped reconstruction:** a widened decoder
  trained on fixed posterior means reconstructs several reserved phrases that
  ASR reads correctly. Some reconstructions still have numeric or word errors.
- **Text inference remains the acceptance target:** reconstruction, volume,
  durations and numerical parity are insufficient. The phoneme prior must emit
  recognizable and correct speech without source audio, then pass listening.

These are multiple contributing failures, not proof of one exclusive cause.

## New architecture and training stages

`math-phoneme-wide-v1` keeps the 32-channel latent/flow and CMU IPA frontend,
but widens the waveform generator from 64 to 256 initial channels. Its decoder
has 855,552 parameters; all inference components total 1,321,025 parameters.
The initial widened decoder reproduces the narrow decoder to approximately
2.8e-8 maximum sample difference. Zero-sum perturbations preserve that function
while breaking channel symmetry so the extra channels can learn.

The ONNX graph is **5,507,081 bytes** in the initial staged export. The curated
math pronunciation dictionary is **877 bytes**. Notices, provenance and large
numerical verification fixtures make the complete development folder larger;
the graph size is not the full distributable package size.

1. `math_teacher_alignment.py` uses the trained LJSpeech phoneme model to label
   durations on target **training clips only**. It excludes validation/test,
   hashes recordings/tokens/prepared data, and preserves the alignment teacher's
   source notices. Labels are automatic and not manually verified.
2. `math_decoder_probe.py` isolates one-clip reconstruction as a diagnostic.
3. `math_acoustic_init.py` reuses compatible acoustic weights with explicit
   component provenance and checks that the probe source matches the transfer.
4. `math_widen.py` widens the decoder without resetting its initial function.
5. `math_acoustic_fit.py` fits the wider decoder on cached posterior means, with
   mel and multiresolution spectral losses. Validation audio is not optimized.
6. `math_prior_fit.py` freezes the acoustic encoder, flow and decoder, then fits
   phoneme means and durations using teacher labels and audio/latent losses.
   Dropout is disabled to match deterministic inference during this small-corpus
   fitting stage. The resulting model intentionally uses **noise scale zero**:
   its acoustic stage learned posterior means, unlike the old stochastic model.

This is acoustic/text supervision on synthesized teacher data. It is not a
claim that the entire teacher network has been distilled or that voice quality
matches the reference.

## Data and assets

`datasets/math-teacher-short-en-v3` has 128 unique short readings: 96 training,
16 validation and 16 test. It covers isolated numbers 0–99 and short
addition/subtraction/multiplication phrases. It does not yet train the full
arithmetic vocabulary or long equations such as `12 + 225 = 237`.

Validation and test readings were checked against training transcripts in both
earlier teacher corpora: no overlaps. The acoustic warm-start originated in
those corpora, so this lineage check matters. The test split remains reserved.

Audio teacher: pinned Kokoro `af_heart`, with retained Apache-2.0 notices.
Alignment teacher: existing imported LJSpeech VITS model, with its source,
weights and dataset notices retained separately. CMU BSD and VITS MIT notices
are retained in exports. ASR is an optional development diagnostic, never an
inference dependency or an API call.

## Local recognition diagnostics

Install `training/requirements-eval.txt` into the existing training environment.
`math_asr_check.py` uses Whisper `tiny.en` locally, without a text prompt or
expected reading supplied to the decoder. The model is downloaded once into
`target/asr-models`, checksum-verified by its loader, and hashed in each report.
The approximately 75.6 MB ASR weights are not part of the TTS voice.

The recognizer correctly reads the full pretrained control, while it returns
generic “you” or other incorrect text for the failed students. It correctly
reads the overfitted training reconstruction “78”. It can still hallucinate or
misrecognize distorted speech and numbers; raw transcripts and confidence
diagnostics are retained, and human listening remains required.

Relevant reports:

- `target/math-asr-initial-investigation.json`
- `target/math-asr-posterior500.json`
- `target/math-wide-acoustic1000-asr.json`
- `target/math-wide-acoustic3000-asr.json`
- `target/math-wide-prior2000-asr.json`

The 2,000-update text-only fit is still inaccurate. Examples ASR reads correctly
include `18 × 3`, `10 − 10` and `10 × 19`; many other validation readings contain
substitutions. Do not promote that checkpoint as a usable arithmetic voice.

## Reproduction and runtime

The staged artifacts are under `checkpoints/math-phoneme-wide-en-v1`. Acoustic
fitting used 3,000 decoder updates; the first prior fit used 2,000 updates, with
a subsequent 4,000-update fit completed (6,000 cumulative prior updates). These staged
tools initialize/fine-tune from the previous generator checkpoint with a new
optimizer. Their outputs are not resumable by the general GAN trainer.

```sh
.venv/bin/python training/math_prior_fit.py \
  --checkpoint checkpoints/math-phoneme-wide-en-v1/acoustic-fit3000/acoustic.pt \
  --prepared checkpoints/math-phoneme-en-v1/prepared.json \
  --alignment checkpoints/math-phoneme-en-v1/teacher-alignment.json \
  --output checkpoints/new-prior-fit --steps 2000 --device mps

.venv/bin/python training/export.py \
  --checkpoint checkpoints/new-prior-fit/prior.pt \
  --output checkpoints/new-voice --new-output

./target/release/tts-speak --backend math-neural \
  --model checkpoints/new-voice --text '18 × 3' --output sample.wav
```

CPU remains supported. The general trainer now also accepts `--device mps`,
saving/restoring MPS RNG state. Switching devices changes floating-point/RNG
behavior; cross-device training is not promised to be bit-identical.

The native library, CLI and C ABI use the same exported VITS contract. No ASR,
Python, acoustic encoder, discriminator or training optimizer is required to
run the TTS voice. Sampled diagnostic exports use a repeatable Gaussian bank
(4096 frames, about 47.6 seconds) to preserve native/Python parity. Staged
posterior-mean exports reject sampling because it would change their learned
input distribution.

Next: listening and numeric accuracy, coverage of all arithmetic vocabulary,
full-expression training and reserved test evaluation. Smaller sizes and
quantization remain secondary to speech quality.

## Latest text-only result: 6,000 prior updates

The exported `voice-prior6000` model passes native parity with maximum sample
error 1.4901161e-7. Its graph is 5,507,081 bytes. Kokoro data-teacher notices
and LJSpeech alignment-teacher notices are included, and the package was resealed.
Studio now selects this experimental checkpoint with the short example `18 × 3`;
it warns that numbers are unreliable and full equations are not yet trained.
This is an experimental default change, not acceptance as a usable voice.

Unprompted local `tiny.en` ASR correctly recovered four of 16 validation phrases
by manual semantic comparison (number words and digits treated equivalently).
This diagnostic is not a formal speech or numeric accuracy benchmark. It also
hallucinates transcripts for poor audio. Human listening is still required.

| Expected | ASR transcript |
| --- | --- |
| fifty seven | Thank you, Sagin. |
| eighteen plus fourteen | 18th Class 13 |
| six minus fifteen | I'm still alive, I'm still alive. |
| eighteen times three | 18 times three. |
| ninety three | 93 |
| ten plus four | Ten plus fun! |
| sixteen minus fourteen | Staying like I see you on teen. |
| sixteen times nineteen | Staying times 19. |
| two | Tanya. |
| nine plus two | Line Class Team |
| ten minus ten | 10 minus 10. |
| thirteen times fourteen | Thirteen times, thirteen. |
| fifty nine | Safety line. |
| eleven plus eighteen | The Wugging Plus 18. |
| five minus eleven | 5 minus 11 |
| ten times nineteen | Teng-times 19. |

Raw ASR results: `target/math-wide-prior6000-asr.json`. Native teacher/student/
fragment comparison: `target/math-wide-prior6000-validation/comparison.json`,
with all three training-corpus lineages audited. Offline shuffled comparison
against prior-fit2000: `target/math-listening-wide6000.html`.
The reserved test split was not used for these development decisions.

The 6,000-step result remains inadequate. Next training work must address
phoneme/latent accuracy and automatic duration-label quality, then expand
coverage; increasing update counts alone has not produced reliable numbers.

## Listening rejection of prior-fit6000

The user rejected the 5.5 MB model as very robotic and far from the required
quality. Mark this candidate FAILED, despite its four ASR-correct phrases.
Do not use ASR recognition to promote voice quality. Further short-corpus
student fitting is paused in favor of preserving pretrained speech weights
and testing compression against the original. A quality-first compressed
reference is not completion of tiny-model distillation.

## Pretrained compression recovery

`training/math_pretrained_compress.py` preserves the pretrained LJSpeech VITS
architecture and stores 172 large weight tensors as symmetric per-axis int8
with `DequantizeLinear`. Activations and model operations remain float32.
This reduces the ONNX graph from 114,015,221 to 29,563,997 bytes. It does not
promise lower runtime RAM or faster synthesis: weights can expand during runtime
optimization. It is post-training stored-weight compression, not distillation
and not an arithmetic-only learned model. The full CMU dictionary is retained.

All source checksums are checked before transformation. Source notices, rights
and provenance are copied. `compression.json` records original model SHA and
fixture differences. Quantization changes waveforms and may change durations;
sample-wise errors are recorded without claiming waveform equivalence.
The new native parity fixtures are ONNX Runtime outputs for the compressed
graph, not Python outputs of the uncompressed network. Native parity passes,
but is only an execution-contract check.

The Rust CLI generated `12 + 225 = 237`, and unprompted local ASR transcribed
it correctly, as it did the original pretrained model. New Studio builds select
`checkpoints/math-pretrained-int8-en-v1` with the full expression and a listening
warning. No extra Studio instance was launched.

```sh
.venv/bin/python -m pip install -r training/requirements-eval.txt
.venv/bin/python training/math_pretrained_compress.py \
  --source voices/vits-ljs-base/voice --output checkpoints/new-compressed-reference
./target/debug/local-tts vits-verify checkpoints/new-compressed-reference
./target/release/tts-speak --backend math-neural \
  --model checkpoints/new-compressed-reference --text '12 + 225 = 237' \
  --output sample.wav
```

Original/compressed native comparison on 16 long v2 validation expressions is
saved under `target/math-pretrained-original-validation` and
`target/math-pretrained-int8-validation`. These are development examples; the
public model's LJSpeech pretraining corpus was not exhaustively audited for
transcript overlap. The offline shuffled listening page is
`target/math-listening-pretrained-int8.html`. It compares the uncompressed
reference, compressed reference, Kokoro teacher and fragment bank.

Quantized convolution checks verify dynamic input lengths, zero-weight channels,
finite outputs and a numerical rounding-error bound. Workspace strict Clippy
and Studio build pass. Human naturalness and numeric accuracy remain required
before acceptance; do not claim this solves the tiny-distillation goal.


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

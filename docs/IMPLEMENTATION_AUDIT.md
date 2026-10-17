# Implementation audit — 2026-10-03

Working local inference and training machinery exists. A speech-quality-validated
own voice and a production release are still incomplete.

| Feature | Implemented and verified | Remaining limits |
|---|---|---|
| Rust inference | Kokoro English FP32/Q8/Q4 and exported VITS; no Python/server at inference | Quality and performance evaluations remain |
| English frontend | CMU dictionary; text-v1 dates/quantities/abbreviations; limited homograph cues; paragraph chunking | Unknown words fail; broader context, unsupported numeric formats and listening evaluation remain |
| egui speech test | Generate/play/WAV output, duration, chunk-boundary cancellation | Current model invocation must finish before cancellation |
| Dataset authoring | WAV import/downmix/resampling, trimming, transcripts, grouped splits | Microphone capture implemented but hardware unverified |
| Guided recording | Six-page in-app beginner walkthrough; original EN/RU/UK starter lists, custom lists, saved-transcript resume | Twelve starter prompts per language are not a complete corpus; guide navigation does not certify quality |
| Validation | Safe paths, decoding, mono/rate, silence, aliases, split warnings and statistics | Rights claims are not automatically established |
| VITS training/export | Default checkpoint fine-tuning, explicit scratch mode, CPU resume/stop, preview, dynamic ONNX, native parity | Project-format and public English LJSpeech import supported; weight provenance gap and real-speech quality remain; CUDA unverified |
| RU/UK pipeline | Separate identifiers; text-v1 dates/quantities with agreement and phrase stress; versioned training/export/native parity | Explicit/saved stress takes precedence; arbitrary case, general stress prediction and listening validation remain |
| C/C++ embedding | Shared/static builds, ownership, PCM, WAV, threading and cancellation | macOS shared consumer tested; static linking and other platforms unverified |
| Integrity | Pinned Kokoro model/tokenizer/style; VITS package hashes and tamper rejection | Hashes are not signatures; editable dictionary excluded |
| Packaging | Native macOS app/SDK/source candidate recipe with notice inventory, hashes, linkage audit and relocation checks | SDK notices complete; one Studio notice gap; real signing/notarization, clean-machine GUI and other platforms remain |
| Voice effects, age and moods | Native EQ/compression/peak protection, duration-preserving pitch and independent formants; Rust/additive C ABI/CLI and Studio original/processed controls | Age presets, learned styles and listening evaluation remain; see [effects](VOICE_EFFECTS.md) |

## Corrections made during this audit

- Added **First dataset: step-by-step guide** in Build dataset, available before
  creating a manifest. All six pages, Back/Next, direct page selection and
  collapse/reopen were checked in the rebuilt macOS app. Studio lint/build and
  its three existing tests passed. This was a help/navigation check; microphone
  hardware and beginner user testing remain unverified.
- Added **Discard manifest edits and reload** to Studio. External changes no
  longer leave unsaved edits without a recovery action. Invalid reloads preserve
  the in-memory edits. Both regression and GUI checks passed.
- Declared speaker/split counts now survive a missing dataset root; valid totals
  stay zero. Regression test passed.
- Restored checkpoint RNG-state tensors on CPU before CUDA RNG APIs use them.
  CPU resume regression passed; CUDA execution remains unverified.
- Added `vits-synthesize` CLI and extended smoke tests to native text inference
  and checksum tamper rejection for English, Russian and Ukrainian.

## Repeatable verification

`python3 scripts/verify.py` runs formatting, the workspace Rust tests, strict Clippy and
workspace builds. `--full` additionally runs exhaustive alignment, synthetic
EN/RU/UK training/resume/graceful stop/export/parity/text inference/tamper checks,
then C++ English inference, cancellation, ownership and concurrency checks.
Full mode needs the training environment and installed English assets. The host
is macOS; Windows full mode is unsupported by the C++/signal harness.

The latest machine-readable result is `target/verification.json`. This local
runner can be invoked by CI, but no hosted CI service is configured here.
Native parity errors for the three synthetic exports were below 0.00000005.

## Dependency review status

The current macOS Cargo metadata snapshot resolved 258 packages, all with license
metadata. None of the reported expressions contained GPL/LGPL/AGPL/MPL. This is
a metadata check of the resolved workspace, including development/build tools,
not a complete distribution audit. Native binary notices, fonts, downloaded
assets and training dependencies still need the complete release notice bundle.
Selected records are retained in [THIRD_PARTY.md](THIRD_PARTY.md).

## Work still required

Owned or cleared speech recordings and suitable compute are needed to validate
own voices. Contextual language frontends, stress handling, release notices,
platform builds, packaging, performance/quality evaluation and the planned
effects/style controls remain in [PHASES.md](PHASES.md).


## Pretrained Cyrillic integration (2026-10-02)

Added native Vosk 0.7 Russian and Piper Ukrainian text-mode inference, with speaker
selection in Studio, CLI commands and additive C ABI backend IDs 3/2. No Python,
server or eSpeak dependency at inference time. Pinned model/config/dictionary
integrity is checked before loading. Vosk dictionary includes !SIL/[unk] entries
outside the TTS inventory; these reserved ASR entries are skipped, while all
ordinary dictionary pronunciations are validated. Input validation rejects unknown
characters instead of dropping letters. Explicit stress annotations share the
existing frontend normalization.

Real-model regression passed for all five Russian/three Ukrainian voices,
non-silent finite audio, stress/decomposition, invalid settings and cancellation;
a Ukrainian paragraph also passed. Generated WAVs/timings are under
`target/pretrained-smoke`. This does not validate listening quality. See
`CYRILLIC.md` for the Ukrainian model card's erroneous dataset link and other
remaining provenance work. Existing own-model training is unchanged.

Studio UI was exercised with Russian F03 and Ukrainian Tetiana: language/model
switching, speaker selection and Generate completed, enabling Play/Save WAV.
The C++ harness passed its existing English ABI ownership/error/concurrency and
cancellation checks and synthesized both new Cyrillic backends. Formatting,
27 default Rust tests, strict Clippy and workspace builds passed. The optional
real-model test adds one regression covering the eight speakers. The repeatable
combined command is `python3 scripts/verify.py --cyrillic`.


## Cyrillic number readings and stress help (2026-10-02)

Added original native RU/UK nominative cardinal expansion below one billion,
feminine thousands/plural scale agreement, leading-zero digit readings and
comma/dot decimal digit readings. Built-in vocabulary carries explicit lexical
stress, avoiding dependence on an external Ukrainian stress dictionary. Numeric
format/resource tests cover both languages and valid frontend normalization for
0–10,000. Dates, signs, embedded digits, attached units and space grouping fail
explicitly. Comma is always a decimal separator; fractional readings preserve
individual digits rather than inferring grammatical fractions.

Both pretrained engines expose prepare_text; Studio previews canonical acute
stress and expanded numbers before loading a model, logs prepared text on
Generate, explains + is optional and unspoken, and uses acute accents in its
Ukrainian starter sentence. Own-model v1/v2 token contracts are unchanged.

External automatic-stress resources were evaluated but not added: the MIT code
package uses a dictionary derived from ULIF data with separate data provenance
requiring verification. General-purpose Ukrainian automatic stress remains open.


Verification passed: 30 default Rust tests, strict Clippy, workspace build, all
real pretrained speakers plus numeric requests, and C++ consumers using comma
decimal inputs. The numeric unit regression validates stress normalization for
20,002 integer readings across RU/UK. No training or model weights changed.
Generated numeric speech is under `target/pretrained-smoke/ru-numbers.wav` and
`uk-numbers.wav`; speech quality still requires listening evaluation.

Studio UI verification passed for an input with + stress and a comma decimal:
Pronunciation preview displayed `приві́т! два́дцять оди́н і три ко́ма нуль п'ять.`,
and Generate completed with playable audio. The previous Russian test sentence
was restored after rebuilding the development app; its prior WAV was preserved
at `target/studio-before-update.wav`. A separate unaccented Ukrainian sentence
was synthesized to `target/ukrainian-unaccented.wav`, demonstrating plain-text
input acceptance, without claiming equivalent pronunciation quality.

## Saved Cyrillic stress corrections (2026-10-02)

Added validated per-voice `pronunciations.json`, whole-word case/Unicode-normalized
lookup, explicit-stress precedence and language-specific spelling validation.
Rust file snapshots save atomically and reject observed external edits. CLI
add/remove/list commands and Studio's Saved word stress editor use this shared
library. No external language data, model changes or training changes were added.

Pretrained engines load corrections with the voice and can reload or replace them
in memory without reloading weights. Additive C ABI v1
`tts_engine_reload_pronunciations` serializes with synthesis. Failed reloads retain
previous corrections; removing the optional file clears them. Studio previews
saved corrections and refreshes them before Generate. Malformed files block
Studio generation until repaired. The snapshot check is not a cross-process lock.

Combined verification passed: 33 default Rust tests, formatting, strict Clippy,
workspace build, all eight real pretrained speakers, saved-stress token equivalence
and synthesis, explicit override precedence, wrong-language rejection and reload
restoration. C++ disposable voice fixtures passed actual synthesis after reload,
malformed/wrong-language reload retention and clear-on-removal for both backends.
Existing English ABI/cancellation/concurrency checks also passed.

Studio UI verified Save word persistence, plain `Привіт!` previewing and generating
as `приві́т!`, and an external CLI edit being protected from a stale Studio save,
followed by successful Reload corrections and removal. CLI add/list/remove were also exercised. Tests used a disposable voice folder;
installed voices were not modified. The rebuilt app was left on Russian F03
with the previous test sentence restored and successful playable output. Generated
library regression WAVs are
`target/pretrained-smoke/ru-saved-stress.wav` and `uk-saved-stress.wav`. Quality and
context-aware automatic stress prediction remain unvalidated.

## Fine-tuning becomes the primary own-voice workflow (2026-10-02)

The plan now requires pretrained-base adaptation, public checkpoint import and
real-speaker evaluation. Scratch training is an explicit fallback. Studio defaults
to Fine-tune, accepts a base `.pt` path and prevents starting without it. The
training CLI requires `--init-checkpoint`, `--resume` or `--from-scratch`.

Native preparation preserves the complete base token inventory, frontend and
language, including symbols absent from target recordings; unsupported symbols
fail. Both grapheme-v2 and legacy grapheme-v1 retain their original normalization.
Generator/discriminator weights load strictly into a matching architecture; new
runs reset optimizer/RNG/step state and use 5e-5 by default. Resume retains the
adapted run's state and learning rate. Export records base hash, rights/source and
initialization lineage in hashed `training-origin.json`; derived weights are no
longer labeled universally as trained from scratch.

Synthetic EN/RU/UK fixtures passed actual adaptation, inherited unused embedding
rows, preserved base file, fresh optimizer step, target resume, wrong-language/
architecture/unknown-symbol rejection, ONNX parity and native synthesis. Maximum
adapted parity errors were below 0.0000001. This does not demonstrate speech
quality or provide a useful pretrained voice. Installed inference models remain
unsupported as training bases; commercially compatible public checkpoint import
is the next required milestone.

The final default verification passed 34 Rust tests, formatting, strict Clippy and
workspace build. A native regression confirms exact base symbol indices and
legacy decomposition versus v2 NFC normalization. Studio default/base-required
validation, one full-size CPU fine-tuning step, export/native parity (maximum
error 0.0000000373), and Test exported voice generation passed using synthetic
data and the existing local legacy checkpoint. Resume disables initialization
controls. The C++ consumer synthesized the adapted export as well. The disposable
artifacts are under `target/finetune-ui-check`; they are engineering fixtures,
not a usable speech voice. The rebuilt app remains open on the training controls.

## Kokoro Q8/Q4 inference (2026-10-02)

Added pinned Q8 and Q4 ONNX packages from the existing Kokoro repository revision.
The Rust loader identifies FP32/Q8/Q4 by model hash, rejects unknown or damaged
weights, and retains existing tokenizer/style checks. Rust exposes the detected
variant. The same folders work with the existing C/C++ backend ID 0 and CLI;
no ABI change or runtime dependency was added. Studio selects the variant and
updates its folder, clears old audio and logs the actual loaded precision.

The Q4 graph contains 76 four-bit MatMulNBits nodes and roughly 301 MB of FP32
initializer payload. Thus Q4 is mixed precision and its 305 MB file is much larger
than the 92 MB Q8 file. Existing retained model/Apache and dictionary/BSD notices
are carried with both packages. Downloader hashing now streams bounded buffers,
uses unique temporary files and reuses only checksum-verified common assets.
A disposable acquisition check passed rejected-download retention, temporary-file
cleanup and successful validated replacement.

Actual FP32/Q8/Q4 native tests passed frontend equivalence, three sentence cases
with numbers, finite non-silent PCM, paragraphs, duration controls, cancellation
and invalid inputs. C++ ABI tests passed actual synthesis, ownership, engine
reuse, errors, cancellation and concurrent calls for both quantized models.
Studio switching and generation passed for Q8 and Q4, enabling Play/Save WAV
and logging the actual model variant. Artifacts/timings are under
`target/kokoro-comparison`. Listening quality is not established by these tests.
Quantized inference assets are not fine-tuning inputs to the current VITS trainer.

Sequential timings on this Apple M4 Pro gave median warm RTF 0.302 (FP32), 0.337
(Q8) and 0.308 (Q4), using two ONNX threads and a Rust debug host. Model file sizes
were 325.53/92.36/305.22 MB respectively. All timed samples were finite/non-silent
with no samples beyond +/-1. Reduced file size did not improve synthesis speed on
these samples. See KOKORO_VARIANTS.md and `target/kokoro-comparison/metrics.json`
for methodology and listening artifacts; process-memory use was not measured.

Final combined `python3 scripts/verify.py --kokoro` passed formatting, 35 default
Rust tests, strict Clippy, workspace build, actual FP32/Q8/Q4 regression and the
C++ variant consumers. The machine-readable report is `target/verification.json`.
The rebuilt Studio was left on Q4 with generated playable English audio.

## Public English base import — 2026-10-03

Added pinned acquisition and restricted `weights_only=True` import of the public
LJSpeech VITS generator. All 838 tensors and 178 indices are preserved, including
the duplicate apostrophe alias and stochastic duration predictor. The source has
no discriminator; fresh initialization is explicit in adaptation lineage. A
shared native CMU IPA frontend/dictionary hash binds preparation to exported
inference. Exports carry the dictionary/BSD notice and complete base records.
Studio now offers acquisition and selection.

Full-size CPU synthetic adaptation to step 1, resume to step 2, SDP parameter
updates, unchanged base, strict import/corrupt source/exclusive destination,
multiple-length/speed ONNX parity, exact preparation/native token agreement and
dictionary tamper rejection passed. Base/adapted maximum sample errors were
0.0000652/0.0000656; text output differed by at most two PCM16 units. C++ produced
22050 Hz English audio from the exported base. Full EN/RU/UK training regressions
passed. These are engineering checks; real speech quality remains unevaluated.

The CMU adapter is not the original eSpeak frontend. Mirror declares Apache-2.0,
but its earlier weight source has no separate license declaration; commercial
provenance confirmation remains open. Details in [public base](PUBLIC_BASE.md).

Studio's **Use English base** selection, full-size one-step CPU adaptation,
**Export and verify ONNX**, and **Test exported voice** were exercised on the
synthetic fixture. The one-step export passed native parity with maximum error
0.0001188. The updated development app bundle is available in target/.

## Fine-tuning preflight and native evaluation — 2026-10-03

Added explicit base weight/config/target dataset preflight without training-side
writes; all project checkpoint loads use restricted deserialization. Native Rust
comparison creates recording/base/adapted WAVs at fixed duration/noise, hashed
level reports and cancellation support. Python orchestration binds base SHA to
target initialization and exact target prepared-dataset digest, exports/verifies
both ONNX models, and retains unique results with checkpoint/report identities.
Studio exposes preflight metadata, checkpoint/split selection, before/after
playback and independent evaluated checkpoint copies. Saved copies reject a
changed latest checkpoint and cannot replace prior snapshots. Train controls
scroll to keep expanded preflight metadata usable in the existing window.

EN/RU/UK synthetic fixture comparisons passed held-out selection, deterministic
native output, copied original/hash checks, dataset binding, existing-output
protection, and restricted-load rejection. Rust tests cover playback tampering,
level diagnostics, cancellation and independent hash-bound checkpoint retention.
The real public English generator/one-step target produced a native held-out
comparison. No new pretrained assets or runtime dependencies were introduced.
Real speaker quality, public RU/UK trainable bases and originating English weight
terms remain open. Details in [evaluation](EVALUATION.md).

Bundle executable installation replaces the file atomically, preserving the
mapped executable of a running instance.

## Evaluated export and responsive playback — 2026-10-03

Studio now exports the completed comparison's checkpoint path and SHA-256 into
an exclusive package beside the report. It rejects a changed checkpoint before
output creation and retains any prior export. Checkpoint loading hashes the same
open handle before and after restricted deserialization. Hashed export metadata
records checkpoint path/hash/step/profile. After native parity succeeds, Test
evaluated voice selects that package and held-out transcript rather than latest.

Live preflight/expanded metadata, native held-out comparison and Keep evaluated
checkpoint passed. The retained 997,947,367-byte one-step checkpoint matched the
report hash. Clicking Play exposed a main-thread CoreAudio initialization stall,
confirmed by a process stack sample. Device creation/playback/teardown now run
on a dedicated worker, with cancellation and one pending clip. Recording/Before/
After clicks kept Studio responsive despite the still-stalled driver. A regression
simulates blocked initialization, rapid requests, Stop, errors and shutdown,
checking that obsolete audio cannot start. Audible device playback is unverified.

Full verification passed Rust tests, strict Clippy, build, alignment, EN/RU/UK
training/export comparisons and C++ synthesis/cancellation. All-language fixtures
check evaluated export/native parity, hashed identity, destination preservation
and stale-hash rejection after resume. The full-size one-step English evaluated
export passed native parity (maximum sample error 0.00011879951) and text
synthesis. Artifacts: `target/verification.json`, `target/verification-current.log`,
`target/public-base-ui-check/evaluations/comparison-g84jadg0/` and
`target/evaluated-export-current.log`. Final playback queue changes additionally
passed Studio tests and strict workspace Clippy.

The remaining evaluated English export/test buttons were subsequently exercised
on the unlocked Mac: native parity passed, the selected package was loaded, and
Studio generated 2.89 seconds of audio. CoreAudio eventually returned an Unknown
property backend error without blocking Studio; audible playback remains unverified. Real speech quality, compatible public RU/UK
training bases and earlier English weight terms remain open.


## Ukrainian public checkpoint adaptation — 2026-10-03

Added pinned acquisition/restricted import of the original Ukrainian Piper medium
training checkpoint, a supported `piper-uk-medium-v1` architecture and shared
native `piper-uk-text-v1` tokens. All 693 generator/111 discriminator tensors and
256 embedding rows are preserved. One chosen original conditioning row remains
fixed through single-target-speaker adaptation, resume and exported inference.
New target optimizers/steps start fresh; the source discriminator is retained.
The importer permits only inert pathlib.PosixPath metadata in addition to normal
restricted PyTorch loading, executes no downloaded source, and checks source
hashes before/after deserialization. Acquisition enforces exact size/hash bounds,
retains previous files on rejection and installs no existing base over an old one.
Studio offers Ukrainian acquisition/selection; native inference needs no new
backend ID, Python, eSpeak or Piper runtime.

Full-size CPU adaptation to step 1 and resume to 2 passed duration/vocoder updates,
fresh optimizer steps, unchanged base, fixed conditioning, export and native
preparation/text agreement. Nine direct native comparisons with the original
Ukrainian Piper ONNX graph passed (maximum sample error 0.00007686019). Base and
adapted PyTorch/ONNX parity passed at 0.00020243227 / 0.00003553182. Text PCM checks
use the equivalent existing 0.001 tolerance; the observed initial text difference
was seven PCM16 units. C++ consumed the adapted Ukrainian VITS package successfully.
Alternate mykyta import preserved every generator tensor and selected row 1.
Preflight, held-out comparison and changed-conditioning rejection passed. The
bounded downloader's oversized/corrupt response, preservation, cleanup and verified
cache fixtures passed, and acquisition/import successfully reused the real cache.

`python3 scripts/verify.py --full` passed the existing workspace, strict Clippy,
alignment, all EN/RU/UK scratch/adaptation/export regressions and C++ checks.
The opt-in Ukrainian real-checkpoint check is available via `--ukrainian-base`.
Evidence: `target/public-uk-base-check/result.json`,
`target/base-discovery/adaptation.log`, `original-parity.log`,
`acquisition-tool-check.log` and `target/verification-uk-increment.log`.

The Russian public generator also loaded with restricted deserialization, but
inspection confirmed a different speaker-conditioned/transformer/multiband-iSTFT
architecture. Its importer/training adapter remains the next implementation
priority; no unmatched tensors were discarded or relabeled as a supported base.
The Ukrainian source declares MIT; supplied CC0 card attribution, underlying
consent/data rights and earlier initialization remain unresolved. All target
recordings here are synthetic fixtures; real personal voice quality is still
unmeasured. See PUBLIC_UK_BASE.md and RUSSIAN_BASE_DISCOVERY.md.

Live Studio verification also passed Ukrainian base selection, compatibility
preflight, held-out comparison, evaluated checkpoint export and testing that exact
export. The comparison is retained under
`target/public-uk-base-check/adapt-gp82gl8d/evaluations/comparison-9aj5omuj`.
Test evaluated voice populated the held-out Ukrainian transcript and generated
1.27 seconds of audio in 0.05 seconds (RTF 0.039). This verifies the UI and native
generation path; audible playback and real personal voice quality are not inferred
from it. The final `python3 scripts/verify.py --ukrainian-base` run passed all
workspace checks and the original-checkpoint adaptation suite; evidence is in
`target/verification.json` and `target/verification-public-uk-final.log`.

## Russian public checkpoint adaptation — 2026-10-03

Added the separate `vosk-ru-multi-v1` profile, pinned acquisition and restricted
same-handle checkpoint import. Every original generator tensor (818), all 62
phone indices and five conditioning rows are retained. Fresh target optimizers
start at step zero; the publisher's iteration is preserved without calling it an
absolute step count. F01 is the default fixed conditioning row, M02 alternate
import preserved all tensors. Native `vosk::Frontend` is now model-independent
and shared by original inference, training preparation and exported own voices.
Dictionary probability selection, normalization, numbers, manual stress and
BOS/pad/EOS indices are unchanged. Internal scalar symbols represent phones;
unrecognized inventories/dictionaries and changed conditioning are rejected.

Full-size CPU adaptation and resume passed at steps 1 and 2, including mel
posterior input, speaker-conditioned encoder, transformer flows, stochastic
duration predictor, duration GAN and multi-resolution subband losses. Waveform
and duration discriminators are fresh because the publisher supplies only G.
Both optimizers, noise-scaled alignment schedule and RNG survive resume. All
generator components update, the base stays unchanged, and inactive conditioning
rows survive within the recorded weight-decay tolerance. This is a project
adaptation objective, not an exact replay of the publisher's training recipe.

Found and corrected the publisher ONNX conversion's missing inverse-STFT overlap
normalization. Real-valued export now matches the complex training inverse at
maximum error 0.00000033527613 on six prepared-token/scale cases. Native base
verification including original training-inverse references passed at
0.00017069466; adapted export at 0.000041587162. The original unnormalized
conversion separately matched the publisher's ONNX graph at 0.00013152137.
Raw normalized exports are deliberately not described as identical to the
publisher's differently scaled waveform. Dynamic relative embeddings also
preserve short/long ONNX inputs. Shared native text/prepared PCM, long-paragraph
chunking, C++ inference, preflight, held-out comparison, source tensor equality,
exclusive import, alternate row import and bad dictionary/conditioning rejection
passed. Real acquisition/import reused every pinned cache file successfully.

`python3 scripts/verify.py --full` passed format, workspace tests, strict Clippy,
build, alignment, EN/RU/UK scratch/adaptation/resume/export and C++ checks.
`tests/russian_base_smoke.py` passed the original Russian checkpoint suite; opt-in
verification is also exposed as `scripts/verify.py --russian-base`.
Evidence: `target/verification.json`, `target/verification-russian-increment.log`,
`target/public-ru-base-check/result.json` and `target/russian-base-smoke.log`.
Studio has Russian acquire/select controls and its bundle was rebuilt. This
supersedes the earlier entry marking the Russian adapter pending.

Training source Apache-2.0, upstream architecture/PQMF MIT and STFT BSD notices
are retained in source and export, without additional package/runtime dependencies.
Dictionary/data/consent/derived voice lineage and personal voice listening quality
remain open. All target recordings here are synthetic tones. See PUBLIC_RU_BASE.md.

Live Studio Russian verification passed base selection, compatibility preflight,
held-out comparison, hash-bound evaluated export and testing that exact package.
The comparison is retained under
`target/public-ru-base-check/adapt-sky_wmrp/evaluations/comparison-fky77ghk`.
Test evaluated voice selected its `exported-voice` folder, populated the held-out
Russian transcript and generated 2.21 seconds of audio in 0.17 seconds (RTF 0.079,
model load excluded). Native export parity was 0.000041587162. Play and Save were
enabled; audible playback was not asserted. This remains synthetic fixture
adaptation, not measured personal voice quality.

The existing real-model regression also passed after the frontend refactor:
all five original Russian and three Ukrainian voices, saved/explicit stress,
number expansion, errors and cancellation. Evidence is in
`target/russian-existing-cyrillic.log` and `target/pretrained-smoke/`.

## Language frontend expansion

Added original text-v1 rules for EN/RU/UK dates, times, signs, units, percentages,
prices, ordinals, fractions, en-dash ranges and selected abbreviations. Phrase
cues select a bounded set of homographs; explicit/saved Cyrillic corrections win.
Ukrainian stress on є is now accepted. No external language data/dependency added.

New target runs opt into versioned outer normalization; old trained VITS package behavior and
resume/evaluation preparation retain their contract. Native/PyTorch output for
raw dates and measurements matched within one 16-bit PCM count in all three
languages after synthetic-tone fine-tuning. Real Kokoro FP32/Q8/Q4 and Vosk/Piper
examples passed finite/non-silent inference and token checks. C++ text preview,
errors, ownership and concurrent calls passed. These are engineering checks, not
native-speaker quality evaluation. See [language documentation](LANGUAGE.md).

Final grammar audit corrected Ukrainian ordinal stress for twentieth, thirtieth
and fortieth, including genitive date/year inflections. Accent-preserving tests
cover these differences from Russian. English fraction plural fallback was
checked against the installed CMU inventory for denominators 2–20.

Verification evidence: `target/language-full-regression.json` and
`target/language-regression.log` retain the full training/resume/export regression;
`target/language-final-verification.log` and `target/verification.json` record
final language checks. `target/language-training-metrics.json` records raw-text
versus prepared-token PCM parity; real model WAVs are in `target/language-smoke/`.
The Studio bundle was rebuilt. Live verification of the new preview remains
unperformed because the Mac was locked; earlier Studio checks do not cover this
new preview behavior.

## Release readiness review and packaging

See [repository review](RELEASE_REVIEW.md) and [release process](RELEASE.md). Studio release startup now installs verified resources outside the bundle; installed training/export uses a bundled native CLI without Cargo. Candidate artifacts carry their own exact smoke-test report, source hashes, dependency audit and notice inventory. Model weights and Python wheels are excluded. Public distribution is still gated as documented.

Release follow-up: separate enabled normal/build dependency trees replace the overinclusive metadata inventory. `--package sdk` packages the independent CLI/shared-library kit; 78 SDK dependency notice records are complete, while Studio retains one dispatch notice gap among 209 records. Each candidate preserves its exact tree, inventory and tests. See RELEASE_REVIEW.md for the correction.

## French-first European expansion — 2026-10-04

Added stable DE/FR/ES/IT/PT/PL/NL IDs across Rust, C ABI, CLI, datasets and
checkpoint validation, preserving existing IDs and trained frontend contracts.
Added original starter prompts and native Chatterbox Q4 multilingual inference,
pinned assets, download tooling, Studio selection and packaged CLI/C++ checks.
French is the default added language. The initial greedy ONNX example failed to
terminate in both Rust and an independent Python run; the backend uses bounded
nucleus sampling with repetition penalty, temperature, min-p and top-p instead.
French native inference/cancellation and project VITS fine-tuning/resume/export
checks passed. Training fixtures are synthetic; speech quality still requires
native-speaker listening. No French public fine-tuning base is imported yet.
New-language numeric expansion is pending; spell numbers as words.
The tokenizer adds permissive dependencies: SDK notice inventory now has 134
packages and no missing texts; Studio has 253 with the existing dispatch gap.
Previous candidate inventories retain their original counts.

A packaged UI check exposed a Polish token exceeding the ONNX embedding range.
The multilingual backend now applies upstream NFKD decomposition before BPE,
separately from project VITS NFC graphemes, and rejects out-of-range text tokens
before inference. Tokenizer regression coverage includes all 84 new-language
starter prompts and canonical-equivalent French spelling. The real-model check
now covers every added language, not just French.

Verification after the Unicode correction: all seven added languages completed
native ONNX synthesis with finite, non-silent PCM. The same run checked preset
and in-flight cancellation and rejected Ukrainian on this backend. French VITS
training tests cover adaptation, resume, evaluated export and native parity on
synthetic clips. Native-speaker intelligibility evaluation is still pending.

German/Spanish follow-up: both languages passed the full synthetic project VITS
training smoke suite, including accented characters, fresh-optimizer adaptation,
resume, held-out comparison, evaluated export hash binding, native parity and
tamper rejection. The C++ consumer loaded one shared Chatterbox engine and
generated French (2.92 s), German (3.80 s) and Spanish (3.64 s) finite non-silent
audio, including umlauts/ß and Spanish inverted punctuation. The release smoke
test now includes all three languages through the relocated shared library.
`scripts/verify.py --european` makes this coverage repeatable. These checks do not
establish native-speaker quality or provide a public DE/ES fine-tuning base.

Public DE/ES base discovery: pinned and inspected the Neon CSS10 source generator
checkpoints (696 finite float32 tensors each). Added reproducible downloads and
read-only restricted inspection, with exact file and tensor-schema hashes.
Tests reject changed/truncated files before deserialization and reject executable
pickle metadata without a fallback. Real source checks confirm both models share
the 129-row grapheme inventory, blank ID 3, speaker aliases for one embedding row,
and learned language conditioning. The report explicitly distinguishes successful
inspection from training readiness. Import/conversion, original-model parity,
full weight notices and underlying data/consent review remain. No Coqui runtime
or new native dependency was added. See PUBLIC_DE_ES_BASES.md.

DE/ES conversion follow-up: the `css10-de-es-v1` adapter now maps all 696 source
tensors using existing MIT VITS components. It retains a single shared trainable
language embedding, the source speaker row and the complete token inventory.
The original publisher's hash-pinned network code is used only as a development
reference: text/posterior encoders, duration training loss, both flow directions
and deterministic full inference match, with maximum observed waveform error
below 0.00003. No original Coqui code package is shipped or required for training.

Both real imported generators passed synthetic fine-tuning, fresh optimizer and
discriminator checks, resume, preflight rejection, held-out comparison, native
ONNX and C++ checks. Prepared-token versus native-text PCM differed by at most
one 16-bit count in each language. The new shared Rust frontend preserves blank
ID 3, NFC accents and explicit source punctuation handling while rejecting other
unknown characters. Studio now has German/Spanish acquisition and selection.
Full weight notices, recording/consent provenance and real speech quality remain
unresolved; these engineering results do not establish commercial clearance.

## Remaining European public bases — 2026-10-05

Added pinned FR/IT/PT/PL/NL acquisition, restricted inspection and import under
`neon-european-v1`. Each checkpoint retains all 696 source tensors and row-zero
speaker/language conditioning. Italian uses its own 140-row inventory; the other
four use 129 rows. Rust dataset preparation and VITS inference validate the
language-specific inventory and share the `neon-graphemes-v1` frontend. Existing
DE/ES profile names and token contracts are preserved.

Studio exposes download/import and selection for all five languages. The shared
conversion/adaptation tests now accept their language codes; `--neon-bases`
runs original-network comparisons, fine-tuning/resume, evaluated export, held-out
comparison and C++ consumption for each. `test_european_bases_release.py` checks
all five through a relocated SDK with paths containing spaces and no Cargo in
PATH. Results are recorded separately under `target/neon-*-conversion.json`,
`target/neon-*-training.json` and `target/neon-release-verification.json`.

These are synthetic engineering fixtures. Real adaptation, native-speaker
quality, model copyright notices and recording/consent lineage remain unresolved.
The new languages still require numbers and abbreviations written as spoken
words. See [European training bases](EUROPEAN_TRAINING_BASES.md).

All five original-network and adaptation/export/C++ runs passed. Dutch exposed
accumulated FP32 waveform rounding (peak 0.000458, RMS 0.00000752); FP64
comparison agreed within 1.7e-13. Conversion checks now bound FP32 peak/RMS
error and require FP64 agreement within 1e-9. All seven European source
comparisons passed this strengthened check. Native exported versus prepared-token
PCM differed by at most two 16-bit counts across the five new languages.


## Reference cloning follow-up (2026-10-05)

Reviewed and hardened the optional Chatterbox reference loader. WAV headers and
sample counts are bounded before model allocation and decoding; missing, silent,
non-finite, wrong-channel/rate and out-of-range-duration references fail. Float
WAV peaks above unity remain accepted because the pinned publisher reference
contains one such sample. Model hashes and the default reference remain pinned.
A reference argument on another C ABI backend fails instead of silently ignoring
it, with the engine output cleared. Existing loader entry points remain unchanged.

Studio exposes English/Russian alongside the European cloning languages, clears
stale audio on reference changes, detects changed reference contents, and offers
return-to-default and native stereo/downmix/sample-rate preparation controls.
Preparation saves to a separate file and runs on a worker thread. Rust callers
can reuse `chatterbox::prepare_reference` and `clips::import_wav_at_rate`.

Workspace compilation and reference/ABI unit checks passed. An explicit publisher
speech reference passed real English and Russian synthesis. The independent C++
reference-loader consumer generated finite non-silent French, German and Spanish
PCM (70080, 91200 and 87360 samples at 24 kHz). Stereo 44.1 kHz preparation passed
24 kHz output and original-file preservation checks. These recordings exercise
reference conditioning using the publisher speaker; they do not assess a new
speaker's similarity. The new Studio controls have compiled but have not had a
live UI interaction check. Release archives have not been rebuilt for this change.

Descriptive voice design and Ukrainian cloning remain planned. The pinned ONNX
model has reference/exaggeration inputs, without a voice-description interface.


## Reference editor and expressiveness (2026-10-05)

Added an asynchronous Studio reference editor with mono/stereo WAV import,
direct 24 kHz resampling, time-range selection, preview, and save-and-select.
Imports are bounded to five minutes; selected references to 0.1–30 seconds.
Invalid/silent selections fail, and saving over the original is rejected.
Preview does not replace generated speech or its effects source. Trimming tests
verify selection bounds and source preservation; all six Studio tests pass.

Exposed the pinned Chatterbox exaggeration input through Rust `set_exaggeration`,
additive C ABI `tts_engine_set_exaggeration`, and a Studio Expressiveness slider.
Finite 0–1 is accepted, default 0.5; invalid settings retain the prior value.
The existing embedding graph receives the value on each token-generation step.
This is delivery intensity, not descriptive voice design or a named mood.

Workspace compilation passed. Real English/Russian reference synthesis passed at
0.7 intensity; the independent C++ consumer passed invalid-value rejection and
French/German/Spanish generation at 0.65, yielding 85440/74880/81600 samples.
Live development Studio import, trim, preview, save-and-select, and slider changes
passed. The UI fixture saved a 0.70-second selection (16800 mono samples at 24 kHz)
to target/reference-ui-check.wav. Playback reported started; no perceptual quality
assessment was performed. The pinned default float reference can be loaded directly,
but the strict preparation importer rejects its single over-unity peak; preparation
was verified with finite-range generated PCM instead. Release archives remain old.


## Portable reference profiles (2026-10-06)

Added original Rust profile storage with copied reference audio, bounded JSON,
SHA-256, suggested language, expressiveness and pinned Chatterbox model revision.
Existing profile directories are rejected. Staged assets are published into an
exclusively created folder; an in-progress/partial folder fails loading. This is
not a crash-atomic directory transaction. Model loading validates profile assets
and configures the existing native backend. No new dependency or model was added.

Studio worker-based save/open, CLI create/synthesize and additive ABI-v1
`tts_engine_load_profile` expose the workflow. Live UI verified restoration of
English/intensity 0.6 and saving an edited name with copied reference. Package
unit tests passed relocation/source removal, tampering, invalid settings/revision,
oversized metadata and existing-directory rejection. CLI saved and synthesized
from a real publisher-reference profile; independent C++ produced 18240 non-silent
finite samples at 24 kHz. Speaker similarity has not been evaluated with a new
speaker. Release archives remain unchanged.

### Native descriptive voice design — 2026-10-06

Added pinned Qwen3 VoiceDesign CPU INT4 inference to Rust, CLI, the additive
C/C++ backend 5 and Studio's **Voice design — Qwen3** selector. Instruction text
uses the pinned WaveKat user-turn template and Qwen JSON tokenizer. Downloads
verify per-file sizes/hashes; loads verify all pinned assets before opening graphs.
The native adapted source retains Apache-2.0 notices and its exact source revision.
No inference-time Python, HTTP or hosted service is involved.

Bounded speech/description characters and tokens, explicit supported-language
checks, cooperative graph-boundary cancellation and a 240-frame EOS requirement
prevent silently returning truncated output. A strict local NPY v1 reader validates
rank, f32 dtype, C layout, payload size and finite values; the trial npyz dependency
was removed after the notice audit found no full upstream license text at its
published source revision.

Real CLI English synthesis produced 60,997 mono 24 kHz samples (2.542 seconds).
The independent C++ consumer produced Russian PCM and checked nonempty finite
nonsilent audio, unsupported Ukrainian, blank/reserved descriptions and pre-request
cancellation. A second C++ run after replacing the NPY reader produced 45,317
samples. These are engineering checks, not listening, style adherence or speaker
identity evaluations. The supported project languages are EN/RU/DE/FR/ES/IT/PT;
UK/PL/NL remain unsupported by this backend.

Core/C ABI library tests: 52 passed, one existing tokenizer test ignored.
Strict workspace/all-target Clippy passed. Notice collection inspected 253 enabled
packages and retained the one pre-existing dispatch@0.2.0 upstream notice gap;
there are no new unresolved dependency notices. WaveKat attribution and full
Apache-2.0 license are now copied into release notices. Existing public release
archives have not been regenerated for this backend.

Studio selector, model path, language and description controls were inspected live.
The optimized app generated 2.71 seconds of audio in 2.51 seconds (RTF 0.928,
model load excluded), enabled Play/Save/Apply effects, and reported Playback started. The debug
build's host-side embedding projection is very slow; use an optimized build for
real-model checks. Ukrainian permissive cloning and descriptive style quality
assessment remain separate pending work.

Studio binary tests: six passed (58 tests total across the core, ABI and Studio).

### Voice-design release candidate — 2026-10-06

Extended release.py/test_release.py with --design-voice, verifying English CLI
and Russian C++ inference after SDK relocation without Cargo/DYLD overrides.
The C++ consumer also verifies cancellation and clean rejection of Ukrainian,
blank descriptions and reserved instruction tokens. External models stay unbundled.

Candidate dist/local-tts-0.1.0-aarch64-apple-darwin-a7d56ab7ee1a passed the full
packaging recipe: failure-path tests, locked workspace tests, strict Clippy,
product-specific builds, notice inventories, native linkage/architecture, ad-hoc
signatures, relocation, source consistency and archive checksum/extraction checks.
The Studio ZIP, SDK and source archives now contain the new backend/profile code.
The pre-existing dispatch notice gap and manual public-release gates remain.
The recipe launches no interactive Studio windows; no Studio processes remained.
Documentation updates after packaging are outside this candidate's recorded snapshot.

### Complete native notice inventories — 2026-10-06

Added original crates/dispatch-compat, MIT, implementing the sole legacy call
used by objc2-foundation through pinned dispatch2 0.3.1. It copies no old dispatch
source. Its test exercises synchronous borrowed callbacks, returned values and
panic recovery without crossing C. The root Cargo patch and lock select this
limited adapter; upstream API expansion must compile against it or fail.

Candidate dist/local-tts-0.1.0-aarch64-apple-darwin-91200d53ff05 passed packaging,
workspace tests, strict Clippy, architecture/linkage/signatures, relocated English
CLI/Russian C++ descriptive synthesis and archive hash/extraction verification.
The enabled Studio inventory covers 252 external packages and SDK 134, both
complete, with zero recorded missing notices. Adapter attribution accompanies the
project MIT license. Older candidates retain their original gaps unchanged.

Opened this exact packaged app once: the UI reported Ready and installed resources
correctly. Closed its window normally and verified no Studio processes remained.
This does not establish clean-machine, minimum-OS, speech-quality or notarization
acceptance. Documentation updates after packaging are outside its source snapshot.

### Qwen cold-load cancellation — 2026-10-07

Added additive Rust load_cancellable and streaming integrity cancellation checks
between 64 KiB reads. The existing noncancellable APIs retain their behavior.
Studio uses the flag during Qwen verification and between every graph/table load;
a single ONNX session load or embedding read remains synchronous. No partial
model is cached when initialization fails. Pre-cancelled missing-path and real
pinned-package interruption tests passed; normal hash equivalence is retained.

The packaged app was opened once with an uncached Qwen model. Generate followed
by Cancel restored controls and displayed Synthesis cancelled on the next UI
observation. Closing the window terminated the process; none remained running.
Studio now spells out the generated WAV → Chatterbox reference → saved profile
workflow; this is reference conditioning, not preservation of Qwen sampling state.

Candidate dist/local-tts-0.1.0-aarch64-apple-darwin-0f14857080db passed the full
packaging recipe, relocated English CLI/Russian C++ synthesis and extracted-archive
checks. Notice gaps remain empty. Documentation additions after packaging are
outside its recorded source snapshot; public-release manual gates remain.

### Inference-only CLI — 2026-10-07

Added separate local-tts-speak/tts-speak Rust application consuming the core
library directly. Supports native backends, UTF-8 text argument/file/stdin,
language/speaker options, Chatterbox reference/profile and Qwen description.
It has no training/download commands, GUI or Python subprocesses. WAV writes
are staged and atomically committed; existing outputs require explicit --force.

Argument errors and output preservation tests passed. Real Kokoro Q8 argument,
file and stdin synthesis passed, alongside strict Clippy/release build. The
standalone dist/tts-speak-0.1.0-macos-arm64 bundle contains executable, guide,
MIT license and complete enabled dependency notices (zero gaps), with checksums
and ad-hoc signature. A copied binary synthesized from an empty temporary working
directory with system-only PATH and no dynamic-runtime overrides. Linkage uses
only system libraries/frameworks. Models remain external. Other CLI backend routes
compile but have not been independently exercised through this new application.

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

Studio English preview guidance now excludes Russian/Ukrainian-only numeric
formats, and offers a one-click English numeric example. Verified Kitten preview
expansion and live synthesis with that example; both compact backends share the
same full-passage English frontend. Studio Clippy passed.

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

2026-10-09: Compact RU/UK arithmetic shares the literal Rust parser and existing
stressed cardinal rules, with fixed-speaker decoder export and schema-2 binary
banks. Russian 8,837,903-byte and Ukrainian 2,800,833-byte packages run through
`MathStudent`, CLI and the shared FFI backend; Studio selectors rebuilt. Native
fixtures, vocabulary/parser coverage, nonfinite/output checks and language guards
pass. Listening quality remains experimental; isolated banks and Russian 4-bit
error require review. See MATH_LATENT.md for reproduction and source-rights limits.

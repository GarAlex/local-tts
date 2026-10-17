# Fine-tuning preflight and listening evaluation

Studio can check a selected base against the target dataset before training,
then generate held-out recording/base/adapted comparisons with the native
library. These tools provide controlled listening samples and audio diagnostics;
no automatic intelligibility, similarity or subjective quality score is claimed.

## Check compatibility

In **Train a voice**, select the base checkpoint and dataset, then click
**Check base and target compatibility**. Expand **Base preflight passed
(snapshot)** for language, frontend, architecture, sample rate, full inventory
size, base speaker, target speaker and split counts. Checkpoint/dataset hashes and
inherited provenance/frontend notes are retained. Editing the paths hides the
old result; training independently validates its inputs again.

```sh
.venv/bin/python training/preflight.py --checkpoint voices/vits-ljs-base/base.pt --dataset /path/to/dataset.json
```

Preflight loads supported full-size/project-tiny/LJSpeech profiles strictly,
checks generator and available discriminator shapes and finite weights, verifies
the dictionary hash and prepares the target through Rust. It uses temporary files
and does not start training, write a checkpoint or change the base/dataset.
Project checkpoint loading now uses restricted `weights_only=True` in preflight,
training initialization/resume, launch and export. This is still a project-format
loader, not a general converter for arbitrary external architectures.

## Compare and keep a checkpoint

Click **Evaluate before / after…**. Choose a target `.pt` checkpoint (default
`latest.pt`), Validation or Test split, and 1–12 clips. **Generate comparison** uses
the training tab's selected base and dataset. It displays each transcript with
**Play Recording**, **Play Before** and **Play After**, plus duration, RMS, peak and
clipping fraction. Evaluate validation throughout development; reserve test data
for final decisions to keep the test set useful. Dataset warnings, including
repeated transcripts across splits, are shown.

```sh
.venv/bin/python training/compare.py --base voices/vits-ljs-base/base.pt --checkpoint checkpoints/my-voice/latest.pt --dataset /path/to/dataset.json --output checkpoints/my-voice/evaluations --split validation --limit 3
```

The checkpoint's recorded initialization must match the selected base SHA-256.
Both architectures/frontends/inventories and dictionary hashes must agree. Native
preparation must match the target checkpoint's exact prepared dataset digest;
changed recordings, tokens, speaker or rights metadata cannot be relabeled as its
held-out set. The tool exports and numerically verifies both models, then calls
Rust VITS inference with the same held-out text, deterministic noise settings
and duration scale 1.0. It uses the single speaker in each model's metadata.
It never selects training clips or silently substitutes another base/dataset.

Comparisons create unique `evaluations/comparison-*/` folders, preserving prior
results. `samples/` contains copies of the original held-out recordings,
before/after WAVs and `report.json` with hashes and unprocessed level statistics.
`comparison.json` records checkpoint identities/steps, dataset/prepared hashes,
source notes, engine settings and the report hash. Temporary ONNX packages are
removed. A completed `comparison.json` indicates the whole workflow finished;
partial folders can remain after failure or Stop. Stop forwards termination to
the active export/native child. Everything stays local after dependencies and
base assets have been installed.

**Keep evaluated checkpoint** saves an independent, byte-identical copy under
that checkpoint's `saved-checkpoints/step-<step>-<hash>.pt`. The current file must
still match the evaluated hash, so an advancing `latest.pt` cannot be saved under
an older evaluation. Installation is exclusive: existing snapshots are retained.
Copies can be large (roughly a gigabyte for the current full training checkpoint).
Select a saved `.pt` later for comparison or export with the ordinary CLI tools.
This feature keeps chosen evaluated checkpoints; training still overwrites
`latest.pt` and does not retain every periodic update automatically.

## Export the evaluated model

**Export evaluated checkpoint** reads the checkpoint path and SHA-256 recorded
in the completed comparison, independently of subsequent edits to the picker.
It rejects a changed checkpoint before deserialization or output creation. The
file handle is hashed again after loading to detect in-place writes. Atomic
replacement during loading preserves the already-open evaluated snapshot.
Export creates an exclusive new `comparison-*/exported-voice/` folder, then
verifies numerical parity with native Rust inference. Existing packages are
retained. Failed exports can leave a partial folder; use a new evaluation/output
for another attempt. Every export includes `exported-checkpoint.json` with the
source path, hash, step and model profile, covered by package checksums.

**Test evaluated voice** appears only after native verification succeeds for the
current comparison's target hash. It selects that package in Test speech and
fills in the first held-out transcript at duration 1.0. The training tab's general
**Export and verify ONNX** / **Test exported voice** still use latest and its
ordinary `voice/` folder.

```sh
.venv/bin/python training/export_and_verify.py --checkpoint /path/to/evaluated.pt --expected-sha256 <64-character-lowercase-sha256> --new-output --output /path/to/new-voice
```

Studio creates and tears down audio devices on a dedicated worker. **Stop
playback** cancels pending startup; an obsolete request cannot begin after a slow
driver returns. Only one pending clip is retained. A stalled OS audio driver can
still prevent audible playback, but it no longer blocks the UI or application
exit. This does not change the library's PCM/WAV output.

## Reusable native API

`local_tts_core::evaluation::compare` compares two compatible exported VITS
packages against validation/test recordings and writes the hashed listening
artifacts. It accepts cooperative cancellation, rejects existing output folders,
and enforces single speakers, matching language/frontend/sample rate/token
inventory/dictionary and at most twelve selected clips of twenty seconds each.
`evaluation::read_audio` verifies the file hash and path before playback.
`training::snapshot_checkpoint` copies an evaluated checkpoint only when its
bytes match the requested hash. All are local Rust functions, reusable by other
applications; no new server or inference runtime is added.

```sh
local-tts vits-compare /path/to/base/voice /path/to/adapted/voice /path/to/dataset.json /path/to/new-output validation 3
```

Direct native comparison accepts exported folders; checkpoint lineage/dataset
binding is provided by the development Python orchestration. Native synthesis,
comparison audio and embedding continue to use the existing Rust/ONNX backend.
Audio-level statistics are calculated before WAV clamping. Playback is verified
against stored WAV hashes; changing an artifact fails instead of playing it under
the old label. Listening samples do not resolve model/data license provenance.

## Verification and limits

`python3 scripts/verify.py --full` now exercises preflight and comparison in the
English, Russian and Ukrainian training fixtures: read-only base inspection,
strict loading, held-out selection, deterministic native audio, exact reference
copy, checkpoint/dataset/report hashes, changed-dataset and existing-output
rejection. A pickle instruction fixture verifies that restricted loading rejects
execution. Evaluated exports also pass hash/provenance/native parity checks,
reject existing folders, and reject an old hash after resume replaces latest.
Rust tests simulate a blocked audio driver, rapid requests, Stop and shutdown,
and cover level diagnostics, cancellation/training-split
rejection, playback tampering and snapshot independence/hash binding.

The installed full-size public English base and its one-step target also produced
a native comparison (`The cat sat.`) here. Those target recordings are synthetic
engineering fixtures; no real personal voice quality or native-speaker assessment
has been established. The Russian public training adapter, Ukrainian source rights lineage and English
originating weight-provenance gap remain open. This increment adds no models, dictionaries
or inference dependencies beyond the existing installed packages.

Live Studio preflight (including expanded metadata), native comparison, all three
playback requests and checkpoint retention were exercised. Playback exposed a
CoreAudio initialization stall; the worker fix kept Studio responsive, and device
setup eventually reported an Unknown property backend error. Audible playback is
still unverified on this device. Evaluated English export/test button interactions
then passed, selecting the exact package and generating 2.89 seconds of native
audio. The Ukrainian public checkpoint path is also exercised by the opt-in
`tests/ukrainian_base_smoke.py`; see PUBLIC_UK_BASE.md for source/quality limits.

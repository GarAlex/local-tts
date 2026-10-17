# Public English fine-tuning base

The pinned LJSpeech VITS generator can now be imported, fine-tuned, resumed,
exported to ONNX and embedded through Rust or the existing C/C++ API. This is a
real pretrained speech generator; the automated adaptation fixtures are synthetic
engineering data and do not produce a useful personal voice.

```sh
# Requires the installed training environment; acquisition needs Internet.
.venv/bin/python scripts/prepare_english_base.py
.venv/bin/python training/launch.py --dataset /path/to/dataset.json --output checkpoints/my-voice --init-checkpoint voices/vits-ljs-base/base.pt --steps 10000 --batch-size 2 --device cpu
.venv/bin/python training/export_and_verify.py --checkpoint checkpoints/my-voice/latest.pt --output checkpoints/my-voice/voice
cargo run -p local-tts-cli -- vits-synthesize checkpoints/my-voice/voice speech.wav 'Hello world.'
```

Studio's Train a voice tab offers **Download / import English base**, then
**Use English base**. Import never overwrites an existing checkpoint. Base files
live at `voices/vits-ljs-base/base.pt`, with the development frontend under
`frontend/` and pinned downloaded records under `source/`. The checkpoint records
an absolute dictionary path: re-import to relocate a training base. An exported
voice contains its own dictionary and notices and is portable. Inference requires
only the native library and the voice folder.

## Import and adaptation contract

- `assets/vits-ljs-assets.json` pins source revisions and every downloaded hash.
  Import checks all files before deserializing external weights with PyTorch
  `weights_only=True`. Downloaded Python is parsed for literal inventory strings,
  never executed. Only this known architecture is accepted.
- All 838 generator tensors load strictly. The original stochastic duration
  predictor is preserved and trained; none of its layers is replaced by a
  deterministic predictor. The source contains no discriminator: adaptation
  explicitly initializes a fresh one, with fresh generator/discriminator
  optimizers, step counter and the existing default learning rate of 5e-5.
- All 178 embedding indices are retained. Blank index zero is represented as NUL
  in the project schema. Upstream duplicates apostrophe; both rows remain and
  text lookup uses the last index, matching upstream's dictionary construction.
- The source checkpoint reports iteration zero. This is retained as publisher
  metadata and does not establish how many training steps produced these weights.
- Native `cmu-ipa-v1` uses the pinned BSD CMU dictionary and existing English
  integer/decimal expansion. Kokoro diphthong shortcuts are expanded into IPA.
  Unknown words and unsupported numeric forms fail explicitly. This is an
  approximate CMU adapter, **not identical to upstream english_cleaners2/eSpeak**:
  stress placement, vowel lengths and contextual pronunciations can differ.
  Training preparation and native inference use the same version and dictionary.
- Resume binds the prepared dataset, full inventory and dictionary hash. Export
  bundles the dictionary/BSD notice, source card, source and declared weight
  licenses, dataset record and import lineage; all are included in checksums.
  The existing `Vits` backend/C ABI accepts these folders without an ABI change.

## Terms and remaining provenance work

The [weight mirror](https://huggingface.co/csukuangfj/vits-ljs) declares Apache-2.0
and says its checkpoint was copied from
[an earlier mirror](https://huggingface.co/yo2266911/mb-vits-models), which does not
supply a separate license declaration in its repository metadata. The Apache
license text retained here is the standard text, with its source URL in the
manifest; it is not a separate grant from the originating model author.
[Upstream VITS code](https://github.com/jaywalnut310/vits/blob/main/LICENSE) is MIT.
[The LJSpeech publisher](https://keithito.com/LJ-Speech-Dataset/) declares the
recordings and texts public domain in the USA and notes possible differences
elsewhere. The dictionary is BSD. These records are kept separately because a
code license alone does not establish weight or dataset rights.

This is an evaluated import candidate, **not a completed commercial provenance
audit**. Confirm originating weight redistribution terms before treating this
base as an unrestricted commercial shipping asset. Fine-tuning retains base
terms and does not resolve that gap. The project importer, native frontend and
training tools introduce no copyleft runtime component. Real consenting-speaker
adaptation, listening/intelligibility/similarity evaluation, and RU/UK trainable
public bases remain required milestones.

## Verification

```sh
python3 scripts/verify.py --full --public-base
```

The optional public-base test requires installed source/imported weights. It
checks every imported generator tensor against the source, exclusive output and
corrupt-source rejection, full-size CPU adaptation and duration-weight updates,
resume, unchanged base hash, native token/PyTorch parity at several lengths and
speeds, shared preparation/text tokenization, carried notices and dictionary
corruption rejection. Outputs are under `target/public-base-check/`.

On this Mac, initial base export maximum absolute sample error was 0.0000582;
expanded base fixtures were 0.0000652, and the two-step adapted model 0.0000656
(tolerance 0.001). The shared text-preparation comparison differed by at most two
PCM16 units. C++ synthesized the public base at 22050 Hz. Studio base selection, one-step
adaptation and export/native verification passed (one-step export error 0.0001188). Synthetic training and
numerical parity do not establish speech quality. CUDA and other platforms remain
unverified.

Studio also offers explicit read-only base/dataset preflight and native held-out
before/after comparison with selectable/kept evaluated checkpoints. See
[evaluation tools](EVALUATION.md). These additions do not resolve the weight
provenance or speech quality gaps above.

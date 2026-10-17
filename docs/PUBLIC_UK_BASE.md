# Ukrainian fine-tuning base

The original Ukrainian Piper medium training checkpoint can be imported and
adapted to one new speaker, with native Rust/C/C++ inference after ONNX export.
This uses training weights rather than the inference-only ONNX package. The
working base is an engineering candidate; source rights gaps and real speech
quality still need evaluation before unrestricted commercial distribution.

## Use it

Studio offers **Download / import Ukrainian base**, then **Use Ukrainian base**
in Train a voice. Select a Ukrainian 22050 Hz dataset and a new checkpoint folder,
check compatibility, and start fine-tuning. Resume, native before/after evaluation,
checkpoint retention and hash-bound evaluated export use the existing workflow.
The imported base defaults to the original `lada` conditioning row.

```sh
.venv/bin/python scripts/prepare_ukrainian_base.py
.venv/bin/python training/launch.py --dataset /path/to/ukrainian-dataset.json --output checkpoints/my-uk-voice --init-checkpoint voices/vits-piper-uk-base/base.pt --steps 10000 --batch-size 2 --device cpu
.venv/bin/python training/export_and_verify.py --checkpoint checkpoints/my-uk-voice/latest.pt --output checkpoints/my-uk-voice/voice
cargo run -p local-tts-cli -- vits-synthesize checkpoints/my-uk-voice/voice ukrainian.wav 'Прив+іт, сві́те!'
```

Base installation is exclusive. To select another original conditioning row,
import into a separate checkpoint; `--speaker` accepts `lada`, `mykyta`, `tetiana`:

```sh
.venv/bin/python scripts/prepare_ukrainian_base.py --speaker mykyta --output voices/vits-piper-uk-mykyta/base.pt
```

The selected row is retained during adaptation/resume and baked into exported
inference; exports have one named target speaker. Selecting another row is not
cross-language transfer or a promise of voice similarity with less training.
The imported source retains three conditioning embeddings, but this trainer
continues to require one target speaker per dataset. Only the `lada` native
baseline has been exercised against the original ONNX graph so far; the alternate
`mykyta` import passed complete tensor preservation and index selection checks.

## Contract

`assets/vits-piper-uk-assets.json` pins the original checkpoint repository revision,
file sizes, SHA-256 hashes, source MIT notice, model card and dataset records.
The download tool bounds responses to each pinned size, retains existing files
on failure and reuses only verified cache entries. Import hashes the checkpoint's
open handle before and after deserialization, loads with `weights_only=True`,
and additionally permits only the inert standard-library `pathlib.PosixPath`
metadata present in this checkpoint. Downloaded Python/shell files are retained
as records and never executed. No unsafe pickle fallback is used.

All 693 generator and 111 discriminator tensors load strictly. The supported
`piper-uk-medium-v1` architecture retains the stochastic duration predictor,
ResBlock2 vocoder, original upsampling and 512-channel speaker conditioning.
Source optimizers are discarded. New adaptation starts at zero with fresh
optimizers and 5e-5 learning rate, using the original discriminator; resume
continues the target run and binds the prepared dataset and conditioning row.
Publisher epoch/global-step metadata is retained, rather than inferred as a
count of generator updates.

`piper-uk-text-v1` is shared by native preparation and inference. It lowercases,
canonicalizes apostrophes and explicit stress, expands the supported plain numbers,
then decomposes Unicode to NFD. It keeps the original 256 embedding rows and exact
BOS/pad/character/pad/EOS indices. Unused rows have reserved internal placeholders,
which cannot be entered as text. Unknown characters and reserved tokens fail.
Stress accepts `+` before a vowel or acute accent after it. This version adds no
automatic word-stress predictor and uses no eSpeak or Piper runtime.

Training and inference both use that project preprocessing. Number/stress
conventions must also be used consistently when authoring transcripts. The
frontend matches the already implemented Ukrainian Piper native frontend; it
is separate from project grapheme-v1/v2 and English CMU IPA. ONNX packages carry
complete source notices, rights records, initialization lineage and checksums.
They require neither Python nor the original checkpoint at inference time.

## Verification and remaining work

```sh
python3 scripts/verify.py --full --ukrainian-base
```

The opt-in test requires acquired source/base weights and the installed original
Ukrainian ONNX package. It checks exact source tensor preservation, source hash
and exclusive install rejection, alternative conditioning selection, full-size
CPU adaptation, duration/vocoder updates, fresh optimizers, resume, unchanged
base, export/parity, matching native training/text tokens, C++ synthesis,
preflight and native held-out comparison, and rejection of changed conditioning.
Download fixtures cover oversized/corrupt responses, preservation, cleanup and
verified cache reuse. Outputs are in `target/public-uk-base-check/`.

Nine native comparisons against the original Piper ONNX graph (three texts at
three speeds) passed with maximum sample error 0.00007686019. Base versus PyTorch
export parity passed at 0.00020243227; the two-step adapted export passed at
0.00003553182 (the existing tolerance is 0.001). PCM16 text versus prepared-token
comparison uses the corresponding 0.001 tolerance, including quantization.
Recordings are synthetic tones: these checks establish mechanics and numerical
agreement, not useful personal voice quality or a recommended recording minimum.
CUDA, Windows and sustained real-data training remain unverified.

## Source records

The [original checkpoint repository](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
declares MIT. [Original Piper source](https://github.com/rhasspy/piper) is MIT.
The supplied Ukrainian model card claims CC0 and links a dataset collection
without Ukrainian data. The [actual Ukrainian dataset project](https://github.com/egorsmkv/ukrainian-tts-datasets)
declares Apache-2.0; its README/license and the supplied card are retained
separately. The card's CC0 attribution is not treated as verified. Complete
consent/rights lineage and any earlier base initialization remain unspecified.
Fine-tuning does not clear those source gaps. This importer introduces no copyleft
runtime or automatic commercial-clearance claim.

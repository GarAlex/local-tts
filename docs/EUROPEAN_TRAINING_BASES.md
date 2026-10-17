# French, Italian, Portuguese, Polish and Dutch fine-tuning

Studio's **Train a voice** panel now provides **Download / import** and **Use base**
buttons for all five languages. Set up the training environment first, import the
base for your recording language, select it, choose your dataset manifest, and
run **Check base and target compatibility** before training. Use consenting-speaker
recordings with accurate transcripts at 22050 Hz. The dataset wizard explains
recording and transcript preparation. Fine-tuning is the primary workflow.

## Command line

```sh
.venv/bin/python scripts/prepare_european_base.py fr
.venv/bin/python scripts/prepare_european_base.py it
.venv/bin/python scripts/prepare_european_base.py pt
.venv/bin/python scripts/prepare_european_base.py pl
.venv/bin/python scripts/prepare_european_base.py nl

.venv/bin/python training/preflight.py \
  --checkpoint voices/vits-neon-fr-base/base.pt --dataset recordings/dataset.json
.venv/bin/python training/launch.py \
  --init-checkpoint voices/vits-neon-fr-base/base.pt \
  --dataset recordings/dataset.json --output checkpoints/my-french-voice \
  --steps 10000 --batch-size 2
```

Replace `fr` with the recording language. Existing base files are never overwritten;
`--output` selects a new destination. Acquisition requires a network connection;
import, training and exported Rust/C++ inference run locally. Python is required
for training/import, not for consuming the exported library. The example step
count is a starting configuration, not a voice-quality guarantee.

## Source contracts

| Language | Pinned source repository | Token rows | Source dataset family |
| --- | --- | ---: | --- |
| French | [Neon CSS10 French](https://huggingface.co/neongeckocom/tts-vits-css10-fr) | 129 | CSS10 |
| Italian | [Neon CV Italian](https://huggingface.co/neongeckocom/tts-vits-cv-it) | 140 | Common Voice |
| Portuguese | [Neon CV Portuguese](https://huggingface.co/neongeckocom/tts-vits-cv-pt) | 129 | Common Voice |
| Polish | [Neon MAI Polish](https://huggingface.co/neongeckocom/tts-vits-mai-pl) | 129 | MAI |
| Dutch | [Neon CSS10 Dutch](https://huggingface.co/neongeckocom/tts-vits-css10-nl) | 129 | CSS10 |

`assets/vits-neon-european.json` pins revisions, SHA-256 hashes, byte sizes,
configuration, model cards and a tensor schema for every source. Import verifies
all required assets and uses restricted `torch.load(weights_only=True)` after
checking the open file, then checks it again. It does not execute publisher Python
or load the optional `model.pt` application package.

`neon-european-v1` retains all 696 generator tensors, including speaker row zero
and the shared language embedding. It initializes a fresh discriminator and
optimizer for target fine-tuning. `neon-graphemes-v1` preserves the original token
row order and blank ID 3. Italian's additional rows change IDs of accented letters;
its inventory must not be replaced by the 129-row inventory. Existing German and
Spanish `css10-de-es-v1` checkpoints remain unchanged.

Native preparation and exported inference share the same Rust frontend. It
normalizes canonical Unicode, lowercase and whitespace, and applies the existing
source punctuation policy. Unsupported characters fail explicitly. Write numbers
and abbreviations as spoken words; numeric expansion and contextual pronunciation
for these languages remain pending. Portuguese dialect selection is not exposed.

## Verification and limits

The engineering test runner is `python3 scripts/verify.py --neon-bases`. It requires
the five imported bases and optional hash-pinned publisher `model.pt` references.
Acquire those with `--test-reference` on the initial preparation command. The
references are used only by explicit comparison tests and are never distributed
in the app or SDK. When a base already exists, download its reference using the
pinned URL/hash in the catalogue to a separate test directory and pass
`--reference` to `tests/css10_conversion.py`.

For each language, tests compare original and converted text/posterior encoders,
flows, duration training objective and complete inference (bounded FP32 error
and near-exact FP64 comparison); then exercise actual
fine-tuning, resume, held-out comparison, evaluated ONNX export, native text/token
parity and a C++ consumer. Fixtures use synthetic tones. They do not establish
intelligibility, pronunciation quality, speaker similarity or real voice adaptation.
Those require recordings and held-out native-speaker listening.

The publisher model cards declare BSD-3-Clause but omit full weight copyright
notices and detailed recording/consent lineage. The pinned Coqui registry supports
matching declarations for FR/NL/PT/PL; its Italian entries concern different
models, so they are not used as evidence for this Italian CV checkpoint. CSS10
repository licensing is recorded separately from audiobook rights. Common Voice
and MAI exact training releases/selections are not established by these cards.
Import/export retains these limitations and source records. These bases are
engineering integrations, not a commercial-clearance determination. No weights
or publisher code packages are bundled with releases.

# German and Spanish public training bases

The Neon CSS10 generators can now be imported as project fine-tuning checkpoints.
The `css10-de-es-v1` adapter maps all 696 tensors and preserves speaker/language
conditioning. Preparation and inference share the native `css10-graphemes-v1`
frontend. Commercial provenance gaps below remain.

## Import and fine-tune

In Studio's **Train a voice** panel, set up the training environment, then use
**Download / import German base** or **Download / import Spanish base**. Select
the corresponding **Use … base** button and a dataset in that language. Run the
compatibility check before training. CLI equivalents:

```sh
.venv/bin/python scripts/prepare_css10_base.py de
.venv/bin/python scripts/prepare_css10_base.py es
.venv/bin/python training/preflight.py \
  --checkpoint voices/vits-css10-de-base/base.pt --dataset datasets/my-voice/dataset.json
.venv/bin/python training/launch.py \
  --dataset datasets/my-voice/dataset.json --output checkpoints/my-german-voice \
  --init-checkpoint voices/vits-css10-de-base/base.pt --steps 10000 --batch-size 2
```

The step count is an example, not a voice-quality guarantee. Use mono 22,050 Hz
recordings, exact transcripts and held-out splits. Fine-tuning starts a fresh
discriminator and optimizers; resume continues that target run. Keep source and
target languages the same. Import never overwrites an existing checkpoint.
Export and listening comparison follow the [evaluation workflow](EVALUATION.md).

## Reproduce acquisition and inspection

```sh
python3 scripts/download_css10_candidates.py all
.venv/bin/python training/inspect_css10.py de --report target/german-base-inspection.json
.venv/bin/python training/inspect_css10.py es --report target/spanish-base-inspection.json
.venv/bin/python tests/css10_candidates.py --installed
```

Use `de` or `es` instead of `all` to acquire one candidate. Each checkpoint is
108,563,599 bytes, plus small metadata records. `--output` selects a parent
directory, with separate `de` and `es` subfolders. Inspection accepts `--source`
for the corresponding language folder. Report paths must be new; existing reports
are preserved. A successful inspection means the pinned assets and recorded
contracts match; `training_ready` remains false for raw source files because
they require conversion with `training/import_css10.py de --output <new-base.pt>`.

`assets/vits-css10-candidates.json` pins immutable model revisions, all file
sizes/hashes, the upstream model registry and CSS10 source records. Acquisition
uses checked temporary downloads. Inspection checks exact source bytes before
restricted `torch.load(weights_only=True)`, checks the same file handle again
after loading, validates all tensor names/shapes and finite float32 contents,
and compares the embedded checkpoint configuration with the downloaded config.
It never imports publisher Python, loads TorchScript, expands pickle globals or
falls back to unrestricted deserialization. No new runtime dependency is added.

## Observed model contracts

| Property | German | Spanish |
|---|---|---|
| Publisher | `neongeckocom/tts-vits-css10-de` | `neongeckocom/tts-vits-css10-es` |
| Revision | `408f0bf6d6d71e24c7544c9425cdc5cd82829734` | `6477ababf77c3102cf1fbbc3092810c290634085` |
| Generator tensors | 696 | 696 |
| Sample rate | 22,050 Hz | 22,050 Hz |
| Source speaker aliases | `de`, `hokuspokus` → row 0 | `es`, `tux` → row 0 |
| Language conditioning | `de` → row 0 | `es` → row 0 |
| Text | 129 grapheme/special-token rows | Same inventory |

Both use a 192-channel character embedding plus four learned language channels,
a 196-channel text transformer, a 256-channel speaker embedding, a stochastic
duration predictor with language conditioning, and a compact VITS decoder.
There are no discriminator weights in these source checkpoints.

The token contract differs from this project's grapheme-v2 profile: pad is 0,
EOS is 1, BOS is 2, blank is 3; blank is inserted between tokens and at both
boundaries, while BOS/EOS are disabled. The upstream multilingual cleaner also
has distinct punctuation handling. The inventory includes umlauts, ß, accented
vowels, ñ and ü, but omits inverted Spanish punctuation. A converter must preserve
all source rows and implement an explicit shared text contract. Relabeling these
weights as an existing project profile would be incorrect. The importer uses a
new architecture profile, preserving one shared trainable language embedding.

Native text normalization applies NFC, lowercasing and whitespace cleanup.
Semicolons/colons become commas and hyphens become spaces. Auxiliary brackets
and quotes are removed, along with `¡`/`¿`; closing punctuation is retained.
Other unknown characters cause an error instead of silent deletion. Spell out
numbers and abbreviations. Private Unicode placeholders represent special token
rows in project metadata and cannot be entered as user text.

## Terms and provenance

The [German model card](https://huggingface.co/neongeckocom/tts-vits-css10-de) and
[Spanish model card](https://huggingface.co/neongeckocom/tts-vits-css10-es) declare
BSD-3-Clause, as does the pinned upstream model registry. The model repositories
do not supply a full copyright notice for the weights. That notice remains a
distribution gap; the tools do not manufacture one or mark commercial clearance.

The [CSS10 source repository](https://github.com/Kyubyong/css10) identifies the
German and Spanish LibriVox books and readers. Its Apache-2.0 repository license
is retained separately; it does not replace the weight license or establish all
underlying audiobook and speaker rights. Territorial text rights and consent
for voice adaptation have not been independently verified. These candidates
must not be advertised as commercially cleared bases yet.

## Implemented acceptance checks and remaining work

Original-versus-converted tests cover text/posterior encoders, forward/reverse
flows, the duration training objective and full deterministic inference at
different text lengths. Observed maximum waveform error is below 0.00003. The
adapter uses the existing MIT VITS components and adds no Coqui dependency to
application inference or project training.

The optional comparison tests execute the publisher's hash-pinned network code
from its `model.pt` package. This is separate from restricted weight-only import
and does not deserialize the publisher's application pickle. It is used only as
a development reference. Models and reference packages remain outside releases.

```sh
python3 scripts/download_css10_candidates.py all --test-reference
python3 scripts/verify.py --css10-bases
```

Both imported bases must already exist for the second command. Full-size
fine-tuning smoke tests use synthetic recordings and cover fresh optimizers,
inventory/conditioning retention, resume, preflight rejection, dynamic ONNX
parity, native text/prepared-token agreement, C++ consumption and held-out
comparison. Studio has import/select controls for both languages. Real
consenting-speaker adaptation, native-speaker evaluation and provenance
resolution remain; engineering checks do not establish speech quality.

The current tests cover hash/size rejection before loading, refusal of executable
pickle metadata, restricted valid loading, both installed source checkpoints and
their special-token contracts. `scripts/verify.py --css10-candidates` combines
these with the normal workspace checks. Reports are saved in
`target/css10-candidate-inspection.json`. Conversion and adaptation reports are
`target/css10-{de,es}-conversion.json` and `target/css10-{de,es}-training.json`.

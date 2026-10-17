# Russian pretrained-base fine-tuning

The original Vosk 0.7 multispeaker training generator is supported as
`vosk-ru-multi-v1`. It is a separate architecture from the English and Ukrainian
bases: speaker-conditioned text encoder, transformer flows, mel posterior,
stochastic duration predictor and four-band iSTFT/PQMF vocoder. Import preserves
all 818 learned tensors, 62 phone indices and five source speaker rows.

In Studio, open **Train a voice**, choose **Use Russian base**, select a Russian
dataset and a new checkpoint folder, then run **Check base and target
compatibility**. Fine-tune is the default. Resume continues the target run.
**Evaluate before / after** compares held-out recordings and native base/adapted
speech; evaluated export/test uses the selected checkpoint snapshot.

The acquisition button requires the Russian inference dictionary to be installed
first. CLI setup and adaptation:

```sh
python3 scripts/download_cyrillic.py ru
.venv/bin/python scripts/prepare_russian_base.py
.venv/bin/python training/preflight.py \
  --checkpoint voices/vits-vosk-ru-base/base.pt --dataset /path/to/dataset.json
.venv/bin/python training/launch.py \
  --dataset /path/to/dataset.json --output checkpoints/russian-voice \
  --init-checkpoint voices/vits-vosk-ru-base/base.pt --steps 10000
.venv/bin/python training/launch.py \
  --dataset /path/to/dataset.json --output checkpoints/russian-voice \
  --resume --steps 20000
.venv/bin/python training/export_and_verify.py \
  --checkpoint checkpoints/russian-voice/latest.pt --output voices/my-russian
target/debug/local-tts vits-synthesize voices/my-russian russian.wav 'Прив+ет, м+ир!'
```

These step counts illustrate the CLI, not measured recording/compute requirements
for a usable voice. Training accepts one target speaker, 22050 Hz mono WAV,
train/validation splits and clips between 8192 samples and twenty seconds. The
current tests use synthetic tones; personal voice quality remains unmeasured.

F01 is the default source conditioning row. Other publisher rows can be selected
with `--speaker F02`, `F03`, `M01` or `M02` and a separate `--output` destination.
They remain source rows rather than adding a new embedding. All rows are retained;
adaptation and export bind one selected row to the target speaker name. A new run
starts at step zero with fresh optimizers and lower default learning rate 5e-5.
The publisher's `iteration=1000` is retained as metadata, not assumed to be an
absolute training-step count. The publisher supplies no discriminators: waveform
and duration discriminators start fresh. Training combines the project's waveform
GAN, duration GAN, mel/KL/stochastic-duration and multi-resolution subband spectral
losses; it is not an exact replay of the publisher's training recipe. Resume saves
both discriminator optimizers, RNG and noise-scaled alignment schedule.

## Shared text contract

`vosk-ru-phones-v1` uses the same model-independent Rust dictionary/manual-stress
frontend as original Vosk inference. Normalization, number expansion, dictionary
probability selection, stressed G2P and BOS/interleaved blank/EOS layout are shared.
Manual `+` before the vowel and combining acute after the vowel are equivalent.
Unknown words require explicit stress. Dates, abbreviations and unsupported scripts
must be spelled out. Automatic contextual stress prediction remains unfinished.

The training schema represents source phone IDs with private-use scalar symbols;
these are internal encodings, not text users should type. Row zero is NUL and row
three represents word space. Preparation and inference verify all 62 indices and
the pinned dictionary. Export packages include the dictionary and its checksum.
Native Rust/C/C++ inference remains offline and in process, with no Python, server,
eSpeak or Vosk Python runtime. The existing VITS backend ID and ABI are unchanged.

## Vocoder conversion

The publisher's ONNX conversion omits inverse-STFT overlap normalization. Its
output has approximately 1.5 times the training waveform gain in the interior,
with overlap-dependent edge differences. Our export uses real-valued convolution,
trigonometry and an explicit squared-window overlap envelope to reproduce
`torch.istft`, including its boundaries. This correction affects the exported
waveform gain; raw waveforms should not be called identical to the published
ONNX package. Regression checks compare our native output with the original
complex training inverse at the existing maximum absolute error 0.001, and
separately compare the publisher's unnormalized conversion against its ONNX graph.
Dynamic relative-position indexing also keeps short/long inputs valid in ONNX.

## Source and distribution records

The [publisher's checkpoint](https://huggingface.co/alphacep/vosk-tts-ru-multi)
is pinned to `0960eb904b9035a9fdbba2882933a4f5a84c525f`. `G_1000.pth` is
167,905,738 bytes, SHA-256
`8c525a91f0c8b0ff7dbbb40764724a68d6a3627cad9d04339cc66490f4b054e2`.
`assets/vits-vosk-ru-assets.json` pins acquisition records. Import uses
`weights_only=True`, hashes the same open checkpoint handle before and after
loading, strictly matches the architecture and installs a new output exclusively.
Existing outputs are not overwritten. Downloads are bounded by exact size/hash.

Training architecture source is pinned to
[alphacep/vosk-tts](https://github.com/alphacep/vosk-tts/tree/b492223c8747a24499e4c1ae905800e85ab887b4/training/vits2),
with modified files and retained source hashes in
`training/vendor/vosk/UPSTREAM.json`. Source Apache-2.0, upstream architecture/PQMF
MIT and STFT BSD notices are retained in training and exported packages. Optional
SLM/spectral discriminator dependencies are excluded. No deployed dependency is
added for this profile.

The model card declares Apache-2.0, but the complete dictionary/data/consent and
derived voice lineage remain unresolved. Fine-tuning does not remove inherited
terms. Export retains this gap alongside the target dataset's separate rights
records; this candidate is not blanket commercial clearance.

## Repeatable checks

```sh
python3 scripts/verify.py --russian-base
```

The opt-in suite requires installed source weights/base and Russian inference
assets. It checks strict import/tensor equality, alternate source conditioning,
full-size adaptation and resume, fresh duration optimizers, unchanged base,
native/PyTorch waveform parity, shared text/prepared-token PCM agreement,
paragraph chunking, original publisher conversion, C++ inference, preflight,
held-out comparison and rejection of changed conditioning/dictionary. Evidence
is retained in `target/public-ru-base-check/result.json`. F01 is exercised through
native export; M02 import/tensor preservation is checked separately. Native parity
for all alternate rows and real-data listening remain separate work.

The 2026-10-03 run passed full-size step 1 and resume to step 2, with both fresh
duration optimizer steps retained. Normalized inverse conversion differed from
the complex training inverse by at most 0.00000033527613 on six held-out/token
cases. The native base check including original training-inverse references passed
at 0.00017069466; adapted native export passed at 0.000041587162. The separate
publisher conversion check passed six cases at 0.00013152137. These are waveform
engineering checks, not listening scores.

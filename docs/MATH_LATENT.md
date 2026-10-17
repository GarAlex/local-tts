# Numeric-only neural arithmetic voice

Latest candidate: `checkpoints/math-latent-hundreds-en-v4`, **9,959,101 bytes**
unpacked including notices and native fixtures. Uses 43 words, 72 tens/units
readings and nine hundreds readings. It remains unaccepted for naturalness and
reliable arithmetic speech. The accepted larger voice remains the initial default.


The user accepted the pretrained stored-int8 sample's voice quality but requested
a numbers-only package below **10,000,000 bytes**. Removing the general-text
networks gives an experimental package of **9,438,483 bytes** including numerical
fixtures and source notices. Its decoder graph is **8,320,688 bytes** and its
cached conditioning is **268,608 bytes**. These are decimal MB, not MiB.

This is **cached acoustic conditioning plus a shared neural waveform decoder**,
not a completed end-to-end distilled text-to-speech model. The decoder remains
the pretrained architecture with packed block-64 four-bit weights. Forty-three
arithmetic words have int8 acoustic conditioning extracted from the trained
text/flow networks offline. They are extracted as medial words in `one WORD one`
phrases to avoid concatenating sentence-final intonation for every word. The
waveform generator runs once on the joined conditioning sequence. No recorded
word PCM, general-text network, Python, or teacher model is needed at runtime.

The smaller vocabulary saves text/flow networks, not just dictionary bytes.
The general full-model four-bit experiment was rejected: its 17.5 MB graph
still missed the requested size and changed durations severely.

The shared native arithmetic parser supports integers through 999,999,999,
decimal digits, signs, addition, subtraction, multiplication, division,
equality and parentheses. It reads the expression literally without evaluating
it. This decoder cannot speak arbitrary prose. The public `MathStudent` adapter
automatically loads either a VITS package or this numeric-only format, so CLI,
Studio and C ABI backend 9 use the same implementation. The lower-level
`math_latent::MathLatent` interface accepts only the checked word vocabulary.

```sh
./target/release/tts-speak --backend math-neural \
  --model checkpoints/math-latent-context-en-v1 \
  --text '12 + 225 = 237' --output sample.wav
```

Rebuild from the trusted local pretrained checkpoint and source voice:

```sh
cargo run --release --locked -p local-tts-core --example math_latent_tokens -- \
  voices/vits-ljs-base/voice target/new-numeric-tokens.json
.venv/bin/python training/math_latent_export.py \
  --checkpoint voices/vits-ljs-base/base.pt \
  --source-voice voices/vits-ljs-base/voice \
  --tokens target/new-numeric-tokens.json \
  --output checkpoints/new-numeric-voice --contextual
cargo run --release --locked -p local-tts-core --example math_latent_verify -- \
  checkpoints/new-numeric-voice
```

Export verifies the source package hashes, checkpoint lineage and original voice
fixtures. Runtime checks checksums, exact vocabulary, finite positive scales,
conditioning bounds, complete nonoverlapping bank coverage, duration scale,
cancellation and output values. Requests are limited to 90 seconds of output;
split larger expressions. Linear interpolation supports duration scales 0.5–2.

Native decoder parity against the compressed ONNX reference passes (zero maximum
sample difference on the three fixture words). The Rust-generated main equation
is correctly transcribed by unprompted local ASR. These checks do **not** establish
naturalness or reliable number accuracy. Word joins, prosody and four-bit acoustic
distortion remain risks; the isolated-word predecessor has numerous longer-phrase
ASR errors. User listening acceptance of the 29.6 MB model does not transfer to
this new design. Keep the accepted larger voice as the quality baseline.

Studio's arithmetic neural voice offers **Compact numbers-only** and **Quality
baseline** buttons. It retains the accepted larger voice as the initial choice
until the compact candidate passes listening. No extra Studio process was opened.

Comparison of 16 long arithmetic validation examples:
`target/math-listening-under10mb.html`. Raw contextual ASR results:
`target/math-latent-context-validation-asr.json`. The public pretrained corpus
was not exhaustively audited for overlap, so these are development examples,
not a certified held-out benchmark.

Native tests cover parity, the main expression, cancellation, invalid text/speed,
the under-10-MB package limit, altered data and invalid resealed offsets/geometry.
Packed-weight tests cover dynamic convolution lengths, zero channels and rounding
error bounds. Strict workspace Clippy and the Studio build are required checks.
Source model, code, dictionary and dataset notices remain in the package; the
existing source lineage/licensing qualifications are preserved.

All 16 contextual development clips have now been checked. Raw ASR has correct
readings for many phrases, but still reports failures including 0.009 as 0.09,
274 as 174 and severe repetition for a million-scale number. These may include
recognizer errors; they are not grounds to accept the package as reliable.

## Compound-number conditioning and binary assets

Schema 2 stores int8 conditioning plus per-channel little-endian float32 scales
in `conditioning-scales.bin`, and native reference waveforms as float32 in
`latent-reference.bin`. Schema-1 JSON scales/references remain supported.
The runtime validates scale offsets, complete nonoverlapping coverage, positive
finite values and the allowed vocabulary. It greedily selects complete tens/units
and hundreds readings; fractional digits remain individual digits. Repeated
identical fractional digits receive a 45 ms gap with short boundary fades, scaled
with requested duration. This does not replace or evaluate digits.

Numeric contexts are valid arithmetic phrases rather than generic one/WORD/one:
for example `four hundred plus one`, `seventy four plus one`, and
`one divided by one`. Cached core data can be reused only from a checksummed
package with matching teacher checkpoint and grammatical-context metadata.
The manifest records the reused bank SHA and the native token-source SHA.

Reproduce the current version from the trusted teacher, regenerating all data:

```sh
cargo run --release --locked -p local-tts-core --example math_latent_tokens -- \
  voices/vits-ljs-base/voice target/new-hundreds-tokens.json --hundreds
.venv/bin/python training/math_latent_export.py \
  --checkpoint voices/vits-ljs-base/base.pt \
  --source-voice voices/vits-ljs-base/voice \
  --tokens target/new-hundreds-tokens.json \
  --output checkpoints/new-hundreds-voice \
  --numeric-contexts --compounds --hundreds
```

The actual build reused the validated 115-entry v3 package with `--reuse-core
checkpoints/math-latent-compound-en-v3`, adding nine hundreds entries. Runtime
size is 8,320,688 bytes of decoder, 1,247,232 bytes of conditioning and 95,232
bytes of scales; the full package is 9,959,101 bytes. These are on-disk sizes,
not runtime RAM estimates; ONNX Runtime may expand stored weights.

Tests exercise schema-1 compatibility, schema-2 parity, the strict under-10-MB
package limit, compound selection, preserved fractional digit sequences, repeated
digit timing, corrupt conditioning and resealed invalid/nonfinite scales.
Five native tests run against the actual v4 package; strict workspace Clippy,
context/packing/listening Python tests and the Studio build pass.

## Development evidence and limits

Paired float/packed decoder probes (`target/math-latent-float-diagnostic`) show
that several initial number-reading failures also occurred with the uncompressed
decoder on identical cached conditioning. This motivated context and compound
changes; quantization is not the only quality concern.

All 16 validation expressions were regenerated through Rust and checked with
unprompted local `tiny.en` ASR. Selected difficult cases were also checked with
`base.en`, against the accepted larger model. Both recognizers now recover the
274/74 example; `base.en` recovers 418 and 0.009. The recognizers disagree on
some decimals. Both compact and larger-model ASR omit part of the million-scale
number, so that is not a validated pronunciation-error diagnosis. The repeated
670.70 equation is still a transcription failure with the current compact voice,
including a regression relative to earlier candidates. Do not call this reliable
arithmetic TTS or claim naturalness acceptance based on these results.

Raw reports: `target/math-latent-hundreds-validation-asr.json` and
`target/math-latent-hundreds-base-asr.json`. Listening against the accepted
29.6 MB voice: `target/math-listening-hundreds-under10mb.html`. Studio's compact
button selects v4, while its initial model remains the accepted larger baseline.
No additional Studio process was launched.

The checked download archive is **8,439,423 bytes**. A local data-package archive is saved as `target/math-numeric-en-v4.zip` with
`target/math-numeric-en-v4.zip.sha256`. It contains the complete checked model
folder, including source notices and fixtures; it excludes training/ASR models
and the TTS executable/runtime. Archive CRC and each extracted file SHA were
verified against the package checksum manifest.

## Russian and Ukrainian numeric candidates (2026-10-09)

The same public `math_speech::MathStudent` API now loads English, Russian and
Ukrainian latent packages. `words_for_language(expression, Language::Ru/Uk)` reads
literal arithmetic, preserving decimal digits and incorrect equations verbatim.
It shares the existing stressed Cyrillic cardinal implementation, including
feminine thousands and plural thousand/million forms. Unsupported languages,
text, exponent notation, incomplete expressions and integers above 999999999
are rejected. Decimal separators remain dots in this arithmetic interface.

| Candidate | Stored decoder weights | Complete folder | ZIP |
|---|---|---:|---:|
| `checkpoints/math-latent-ru-q4-v1` | 4-bit | 8,837,903 bytes | 7,373,673 bytes |
| `checkpoints/math-latent-uk-q8-v1` | 8-bit | 2,800,833 bytes | 2,400,910 bytes |

Each includes 55 stressed lexical entries, binary conditioning/scales, native
reference fixtures, SHA checksums and original notices/provenance. Runtime uses
float operations after unpacking stored weights. The source speaker is fixed:
Vosk F01 for Russian and Piper Lada for Ukrainian. The caller uses `default` and
must pass the matching language. No training/Python is required for playback.
C ABI backend 9 shares this library adapter. Studio selects both through its
arithmetic neural voice panel and previews the chosen package's language.
The Studio executable was rebuilt; no additional app instance was launched.

These first Cyrillic banks use isolated stressed words, rather than the English
candidate's contextual compounds. They are experimental: finite audio/native
fixture parity establishes execution correctness, not naturalness or accurate
listening. Russian 4-bit float-decoder differences reach 0.216 on three fixture
words, versus 0.0172 for Ukrainian 8-bit. The larger Russian 8-bit comparison
(`checkpoints/math-latent-ru-v1`, 14,770,840 bytes) is retained to separate
compression damage from acoustic joins. Contextual banks and listening review
remain next work; no Cyrillic ASR quality result is claimed.

Listening: `target/math-cyrillic-listening.html`; CLI WAVs:
`target/math-numeric-ru-12-plus-225.wav` and
`target/math-numeric-uk-12-plus-225.wav`. Verified archives and sidecar SHA files
are `target/math-numeric-ru-v1.zip` and `target/math-numeric-uk-v1.zip`.

Reproduce without overwriting existing folders:

```sh
cargo run --locked -p local-tts-core --example math_cyrillic_tokens -- \
  ru voices/vosk-ru/dictionary target/math-ru-new-tokens.json
cargo run --locked -p local-tts-core --example math_cyrillic_tokens -- \
  uk unused target/math-uk-new-tokens.json
.venv/bin/python training/math_cyrillic_export.py \
  --checkpoint voices/vits-vosk-ru-base/base.pt --tokens target/math-ru-new-tokens.json \
  --output checkpoints/math-ru-new --bits 4
.venv/bin/python training/math_cyrillic_export.py \
  --checkpoint voices/vits-piper-uk-base/base.pt --tokens target/math-uk-new-tokens.json \
  --output checkpoints/math-uk-new --bits 8
cargo run --locked -p local-tts-speak -- --backend math-neural \
  --model checkpoints/math-latent-uk-q8-v1 --language uk \
  --text '12 + 225 = 237' --output numeric-uk.wav
cargo test --locked -p local-tts-core --test math_cyrillic -- --include-ignored
```

License records are preserved, not replaced by compression. Russian publisher
Apache-2.0 declarations retain unresolved recording/dictionary/consent lineage;
Ukrainian MIT source/repository and Apache-2.0 dataset records retain the earlier
CC0 attribution and initialization/consent qualifications. These packages are
not represented as having newly verified blanket commercial voice rights.
The exporter also works with matching, legitimately fine-tuned checkpoints;
retain the resulting checkpoint's own rights records.

## European decoder components (2026-10-10)

French, German, Spanish, Italian, Portuguese, Polish and Dutch now have actual
8-bit decoder component exports under `target/math-european-decoders-q8-v1`.
Each graph is 1,770,045 bytes. These are development components, **not loadable
arithmetic voice packages**: numeric language rules, acoustic inventories and
listening validation are still missing. The earlier content-filter restriction
prevented continuation of the number-rule implementation.

Reproduce into a new folder:

```sh
.venv/bin/python training/math_decoder_preflight.py \
  --voices voices --output target/math-european-decoders-new
cargo run --locked -p local-tts-core --example math_decoder_probe -- \
  target/math-european-decoders-new
```

The Python report records compression errors on random latent probes at lengths
17, 32 and 73; they do not establish intelligibility. Rust independently checks
SHA integrity, output length and finite samples at those lengths. Original
rights qualifications and notices are retained with each decoder component.

## Cyrillic contextual candidates (2026-10-11)

New opt-in packages cache stressed RU/UK words extracted from valid arithmetic
phrases, replacing isolated-word prosody while preserving the same decoder and
source checkpoint. Both remain experimental, with no listening-quality claim.

- Russian: `checkpoints/math-latent-ru-context-q4-v2`, 8,506,125 bytes.
- Ukrainian: `checkpoints/math-latent-uk-context-q8-v2`, 2,540,863 bytes.
- Comparison: `target/math-cyrillic-context-listening-v2/index.html` places full
  source, previous isolated candidate and new contextual candidate side by side
  for four expressions per language. Reports retain source/model/conditioning
  hashes. Eight new WAVs were generated via the Rust CLI; both native fixture
  verifications have maximum error zero.
- Verified packages: `target/math-numeric-ru-context-v2.zip` and
  `target/math-numeric-uk-context-v2.zip`, with `.sha256` sidecars. Source notices
  and rights qualifications remain included. These are cached-conditioning
  models, not end-to-end distilled students.
- Studio adds separate contextual candidate buttons and retains previous
  candidates. The rebuilt app was not launched; no extra Dock instance created.

Token generation now accepts `--contextual`, selecting carrier phrases through
our existing literal arithmetic parser. Each stored token span must exactly
match the isolated native word's body tokens. The exporter validates spans,
uses attention durations to extract their acoustic frames, then crops activity
and quantizes conditioning as before. Context spans currently support RU/UK only.

Reproduce with new paths:

```sh
cargo run --locked -p local-tts-core --example math_cyrillic_tokens -- \
  uk unused target/math-uk-context-new.json --contextual
.venv/bin/python training/math_cyrillic_export.py \
  --checkpoint voices/vits-piper-uk-base/base.pt \
  --tokens target/math-uk-context-new.json --output checkpoints/math-uk-context-new \
  --bits 8 --contextual
```

Use `ru voices/vosk-ru/dictionary` for Russian token preparation, the Russian
checkpoint and `--bits 4` for its export. Existing paths are never overwritten.
The main equation duration changed from 5.62 to 3.13 seconds for Russian and from
5.46 to 3.73 seconds for Ukrainian. Shorter speech alone does not prove improvement:
listeners must check every number, operation, repeated digit and word boundary.

## Local Cyrillic ASR diagnostics (2026-10-11)

The optional ASR tool now accepts `--language en|ru|uk` and multilingual `tiny`
or `base`. English-only `.en` checkpoints are rejected for Cyrillic before any
model load/download. Four CLI guard combinations are tested. Existing English
invocations retain their defaults. The multilingual Whisper base evaluation
checkpoint is cached only under `target/asr-models`; it is not shipped with TTS
packages and no remote inference API is used.

Unprompted CPU recognition ran on all 24 clips (full source, isolated candidate,
contextual candidate; four cases; two languages). Raw reports are
`target/math-context-ru-base-asr-v1.json` and
`target/math-context-uk-base-asr-v1.json`, with model/audio hashes, package version,
explicit language and decoder diagnostics. Transcripts were added to the
contextual comparison page without changing the voice packages.

Results do **not** validate the new contextual candidates. Russian full source
and isolated candidate retain the main equation and the multiplication/division
reading in ASR, whereas the contextual candidate drops the equality term and
loses much of the operator example. Its leading-zero example also produces
repetitive recognition. Ukrainian recognition is unreliable even on full-source
controls, so its transcripts cannot establish precise numeric accuracy or rank
voice quality. Both contextual candidates remain diagnostic; Studio displays a
recognition-regression note when selected. Previous candidates remain available.

The source models, original data, decoder compression and cached joins still need
separate investigation; these transcripts alone do not identify the cause.
Neither shorter duration nor native numerical parity is evidence of intelligible
speech. Human review must check every number/operator before acceptance.

## Decoder precision and independent-entry diagnostics (2026-10-12)

The Russian contextual bank was exported with an 8-bit decoder while keeping its
conditioning, scales, entry frame counts, source checkpoint and speaker exactly
identical to the 4-bit contextual package. Both bank files compare byte-for-byte.
The 14,439,059-byte package `math-latent-ru-context-q8-diagnostic-v2` exceeds the
size target and is diagnostic only, with no Studio selector. Native fixture
verification has maximum error zero. Four CLI renders plus unprompted local
multilingual-base recognition preserve the contextual equality/operator failures.
Thus increasing decoder precision alone does not fix this regression; it does
not prove compression has no audible effect. See
`target/math-ru-decoder-ablation-v1/index.html` and its hashed report.

An opt-in manifest boolean `decode_per_entry` now requests independent decoding
of each selected cached entry, followed by 3 ms waveform edge fades and the
existing repeated-digit gaps. Missing/false retains the original one-pass joined
latent decoder. A single entry retains its native reference waveform unchanged.
This isolates cross-entry neural convolution effects without new model weights
or a remote service. Word interpolation, cancellation checks, finite output and
length validation, and the 90-second limit apply. CLI, Studio and FFI share the
same `MathStudent`/`MathLatent` implementation.

Diagnostic copies of the original isolated-word banks:

- `checkpoints/math-latent-ru-per-entry-q4-v3`: 8,837,962 bytes.
- `checkpoints/math-latent-uk-per-entry-q8-v3`: 2,800,892 bytes.

The source weights, conditioning and notices are unchanged; only decode mode
and experimental status metadata differ. The exporter supports the new
`--decode-per-entry` flag for reproducible fresh exports. Native tests check
exact equality with independently decoded/faded words at three duration scales,
repeat-zero gap length, finite samples, cancellation and reference parity in
both languages. Existing English and Cyrillic package regression tests pass.

Eight CLI samples and local ASR comparisons are available at
`target/math-per-entry-listening-v3/index.html`. Russian main equation/operators
remain recognized, but decimal recognition problems persist. Ukrainian results
are not improved convincingly and its full-model recognition controls are
unreliable. Independent decoding is therefore a diagnostic option, not an
accepted quality improvement. Studio exposes separate diagnostic selectors;
previous candidates/defaults remain available. No extra Studio process launched.

## Double-zero phrase diagnostic (2026-10-12)

Schema 3 adds exactly one allowed cached phrase: the language's zero word twice.
The entry is mandatory for schema 3; other arbitrary phrases remain rejected.
Binary bank/scale validation matches schema 2. Greedy two-word selection already
handles the phrase, and literal fractional digits are preserved. Longer zero
runs select successive pairs plus an individual zero when needed. Schemas 1/2
keep their existing behavior and vocabulary contracts.

`math_cyrillic_tokens --double-zero` generates the isolated core plus a native
two-zero span from the valid carrier `0.00 + 1`. Export uses `--double-zero
--decode-per-entry`, with only that phrase extracted from the carrier. All
original single-word conditioning/scales were verified byte-identical prefixes
of the new banks; no original words or decoder weights were replaced.

Diagnostic packages (56 entries each):

- Russian `checkpoints/math-latent-ru-double-zero-q4-v4`: 8,844,971 bytes.
- Ukrainian `checkpoints/math-latent-uk-double-zero-q8-v4`: 2,807,709 bytes.

The Russian `0.009 + 1 = 1.009` probe is recognized correctly by the same local
unprompted multilingual-base ASR that failed earlier candidates. This is a
narrow diagnostic improvement, **not** broad accuracy acceptance. Additional
`0.001 + 0.002 = 0.003` and `0.0001 + 0.0002 = 0.0003` probes still lose zeroes or
operators in recognition. Ukrainian recognition regresses into repetition;
that package remains diagnostic and is not promoted. The unaffected Russian
`12 + 225 = 237` WAV is byte-identical to the separate-entry v3 WAV.

Native tests cover schema-3 parity, package size, preservation of literal zero
counts, actual output geometry for one/two zero-pair entries, finite output,
independent decoding at three speeds and cancellation. Prior English/Cyrillic
schema-2 regression tests pass. Studio retains previous candidates and adds
explicitly labeled diagnostic selectors. No new app instance was launched.

Listen and inspect transcripts: `target/math-double-zero-listening-v4.html`.
The Russian diagnostic ZIP is `target/math-numeric-ru-double-zero-v4.zip`
(7,380,877 bytes), with SHA sidecar and verified archive/package checksums.
Source notices and existing rights qualifications remain included.

Reproduce using fresh output names:

```sh
cargo run --locked -p local-tts-core --example math_cyrillic_tokens -- \
  ru voices/vosk-ru/dictionary target/math-ru-double-zero-new.json --double-zero
.venv/bin/python training/math_cyrillic_export.py \
  --checkpoint voices/vits-vosk-ru-base/base.pt --tokens target/math-ru-double-zero-new.json \
  --output checkpoints/math-ru-double-zero-new --bits 4 --double-zero --decode-per-entry
```

Do not combine `--double-zero` with `--contextual`: the experiment deliberately
retains the isolated single-word core. Ukrainian uses its native frontend and
checkpoint with `--bits 8`; its current diagnostic results do not justify use.

## Full-source decimal controls (2026-10-13)

`math_numeric_compare` now optionally accepts `--expressions JSON`, an array of
1–32 unique literal arithmetic strings. Input size, syntax and duplicates are
validated before model loading/audio directory creation. The report records the
probe-file SHA. Existing four-case behavior remains available without this flag.
Five invalid-input cases verify that rejection creates no audio/output directory.
Clippy passed for the example.

Generated four decimal equation pairs per language using the full pretrained
source and the same schema-3 compact candidate. All 16 mono WAVs generated;
identical source-checkpoint lineage, intended wording and audio SHA values are
recorded. Local unprompted multilingual-base recognition ran on all controls.
Listening page: `target/math-decimal-controls-v1.html`. Probe file:
`target/math-decimal-controls-v1.json`; reports and WAVs are under
`target/math-ru-decimal-controls-v1` and `target/math-uk-decimal-controls-v1`.

Interpretation remains limited: full-source Russian recognition also uses
ambiguous `0.0.1` renderings on some probes. On the longer
`0.0001 + 0.0002 = 0.0003` probe, full-source recognition preserves the readings,
while the compact transcript omits a zero from the first operand. Both recognize
`0.1 + 0.2 = 0.3` correctly. Ukrainian full-source recognition remains inconsistent,
so its ASR is not a reliable accuracy judge. These controls do not establish
naturalness, identify a definitive mechanism, or justify further cache additions
based solely on every recognition error. No candidate/default changed this turn.

```sh
cargo run --locked -p local-tts-core --example math_numeric_compare -- \
  ru target/public-ru-base-check/base-voice checkpoints/math-latent-ru-double-zero-q4-v4 \
  target/new-decimal-comparison --expressions target/math-decimal-controls-v1.json
```

## User listening verdict (2026-10-13)

The user rejected the latest Russian double-zero sample as “Still robotic or
unclear.” Treat v4 as failed listening acceptance, not as a successful voice
based on its narrow ASR improvement. The Studio selector and comparison page now
state the rejection. Artifacts remain unchanged for reproducible comparisons.

Further phrase-cache patches are not the main quality route. The new
`training/math_sequence_student.py` prototype explores full-utterance duration
and acoustic-latent prediction into a pretrained waveform decoder. It is
**untrained**, has no free-running/native export yet, and has not generated a
playable voice. Tests validate data-flow/padding/gradient contracts, not speech
quality. See PHASES.md for the required training and listening milestones.

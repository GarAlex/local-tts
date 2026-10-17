# Pretrained Russian and Ukrainian voices

Studio's Test speech tab now selects English/Kokoro, Russian/Vosk,
Ukrainian/Piper or an exported own VITS voice. Russian and Ukrainian packages
are installed in this workspace. Select a speaker, enter text, Generate,
then Play or Save WAV. All inference runs in the Rust library using ONNX Runtime;
Python and network access are needed only for acquisition/training.

| Backend | Package | Speakers | Frontend |
|---|---|---|---|
| Russian | Vosk 0.7 multi | F01, F02, F03, M01, M02 | Dictionary pronunciation, saved corrections and explicit stress overrides |
| Ukrainian | Piper ukrainian_tts medium | lada, mykyta, tetiana | Lowercase decomposed Unicode text with saved/explicit stress |

```sh
python3 scripts/download_cyrillic.py all
cargo run -p local-tts-cli -- vosk-synthesize voices/vosk-ru russian.wav F03 'Привет, мир!'
cargo run -p local-tts-cli -- piper-synthesize voices/piper-uk ukrainian.wav lada 'Прив+іт, св+іте!'
python3 scripts/verify.py --cyrillic
```

Use `ru` or `uk` to download just one package. Asset manifests pin SHA-256
checksums; Ukrainian URLs pin a Hugging Face commit. The Russian publisher
archive has a versioned URL and pinned archive/member hashes. Downloaders preserve
existing files when a fetched checksum differs. Loaders verify model/config hashes
and the Russian dictionary against checksums compiled into the library.

Russian defaults to F03, an artificial female voice according to the publisher's
model card. F01 is attributed to Tiflocomp Irina and F02 to Sova Natasha;
M01 and M02 are artificial voices. We use the 0.7 ONNX package corresponding to
the Apache-2.0 Hugging Face checkpoint, rather than the much larger 0.9 package.

Both frontends accept `+` immediately before a stressed vowel or U+0301 acute
after it: `з+амок` and `за́мок` are equivalent, while `зам+ок` changes stress.
Russian overrides bypass dictionary lookup. Unknown Russian words require an
explicit stress annotation; unsupported characters fail with an error. Ukrainian
accepts unaccented text but has no automatic stress predictor, so annotate stress
for more predictable pronunciation. Curly Ukrainian apostrophes normalize to `'`.
Plain unsigned integers below one billion now expand to language-specific cardinal
words, with feminine thousands and plural agreement. Leading zeroes read digit by
digit. Decimal fractions read each digit, preserving written precision: `3,05`
becomes Ukrainian `три ко́ма нуль п'ять` or Russian `три запята́я ноль пять`;
`3.05` uses `кра́пка`/`то́чка`. Comma and dot followed immediately by a digit are
always decimal separators, never thousands separators. Fractional parts are
limited to nine digits. Space-grouped numbers such as `1 000` are rejected as
ambiguous. Write `1000` or spell out the intended reading.

Generated number words include stress internally; users do not need to annotate
digits. The current `text-v1` layer additionally handles ISO/DD.MM.YYYY dates,
24-hour times, signed quantities, units, percentages, prices, small ordinals,
fractions, integer en-dash ranges and whitelisted abbreviations. Measurements use
gender/number agreement; fractional units and ranges use genitive forms. Date
readings use a fixed genitive phrase. Arbitrary surrounding case and word stress
are not predicted. Limited phrase cues distinguish замок/атлас (and Russian мука),
after explicit/saved corrections. See [supported syntax and limits](LANGUAGE.md).
Latin text is not transliterated, except the documented abbreviation/unit whitelist.

Russian tokenization follows the original Vosk 0.7 blank insertion and phoneme
rules. Ukrainian uses Piper's text-mode lowercase/NFD and BOS/blank/EOS layout;
no eSpeak or Piper implementation is linked. Large requests split at sentence/word
boundaries (180 normalized characters per Russian chunk, 240 per Ukrainian
chunk), with 120 ms pauses between chunks. Whole input is validated before model
execution. Input is limited to 10,000 characters and output to ten minutes.
Cancellation is cooperative at model invocation boundaries. Inference noise makes
successive outputs differ; waveform equality is not a quality test.

## Embedding

Rust: load `vosk::Vosk` or `piper::Piper`, reuse the engine and call `Synthesizer`
with `Language::Ru`/`Language::Uk` and an exact speaker name. Both engines expose
`synthesize_cancellable`; `tokenize` exposes frontend output for diagnostics. Public `piper::prepare_text`
and `vosk::prepare_text` return the normalized, expanded text, with canonical
acute stress marks, without saved corrections. The engine methods of the same name
also apply the loaded corrections; Studio previews and logs this corrected text.
`Piper::speakers()` and `vosk::SPEAKERS` list supported speakers.

C/C++: ABI v1 adds `TTS_VOSK_RU=3` and `TTS_PIPER_UK=2`, preserving existing
backend IDs. Strings are UTF-8. The same audio ownership, errors and cancellation
rules apply. For the C++ consumer built by `scripts/test_cpp.py --cyrillic`:

```sh
target/cpp/speak voices/vosk-ru russian.wav 1 F03 'Привет, мир!' vosk-ru
target/cpp/speak voices/piper-uk ukrainian.wav 2 lada 'Прив+іт, св+іте!' piper-uk
```

## Saved word stress

In Studio, select a pretrained Russian/Ukrainian voice, open **Saved word stress**,
enter one word with its stressed vowel (`Прив+іт` or `Приві́т`), and choose
**Save word**. Then ordinary `Привіт!` receives the saved `приві́т!` reading.
**Remove** deletes that entry. Saves apply immediately to this voice folder and
survive app restarts. All speakers in the folder share the corrections.

Each inflected form needs its own entry. Lookup matches complete words after
case/Unicode/apostrophe normalization and preserves letters. It does not predict
stress for unknown words or choose a homograph's meaning from context. Explicit
stress in a sentence wins: a saved `за́мок` does not replace typed `зам+ок`.
Russian saved corrections also allow unknown words to use an explicit-stress
pronunciation. Ukrainian uncorrected words retain the model's default reading.

The optional `pronunciations.json` lives beside the model:

```json
{
  "schema_version": 1,
  "language": "uk",
  "entries": {
    "привіт": "приві́т",
    "світе": "сві́те"
  }
}
```

Keys must be canonical lowercase unaccented spellings; values must preserve their
letters and specify exactly one stressed vowel. Values also accept `+` before a
vowel when loading and are saved with acute accents. Russian uses `"ru"`. Files
are limited to one MiB/10,000 entries; each accented word to 96 characters.
Malformed, wrong-language and nonregular/symlink files fail explicitly. An absent
file means no corrections. Corrections are editable user data, excluded from the
pinned model/config/dictionary checksums.

```sh
cargo run -p local-tts-cli -- pronunciation-add voices/piper-uk uk 'Прив+іт'
cargo run -p local-tts-cli -- pronunciation-list voices/piper-uk uk
cargo run -p local-tts-cli -- piper-synthesize voices/piper-uk ukrainian.wav lada 'Привіт!'
cargo run -p local-tts-cli -- pronunciation-remove voices/piper-uk uk 'привіт'
```

CLI/Studio saves validate and atomically replace the file, rejecting observed
external changes since loading. This is optimistic edit protection, not a lock
against simultaneous external writers. Use **Reload corrections** after editing
outside Studio; previews use the loaded snapshot. Generate automatically reloads
corrections, including for a cached model. An invalid file blocks Studio generation
until fixed, rather than silently ignoring corrections.

Rust callers can use `pronunciations::{Pronunciations, PronunciationFile}` to
validate, edit, persist and apply corrections without loading a neural model.
`Vosk::load`/`Piper::load` read the optional file once. Their
`reload_pronunciations()` methods refresh it without reloading weights; a failed
reload retains the previous engine corrections. Removing the file and reloading
clears them. `set_pronunciations(data)` replaces only in-memory corrections and
rejects a language mismatch. C/C++ exposes `tts_engine_reload_pronunciations` for
the same reload behavior, serialized with synthesis on that engine.

This feature uses original project code and user-authored entries, adds no external
stress dictionary, and requires no retraining. Own VITS training/inference retains
its existing versioned token contract; these saved corrections apply only to the
two pretrained Cyrillic backends.

## License records and remaining evaluation

The Vosk Hugging Face card declares Apache-2.0 and identifies the checkpoint as
0.7 multi. Its card and the source Apache license are retained with the package;
`assets/VOSK-LICENSE.txt` covers the adapted frontend rules. The package does not
supply a separate dictionary/data license or a detailed rights chain for every
training voice; this is recorded rather than inferred from a code license.

The Piper repository declares MIT. Its Ukrainian voice card claims CC0 and links
OHF-Voice/voice-datasets, which currently contains no Ukrainian dataset. The
actual Ukrainian dataset project's README identifies lada/mykyta/tetiana and
Piper, and its repository license is Apache-2.0. Both the supplied card and the
actual dataset README/license are retained; the card's CC0 claim is **not treated
as verified**. Standard MIT/Apache notices apply; these records do not establish a
royalty requirement. Full release provenance/notice auditing remains open.

The real-model regression exercises all eight speakers, finite non-silent PCM,
stress equivalence, Ukrainian decomposition, invalid language/speaker/settings,
cancellation and a Ukrainian paragraph. C++ consumers exercise both new backends.
Generated samples and timings are in `target/pretrained-smoke`. These are native
integration checks; limited phrase stress and structured numeric normalization
are implemented. Native-speaker listening, broader contextual disambiguation and
general Ukrainian automatic stress remain unfinished.

Own-voice training primarily fine-tunes a compatible project grapheme VITS checkpoint
into a new voice, with scratch training as an explicit fallback. The existing trainer
does not fine-tune these downloaded ONNX checkpoints; the original Ukrainian Piper training checkpoint can now be imported and
fine-tuned through a separate supported profile (see PUBLIC_UK_BASE.md). Russian
public-base import and adaptation are now implemented through the Vosk architecture
adapter (see PUBLIC_RU_BASE.md). Training
quality is still covered only by engineering fixtures until real recordings are
provided.


## Stress notation and own-model compatibility

`+` is an input convenience for lexical word stress, not a spoken plus sign or an
expressive emphasis instruction. `Прив+іт`, `Приві́т` and unaccented `Привіт` are
all accepted; the first two specify the stressed vowel explicitly. Studio's
Ukrainian starter example now uses the acute accent, and its help explains that
ordinary unaccented text also works. Pronunciation preview converts + to acute
marks without requiring a model load. It does not add stress to arbitrary words.

The numeric expander is original project code with built-in lexical stress; it
adds no external dictionary or runtime dependency. A candidate general-purpose
Ukrainian stress dictionary derives from ULIF data, whose data terms need separate
verification; it has not been downloaded or shipped.

Number expansion applies to pretrained RU/UK inference through Rust and the C
ABI. Own VITS grapheme-v1/v2 preparation/inference is unchanged: automatically
expanding only inference text would break the shared training token contract.
Callers can use `cyrillic_numbers::expand` while authoring their own transcripts,
with matching preprocessing at inference, or wait for a future versioned frontend.

Own models adapted from the original Ukrainian Piper checkpoint use
`piper-uk-text-v1`, which applies the same number/stress normalization in both
native preparation and inference. This does not change older grapheme models.
See [Ukrainian fine-tuning base](PUBLIC_UK_BASE.md).

Russian public-base adaptation now uses `vosk-ru-phones-v1`, sharing the original
Vosk dictionary/manual-stress frontend with native inference and retaining all
source phone indices. The exported own voice uses a corrected normalized iSTFT
vocoder, so its gain differs from the publisher's unnormalized ONNX conversion.
Fine-tuning/resume/export mechanics are tested with engineering tones; real speech
quality and source rights remain open. See [Russian base](PUBLIC_RU_BASE.md).

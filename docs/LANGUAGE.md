# Text normalization and contextual pronunciation

`local_tts_core::text::prepare(text, language)` is a model-free, offline Rust
interface for the current `text-v1` contract. Kokoro, pretrained Vosk and pretrained
Piper use it automatically, with engine-specific phonemes and saved corrections.
Studio's **Pronunciation preview** shows the expanded words for English, Russian
and Ukrainian. English generation also logs the actual phonemes.

These are original project rules and lexical tables. No new external language
dataset, dictionary, model weights or runtime dependency has been added.

## Supported forms

| Form | English | Russian / Ukrainian |
|---|---|---|
| Integers | 0–999999999; leading zeroes spoken digit by digit | Same bounds, native cardinal vocabulary and feminine thousands |
| Bare decimals | `3.05` → three point zero five | Dot/comma; separator and fractional digits read explicitly |
| Grouping | `1,234,567.05` | Nonbreaking U+00A0 or narrow nonbreaking U+202F spaces; dot/comma remain decimal separators |
| Signs | `-5`, `−5`, `+5` → minus/plus five | Native spoken minus/plus; signs must touch digits |
| Dates | `2026-10-03` → October third, two thousand twenty six | ISO or `03.10.2026`; genitive day/month/year phrase |
| Clock times | `12:30` → twelve hours thirty minutes | Native hour/minute counts, with gender and number agreement |
| Ordinals | `21st`, `12th`, `100th`, `0th` | `21-й`, `22-го`, values 1–99 |
| Fractions | `1/2`, `2/3`, `3/4`; denominators 2–20 | Native feminine counts and fractional adjectives, same denominator bounds |
| Ranges | `0.5–2.5 kg` → from zero point five to two point five kilograms | Integer `5–21 кг` → genitive from/to phrase with genitive unit |
| Measurements | `2.5 kg`, `60 km/h`, `−5°C` | Integer gender/number agreement; decimals use whole/fraction words and genitive singular unit |
| Percentages | `5%`, `2.5%` | Native percent noun and quantity agreement |
| Prices | `$5.20`, `€2.50`, `£1.01`, `3 USD` | Native currency and minor-unit forms; also RUB/₽/руб and UAH/₴/грн |
| Labels | `No. 42`, `№42`, `#42` | `№42`, `#42` |

Dates require real calendar days and years **1900–2099**, including Gregorian
leap-year validation. Numeric slash dates are ambiguous and rejected. Date
phrases in RU/UK use genitive: `третьего октября ... года` /
`третього жовтня ... року`. Write a different case explicitly when your sentence
needs it; the normalizer does not parse arbitrary sentence grammar.

Times accept one/two hour digits and two minute digits in a **24-hour** reading,
00:00–23:59. Seconds, AM/PM suffixes and time zones need a spoken spelling.
Signs, ordinal suffixes and range separators must touch their digits.

Measurement keys: `m`, `km`, `cm`, `mm`, `g`, `kg`, `l`/`L`, `ml`/`mL`, `h`, `min`,
`s`, `km/h`, `m/s`, `°C`, `°F`. Cyrillic aliases include м/км/см/мм/г/кг/л/мл,
ч, мин, хв, с, км/ч, км/год and м/с; `год` means hours only in Ukrainian.
Quantities may touch the unit or use whitespace. For example, RU `21 мин` uses
`двадцать одна минута`, and UK `22 хв` uses `двадцять дві хвилини`.

Bare decimals preserve the earlier digit reading. With a measurement, `2,5 кг`
becomes RU `две целых пять десятых килограмма` or UK
`дві цілих п'ять десятих кілограма`. Fractional precision is limited to nine digits;
prices accept at most two fractional digits, with one digit scaled to tenths of a
major currency unit. Zero minor units are omitted. Price symbols/codes identify
one currency, so conflicting markers are rejected.

Ranges use **U+2013 en dash**, without spaces, and an optional unit at the end.
ASCII `5-10` is ambiguous with subtraction and is rejected. Signed, currency and
fractional Cyrillic ranges need a spoken spelling. Fractions with measurements
use `one half of a kilogram` or a genitive Cyrillic unit. Mixed numbers such as
`1 1/2` should be written as words. Unit powers, scientific notation, telephone
numbers, versions, arbitrary alphanumeric identifiers and unsupported units need
explicit spoken text; ordinary words are still passed to the chosen model's
frontend and may fail its dictionary/inventory checks.

ASCII-space grouping (`1 000`) is rejected because it can also be a list. Use
`1000`, English commas or the documented Cyrillic nonbreaking separators.
Normalized output and input are both capped at 10,000 characters. Unsupported
syntax fails before neural inference; it is not silently truncated.

## Abbreviations

The whitelist expands English `Dr.`, `Mr.`, `Mrs.`, `Ms.`, `Prof.`, `e.g.`, `i.e.`,
`etc.`, `vs.` and numeric `No.`. `No.` without following digits remains the word
"no". Ambiguous `St.` is deliberately not interpreted as street or saint.
RU recognizes т.е./т. е., т.д./т. д., т.п./т. п., напр. and проф.; UK recognizes
т.д./т. д., т.п./т. п., напр. and проф. Cyrillic abbreviations also accept capitals.
CPU, GPU and API expand to language-specific words. Whole-token boundaries keep
these rules out of longer words. Arbitrary acronyms are not automatically spelled
or transliterated. Curly quotes and apostrophes normalize to supported punctuation.

## Contextual readings and precedence

The current rules use adjacent lexical cues and do not cross punctuation:

- RU: door/padlock adjectives select `замо́к`, castle adjectives select `за́мок`;
  wheat/rye/rice adjectives select `мука́`, suffering adjectives select `му́ка`;
  geographic/school versus silk adjectives select the two readings of атлас.
- UK: corresponding door/castle adjectives select the two readings of замок;
  geographic/school versus silk adjectives select the two readings of атлас.
- EN: have/has/had read selects the past reading; to/will/can/must/should/please
  read selects the present reading. Immediate "read yesterday", "lead pipe/metal",
  "wind clock/watch", "bass guitar/drum/player" and "bass fish/fishing" have explicit
  readings. A selected cue must be adjacent, not across a sentence boundary.

RU/UK precedence is **explicit input stress → saved word correction → phrase rule
→ existing dictionary/model behavior**. Unknown Russian words still need an
explicit stress spelling. `+` before a vowel is an annotation; `+5` is a spoken
numeric plus sign. Stress on Ukrainian є now works (`+Європа` → `є́вропа`).

This is a bounded rule set, not a general disambiguator or Ukrainian word-stress
predictor. Unresolved English homographs retain their dictionary reading;
phoneme input remains available. Saved Cyrillic entries apply to one word form,
so inflections need their own entries. No retraining is required for the frontend
rules themselves. Native-speaker listening remains necessary to assess quality.

Grammar checks consulted: [Ukrainian Ministry of Education orthography](https://mon.gov.ua/osvita-2/zagalna-serednya-osvita/ukrainskiy-pravopis),
[Gramota on fractional quantities](https://gramota.ru/poisk?mode=spravka&page=3&query=%D0%94%D0%B5%D0%B2%D1%8F%D1%82%D1%8C&simple=0),
[Cambridge's present/past read pronunciations](https://dictionary.cambridge.org/pronunciation/english/read)
and [GOV.UK's UK/US date-format discussion](https://www.gov.uk/government/publications/govuk-content-principles-conventions-and-research-background/govuk-content-principles-conventions-and-research-background).
Ukrainian ordinal stress was also checked against the authorized СУМ entries for
[двадцятий](https://slovnyk.ua/index.php?swrd=двадцятий),
[тридцятий](https://slovnyk.ua/index.php?swrd=тридцятий) and
[сороковий](https://slovnyk.ua/index.php?swrd=сороковий).
These references informed checks; their dictionaries/text were not bundled.

## Own-voice contracts and embedding

New scratch runs and new fine-tuning targets (except legacy grapheme-v1 bases)
record `"normalization": "text-v1"` in prepared configuration/checkpoints and
exported `voice.json`. Training and inference apply this same outer contract
before the existing token frontend. Token inventory/architecture IDs are unchanged.
Unknown normalization versions fail loading/preparation. Exports without the
field retain their original text behavior; existing resume/evaluation preparation
retains its saved version and byte-stable configuration, rather than upgrading.

`training::prepare_from_base` preserves the base contract for resume/evaluation;
`training::prepare_from_base_current` explicitly opts a new target into text-v1.
The Studio launcher selects the latter for a new fine-tune. CLI equivalents:

```sh
# Model-free preview:
cargo run -p local-tts-cli -- text-prepare uk '03.10.2026; 2,5 кг'
# New target using a legacy compatible base's token inventory:
local-tts training-prepare dataset.json prepared.json base-training.json --text-v1
# Resume/evaluate with the checkpoint's exact contract:
local-tts training-prepare dataset.json prepared.json checkpoint-training.json
```

Comparing a legacy base with a text-v1 target expands the text for both engines
and records the prepared spelling and both contracts. The report warns that this
includes frontend differences; English contextual phoneme choices may still
differ, so it does not isolate the effect of adapting weights.

C/C++ has additive ABI-v1 `tts_text_prepare(text, language, &output, &error)`.
It needs no model, produces owned UTF-8 and does not load saved voice corrections.
Free success output with `tts_string_free` and error output with `tts_error_free`.
The main synthesis ABI automatically uses the loaded engine's own text contract.
See `examples/cpp/text_test.cpp` for ownership, errors and concurrent calls.

## Verification

`cargo test --workspace` covers normalization, malformed inputs, calendar validity,
number/unit agreement, phrase boundaries, explicit/saved stress precedence,
idempotence, version handling and byte-stable legacy/current preparation.
`python3 scripts/verify.py --language` additionally uses installed real Kokoro
FP32/Q8/Q4, Vosk and Piper packages, C++ consumers and synthetic numeric dataset
fine-tuning/export checks. The latter compare native raw-text output with PyTorch
output for the exact prepared tokens. Synthetic tones establish engineering
contracts, not pronunciation or usable voice quality.

## Added European languages

The language catalogue also accepts de/fr/es/it/pt/pl/nl. Their initial text path
uses NFC Unicode and lowercase graphemes, preserving accents and ligatures.
Write numeric forms and abbreviations as spoken words; structured expansion and
context rules above remain EN/RU/UK-only. See [French and European guide](FRENCH.md).

## Narration mode (NeMo)

`English::set_fallback(true)` — `tts-speak --guess-unknown`, and PromoShot's
local voices — switches English text normalization from the strict, versioned
text-v1 contract to NVIDIA NeMo's English text-normalization grammars
(r1.2.0, deterministic, cased), compiled once by `training/nemo/build_grammars.sh`
and run natively by `crate::nemo`. Trained VITS voices keep text-v1 unless the
mode is switched on.

The pipeline: accented Latin letters are folded (Kīlauea → Kilauea); a few
forms NeMo reads awkwardly or leaves as written are rewritten into ones it
reads well — web addresses lose `https://` and `www.`, multipliers (2.5x →
2.5 times), a score beside a game word (won 112-108 → 112 to 108), units per
unit (km/h → km per hour), a unit used as an adjective after a/an (a 5 kg
camera → a 5 kilogram camera), US and European time zones after a time
(9 a.m. PT → Pacific Time), Roman numerals after a numbering word (Chapter IV,
World War II) or a regnal name (Louis XIV → the fourteenth) or another
capitalized word up to XXXIX (Rocky II); NeMo normalizes a sentence at a time;
its British "one hundred and eight" loses the "and";
then each word is read from the user lexicon, the dictionary, or the
unknown-word fallback. All-capital tokens NeMo produced (AM, PM) are spelled,
as are short isolated ones the text wrote (EST, CEO, the IT team), unless they
are common words shouted for emphasis (Download NOW) or part of an all-caps
headline; longer ones (NASA) are read from the dictionary. Symbols NeMo leaves
become a pause rather than an error.

Parity: on NeMo's 494 English test cases the native runner's output is
identical to Python NeMo's on 492; the two others are punctuation pile-ups
(`U.S.A..`, `4A.M.?-34`) where equally weighted paths break differently.
Python NeMo matches the cases' expected answers on 490, the native runner on 488.
A typical sentence normalizes in under 20 ms on an M4 Pro; loading the grammars
takes about 10 ms, once per process.

Words are read from the user's lexicon first, then from a small bundled
narration lexicon (`assets/narration-en.dict`) for names and brands CMUdict
lacks or reads with an older pronunciation (Meta, which CMUdict records as
"Mita"; Norte; Canva; Kubernetes), then from CMUdict. Lexicon entries are
used exactly as written, without the American flap or destressing.

Spelled letters reach the voice one word per letter, so "3:19 AM" stays
two letters. Guesses from the unknown-word model carry its confidence; below
−0.08 (`Guess::unsure`) a guess differed from CMUdict on 62–81% of held-out
words, so those are reported first for checking by ear. A rule-based
"foreign-style" reading was measured against the model on the same held-out
words and lost even on the model's least confident vowel-final words
(37% vs 23.5% phone error): CMUdict records how English speakers say foreign
names, and the model has learned that. Names it gets wrong are pinned in the
project's lexicon.


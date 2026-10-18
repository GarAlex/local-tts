//! Dictionary-based English frontend; never silently substitutes a pronunciation.
use std::{collections::HashMap, path::Path};

pub struct English {
    dictionary: HashMap<String, String>,
    /// User lexicon entries; these win over the dictionary and contextual readings.
    overrides: HashMap<String, String>,
    /// Off unless asked for: guesses for words the dictionary lacks.
    fallback: Option<crate::g2p::G2p>,
    /// What the fallback guessed since the last [`English::take_guesses`].
    guesses: std::sync::Mutex<Vec<Guess>>,
    /// The dictionary file, re-read in its ARPAbet form when the fallback is on.
    source: std::path::PathBuf,
    /// ARPAbet by word, filled only while the fallback is on.
    arpabet: HashMap<String, String>,
}

/// A pronunciation the fallback made up for a word the dictionary lacks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Guess {
    /// The word as written (its original capitals).
    pub word: String,
    /// ARPAbet with stress digits — the line to put in a lexicon to keep or fix it.
    pub arpabet: String,
    /// `parts` (split at capitals into known words), `letters` (an acronym),
    /// `compound` (known words run together) or `model` (the trained guesser).
    pub method: &'static str,
}

/// Size cap for a user lexicon file.
const MAX_LEXICON_BYTES: u64 = 1_048_576;

impl English {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut dictionary = HashMap::new();
        for line in data.lines() {
            if line.starts_with(";;;") {
                continue;
            }
            let mut parts = line.split('#').next().unwrap_or("").split_whitespace();
            let Some(word) = parts.next() else { continue };
            if word.contains('(') {
                continue;
            }
            let pronunciation = parts.map(phone).collect::<Result<Vec<_>, _>>()?.join("");
            dictionary
                .entry(word.to_lowercase())
                .or_insert(pronunciation);
        }
        if dictionary.is_empty() {
            return Err("pronunciation dictionary is empty".into());
        }
        Ok(Self {
            dictionary,
            overrides: HashMap::new(),
            fallback: None,
            guesses: Default::default(),
            source: path.to_path_buf(),
            arpabet: HashMap::new(),
        })
    }

    /// Guess words the dictionary lacks instead of refusing them: split at
    /// capitals into known words (PromoShot, YouTube, ChatGPT), spell short
    /// acronyms (API), split run-together known words (screenrecorder), and
    /// otherwise ask a small model trained on this dictionary. Every guess
    /// is recorded; read them with [`English::take_guesses`]. Off by default.
    pub fn set_fallback(&mut self, enabled: bool) -> Result<(), String> {
        if !enabled {
            self.fallback = None;
            self.arpabet.clear();
            return Ok(());
        }
        let data = std::fs::read_to_string(&self.source).map_err(|e| e.to_string())?;
        for line in data.lines().filter(|l| !l.starts_with(";;;")) {
            let mut parts = line.split('#').next().unwrap_or("").split_whitespace();
            let Some(word) = parts.next().filter(|w| !w.contains('(')) else {
                continue;
            };
            let phones: Vec<&str> = parts.collect();
            if !phones.is_empty() {
                self.arpabet
                    .entry(word.to_lowercase())
                    .or_insert_with(|| phones.join(" "));
            }
        }
        self.fallback = Some(crate::g2p::G2p::bundled()?);
        Ok(())
    }

    /// The guesses made since the last call, each word once.
    pub fn take_guesses(&self) -> Vec<Guess> {
        std::mem::take(&mut *self.guesses.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Adds CMU-format user pronunciations (`WORD  P R AA1 M OW0`), one per line.
    /// Lines starting with `;;;` or `#` are comments. Words may contain only ASCII
    /// letters and apostrophes, matching how text is split. Later entries replace
    /// earlier ones. Returns the number of entries read.
    pub fn add_lexicon(&mut self, path: &Path) -> Result<usize, String> {
        let name = path.display();
        let size = std::fs::metadata(path)
            .map_err(|e| format!("lexicon {name}: {e}"))?
            .len();
        if size > MAX_LEXICON_BYTES {
            return Err(format!("lexicon {name} exceeds one MiB"));
        }
        let data = std::fs::read_to_string(path).map_err(|e| format!("lexicon {name}: {e}"))?;
        let mut count = 0;
        for (number, line) in data.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("");
            if line.starts_with(";;;") {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(word) = parts.next() else { continue };
            let at = || format!("lexicon {name} line {}", number + 1);
            if !word.chars().all(|c| c.is_ascii_alphabetic() || c == '\'') {
                return Err(format!(
                    "{}: '{word}' must contain only ASCII letters and apostrophes",
                    at()
                ));
            }
            let phones = parts
                .map(phone)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("{}: {e}", at()))?;
            if phones.is_empty() {
                return Err(format!("{}: '{word}' has no phonemes", at()));
            }
            self.overrides.insert(word.to_lowercase(), phones.join(""));
            count += 1;
        }
        Ok(count)
    }

    /// IPA spelling for the upstream VITS inventory (not Kokoro's vowel shortcuts).
    pub fn phonemize_ipa(&self, text: &str) -> Result<String, String> {
        self.ipa(text, false)
    }
    /// IPA with text-v1 normalization and limited contextual homograph readings.
    pub fn phonemize_ipa_current(&self, text: &str) -> Result<String, String> {
        self.ipa(text, true)
    }
    fn ipa(&self, text: &str, current: bool) -> Result<String, String> {
        Ok(self
            .phonemize_impl(text, current, false)?
            .chars()
            .map(|c| match c {
                'A' => "eɪ".into(),
                'I' => "aɪ".into(),
                'O' => "oʊ".into(),
                'W' => "aʊ".into(),
                'Y' => "ɔɪ".into(),
                _ => c.to_string(),
            })
            .collect())
    }

    pub fn phonemize(&self, text: &str) -> Result<String, String> {
        self.phonemize_impl(text, true, false)
    }
    /// [`Self::phonemize`] with American connected-speech cues for Kokoro's
    /// US voices: function words lose their citation stress and an
    /// intervocalic t/d before an unstressed vowel becomes the flap ɾ, as
    /// in the misaki phonemizer Kokoro was trained with. Without them every
    /// word is stressed and every t is hard, which sounds foreign. User
    /// lexicon entries are used exactly as written.
    pub fn phonemize_american(&self, text: &str) -> Result<String, String> {
        self.phonemize_impl(text, true, true)
    }
    fn phonemize_impl(&self, text: &str, current: bool, american: bool) -> Result<String, String> {
        // Narration mode (the fallback on): NeMo's English grammars read the
        // money, dates, times, measures, URLs… that strict text-v1 refuses.
        let narration = current && self.fallback.is_some();
        let prepared = narration.then(|| crate::english_tolerant::prepare(text));
        let text = prepared.as_deref().unwrap_or(text);
        let original = text;
        let text = if narration {
            crate::nemo::normalize(text)?
        } else if current {
            crate::text::expand(text, crate::Language::En)?
        } else {
            crate::english_numbers::expand(text)?
        };
        let readings = if current {
            crate::context::english_phones(&text)
        } else {
            Vec::new()
        };
        // Capitals are lost by normalization; the fallback splits on them.
        let written: HashMap<String, String> = original
            .split(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
            .filter(|w| !w.is_empty())
            .map(|w| (w.to_lowercase(), w.to_string()))
            .collect();
        // How capitals were written, for spelling acronyms in narration.
        let tokens: Vec<&str> = original
            .split(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
            .filter(|w| !w.is_empty())
            .collect();
        let caps = |w: &str| w.len() >= 2 && w.chars().all(|c| c.is_ascii_uppercase());
        let written_caps: std::collections::HashSet<&str> =
            tokens.iter().copied().filter(|w| caps(w)).collect();
        // A run of capitalized words is a shouted headline: read as words.
        let shouted: std::collections::HashSet<&str> = (0..tokens.len())
            .filter(|&i| {
                caps(tokens[i])
                    && (i > 0 && caps(tokens[i - 1]) || tokens.get(i + 1).is_some_and(|n| caps(n)))
            })
            .map(|i| tokens[i])
            .collect();
        let spell = |w: &str| -> Option<String> {
            if !narration || !(2..=5).contains(&w.len()) || !caps(w) {
                return None;
            }
            if !written_caps.contains(w) {
                // NeMo's own capitals: AM, PM, a domain's DE — but a domain
                // ending that is a word (dot APP, dot COM) is read as one.
                if WORD_DOMAINS.contains(&w.to_lowercase().as_str()) {
                    return None;
                }
                return acronym(w);
            }
            if shouted.contains(w) || w.len() > 3 || SHOUTED_WORDS.contains(&w) {
                return None; // a headline, NASA from the dictionary, "Download NOW"
            }
            acronym(w) // EST, CEO, the IT team
        };
        let mut index = 0;
        let mut out = String::new();
        let mut word = String::new();
        let mut flush = |word: &mut String, out: &mut String| -> Result<(), String> {
            if !word.is_empty() {
                let spelled = if self.overrides.contains_key(&word.to_lowercase()) {
                    None
                } else {
                    spell(word)
                };
                let start = out.len();
                let overridden = self.overrides.contains_key(&word.to_lowercase());
                if let Some(pronunciation) = self.overrides.get(&word.to_lowercase()) {
                    out.push_str(pronunciation);
                } else if spelled.is_some() {
                    out.push_str(&spoken_letters(word).ok_or("cannot spell")?);
                } else if let Some(Some(phones)) = readings.get(index) {
                    out.push_str(
                        &phones
                            .split_whitespace()
                            .map(phone)
                            .collect::<Result<Vec<_>, _>>()?
                            .join(""),
                    );
                } else {
                    if let Some(pronunciation) = self.dictionary.get(&word.to_lowercase()) {
                        out.push_str(pronunciation);
                    } else if current && matches!(word.as_str(), "miz" | "zeroth") {
                        // Original readings for two normalization words absent in CMU.
                        let phones = if word == "miz" {
                            "M IH1 Z"
                        } else {
                            "Z IH1 R OW0 TH"
                        };
                        out.push_str(
                            &phones
                                .split_whitespace()
                                .map(phone)
                                .collect::<Result<Vec<_>, _>>()?
                                .join(""),
                        );
                    } else if current && let Some(base) = self.fraction_plural(word) {
                        // These ordinal stems end in unvoiced /θ/: their plural is /s/.
                        out.push_str(base);
                        out.push('s');
                    } else if let Some(guess) = self.guess(word, &written) {
                        out.push_str(&guess);
                    } else {
                        return Err(format!(
                            "Unknown word '{word}'. Add a CMU-format pronunciation to the dictionary or use phoneme input."
                        ));
                    }
                }
                // Spelled letters keep their own stress: "AM" is not the verb.
                if american && !overridden && spelled.is_none() {
                    let spoken = out.split_off(start);
                    out.push_str(&connected_us(&word.to_lowercase(), &spoken));
                }
                index += 1;
                word.clear();
            }
            Ok(())
        };
        for c in text.chars() {
            if c.is_ascii_alphabetic() || c == '\'' {
                word.push(c);
            } else {
                flush(&mut word, &mut out)?;
                match c {
                    c if c.is_whitespace() => {
                        if !out.ends_with(' ') {
                            out.push(' ');
                        }
                    }
                    '.' | ',' | '!' | '?' | ';' | ':' => out.push(c),
                    '-' | '—' | '(' | ')' | '"' => out.push(' '),
                    // Narration never fails on a stray symbol NeMo left: a pause.
                    c if narration && !c.is_alphanumeric() => {
                        if !out.ends_with(' ') {
                            out.push(' ');
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Unsupported character '{c}'. Spell out this symbol or use phoneme input."
                        ));
                    }
                }
            }
        }
        flush(&mut word, &mut out)?;
        let out = out.trim().to_string();
        if out.is_empty() {
            return Err("enter some text".into());
        }
        Ok(out)
    }
    /// The fallback's pronunciation for an unknown `word` (lowercase, as
    /// normalized), recorded as a [`Guess`]; None when the fallback is off.
    fn guess(&self, word: &str, written: &HashMap<String, String>) -> Option<String> {
        let model = self.fallback.as_ref()?;
        let lower = word.to_lowercase();
        let original = written
            .get(&lower)
            .cloned()
            .unwrap_or_else(|| word.to_string());
        let resolve = |written: &str| -> Option<(String, &'static str)> {
            let lower = written.to_lowercase();
            self.arpabet(&lower)
                .map(|a| (a, "dictionary"))
                .or_else(|| self.split_at_apostrophes(&lower).map(|a| (a, "parts")))
                .or_else(|| self.split_at_capitals(written).map(|a| (a, "parts")))
                .or_else(|| acronym(written).map(|a| (a, "letters")))
                // A short lowercase run with no vowel (a URL's /mcp) is initials.
                .or_else(|| {
                    ((2..=5).contains(&lower.len())
                        && !lower.contains(['a', 'e', 'i', 'o', 'u', 'y']))
                    .then(|| acronym(&lower.to_uppercase()))
                    .flatten()
                    .map(|a| (a, "letters"))
                })
                .or_else(|| self.compound(&lower).map(|a| (a, "compound")))
                .or_else(|| {
                    // A full diphthong the model left unstressed (phytotaxa's
                    // AY0) reads reduced; give it secondary stress.
                    model.guess(&lower).map(|p| {
                        let p: Vec<String> = p
                            .into_iter()
                            .map(|ph| match ph.as_str() {
                                "AY0" | "EY0" | "AW0" | "OY0" => ph.replace('0', "2"),
                                _ => ph,
                            })
                            .collect();
                        (p.join(" "), "model")
                    })
                })
        };
        // KDKA's, APIs, PromoShot's: the base word, then the ending its last
        // sound takes. Guessing the whole token made "KDKA's" into "kiz".
        let base = original
            .strip_suffix("'s")
            .or_else(|| {
                original
                    .strip_suffix('s')
                    .filter(|b| b.len() >= 2 && b.chars().all(|c| c.is_ascii_uppercase()))
            })
            .filter(|b| !b.is_empty());
        let (arpabet, method, ending) = match base {
            Some(base) => {
                let (stem, method) = resolve(base)?;
                let last = stem.split_whitespace().last().unwrap_or_default();
                let ending = match last.trim_end_matches(['0', '1', '2']) {
                    "S" | "Z" | "SH" | "ZH" | "CH" | "JH" => "IH0 Z",
                    "P" | "T" | "K" | "F" | "TH" => "S",
                    _ => "Z",
                };
                (format!("{stem} {ending}"), method, Some(ending))
            }
            None => {
                let (arpabet, method) = resolve(&original)?;
                (arpabet, method, None)
            }
        };
        let joined = |a: &str| -> Option<String> {
            Some(
                a.split_whitespace()
                    .map(phone)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?
                    .join(""),
            )
        };
        let spoken = if method == "letters" {
            // One word per letter, so K D K A's stays four letters.
            let mut letters = spoken_letters(base.unwrap_or(&original))?;
            if let Some(ending) = ending {
                letters.push_str(&joined(ending)?);
            }
            letters
        } else {
            joined(&arpabet)?
        };
        let mut guesses = self.guesses.lock().unwrap_or_else(|e| e.into_inner());
        if !guesses.iter().any(|g| g.word == original) {
            guesses.push(Guess {
                word: original,
                arpabet,
                method,
            });
        }
        Some(spoken)
    }

    /// ARPAbet for a dictionary word, read back from the CMU source form.
    fn arpabet(&self, word: &str) -> Option<String> {
        self.arpabet.get(word).cloned()
    }

    /// `nor'easter` → noreaster or nor + easter: an inner apostrophe (not a
    /// possessive) dropped, else each piece a dictionary word.
    fn split_at_apostrophes(&self, word: &str) -> Option<String> {
        let inner = word.trim_matches('\'');
        if !inner.contains('\'') || word.ends_with("'s") {
            return None;
        }
        if let Some(whole) = self.arpabet(&inner.replace('\'', "")) {
            return Some(whole);
        }
        let parts: Option<Vec<String>> = inner
            .split('\'')
            .filter(|p| !p.is_empty())
            .map(|p| self.arpabet(p))
            .collect();
        parts.filter(|p| p.len() >= 2).map(|p| p.join(" "))
    }

    /// `PromoShot` → promo + shot, `ChatGPT` → chat + G P T: split where the
    /// capitals change, each part a dictionary word or a spelled acronym.
    fn split_at_capitals(&self, original: &str) -> Option<String> {
        let chars: Vec<char> = original.chars().collect();
        let mut parts = Vec::new();
        let mut start = 0;
        for i in 1..chars.len() {
            let (a, b) = (chars[i - 1], chars[i]);
            let next_lower = chars.get(i + 1).is_some_and(|c| c.is_ascii_lowercase());
            if (a.is_ascii_lowercase() && b.is_ascii_uppercase())
                || (a.is_ascii_uppercase() && b.is_ascii_uppercase() && next_lower)
            {
                parts.push(chars[start..i].iter().collect::<String>());
                start = i;
            }
        }
        parts.push(chars[start..].iter().collect());
        if parts.len() < 2 {
            return None;
        }
        let spoken: Option<Vec<String>> = parts
            .iter()
            .map(|p| self.arpabet(&p.to_lowercase()).or_else(|| acronym(p)))
            .collect();
        spoken.map(|s| s.join(" "))
    }

    /// `screenrecorder` → screen + recorder: exactly two known words of at
    /// least four letters each. Shorter pieces are too easy to find by
    /// accident (phytotaxa is not phy + tot + axa); the model reads those.
    fn compound(&self, word: &str) -> Option<String> {
        if word.len() < 6 || !word.is_ascii() {
            return None;
        }
        let n = word.len();
        // best[i]: the fewest parts covering word[..i], with their split points.
        let mut best: Vec<Option<Vec<usize>>> = vec![None; n + 1];
        best[0] = Some(Vec::new());
        for end in 4..=n {
            for begin in 0..=end.saturating_sub(4) {
                let Some(prefix) = best[begin].clone() else {
                    continue;
                };
                if prefix.len() >= 2 || !self.arpabet.contains_key(&word[begin..end]) {
                    continue;
                }
                let mut cuts = prefix;
                cuts.push(end);
                if best[end].as_ref().is_none_or(|b| cuts.len() < b.len()) {
                    best[end] = Some(cuts);
                }
            }
        }
        let cuts = best[n].clone().filter(|c| c.len() >= 2)?;
        let mut begin = 0;
        let mut spoken = Vec::new();
        for end in cuts {
            spoken.push(self.arpabet(&word[begin..end])?);
            begin = end;
        }
        Some(spoken.join(" "))
    }

    fn fraction_plural(&self, word: &str) -> Option<&str> {
        let stem = word.strip_suffix('s')?;
        if [
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
            "twentieth",
        ]
        .contains(&stem)
        {
            self.dictionary.get(stem).map(String::as_str)
        } else {
            None
        }
    }
}

/// Words English speech leaves unstressed in running text. Their CMU
/// citation forms carry primary stress, which Kokoro then voices.
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "nor", "if", "as", "than", "of", "to", "in", "on", "at",
    "by", "for", "from", "with", "into", "onto", "is", "are", "was", "were", "be", "been", "am",
    "has", "have", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "it's", "they", "them", "their", "i'm", "you're", "we're",
    "they're", "he's", "she's", "i'll", "you'll", "we'll", "i've", "you've", "we've",
];

/// One word's Kokoro phonemes with American connected-speech cues.
fn connected_us(word: &str, spoken: &str) -> String {
    if FUNCTION_WORDS.contains(&word) {
        return spoken.replace(['ˈ', 'ˌ'], "");
    }
    flap(spoken)
}

/// American flapping inside a word: t after a vowel or ɹ and before an
/// unstressed vowel is ɾ (better, water, city, party). Only t: a flapped d
/// is far less regular (video keeps its d). Not before a
/// syllabic n (button, written), where American English has a glottal
/// stop instead; not before a stressed vowel (hotel, attack).
fn flap(spoken: &str) -> String {
    const VOWELS: &str = "aeiouæɑɐɒɔəɛɜɪʊʌAIOWYᵊᵻ";
    let chars: Vec<char> = spoken.chars().collect();
    let mut out = String::with_capacity(spoken.len());
    for (i, &c) in chars.iter().enumerate() {
        let before = i.checked_sub(1).map(|j| chars[j]);
        let after = chars.get(i + 1).copied();
        let flapped = c == 't'
            && before.is_some_and(|b| VOWELS.contains(b) || b == 'ɹ')
            && after.is_some_and(|a| VOWELS.contains(a))
            && !(after == Some('ə')
                && chars.get(i + 2) == Some(&'n')
                && chars.get(i + 3).is_none_or(|n| !VOWELS.contains(*n)));
        out.push(if flapped { 'ɾ' } else { c });
    }
    out
}

/// Short words promo copy capitalizes for emphasis ("Download NOW", "the
/// NEW app"): read as words, where other short capitals (EST, CEO) are spelled.
const SHOUTED_WORDS: &[&str] = &[
    "NO", "GO", "UP", "ON", "IN", "TO", "DO", "BE", "WE", "MY", "ME", "SO", "OR", "AT", "BY", "IF",
    "OF", "AN", "AS", "IS", "HE", "NEW", "NOW", "GET", "BUY", "WIN", "TOP", "HOT", "BIG", "YES",
    "ALL", "OFF", "OUR", "YOU", "THE", "AND", "FOR", "ONE", "TWO", "TEN", "FUN", "WOW", "TRY",
    "ADD", "OUT", "NOT", "ANY", "WAY", "DAY", "PRO", "MAX", "LOW", "OLD", "SEE", "HOW", "WHY",
    "WHO", "ARE", "CAN", "HAS", "HAD", "WAS", "LET", "SET", "RUN", "FAN", "BOX", "SIX", "RED",
    "WON", "OWN", "USE", "ASK", "ART", "APP", "JOB", "MAP", "TIP", "YET", "NOR", "BUT",
];

/// Domain endings read as words even when NeMo capitalizes them (dot APP).
const WORD_DOMAINS: &[&str] = &[
    "com", "net", "org", "app", "dev", "edu", "gov", "info", "biz", "shop", "store", "blog",
    "tech", "site", "online", "cloud", "art", "design", "news", "live", "page", "studio",
];

/// A word spelled letter by letter, as Kokoro phonemes: one word per
/// letter, so a spelled "AM" after "nineteen" stays two letters instead of
/// slurring into one syllable.
fn spoken_letters(word: &str) -> Option<String> {
    let letters = acronym(&word.to_uppercase().replace('\'', ""))?;
    let mut out = Vec::new();
    let mut current = String::new();
    // acronym() writes each letter's phones in order; split them back by letter.
    let mut names = word
        .to_uppercase()
        .replace('\'', "")
        .into_bytes()
        .into_iter();
    let mut remaining: Vec<&str> = letters.split_whitespace().collect();
    remaining.reverse();
    for b in names.by_ref() {
        let count = LETTER_NAMES[(b - b'A') as usize].split_whitespace().count();
        for _ in 0..count {
            current.push_str(&phone(remaining.pop()?).ok()?);
        }
        out.push(std::mem::take(&mut current));
    }
    Some(out.join(" "))
}

/// Letter names in ARPAbet, for spelling an acronym.
const LETTER_NAMES: [&str; 26] = [
    "EY1",
    "B IY1",
    "S IY1",
    "D IY1",
    "IY1",
    "EH1 F",
    "JH IY1",
    "EY1 CH",
    "AY1",
    "JH EY1",
    "K EY1",
    "EH1 L",
    "EH1 M",
    "EH1 N",
    "OW1",
    "P IY1",
    "K Y UW1",
    "AA1 R",
    "EH1 S",
    "T IY1",
    "Y UW1",
    "V IY1",
    "D AH1 B AH0 L Y UW0",
    "EH1 K S",
    "W AY1",
    "Z IY1",
];

/// An all-capitals word of two to five letters (API, SDK, GPT), spelled out:
/// secondary stress on each letter, the main stress on the last.
fn acronym(word: &str) -> Option<String> {
    if !(2..=5).contains(&word.len()) || !word.chars().all(|c| c.is_ascii_uppercase()) {
        return None;
    }
    let letters: Vec<String> = word
        .bytes()
        .map(|b| LETTER_NAMES[(b - b'A') as usize].to_string())
        .collect();
    let last = letters.len() - 1;
    Some(
        letters
            .iter()
            .enumerate()
            .map(|(i, name)| {
                if i == last {
                    name.clone()
                } else {
                    name.replace('1', "2")
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn phone(value: &str) -> Result<String, String> {
    let base = value.trim_end_matches(['0', '1', '2']);
    let stress = if value.ends_with('1') {
        "ˈ"
    } else if value.ends_with('2') {
        "ˌ"
    } else {
        ""
    };
    let ipa = match base {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => {
            if value.ends_with('0') {
                "ə"
            } else {
                "ʌ"
            }
        }
        "AO" => "ɔ",
        "AW" => "W",
        "AY" => "I",
        "B" => "b",
        "CH" => "ʧ",
        "D" => "d",
        "DH" => "ð",
        "EH" => "ɛ",
        "ER" => {
            if value.ends_with('0') {
                "əɹ"
            } else {
                "ɜɹ"
            }
        }
        "EY" => "A",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "IH" => "ɪ",
        "IY" => "i",
        "JH" => "ʤ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "OW" => "O",
        "OY" => "Y",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "UH" => "ʊ",
        "UW" => "u",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        _ => return Err(format!("unknown CMU phoneme {value}")),
    };
    Ok(format!("{stress}{ipa}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stress_and_schwa() {
        assert_eq!(phone("AH0").unwrap(), "ə");
        assert_eq!(phone("EH1").unwrap(), "ˈɛ");
    }
    #[test]
    fn ipa_expands_vowels_without_changing_kokoro_tokens() {
        let frontend = English {
            dictionary: HashMap::from([("test".into(), "ˈA I O W Y".into())]),
            overrides: HashMap::new(),
            fallback: None,
            guesses: Default::default(),
            source: Default::default(),
            arpabet: HashMap::new(),
        };
        assert_eq!(frontend.phonemize("test!").unwrap(), "ˈA I O W Y!");
        assert_eq!(frontend.phonemize_ipa("test!").unwrap(), "ˈeɪ aɪ oʊ aʊ ɔɪ!");
        assert!(frontend.phonemize_ipa("unknown").is_err());
    }
    #[test]
    fn unknown_words_fail() {
        let frontend = English {
            dictionary: HashMap::from([("hello".into(), "həlˈoʊ".into())]),
            overrides: HashMap::new(),
            fallback: None,
            guesses: Default::default(),
            source: Default::default(),
            arpabet: HashMap::new(),
        };
        assert_eq!(frontend.phonemize("Hello!").unwrap(), "həlˈoʊ!");
        assert!(frontend.phonemize("unknown").is_err());
        assert!(frontend.phonemize("123").is_err());
    }
    #[test]
    fn lexicon_entries_add_and_override_words() {
        let mut frontend = English {
            dictionary: HashMap::from([("read".into(), "ɹˈid".into())]),
            overrides: HashMap::new(),
            fallback: None,
            guesses: Default::default(),
            source: Default::default(),
            arpabet: HashMap::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lexicon.dict");
        std::fs::write(
            &path,
            ";;; brand words\nPromoShot  P R OW1 M OW0 SH AA2 T # product\n\nread R EH1 D\n",
        )
        .unwrap();
        assert_eq!(frontend.add_lexicon(&path).unwrap(), 2);
        assert_eq!(frontend.phonemize("promoshot").unwrap(), "pɹˈOmOʃˌɑt");
        assert_eq!(frontend.phonemize("Read.").unwrap(), "ɹˈɛd.");
        for bad in ["mp4 EH1 M\n", "word XX1\n", "lonely\n"] {
            std::fs::write(&path, bad).unwrap();
            assert!(frontend.add_lexicon(&path).is_err(), "{bad}");
        }
    }
    #[test]
    fn american_cues_unstress_function_words_and_flap_t() {
        assert_eq!(connected_us("for", "fˈɔɹ"), "fɔɹ");
        assert_eq!(flap("bˈɛtəɹ"), "bˈɛɾəɹ");
        assert_eq!(flap("sˈɪti"), "sˈɪɾi");
        assert_eq!(flap("pˈɑɹti"), "pˈɑɹɾi");
        assert_eq!(flap("hOtˈɛl"), "hOtˈɛl");
        assert_eq!(flap("bˈʌtən"), "bˈʌtən");
        assert_eq!(flap("vˈɪdiO"), "vˈɪdiO");
        assert_eq!(flap("tˈim"), "tˈim");
    }
    #[test]
    fn the_fallback_splits_spells_and_guesses_and_reports_each_word() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cmudict.dict");
        std::fs::write(
            &path,
            "promo P R OW1 M OW0\nshot SH AA1 T\nchat CH AE1 T\nscreen S K R IY1 N\n\
             recorder R IH0 K AO1 R D ER0\nmeet M IY1 T\n",
        )
        .unwrap();
        let mut frontend = English::load(&path).unwrap();
        assert!(frontend.phonemize("Meet Figma.").is_err(), "off by default");
        frontend.set_fallback(true).unwrap();
        let spoken = frontend
            .phonemize("Meet PromoShot, ChatGPT, the SDK and screenrecorder, by Figma.")
            .unwrap();
        assert!(
            spoken.contains("ˌɛs dˌi kˈA"),
            "SDK spelled letter by letter: {spoken}"
        );
        let guesses = frontend.take_guesses();
        let by_word = |w: &str| guesses.iter().find(|g| g.word == w).unwrap().clone();
        assert_eq!(by_word("PromoShot").arpabet, "P R OW1 M OW0 SH AA1 T");
        assert_eq!(by_word("PromoShot").method, "parts");
        assert_eq!(by_word("ChatGPT").arpabet, "CH AE1 T JH IY2 P IY2 T IY1");
        // Short capitals are spelled by rule in narration, not guessed.
        assert!(guesses.iter().all(|g| g.word != "SDK"));
        assert_eq!(by_word("screenrecorder").method, "compound");
        assert_eq!(by_word("Figma").method, "model");
        assert!(
            by_word("Figma").arpabet.starts_with("F IH1 G"),
            "{:?}",
            by_word("Figma")
        );
        assert!(frontend.take_guesses().is_empty(), "taken once");
    }
    #[test]
    fn possessives_plurals_and_tolerant_forms_resolve_their_base() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cmudict.dict");
        std::fs::write(&path, "promo P R OW1 M OW0\nshot SH AA1 T\nnews N UW1 Z\n").unwrap();
        let mut frontend = English::load(&path).unwrap();
        frontend.set_fallback(true).unwrap();
        frontend
            .phonemize("KDKA's news, PromoShot's news, URLs.")
            .unwrap();
        let guesses = frontend.take_guesses();
        let by_word = |w: &str| {
            guesses
                .iter()
                .find(|g| g.word == w)
                .unwrap()
                .arpabet
                .clone()
        };
        assert_eq!(by_word("KDKA's"), "K EY2 D IY2 K EY2 EY1 Z");
        assert_eq!(by_word("PromoShot's"), "P R OW1 M OW0 SH AA1 T S");
        assert_eq!(by_word("URL's"), "Y UW2 AA2 R EH1 L Z");
        // The tolerant pass only runs with the fallback on.
        let mut strict = English::load(&path).unwrap();
        assert!(strict.phonemize("news at 3 p.m.").is_err());
        strict.set_fallback(true).unwrap();
        assert!(strict.phonemize("news at 3 p.m.").is_ok());
    }
    #[test]
    fn spelled_letters_stay_separate_and_apostrophes_split() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cmudict.dict");
        std::fs::write(
            &path,
            "am AE1 M\nat AE1 T\nnor N AO1 R\neaster IY1 S T ER0\nfive F AY1 V\n",
        )
        .unwrap();
        let mut frontend = English::load(&path).unwrap();
        frontend.set_fallback(true).unwrap();
        // NeMo writes "five AM": the letters A M, stressed, not the verb "am".
        let spoken = frontend.phonemize_american("at 5 a.m.").unwrap();
        assert!(spoken.ends_with("ˌA ˈɛm"), "{spoken}");
        frontend.phonemize("a nor'easter").unwrap();
        let guess = frontend
            .take_guesses()
            .into_iter()
            .find(|g| g.word == "nor'easter")
            .unwrap();
        assert_eq!(guess.arpabet, "N AO1 R IY1 S T ER0");
    }
}

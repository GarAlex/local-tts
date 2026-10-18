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
        let original = text;
        let text = if current {
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
        let mut index = 0;
        let mut out = String::new();
        let mut word = String::new();
        let mut flush = |word: &mut String, out: &mut String| -> Result<(), String> {
            if !word.is_empty() {
                let start = out.len();
                let overridden = self.overrides.contains_key(&word.to_lowercase());
                if let Some(pronunciation) = self.overrides.get(&word.to_lowercase()) {
                    out.push_str(pronunciation);
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
                if american && !overridden {
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
        let (arpabet, method) = self
            .split_at_capitals(&original)
            .map(|a| (a, "parts"))
            .or_else(|| acronym(&original).map(|a| (a, "letters")))
            .or_else(|| self.compound(&lower).map(|a| (a, "compound")))
            .or_else(|| model.guess(&lower).map(|p| (p.join(" "), "model")))?;
        let spoken = arpabet
            .split_whitespace()
            .map(phone)
            .collect::<Result<Vec<_>, _>>()
            .ok()?
            .join("");
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

    /// `screenrecorder` → screen + recorder: two or three known words of at
    /// least three letters each, the fewest parts winning.
    fn compound(&self, word: &str) -> Option<String> {
        if word.len() < 6 || !word.is_ascii() {
            return None;
        }
        let n = word.len();
        // best[i]: the fewest parts covering word[..i], with their split points.
        let mut best: Vec<Option<Vec<usize>>> = vec![None; n + 1];
        best[0] = Some(Vec::new());
        for end in 3..=n {
            for begin in 0..=end.saturating_sub(3) {
                let Some(prefix) = best[begin].clone() else {
                    continue;
                };
                if prefix.len() >= 3 || !self.arpabet.contains_key(&word[begin..end]) {
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
        frontend
            .phonemize("Meet PromoShot, ChatGPT, the SDK and screenrecorder, by Figma.")
            .unwrap();
        let guesses = frontend.take_guesses();
        let by_word = |w: &str| guesses.iter().find(|g| g.word == w).unwrap().clone();
        assert_eq!(by_word("PromoShot").arpabet, "P R OW1 M OW0 SH AA1 T");
        assert_eq!(by_word("PromoShot").method, "parts");
        assert_eq!(by_word("ChatGPT").arpabet, "CH AE1 T JH IY2 P IY2 T IY1");
        assert_eq!(by_word("SDK").arpabet, "EH2 S D IY2 K EY1");
        assert_eq!(by_word("SDK").method, "letters");
        assert_eq!(by_word("screenrecorder").method, "compound");
        assert_eq!(by_word("Figma").method, "model");
        assert!(
            by_word("Figma").arpabet.starts_with("F IH1 G"),
            "{:?}",
            by_word("Figma")
        );
        assert!(frontend.take_guesses().is_empty(), "taken once");
    }
}

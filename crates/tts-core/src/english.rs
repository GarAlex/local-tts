//! Dictionary-based English frontend; never silently substitutes a pronunciation.
use std::{collections::HashMap, path::Path};

pub struct English {
    dictionary: HashMap<String, String>,
    /// User lexicon entries; these win over the dictionary and contextual readings.
    overrides: HashMap<String, String>,
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
        })
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
}

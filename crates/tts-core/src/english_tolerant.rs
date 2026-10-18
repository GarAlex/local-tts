//! The pre-pass narration mode runs before NeMo's normalizer
//! ([`crate::nemo`]): only what NeMo's English grammars leave alone.
//!
//! - Accented Latin letters lose their marks: Kīlauea → Kilauea, ß → ss
//!   (NeMo passes them through, and the dictionary spells without them).
//! - Multipliers: 2.5x and 10× → 2.5 times, 10 times (NeMo reads "x").
//! - Web addresses lose `https://` and `www.`: narration says "promoshot
//!   dot app slash docs", where NeMo would read the scheme out in full.
//! - Roman numerals after a word that introduces a number: Chapter IV,
//!   World War II, Super Bowl LVIII → digits, which NeMo reads (its cased
//!   English grammars leave Roman numerals as written).
use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

pub(crate) fn prepare(text: &str) -> String {
    let folded = numbered_roman(&spoken_urls(&fold(text)));
    let chars: Vec<char> = folded.chars().collect();
    let mut out = String::with_capacity(folded.len());
    let mut i = 0;
    while i < chars.len() {
        let boundary = i == 0 || !chars[i - 1].is_alphanumeric();
        if boundary && let Some(used) = multiplier(&chars[i..]) {
            out.extend(&chars[i..i + used - 1]);
            out.push_str(" times");
            i += used;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `https://www.promoshot.app/docs/` → `promoshot.app/docs`.
fn spoken_urls(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let lower = word.to_ascii_lowercase();
            let mut rest = word;
            for scheme in ["https://", "http://"] {
                if lower.starts_with(scheme) {
                    rest = &word[scheme.len()..];
                }
            }
            if rest.to_ascii_lowercase().starts_with("www.") && rest.len() > 4 {
                rest = &rest[4..];
            }
            if rest.len() == word.len() {
                return word.to_string();
            }
            // A trailing slash says nothing aloud; keep sentence punctuation.
            let core = rest.trim_end_matches(|c: char| ",.;:!?)".contains(c));
            format!("{}{}", core.trim_end_matches('/'), &rest[core.len()..])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Words after which a Roman numeral is a number.
const NUMBERING: &[&str] = &[
    "chapter", "part", "volume", "vol.", "book", "act", "scene", "episode", "season", "phase",
    "level", "stage", "section", "article", "appendix", "round", "war", "bowl", "title", "canto",
    "psalm", "movement", "op.", "no.",
];

/// `Chapter IV` → `Chapter 4`: a valid Roman numeral (1–3999, standard form)
/// right after a numbering word.
fn numbered_roman(text: &str) -> String {
    let words: Vec<&str> = text.split(' ').collect();
    let mut out: Vec<String> = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let core = word.trim_end_matches(|c: char| ",.;:!?)".contains(c));
        let tail = &word[core.len()..];
        let after_numbering =
            i > 0 && NUMBERING.contains(&words[i - 1].to_lowercase().trim_end_matches(','));
        match roman(core).filter(|_| after_numbering) {
            Some(n) => out.push(format!("{n}{tail}")),
            None => out.push(word.to_string()),
        }
    }
    out.join(" ")
}

/// The value of a standard-form Roman numeral in capitals, 1–3999.
fn roman(word: &str) -> Option<u32> {
    if word.is_empty() || !word.chars().all(|c| "IVXLCDM".contains(c)) {
        return None;
    }
    let value = |c: char| match c {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        'D' => 500,
        _ => 1000,
    };
    let digits: Vec<u32> = word.chars().map(value).collect();
    let mut total: i64 = 0;
    for (i, &d) in digits.iter().enumerate() {
        if digits.get(i + 1).is_some_and(|&next| next > d) {
            total -= i64::from(d);
        } else {
            total += i64::from(d);
        }
    }
    let total = u32::try_from(total)
        .ok()
        .filter(|t| (1..=3999).contains(t))?;
    // Only the canonical spelling counts: IIII or VX are not numerals here.
    (to_roman(total) == word).then_some(total)
}

fn to_roman(mut n: u32) -> String {
    let table = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (value, letters) in table {
        while n >= value {
            out.push_str(letters);
            n -= value;
        }
    }
    out
}

/// Accented Latin letters to their base letters; ligatures and a few
/// letters with no decomposition spelled out. Other scripts pass through.
fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ß' => out.push_str("ss"),
            'æ' => out.push_str("ae"),
            'Æ' => out.push_str("Ae"),
            'œ' => out.push_str("oe"),
            'Œ' => out.push_str("Oe"),
            'ø' => out.push('o'),
            'Ø' => out.push('O'),
            'ł' => out.push('l'),
            'Ł' => out.push('L'),
            'đ' | 'ð' => out.push('d'),
            'Đ' | 'Ð' => out.push('D'),
            'þ' => out.push_str("th"),
            'Þ' => out.push_str("Th"),
            'ı' => out.push('i'),
            c if c.is_ascii() => out.push(c),
            c => out.extend(c.to_string().nfd().filter(|d| !is_combining_mark(*d))),
        }
    }
    out
}

fn digits(chars: &[char]) -> usize {
    chars.iter().take_while(|c| c.is_ascii_digit()).count()
}

fn ends_word(chars: &[char], at: usize) -> bool {
    chars.get(at).is_none_or(|c| !c.is_alphanumeric())
}

/// `2.5x`, `10x`, `3×` → the number, then "times" (the caller adds it).
/// Answers the length including the x.
fn multiplier(chars: &[char]) -> Option<usize> {
    let mut at = digits(chars);
    if at == 0 {
        return None;
    }
    if chars.get(at) == Some(&'.') && digits(&chars[at + 1..]) > 0 {
        at += 1 + digits(&chars[at + 1..]);
    }
    if !matches!(chars.get(at), Some('x' | 'X' | '×')) || !ends_word(chars, at + 1) {
        return None;
    }
    Some(at + 1)
}

#[cfg(test)]
mod tests {
    use super::prepare;

    #[test]
    fn accents_and_multipliers_become_readable() {
        assert_eq!(
            prepare("Kīlauea, Zürich, Straße"),
            "Kilauea, Zurich, Strasse"
        );
        assert_eq!(
            prepare("It is 2.5x faster, even 10× faster"),
            "It is 2.5 times faster, even 10 times faster"
        );
        assert_eq!(
            prepare("Chapter IV, World War II and Super Bowl LVIII."),
            "Chapter 4, World War 2 and Super Bowl 58."
        );
        assert_eq!(
            prepare("Visit https://www.promoshot.app/docs/, or http://example.com."),
            "Visit promoshot.app/docs, or example.com."
        );
        for keep in [
            "0x1f",
            "3:19 a.m.",
            "1990s",
            "max 5",
            "I like it",
            "the IV drip",
            "Act IIII",
        ] {
            assert_eq!(prepare(keep), keep);
        }
    }
}

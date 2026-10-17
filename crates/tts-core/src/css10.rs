//! Versioned CSS10 grapheme contract shared by preparation and native inference.
use crate::Language;
use unicode_normalization::UnicodeNormalization;

pub const VERSION: &str = "css10-graphemes-v1";

pub fn symbols() -> Vec<char> {
    serde_json::from_str(include_str!("../../../assets/css10-symbols.json"))
        .expect("embedded CSS10 inventory")
}

pub fn normalize(text: &str, language: Language) -> Result<String, String> {
    if !matches!(language, Language::De | Language::Es) {
        return Err("CSS10 frontend requires German or Spanish".into());
    }
    normalize_with_inventory(text, &symbols())
}

pub(crate) fn normalize_with_inventory(text: &str, inventory: &[char]) -> Result<String, String> {
    if text.chars().count() > 10000 {
        return Err("text exceeds 10000 characters".into());
    }
    let mut output = String::new();
    for c in text.nfc().flat_map(char::to_lowercase) {
        match c {
            ';' | ':' => output.push(','),
            '-' => output.push(' '),
            '<' | '>' | '(' | ')' | '[' | ']' | '"' | '¡' | '¿' => (),
            c if c.is_whitespace() => output.push(' '),
            c => output.push(c),
        }
    }
    let output = output.split_whitespace().collect::<Vec<_>>().join(" ");
    if output.is_empty() {
        return Err("enter text".into());
    }
    for c in output.chars() {
        // Private placeholders represent source specials, never user text.
        if !inventory[4..].contains(&c) {
            return Err(format!(
                "character '{c}' absent from CSS10 text inventory; write numbers and abbreviations as spoken words"
            ));
        }
    }
    Ok(output)
}

/// Encode already-normalized text with source blank ID 3, including boundaries.
pub fn encode(text: &str) -> Result<Vec<i64>, String> {
    encode_with_inventory(text, &symbols())
}

pub(crate) fn encode_with_inventory(text: &str, inventory: &[char]) -> Result<Vec<i64>, String> {
    if text.is_empty() {
        return Err("enter text".into());
    }
    let mut ids = vec![3];
    for c in text.chars() {
        let index = inventory[4..]
            .iter()
            .position(|v| *v == c)
            .ok_or_else(|| format!("character '{c}' absent from CSS10 text inventory"))?;
        ids.extend([index as i64 + 4, 3]);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_blanks_accents_and_explicit_punctuation_policy() {
        let symbols = symbols();
        assert_eq!(symbols.len(), 129);
        assert_eq!(&symbols[..4], &['\0', '\u{e001}', '\u{e002}', '\u{e003}']);
        assert_eq!(encode("a").unwrap(), vec![3, 4, 3]);
        assert_eq!(
            normalize("  GRÜẞE: (Köln)-Bonn! ", Language::De).unwrap(),
            "grüße, köln bonn!"
        );
        assert_eq!(
            normalize("¡Buenos di\u{301}as! ¿Niño?", Language::Es).unwrap(),
            "buenos días! niño?"
        );
        for text in ["", "()", "\0", "\u{e003}", "12", "hello 🙂"] {
            assert!(normalize(text, Language::Es).is_err());
        }
        assert!(normalize("bonjour", Language::Fr).is_err());
        assert!(encode("\0").is_err());
    }
}

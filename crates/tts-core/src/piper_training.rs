//! Versioned Ukrainian text tokens for the pinned Piper training checkpoint.
//! Keeps all 256 embedding rows; unused rows are not valid input characters.
use crate::Language;
use unicode_normalization::UnicodeNormalization;
pub const VERSION: &str = "piper-uk-text-v1";

pub fn symbols() -> Vec<char> {
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../../../assets/piper-uk-config.json")).unwrap();
    let mut symbols: Vec<char> = (0..256)
        .map(|i| char::from_u32(0xf0000 + i).unwrap())
        .collect();
    for (text, ids) in config["phoneme_id_map"].as_object().unwrap() {
        let id = ids[0].as_u64().unwrap() as usize;
        symbols[id] = if id == 0 {
            '\0'
        } else {
            text.chars().next().unwrap()
        };
    }
    symbols
}
pub fn normalize(text: &str, language: Language) -> Result<String, String> {
    if language != Language::Uk {
        return Err("Piper text training frontend requires Ukrainian".into());
    }
    // Preserve old voice packages: text-v1 is an explicit outer training contract.
    let expanded = crate::cyrillic_numbers::expand(text, language)?;
    let text = crate::frontend::normalize(&expanded.replace(['’', 'ʼ'], "'"), language)?;
    let inventory = symbols();
    let text: String = text.nfd().collect();
    for c in text.chars() {
        if "_^$".contains(c) || c == '\0' || (0xf0000..=0xf00ff).contains(&(c as u32)) {
            return Err("reserved token in Ukrainian text".into());
        }
        if !inventory.contains(&c) {
            return Err(format!("unsupported Ukrainian character {c:?}"));
        }
    }
    Ok(text)
}
/// `text` must have been normalized with this version. BOS/pad/characters/pad/EOS.
pub fn encode(text: &str) -> Result<Vec<i64>, String> {
    let inventory = symbols();
    let mut tokens = vec![1, 0];
    for c in text.chars() {
        if "_^$".contains(c) || c == '\0' || (0xf0000..=0xf00ff).contains(&(c as u32)) {
            return Err("reserved token in Ukrainian text".into());
        }
        let id = inventory
            .iter()
            .position(|&s| s == c)
            .ok_or_else(|| format!("unsupported Ukrainian character {c:?}"))?;
        tokens.extend([id as i64, 0]);
    }
    tokens.push(2);
    if text.is_empty() || tokens.len() > 1024 {
        return Err("empty Ukrainian text or chunk exceeds 1024 tokens".into());
    }
    Ok(tokens)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_indices_and_matches_piper_specials_and_decomposition() {
        let inventory = symbols();
        assert_eq!(inventory.len(), 256);
        assert_eq!(&inventory[..3], &['\0', '^', '$']);
        let text = normalize("Ґ+Їжак, 12!", Language::Uk).unwrap();
        assert!(text.contains("і\u{308}\u{301}"));
        let ids = encode(&text).unwrap();
        assert_eq!(&ids[..2], &[1, 0]);
        assert_eq!(ids.last(), Some(&2));
        assert_eq!(
            normalize("Прив+іт", Language::Uk).unwrap(),
            normalize("Приві́т", Language::Uk).unwrap()
        );
        for text in ["^", "$", "_", "\0", "\u{f000a}", "hello"] {
            assert!(normalize(text, Language::Uk).is_err());
        }
        assert!(normalize("привіт", Language::Ru).is_err());
        assert!(encode(&"а".repeat(511)).is_err());
    }
}

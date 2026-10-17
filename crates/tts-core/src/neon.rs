//! Native grapheme contracts for pinned European Neon training bases.
use crate::Language;

pub const VERSION: &str = "neon-graphemes-v1";
pub fn supports(language: Language) -> bool {
    matches!(
        language,
        Language::Fr | Language::It | Language::Pt | Language::Pl | Language::Nl
    )
}
pub fn symbols(language: Language) -> Result<Vec<char>, String> {
    if !supports(language) {
        return Err("Neon frontend requires French, Italian, Portuguese, Polish or Dutch".into());
    }
    Ok(if language == Language::It {
        serde_json::from_str(include_str!("../../../assets/neon-it-symbols.json"))
            .expect("embedded Italian Neon inventory")
    } else {
        crate::css10::symbols()
    })
}
pub fn normalize(text: &str, language: Language) -> Result<String, String> {
    crate::css10::normalize_with_inventory(text, &symbols(language)?)
}
pub fn encode(text: &str, language: Language) -> Result<Vec<i64>, String> {
    crate::css10::encode_with_inventory(text, &symbols(language)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_each_source_inventory_and_accents() {
        for (lang, text, expected) in [
            (
                Language::Fr,
                "LE GARÇON: près du café.",
                "le garçon, près du café.",
            ),
            (
                Language::It,
                "LA CITTÀ È GIÀ SVEGLIA.",
                "la città è già sveglia.",
            ),
            (
                Language::Pt,
                "O CORAÇÃO ESTÁ TRANQUILO.",
                "o coração está tranquilo.",
            ),
            (
                Language::Pl,
                "ŁÓDŹ PŁYNIE PRZEZ RZEKĘ.",
                "łódź płynie przez rzekę.",
            ),
            (Language::Nl, "Het café is open.", "het café is open."),
        ] {
            assert_eq!(normalize(text, lang).unwrap(), expected);
            let inventory = symbols(lang).unwrap();
            assert_eq!(
                inventory.len(),
                if lang == Language::It { 140 } else { 129 }
            );
            let ids = encode(expected, lang).unwrap();
            assert!(ids.iter().step_by(2).all(|id| *id == 3));
            let recovered: String = ids
                .iter()
                .skip(1)
                .step_by(2)
                .map(|id| inventory[*id as usize])
                .collect();
            assert_eq!(recovered, expected);
            for bad in ["12", "🙂", "\0", "\u{e003}"] {
                assert!(normalize(bad, lang).is_err());
            }
        }
        assert_ne!(
            encode("è", Language::It).unwrap(),
            encode("è", Language::Fr).unwrap()
        );
        assert_eq!(normalize("cafe\u{301}", Language::Fr).unwrap(), "café");
        assert!(normalize("hello", Language::En).is_err());
    }
}

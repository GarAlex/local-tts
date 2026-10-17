//! Versioned own-voice text normalization. Stress markers require annotated training.
use crate::Language;
use unicode_normalization::UnicodeNormalization;

pub const VERSION: &str = "grapheme-v2";
/// Canonical matching for dataset tools; malformed annotations remain visible for preparation errors.
pub fn match_key(text: &str, language: Language) -> String {
    normalize(text, language).unwrap_or_else(|_| crate::training::normalize(text))
}
pub fn normalize(text: &str, language: Language) -> Result<String, String> {
    if text.chars().count() > 10000 {
        return Err("text exceeds 10000 characters".into());
    }
    let lower: String = text.nfc().flat_map(char::to_lowercase).collect();
    let lower = lower.split_whitespace().collect::<Vec<_>>().join(" ");
    if !language.is_cyrillic() {
        return Ok(lower);
    }
    let vowels = match language {
        Language::Ru => "аеёиоуыэюя",
        Language::Uk => "аеєиіоїуюя",
        _ => unreachable!(),
    };
    let mut chars = lower.chars().peekable();
    let mut output = String::new();
    let mut stress_count = 0;
    while let Some(c) = chars.next() {
        if c == '+' {
            let vowel = chars
                .next()
                .filter(|c| vowels.contains(*c))
                .ok_or("put + immediately before a vowel in a stress override")?;
            stress_count += 1;
            output.push(vowel);
            output.push('\u{301}');
        } else if c == '\u{301}' {
            if !output.chars().last().is_some_and(|c| vowels.contains(c)) {
                return Err("stress accent must follow a vowel".into());
            }
            stress_count += 1;
            output.push(c);
        } else {
            if !c.is_alphabetic() && !['\'', '’'].contains(&c) {
                stress_count = 0;
            }
            output.push(c);
        }
        if stress_count > 1 {
            return Err("only one explicit stress marker is supported per word".into());
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stress_spellings_match_and_language_letters_survive() {
        assert_eq!(normalize("  З+АМОК  ", Language::Ru).unwrap(), "за́мок");
        assert_eq!(
            normalize("за́мок", Language::Ru).unwrap(),
            normalize("з+амок", Language::Ru).unwrap()
        );
        assert_eq!(normalize("Ґ +Ї Ї І Є", Language::Uk).unwrap(), "ґ ї́ ї і є");
        assert_eq!(normalize("+Європа", Language::Uk).unwrap(), "є́вропа");
        assert_eq!(normalize("Е\u{308} И\u{306}", Language::Ru).unwrap(), "ё й");
        assert_eq!(normalize("п’+ять", Language::Uk).unwrap(), "п’я́ть");
        assert_ne!(
            normalize("зам+ок", Language::Ru).unwrap(),
            normalize("з+амок", Language::Ru).unwrap()
        );
    }
    #[test]
    fn malformed_stress_is_rejected() {
        for text in ["+", "+б", "а́́", "б́", "з+ам+ок"] {
            assert!(normalize(text, Language::Ru).is_err(), "{text}");
        }
        assert!(normalize("+ы", Language::Uk).is_err());
        assert_eq!(
            normalize("  Plain TEXT ", Language::En).unwrap(),
            "plain text"
        );
    }
}

//! Original phrase rules, deliberately limited to explicit lexical cues.
//! No general part-of-speech tagger or automatic stress predictor is claimed.
use crate::Language;

struct Word<'a> {
    start: usize,
    end: usize,
    text: &'a str,
}
fn words(text: &str) -> Vec<Word<'_>> {
    let mut result = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() || c == '\u{301}' || (c == '\'' && start.is_some()) {
            start.get_or_insert(i);
        } else if let Some(begin) = start.take() {
            result.push(Word {
                start: begin,
                end: i,
                text: &text[begin..i],
            });
        }
    }
    if let Some(begin) = start {
        result.push(Word {
            start: begin,
            end: text.len(),
            text: &text[begin..],
        });
    }
    result
}
fn neighbour<'a>(text: &str, words: &'a [Word<'a>], index: usize, next: bool) -> &'a str {
    let other = if next {
        index.checked_add(1)
    } else {
        index.checked_sub(1)
    };
    let Some(other) = other.filter(|&i| i < words.len()) else {
        return "";
    };
    let (left, right) = if next { (index, other) } else { (other, index) };
    if text[words[left].end..words[right].start]
        .chars()
        .all(char::is_whitespace)
    {
        words[other].text
    } else {
        ""
    }
}

/// Adds stress only for supported adjacent phrase cues. Already stressed words survive.
pub fn stress(text: &str, language: Language) -> Result<String, String> {
    let text = crate::frontend::normalize(text, language)?;
    if !language.is_cyrillic() {
        return Ok(text);
    }
    let spans = words(&text);
    let mut out = String::new();
    let mut end = 0;
    for (i, span) in spans.iter().enumerate() {
        out.push_str(&text[end..span.start]);
        let previous = neighbour(&text, &spans, i, false).replace('\u{301}', "");
        let replacement = if span.text.contains('\u{301}') {
            None
        } else {
            match (language, span.text, previous.as_str()) {
                (Language::Ru, "замок", "дверной" | "навесной" | "кодовый") => {
                    Some("замо́к")
                }
                (Language::Ru, "замок", "средневековый" | "старинный" | "королевский") => {
                    Some("за́мок")
                }
                (Language::Ru, "мука", "пшеничная" | "ржаная" | "рисовая") => {
                    Some("мука́")
                }
                (Language::Ru, "мука", "невыносимая" | "страшная") => {
                    Some("му́ка")
                }
                (Language::Ru, "атлас", "географический" | "школьный") => {
                    Some("а́тлас")
                }
                (Language::Ru, "атлас", "шёлковый" | "шелковый") => {
                    Some("атла́с")
                }
                (Language::Uk, "замок", "дверний" | "навісний" | "кодовий") => {
                    Some("замо́к")
                }
                (Language::Uk, "замок", "середньовічний" | "старовинний" | "королівський") => {
                    Some("за́мок")
                }
                (Language::Uk, "атлас", "географічний" | "шкільний") => {
                    Some("а́тлас")
                }
                (Language::Uk, "атлас", "шовковий") => Some("атла́с"),
                _ => None,
            }
        };
        out.push_str(replacement.unwrap_or(span.text));
        end = span.end;
    }
    out.push_str(&text[end..]);
    if out.chars().count() > 10000 {
        return Err("contextual text exceeds 10000 characters".into());
    }
    Ok(out)
}

/// One optional original ARPABET reading per English word, with sentence boundaries.
/// Unresolved homographs retain their dictionary pronunciation.
pub(crate) fn english_phones(text: &str) -> Vec<Option<&'static str>> {
    let text = text.to_ascii_lowercase();
    let spans = words(&text);
    spans
        .iter()
        .enumerate()
        .map(|(i, span)| {
            let previous = neighbour(&text, &spans, i, false);
            let next = neighbour(&text, &spans, i, true);
            match (span.text, previous, next) {
                ("read", "have" | "has" | "had", _) => Some("R EH1 D"),
                ("read", "to" | "will" | "can" | "must" | "should" | "please", _) => {
                    Some("R IY1 D")
                }
                ("read", _, "yesterday") => Some("R EH1 D"),
                ("lead", _, "pipe" | "pipes" | "metal" | "poisoning") => Some("L EH1 D"),
                ("lead", "to" | "will" | "can" | "must", _) => Some("L IY1 D"),
                ("wind", _, "clock" | "clocks" | "watch" | "watches") => Some("W AY1 N D"),
                ("wind", "the", "blows" | "blew") => Some("W IH1 N D"),
                ("bass", _, "guitar" | "guitars" | "player" | "players" | "drum" | "drums") => {
                    Some("B EY1 S")
                }
                ("bass", _, "fish" | "fishing") => Some("B AE1 S"),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phrase_cues_do_not_cross_punctuation_or_override_annotations() {
        assert_eq!(
            stress("Дверной замок, старинный замок.", Language::Ru).unwrap(),
            "дверной замо́к, старинный за́мок."
        );
        assert_eq!(
            stress("Дверний замок, старовинний замок.", Language::Uk).unwrap(),
            "дверний замо́к, старовинний за́мок."
        );
        assert_eq!(
            stress("Дверной з+амок. Старинный, замок.", Language::Ru).unwrap(),
            "дверной за́мок. старинный, замок."
        );
        assert_eq!(
            stress("пшеничная мука и шёлковый атлас", Language::Ru).unwrap(),
            "пшеничная мука́ и шёлковый атла́с"
        );
        assert_eq!(
            stress("географічний атлас і шовковий атлас", Language::Uk).unwrap(),
            "географічний а́тлас і шовковий атла́с"
        );
        assert_eq!(
            stress("замок замки xзамок", Language::Ru).unwrap(),
            "замок замки xзамок"
        );
    }
    #[test]
    fn english_tense_and_noun_cues_stay_local() {
        assert_eq!(
            english_phones("I have read. I will read. A lead pipe. A bass guitar."),
            vec![
                None,
                None,
                Some("R EH1 D"),
                None,
                None,
                Some("R IY1 D"),
                None,
                Some("L EH1 D"),
                None,
                None,
                Some("B EY1 S"),
                None
            ]
        );
        assert_eq!(
            english_phones("Have. Read a bass."),
            vec![None, None, None, None]
        );
    }
}

//! Native, stressed arithmetic conditioning inventory for RU/UK teachers.
use local_tts_core::{Language, math_speech, piper_training, vosk};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if !(a.len() == 4
        || (a.len() == 5 && matches!(a[4].as_str(), "--contextual" | "--double-zero")))
    {
        return Err(
            "usage: math_cyrillic_tokens ru|uk <dictionary> <new-output.json> [--contextual|--double-zero]"
                .into(),
        );
    }
    let language = match a[1].as_str() {
        "ru" => Language::Ru,
        "uk" => Language::Uk,
        _ => return Err("expected ru or uk".into()),
    };
    let ru = if language == Language::Ru {
        Some(vosk::Frontend::load(std::path::Path::new(&a[2]))?)
    } else {
        None
    };
    let encode = |text: &str| -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        if let Some(frontend) = &ru {
            Ok(frontend.tokenize(text)?)
        } else {
            Ok(piper_training::encode(&piper_training::normalize(
                text, language,
            )?)?)
        }
    };
    let double_zero = a.get(4).is_some_and(|v| v == "--double-zero");
    let mut carriers = Vec::new();
    if a.len() == 5 {
        if double_zero {
            carriers.push(math_speech::words_for_language("0.00 + 1", language)?);
        }
        for n in (0..1000).chain([1000, 2000, 5000, 1000000, 2000000, 5000000]) {
            carriers.push(math_speech::words_for_language(
                &format!("{n} + 1"),
                language,
            )?);
        }
        for expression in [
            "1 - 1", "1 * 1", "1 / 1", "1 = 1", "-1", "+1", "(1 + 1)", "0.1",
        ] {
            carriers.push(math_speech::words_for_language(expression, language)?);
        }
    }
    let mut records = Vec::new();
    let mut inventory = math_speech::vocabulary_for_language(language)?;
    let zero = math_speech::words_for_language("0", language)?.join(" ");
    let pair = format!("{zero} {zero}");
    if double_zero {
        inventory.push(pair.clone());
    }
    for word in inventory {
        let tokens = encode(&word)?;
        let mut record = serde_json::json!({"word":word,"tokens":tokens});
        if a.len() == 5 && (!double_zero || word == pair) {
            let parts: Vec<_> = word.split_whitespace().collect();
            let carrier = carriers
                .iter()
                .find(|words| {
                    words
                        .windows(parts.len())
                        .any(|window| window.iter().map(String::as_str).eq(parts.iter().copied()))
                })
                .ok_or("word has no valid arithmetic carrier")?;
            let index = carrier
                .windows(parts.len())
                .position(|window| window.iter().map(String::as_str).eq(parts.iter().copied()))
                .unwrap();
            let phrase = carrier.join(" ");
            let context_tokens = encode(&phrase)?;
            let start = if index == 0 {
                2
            } else {
                encode(&carrier[..index].join(" "))?.len() + 1
            };
            let suffix = if language == Language::Ru { 2 } else { 1 };
            let body = &tokens[2..tokens.len() - suffix];
            let end = start + body.len();
            if context_tokens.get(start..end) != Some(body) {
                return Err("native contextual token span differs from word".into());
            }
            record["context_tokens"] = serde_json::json!(context_tokens);
            record["context_start"] = serde_json::json!(start);
            record["context_end"] = serde_json::json!(end);
            record["context_text"] = serde_json::json!(phrase);
        }
        records.push(record);
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[3])?;
    serde_json::to_writer(file, &records)?;
    Ok(())
}

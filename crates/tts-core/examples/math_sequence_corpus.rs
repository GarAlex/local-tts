//! Prepare disjoint full-utterance numeric phoneme targets through native frontends.
use local_tts_core::{Language, math_speech, piper_training, vosk};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: math_sequence_corpus ru|uk <dictionary> <count> <new-output.json>".into(),
        );
    }
    let language = match args[1].as_str() {
        "ru" => Language::Ru,
        "uk" => Language::Uk,
        _ => return Err("requires ru or uk".into()),
    };
    let ru = if language == Language::Ru {
        Some(vosk::Frontend::load(std::path::Path::new(&args[2]))?)
    } else {
        None
    };
    let mut records = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for record in math_speech::corpus(args[3].parse()?)? {
        let spoken = math_speech::words_for_language(&record.expression, language)?.join(" ");
        if !seen.insert(spoken.clone()) {
            return Err("spoken curriculum overlap across splits".into());
        }
        let tokens = if let Some(frontend) = &ru {
            frontend.tokenize(&spoken)?
        } else {
            piper_training::encode(&piper_training::normalize(&spoken, language)?)?
        };
        records.push(serde_json::json!({"expression":record.expression,"spoken":spoken,"split":record.split,"tokens":tokens}));
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    serde_json::to_writer(
        file,
        &serde_json::json!({"schema_version":1,"language":args[1],"records":records}),
    )?;
    Ok(())
}

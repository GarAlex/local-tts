//! Deterministic synthetic teacher corpus for output-based arithmetic distillation.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    dataset::{Dataset, Rights, Utterance},
    kokoro::Kokoro,
    math_speech::{corpus, short_corpus},
};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 && !(args.len() == 5 && args[4] == "--short") {
        return Err(
            "usage: math_dataset <kokoro-folder> <new-output-folder> <count> [--short]".into(),
        );
    }
    let count: usize = args[3].parse()?;
    let short = args.len() == 5;
    let curriculum = if short {
        short_corpus(count)?
    } else {
        corpus(count)?
    };
    let root = Path::new(&args[2]);
    if root.exists() {
        return Err("output already exists".into());
    }
    let mut teacher = Kokoro::load(Path::new(&args[1]))?;
    std::fs::create_dir_all(root.join("audio"))?;
    let mut utterances = Vec::new();
    let mut expressions = Vec::new();
    for (i, line) in curriculum.into_iter().enumerate() {
        let expression = line.expression;
        let spoken = line.spoken;
        let split = line.split;
        let audio = teacher.synthesize(&SynthesisRequest {
            text: &spoken,
            language: Language::En,
            speaker: "af_heart",
            duration_scale: 1.0,
        })?;
        let mut audio = local_tts_core::clips::resample(&audio, 22050)?;
        if short && audio.samples.len() < 8192 {
            audio.samples.resize(8192, 0.0);
        }
        let relative = format!("audio/{i:05}.wav");
        audio
            .write_wav(&root.join(&relative))
            .map_err(|e| e.to_string())?;
        expressions.push(serde_json::json!({"id":format!("math-{i:05}"),"expression":expression,"spoken":spoken,"split":split}));
        utterances.push(Utterance {
            id: format!("math-{i:05}"),
            audio: relative,
            text: spoken,
            speaker: "math-en".into(),
            split,
        });
        println!("teacher clip {}/{}", i + 1, count);
    }
    let dataset = Dataset {
        schema_version: 1,
        language: Language::En,
        sample_rate: 22050,
        rights: Rights {
            audio: "Synthetic Kokoro af_heart; see retained teacher license and provenance".into(),
            transcripts: "Original deterministic project arithmetic corpus (MIT)".into(),
            speaker_consent: "Packaged teacher voice; no new human recording or custom voice clone"
                .into(),
        },
        utterances,
    };
    std::fs::write(
        root.join("dataset.json"),
        serde_json::to_vec_pretty(&dataset)?,
    )?;
    std::fs::write(
        root.join("curriculum.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"version":if short {"arithmetic-short-v3"} else {"arithmetic-v2"},"seed":if short {"0x6d6174687633"} else {"0x6d6174687632"},"count":count,"unique_spoken":true,"structures_per_block":if short {4} else {16},"split":"six train blocks, one validation, one test", "minimum_samples":if short {Some(8192)} else {None}}),
        )?,
    )?;
    std::fs::write(
        root.join("expressions.json"),
        serde_json::to_vec_pretty(&expressions)?,
    )?;
    for name in ["APACHE-LICENSE.txt", "MODEL_CARD.md", "provenance.json"] {
        std::fs::copy(
            Path::new(&args[1]).join(name),
            root.join(format!("TEACHER-{name}")),
        )?;
    }
    Ok(())
}

//! Compare a compact numeric candidate against its exact full pretrained source.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    math_speech::{MathStudent, words_for_language},
    vits::Vits,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
fn sha(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().collect();
    if !(args.len() == 5 || (args.len() == 7 && args[5] == "--expressions")) {
        return Err(
            "usage: math_numeric_compare ru|uk <full-voice> <compact-voice> <new-output> [--expressions JSON]".into(),
        );
    }
    let language = match args[1].as_str() {
        "ru" => Language::Ru,
        "uk" => Language::Uk,
        _ => return Err("requires ru or uk".into()),
    };
    let full = Path::new(&args[2]);
    let compact = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    if output.exists() {
        return Err("output already exists".into());
    }
    let source: Value =
        serde_json::from_slice(&std::fs::read(full.join("exported-checkpoint.json"))?)?;
    let candidate: Value =
        serde_json::from_slice(&std::fs::read(compact.join("arithmetic-neural.json"))?)?;
    if source["sha256"].as_str().is_none()
        || source["sha256"] != candidate["source_checkpoint_sha256"]
    {
        return Err("full voice and candidate checkpoint lineage differ".into());
    }
    let expressions: Vec<String> = if args.len() == 7 {
        let bytes = std::fs::read(&args[6])?;
        if bytes.len() > 128_000 {
            return Err("expression file exceeds 128 KB".into());
        }
        serde_json::from_slice(&bytes)?
    } else {
        [
            "12 + 225 = 237",
            "(2 * 3) / 4",
            "0.009 + 1 = 1.009",
            "670.70 = 670.70",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    };
    if expressions.is_empty() || expressions.len() > 32 {
        return Err("expected 1..32 expressions".into());
    }
    let mut unique = std::collections::BTreeSet::new();
    for expression in &expressions {
        words_for_language(expression, language)?;
        if !unique.insert(expression) {
            return Err("duplicate expression".into());
        }
    }
    let mut teacher = Vits::load(full)?;
    let mut student = MathStudent::load(compact)?;
    if teacher.manifest.language != language || student.language() != language {
        return Err("voice language mismatch".into());
    }
    let speaker = teacher.manifest.speakers[0].clone();
    std::fs::create_dir_all(output)?;
    let mut cases = Vec::new();
    for (index, expression) in expressions.iter().enumerate() {
        let spoken = words_for_language(expression, language)?.join(" ");
        let mut recordings = Vec::new();
        for kind in ["full", "compact"] {
            let start = Instant::now();
            let audio = if kind == "full" {
                teacher.synthesize(&SynthesisRequest {
                    text: &spoken,
                    language,
                    speaker: &speaker,
                    duration_scale: 1.0,
                })?
            } else {
                student.synthesize(&SynthesisRequest {
                    text: expression,
                    language,
                    speaker: "default",
                    duration_scale: 1.0,
                })?
            };
            let elapsed = start.elapsed().as_secs_f64();
            if audio.samples.is_empty() || audio.samples.iter().any(|v| !v.is_finite()) {
                return Err("invalid waveform".into());
            }
            let file = format!("{index:02}-{kind}.wav");
            audio.write_wav(&output.join(&file))?;
            recordings.push(json!({"kind":kind,"file":file,"sha256":sha(&output.join(&file))?,"sample_rate":audio.sample_rate,"duration_seconds":audio.samples.len() as f64/audio.sample_rate as f64,"generation_seconds":elapsed}));
        }
        cases.push(json!({"expression":expression,"spoken":spoken,"recordings":recordings}));
        println!("{} case {} complete", args[1], index + 1);
    }
    let report = json!({"language":args[1],"source_checkpoint_sha256":source["sha256"],"full_model_sha256":sha(&full.join("model.onnx"))?,"compact_model_sha256":sha(&compact.join("model.onnx"))?,"cases":cases,"expression_file_sha256":if args.len()==7 {Some(sha(Path::new(&args[6]))?)} else {None},"quality_status":"listening pending; generated probes are not independent held-out training evaluation"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

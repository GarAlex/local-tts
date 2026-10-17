//! Export three-way audio on a teacher dataset's explicitly reserved validation or test split.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    dataset::{Dataset, Split},
    math_speech::{MathStudent, MathVoice, words},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};
#[derive(Deserialize)]
struct Expression {
    id: String,
    expression: String,
    spoken: String,
}
fn sha(path: &Path) -> Result<String, std::io::Error> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
fn rms(samples: &[f32]) -> f64 {
    (samples.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: math_student_compare <dataset-folder> <fragment-bank> <student-export> <new-output> [--validation] [training-dataset-folders...]".into());
    }
    let validation = args[5..].iter().any(|a| a == "--validation");
    let dataset_root = Path::new(&args[1]);
    let output = Path::new(&args[4]);
    if output.exists() {
        return Err("output already exists".into());
    }
    let dataset: Dataset =
        serde_json::from_slice(&std::fs::read(dataset_root.join("dataset.json"))?)?;
    let expressions: Vec<Expression> =
        serde_json::from_slice(&std::fs::read(dataset_root.join("expressions.json"))?)?;
    let expressions: BTreeMap<_, _> = expressions.into_iter().map(|e| (e.id.clone(), e)).collect();
    let mut seen = BTreeSet::new();
    let mut test = Vec::new();
    for record in &dataset.utterances {
        if !seen.insert(&record.text) {
            return Err("duplicate spoken text across dataset".into());
        }
        if (validation && matches!(record.split, Split::Validation))
            || (!validation && matches!(record.split, Split::Test))
        {
            test.push(record);
        }
    }
    if test.is_empty() {
        return Err("dataset has no examples in the selected held-out split".into());
    }
    let mut lineage = Vec::new();
    for folder in args[5..].iter().filter(|a| a.as_str() != "--validation") {
        let path = Path::new(folder).join("dataset.json");
        let earlier: Dataset = serde_json::from_slice(&std::fs::read(&path)?)?;
        let used: BTreeSet<_> = earlier
            .utterances
            .iter()
            .filter(|u| {
                matches!(u.split, Split::Train)
                    || (!validation && matches!(u.split, Split::Validation))
            })
            .map(|u| &u.text)
            .collect();
        if test.iter().any(|u| used.contains(&u.text)) {
            return Err(
                format!("test reading overlaps training/validation lineage: {folder}").into(),
            );
        }
        lineage.push(serde_json::json!({"folder":folder,"dataset_sha256":sha(&path)?}));
    }
    let bank = MathVoice::load(Path::new(&args[2]))?;
    let mut student = MathStudent::load(Path::new(&args[3]))?;
    std::fs::create_dir_all(output)?;
    let mut rows = Vec::new();
    for (i, record) in test.iter().enumerate() {
        let e = expressions.get(&record.id).ok_or("missing expression")?;
        if words(&e.expression)?.join(" ") != record.text || e.spoken != record.text {
            return Err("expression/transcript mismatch".into());
        }
        let teacher_path = dataset_root.join(&record.audio).canonicalize()?;
        if !teacher_path.starts_with(dataset_root.canonicalize()?) {
            return Err("audio path escapes dataset".into());
        }
        let teacher = local_tts_core::read_wav(&teacher_path).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let fragments = bank.expression(&e.expression)?;
        let bank_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let neural = student.synthesize(&SynthesisRequest {
            text: &e.expression,
            language: Language::En,
            speaker: "default",
            duration_scale: 1.0,
        })?;
        let student_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut files = BTreeMap::new();
        let mut durations = BTreeMap::new();
        let mut levels = BTreeMap::new();
        for (name, audio) in [
            ("teacher", teacher),
            ("fragments", fragments),
            ("student", neural),
        ] {
            let file = format!("{i:02}-{name}.wav");
            audio
                .write_wav(&output.join(&file))
                .map_err(|e| e.to_string())?;
            durations.insert(name, audio.samples.len() as f64 / audio.sample_rate as f64);
            levels.insert(name, rms(&audio.samples));
            files.insert(name, file);
        }
        rows.push(serde_json::json!({"id":record.id,"expression":e.expression,"spoken":record.text,"files":files,"seconds":durations,"rms":levels,"fragment_ms":bank_ms,"student_ms":student_ms,"teacher_audio_sha256":sha(&teacher_path)?,"perceptual_evaluation":"pending"}));
    }
    let report = serde_json::json!({"dataset_sha256":sha(&dataset_root.join("dataset.json"))?,"student_model_sha256":sha(&Path::new(&args[3]).join("model.onnx"))?,"fragment_manifest_sha256":sha(&Path::new(&args[2]).join("math-voice.json"))?,"schema_version":2,"split":if validation {"validation"} else {"test"},"examples":rows,"audited_training_lineage":lineage,"metrics_note":"Duration, RMS and latency do not establish intelligibility or naturalness. Held-out split independence is checked within this dataset; training lineage must also be reviewed."});
    std::fs::write(
        output.join("comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Arithmetic voice comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#19202b}section{border-top:1px solid #ccd3dd;padding:16px 0}.voices{display:flex;gap:16px;flex-wrap:wrap}audio{width:300px}small{color:#526071}</style><h1>Arithmetic voice comparison</h1><p>Listen for missing or changed numbers, operator words, joins and naturalness. These are raw outputs; the student remains experimental. Loudness differences are not quality scores.</p>",
    );
    for row in report["examples"].as_array().unwrap() {
        html.push_str(&format!(
            "<section><h2>{}</h2><p>{}</p><div class=\"voices\">",
            escape(row["expression"].as_str().unwrap()),
            escape(row["spoken"].as_str().unwrap())
        ));
        for name in ["teacher", "fragments", "student"] {
            html.push_str(&format!("<div><p>{name}</p><audio controls preload=\"none\" src=\"{}\"></audio><p><small>{:.2} seconds</small></p></div>",escape(row["files"][name].as_str().unwrap()),row["seconds"][name].as_f64().unwrap()));
        }
        html.push_str("</div></section>");
    }
    html.push_str("</html>");
    std::fs::write(output.join("comparison.html"), html)?;
    println!("Exported {} reserved held-out triples", test.len());
    Ok(())
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

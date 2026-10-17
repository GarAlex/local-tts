//! Export identical readings from the fragment baseline and neural teacher.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    kokoro::Kokoro,
    math_speech::{MathVoice, words},
};
use std::{path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: math_compare <bank-folder> <kokoro-folder> <new-output-folder>".into());
    }
    let root = Path::new(&args[3]);
    if root.exists() {
        return Err("output already exists".into());
    }
    let bank = MathVoice::load(Path::new(&args[1]))?;
    let mut teacher = Kokoro::load(Path::new(&args[2]))?;
    std::fs::create_dir_all(root)?;
    let mut rows = Vec::new();
    for (i, expression) in [
        "0 + 0 = 0",
        "7 + 18 = 25",
        "99 + 1 = 100",
        "1000 - 1 = 999",
        "-12 + 225 = 213",
        "12 + 225 = 237",
        "0.005 × 20 = 0.1",
        "(23 + 4) / 9 = 3",
        "123456789 - 89 = 123456700",
        "1 + 1 = 3",
    ]
    .iter()
    .enumerate()
    {
        let spoken = words(expression)?.join(" ");
        let start = Instant::now();
        let fragments = bank.expression(expression)?;
        let bank_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let neural = teacher.synthesize(&SynthesisRequest {
            text: &spoken,
            language: Language::En,
            speaker: "af_heart",
            duration_scale: 1.0,
        })?;
        let teacher_ms = start.elapsed().as_secs_f64() * 1000.0;
        let bank_file = format!("{i:02}-fragments.wav");
        let teacher_file = format!("{i:02}-teacher.wav");
        fragments
            .write_wav(&root.join(&bank_file))
            .map_err(|e| e.to_string())?;
        neural
            .write_wav(&root.join(&teacher_file))
            .map_err(|e| e.to_string())?;
        rows.push(serde_json::json!({"expression":expression,"spoken":spoken,"fragments":bank_file,"teacher":teacher_file,"fragment_ms":bank_ms,"teacher_ms":teacher_ms,"fragment_seconds":fragments.samples.len() as f64/fragments.sample_rate as f64,"teacher_seconds":neural.samples.len() as f64/neural.sample_rate as f64,"listening_status":"pending"}));
    }
    std::fs::write(
        root.join("comparison.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    println!(
        "Exported {} comparison pairs to {}",
        rows.len(),
        root.display()
    );
    Ok(())
}

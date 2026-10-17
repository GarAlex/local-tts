//! Native synthesis and effects without a WAV roundtrip or inference service.
use local_tts_core::{
    Language, SynthesisRequest, Synthesizer,
    effects::{VoiceEffects, process},
    kokoro::Kokoro,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(3..=5).contains(&args.len()) {
        return Err("usage: voice_effects <kokoro-folder> <new-output-folder> <text> [pitch-semitones] [formant-semitones]".into());
    }
    let settings = VoiceEffects {
        pitch_semitones: args.get(3).map_or(Ok(-4.0), |x| x.parse())?,
        formant_semitones: args.get(4).map_or(Ok(-2.0), |x| x.parse())?,
        ..Default::default()
    };
    settings.validate().map_err(std::io::Error::other)?;
    let mut engine = Kokoro::load(Path::new(&args[0]))?;
    let original = engine.synthesize(&SynthesisRequest {
        text: &args[2],
        language: Language::En,
        speaker: "af_heart",
        duration_scale: 1.0,
    })?;
    let (processed, stats) = process(&original, settings).map_err(std::io::Error::other)?;
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?; // Refuse to overwrite an existing output directory.
    original.write_wav(&output.join("original.wav"))?;
    processed.write_wav(&output.join("processed.wav"))?;
    let report = serde_json::json!({
        "sample_rate": original.sample_rate, "samples": original.samples.len(),
        "pitch_semitones": settings.pitch_semitones, "formant_semitones": settings.formant_semitones,
        "input_peak": stats.input_peak, "output_peak": stats.output_peak, "peak_gain": stats.peak_gain,
    });
    std::fs::write(
        output.join("metrics.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "Saved native original and processed clips to {}",
        output.display()
    );
    Ok(())
}

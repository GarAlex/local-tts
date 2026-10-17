use local_tts_core::{Language, SynthesisRequest, Synthesizer, kokoro::Kokoro};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: speak <voice-folder> <output.wav>".into());
    }
    let mut engine = Kokoro::load(Path::new(&args[1]))?;
    let audio = engine.synthesize(&SynthesisRequest {
        text: "Hello world.",
        language: Language::En,
        speaker: "af_heart",
        duration_scale: 1.0,
    })?;
    audio.write_wav(Path::new(&args[2]))?;
    Ok(())
}

use local_tts_core::{kokoro::Kokoro, math_speech::MathVoice};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: math_pack <kokoro-folder> <new-output-folder> <test.wav>".into());
    }
    let mut teacher = Kokoro::load(Path::new(&args[1]))?;
    let size = MathVoice::build(
        Path::new(&args[2]),
        &mut teacher,
        "af_heart",
        "Kokoro af_heart; retain Apache-2.0 teacher license and pinned model provenance",
    )?;
    let voice = MathVoice::load(Path::new(&args[2]))?;
    voice
        .expression("12 + 225 = 237")?
        .write_wav(Path::new(&args[3]))
        .map_err(|e| e.to_string())?;
    println!("Math voice runtime data: {size} bytes");
    Ok(())
}

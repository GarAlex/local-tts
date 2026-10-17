fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: math_sequence_verify <sequence-voice>")?;
    let path = std::path::Path::new(&root);
    let mut voice = local_tts_core::math_sequence::MathSequence::load(path)?;
    println!("Native continuous sequence parity: {}", voice.verify(path)?);
    Ok(())
}

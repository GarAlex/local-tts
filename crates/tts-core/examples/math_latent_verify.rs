use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: math_latent_verify <voice-folder>")?;
    let mut model = local_tts_core::math_latent::MathLatent::load(Path::new(&root))?;
    println!(
        "Native latent decoder parity passed; maximum error {}",
        model.verify(Path::new(&root))?
    );
    Ok(())
}

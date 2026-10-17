//! Developer-only shape/finite checks for standalone decoder components.
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args()
        .nth(1)
        .ok_or("usage: math_decoder_probe <preflight-root>")?;
    for language in ["de", "es", "fr", "it", "pt", "pl", "nl"] {
        let folder = Path::new(&root).join(language);
        let hashes: BTreeMap<String, String> =
            serde_json::from_slice(&std::fs::read(folder.join("checksums.json"))?)?;
        let path = folder.join("decoder.onnx");
        let digest = format!("{:x}", Sha256::digest(std::fs::read(&path)?));
        if hashes.get("decoder.onnx") != Some(&digest) {
            return Err("decoder checksum differs".into());
        }
        let mut session = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(&path)?;
        for frames in [17, 32, 73] {
            let latent = vec![0.0f32; 192 * frames];
            let output = session.run(
                ort::inputs!["latent" => Tensor::from_array(([1usize, 192, frames], latent))?],
            )?;
            let (_, audio) = output[0].try_extract_tensor::<f32>()?;
            if audio.len() != frames * 256 || audio.iter().any(|v| !v.is_finite()) {
                return Err("invalid decoder output".into());
            }
        }
        println!("{language}: SHA and three native dynamic-length checks passed");
    }
    Ok(())
}

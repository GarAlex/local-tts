use crate::{Audio, Language, SynthesisRequest, Synthesizer, english::English};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use std::{collections::HashMap, path::Path};

#[derive(Debug)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

/// Published ONNX variants. Quantization covers selected operations, not every tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    F32,
    Q8,
    Q4,
}
impl Variant {
    pub const ALL: [Self; 3] = [Self::F32, Self::Q8, Self::Q4];
    pub fn id(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "FP32",
            Self::Q8 => "Q8",
            Self::Q4 => "Q4 (mixed FP32)",
        }
    }
    pub fn folder(self) -> &'static str {
        match self {
            Self::F32 => "kokoro-en",
            Self::Q8 => "kokoro-en-q8",
            Self::Q4 => "kokoro-en-q4",
        }
    }
    fn from_digest(digest: &str) -> Result<Self, String> {
        let variants: serde_json::Value =
            serde_json::from_str(include_str!("../../../assets/kokoro-variants.json"))
                .map_err(|e| e.to_string())?;
        Self::ALL
            .into_iter()
            .find(|variant| variants[variant.id()]["sha256"].as_str() == Some(digest))
            .ok_or_else(|| "checksum mismatch for model.onnx; unsupported Kokoro variant".into())
    }
}

pub struct Kokoro {
    variant: Variant,
    session: Session,
    vocab: HashMap<char, i64>,
    styles: Vec<f32>,
    frontend: English,
}

impl Kokoro {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let assets: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/english-assets.json"))?;
            let variant = Variant::from_digest(
                &crate::integrity::hash_file(root, "model.onnx").map_err(std::io::Error::other)?,
            )
            .map_err(std::io::Error::other)?;
            let mut hashes = std::collections::BTreeMap::new();
            for name in ["tokenizer.json", "af_heart.bin"] {
                hashes.insert(
                    name.into(),
                    assets[name]["sha256"]
                        .as_str()
                        .ok_or("missing pinned checksum")?
                        .to_string(),
                );
            }
            crate::integrity::verify(root, &hashes, &["tokenizer.json", "af_heart.bin"])
                .map_err(std::io::Error::other)?;
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("tokenizer.json"))?)?;
            let mut vocab = HashMap::new();
            for (key, id) in config["model"]["vocab"]
                .as_object()
                .ok_or("missing vocab")?
            {
                let mut chars = key.chars();
                let c = chars.next().ok_or("empty vocab entry")?;
                if chars.next().is_some() {
                    return Err("non-character vocab entry".into());
                }
                vocab.insert(c, id.as_i64().ok_or("invalid token id")?);
            }
            let bytes = std::fs::read(root.join("af_heart.bin"))?;
            if bytes.len() % (256 * 4) != 0 || bytes.len() < 256 * 4 {
                return Err("invalid voice style data".into());
            }
            let styles = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            Ok(Self {
                variant,
                session: Session::builder()?
                    .with_intra_threads(2)?
                    .commit_from_file(root.join("model.onnx"))?,
                vocab,
                styles,
                frontend: English::load(&root.join("cmudict.dict"))
                    .map_err(std::io::Error::other)?,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    pub fn phonemize(&self, text: &str) -> Result<String, Error> {
        self.frontend.phonemize(text).map_err(Error)
    }

    /// Adds CMU-format user pronunciations; see [`English::add_lexicon`].
    pub fn add_lexicon(&mut self, path: &Path) -> Result<usize, Error> {
        self.frontend.add_lexicon(path).map_err(Error)
    }

    pub fn synthesize_phonemes(
        &mut self,
        phonemes: &str,
        duration_scale: f32,
    ) -> Result<Audio, Error> {
        if !duration_scale.is_finite() || !(0.5..=2.0).contains(&duration_scale) {
            return Err(Error("duration scale must be 0.5–2.0".into()));
        }
        let tokens: Vec<i64> = phonemes
            .chars()
            .map(|c| {
                self.vocab
                    .get(&c)
                    .copied()
                    .ok_or_else(|| Error(format!("unsupported phoneme '{c}'")))
            })
            .collect::<Result<_, _>>()?;
        if tokens.is_empty() || tokens.len() > 510 {
            return Err(Error(
                "phoneme input must contain 1–510 characters; shorten the sentence".into(),
            ));
        }
        let row = tokens.len() - 1;
        let style = self
            .styles
            .get(row * 256..(row + 1) * 256)
            .ok_or_else(|| Error("voice style row unavailable".into()))?
            .to_vec();
        let mut padded = vec![0];
        padded.extend(tokens);
        padded.push(0);
        let run = || -> Result<Audio, Box<dyn std::error::Error>> {
            let outputs = self.session.run(ort::inputs![
                "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, padded.len()), padded)?)?,
                "style" => Tensor::from_array(Array2::from_shape_vec((1, 256), style)?)?,
                "speed" => Tensor::from_array(([1usize], vec![1.0 / duration_scale]))?
            ])?;
            let (_, samples) = outputs[0].try_extract_tensor::<f32>()?;
            if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
                return Err("model produced invalid audio".into());
            }
            Ok(Audio {
                sample_rate: 24000,
                samples: samples.to_vec(),
            })
        };
        run().map_err(|e| Error(e.to_string()))
    }
}

impl Synthesizer for Kokoro {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Error> {
        self.synthesize_cancellable(request, &std::sync::atomic::AtomicBool::new(false))
    }
}
impl Kokoro {
    /// Observe cancellation before and after each model invocation. Partial audio is discarded.
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Audio, Error> {
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        if request.language != Language::En || request.speaker != "af_heart" {
            return Err(Error(
                "this package supports English / af_heart only".into(),
            ));
        }
        if request.text.chars().count() > 10000 {
            return Err(Error("text exceeds 10000 characters".into()));
        }
        let phonemes = self.phonemize(request.text)?;
        let chunks = crate::paragraph::chunks(&phonemes, 510).map_err(Error)?;
        let mut output = None;
        for chunk in chunks {
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
            crate::paragraph::append(
                &mut output,
                self.synthesize_phonemes(&chunk, request.duration_scale)?,
            )
            .map_err(Error)?;
        }
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        output.ok_or_else(|| Error("enter text".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_pinned_variants_are_recognized() {
        let variants: serde_json::Value =
            serde_json::from_str(include_str!("../../../assets/kokoro-variants.json")).unwrap();
        for variant in Variant::ALL {
            assert_eq!(
                Variant::from_digest(variants[variant.id()]["sha256"].as_str().unwrap()).unwrap(),
                variant
            );
        }
        assert!(Variant::from_digest(&"0".repeat(64)).is_err());
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("model.onnx"),
            b"unsupported or damaged model",
        )
        .unwrap();
        assert!(
            Kokoro::load(root.path())
                .err()
                .unwrap()
                .to_string()
                .contains("checksum mismatch")
        );
    }
}

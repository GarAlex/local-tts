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

/// The voice every Kokoro package has; the others are optional files.
pub const DEFAULT_VOICE: &str = "af_heart";

/// One pinned English voice style (`<name>.bin` in a package folder).
#[derive(Clone, Debug)]
pub struct KokoroVoice {
    pub name: String,
    /// `us` or `gb`. The text frontend is American for both.
    pub accent: String,
    pub gender: String,
    /// Upstream's own listening grade, A best.
    pub grade: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
}

/// Every pinned English voice, the default first, then by grade order of
/// the upstream list.
pub fn voices() -> Vec<KokoroVoice> {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("../../../assets/kokoro-voices.json"))
            .expect("bundled Kokoro voice pins are valid");
    let mut voices: Vec<KokoroVoice> = doc["voices"]
        .as_object()
        .expect("voices object")
        .iter()
        .map(|(name, v)| KokoroVoice {
            name: name.clone(),
            accent: v["accent"].as_str().unwrap_or_default().into(),
            gender: v["gender"].as_str().unwrap_or_default().into(),
            grade: v["grade"].as_str().unwrap_or_default().into(),
            url: v["url"].as_str().unwrap_or_default().into(),
            sha256: v["sha256"].as_str().unwrap_or_default().into(),
            bytes: v["bytes"].as_u64().unwrap_or_default(),
        })
        .collect();
    voices.sort_by_key(|v| (v.name != DEFAULT_VOICE, v.accent != "us", v.name.clone()));
    voices
}

fn parse_style(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(256 * 4) || bytes.len() < 256 * 4 {
        return Err("invalid voice style data".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

pub struct Kokoro {
    variant: Variant,
    session: Session,
    vocab: HashMap<char, i64>,
    styles: HashMap<String, Vec<f32>>,
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
            let mut styles = HashMap::new();
            styles.insert(
                DEFAULT_VOICE.to_string(),
                parse_style(&std::fs::read(root.join("af_heart.bin"))?)
                    .map_err(std::io::Error::other)?,
            );
            // Further voices are optional; one that is present must match
            // its pin, so a tampered or truncated file never loads.
            for voice in voices().into_iter().filter(|v| v.name != DEFAULT_VOICE) {
                let file = format!("{}.bin", voice.name);
                if !root.join(&file).is_file() {
                    continue;
                }
                let pin = std::collections::BTreeMap::from([(file.clone(), voice.sha256.clone())]);
                crate::integrity::verify(root, &pin, &[file.as_str()])
                    .map_err(std::io::Error::other)?;
                styles.insert(
                    voice.name.clone(),
                    parse_style(&std::fs::read(root.join(&file))?)
                        .map_err(std::io::Error::other)?,
                );
            }
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

    /// The voices this package has installed, the default first.
    pub fn speakers(&self) -> Vec<String> {
        voices()
            .into_iter()
            .map(|v| v.name)
            .filter(|name| self.styles.contains_key(name))
            .collect()
    }

    pub fn phonemize(&self, text: &str) -> Result<String, Error> {
        self.frontend.phonemize_american(text).map_err(Error)
    }

    fn unknown_speaker(&self, speaker: &str) -> Error {
        if voices().iter().any(|v| v.name == speaker) {
            Error(format!(
                "Kokoro voice {speaker} is not installed in this package; reinstall it to add every pinned voice"
            ))
        } else {
            Error(format!(
                "unknown Kokoro voice {speaker}; installed: {}",
                self.speakers().join(", ")
            ))
        }
    }

    /// Adds CMU-format user pronunciations; see [`English::add_lexicon`].
    pub fn add_lexicon(&mut self, path: &Path) -> Result<usize, Error> {
        self.frontend.add_lexicon(path).map_err(Error)
    }

    /// Phonemes in the default voice.
    pub fn synthesize_phonemes(
        &mut self,
        phonemes: &str,
        duration_scale: f32,
    ) -> Result<Audio, Error> {
        self.synthesize_phonemes_as(DEFAULT_VOICE, phonemes, duration_scale)
    }

    pub fn synthesize_phonemes_as(
        &mut self,
        speaker: &str,
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
            .get(speaker)
            .ok_or_else(|| self.unknown_speaker(speaker))?
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
        if request.language != Language::En {
            return Err(Error("Kokoro packages support English only".into()));
        }
        if !self.styles.contains_key(request.speaker) {
            return Err(self.unknown_speaker(request.speaker));
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
                self.synthesize_phonemes_as(request.speaker, &chunk, request.duration_scale)?,
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

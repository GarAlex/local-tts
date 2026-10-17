//! Continuous numeric student inference: phonemes → context/durations → latents → waveform.
use crate::{Audio, Language, kokoro::Error};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool};
#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    architecture: String,
    language: Language,
    sample_rate: u32,
    channels: usize,
    hidden: usize,
    hop: usize,
    vocabulary: i64,
}
#[derive(Deserialize)]
struct Lexicon {
    words: BTreeMap<String, Vec<i64>>,
}
pub struct MathSequence {
    pub language: Language,
    words: BTreeMap<String, Vec<i64>>,
    encoder: Session,
    latent: Session,
    decoder: Session,
}
impl MathSequence {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            for (file, limit) in [
                ("sequence.json", 65536),
                ("checksums.json", 65536),
                ("words.json", 100000),
                ("reference.json", 65536),
                ("reference.bin", 2000000),
                ("encoder.onnx", 10000000),
                ("latent.onnx", 10000000),
                ("decoder.onnx", 60000000),
            ] {
                if std::fs::metadata(root.join(file))?.len() > limit {
                    return Err("sequence package component too large".into());
                }
            }
            let hashes = serde_json::from_slice(&std::fs::read(root.join("checksums.json"))?)?;
            crate::integrity::verify(
                root,
                &hashes,
                &[
                    "sequence.json",
                    "words.json",
                    "encoder.onnx",
                    "latent.onnx",
                    "decoder.onnx",
                    "reference.json",
                    "reference.bin",
                ],
            )
            .map_err(std::io::Error::other)?;
            let manifest: Manifest =
                serde_json::from_slice(&std::fs::read(root.join("sequence.json"))?)?;
            let vocabulary = match manifest.language {
                Language::Ru => 62,
                Language::Uk => 256,
                _ => return Err("sequence voice requires ru or uk".into()),
            };
            if manifest.schema_version != 1
                || manifest.architecture != "arithmetic-sequence-v1"
                || manifest.sample_rate != 22050
                || manifest.channels != 192
                || manifest.hidden != 96
                || manifest.hop != 256
                || manifest.vocabulary != vocabulary
            {
                return Err("invalid sequence voice contract".into());
            }
            let lexicon: Lexicon =
                serde_json::from_slice(&std::fs::read(root.join("words.json"))?)?;
            let required = crate::math_speech::vocabulary_for_language(manifest.language)
                .map_err(std::io::Error::other)?;
            if lexicon.words.len() != required.len()
                || required
                    .iter()
                    .any(|word| !lexicon.words.contains_key(word))
            {
                return Err("numeric pronunciation vocabulary differs".into());
            }
            for ids in lexicon.words.values() {
                if ids.len() < 5
                    || ids.len() > 512
                    || ids.len() % 2 != 1
                    || ids[..2] != [1, 0]
                    || ids[ids.len() - 2..] != [0, 2]
                    || ids.iter().any(|&id| id < 0 || id >= vocabulary)
                    || ids.iter().skip(1).step_by(2).any(|&id| id != 0)
                {
                    return Err("invalid native numeric token boundaries".into());
                }
            }
            let session = |file: &str| -> Result<Session, ort::Error> {
                Session::builder()?
                    .with_intra_threads(2)?
                    .commit_from_file(root.join(file))
            };
            Ok(Self {
                language: manifest.language,
                words: lexicon.words,
                encoder: session("encoder.onnx")?,
                latent: session("latent.onnx")?,
                decoder: session("decoder.onnx")?,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    pub fn synthesize_words(
        &mut self,
        words: &[String],
        scale: f32,
        cancel: &AtomicBool,
    ) -> Result<Audio, Error> {
        let mut run = || -> Result<Audio, Box<dyn std::error::Error>> {
            crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
            if words.is_empty()
                || words.len() > 512
                || !scale.is_finite()
                || !(0.5..=2.0).contains(&scale)
            {
                return Err("invalid sequence words or duration scale".into());
            }
            let mut ids = vec![1i64, 0];
            for (i, word) in words.iter().enumerate() {
                let part = self
                    .words
                    .get(word)
                    .ok_or("unknown numeric pronunciation")?;
                if i > 0 {
                    ids.extend([3, 0]);
                }
                ids.extend_from_slice(&part[2..part.len() - 1]);
                if ids.len() > 1023 {
                    return Err("numeric input exceeds 1024 phoneme tokens".into());
                }
            }
            ids.push(2);
            let count = ids.len();
            let encoded = self
                .encoder
                .run(ort::inputs!["tokens"=>Tensor::from_array(([1usize,count],ids))?])?;
            if encoded.len() != 2 {
                return Err("invalid encoder output count".into());
            }
            let (hidden_shape, hidden) = encoded[0].try_extract_tensor::<f32>()?;
            let (duration_shape, duration) = encoded[1].try_extract_tensor::<f32>()?;
            if hidden_shape.iter().copied().ne([1, 96, count as i64])
                || duration_shape.iter().copied().ne([1, count as i64])
                || hidden.iter().any(|v| !v.is_finite())
                || duration
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0 || *v > 128.001)
            {
                return Err("invalid encoder shapes or values".into());
            }
            let durations: Vec<usize> = duration
                .iter()
                .map(|v| (v * scale).round_ties_even().clamp(1.0, 256.0) as usize)
                .collect();
            let frames: usize = durations.iter().sum();
            if frames == 0 || frames > 7740 {
                return Err("predicted speech exceeds frame limit".into());
            }
            let mut features = vec![0.0f32; 98 * frames];
            let mut offset = 0;
            for (token, &length) in durations.iter().enumerate() {
                for channel in 0..96 {
                    features[channel * frames + offset..channel * frames + offset + length]
                        .fill(hidden[channel * count + token]);
                }
                for frame in 0..length {
                    let progress = frame as f32 / (length - 1).max(1) as f32;
                    features[96 * frames + offset + frame] = progress;
                    features[97 * frames + offset + frame] =
                        (progress * std::f32::consts::PI).sin();
                }
                offset += length;
            }
            drop(encoded);
            crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
            let predicted = self.latent.run(
                ort::inputs!["features"=>Tensor::from_array(([1usize,98,frames],features))?],
            )?;
            if predicted.len() != 1 {
                return Err("invalid latent output count".into());
            }
            let (shape, values) = predicted[0].try_extract_tensor::<f32>()?;
            if shape.iter().copied().ne([1, 192, frames as i64])
                || values.iter().any(|v| !v.is_finite())
            {
                return Err("invalid continuous acoustic output".into());
            }
            let latent = values.to_vec();
            drop(predicted);
            crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
            let decoded = self
                .decoder
                .run(ort::inputs!["latent"=>Tensor::from_array(([1usize,192,frames],latent))?])?;
            if decoded.len() != 1 {
                return Err("invalid decoder output count".into());
            }
            let (shape, audio) = decoded[0].try_extract_tensor::<f32>()?;
            if shape.iter().copied().ne([1, 1, (frames * 256) as i64])
                || audio.iter().any(|v| !v.is_finite())
            {
                return Err("invalid sequence waveform".into());
            }
            crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
            Ok(Audio {
                sample_rate: 22050,
                samples: audio.to_vec(),
            })
        };
        run().map_err(|e| Error(e.to_string()))
    }
    pub fn verify(&mut self, root: &Path) -> Result<f32, Error> {
        #[derive(Deserialize)]
        struct Reference {
            expression: String,
            duration_scale: f32,
            offset: usize,
            count: usize,
        }
        let mut verify = || -> Result<f32, Box<dyn std::error::Error>> {
            let references: Vec<Reference> =
                serde_json::from_slice(&std::fs::read(root.join("reference.json"))?)?;
            let bank = std::fs::read(root.join("reference.bin"))?;
            if references.is_empty() || references.len() > 16 {
                return Err("invalid reference count".into());
            }
            let mut error = 0.0f32;
            for reference in references {
                let size = reference
                    .count
                    .checked_mul(4)
                    .ok_or("reference size overflow")?;
                let end = reference
                    .offset
                    .checked_add(size)
                    .ok_or("reference offset overflow")?;
                if reference.offset % 4 != 0 || size == 0 {
                    return Err("invalid reference geometry".into());
                }
                let expected = bank
                    .get(reference.offset..end)
                    .ok_or("reference outside bank")?;
                let words =
                    crate::math_speech::words_for_language(&reference.expression, self.language)
                        .map_err(std::io::Error::other)?;
                let audio = self.synthesize_words(
                    &words,
                    reference.duration_scale,
                    &AtomicBool::new(false),
                )?;
                if audio.samples.len() != reference.count {
                    return Err("native sequence reference length differs".into());
                }
                for (a, b) in audio.samples.iter().zip(expected.chunks_exact(4)) {
                    let b = f32::from_le_bytes(b.try_into()?);
                    if !b.is_finite() {
                        return Err("nonfinite sequence reference".into());
                    }
                    error = error.max((a - b).abs());
                }
            }
            if error > 1e-3 {
                return Err(format!("native sequence parity failed: {error}").into());
            }
            Ok(error)
        };
        verify().map_err(|e| Error(e.to_string()))
    }
}

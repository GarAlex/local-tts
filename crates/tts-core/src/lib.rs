//! Offline synthesis, voice training contracts and dataset authoring.

#[cfg(feature = "download")]
pub mod catalog;
pub mod chatterbox;
pub mod clips;
pub mod collection;
pub mod compact_english;
pub mod context;
pub mod contextual;
pub mod css10;
pub mod cyrillic_numbers;
pub mod dataset;
pub mod effects;
pub mod english;
pub mod english_numbers;
pub mod evaluation;
pub mod frontend;
pub mod g2p;
pub mod integrity;
pub mod kokoro;
pub mod languages;
pub mod math_latent;
pub mod math_sequence;
pub mod math_speech;
pub mod neon;
pub mod paragraph;
pub mod piper;
pub mod piper_training;
pub mod prompts;
pub mod pronunciations;
pub mod qwen_design;
pub mod text;
mod text_units;
pub mod training;
pub mod vits;
pub mod voice;
pub mod voice_profile;
pub mod vosk;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    En,
    Ru,
    Uk,
    De,
    Fr,
    Es,
    It,
    Pt,
    Pl,
    Nl,
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest<'a> {
    pub text: &'a str,
    pub language: Language,
    pub speaker: &'a str,
    /// Duration multiplier: values greater than one produce slower speech.
    pub duration_scale: f32,
}

#[derive(Debug, Clone)]
pub struct Audio {
    pub sample_rate: u32,
    /// Mono PCM samples in [-1, 1].
    pub samples: Vec<f32>,
}

impl Audio {
    pub fn write_wav(
        &self,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.sample_rate == 0 || self.samples.iter().any(|s| !s.is_finite()) {
            return Err("invalid audio".into());
        }
        let mut writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: self.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for sample in &self.samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

pub fn read_wav(path: &std::path::Path) -> Result<Audio, Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate == 0 {
        return Err("expected mono WAV with nonzero sample rate".into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
        return Err("empty or invalid audio".into());
    }
    Ok(Audio {
        sample_rate: spec.sample_rate,
        samples,
    })
}

/// Implemented by native inference backends. The caller owns playback.
pub trait Synthesizer {
    type Error: std::error::Error + Send + Sync + 'static;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Self::Error>;
}

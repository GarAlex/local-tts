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
mod english_tolerant;
pub mod evaluation;
pub mod frontend;
pub mod g2p;
pub mod integrity;
pub mod kokoro;
pub mod languages;
pub mod math_latent;
pub mod math_sequence;
pub mod math_speech;
pub mod nemo;
pub mod neon;
pub mod onnx_output;
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

/// One spoken word of the text as the engine read it — normalized words
/// ("forty", "nine", "dollars"), not the caller's spelling — and when it
/// is said, in seconds from the start of the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpokenWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// Audio and the words in it, from the model's own duration predictor.
#[derive(Debug, Clone)]
pub struct TimedAudio {
    /// The same samples [`Synthesizer::synthesize`] returns for the request.
    pub audio: Audio,
    pub words: Vec<SpokenWord>,
    /// Seconds per duration unit the model predicted in.
    pub frame_seconds: f64,
}

/// A backend that can say when each word is spoken. Additive: requests
/// and [`Synthesizer`] are unchanged.
pub trait TimedSynthesizer: Synthesizer {
    fn synthesize_timed(
        &mut self,
        request: &SynthesisRequest<'_>,
    ) -> Result<TimedAudio, Self::Error>;
}

/// Word times from per-token durations: a word spans the frames of its
/// first to its last token. `words` are (word index, spoken text, token
/// range — first inclusive, end exclusive); integer frames times one
/// constant, offset by the chunk's start and held inside its audio.
pub(crate) fn words_from_durations(
    words: &[(usize, String, std::ops::Range<usize>)],
    durations: &[i64],
    frame_seconds: f64,
    offset_seconds: f64,
    limit_seconds: f64,
) -> Vec<(usize, SpokenWord)> {
    let mut cumulative = Vec::with_capacity(durations.len() + 1);
    let mut total = 0i64;
    cumulative.push(0i64);
    for d in durations {
        total += (*d).max(0);
        cumulative.push(total);
    }
    let at = |frames: i64| offset_seconds + (frames as f64 * frame_seconds).min(limit_seconds);
    words
        .iter()
        .filter(|(_, _, r)| r.start < r.end && r.end <= durations.len())
        .map(|(index, text, r)| {
            (
                *index,
                SpokenWord {
                    text: text.clone(),
                    start: at(cumulative[r.start]),
                    end: at(cumulative[r.end]),
                },
            )
        })
        .collect()
}

/// One word per spoken word: a word whose phones two chunks split (a
/// spelled acronym's letters may be) runs from its first start to its last
/// end.
pub(crate) fn merge_split_words(words: Vec<(usize, SpokenWord)>) -> Vec<SpokenWord> {
    let mut out: Vec<(usize, SpokenWord)> = Vec::with_capacity(words.len());
    for (index, word) in words {
        match out.last_mut() {
            Some((last, previous)) if *last == index => previous.end = word.end,
            _ => out.push((index, word)),
        }
    }
    out.into_iter().map(|(_, w)| w).collect()
}

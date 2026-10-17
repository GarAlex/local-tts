//! Local multilingual inference using pinned MIT Chatterbox ONNX exports.
//! Q4 language model, optional speaker reference, bounded nucleus sampling.
use crate::{Audio, Language, SynthesisRequest, Synthesizer, kokoro::Error};
use ndarray::{Array2, Array3, Axis, concatenate};
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use rand::{Rng, SeedableRng};
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
    sync::atomic::AtomicBool,
};
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

const START: i64 = 6561;
const STOP: i64 = 6562;
const MAX_STEPS: usize = 750;
// Row count in the pinned embed_tokens graph; tokenizer metadata includes extra IDs.
const TEXT_VOCAB: i64 = 2352;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct Chatterbox {
    tokenizer: Tokenizer,
    embed: Session,
    language_model: Session,
    decoder: Session,
    conditioning: Array3<f32>,
    prompt: Vec<i64>,
    speaker: Array2<f32>,
    features: Array3<f32>,
    exaggeration: f32,
}

impl Chatterbox {
    pub fn load(root: &Path) -> std::result::Result<Self, Error> {
        Self::load_with_reference(root, None)
    }

    /// Load Chatterbox with an optional speaker reference recording.
    ///
    /// The reference is encoded once during model loading and is then used for
    /// every synthesis call. `None` keeps the pinned publisher reference voice.
    /// Custom references must be mono, 24 kHz WAV files no longer than 30 s.
    pub fn load_with_reference(
        root: &Path,
        reference: Option<&Path>,
    ) -> std::result::Result<Self, Error> {
        let default_reference = root.join("default_voice.wav");
        // Validate and bound user audio before allocating model sessions.
        let audio = read_reference(reference.unwrap_or(&default_reference))
            .map_err(|e| Error(e.to_string()))?;
        let load = || -> Result<Self> {
            let pins: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/chatterbox-assets.json"))?;
            let hashes: BTreeMap<String, String> = pins["files"]
                .as_object()
                .ok_or("invalid pins")?
                .iter()
                .map(|(name, record)| {
                    Ok((
                        name.clone(),
                        record["sha256"].as_str().ok_or("missing hash")?.to_owned(),
                    ))
                })
                .collect::<Result<_>>()?;
            crate::integrity::verify(root, &hashes, &[])?;
            let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))?;
            let mut encoder = session(root, "speech_encoder")?;
            let output=encoder.run(ort::inputs!["audio_values"=>Tensor::from_array(([1,audio.samples.len()],audio.samples))?])?;
            let conditioning = output["audio_features"]
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality()?;
            let prompt = output["audio_tokens"]
                .try_extract_tensor::<i64>()?
                .1
                .to_vec();
            let speaker = output["speaker_embeddings"]
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality()?;
            let features = output["speaker_features"]
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality()?;
            drop(output);
            drop(encoder);
            Ok(Self {
                tokenizer,
                conditioning,
                prompt,
                speaker,
                features,
                exaggeration: 0.5,
                embed: session(root, "embed_tokens")?,
                language_model: session(root, "language_model_q4")?,
                decoder: session(root, "conditional_decoder")?,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    /// Set the pinned model's emotion intensity input. Range 0–1, default 0.5.
    /// This does not select a mood or design a speaker identity. Higher values
    /// may speed speech up; quality depends on the reference and model.
    pub fn set_exaggeration(&mut self, value: f32) -> std::result::Result<(), Error> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Error(
                "expressiveness must be finite and between 0 and 1".into(),
            ));
        }
        self.exaggeration = value;
        Ok(())
    }
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &AtomicBool,
    ) -> std::result::Result<Audio, Error> {
        let run = || -> Result<Audio> {
            if request.language == Language::Uk {
                return Err(
                    "Chatterbox does not support Ukrainian; select the Ukrainian Piper backend"
                        .into(),
                );
            }
            if !matches!(request.speaker, "" | "default") {
                return Err(
                    "Chatterbox speaker must be default; choose a reference at engine load".into(),
                );
            }
            if request.duration_scale != 1.0 {
                return Err("Chatterbox currently requires duration scale 1.0".into());
            }
            crate::paragraph::check_cancel(cancel)?;
            let prepared = crate::text::prepare(request.text, request.language)?;
            let mut audio = None;
            for chunk in crate::paragraph::chunks(&prepared, 180)? {
                crate::paragraph::check_cancel(cancel)?;
                let next = self.chunk(&chunk, request.language, cancel)?;
                crate::paragraph::append(&mut audio, next)?;
            }
            crate::paragraph::check_cancel(cancel)?;
            audio.ok_or_else(|| "empty synthesis".into())
        };
        let mut run = run;
        run().map_err(|e| Error(e.to_string()))
    }
    fn chunk(&mut self, text: &str, language: Language, cancel: &AtomicBool) -> Result<Audio> {
        let mut ids = encode(&self.tokenizer, text, language)?;
        let mut positions: Vec<i64> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| if id >= START { 0 } else { i as i64 - 1 })
            .collect();
        let mut cache: Vec<(String, DynValue)> = Vec::new();
        for layer in 0..30 {
            for kind in ["key", "value"] {
                cache.push((
                    format!("past_key_values.{layer}.{kind}"),
                    Tensor::<f32>::new(&ort::memory::Allocator::default(), [1_i64, 16, 0, 64])?
                        .into_dyn(),
                ));
            }
        }
        let mut total = 0;
        let mut generated = Vec::new();
        let mut seen = HashSet::from([START]);
        let mut finished = false;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        for step in 0..MAX_STEPS {
            crate::paragraph::check_cancel(cancel)?;
            let mut embedding = self.embed.run(ort::inputs![
                "input_ids"=>Tensor::from_array(([1,ids.len()],ids))?,
                "position_ids"=>Tensor::from_array(([1,positions.len()],positions))?,
                "exaggeration"=>Tensor::from_array(([1],vec![self.exaggeration]))?
            ])?;
            let values = embedding
                .remove("inputs_embeds")
                .ok_or("missing text embeddings")?;
            let values = if step == 0 {
                let text = values
                    .try_extract_array::<f32>()?
                    .into_dimensionality::<ndarray::Ix3>()?;
                Tensor::from_array(concatenate(Axis(1), &[self.conditioning.view(), text])?)?
                    .into_dyn()
            } else {
                values
            };
            let (shape, _) = values.try_extract_tensor::<f32>()?;
            total += shape[1] as usize;
            cache.push(("inputs_embeds".into(), values));
            cache.push((
                "attention_mask".into(),
                Tensor::from_array(([1, total], vec![1_i64; total]))?.into_dyn(),
            ));
            let outputs = self.language_model.run(cache)?;
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            let scores = logits
                .get(logits.len().checked_sub(8194).ok_or("short logits")?..)
                .ok_or("invalid logits")?;
            let next = next_token(scores, &seen, rng.random())?;
            if next == STOP {
                finished = true;
                break;
            }
            // Speech vocabulary ends at 6560; a special/non-speech output fails explicitly.
            if next >= START {
                return Err("model generated an invalid speech token".into());
            }
            generated.push(next);
            seen.insert(next);
            cache = outputs
                .into_iter()
                .filter_map(|(name, value)| {
                    name.strip_prefix("present.")
                        .map(|suffix| (format!("past_key_values.{suffix}"), value))
                })
                .collect();
            if cache.len() != 60 {
                return Err("incomplete model attention cache".into());
            }
            ids = vec![next];
            positions = vec![step as i64 + 1];
        }
        if !finished || generated.is_empty() {
            return Err(
                "model did not finish within the token limit; try a shorter sentence".into(),
            );
        }
        crate::paragraph::check_cancel(cancel)?;
        let mut speech = self.prompt.clone();
        speech.extend(generated);
        let result = self.decoder.run(ort::inputs![
            "speech_tokens"=>Tensor::from_array(([1,speech.len()],speech))?,
            "speaker_embeddings"=>Tensor::from_array(self.speaker.clone())?,
            "speaker_features"=>Tensor::from_array(self.features.clone())?
        ])?;
        let (_, samples) = result["waveform"].try_extract_tensor::<f32>()?;
        if samples.is_empty()
            || samples.len() > 24000 * 60
            || samples.iter().any(|s| !s.is_finite())
        {
            return Err("invalid or excessive Chatterbox audio".into());
        }
        crate::paragraph::check_cancel(cancel)?;
        Ok(Audio {
            sample_rate: 24000,
            samples: samples.to_vec(),
        })
    }
}
/// Prepare a mono/stereo WAV at a supported sample rate for reference cloning.
/// Recording length is bounded before decoding; no automatic trimming is done.
pub fn prepare_reference(path: &Path) -> std::result::Result<Audio, Error> {
    let run = || -> Result<Audio> {
        let reader = hound::WavReader::open(path)?;
        let rate = reader.spec().sample_rate;
        if rate == 0
            || reader.duration() as u64 * 10 < u64::from(rate)
            || reader.duration() as u64 > u64::from(rate) * 30
        {
            return Err("choose a speech recording between 0.1 and 30 seconds".into());
        }
        drop(reader);
        let audio = crate::clips::import_wav_at_rate(path, 24000)?;
        if !audio.samples.iter().any(|s| s.abs() > 0.0001) {
            return Err("reference recording is silent".into());
        }
        Ok(audio)
    };
    run().map_err(|e| Error(e.to_string()))
}
/// Inspect the WAV header before decoding so invalid/oversized references never
/// allocate an unbounded sample buffer. Speech quality needs listening evaluation.
pub(crate) fn read_reference(path: &Path) -> Result<Audio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != 24000
        || reader.duration() < 2400
        || reader.duration() > 24000 * 30
    {
        return Err("reference must be mono 24 kHz WAV, 0.1 to 30 seconds".into());
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .take(720001)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .take(720001)
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    if samples.len() < 2400 || samples.len() > 720000 || samples.iter().any(|s| !s.is_finite()) {
        return Err("reference must contain finite PCM, 0.1 to 30 seconds".into());
    }
    let audio = Audio {
        sample_rate: spec.sample_rate,
        samples,
    };
    if !audio.samples.iter().any(|s| s.abs() > 0.0001) {
        return Err("reference must contain non-silent finite PCM".into());
    }
    Ok(audio)
}
fn session(root: &Path, name: &str) -> Result<Session> {
    Ok(Session::builder()?
        .with_intra_threads(2)?
        .commit_from_file(root.join("onnx").join(format!("{name}.onnx")))?)
}
fn encode(tokenizer: &Tokenizer, text: &str, language: Language) -> Result<Vec<i64>> {
    // Chatterbox's multilingual training frontend decomposes Unicode. Keep this
    // model-specific: project VITS graphemes deliberately retain NFC spellings.
    let text: String = text.nfkd().flat_map(char::to_lowercase).collect();
    if text.contains(['[', ']']) {
        return Err("reserved model tokens are not allowed in speech text".into());
    }
    let text = text
        .replace('’', "'")
        .replace(['“', '”'], "\"")
        .replace('…', ".");
    let encoded = tokenizer.encode(format!("[{}]{text}", language.code()), true)?;
    let ids: Vec<i64> = encoded.get_ids().iter().map(|&id| i64::from(id)).collect();
    if ids
        .iter()
        .any(|&id| id == 1 || (TEXT_VOCAB..START).contains(&id) || id > 6563)
        || ids.len() > 512
    {
        return Err("unsupported character or too many text tokens".into());
    }
    Ok(ids)
}
// Standard repetition penalty, temperature .8, min-p .05 and top-p .95.
// Use a fixed seed per chunk for a repeatable caller experience on one runtime.
fn next_token(scores: &[f32], seen: &HashSet<i64>, draw: f64) -> Result<i64> {
    if scores.is_empty() || scores.iter().any(|x| !x.is_finite()) || !(0.0..1.0).contains(&draw) {
        return Err("invalid model logits or sampling value".into());
    }
    let adjusted: Vec<f64> = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| {
            let score = if seen.contains(&(id as i64)) {
                if score < 0.0 {
                    score * 1.2
                } else {
                    score / 1.2
                }
            } else {
                score
            };
            f64::from(score) / 0.8
        })
        .collect();
    let max = adjusted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<(usize, f64)> = adjusted
        .iter()
        .enumerate()
        .map(|(i, s)| (i, (s - max).exp()))
        .filter(|(_, p)| *p >= 0.05)
        .collect();
    weights.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let total: f64 = weights.iter().map(|(_, p)| p).sum();
    let mut retained = 0.0;
    let count = weights
        .iter()
        .take_while(|(_, p)| {
            if retained >= total * 0.95 {
                return false;
            }
            retained += p;
            true
        })
        .count();
    let mut threshold = draw * retained;
    for &(id, p) in &weights[..count] {
        if threshold < p {
            return Ok(id as i64);
        }
        threshold -= p;
    }
    weights
        .get(count.saturating_sub(1))
        .map(|&(id, _)| id as i64)
        .ok_or_else(|| "empty sampling distribution".into())
}

impl Synthesizer for Chatterbox {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> std::result::Result<Audio, Error> {
        self.synthesize_cancellable(request, &AtomicBool::new(false))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepares_stereo_reference_without_changing_source() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for i in 0..44100 {
            let value = (8000.0 * (i as f32 * 0.03).sin()) as i16;
            writer.write_sample(value).unwrap();
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        let source = std::fs::read(&path).unwrap();
        let audio = prepare_reference(&path).unwrap();
        assert_eq!(audio.sample_rate, 24000);
        assert_eq!(audio.samples.len(), 24000);
        assert!(audio.samples.iter().all(|s| s.is_finite()));
        assert!(audio.samples.iter().any(|s| s.abs() > 0.1));
        assert_eq!(std::fs::read(&path).unwrap(), source);
        let output = dir.path().join("mono.wav");
        audio.write_wav(&output).unwrap();
        assert!(read_reference(&output).is_ok());
    }
    #[test]
    fn reference_validation_rejects_bad_headers_silence_and_levels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference.wav");
        for (channels, rate, length, value, accepted) in [
            (1, 24000, 2400, 0.2, true),
            (2, 24000, 4800, 0.2, false),
            (1, 22050, 2400, 0.2, false),
            (1, 24000, 2399, 0.2, false),
            (1, 24000, 720001, 0.2, false),
            (1, 24000, 2400, 0.0, false),
            (1, 24000, 2400, 1.1, true),
            (1, 24000, 2400, f32::NAN, false),
        ] {
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels,
                    sample_rate: rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for _ in 0..length {
                writer.write_sample(value).unwrap();
            }
            writer.finalize().unwrap();
            assert_eq!(read_reference(&path).is_ok(), accepted);
        }
        assert!(read_reference(&dir.path().join("missing.wav")).is_err());
    }
    #[test]
    #[ignore = "requires the downloaded Chatterbox tokenizer"]
    fn multilingual_tokenizer_preserves_model_unicode_contract() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/chatterbox-multilingual");
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
        for language in Language::ALL
            .into_iter()
            .filter(|l| l.is_european_extension())
        {
            for prompt in crate::prompts::PromptList::starter(language).lines {
                let ids = encode(&tokenizer, &prompt, language).unwrap();
                assert!(!ids.contains(&1));
                assert!(
                    ids.iter()
                        .all(|&id| id < TEXT_VOCAB || (6561..=6563).contains(&id))
                );
            }
        }
        assert_eq!(
            encode(&tokenizer, "École", Language::Fr).unwrap(),
            encode(&tokenizer, "E\u{301}cole", Language::Fr).unwrap()
        );
        assert!(encode(&tokenizer, "［fr］bonjour", Language::Fr).is_err());
    }
    #[test]
    fn sampling_penalty_thresholds_and_ties_are_stable() {
        assert_eq!(
            next_token(&[1.1, 1.2], &HashSet::from([1]), 0.0).unwrap(),
            0
        );
        assert_eq!(
            next_token(&[-1.1, -1.0], &HashSet::from([1]), 0.0).unwrap(),
            0
        );
        assert_eq!(next_token(&[1.0, 1.0], &HashSet::new(), 0.0).unwrap(), 0);
        assert_eq!(
            next_token(&[0.0, -100.0], &HashSet::new(), 0.999).unwrap(),
            0
        );
        assert_eq!(next_token(&[0.0, 0.0], &HashSet::new(), 0.75).unwrap(), 1);
        assert!(next_token(&[f32::NAN], &HashSet::new(), 0.0).is_err());
    }
}

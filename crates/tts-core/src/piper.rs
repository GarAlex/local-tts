//! Pinned Ukrainian text-based Piper voice; no Piper/eSpeak runtime dependency.
use crate::{Audio, Language, SynthesisRequest, Synthesizer, kokoro::Error};
use ndarray::Array2;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct Config {
    audio: AudioConfig,
    phoneme_type: String,
    phoneme_id_map: BTreeMap<String, Vec<i64>>,
    speaker_id_map: BTreeMap<String, i64>,
    inference: Inference,
}
#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}
#[derive(Deserialize)]
struct Inference {
    noise_scale: f32,
    noise_w: f32,
}

pub struct Piper {
    pronunciations: crate::pronunciations::Pronunciations,
    pronunciation_path: PathBuf,
    session: Session,
    config: Config,
}
impl Piper {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            let assets: serde_json::Value =
                serde_json::from_str(include_str!("../../../assets/ukrainian-assets.json"))?;
            let hashes = ["model.onnx", "config.json"]
                .into_iter()
                .map(|name| {
                    Ok((
                        name.to_string(),
                        assets[name]["sha256"]
                            .as_str()
                            .ok_or("missing pinned checksum")?
                            .to_string(),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
            crate::integrity::verify(root, &hashes, &["model.onnx", "config.json"])
                .map_err(std::io::Error::other)?;
            let config: Config = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
            if config.phoneme_type != "text" || config.audio.sample_rate != 22050 {
                return Err(
                    "unsupported Piper package; only the pinned Ukrainian text voice is supported"
                        .into(),
                );
            }
            let pronunciation_path = root.join(crate::pronunciations::FILE_NAME);
            let pronunciations =
                crate::pronunciations::PronunciationFile::open(&pronunciation_path, Language::Uk)
                    .map_err(std::io::Error::other)?
                    .data()
                    .clone();
            Ok(Self {
                pronunciations,
                pronunciation_path,
                session: Session::builder()?
                    .with_intra_threads(2)?
                    .commit_from_file(root.join("model.onnx"))?,
                config,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    pub fn speakers(&self) -> impl Iterator<Item = &str> {
        self.config.speaker_id_map.keys().map(String::as_str)
    }
    /// Lowercase NFD codepoints, BOS/padding/EOS as in Piper's text frontend.
    /// Unsupported characters fail explicitly instead of silently disappearing.
    /// Reload user corrections without reloading the neural model. Failed reloads retain old corrections.
    pub fn reload_pronunciations(&mut self) -> Result<(), Error> {
        let file =
            crate::pronunciations::PronunciationFile::open(&self.pronunciation_path, Language::Uk)
                .map_err(Error)?;
        self.set_pronunciations(file.data().clone())
    }
    /// Install validated in-memory corrections; does not change the saved file.
    pub fn set_pronunciations(
        &mut self,
        data: crate::pronunciations::Pronunciations,
    ) -> Result<(), Error> {
        if data.language() != Language::Uk {
            return Err(Error("pronunciation language mismatch".into()));
        }
        self.pronunciations = data;
        Ok(())
    }
    /// Return expanded text with saved corrections; explicit stress takes precedence.
    pub fn prepare_text(&self, text: &str) -> Result<String, Error> {
        let expanded = crate::text::expand(text, Language::Uk).map_err(Error)?;
        let corrected = self.pronunciations.apply(&expanded).map_err(Error)?;
        crate::context::stress(&corrected, Language::Uk).map_err(Error)
    }
    pub fn tokenize(&self, text: &str) -> Result<Vec<i64>, Error> {
        tokenize(&self.prepare_text(text)?, &self.config.phoneme_id_map)
    }
    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancel: &AtomicBool,
    ) -> Result<Audio, Error> {
        if request.language != Language::Uk {
            return Err(Error("this Piper voice supports Ukrainian only".into()));
        }
        let sid = *self
            .config
            .speaker_id_map
            .get(request.speaker)
            .ok_or_else(|| {
                Error("unknown Ukrainian speaker; use lada, mykyta or tetiana".into())
            })?;
        if !request.duration_scale.is_finite() || !(0.5..=2.0).contains(&request.duration_scale) {
            return Err(Error("duration scale must be 0.5–2.0".into()));
        }
        let normalized = self.prepare_text(request.text)?;
        let parts = crate::paragraph::chunks(&normalized, 240).map_err(Error)?;
        // Validate the complete input before running any model chunk.
        let tokens = parts
            .iter()
            .map(|part| self.tokenize(part))
            .collect::<Result<Vec<_>, _>>()?;
        let mut audio = None;
        for ids in tokens {
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
            let run = || -> Result<Audio, Box<dyn std::error::Error>> {
                let output = self.session.run(ort::inputs![
                    "input" => Tensor::from_array(Array2::from_shape_vec((1, ids.len()), ids.clone())?)?,
                    "input_lengths" => Tensor::from_array(([1usize], vec![ids.len() as i64]))?,
                    "scales" => Tensor::from_array(([3usize], vec![self.config.inference.noise_scale, request.duration_scale, self.config.inference.noise_w]))?,
                    "sid" => Tensor::from_array(([1usize], vec![sid]))?
                ])?;
                let (_, samples) = output[0].try_extract_tensor::<f32>()?;
                Ok(Audio {
                    sample_rate: self.config.audio.sample_rate,
                    samples: samples.to_vec(),
                })
            };
            let mut run = run;
            crate::paragraph::append(&mut audio, run().map_err(|e| Error(e.to_string()))?)
                .map_err(Error)?;
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
        }
        audio.ok_or_else(|| Error("empty input".into()))
    }
}
impl Synthesizer for Piper {
    type Error = Error;
    fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<Audio, Error> {
        self.synthesize_cancellable(request, &AtomicBool::new(false))
    }
}
/// Current number/date/unit normalization with limited contextual stress.
pub fn prepare_text(text: &str) -> Result<String, String> {
    crate::text::prepare(text, Language::Uk)
}
fn tokenize(text: &str, map: &BTreeMap<String, Vec<i64>>) -> Result<Vec<i64>, Error> {
    let normalized = prepare_text(text).map_err(Error)?;
    if normalized.is_empty() {
        return Err(Error("empty input".into()));
    }
    let get = |c: &str| {
        map.get(c).ok_or_else(|| {
            Error(format!(
                "unsupported Ukrainian character {c:?}; spell out dates and abbreviations"
            ))
        })
    };
    let mut ids = get("^")?.clone();
    ids.extend(get("_")?);
    for c in normalized.nfd() {
        if "_^$".contains(c) {
            return Err(Error("reserved token in Ukrainian text".into()));
        }
        ids.extend(get(&c.to_string())?);
        ids.extend(get("_")?);
    }
    ids.extend(get("$")?);
    if ids.len() > 1024 {
        return Err(Error("Ukrainian chunk exceeds 1024 tokens".into()));
    }
    Ok(ids)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn map() -> BTreeMap<String, Vec<i64>> {
        let config: Config =
            serde_json::from_str(include_str!("../../../assets/piper-uk-config.json")).unwrap();
        config.phoneme_id_map
    }
    #[test]
    fn matches_piper_decomposition_and_padding() {
        assert_eq!(
            tokenize("ЇЙ", &map()).unwrap(),
            [1, 0, 23, 0, 47, 0, 22, 0, 46, 0, 2]
        );
        assert_eq!(
            tokenize("ґ+анок", &map()).unwrap(),
            tokenize("ґа́нок", &map()).unwrap()
        );
        assert_eq!(
            tokenize("п’ять", &map()).unwrap(),
            tokenize("п'ять", &map()).unwrap()
        );
        assert_eq!(
            tokenize("123", &map()).unwrap(),
            tokenize("сто два́дцять три", &map()).unwrap()
        );
        assert!(tokenize("ы", &map()).is_err());
        assert!(tokenize(" ", &map()).is_err());
        assert!(tokenize("^", &map()).is_err());
    }
}

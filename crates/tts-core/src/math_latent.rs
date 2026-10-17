//! Numeric-only cached acoustic conditioning and a shared neural decoder.
use crate::{Audio, Language, kokoro::Error, math_speech::vocabulary_for_language};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct Entry {
    offset: usize,
    frames: usize,
    #[serde(default)]
    scales: Vec<f32>,
    #[serde(default)]
    scales_offset: Option<usize>,
}
#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    architecture: String,
    language: String,
    sample_rate: u32,
    channels: usize,
    hop: usize,
    model: String,
    conditioning: String,
    #[serde(default)]
    conditioning_scales: Option<String>,
    #[serde(default)]
    repeat_digit_pause_ms: u32,
    #[serde(default)]
    decode_per_entry: bool,
    entries: BTreeMap<String, Entry>,
}
pub struct MathLatent {
    pub language: Language,
    decode_per_entry: bool,
    session: Session,
    words: BTreeMap<String, Vec<f32>>,
    frames: BTreeMap<String, usize>,
    repeat_digit_pause_ms: u32,
}
impl MathLatent {
    pub fn load(root: &Path) -> Result<Self, Error> {
        let load = || -> Result<Self, Box<dyn std::error::Error>> {
            if std::fs::metadata(root.join("arithmetic-neural.json"))?.len() > 400_000 {
                return Err("manifest too large".into());
            }
            let m: Manifest =
                serde_json::from_slice(&std::fs::read(root.join("arithmetic-neural.json"))?)?;
            let language = match m.language.as_str() {
                "en" => Language::En,
                "ru" => Language::Ru,
                "uk" => Language::Uk,
                _ => return Err("unsupported arithmetic language".into()),
            };
            if !matches!(m.schema_version, 1..=3)
                || m.architecture != "arithmetic-latent-v1"
                || m.sample_rate != 22050
                || m.channels != 192
                || m.hop != 256
                || m.model != "model.onnx"
                || m.conditioning != "conditioning.bin"
                || m.repeat_digit_pause_ms > 100
            {
                return Err("invalid arithmetic neural contract".into());
            }
            let hashes = serde_json::from_slice(&std::fs::read(root.join("checksums.json"))?)?;
            crate::integrity::verify(
                root,
                &hashes,
                &["arithmetic-neural.json", "model.onnx", "conditioning.bin"],
            )
            .map_err(std::io::Error::other)?;
            let scale_bank = if m.schema_version >= 2 {
                if m.conditioning_scales.as_deref() != Some("conditioning-scales.bin") {
                    return Err("invalid scale-bank contract".into());
                }
                crate::integrity::verify(root, &hashes, &["conditioning-scales.bin"])
                    .map_err(std::io::Error::other)?;
                if std::fs::metadata(root.join("conditioning-scales.bin"))?.len() > 200_000 {
                    return Err("scale bank too large".into());
                }
                Some(std::fs::read(root.join("conditioning-scales.bin"))?)
            } else {
                if m.conditioning_scales.is_some() {
                    return Err("unexpected scale bank".into());
                }
                None
            };
            if std::fs::metadata(root.join(&m.conditioning))?.len() > 2_000_000 {
                return Err("conditioning bank too large".into());
            }
            let bank = std::fs::read(root.join(&m.conditioning))?;
            let core = vocabulary_for_language(language).map_err(std::io::Error::other)?;
            let mut allowed: std::collections::BTreeSet<_> = core.iter().cloned().collect();
            if m.schema_version == 3 {
                let zero = crate::math_speech::words_for_language("0", language)
                    .map_err(std::io::Error::other)?
                    .join(" ");
                let pair = format!("{zero} {zero}");
                if !m.entries.contains_key(&pair) {
                    return Err("schema-3 requires double-zero conditioning".into());
                }
                allowed.insert(pair);
            }
            if m.schema_version >= 2 && language == Language::En {
                allowed.extend(
                    (21..100)
                        .filter(|n| n % 10 != 0)
                        .map(crate::english_numbers::cardinal),
                );
                allowed.extend((1..10).map(|n| crate::english_numbers::cardinal(n * 100)));
            }
            if core.iter().any(|w| !m.entries.contains_key(w))
                || m.entries.keys().any(|w| !allowed.contains(w))
            {
                return Err("arithmetic vocabulary differs".into());
            }
            let mut words = BTreeMap::new();
            let mut frames = BTreeMap::new();
            let mut ranges = Vec::new();
            let mut scale_ranges = Vec::new();
            for (word, mut e) in m.entries {
                if let Some(bank) = &scale_bank {
                    if !e.scales.is_empty() {
                        return Err("ambiguous scales".into());
                    }
                    let offset = e.scales_offset.ok_or("missing scale offset")?;
                    let end = offset.checked_add(192 * 4).ok_or("scale overflow")?;
                    if offset % 4 != 0 {
                        return Err("unaligned scale offset".into());
                    }
                    let bytes = bank.get(offset..end).ok_or("scales outside bank")?;
                    e.scales = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
                        .collect();
                    scale_ranges.push((offset, end));
                } else if e.scales_offset.is_some() {
                    return Err("unexpected scale offset".into());
                }
                if !(1..=1024).contains(&e.frames)
                    || e.scales.len() != 192
                    || e.scales
                        .iter()
                        .any(|s| !s.is_finite() || *s <= 0.0 || *s > 100.0)
                {
                    return Err("invalid conditioning entry".into());
                }
                let end = e
                    .offset
                    .checked_add(192 * e.frames)
                    .ok_or("conditioning overflow")?;
                let bytes = bank.get(e.offset..end).ok_or("conditioning outside bank")?;
                ranges.push((e.offset, end));
                let decoded = bytes
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (*b as i8 as f32) * e.scales[i / e.frames])
                    .collect();
                frames.insert(word.clone(), e.frames);
                words.insert(word, decoded);
            }
            ranges.sort_unstable();
            let mut end = 0;
            for (start, next) in ranges {
                if start != end {
                    return Err("overlapping or incomplete conditioning bank".into());
                }
                end = next;
            }
            if end != bank.len() {
                return Err("unaccounted conditioning data".into());
            }
            if let Some(bank) = scale_bank {
                scale_ranges.sort_unstable();
                let mut end = 0;
                for (start, next) in scale_ranges {
                    if start != end {
                        return Err("overlapping or incomplete scale bank".into());
                    }
                    end = next;
                }
                if end != bank.len() {
                    return Err("unaccounted scales".into());
                }
            }
            let session = Session::builder()?
                .with_intra_threads(2)?
                .commit_from_file(root.join(m.model))?;
            Ok(Self {
                language,
                decode_per_entry: m.decode_per_entry,
                session,
                words,
                frames,
                repeat_digit_pause_ms: m.repeat_digit_pause_ms,
            })
        };
        load().map_err(|e| Error(e.to_string()))
    }
    pub fn synthesize_words(
        &mut self,
        words: &[String],
        duration: f32,
        cancel: &AtomicBool,
    ) -> Result<Audio, Error> {
        if words.is_empty()
            || words.len() > 512
            || !duration.is_finite()
            || !(0.5..=2.0).contains(&duration)
        {
            return Err(Error("invalid word count or duration scale".into()));
        }
        let mut selected = Vec::new();
        let mut total = 0;
        let mut index = 0;
        let mut previous = None;
        let mut pauses = Vec::new();
        let pause_samples = (self.repeat_digit_pause_ms as f32 * 22.05 * duration).round() as usize;
        while index < words.len() {
            crate::paragraph::check_cancel(cancel).map_err(Error)?;
            let combined = words
                .get(index + 1)
                .map(|next| format!("{} {next}", words[index]));
            let word = if let Some(combined) = combined.filter(|w| self.frames.contains_key(w)) {
                index += 2;
                combined
            } else {
                let word = words[index].clone();
                index += 1;
                word
            };
            let frames = *self
                .frames
                .get(&word)
                .ok_or_else(|| Error(format!("unknown arithmetic word {word}")))?;
            if previous.as_ref() == Some(&word)
                && pause_samples > 0
                && (0..10).any(|n| {
                    crate::math_speech::words_for_language(&n.to_string(), self.language)
                        .is_ok_and(|v| v == [word.clone()])
                })
            {
                pauses.push(total * 256);
            }
            previous = Some(word.clone());
            let scaled = (frames as f32 * duration).round().max(1.0) as usize;
            total += scaled;
            if total > 7740 || total * 256 + pauses.len() * pause_samples > 22050 * 90 {
                return Err(Error(
                    "arithmetic speech exceeds 90 seconds; split the expression".into(),
                ));
            }
            selected.push((word, frames, scaled));
        }
        let chunks: Vec<usize> = selected.iter().map(|(_, _, scaled)| *scaled).collect();
        let mut latent = vec![0.0; 192 * total];
        let mut offset = 0;
        for (word, frames, scaled) in selected {
            let source = &self.words[&word];
            for channel in 0..192 {
                for frame in 0..scaled {
                    let pos = if scaled == 1 {
                        0.0
                    } else {
                        frame as f32 * (frames - 1) as f32 / (scaled - 1) as f32
                    };
                    let lo = pos.floor() as usize;
                    let hi = (lo + 1).min(frames - 1);
                    let mix = pos - lo as f32;
                    latent[channel * total + offset + frame] = source[channel * frames + lo]
                        * (1.0 - mix)
                        + source[channel * frames + hi] * mix;
                }
            }
            offset += scaled;
        }
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        let run = || -> Result<Audio, Box<dyn std::error::Error>> {
            if self.decode_per_entry && chunks.len() > 1 {
                let mut samples = Vec::with_capacity(total * 256);
                let mut offset = 0;
                for frames in chunks {
                    crate::paragraph::check_cancel(cancel).map_err(std::io::Error::other)?;
                    let mut part = Vec::with_capacity(192 * frames);
                    for channel in 0..192 {
                        part.extend_from_slice(
                            &latent[channel * total + offset..channel * total + offset + frames],
                        );
                    }
                    let output = self.session.run(
                        ort::inputs!["latent"=>Tensor::from_array(([1usize,192,frames],part))?],
                    )?;
                    let (_, raw) = output[0].try_extract_tensor::<f32>()?;
                    if raw.len() != frames * 256 || raw.iter().any(|s| !s.is_finite()) {
                        return Err("invalid per-entry decoder output".into());
                    }
                    let mut part = raw.to_vec();
                    let fade = 66.min(part.len() / 2);
                    let count = part.len();
                    for i in 0..fade {
                        let gain = i as f32 / fade as f32;
                        part[i] *= gain;
                        part[count - 1 - i] *= gain;
                    }
                    samples.extend(part);
                    offset += frames;
                }
                return Ok(Audio {
                    sample_rate: 22050,
                    samples,
                });
            }
            let output = self
                .session
                .run(ort::inputs!["latent"=>Tensor::from_array(([1usize,192,total],latent))?])?;
            let (_, samples) = output[0].try_extract_tensor::<f32>()?;
            if samples.len() != total * 256 || samples.iter().any(|s| !s.is_finite()) {
                return Err("invalid decoder output".into());
            }
            Ok(Audio {
                sample_rate: 22050,
                samples: samples.to_vec(),
            })
        };
        let mut audio = run().map_err(|e| Error(e.to_string()))?;
        if !pauses.is_empty() {
            // A short fade avoids clicks at the explicit repeated-digit gap.
            let fade = 66usize;
            for &boundary in &pauses {
                for i in 0..fade {
                    let gain = i as f32 / fade as f32;
                    if boundary > i {
                        audio.samples[boundary - 1 - i] *= gain;
                    }
                    if boundary + i < audio.samples.len() {
                        audio.samples[boundary + i] *= gain;
                    }
                }
            }
            let mut joined = Vec::with_capacity(audio.samples.len() + pauses.len() * pause_samples);
            let mut start = 0;
            for boundary in pauses {
                joined.extend_from_slice(&audio.samples[start..boundary]);
                joined.resize(joined.len() + pause_samples, 0.0);
                start = boundary;
            }
            joined.extend_from_slice(&audio.samples[start..]);
            audio.samples = joined;
        }
        crate::paragraph::check_cancel(cancel).map_err(Error)?;
        Ok(audio)
    }
    pub fn verify(&mut self, root: &Path) -> Result<f32, Error> {
        #[derive(Deserialize)]
        struct Case {
            word: String,
            #[serde(default)]
            samples: Vec<f32>,
            #[serde(default)]
            offset: Option<usize>,
            #[serde(default)]
            count: Option<usize>,
        }
        let cases: Vec<Case> = serde_json::from_slice(
            &std::fs::read(root.join("latent-reference.json")).map_err(|e| Error(e.to_string()))?,
        )
        .map_err(|e| Error(e.to_string()))?;
        if cases.is_empty() {
            return Err(Error("no reference cases".into()));
        }
        let mut maximum = 0.0f32;
        let binary = if cases.iter().any(|c| c.offset.is_some()) {
            Some(
                std::fs::read(root.join("latent-reference.bin"))
                    .map_err(|e| Error(e.to_string()))?,
            )
        } else {
            None
        };
        for mut case in cases {
            if let Some(offset) = case.offset {
                let size = case
                    .count
                    .ok_or_else(|| Error("missing reference count".into()))?
                    .checked_mul(4)
                    .ok_or_else(|| Error("reference overflow".into()))?;
                let end = offset
                    .checked_add(size)
                    .ok_or_else(|| Error("reference overflow".into()))?;
                if !case.samples.is_empty() {
                    return Err(Error("ambiguous reference samples".into()));
                }
                let bytes = binary
                    .as_ref()
                    .and_then(|b| b.get(offset..end))
                    .ok_or_else(|| Error("reference outside bank".into()))?;
                case.samples = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
                    .collect();
            }
            let audio = self.synthesize_words(&[case.word], 1.0, &AtomicBool::new(false))?;
            if audio.samples.len() != case.samples.len()
                || case.samples.iter().any(|s| !s.is_finite())
            {
                return Err(Error("reference length or values invalid".into()));
            }
            for (a, b) in audio.samples.iter().zip(case.samples) {
                maximum = maximum.max((a - b).abs());
            }
        }
        if maximum > 1e-4 {
            return Err(Error(format!("native decoder parity failed: {maximum}")));
        }
        Ok(maximum)
    }
}

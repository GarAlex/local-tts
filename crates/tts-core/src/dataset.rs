use crate::Language;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Component, Path},
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub schema_version: u32,
    pub language: Language,
    pub sample_rate: u32,
    /// Records the source of permission; validation does not establish legal rights.
    pub rights: Rights,
    pub utterances: Vec<Utterance>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Rights {
    pub audio: String,
    pub transcripts: String,
    pub speaker_consent: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Utterance {
    pub id: String,
    pub audio: String,
    pub text: String,
    pub speaker: String,
    pub split: Split,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Serialize, Default)]
pub struct ValidationReport {
    pub utterances: usize,
    pub duration_seconds: f64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub speakers: BTreeMap<String, GroupStats>,
    pub splits: BTreeMap<String, GroupStats>,
}

#[derive(Debug, Serialize, Default)]
pub struct GroupStats {
    /// Includes every declared manifest entry, even when validation fails.
    pub utterances: usize,
    /// Only error-free entries with unique, fully decoded audio contribute below.
    pub valid_clips: usize,
    pub duration_seconds: f64,
    pub shortest_seconds: Option<f64>,
    pub longest_seconds: Option<f64>,
}
impl GroupStats {
    fn add_duration(&mut self, seconds: f64) {
        self.valid_clips += 1;
        self.duration_seconds += seconds;
        self.shortest_seconds = Some(self.shortest_seconds.map_or(seconds, |n| n.min(seconds)));
        self.longest_seconds = Some(self.longest_seconds.map_or(seconds, |n| n.max(seconds)));
    }
}
impl ValidationReport {
    pub fn statistics_text(&self) -> String {
        let mut lines = vec!["Validated audio by speaker / split:".to_string()];
        for (kind, groups) in [("Speaker", &self.speakers), ("Split", &self.splits)] {
            for (name, stats) in groups {
                lines.push(format!(
                    "{kind} {name}: {} / {} valid clips, {:.2} min",
                    stats.valid_clips,
                    stats.utterances,
                    stats.duration_seconds / 60.0
                ));
            }
        }
        lines.join("\n")
    }
}

pub fn read(path: &Path) -> Result<Dataset, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(crate) fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

pub fn validate(dataset: &Dataset, root: &Path) -> ValidationReport {
    let mut report = ValidationReport {
        utterances: dataset.utterances.len(),
        ..Default::default()
    };
    if dataset.schema_version != 1 {
        report
            .errors
            .push("unsupported schema_version; expected 1".into());
    }
    if dataset.sample_rate < 8000 || dataset.sample_rate > 96000 {
        report
            .errors
            .push("sample_rate must be between 8000 and 96000".into());
    }
    for (name, value) in [
        ("audio", &dataset.rights.audio),
        ("transcripts", &dataset.rights.transcripts),
        ("speaker_consent", &dataset.rights.speaker_consent),
    ] {
        if value.trim().is_empty() {
            report
                .errors
                .push(format!("rights.{name} must document permission"));
        }
    }
    if dataset.utterances.is_empty() {
        report.errors.push("dataset has no utterances".into());
    }
    let mut ids = HashSet::new();
    let mut paths = HashSet::new();
    let mut texts = std::collections::HashMap::new();
    for item in &dataset.utterances {
        let split_name = match item.split {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        };
        report
            .speakers
            .entry(item.speaker.clone())
            .or_default()
            .utterances += 1;
        report
            .splits
            .entry(split_name.into())
            .or_default()
            .utterances += 1;
    }
    let canonical_root = match root.canonicalize() {
        Ok(path) => path,
        Err(e) => {
            report.errors.push(format!("dataset root: {e}"));
            return report;
        }
    };
    for item in &dataset.utterances {
        let errors_before = report.errors.len();
        let split_name = match item.split {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        };
        let prefix = &item.id;
        if item.id.trim().is_empty() || !ids.insert(&item.id) {
            report
                .errors
                .push(format!("{prefix}: empty or duplicate id"));
        }
        if item.text.trim().is_empty() {
            report.errors.push(format!("{prefix}: empty transcript"));
        }
        if item.speaker.trim().is_empty() {
            report.errors.push(format!("{prefix}: empty speaker"));
        }
        let normalized = crate::frontend::match_key(&item.text, dataset.language);
        if let Some(previous) = texts.insert(normalized, item.split)
            && previous != item.split
        {
            report.warnings.push(format!(
                "{prefix}: repeated transcript across dataset splits"
            ));
        }
        if !safe_relative(&item.audio) {
            report.errors.push(format!(
                "{prefix}: audio must be a relative path without traversal"
            ));
            continue;
        }
        let path = match root.join(&item.audio).canonicalize() {
            Ok(path) if path.starts_with(&canonical_root) => path,
            Ok(_) => {
                report
                    .errors
                    .push(format!("{prefix}: audio resolves outside dataset root"));
                continue;
            }
            Err(e) => {
                report
                    .errors
                    .push(format!("{prefix}: audio unavailable: {e}"));
                continue;
            }
        };
        if !paths.insert(path.clone()) {
            report.errors.push(format!(
                "{prefix}: duplicate audio file (possible split leakage)"
            ));
        }
        let mut reader = match hound::WavReader::open(path) {
            Ok(reader) => reader,
            Err(e) => {
                report.errors.push(format!("{prefix}: invalid WAV: {e}"));
                continue;
            }
        };
        let spec = reader.spec();
        if spec.channels != 1 {
            report.errors.push(format!("{prefix}: expected mono WAV"));
        }
        if spec.sample_rate != dataset.sample_rate {
            report.errors.push(format!(
                "{prefix}: sample rate {} differs from {}",
                spec.sample_rate, dataset.sample_rate
            ));
        }
        if spec.sample_rate == 0 {
            report.errors.push(format!("{prefix}: zero sample rate"));
            continue;
        }
        let duration = f64::from(reader.duration()) / f64::from(spec.sample_rate);
        report.duration_seconds += duration;
        if duration == 0.0 {
            report.errors.push(format!("{prefix}: empty audio"));
        }
        if !(0.5..=20.0).contains(&duration) {
            report.warnings.push(format!(
                "{prefix}: duration {duration:.2}s outside recommended 0.5–20s"
            ));
        }
        let (mut peak, mut energy, mut count) = (0.0_f64, 0.0_f64, 0_u64);
        let mut add = |sample: Result<f64, hound::Error>| -> Result<(), String> {
            let value = sample.map_err(|e| e.to_string())?;
            if !value.is_finite() {
                return Err("non-finite sample".into());
            }
            peak = peak.max(value.abs());
            energy += value * value;
            count += 1;
            Ok(())
        };
        let decoded = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .try_for_each(|s| add(s.map(f64::from))),
            hound::SampleFormat::Int => {
                let scale = 2_f64.powi(i32::from(spec.bits_per_sample) - 1);
                reader
                    .samples::<i32>()
                    .try_for_each(|s| add(s.map(|v| f64::from(v) / scale)))
            }
        };
        if let Err(e) = decoded {
            report
                .errors
                .push(format!("{prefix}: sample decoding failed: {e}"));
        }
        if count > 0 {
            if peak >= 0.999 {
                report.warnings.push(format!(
                    "{prefix}: audio reaches full scale; inspect for clipping"
                ));
            }
            if (energy / count as f64).sqrt() < 0.0001 {
                report
                    .errors
                    .push(format!("{prefix}: silent or near-silent audio"));
            }
        }
        if report.errors.len() == errors_before && count > 0 {
            report
                .speakers
                .get_mut(&item.speaker)
                .unwrap()
                .add_duration(duration);
            report
                .splits
                .get_mut(split_name)
                .unwrap()
                .add_duration(duration);
        }
    }
    for split in [Split::Train, Split::Validation, Split::Test] {
        if !dataset.utterances.iter().any(|u| u.split == split) {
            report.warnings.push(format!("no {split:?} utterances"));
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paths_cannot_escape() {
        for path in ["", "../voice.wav", "/voice.wav", "audio/../voice.wav"] {
            assert!(!safe_relative(path));
        }
        assert!(safe_relative("audio/voice.wav"));
    }
}

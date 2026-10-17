//! Held-out comparisons through the same native VITS engine used by embedded apps.
use crate::{
    Audio, Language, SynthesisRequest,
    dataset::{self, Split},
    vits::Vits,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioFile {
    pub file: String,
    pub sha256: String,
    pub seconds: f64,
    pub rms: f64,
    pub peak: f32,
    /// Fraction of unprocessed samples reaching/exceeding full scale, before WAV clamping.
    pub clipped_fraction: f64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Sample {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepared_text: Option<String>,
    pub reference: AudioFile,
    pub before: AudioFile,
    pub after: AudioFile,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub schema_version: u32,
    pub language: Language,
    pub frontend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_normalization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_normalization: Option<String>,
    pub split: Split,
    pub dataset_sha256: String,
    pub before_model_sha256: String,
    pub after_model_sha256: String,
    pub duration_scale: f32,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub samples: Vec<Sample>,
}

fn stats(root: &Path, name: &str, audio: &Audio) -> Result<AudioFile, String> {
    if audio.sample_rate == 0
        || audio.samples.is_empty()
        || audio.samples.iter().any(|s| !s.is_finite())
    {
        return Err("empty or non-finite comparison audio".into());
    }
    let n = audio.samples.len() as f64;
    Ok(AudioFile {
        file: name.into(),
        sha256: crate::integrity::hash_file(root, name)?,
        seconds: n / f64::from(audio.sample_rate),
        rms: (audio
            .samples
            .iter()
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>()
            / n)
            .sqrt(),
        peak: audio.samples.iter().map(|s| s.abs()).fold(0.0, f32::max),
        clipped_fraction: audio.samples.iter().filter(|s| s.abs() >= 1.0).count() as f64 / n,
    })
}
fn save(root: &Path, name: &str, audio: &Audio) -> Result<AudioFile, String> {
    audio
        .write_wav(&root.join(name))
        .map_err(|e| e.to_string())?;
    stats(root, name, audio)
}
/// Validate the stored hash and package path before playing an evaluation artifact.
pub fn read_audio(root: &Path, file: &AudioFile) -> Result<Audio, String> {
    if crate::integrity::hash_file(root, &file.file)? != file.sha256 {
        return Err("comparison audio checksum mismatch".into());
    }
    crate::read_wav(&root.join(&file.file)).map_err(|e| e.to_string())
}
/// Never evaluates training clips or overwrites an existing output directory.
/// Partial files can remain after cancellation/failure; only report.json indicates completion.
pub fn compare(
    before: &Path,
    after: &Path,
    manifest: &Path,
    output: &Path,
    split: Split,
    limit: usize,
    cancel: &AtomicBool,
) -> Result<Comparison, String> {
    if split == Split::Train || !(1..=12).contains(&limit) {
        return Err("choose validation/test and 1–12 clips".into());
    }
    crate::paragraph::check_cancel(cancel)?;
    let data = dataset::read(manifest).map_err(|e| e.to_string())?;
    let root = manifest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let validation = dataset::validate(&data, root);
    if !validation.errors.is_empty() {
        return Err(validation.errors.join("\n"));
    }
    let clips: Vec<_> = data
        .utterances
        .iter()
        .filter(|u| u.split == split)
        .take(limit)
        .collect();
    if clips.is_empty() {
        return Err("no recordings in selected held-out split".into());
    }
    // Symbol indices and dictionary must match, not just the frontend's name.
    let tokens_before: serde_json::Value = serde_json::from_slice(
        &std::fs::read(before.join("tokens.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let tokens_after: serde_json::Value = serde_json::from_slice(
        &std::fs::read(after.join("tokens.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if tokens_before != tokens_after {
        return Err("comparison token inventories differ".into());
    }
    let mut base = Vits::load(before).map_err(|e| e.to_string())?;
    let mut target = Vits::load(after).map_err(|e| e.to_string())?;
    if base.manifest.language != data.language
        || target.manifest.language != data.language
        || base.manifest.frontend != target.manifest.frontend
        || base.manifest.sample_rate != data.sample_rate
        || target.manifest.sample_rate != data.sample_rate
        || base.manifest.speakers.len() != 1
        || target.manifest.speakers.len() != 1
        || clips
            .iter()
            .any(|u| u.speaker != target.manifest.speakers[0])
    {
        return Err("comparison language/frontend/sample rate/speaker differs".into());
    }
    if base.manifest.frontend == crate::training::IPA_FRONTEND
        && crate::integrity::hash_file(before, "cmudict.dict")?
            != crate::integrity::hash_file(after, "cmudict.dict")?
    {
        return Err("comparison dictionaries differ".into());
    }
    let mut report = Comparison {
        schema_version: 1,
        language: data.language,
        frontend: base.manifest.frontend.clone(),
        before_normalization: base.manifest.normalization.clone(),
        after_normalization: target.manifest.normalization.clone(),
        split,
        dataset_sha256: crate::integrity::hash_file(
            root,
            manifest
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("invalid manifest filename")?,
        )?,
        before_model_sha256: crate::integrity::hash_file(before, &base.manifest.model)?,
        after_model_sha256: crate::integrity::hash_file(after, &target.manifest.model)?,
        duration_scale: 1.0,
        warnings: validation.warnings,
        samples: Vec::new(),
    };
    let mixed_normalization = base.manifest.normalization != target.manifest.normalization;
    if mixed_normalization {
        report.warnings.push("Base/target text normalization differs. Both receive target-expanded text; English contextual phoneme choices may still differ. This comparison includes frontend differences, not just weight adaptation.".into());
    }
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    std::fs::create_dir(output)
        .map_err(|e| format!("comparison output must be a new directory: {e}"))?;
    for (index, clip) in clips.iter().enumerate() {
        crate::paragraph::check_cancel(cancel)?;
        let source = root
            .join(&clip.audio)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if !source.starts_with(&root) {
            return Err("recording escapes dataset".into());
        }
        let reader = hound::WavReader::open(&source).map_err(|e| e.to_string())?;
        if reader.duration() > data.sample_rate * 20 {
            return Err("comparison clips must be at most twenty seconds".into());
        }
        let name = format!("{index:03}-reference.wav");
        std::fs::write(
            output.join(&name),
            std::fs::read(source).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let audio = crate::read_wav(&output.join(&name)).map_err(|e| e.to_string())?;
        let reference = stats(output, &name, &audio)?;
        let prepared_text = if mixed_normalization && target.manifest.normalization.is_some() {
            Some(crate::text::prepare(&clip.text, data.language)?)
        } else {
            None
        };
        let request = SynthesisRequest {
            text: prepared_text.as_deref().unwrap_or(&clip.text),
            language: data.language,
            speaker: &base.manifest.speakers[0].clone(),
            duration_scale: 1.0,
        };
        let before_audio = base
            .synthesize_cancellable(&request, cancel)
            .map_err(|e| e.to_string())?;
        let before = save(output, &format!("{index:03}-before.wav"), &before_audio)?;
        let speaker = target.manifest.speakers[0].clone();
        let after_audio = target
            .synthesize_cancellable(
                &SynthesisRequest {
                    speaker: &speaker,
                    ..request
                },
                cancel,
            )
            .map_err(|e| e.to_string())?;
        let after = save(output, &format!("{index:03}-after.wav"), &after_audio)?;
        report.samples.push(Sample {
            id: clip.id.clone(),
            text: clip.text.clone(),
            prepared_text,
            reference,
            before,
            after,
        });
    }
    crate::paragraph::check_cancel(cancel)?;
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_checks_integrity_and_levels_are_not_quality_scores() {
        let dir = tempfile::tempdir().unwrap();
        let audio = Audio {
            sample_rate: 10,
            samples: vec![-1.2, 0.0, 0.5, 1.0],
        };
        let file = save(dir.path(), "sample.wav", &audio).unwrap();
        assert_eq!(file.seconds, 0.4);
        assert_eq!(file.peak, 1.2);
        assert_eq!(file.clipped_fraction, 0.5);
        read_audio(dir.path(), &file).unwrap();
        std::fs::write(dir.path().join("sample.wav"), b"changed").unwrap();
        assert!(
            read_audio(dir.path(), &file)
                .unwrap_err()
                .contains("checksum")
        );
    }
    #[test]
    fn rejects_training_split_and_cancellation_before_loading() {
        let p = Path::new("missing");
        assert!(
            compare(p, p, p, p, Split::Train, 1, &AtomicBool::new(false))
                .unwrap_err()
                .contains("validation/test")
        );
        assert!(
            compare(p, p, p, p, Split::Validation, 1, &AtomicBool::new(true))
                .unwrap_err()
                .contains("cancelled")
        );
    }
}

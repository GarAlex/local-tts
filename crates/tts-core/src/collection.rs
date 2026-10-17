//! Durable dataset editing. Audio imports never replace the source recording.
use crate::{
    Audio, Language,
    dataset::{self, Dataset, Rights, Split, Utterance},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub struct Collection {
    pub data: Dataset,
    manifest: PathBuf,
    disk_digest: Vec<u8>,
}
impl Collection {
    pub fn create(root: &Path, language: Language) -> Result<Self, String> {
        fs::create_dir_all(root).map_err(|e| e.to_string())?;
        let manifest = root
            .canonicalize()
            .map_err(|e| e.to_string())?
            .join("dataset.json");
        let data = Dataset {
            schema_version: 1,
            language,
            sample_rate: 22050,
            rights: Rights {
                audio: String::new(),
                transcripts: String::new(),
                speaker_consent: String::new(),
            },
            utterances: Vec::new(),
        };
        let bytes = serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?;
        let mut temp = tempfile::NamedTempFile::new_in(root).map_err(|e| e.to_string())?;
        temp.write_all(&bytes).map_err(|e| e.to_string())?;
        temp.as_file().sync_all().map_err(|e| e.to_string())?;
        temp.persist_noclobber(&manifest)
            .map_err(|e| format!("dataset already exists or cannot be created: {e}"))?;
        Ok(Self {
            data,
            manifest,
            disk_digest: Sha256::digest(&bytes).to_vec(),
        })
    }
    pub fn open(path: &Path) -> Result<Self, String> {
        let manifest = path.canonicalize().map_err(|e| e.to_string())?;
        let bytes = fs::read(&manifest).map_err(|e| e.to_string())?;
        let data: Dataset = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if data.schema_version != 1 || data.sample_rate != 22050 {
            return Err("dataset editor supports schema v1 / 22050 Hz; import clips into a new dataset for other rates".into());
        }
        Ok(Self {
            data,
            manifest,
            disk_digest: Sha256::digest(&bytes).to_vec(),
        })
    }
    pub fn manifest(&self) -> &Path {
        &self.manifest
    }
    pub fn root(&self) -> &Path {
        self.manifest.parent().unwrap()
    }
    pub fn save(&mut self) -> Result<(), String> {
        if self.data.schema_version != 1 || self.data.sample_rate != 22050 {
            return Err("dataset editor requires schema v1 and 22050 Hz".into());
        }
        let mut ids = HashSet::new();
        for u in &self.data.utterances {
            if u.id.trim().is_empty()
                || !ids.insert(&u.id)
                || !dataset::safe_relative(&u.audio)
                || u.text.trim().is_empty()
                || u.speaker.trim().is_empty()
            {
                return Err(
                    "every clip needs a unique id, safe relative path, transcript and speaker"
                        .into(),
                );
            }
        }
        if Sha256::digest(fs::read(&self.manifest).map_err(|e| e.to_string())?).as_slice()
            != self.disk_digest
        {
            return Err("manifest changed outside Studio; reopen it before saving".into());
        }
        let bytes = serde_json::to_vec_pretty(&self.data).map_err(|e| e.to_string())?;
        let mut temp = tempfile::NamedTempFile::new_in(self.root()).map_err(|e| e.to_string())?;
        temp.write_all(&bytes).map_err(|e| e.to_string())?;
        temp.as_file().sync_all().map_err(|e| e.to_string())?;
        temp.persist(&self.manifest).map_err(|e| e.to_string())?;
        self.disk_digest = Sha256::digest(bytes).to_vec();
        Ok(())
    }
    pub fn add_clip(
        &mut self,
        audio: &Audio,
        text: &str,
        speaker: &str,
        split: Split,
    ) -> Result<String, String> {
        if text.trim().is_empty() || speaker.trim().is_empty() {
            return Err("enter the exact transcript and speaker".into());
        }
        if audio.sample_rate != 22050
            || !(11025..=441000).contains(&audio.samples.len())
            || audio
                .samples
                .iter()
                .any(|s| !s.is_finite() || s.abs() > 1.0)
        {
            return Err("clip must be finite mono audio at 22050 Hz, between half a second and twenty seconds".into());
        }
        let rms = (audio
            .samples
            .iter()
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>()
            / audio.samples.len() as f64)
            .sqrt();
        if rms < 0.0001 {
            return Err("clip is silent or near-silent".into());
        }
        let dir = self.root().join("audio");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let dir = dir.canonicalize().map_err(|e| e.to_string())?;
        if !dir.starts_with(self.root()) {
            return Err("audio folder resolves outside dataset".into());
        }
        let mut number = 1;
        let (id, path) = loop {
            let id = format!("clip-{number:05}");
            let path = dir.join(format!("{id}.wav"));
            if !path.exists() && !self.data.utterances.iter().any(|u| u.id == id) {
                break (id, path);
            }
            number += 1;
        };
        let temp = tempfile::NamedTempFile::new_in(&dir).map_err(|e| e.to_string())?;
        audio.write_wav(temp.path()).map_err(|e| e.to_string())?;
        temp.as_file().sync_all().map_err(|e| e.to_string())?;
        temp.persist_noclobber(&path).map_err(|e| e.to_string())?;
        self.data.utterances.push(Utterance {
            id: id.clone(),
            audio: format!("audio/{id}.wav"),
            text: text.trim().into(),
            speaker: speaker.trim().into(),
            split,
        });
        if let Err(error) = self.save() {
            self.data.utterances.pop();
            // Roll back only the new file created by this operation.
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        Ok(id)
    }
    /// Groups identical normalized transcripts to avoid splitting the same prompt.
    pub fn assign_splits(&mut self, seed: u64) -> Result<(), String> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, u) in self.data.utterances.iter().enumerate() {
            groups
                .entry(crate::frontend::match_key(&u.text, self.data.language))
                .or_default()
                .push(i);
        }
        if groups.len() < 2 {
            return Err(
                "at least two distinct transcripts are needed for a validation split".into(),
            );
        }
        let mut ranked: Vec<_> = groups
            .into_iter()
            .map(|(text, ids)| {
                let mut hash = Sha256::new();
                hash.update(seed.to_le_bytes());
                hash.update(text.as_bytes());
                (hash.finalize().to_vec(), ids)
            })
            .collect();
        ranked.sort_by(|a, b| a.0.cmp(&b.0));
        let count = ranked.len();
        let validation = (count / 10).max(1);
        let test = if count >= 3 { (count / 10).max(1) } else { 0 };
        for (i, (_, ids)) in ranked.into_iter().enumerate() {
            let split = if i < test {
                Split::Test
            } else if i < test + validation {
                Split::Validation
            } else {
                Split::Train
            };
            for id in ids {
                self.data.utterances[id].split = split;
            }
        }
        Ok(())
    }
}

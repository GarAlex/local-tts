use crate::recording::Recorder;
use eframe::egui;
use local_tts_core::{
    Audio, Language, clips,
    collection::Collection,
    dataset::{self, Split},
};
use std::{path::PathBuf, sync::mpsc, thread, time::Duration};

pub enum Action {
    Play(Audio),
    Train(PathBuf),
}
pub struct Editor {
    guide: crate::dataset_guide::Guide,
    collection: Option<Collection>,
    path: String,
    language: Language,
    speaker: String,
    transcript: String,
    prompts: Option<local_tts_core::prompts::PromptList>,
    split: Split,
    selected: Option<usize>,
    dirty: bool,
    clip: Option<Audio>,
    trim_start: f64,
    trim_end: f64,
    recorder: Option<Recorder>,
    pending: bool,
    tx: mpsc::Sender<Result<Audio, String>>,
    rx: mpsc::Receiver<Result<Audio, String>>,
    pub status: String,
}
impl Editor {
    pub fn new(root: &std::path::Path) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            guide: crate::dataset_guide::Guide::default(),
            collection: None,
            path: root.join("datasets/my-voice").display().to_string(),
            language: Language::En,
            speaker: "speaker1".into(),
            transcript: String::new(),
            prompts: None,
            split: Split::Train,
            selected: None,
            dirty: false,
            clip: None,
            trim_start: 0.0,
            trim_end: 0.0,
            recorder: None,
            pending: false,
            tx,
            rx,
            status: "Create a dataset or open an existing manifest.".into(),
        }
    }
    pub fn active(&self) -> bool {
        self.pending || self.recorder.is_some()
    }
    pub fn dirty(&self) -> bool {
        self.dirty || self.clip.is_some() || self.active()
    }
    pub fn discard_dirty_flag(&mut self) {
        self.dirty = false;
        self.clip = None;
        self.recorder = None;
    }
    fn save(&mut self) {
        if let Some(collection) = self.collection.as_mut() {
            match collection.save() {
                Ok(()) => {
                    self.dirty = false;
                    self.status = "Dataset saved".into();
                }
                Err(e) => self.status = e,
            }
        }
    }
    fn reload_saved(&mut self) {
        let Some(collection) = self.collection.as_ref() else {
            return;
        };
        match Collection::open(collection.manifest()) {
            Ok(collection) => {
                self.collection = Some(collection);
                self.selected = None;
                self.dirty = false;
                self.status = "Reloaded saved dataset; unsaved manifest edits discarded.".into();
            }
            Err(error) => self.status = error,
        }
    }
    fn convert_recording(&mut self) {
        let Some(recorder) = self.recorder.take() else {
            return;
        };
        match recorder.finish() {
            Ok(audio) => {
                self.pending = true;
                let tx = self.tx.clone();
                thread::spawn(move || {
                    let _ = tx.send(clips::resample(&audio, 22050));
                });
                self.status = "Preparing recording…".into();
            }
            Err(e) => self.status = e,
        }
    }
    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(result) = self.rx.try_recv() {
            self.pending = false;
            match result {
                Ok(audio) => {
                    self.trim_start = 0.0;
                    self.trim_end = audio.samples.len() as f64 / f64::from(audio.sample_rate);
                    self.clip = Some(audio);
                    self.status =
                        "Clip ready. Verify the transcript, trim, then add it to the dataset."
                            .into();
                }
                Err(e) => self.status = e,
            }
        }
        if let Some(recorder) = &self.recorder {
            let (seconds, done) = recorder.progress();
            self.status = format!("Recording {seconds:.1}s / 20s — {}", recorder.device);
            if done {
                self.convert_recording();
            }
        }
        if self.active() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> Option<Action> {
        self.poll(ctx);
        let mut action = None;
        let mut reload = false;
        self.guide.ui(
            ui,
            self.collection.as_ref().map(|c| &c.data),
            self.dirty,
            self.clip.is_some() || self.active(),
        );
        if self.recorder.is_some() && ui.button("Stop recording").clicked() {
            self.convert_recording();
        }
        ui.add_enabled_ui(!self.active(), |ui| {
            ui.horizontal(|ui| {
                ui.label("Dataset folder"); ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(500.0));
                if ui.button("Browse folder…").clicked() && let Some(path) = rfd::FileDialog::new().pick_folder() { self.path = path.display().to_string(); }
            });
            ui.horizontal(|ui| {
                ui.label("New dataset language");
                language_choice(ui, &mut self.language);
                if ui.button("Create dataset").clicked() {
                if self.dirty || self.clip.is_some() { self.status = "Save edits and add/discard the current take before changing datasets.".into(); }
                    else { match Collection::create(std::path::Path::new(&self.path), self.language) {
                        Ok(collection) => { self.prompts = None; self.collection = Some(collection); self.selected = None; self.clip = None; self.status = "Created dataset. Document rights and add clips.".into(); }, Err(e) => self.status = e,
                    } }
                }
                if ui.button("Open manifest…").clicked() {
                    if self.dirty || self.clip.is_some() { self.status = "Save edits and add/discard the current take before changing datasets.".into(); }
                    else if let Some(path) = rfd::FileDialog::new().add_filter("Dataset", &["json"]).pick_file() {
                        match Collection::open(&path) {
                            Ok(collection) => {
                                self.path = collection.root().display().to_string(); self.language = collection.data.language;
                                self.speaker = collection.data.utterances.first().map(|u| u.speaker.clone()).unwrap_or_else(|| "speaker1".into());
                                self.prompts = None; self.collection = Some(collection); self.selected = None; self.clip = None; self.status = "Dataset opened".into();
                            }, Err(e) => self.status = e,
                        }
                    }
                }
            });
            let Some(collection) = self.collection.as_mut() else { return; };
            ui.label(format!("{} • {:?} • {} clips • mono 22050 Hz", collection.manifest().display(), collection.data.language, collection.data.utterances.len()));
            egui::CollapsingHeader::new("Recording/transcript rights and speaker permission").default_open(collection.data.rights.audio.is_empty()).show(ui, |ui| {
                for (label, value) in [("Audio rights", &mut collection.data.rights.audio), ("Transcript rights", &mut collection.data.rights.transcripts), ("Speaker permission", &mut collection.data.rights.speaker_consent)] {
                    ui.label(label); if ui.add(egui::TextEdit::singleline(value).desired_width(f32::INFINITY)).changed() { self.dirty = true; }
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Speaker"); ui.text_edit_singleline(&mut self.speaker);
                split_choice(ui, "new-clip-split", &mut self.split);
                if ui.add_enabled(self.clip.is_none(), egui::Button::new("Import WAV…")).clicked() && let Some(path) = rfd::FileDialog::new().add_filter("WAV audio", &["wav"]).pick_file() {
                    self.pending = true; self.status = "Importing/downmixing/resampling…".into(); let tx = self.tx.clone();
                    thread::spawn(move || { let _ = tx.send(clips::import_wav(&path)); });
                }
                if ui.add_enabled(self.clip.is_none() && !self.pending, egui::Button::new("Record microphone")).clicked() {
                    match Recorder::start() { Ok(recorder) => self.recorder = Some(recorder), Err(e) => self.status = e }
                }
            });
            ui.label("Recording uses the default microphone only while active. Audio stays local.");
            egui::CollapsingHeader::new("Guided recording prompts").show(ui, |ui| {
                ui.label("Starter lists contain twelve original practice prompts; they are not a complete training corpus.");
                ui.horizontal(|ui| {
                    if ui.add_enabled(self.clip.is_none(), egui::Button::new("Load starter prompts")).clicked() {
                        self.prompts = Some(local_tts_core::prompts::PromptList::starter(collection.data.language));
                    }
                    if ui.add_enabled(self.clip.is_none(), egui::Button::new("Import prompt list…")).clicked()
                        && let Some(path) = rfd::FileDialog::new().add_filter("UTF-8 text", &["txt"]).pick_file()
                    {
                        let result = (|| {
                            if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 1_000_000 {return Err("prompt list exceeds one megabyte".into());}
                            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
                            local_tts_core::prompts::PromptList::parse(collection.data.language, &text)
                        })();
                        match result {Ok(list) => self.prompts = Some(list), Err(e) => self.status = e}
                    }
                });
                if let Some(list) = &self.prompts {
                    match list.remaining(&collection.data, &self.speaker) {
                        Ok(remaining) => {
                            ui.label(format!("{} / {} prompts recorded for {}", list.lines.len()-remaining.len(), list.lines.len(), self.speaker));
                            if ui.add_enabled(self.clip.is_none() && !remaining.is_empty(), egui::Button::new("Use next unrecorded prompt")).clicked() {
                                self.transcript = remaining[0].into();
                                self.status = "Read the prompt, record/import, verify the exact words, then Add clip. Reload this list later to resume.".into();
                            }
                        }
                        Err(e) => {ui.label(e);}
                    }
                }
            });
            ui.label("Exact transcript for the new clip");
            if collection.data.language.is_cyrillic() {ui.label("Optional stress: + before a vowel (з+амок) or a combining acute accent. Match the recording's actual pronunciation.");}
            ui.add(egui::TextEdit::multiline(&mut self.transcript).desired_rows(2).desired_width(f32::INFINITY));
            let mut added = false;
            if let Some(clip) = &self.clip {
                let duration = clip.samples.len() as f64 / f64::from(clip.sample_rate);
                waveform(ui, clip, self.trim_start, self.trim_end);
                ui.horizontal(|ui| {
                    ui.label("Start (s)"); ui.add(egui::DragValue::new(&mut self.trim_start).range(0.0..=duration).speed(0.01));
                    ui.label("End (s)"); ui.add(egui::DragValue::new(&mut self.trim_end).range(0.0..=duration).speed(0.01));
                    ui.label(format!("Selected {:.2}s", self.trim_end - self.trim_start));
                    if ui.button("Preview trim").clicked() {
                        match clips::trim(clip, self.trim_start, self.trim_end) { Ok(audio) => action = Some(Action::Play(audio)), Err(e) => self.status = e }
                    }
                    if ui.button("Add clip").clicked() {
                        match clips::trim(clip, self.trim_start, self.trim_end).and_then(|audio| collection.add_clip(&audio, &self.transcript, &self.speaker, self.split)) {
                            Ok(id) => { self.dirty = false; self.status = format!("Saved {id}"); added = true; }, Err(e) => self.status = e,
                        }
                    }
                });
                if ui.button("Discard current take").clicked() { added = true; }
            }
            if added { self.clip = None; self.transcript.clear(); }
            ui.separator();
            egui::ScrollArea::vertical().id_salt("clips").max_height(130.0).show(ui, |ui| {
                for (i, clip) in collection.data.utterances.iter().enumerate() {
                    if ui.selectable_label(self.selected == Some(i), format!("{} • {:?} • {}", clip.id, clip.split, clip.text)).clicked() { self.selected = Some(i); }
                }
            });
            if let Some(index) = self.selected && let Some(clip) = collection.data.utterances.get_mut(index) {
                ui.horizontal(|ui| {
                    ui.label("Edit transcript"); if ui.add(egui::TextEdit::singleline(&mut clip.text).desired_width(450.0)).changed() { self.dirty = true; }
                    let before = clip.split; split_choice(ui, "edit-clip-split", &mut clip.split); self.dirty |= before != clip.split;
                });
            }
            ui.horizontal(|ui| {
                if ui.button("Assign held-out splits").clicked() {
                    match collection.assign_splits(1234) { Ok(()) => { self.dirty = true; self.status = "Splits assigned; Save edits to apply. Identical prompts stay together.".into(); }, Err(e) => self.status = e }
                }
                if ui.button("Check saved dataset").clicked() {
                    if self.dirty { self.status = "Save edits before checking.".into(); }
                    else {
                        let report = dataset::validate(&collection.data, collection.root());
                        self.status = format!("{:.1} minutes • {} errors • {} warnings", report.duration_seconds / 60.0, report.errors.len(), report.warnings.len());
                        self.status.push_str(&format!("\n{}", report.statistics_text()));
                        for problem in report.errors.iter().chain(report.warnings.iter()) { self.status.push_str(&format!("\n{problem}")); }
                    }
                }
                if ui.button("Use for training").clicked() {
                    if self.dirty { self.status = "Save edits before training.".into(); }
                    else { action = Some(Action::Train(collection.manifest().to_path_buf())); }
                }
            });
            if ui.button(if self.dirty { "Save edits *" } else { "Save edits" }).clicked() { self.save(); }
            if ui.add_enabled(self.clip.is_none(), egui::Button::new("Discard manifest edits and reload")).clicked() { reload = true; }
        });
        if reload {
            self.reload_saved();
        }
        ui.label(&self.status);
        action
    }
}
fn language_choice(ui: &mut egui::Ui, language: &mut Language) {
    egui::ComboBox::from_id_salt("dataset-language")
        .selected_text(language.label())
        .show_ui(ui, |ui| {
            for choice in Language::ALL {
                ui.selectable_value(language, choice, choice.label());
            }
        });
}
fn split_choice(ui: &mut egui::Ui, id: &str, split: &mut Split) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(format!("{split:?}"))
        .show_ui(ui, |ui| {
            ui.selectable_value(split, Split::Train, "Train");
            ui.selectable_value(split, Split::Validation, "Validation");
            ui.selectable_value(split, Split::Test, "Test");
        });
}
fn waveform(ui: &mut egui::Ui, audio: &Audio, start: f64, end: f64) {
    let (response, painter) =
        ui.allocate_painter(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
    let rect = response.rect;
    let duration = audio.samples.len() as f64 / f64::from(audio.sample_rate);
    painter.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    let selection = egui::Rect::from_min_max(
        egui::pos2(
            rect.left() + rect.width() * (start / duration) as f32,
            rect.top(),
        ),
        egui::pos2(
            rect.left() + rect.width() * (end / duration) as f32,
            rect.bottom(),
        ),
    );
    painter.rect_filled(
        selection.intersect(rect),
        0.0,
        ui.visuals().selection.bg_fill,
    );
    let bins = (rect.width() as usize).max(1);
    for x in 0..bins {
        let from = x * audio.samples.len() / bins;
        let to = ((x + 1) * audio.samples.len() / bins)
            .max(from + 1)
            .min(audio.samples.len());
        let peak = audio.samples[from..to]
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        let height = peak * rect.height() / 2.0;
        painter.line_segment(
            [
                egui::pos2(rect.left() + x as f32, rect.center().y - height),
                egui::pos2(rect.left() + x as f32, rect.center().y + height),
            ],
            egui::Stroke::new(1.0, ui.visuals().text_color()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reload_recovers_external_edits_without_overwriting_them() {
        let root = std::env::temp_dir().join(format!("tts-editor-reload-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut editor = Editor::new(&root);
        // Use a uniquely named child so existing temp files cannot be overwritten.
        let folder = root.join(format!(
            "{:?}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        editor.collection = Some(Collection::create(&folder, Language::En).unwrap());
        let path = editor.collection.as_ref().unwrap().manifest().to_path_buf();
        let mut external = Collection::open(&path).unwrap();
        external.data.rights.audio = "external saved rights".into();
        external.save().unwrap();
        editor.collection.as_mut().unwrap().data.rights.audio = "unsaved edit".into();
        editor.dirty = true;
        editor.save();
        assert!(editor.dirty);
        assert!(editor.status.contains("outside Studio"));
        editor.reload_saved();
        assert!(!editor.dirty);
        assert_eq!(
            editor.collection.as_ref().unwrap().data.rights.audio,
            "external saved rights"
        );
        std::fs::write(&path, "invalid").unwrap();
        editor.dirty = true;
        editor.reload_saved();
        assert!(editor.dirty);
        assert_eq!(
            editor.collection.as_ref().unwrap().data.rights.audio,
            "external saved rights"
        );
        std::fs::remove_dir_all(folder).unwrap();
    }
}

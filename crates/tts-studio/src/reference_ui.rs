//! Reference recording import, trim, preview and save without blocking egui.
use eframe::egui;
use local_tts_core::{Audio, clips};
use std::{
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
};

enum ResultEvent {
    Imported(Result<(PathBuf, Audio), String>),
    Saved(Result<PathBuf, String>),
}
pub enum Action {
    Play(Audio),
    Select(PathBuf),
}
pub struct Panel {
    source: Option<PathBuf>,
    audio: Option<Arc<Audio>>,
    start: f64,
    end: f64,
    pending: bool,
    status: String,
    tx: mpsc::Sender<ResultEvent>,
    rx: mpsc::Receiver<ResultEvent>,
}
impl Default for Panel {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            source: None,
            audio: None,
            start: 0.0,
            end: 0.0,
            pending: false,
            status: String::new(),
            tx,
            rx,
        }
    }
}
impl Panel {
    pub fn pending(&self) -> bool {
        self.pending
    }
    fn selected(&self) -> Result<Audio, String> {
        let audio = self.audio.as_ref().ok_or("Import a recording first")?;
        if self.end - self.start < 0.1 || self.end - self.start > 30.0 {
            return Err("Select between 0.1 and 30 seconds of speech".into());
        }
        let audio = clips::trim(audio, self.start, self.end)?;
        if !audio.samples.iter().any(|s| s.abs() > 0.0001) {
            return Err("Selected recording is silent".into());
        }
        Ok(audio)
    }
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let mut action = None;
        while let Ok(event) = self.rx.try_recv() {
            self.pending = false;
            match event {
                ResultEvent::Imported(Ok((path, audio))) => {
                    self.start = 0.0;
                    self.end = (audio.samples.len() as f64 / audio.sample_rate as f64).min(15.0);
                    self.source = Some(path);
                    self.audio = Some(Arc::new(audio));
                    self.status =
                        "Choose a clean 5–15 second speech segment, preview it, then save.".into();
                }
                ResultEvent::Saved(Ok(path)) => {
                    self.status = format!("Saved {}", path.display());
                    action = Some(Action::Select(path));
                }
                ResultEvent::Imported(Err(e)) | ResultEvent::Saved(Err(e)) => self.status = e,
            }
        }
        egui::CollapsingHeader::new("Prepare / trim speaker reference").show(ui, |ui| {
            ui.label("Import mono/stereo WAV, select clean speech and save a mono 24 kHz reference. Originals are preserved.");
            ui.add_enabled_ui(!self.pending, |ui| {
                if ui.button("Import recording…").clicked()
                    && let Some(path) = rfd::FileDialog::new().add_filter("WAV", &["wav"]).pick_file()
                {
                    self.pending = true; self.audio = None; self.source = None;
                    let tx = self.tx.clone();
                    thread::spawn(move || {
                        let result = clips::import_wav_at_rate(&path, 24000).map(|audio| (path, audio));
                        let _ = tx.send(ResultEvent::Imported(result));
                    });
                }
                if let Some(audio) = &self.audio {
                    let duration = audio.samples.len() as f64 / audio.sample_rate as f64;
                    ui.label(format!("{} • {:.2} seconds", self.source.as_ref().unwrap().display(), duration));
                    ui.horizontal(|ui| {
                        ui.label("Start (seconds)"); ui.add(egui::DragValue::new(&mut self.start).speed(0.1).range(0.0..=duration));
                        ui.label("End (seconds)"); ui.add(egui::DragValue::new(&mut self.end).speed(0.1).range(0.0..=duration));
                    });
                    ui.label(format!("Selected: {:.2} seconds", self.end - self.start));
                    ui.horizontal(|ui| {
                        if ui.button("Preview selection").clicked() {
                            match self.selected() { Ok(audio) => action = Some(Action::Play(audio)), Err(e) => self.status = e }
                        }
                        if ui.button("Save and use reference…").clicked() {
                            match self.selected() {
                                Err(e) => self.status = e,
                                Ok(audio) => if let Some(path) = rfd::FileDialog::new().add_filter("WAV", &["wav"]).set_file_name("speaker-reference.wav").save_file() {
                                    if self.source.as_ref().and_then(|p| p.canonicalize().ok()) == path.canonicalize().ok() && path.exists() {
                                        self.status = "Choose a different file to preserve the original".into();
                                    } else {
                                        self.pending = true; let tx = self.tx.clone();
                                        thread::spawn(move || {
                                            let result = audio.write_wav(&path).map(|_| path).map_err(|e| e.to_string());
                                            let _ = tx.send(ResultEvent::Saved(result));
                                        });
                                    }
                                }
                            }
                        }
                    });
                }
            });
            if self.pending { ui.spinner(); }
            ui.label(&self.status);
        });
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trimmed_selection_is_bounded_and_source_is_preserved() {
        let source = Arc::new(Audio {
            sample_rate: 24000,
            samples: vec![0.1; 24000 * 40],
        });
        let mut panel = Panel {
            audio: Some(source.clone()),
            start: 5.0,
            end: 15.0,
            ..Panel::default()
        };
        assert_eq!(panel.selected().unwrap().samples.len(), 24000 * 10);
        assert_eq!(source.samples.len(), 24000 * 40);
        panel.end = 36.0;
        assert!(panel.selected().is_err());
        panel.end = 4.0;
        assert!(panel.selected().is_err());
    }
}

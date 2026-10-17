use eframe::egui;
use local_tts_core::{Language, voice_profile::VoiceProfile};
use std::{path::PathBuf, sync::mpsc, thread};
#[derive(Default)]
pub struct Panel {
    name: String,
    status: String,
    receiver: Option<mpsc::Receiver<Result<(PathBuf, VoiceProfile), String>>>,
}
pub struct Selection {
    pub reference: PathBuf,
    pub profile: VoiceProfile,
}
impl Panel {
    pub fn pending(&self) -> bool {
        self.receiver.is_some()
    }
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        reference: &str,
        language: Language,
        exaggeration: f32,
    ) -> Option<Selection> {
        let mut selection = None;
        if let Some(rx) = &self.receiver
            && let Ok(result) = rx.try_recv()
        {
            self.receiver = None;
            match result {
                Ok((folder, profile)) => {
                    self.name = profile.name.clone();
                    self.status = format!("Profile ready: {}", folder.display());
                    selection = Some(Selection {
                        reference: folder.join("reference.wav"),
                        profile,
                    });
                }
                Err(error) => self.status = error,
            }
        }
        egui::CollapsingHeader::new("Saved voice profiles").show(ui, |ui| {
            ui.label("A profile stores a copy of your reference, its checksum, language and expressiveness. Model weights are installed separately.");
            ui.add_enabled_ui(!self.pending(), |ui| {
                ui.horizontal(|ui| { ui.label("Profile name"); ui.text_edit_singleline(&mut self.name); });
                ui.horizontal(|ui| {
                    if ui.add_enabled(!reference.trim().is_empty(), egui::Button::new("Save new profile…")).clicked()
                        && let Some(folder) = rfd::FileDialog::new().set_title("Choose a NEW profile folder name").set_file_name("my-voice-profile").save_file()
                    {
                        let reference = PathBuf::from(reference); let name = self.name.clone();
                        let (tx, rx) = mpsc::channel(); self.receiver = Some(rx);
                        thread::spawn(move || { let result = VoiceProfile::create(&folder, &reference, &name, language, exaggeration).map(|profile| (folder, profile)); let _ = tx.send(result); });
                    }
                    if ui.button("Open profile…").clicked()
                        && let Some(folder) = rfd::FileDialog::new().pick_folder()
                    {
                        let (tx, rx) = mpsc::channel(); self.receiver = Some(rx);
                        thread::spawn(move || { let result = VoiceProfile::load(&folder).map(|profile| (folder, profile)); let _ = tx.send(result); });
                    }
                });
            });
            if self.pending() { ui.spinner(); }
            ui.label(&self.status);
        });
        selection
    }
}

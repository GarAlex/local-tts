use eframe::egui;
use local_tts_core::{
    Language,
    pronunciations::{FILE_NAME, PronunciationFile},
};
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct Editor {
    key: Option<(PathBuf, Language)>,
    file: Option<PronunciationFile>,
    form: String,
    status: String,
}
impl Editor {
    pub fn sync(&mut self, root: &Path, language: Language) {
        let key = (root.join(FILE_NAME), language);
        if self.key.as_ref() != Some(&key) {
            self.key = Some(key);
            self.form.clear();
            self.status.clear();
            if let Err(error) = self.reload() {
                self.status = error;
            }
        }
    }
    pub fn reload(&mut self) -> Result<(), String> {
        let (path, language) = self.key.as_ref().ok_or("select a Cyrillic voice first")?;
        match PronunciationFile::open(path, *language) {
            Ok(file) => {
                self.file = Some(file);
                Ok(())
            }
            Err(error) => {
                self.file = None;
                Err(error)
            }
        }
    }
    pub fn prepare(&self, text: &str) -> Result<String, String> {
        let (_, language) = self.key.as_ref().ok_or("select a Cyrillic voice first")?;
        let file = self
            .file
            .as_ref()
            .ok_or("reload or fix the pronunciation dictionary before generating")?;
        let text = local_tts_core::text::expand(text, *language)?;
        let corrected = file.data().apply(&text)?;
        local_tts_core::context::stress(&corrected, *language)
    }
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Saved word stress").show(ui,|ui| {
            ui.label("Save a correction once, then type the word normally. Explicit stress in a sentence takes precedence.");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.form).hint_text("One word, e.g. прив+іт or вал+ерия").desired_width(360.0));
                if ui.add_enabled(self.file.is_some(),egui::Button::new("Save word")).clicked() {
                    let result=(||->Result<String,String> {
                        let file=self.file.as_mut().ok_or("reload the dictionary first")?;
                        let mut data=file.data().clone(); let word=data.insert(&self.form)?; file.save(data)?;
                        Ok(format!("Saved {word}"))
                    })();
                    match result { Ok(message)=> { self.status=message; self.form.clear(); }, Err(error)=>self.status=error }
                }
                if ui.button("Reload corrections").clicked() {
                    self.status=match self.reload() { Ok(())=>"Reloaded saved corrections".into(), Err(error)=>error };
                }
            });
            if let Some(file)=&self.file {
                let rows:Vec<_>=file.data().entries().map(|(key,value)|(key.to_string(),value.to_string())).collect();
                ui.label(format!("{} saved word(s)",rows.len()));
                let mut remove=None;
                egui::ScrollArea::vertical().id_salt("stress-entries").max_height(100.0).show(ui,|ui| {
                    for (key,value) in rows { ui.horizontal(|ui| { ui.label(format!("{key} → {value}")); if ui.button("Remove").clicked() { remove=Some(key); } }); }
                });
                if let Some(word)=remove {
                    let result=(||->Result<(),String> { let file=self.file.as_mut().unwrap(); let mut data=file.data().clone(); data.remove(&word)?; file.save(data) })();
                    self.status=match result { Ok(())=>format!("Removed {word}"), Err(error)=>error };
                }
            }
            if !self.status.is_empty() { ui.label(&self.status); }
            ui.label("Corrections are saved immediately in this voice folder. Each word form needs its own entry.");
        });
    }
}

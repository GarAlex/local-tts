//! Project checkpoint preflight and held-out listening comparisons.
use eframe::egui;
use local_tts_core::{
    dataset::Split,
    evaluation::{AudioFile, Comparison},
};
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Deserialize)]
pub struct Checkpoint {
    pub checkpoint: String,
    pub sha256: String,
    pub step: usize,
    pub language: String,
    pub frontend: String,
    pub speaker: String,
    pub sample_rate: u32,
    pub symbols: usize,
    pub model_profile: String,
    pub discriminator: String,
    pub notes: Vec<String>,
}
#[derive(Deserialize)]
pub struct Preflight {
    #[serde(flatten)]
    pub base: Checkpoint,
    pub dataset: String,
    pub dataset_sha256: String,
    pub target_speaker: String,
    pub clips: BTreeMap<String, usize>,
}
#[derive(Deserialize)]
pub struct Result {
    pub directory: PathBuf,
    pub report: Comparison,
    pub base: Checkpoint,
    pub target: Checkpoint,
}
impl Checkpoint {
    fn ui(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "{} • {} • {} Hz • {} symbols • step {}",
            self.language, self.frontend, self.sample_rate, self.symbols, self.step
        ));
        ui.label(format!(
            "Speaker: {} • architecture: {} • discriminator: {}",
            self.speaker, self.model_profile, self.discriminator
        ));
        ui.collapsing("Checkpoint identity", |ui| {
            ui.label(&self.checkpoint);
            ui.monospace(&self.sha256);
        });
        for note in &self.notes {
            ui.label(note);
        }
    }
}
#[derive(Deserialize)]
pub struct Exported {
    pub voice: PathBuf,
    pub checkpoint_sha256: String,
}
#[derive(Default)]
pub struct Panel {
    pub preflight: Option<Preflight>,
    pub open: bool,
    target: String,
    split_test: bool,
    limit: usize,
    pub result: Option<Result>,
    pub exported: Option<Exported>,
}
pub enum Action {
    Compare {
        checkpoint: String,
        split: Split,
        limit: usize,
    },
    Play(PathBuf, AudioFile),
    Export {
        checkpoint: String,
        sha256: String,
        output: PathBuf,
    },
    Test(PathBuf),
    Keep {
        checkpoint: PathBuf,
        sha256: String,
        step: usize,
    },
}
impl Panel {
    pub fn preflight_ui(
        &self,
        ui: &mut egui::Ui,
        base: &str,
        dataset: &str,
        root: &std::path::Path,
    ) {
        if let Some(report) = &self.preflight
            && std::path::Path::new(&report.base.checkpoint) == root.join(base)
            && std::path::Path::new(&report.dataset) == root.join(dataset)
        {
            ui.collapsing("Base preflight passed (snapshot)", |ui| {
                report.base.ui(ui);
                ui.label(format!(
                    "Target: {} • train {} / validation {} / test {}",
                    report.target_speaker,
                    report.clips.get("train").unwrap_or(&0),
                    report.clips.get("validation").unwrap_or(&0),
                    report.clips.get("test").unwrap_or(&0)
                ));
                ui.monospace(format!("Dataset SHA-256: {}", report.dataset_sha256));
                ui.label(
                    "Preflight describes the checked snapshot; training validates inputs again.",
                );
            });
        }
    }
    pub fn show(&mut self, ctx: &egui::Context, output: &str, busy: bool) -> Option<Action> {
        if !self.open {
            return None;
        }
        if self.target.is_empty() {
            self.target = PathBuf::from(output)
                .join("latest.pt")
                .display()
                .to_string();
        }
        if self.limit == 0 {
            self.limit = 3;
        }
        let mut action = None;
        egui::Window::new("Evaluate fine-tuning").open(&mut self.open).default_width(710.0).vscroll(true).show(ctx,|ui| {
            ui.add_enabled_ui(!busy,|ui| {
                super::path_field(ui,"Target checkpoint to compare",&mut self.target,Some("pt"));
                ui.horizontal(|ui| {
                    if ui.button("Use latest checkpoint").clicked() {self.target=PathBuf::from(output).join("latest.pt").display().to_string();}
                    ui.radio_value(&mut self.split_test,false,"Validation"); ui.radio_value(&mut self.split_test,true,"Test");
                    ui.label("Clips");ui.add(egui::DragValue::new(&mut self.limit).range(1..=12));
                });
                ui.label("Uses the training tab's base and dataset. Same held-out text, original recording, base voice and adapted voice. Duration is fixed at 1.0. Each evaluation creates a new local report.");
                if ui.button("Generate comparison").clicked() {
                    self.result=None; self.exported=None;
                    action=Some(Action::Compare { checkpoint:self.target.clone(), split:if self.split_test {Split::Test}else{Split::Validation},limit:self.limit });
                }
            });
            if let Some(result)=&self.result {
                ui.separator();ui.label(format!("Compared {} clips • {:?} • {}",result.report.samples.len(),result.report.split,result.report.frontend));
                ui.collapsing("Base checkpoint",|ui| result.base.ui(ui));
                ui.collapsing("Adapted checkpoint",|ui| result.target.ui(ui));
                if ui.add_enabled(!busy,egui::Button::new("Keep evaluated checkpoint")).clicked() {
                    action=Some(Action::Keep {checkpoint:PathBuf::from(&result.target.checkpoint),sha256:result.target.sha256.clone(),step:result.target.step});
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(!busy && self.exported.is_none(),egui::Button::new("Export evaluated checkpoint")).clicked() {
                        action=Some(Action::Export {checkpoint:result.target.checkpoint.clone(),sha256:result.target.sha256.clone(),output:result.directory.parent().unwrap_or(&result.directory).join("exported-voice")});
                    }
                    if let Some(exported)=&self.exported
                        && exported.checkpoint_sha256==result.target.sha256
                        && ui.add_enabled(!busy,egui::Button::new("Test evaluated voice")).clicked() {
                        action=Some(Action::Test(exported.voice.clone()));
                    }
                });
                ui.label("Export is bound to this evaluated hash and creates a new package. A changed checkpoint is rejected; an existing package is retained.");
                for warning in &result.report.warnings { ui.label(format!("Dataset warning: {warning}")); }
                ui.label("RMS, peak and clipping describe audio levels. Listen for intelligibility, speaker character and pronunciation; these values do not score voice quality.");
                for sample in &result.report.samples {
                    ui.separator();ui.label(format!("{}: {}",sample.id,sample.text));
                    ui.horizontal(|ui| {
                        for (label,file) in [("Recording",&sample.reference),("Before",&sample.before),("After",&sample.after)] {
                            if ui.button(format!("Play {label}")).clicked() { action=Some(Action::Play(result.directory.clone(),AudioFile {
                                file:file.file.clone(),sha256:file.sha256.clone(),seconds:file.seconds,rms:file.rms,peak:file.peak,clipped_fraction:file.clipped_fraction })); }
                        }
                    });
                    for (label,file) in [("Recording",&sample.reference),("Before",&sample.before),("After",&sample.after)] {
                        ui.label(format!("{label}: {:.2}s • RMS {:.4} • peak {:.4} • clipped {:.2}%",file.seconds,file.rms,file.peak,100.0*file.clipped_fraction));
                    }
                }
                ui.label(format!("Saved: {}",result.directory.display()));
            }
        });
        action
    }
}

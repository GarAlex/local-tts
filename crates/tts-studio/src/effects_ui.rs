use eframe::egui;
use local_tts_core::{
    Audio,
    effects::{Stats, VoiceEffects},
};
use std::sync::Arc;

#[derive(Default)]
pub struct Panel {
    pub settings: VoiceEffects,
    pub revision: u64,
    original: Option<Arc<Audio>>,
    processed: Option<Arc<Audio>>,
    applied: Option<VoiceEffects>,
    stats: Option<Stats>,
    use_processed: bool,
}
pub enum Action {
    Apply,
    Select(Arc<Audio>),
}
impl Panel {
    pub fn clear(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.original = None;
        self.processed = None;
        self.applied = None;
        self.stats = None;
        self.use_processed = false;
    }
    pub fn set_original(&mut self, audio: Arc<Audio>) {
        self.clear();
        self.original = Some(audio);
    }
    pub fn source(&self) -> Option<Arc<Audio>> {
        self.original.clone()
    }
    pub fn accept(
        &mut self,
        revision: u64,
        settings: VoiceEffects,
        audio: Arc<Audio>,
        stats: Stats,
    ) -> bool {
        if revision != self.revision || self.original.is_none() {
            return false;
        }
        self.processed = Some(audio);
        self.applied = Some(settings);
        self.stats = Some(stats);
        self.use_processed = true;
        true
    }
    pub fn selected(&self) -> Option<Arc<Audio>> {
        if self.use_processed {
            self.processed.clone()
        } else {
            self.original.clone()
        }
    }
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let mut action = None;
        egui::CollapsingHeader::new("Voice effects").default_open(true).show(ui, |ui| {
            ui.label("Apply to generated speech. Every pass starts from the original.");
            ui.add(egui::Slider::new(&mut self.settings.bass_db, -12.0..=12.0).suffix(" dB").text("Bass / warmth"));
            ui.add(egui::Slider::new(&mut self.settings.brightness_db, -12.0..=12.0).suffix(" dB").text("Brightness"));
            ui.add(egui::Slider::new(&mut self.settings.compression, 0.0..=1.0).text("Compression"));
            ui.add(egui::Slider::new(&mut self.settings.pitch_semitones, -12.0..=12.0).suffix(" st").text("Pitch"));
            ui.add(egui::Slider::new(&mut self.settings.formant_semitones, -6.0..=6.0).suffix(" st").text("Formants"));
            ui.label("Pitch preserves duration and compensates formants. Formants change vocal resonance.");
            ui.horizontal(|ui| {
                if ui.add_enabled(self.original.is_some(), egui::Button::new("Apply effects")).clicked() { action = Some(Action::Apply); }
                if ui.button("Reset controls").clicked() { self.settings = VoiceEffects::default(); }
            });
            let previous = self.use_processed;
            ui.horizontal(|ui| {
                ui.add_enabled_ui(self.original.is_some(), |ui| { ui.radio_value(&mut self.use_processed, false, "Original"); });
                ui.add_enabled_ui(self.processed.is_some(), |ui| { ui.radio_value(&mut self.use_processed, true, "Processed"); });
            });
            if previous != self.use_processed && let Some(audio) = self.selected() { action = Some(Action::Select(audio)); }
            if self.applied.is_some_and(|settings| settings != self.settings) {
                ui.label("Controls changed — Apply effects to update processed audio.");
            }
            if let Some(stats) = self.stats {
                ui.label(format!("Peak {:.3} → {:.3} • protection {:.2} dB • compression up to {:.2} dB",
                    stats.input_peak, stats.output_peak, 20.0 * stats.peak_gain.log10(), stats.max_compression_db));
            }
            ui.label("Play and Save WAV use the selected version. Age presets and learned moods are planned.");
        });
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_pass_uses_original_and_stale_results_are_ignored() {
        let original = Arc::new(Audio {
            sample_rate: 24000,
            samples: vec![0.2; 10],
        });
        let settings = VoiceEffects {
            bass_db: 6.0,
            ..Default::default()
        };
        let (processed, stats) = local_tts_core::effects::process(&original, settings).unwrap();
        let processed = Arc::new(processed);
        let mut panel = Panel::default();
        panel.set_original(original.clone());
        let revision = panel.revision;
        assert!(panel.accept(revision, settings, processed.clone(), stats));
        assert!(Arc::ptr_eq(&panel.selected().unwrap(), &processed));
        assert!(Arc::ptr_eq(&panel.source().unwrap(), &original));
        panel.use_processed = false;
        assert!(Arc::ptr_eq(&panel.selected().unwrap(), &original));
        panel.set_original(original.clone());
        assert!(!panel.accept(revision, settings, processed, stats));
        assert!(Arc::ptr_eq(&panel.selected().unwrap(), &original));
        panel.clear();
        assert!(panel.source().is_none() && panel.selected().is_none());
    }
}

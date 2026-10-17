use eframe::egui;
use local_tts_core::dataset::{Dataset, Split};

#[derive(Default)]
pub struct Guide {
    step: usize,
}

struct Page {
    title: &'static str,
    introduction: &'static str,
    instructions: &'static [&'static str],
    reminder: &'static str,
}

const PAGES: [Page; 6] = [
    Page {
        title: "Set up",
        introduction: "A voice dataset pairs short recordings with the exact words spoken. You will use these pairs to fine-tune a compatible base model to your voice.",
        instructions: &[
            "Choose a new Dataset folder, choose the language of your recordings, then Create dataset. Keep one language and one speaker in each dataset; keep the same Speaker name for every clip.",
            "Use a quiet room and a fixed microphone position. Make a short practice take and listen for background noise, echo and distortion. Speak naturally and consistently; do not add music, bass, pitch effects or noise effects to training recordings.",
            "Keep the whole dataset folder together: dataset.json stores the transcripts and audio/ stores the recordings. Back up this folder after each session. Open manifest… lets you continue later.",
        ],
        reminder: "This guide explains the controls below. Back/Next only changes the instructions; it does not create files, record audio or mark tasks complete.",
    },
    Page {
        title: "Document permission",
        introduction: "Open Recording/transcript rights and speaker permission. Describe the actual source and permission in each field, then Save edits.",
        instructions: &[
            "Audio rights: describe who made and owns the recordings. If true, an example is ‘Recorded by me; I own these recordings.’ For supplied audio, record the source and agreement reference.",
            "Transcript rights: describe where the words came from. If true, use ‘Original sentences written by me’ or ‘Bundled MIT starter prompts’. Publicly available text is not automatically cleared for your use.",
            "Speaker permission: describe permission to train and use the synthetic voice commercially. For your own voice, state that it is your voice and you authorize this use; for another speaker, keep their agreement and reference it here.",
        ],
        reminder: "Examples are not automatic declarations. Only enter statements that are true. Keep agreements separately and avoid private personal details in the manifest; the checker only checks that these fields are filled in.",
    },
    Page {
        title: "Choose sentences",
        introduction: "First practice the full recording loop with the built-in prompts, then build a larger collection of varied sentences for your intended use.",
        instructions: &[
            "Expand Guided recording prompts, press Load starter prompts, then Use next unrecorded prompt. This fills Exact transcript for the new clip with the sentence to read.",
            "For your own scripts, prepare a UTF-8 .txt file with one sentence per line, then Import prompt list…. Use short, natural sentences with varied sounds, words and punctuation. Write numbers as the words you intend to say.",
            "Keep the prompt file. When returning, Open manifest…, select the same Speaker, reload the list and use the next unrecorded prompt. Progress comes from saved transcripts, so a corrected transcript may make a prompt appear again.",
        ],
        reminder: "There are 12 starter prompts per language. They teach the workflow, but are not enough for a usable voice or balanced sound coverage. Recording time and voice quality matter more than a clip count alone.",
    },
    Page {
        title: "Record and review",
        introduction: "Repeat this loop for each sentence. Adding a clip saves its audio and transcript to your dataset.",
        instructions: &[
            "Leave the new clip split on Train. Read the displayed sentence using Record microphone, then Stop recording; capture stops automatically at 20 seconds. The operating system may ask for microphone permission. Alternatively, Import WAV… from an existing recording.",
            "Set Start (s) and End (s) to remove long pauses while keeping the complete first and last sounds. The selected take must be 0.5–20 seconds. Imports are converted to mono 22050 Hz automatically.",
            "Press Preview trim and listen. Exact transcript must match every spoken word. Correct the transcript if you changed the wording; retake recordings with noise, distortion, interruptions or a delivery you do not want the model to learn. Use Discard current take to try again.",
            "Press Add clip, then Use next unrecorded prompt. For Russian/Ukrainian, + before a vowel or an acute accent can mark stress; do not say the marker aloud. Annotations are optional and should match the actual pronunciation consistently.",
        ],
        reminder: "Recording happens only while active, and audio stays local. Take breaks and keep the same room, microphone position and speaking style across sessions. Keep original source files when importing.",
    },
    Page {
        title: "Split and check",
        introduction: "Reserve some recordings so you can check how well the model speaks sentences it did not learn from.",
        instructions: &[
            "Train clips teach the model. Validation clips help monitor progress and compare checkpoints. Test clips are reserved for a final assessment. Do not move evaluation clips into Train just to improve an evaluation score.",
            "After collecting clips, press Assign held-out splits, then Save edits. This keeps identical prompts together and reserves roughly 10% each for validation and test. Two distinct prompts allow train/validation; at least three allow a test group too. These are structural minimums, not enough data for a good voice.",
            "Press Check saved dataset. Read the errors, warnings and valid minutes per split shown below the controls. Fix missing rights, unreadable/silent audio or incorrect transcripts, save changes and check again. Select an existing clip to edit its transcript or split.",
        ],
        reminder: "When adding more clips, review the splits again. Passing this checker does not judge pronunciation, recording quality, true permission or compatibility with a base model; listening and the training compatibility check are still needed.",
    },
    Page {
        title: "Fine-tune your voice",
        introduction: "When the saved dataset checks out, press Use for training to send its manifest to the Train a voice tab.",
        instructions: &[
            "Choose a compatible training base for your dataset language and keep fine-tuning as the training mode. Test-speech voices are not all training bases: Kokoro is currently for inference and cannot be fine-tuned here.",
            "Run Check dataset and Check base and target compatibility in the training tab. Resolve reported problems before Start training. Start a separate target run; use resume later to continue that same run.",
            "Compare adapted speech against the original recordings and base voice using held-out sentences. Listen for pronunciation, intelligibility and voice similarity, then export and test the voice in the native library. A short practice run verifies the workflow; it does not establish usable voice quality.",
        ],
        reminder: "A compatible base still has its own weight and data terms. Your dataset permission does not replace those terms. Keep provenance with the model you intend to distribute.",
    },
];

impl Guide {
    pub fn ui(&mut self, ui: &mut egui::Ui, dataset: Option<&Dataset>, unsaved: bool, take: bool) {
        egui::CollapsingHeader::new("First dataset: step-by-step guide")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (index, page) in PAGES.iter().enumerate() {
                        if ui.selectable_label(self.step == index, format!("{}. {}", index + 1, page.title)).clicked() {
                            self.step = index;
                        }
                    }
                });
                let page = &PAGES[self.step];
                ui.label(egui::RichText::new(page.introduction).strong());
                for instruction in page.instructions {
                    ui.horizontal_top(|ui| {
                        ui.label("•");
                        ui.add(egui::Label::new(*instruction).wrap());
                    });
                }
                ui.add(egui::Label::new(page.reminder).wrap());
                ui.horizontal(|ui| {
                    if ui.add_enabled(self.step > 0, egui::Button::new("Back")).clicked() { self.step -= 1; }
                    ui.label(format!("Step {} of {}", self.step + 1, PAGES.len()));
                    if ui.add_enabled(self.step + 1 < PAGES.len(), egui::Button::new("Next")).clicked() { self.step += 1; }
                });
                ui.separator();
                if let Some(dataset) = dataset {
                    let count = |split| dataset.utterances.iter().filter(|clip| clip.split == split).count();
                    ui.label(format!(
                        "Current dataset: {} saved clips • {} train / {} validation / {} test",
                        dataset.utterances.len(), count(Split::Train), count(Split::Validation), count(Split::Test)
                    ));
                    if unsaved { ui.label("Manifest changes need Save edits."); }
                    if take { ui.label("Current take has not been added yet; add or discard it before leaving this dataset."); }
                } else {
                    ui.label("No dataset open yet. Start with Create dataset below.");
                }
            });
        ui.separator();
    }
}

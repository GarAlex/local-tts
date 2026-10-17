use eframe::egui;
mod collection_ui;
mod dataset_guide;
mod effects_ui;
mod playback;
mod profile_ui;
mod pronunciation_ui;
mod recording;
mod reference_ui;
mod runtime;
mod training_ui;
use local_tts_core::{
    Audio, Language, SynthesisRequest, dataset,
    kokoro::{Kokoro, Variant as KokoroVariant},
    piper::Piper,
    vits::Vits,
    vosk::Vosk,
};
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

enum Event {
    Log(String),
    Audio(Audio),
    Effects(
        u64,
        local_tts_core::effects::VoiceEffects,
        Arc<Audio>,
        local_tts_core::effects::Stats,
    ),
    Playback(u64, Result<(), String>),
    Preflight(Box<training_ui::Preflight>),
    Comparison(Box<training_ui::Result>),
    EvaluatedExport(training_ui::Exported),
    Done(Result<(), String>),
}
#[derive(PartialEq)]
enum Tab {
    Test,
    Dataset,
    Train,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum TestVoice {
    MathStudent,
    Math,
    Inflect,
    Kitten,
    Design,
    English,
    Own,
    Ukrainian,
    Russian,
    European,
}
impl TestVoice {
    fn language(self) -> Option<Language> {
        match self {
            Self::Russian => Some(Language::Ru),
            Self::Ukrainian => Some(Language::Uk),
            _ => None,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::MathStudent => "Arithmetic neural voice (EN/RU/UK, experimental)",
            Self::Math => "English — Arithmetic (small word bank)",
            Self::Inflect => "English — Inflect Nano (experimental)",
            Self::Kitten => "English — Kitten Nano INT8 (experimental)",
            Self::Design => "Voice design — Qwen3",
            Self::English => "English — Kokoro",
            Self::European => "Multilingual / voice cloning — Chatterbox",
            Self::Own => "My exported VITS voice",
            Self::Ukrainian => "Ukrainian — Piper",
            Self::Russian => "Russian — Vosk",
        }
    }
}
enum Engine {
    MathStudent(local_tts_core::math_speech::MathStudent),
    Math(local_tts_core::math_speech::MathVoice),
    Compact(Box<local_tts_core::compact_english::CompactEnglish>),
    QwenDesign(Box<local_tts_core::qwen_design::QwenDesign>),
    Kokoro(Kokoro),
    Vits(Vits),
    Piper(Piper),
    Vosk(Vosk),
    Chatterbox(Box<local_tts_core::chatterbox::Chatterbox>),
}
struct Studio {
    tab: Tab,
    editor: collection_ui::Editor,
    close_dialog: bool,
    pronunciation_editor: pronunciation_ui::Editor,
    root: PathBuf,
    voice: String,
    reference_voice: String,
    reference_editor: reference_ui::Panel,
    profile_editor: profile_ui::Panel,
    exaggeration: f32,
    voice_description: String,
    text: String,
    phonemes: bool,
    test_voice: TestVoice,
    european_language: Language,
    kokoro_variant: KokoroVariant,
    speaker: String,
    scale: f32,
    dataset: String,
    output: String,
    python: String,
    cli: PathBuf,
    steps: usize,
    batch: usize,
    cuda: bool,
    resume: bool,
    from_scratch: bool,
    base_checkpoint: String,
    evaluation: training_ui::Panel,
    busy: bool,
    synthesis_cancel: Option<Arc<AtomicBool>>,
    status: String,
    logs: Vec<String>,
    tx: mpsc::Sender<Event>,
    rx: mpsc::Receiver<Event>,
    child: Arc<Mutex<Option<Child>>>,
    engine: Arc<Mutex<Option<(String, Engine)>>>,
    audio: Option<Arc<Audio>>,
    effects: effects_ui::Panel,
    playback: playback::Player,
}
impl Studio {
    fn new(runtime: runtime::Runtime) -> Self {
        let root = runtime.root;
        let (tx, rx) = mpsc::channel();
        let environment = root.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        let python = if environment.exists() {
            environment.display().to_string()
        } else if Path::new("/opt/homebrew/bin/python3.12").exists() {
            "/opt/homebrew/bin/python3.12".to_string()
        } else {
            "python3".to_string()
        };
        let playback_tx = tx.clone();
        let playback = playback::Player::new(move |id, result| {
            let _ = playback_tx.send(Event::Playback(id, result));
        });
        Self {
            tab: Tab::Test,
            editor: collection_ui::Editor::new(&root),
            close_dialog: false,
            pronunciation_editor: pronunciation_ui::Editor::default(),
            voice: root.join("voices/kokoro-en").display().to_string(),
            reference_voice: String::new(),
            reference_editor: reference_ui::Panel::default(),
            profile_editor: profile_ui::Panel::default(),
            exaggeration: 0.5,
            voice_description: "A warm, calm adult male voice with a clear British accent.".into(),
            text: "Hello world. This is a local speech test.".into(),
            phonemes: false,
            test_voice: TestVoice::English,
            european_language: Language::Fr,
            kokoro_variant: KokoroVariant::F32,
            speaker: "lada".into(),
            scale: 1.0,
            dataset: String::new(),
            output: root.join("checkpoints/my-voice").display().to_string(),
            python,
            cli: runtime.cli,
            steps: 10000,
            batch: 2,
            cuda: false,
            resume: false,
            from_scratch: false,
            base_checkpoint: String::new(),
            evaluation: training_ui::Panel::default(),
            busy: false,
            synthesis_cancel: None,
            status: "Ready".into(),
            logs: Vec::new(),
            tx,
            rx,
            child: Arc::new(Mutex::new(None)),
            engine: Arc::new(Mutex::new(None)),
            audio: None,
            effects: effects_ui::Panel::default(),
            playback,
            root,
        }
    }
    fn begin(&mut self, status: &str) {
        self.busy = true;
        self.status = status.into();
        self.logs.clear();
    }
    fn process(&mut self, mut command: Command, label: &str) {
        self.begin(label);
        command
            .current_dir(&self.root)
            .env("LOCAL_TTS_CLI", &self.cli)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let tx = self.tx.clone();
        let holder = self.child.clone();
        thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let mut child = command.spawn().map_err(|e| e.to_string())?;
                let stdout = child.stdout.take().unwrap();
                let stderr = child.stderr.take().unwrap();
                let tx_out = tx.clone();
                let tx_err = tx.clone();
                let out = thread::spawn(move || {
                    for line in BufReader::new(stdout).lines() {
                        let line = line.unwrap_or_else(|e| e.to_string());
                        let event = serde_json::from_str::<serde_json::Value>(&line)
                            .ok()
                            .and_then(|value| match value["event"].as_str()? {
                                "preflight" => serde_json::from_value(value)
                                    .ok()
                                    .map(|r| Event::Preflight(Box::new(r))),
                                "comparison" => serde_json::from_value(value)
                                    .ok()
                                    .map(|r| Event::Comparison(Box::new(r))),
                                "evaluated_export" => serde_json::from_value(value)
                                    .ok()
                                    .map(Event::EvaluatedExport),
                                _ => None,
                            });
                        let _ = tx_out.send(event.unwrap_or(Event::Log(line)));
                    }
                });
                let err = thread::spawn(move || {
                    for line in BufReader::new(stderr).lines() {
                        let _ = tx_err.send(Event::Log(line.unwrap_or_else(|e| e.to_string())));
                    }
                });
                *holder.lock().unwrap() = Some(child);
                let status = loop {
                    let status = holder
                        .lock()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .try_wait()
                        .map_err(|e| e.to_string())?;
                    if let Some(status) = status {
                        break status;
                    }
                    thread::sleep(Duration::from_millis(100));
                };
                let _ = out.join();
                let _ = err.join();
                *holder.lock().unwrap() = None;
                if status.success() {
                    Ok(())
                } else {
                    Err(format!("process exited with {status}"))
                }
            })();
            let _ = tx.send(Event::Done(result));
        });
    }
    fn stop(&mut self) {
        if let Some(child) = self.child.lock().unwrap().as_mut() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .args(["-TERM", &child.id().to_string()])
                    .status();
            }
            #[cfg(not(unix))]
            {
                let _ = child.kill();
            }
            self.status = "Stopping; training saves at the next step boundary".into();
        }
    }
    fn generate(&mut self) {
        if let Some(language) = self.test_voice.language() {
            self.pronunciation_editor
                .sync(Path::new(&self.voice), language);
            if let Err(error) = self.pronunciation_editor.reload() {
                self.status = format!("Failed: {error}");
                return;
            }
        }
        let reference_digest =
            if self.test_voice == TestVoice::European && !self.reference_voice.trim().is_empty() {
                let reference = Path::new(&self.reference_voice);
                let name = reference.file_name().and_then(|n| n.to_str());
                match name.and_then(|name| {
                    local_tts_core::integrity::hash_file(
                        reference
                            .parent()
                            .filter(|p| !p.as_os_str().is_empty())
                            .unwrap_or(Path::new(".")),
                        name,
                    )
                    .ok()
                }) {
                    Some(digest) => digest,
                    None => {
                        self.status =
                            "Cannot read reference WAV; choose an accessible recording".into();
                        return;
                    }
                }
            } else {
                String::new()
            };
        self.begin("Generating speech…");
        let cancel = Arc::new(AtomicBool::new(false));
        self.synthesis_cancel = Some(cancel.clone());
        self.audio = None;
        self.effects.clear();
        self.playback.stop();
        let tx = self.tx.clone();
        let cache = self.engine.clone();
        let path = self.voice.clone();
        let reference_voice = self.reference_voice.clone();
        let exaggeration = self.exaggeration;
        let voice_description = self.voice_description.clone();
        let test_voice = self.test_voice;
        let european_language = self.european_language;
        let speaker = self.speaker.clone();
        let key = format!("{test_voice:?}:{path}:{reference_voice}:{reference_digest}");
        let text = self.text.clone();
        let phonemes = self.phonemes;
        let scale = self.scale;
        thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let mut cache = cache.lock().map_err(|e| e.to_string())?;
                if cache.as_ref().is_none_or(|(loaded, _)| loaded != &key) {
                    *cache = Some((
                        key,
                        match test_voice {
                            TestVoice::MathStudent => Engine::MathStudent(
                                local_tts_core::math_speech::MathStudent::load(Path::new(&path))
                                    .map_err(|e| e.to_string())?,
                            ),
                            TestVoice::Math => Engine::Math(
                                local_tts_core::math_speech::MathVoice::load(Path::new(&path))?,
                            ),
                            TestVoice::Inflect | TestVoice::Kitten => Engine::Compact(Box::new(
                                local_tts_core::compact_english::CompactEnglish::load(
                                    Path::new(&path),
                                    if test_voice == TestVoice::Inflect {
                                        local_tts_core::compact_english::CompactKind::Inflect
                                    } else {
                                        local_tts_core::compact_english::CompactKind::Kitten
                                    },
                                )
                                .map_err(|e| e.to_string())?,
                            )),
                            TestVoice::Design => Engine::QwenDesign(Box::new(
                                local_tts_core::qwen_design::QwenDesign::load_cancellable(
                                    Path::new(&path),
                                    &cancel,
                                )
                                .map_err(|e| e.to_string())?,
                            )),
                            TestVoice::European => Engine::Chatterbox(Box::new(
                                local_tts_core::chatterbox::Chatterbox::load_with_reference(
                                    Path::new(&path),
                                    (!reference_voice.trim().is_empty())
                                        .then(|| Path::new(&reference_voice)),
                                )
                                .map_err(|e| e.to_string())?,
                            )),
                            TestVoice::Own => Engine::Vits(
                                Vits::load(Path::new(&path)).map_err(|e| e.to_string())?,
                            ),
                            TestVoice::English => Engine::Kokoro(
                                Kokoro::load(Path::new(&path)).map_err(|e| e.to_string())?,
                            ),
                            TestVoice::Russian => Engine::Vosk(
                                Vosk::load(Path::new(&path)).map_err(|e| e.to_string())?,
                            ),
                            TestVoice::Ukrainian => Engine::Piper(
                                Piper::load(Path::new(&path)).map_err(|e| e.to_string())?,
                            ),
                        },
                    ));
                }
                let engine = &mut cache.as_mut().unwrap().1;
                if let Engine::Chatterbox(model) = engine {
                    model
                        .set_exaggeration(exaggeration)
                        .map_err(|e| e.to_string())?;
                }
                if let Engine::Kokoro(voice) = engine {
                    let _ = tx.send(Event::Log(format!(
                        "Kokoro variant: {}",
                        voice.variant().label()
                    )));
                }
                let prepared = match engine {
                    Engine::Kokoro(voice) if !phonemes => {
                        let _ = tx.send(Event::Log(format!(
                            "Phonemes: {}",
                            voice.phonemize(&text).map_err(|e| e.to_string())?
                        )));
                        Some(local_tts_core::text::prepare(&text, Language::En)?)
                    }
                    Engine::Vosk(voice) => {
                        voice.reload_pronunciations().map_err(|e| e.to_string())?;
                        Some(voice.prepare_text(&text).map_err(|e| e.to_string())?)
                    }
                    Engine::Piper(voice) => {
                        voice.reload_pronunciations().map_err(|e| e.to_string())?;
                        Some(voice.prepare_text(&text).map_err(|e| e.to_string())?)
                    }
                    _ => None,
                };
                if let Some(text) = prepared {
                    let _ = tx.send(Event::Log(format!("Prepared text: {text}")));
                }
                let start = std::time::Instant::now();
                let audio = match engine {
                    Engine::MathStudent(model) => model.synthesize_cancellable(
                        &SynthesisRequest {
                            text: &text,
                            language: model.language(),
                            speaker: "default",
                            duration_scale: scale,
                        },
                        &cancel,
                    ),
                    Engine::Math(model) => model
                        .expression(&text)
                        .map_err(local_tts_core::kokoro::Error),
                    Engine::Compact(model) => model.synthesize_cancellable(
                        &SynthesisRequest {
                            text: &text,
                            language: Language::En,
                            speaker: &speaker,
                            duration_scale: scale,
                        },
                        &cancel,
                    ),
                    Engine::QwenDesign(model) => model
                        .synthesize_design(&text, european_language, &voice_description, &cancel)
                        .map_err(|e| local_tts_core::kokoro::Error(e.to_string())),
                    Engine::Chatterbox(voice) => voice.synthesize_cancellable(
                        &SynthesisRequest {
                            text: &text,
                            language: european_language,
                            speaker: "default",
                            duration_scale: 1.0,
                        },
                        &cancel,
                    ),
                    Engine::Vosk(voice) => voice.synthesize_cancellable(
                        &SynthesisRequest {
                            text: &text,
                            language: Language::Ru,
                            speaker: &speaker,
                            duration_scale: scale,
                        },
                        &cancel,
                    ),
                    Engine::Piper(voice) => voice.synthesize_cancellable(
                        &SynthesisRequest {
                            text: &text,
                            language: Language::Uk,
                            speaker: &speaker,
                            duration_scale: scale,
                        },
                        &cancel,
                    ),
                    Engine::Vits(voice) => {
                        let speaker = voice.manifest.speakers[0].clone();
                        voice.synthesize_cancellable(
                            &SynthesisRequest {
                                text: &text,
                                language: voice.manifest.language,
                                speaker: &speaker,
                                duration_scale: scale,
                            },
                            &cancel,
                        )
                    }
                    Engine::Kokoro(engine) if phonemes => {
                        local_tts_core::paragraph::check_cancel(&cancel)?;
                        let audio = engine.synthesize_phonemes(&text, scale);
                        local_tts_core::paragraph::check_cancel(&cancel)?;
                        audio
                    }
                    Engine::Kokoro(engine) => {
                        let phones = engine.phonemize(&text).map_err(|e| e.to_string())?;
                        let _ = tx.send(Event::Log(format!("Pronunciation: {phones}")));
                        engine.synthesize_cancellable(
                            &SynthesisRequest {
                                text: &text,
                                language: Language::En,
                                speaker: "af_heart",
                                duration_scale: scale,
                            },
                            &cancel,
                        )
                    }
                }
                .map_err(|e| e.to_string())?;
                let duration = audio.samples.len() as f64 / audio.sample_rate as f64;
                let elapsed = start.elapsed().as_secs_f64();
                let _ = tx.send(Event::Log(format!("{duration:.2}s audio generated in {elapsed:.2}s; RTF {:.3} (model load excluded)", elapsed / duration)));
                let _ = tx.send(Event::Audio(audio));
                Ok(())
            })();
            let _ = tx.send(Event::Done(result));
        });
    }
    fn validate(&mut self) {
        self.begin("Checking dataset…");
        let tx = self.tx.clone();
        let path = self.dataset.clone();
        thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let path = Path::new(&path);
                let data = dataset::read(path).map_err(|e| e.to_string())?;
                let root = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let report = dataset::validate(&data, root);
                let _ = tx.send(Event::Log(report.statistics_text()));
                let _ = tx.send(Event::Log(serde_json::to_string_pretty(&report).unwrap()));
                if report.errors.is_empty() {
                    Ok(())
                } else {
                    Err("Dataset has errors; correct them before training".into())
                }
            })();
            let _ = tx.send(Event::Done(result));
        });
    }
    fn train(&mut self) {
        if !self.resume && !self.from_scratch && self.base_checkpoint.trim().is_empty() {
            self.status = "Select a base checkpoint to fine-tune".into();
            return;
        }
        // Preparation runs in the child process so WAV scanning does not block the UI.
        let mut command = Command::new(&self.python);
        command
            .arg(self.root.join("training/launch.py"))
            .arg("--dataset")
            .arg(&self.dataset)
            .arg("--output")
            .arg(&self.output)
            .arg("--steps")
            .arg(self.steps.to_string())
            .arg("--batch-size")
            .arg(self.batch.to_string())
            .arg("--device")
            .arg(if self.cuda { "cuda" } else { "cpu" });
        if self.resume {
            command.arg("--resume");
        } else if self.from_scratch {
            command.arg("--from-scratch");
        } else {
            command.arg("--init-checkpoint").arg(&self.base_checkpoint);
        }
        self.process(command, "Preparing and training…");
    }
    fn play(&mut self) {
        if let Some(audio) = &self.audio {
            self.playback.play(audio.as_ref().clone());
            self.status =
                "Starting playback… audio-device initialization runs in the background".into();
        }
    }
    fn apply_effects(&mut self) {
        let Some(source) = self.effects.source() else {
            return;
        };
        let revision = self.effects.revision;
        let settings = self.effects.settings;
        self.begin("Processing voice effects…");
        self.playback.stop();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let result = local_tts_core::effects::process(&source, settings).map(|(audio, stats)| {
                let _ = tx.send(Event::Effects(revision, settings, Arc::new(audio), stats));
                let _ = tx.send(Event::Log(format!("Effects applied: bass {:.1} dB, brightness {:.1} dB, compression {:.2}, pitch {:.1} st, formants {:.1} st; {} original samples retained",
                    settings.bass_db, settings.brightness_db, settings.compression,
                    settings.pitch_semitones, settings.formant_semitones, source.samples.len())));
            });
            let _ = tx.send(Event::Done(result));
        });
    }
    fn preview_audio(&mut self, audio: Audio) {
        self.effects.clear();
        self.audio = Some(Arc::new(audio));
        self.play();
    }
}
impl Drop for Studio {
    fn drop(&mut self) {
        self.stop();
    }
}
fn path_field(ui: &mut egui::Ui, label: &str, value: &mut String, extension: Option<&str>) {
    ui.label(label);
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(value).desired_width(650.0));
        if ui.button("Browse…").clicked() {
            let path = if let Some(extension) = extension {
                rfd::FileDialog::new()
                    .add_filter(extension, &[extension])
                    .pick_file()
            } else {
                rfd::FileDialog::new().pick_folder()
            };
            if let Some(path) = path {
                *value = path.display().to_string();
            }
        }
    });
}
impl eframe::App for Studio {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) && self.editor.dirty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_dialog = true;
        }
        if self.close_dialog {
            egui::Window::new("Unsaved dataset edits")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Save your dataset edits before closing, or discard them explicitly.");
                    if ui.button("Return to dataset").clicked() {
                        self.close_dialog = false;
                        self.tab = Tab::Dataset;
                    }
                    if ui.button("Discard edits and close").clicked() {
                        self.close_dialog = false;
                        self.editor.discard_dirty_flag();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
        }
        while let Ok(event) = self.rx.try_recv() {
            match event {
                Event::Log(line) => {
                    self.logs.push(line);
                    if self.logs.len() > 1000 {
                        self.logs.remove(0);
                    }
                }
                Event::Audio(audio) => {
                    let audio = Arc::new(audio);
                    self.effects.set_original(audio.clone());
                    self.audio = Some(audio);
                }
                Event::Effects(revision, settings, audio, stats) => {
                    if self
                        .effects
                        .accept(revision, settings, audio.clone(), stats)
                    {
                        self.audio = Some(audio);
                    }
                }
                Event::Playback(id, result) => {
                    if self.playback.accept(id) {
                        self.status = match result {
                            Ok(()) => "Playback started".into(),
                            Err(e) => format!("Playback failed: {e}"),
                        };
                    }
                }
                Event::Preflight(report) => self.evaluation.preflight = Some(*report),
                Event::Comparison(report) => {
                    self.evaluation.exported = None;
                    self.evaluation.result = Some(*report);
                }
                Event::EvaluatedExport(exported) => {
                    if self
                        .evaluation
                        .result
                        .as_ref()
                        .is_some_and(|r| r.target.sha256 == exported.checkpoint_sha256)
                    {
                        self.evaluation.exported = Some(exported);
                    }
                }
                Event::Done(result) => {
                    self.synthesis_cancel = None;
                    self.busy = false;
                    self.status = match result {
                        Ok(()) => "Completed".into(),
                        Err(e) if e == "synthesis cancelled" => "Synthesis cancelled".into(),
                        Err(e) => format!("Failed: {e}"),
                    };
                }
            }
        }
        if self.busy
            || self.playback.pending()
            || self.reference_editor.pending()
            || self.profile_editor.pending()
        {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Local TTS Studio");
            ui.add_enabled_ui(!self.editor.active(), |ui| {
                ui.horizontal(|ui| { ui.selectable_value(&mut self.tab, Tab::Test, "Test speech"); ui.selectable_value(&mut self.tab, Tab::Dataset, "Build dataset"); ui.selectable_value(&mut self.tab, Tab::Train, "Train a voice"); });
            });
            ui.separator();
            ui.add_enabled_ui(!self.busy, |ui| match self.tab {
                Tab::Dataset => {
                    egui::ScrollArea::vertical().id_salt("dataset-editor").max_height(ui.available_height()).show(ui, |ui| {
                        match self.editor.ui(ui, ctx) {
                            Some(collection_ui::Action::Play(audio)) => { self.preview_audio(audio); }
                            Some(collection_ui::Action::Train(path)) => { self.dataset = path.display().to_string(); self.tab = Tab::Train; }
                            None => {}
                        }
                    });
                }
                Tab::Test => {
                    egui::ScrollArea::vertical().id_salt("test-controls").max_height((ui.available_height()-100.0).max(180.0)).show(ui, |ui| {
                    let previous = self.test_voice;
                    egui::ComboBox::from_label("Voice model").selected_text(self.test_voice.label()).show_ui(ui, |ui| {
                        for choice in [TestVoice::MathStudent, TestVoice::Math, TestVoice::Inflect, TestVoice::Kitten, TestVoice::English, TestVoice::Russian, TestVoice::Ukrainian, TestVoice::European, TestVoice::Design, TestVoice::Own] { ui.selectable_value(&mut self.test_voice, choice, choice.label()); }
                    });
                    if previous != self.test_voice {
                        self.voice = match self.test_voice { TestVoice::MathStudent => self.root.join("checkpoints/math-pretrained-int8-en-v1"), TestVoice::Math => self.root.join("voices/math-en-v1"), TestVoice::Inflect => self.root.join("voices/inflect-nano-en"), TestVoice::Kitten => self.root.join("voices/kitten-nano-en"), TestVoice::Design => self.root.join("voices/qwen-voice-design"), TestVoice::English => self.root.join("voices").join(self.kokoro_variant.folder()), TestVoice::Ukrainian => self.root.join("voices/piper-uk"), TestVoice::Russian => self.root.join("voices/vosk-ru"), TestVoice::European => self.root.join("voices/chatterbox-multilingual"), TestVoice::Own => Path::new(&self.output).join("voice") }.display().to_string();
                        self.reference_voice.clear();
                        if self.test_voice == TestVoice::Design { self.european_language = Language::En; }
                        self.text = match self.test_voice { TestVoice::MathStudent => "12 + 225 = 237", TestVoice::Math => "12 + 225 = 237", TestVoice::Inflect | TestVoice::Kitten => "Hello world. This is a compact English voice.", TestVoice::Design => "Hello, this is a designed voice.", TestVoice::English => "Hello world. This is a local speech test.", TestVoice::Ukrainian => "Приві́т, сві́те! Це украї́нська мо́ва.", TestVoice::Russian => "Привет, мир! Это русский голос.", TestVoice::European => "Bonjour, le soleil entre par la fenêtre.", TestVoice::Own => "" }.into();
                        if self.test_voice==TestVoice::European {
                            self.text=local_tts_core::prompts::PromptList::starter(self.european_language).lines[0].clone();
                        }
                        self.speaker = match self.test_voice { TestVoice::Russian => "F03", TestVoice::MathStudent | TestVoice::Math | TestVoice::Inflect | TestVoice::Kitten => "default", _ => "lada" }.into();
                        self.audio = None; self.effects.clear(); self.playback.stop(); self.status = "Ready".into(); self.logs.clear();
                    }
                    path_field(ui, "Voice folder", &mut self.voice, None);
                    match self.test_voice {
                        TestVoice::Design => {
                            egui::ComboBox::from_label("Design language").selected_text(self.european_language.label()).show_ui(ui, |ui| {
                                for language in Language::ALL.into_iter().filter(|l| local_tts_core::qwen_design::QwenDesign::supports(*l)) {
                                    ui.selectable_value(&mut self.european_language, language, language.label());
                                }
                            });
                            ui.label("Describe the voice (up to 500 characters)");
                            ui.add(egui::TextEdit::multiline(&mut self.voice_description).desired_rows(3).desired_width(f32::INFINITY));
                            ui.label("Qwen3 VoiceDesign • native CPU • 24 kHz • INT4 • approximately 5.85 GB download. Short utterances only; text up to 300 characters. No reference audio required.");
                            ui.label("Describe age, gender, pitch, accent and delivery. Identity may vary. To reuse a result: Save WAV, switch to Multilingual / voice cloning, import that WAV as the reference, then save a voice profile.");
                            if ui.button("Download voice-design model").clicked() {
                                let mut command = Command::new(&self.python);
                                command.arg(self.root.join("scripts/download_qwen_design.py")).arg("--output").arg(&self.voice);
                                self.process(command, "Downloading pinned voice-design assets…");
                            }
                        }
                        TestVoice::English => {
                            let previous_variant = self.kokoro_variant;
                            egui::ComboBox::from_label("Kokoro variant").selected_text(self.kokoro_variant.label()).show_ui(ui, |ui| {
                                for variant in KokoroVariant::ALL { ui.selectable_value(&mut self.kokoro_variant, variant, variant.label()); }
                            });
                            if previous_variant != self.kokoro_variant {
                                self.voice = self.root.join("voices").join(self.kokoro_variant.folder()).display().to_string();
                                self.audio = None; self.effects.clear(); self.playback.stop(); self.status = "Ready".into(); self.logs.clear();
                            }
                            ui.label("Kokoro • af_heart • CPU • 24 kHz");
                            if ui.button("Download English model").clicked() {
                                let mut command = Command::new(&self.python); command.arg(self.root.join("scripts/download_english.py")).arg(self.kokoro_variant.id());
                                self.process(command, "Downloading English assets…");
                            }
                            ui.checkbox(&mut self.phonemes, "Input is Kokoro phonemes");
                            ui.label("Text supports ISO dates, times, units, prices, percentages and ordinals.");
                        }
                        TestVoice::Ukrainian => {
                            ui.label("Ukrainian • CPU • 22.05 kHz");
                            egui::ComboBox::from_label("Speaker").selected_text(&self.speaker).show_ui(ui, |ui| { for speaker in ["lada", "mykyta", "tetiana"] { ui.selectable_value(&mut self.speaker, speaker.into(), speaker); } });
                            if ui.button("Download Ukrainian model").clicked() {
                                let mut command = Command::new(&self.python); command.arg(self.root.join("scripts/download_cyrillic.py")).arg("uk");
                                self.process(command, "Downloading Ukrainian assets…");
                            }
                            ui.label("Stress is optional: Прив+іт = Приві́т. The + marker is not spoken.");
                            ui.label("Limited phrase stress cues; no general stress predictor. Dates, units and prices supported.");
                        }
                        TestVoice::Russian => {
                            ui.label("Russian • CPU • 22.05 kHz");
                            egui::ComboBox::from_label("Speaker").selected_text(&self.speaker).show_ui(ui, |ui| { for speaker in local_tts_core::vosk::SPEAKERS { ui.selectable_value(&mut self.speaker, speaker.into(), speaker); } });
                            if ui.button("Download Russian model").clicked() {
                                let mut command = Command::new(&self.python); command.arg(self.root.join("scripts/download_cyrillic.py")).arg("ru");
                                self.process(command, "Downloading Russian assets…");
                            }
                            ui.label("Dictionary and phrase cues; explicit/saved stress takes priority. Dates, units and prices supported.");
                        }
                        TestVoice::European => {
                            let previous_language=self.european_language;
                            egui::ComboBox::from_label("Speech language").selected_text(self.european_language.label()).show_ui(ui,|ui| {
                                for language in Language::ALL.into_iter().filter(|l| *l != Language::Uk) {
                                    ui.selectable_value(&mut self.european_language,language,language.label());
                                }
                            });
                            if previous_language!=self.european_language {
                                self.text=local_tts_core::prompts::PromptList::starter(self.european_language).lines[0].clone();
                                self.audio=None;self.effects.clear();self.playback.stop();self.logs.clear();
                            }
                            ui.label("Chatterbox • CPU • 24 kHz • approximately 1.56 GB download");
                            let old_reference = self.reference_voice.clone();
                            if let Some(selection) = self.profile_editor.ui(ui, &self.reference_voice, self.european_language, self.exaggeration) {
                                self.reference_voice = selection.reference.display().to_string();
                                self.exaggeration = selection.profile.exaggeration;
                                self.european_language = selection.profile.language;
                                self.text = local_tts_core::prompts::PromptList::starter(self.european_language).lines[0].clone();
                                self.audio = None; self.effects.clear(); self.playback.stop();
                                *self.engine.lock().unwrap() = None;
                            }

                            if let Some(action) = self.reference_editor.ui(ui) {
                                match action {
                                    reference_ui::Action::Play(audio) => self.playback.play(audio),
                                    reference_ui::Action::Select(path) => self.reference_voice = path.display().to_string(),
                                }
                            }
                            path_field(ui, "Optional speaker reference (mono 24 kHz WAV, up to 30 s)", &mut self.reference_voice, Some("wav"));
                            if ui.button("Use default voice").clicked() { self.reference_voice.clear(); }
                            if old_reference != self.reference_voice {
                                self.audio = None; self.effects.clear(); self.playback.stop();
                                *self.engine.lock().unwrap() = None;
                            }
                            ui.label("Use a clean speech recording, ideally 5–15 seconds. Mono 24 kHz WAV is required. English and Russian cloning are available; Ukrainian is unsupported by this model.");
                            ui.label("The recording is encoded when the model loads. Clear it to use the packaged default voice.");
                            ui.add(egui::Slider::new(&mut self.exaggeration, 0.0..=1.0).text("Expressiveness"));
                            ui.label("Default 0.5. Adjusts delivery intensity, not a named mood or speaker age. Generate again to hear changes; higher values may speed up speech.");
                            ui.label("Write numbers and abbreviations as spoken words. Native-speaker quality evaluation is pending.");
                            if ui.button("Download multilingual model").clicked() {
                                let mut command=Command::new(&self.python);
                                command.arg(self.root.join("scripts/download_chatterbox.py")).arg("--output").arg(&self.voice);
                                self.process(command,"Downloading multilingual model…");
                            }
                        }
                        TestVoice::MathStudent => {
                            let bytes: u64 = ["model.onnx", "encoder.onnx", "latent.onnx", "decoder.onnx"].iter().filter_map(|name| std::fs::metadata(PathBuf::from(&self.voice).join(name)).ok()).map(|m| m.len()).sum();
                            let size = if bytes > 0 {format!("{:.2} MB graphs", bytes as f64 / 1_000_000.0)} else {"model not found".into()};
                            ui.label(format!("Experimental neural arithmetic voice • 22.05 kHz • {size}"));
                            if ui.button("Russian continuous student v6 (experimental, 9.51 MB)").clicked() {
                                self.voice=self.root.join("checkpoints/math-sequence-ru-native-v6").display().to_string();
                            }
                            if ui.button("Russian timing-focused student v7 (experimental)").clicked() {
                                self.voice=self.root.join("checkpoints/math-sequence-ru-native-v7").display().to_string();
                            }
                            if ui.button("Ukrainian continuous student v10 (experimental)").clicked() {
                                self.voice=self.root.join("checkpoints/math-sequence-uk-native-v10").display().to_string();
                            }
                            if ui.button("Ukrainian continuous student v11 (experimental)").clicked() {
                                self.voice=self.root.join("checkpoints/math-sequence-uk-native-v11").display().to_string();
                            }
                            if PathBuf::from(&self.voice).join("sequence.json").exists() { ui.label("Continuous speech student: numeric accuracy and listening approval pending."); }
                            if PathBuf::from(&self.voice).join("arithmetic-neural.json").exists() {
                                ui.label("Numbers-only candidate: word joins and naturalness await listening.");
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Compact numbers-only (9.96 MB package)").clicked() {
                                    self.voice=self.root.join("checkpoints/math-latent-hundreds-en-v4").display().to_string();
                                }
                                if ui.button("Quality baseline (29.6 MB model)").clicked() {
                                    self.voice=self.root.join("checkpoints/math-pretrained-int8-en-v1").display().to_string();
                                }
                            });
                            ui.horizontal(|ui| {
                                for (label, folder) in [("Russian numeric candidate", "math-latent-ru-q4-v1"), ("Ukrainian numeric candidate", "math-latent-uk-q8-v1")] {
                                    if ui.button(label).clicked() { self.voice=self.root.join("checkpoints").join(folder).display().to_string(); }
                                }
                            });
                            ui.horizontal(|ui| {
                                for (label, folder) in [("Russian contextual candidate", "math-latent-ru-context-q4-v2"), ("Ukrainian contextual candidate", "math-latent-uk-context-q8-v2")] {
                                    if ui.button(label).clicked() { self.voice=self.root.join("checkpoints").join(folder).display().to_string(); }
                                }
                            });
                            if self.voice.contains("math-latent-ru-context-") || self.voice.contains("math-latent-uk-context-") { ui.label("Diagnostic contextual candidate: recognition regressions detected; compare with the source voice before use."); }
                            ui.horizontal(|ui| {
                                for (label, folder) in [("Russian separate-word diagnostic", "math-latent-ru-per-entry-q4-v3"), ("Ukrainian separate-word diagnostic", "math-latent-uk-per-entry-q8-v3")] {
                                    if ui.button(label).clicked() { self.voice=self.root.join("checkpoints").join(folder).display().to_string(); }
                                }
                            });
                            if self.voice.contains("math-latent-ru-per-entry-") || self.voice.contains("math-latent-uk-per-entry-") { ui.label("Diagnostic separate-word decoding: numeric accuracy and naturalness are unverified."); }
                            ui.horizontal(|ui| {
                                for (label, folder) in [("Russian double-zero (rejected quality)", "math-latent-ru-double-zero-q4-v4"), ("Ukrainian double-zero diagnostic", "math-latent-uk-double-zero-q8-v4")] {
                                    if ui.button(label).clicked() { self.voice=self.root.join("checkpoints").join(folder).display().to_string(); }
                                }
                            });
                            if self.voice.contains("math-latent-ru-double-zero-") { ui.label("Rejected by listening: robotic/unclear. Retained only for regression comparison."); }
                            if self.voice.contains("double-zero-") { ui.label("Diagnostic double-zero phrases: broader decimal accuracy remains unresolved; Ukrainian recognition regressed."); }
                            ui.label("Enter numbers and + − × ÷ = or parentheses. Decimals use a dot; arithmetic is read as written.");
                            if ui.button("Use arithmetic example").clicked() {self.text="12 + 225 = 237".into();}
                        }
                        TestVoice::Math => {
                            ui.label("Small English arithmetic voice • 16 kHz • no neural model required");
                            ui.label("Enter numbers and + − × ÷ = or parentheses. Decimals use a dot. Expressions are read as written; arithmetic is not checked.");
                            ui.label("Fragment baseline: joins can sound less natural than neural speech. Voice bank is about 0.79 MB including notices.");
                            if ui.button("Use arithmetic example").clicked() { self.text="12 + 225 = 237".into(); }
                        }
                        TestVoice::Inflect | TestVoice::Kitten => {
                            ui.label("Experimental compact English • native CPU • 24 kHz • BSD dictionary frontend; not validated against Apple voices.");
                            ui.label("Text supports ISO dates, times, units, prices, percentages and ordinals. Open Pronunciation preview to inspect the spoken words.");
                            if self.test_voice==TestVoice::Kitten { egui::ComboBox::from_label("Kitten voice").selected_text(&self.speaker).show_ui(ui,|ui| {for name in ["default","Bella","Jasper","Luna","Bruno","Rosie","Hugo","Kiki","Leo"] {ui.selectable_value(&mut self.speaker,name.into(),name);}}); }
                            if ui.button("Download compact English model").clicked() {let mut command=Command::new(&self.python);command.arg(self.root.join("scripts/download_compact_english.py")).arg(if self.test_voice==TestVoice::Inflect {"inflect"} else {"kitten"}).arg("--output").arg(&self.voice);self.process(command,"Downloading pinned compact model…");}
                        }
                        TestVoice::Own => { ui.label("Own VITS model • language and speaker from voice.json"); ui.label("Text (use the spelling and stress conventions of the training voice)"); }
                    }
                    if let Some(language)=self.test_voice.language() {
                        self.pronunciation_editor.sync(Path::new(&self.voice),language);
                        self.pronunciation_editor.ui(ui);
                    }
                    ui.add(egui::TextEdit::multiline(&mut self.text).desired_rows(5).desired_width(f32::INFINITY));
                    if matches!(self.test_voice, TestVoice::Math | TestVoice::MathStudent) {
                        egui::CollapsingHeader::new("Arithmetic pronunciation preview").show(ui, |ui| {
                            let language = if self.test_voice == TestVoice::MathStudent {
                                std::fs::read(PathBuf::from(&self.voice).join("sequence.json")).or_else(|_| std::fs::read(PathBuf::from(&self.voice).join("arithmetic-neural.json"))).ok()
                                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                                    .and_then(|v| match v["language"].as_str() {Some("ru")=>Some(Language::Ru),Some("uk")=>Some(Language::Uk),_=>None}).unwrap_or(Language::En)
                            } else {Language::En};
                            match local_tts_core::math_speech::words_for_language(&self.text, language) {
                                Ok(words) => {ui.label(words.join(" "));},
                                Err(error) => {ui.colored_label(egui::Color32::LIGHT_RED,error);}
                            }
                        });
                    }
                    if matches!(self.test_voice, TestVoice::Russian | TestVoice::Ukrainian) || (matches!(self.test_voice, TestVoice::English) && !self.phonemes) || matches!(self.test_voice, TestVoice::Inflect | TestVoice::Kitten) {
                        egui::CollapsingHeader::new("Pronunciation preview").show(ui, |ui| {
                            let prepared = if matches!(self.test_voice, TestVoice::English | TestVoice::Inflect | TestVoice::Kitten) {
                                local_tts_core::text::prepare(&self.text, Language::En)
                            } else { self.pronunciation_editor.prepare(&self.text) };
                            match prepared {
                                Ok(text) => { egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| { ui.label(text); }); }
                                Err(error) => { ui.colored_label(egui::Color32::LIGHT_RED, error); }
                            }
                            if matches!(self.test_voice, TestVoice::English | TestVoice::Inflect | TestVoice::Kitten) {
                                ui.label("English example: On 2026-10-03 at 12:30, Dr. Smith paid $5.20 for 2.5 kg. It was 5% off on the 21st day.");
                                if ui.button("Use English numeric example").clicked() { self.text = "On 2026-10-03 at 12:30, Dr. Smith paid $5.20 for 2.5 kg. It was 5% off on the 21st day.".into(); }
                                ui.label("English dates use YYYY-MM-DD and decimals use a dot. Context choices appear in generated phonemes. Ambiguous notation needs a spoken spelling.");
                            } else {
                                ui.label("Examples: 2026-10-03, 03.10.2026, 12:30, 2,5 кг, 5%. Ambiguous dates and unsupported notation need a spoken spelling.");
                                ui.label("Accent marks show stress. Quantity agreement is nominative; surrounding case is not predicted. Phrase cues cover a few homographs, not arbitrary words.");
                            }
                        });
                    }
                    if !matches!(self.test_voice, TestVoice::European | TestVoice::Design | TestVoice::Math) {ui.add(egui::Slider::new(&mut self.scale, 0.5..=2.0).text("Duration multiplier"));}
                    if let Some(action) = self.effects.ui(ui) {
                        match action {
                            effects_ui::Action::Apply => self.apply_effects(),
                            effects_ui::Action::Select(audio) => { self.playback.stop(); self.audio = Some(audio); self.status = "Audio selection changed".into(); }
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.add_enabled(!self.busy, egui::Button::new("Generate")).clicked() { self.generate(); }
                        if ui.add_enabled(self.audio.is_some(), egui::Button::new("Play")).clicked() { self.play(); }
                        if ui.add_enabled(self.audio.is_some(), egui::Button::new("Save WAV…")).clicked()
                            && let Some(path) = rfd::FileDialog::new().add_filter("Wave audio", &["wav"]).set_file_name("speech.wav").save_file()
                        { self.status = match self.audio.as_ref().unwrap().write_wav(&path) { Ok(()) => format!("Saved {}", path.display()), Err(e) => e.to_string() }; }
                    });
                    });
                }
                Tab::Train => {
                    egui::ScrollArea::vertical().id_salt("training-controls").max_height((ui.available_height()-120.0).max(180.0)).show(ui,|ui| {
                    ui.label("Single-speaker VITS • fine-tune your voice • 10 languages • compatible same-language checkpoint required");
                    ui.add_enabled_ui(!self.resume, |ui| {
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.from_scratch, false, "Fine-tune (recommended)");
                            ui.radio_value(&mut self.from_scratch, true, "Train from scratch");
                        });
                        if !self.from_scratch {
                            path_field(ui, "Base checkpoint (.pt)", &mut self.base_checkpoint, Some("pt"));
                            ui.label("Use a project checkpoint in the same language, or import one of the bases below. ONNX files are inference-only.");
                            ui.label("Public fine-tuning bases are available for all ten languages below. Chatterbox is currently inference-only.");
                            ui.horizontal(|ui| {
                                let base = self.root.join("voices/vits-ljs-base/base.pt");
                                if ui.add_enabled(!base.exists(), egui::Button::new("Download / import English base")).clicked() {
                                    let mut command = Command::new(&self.python);
                                    command.arg(self.root.join("scripts/prepare_english_base.py"));
                                    self.process(command, "Acquiring and importing English training base…");
                                }
                                if ui.add_enabled(base.exists(), egui::Button::new("Use English base")).clicked() {
                                    self.base_checkpoint = base.display().to_string();
                                    self.from_scratch = false;
                                    self.status = "English base selected; use 22050 Hz English recordings".into();
                                }
                            });
                            ui.horizontal(|ui| {
                                let base=self.root.join("voices/vits-piper-uk-base/base.pt");
                                if ui.add_enabled(!base.exists(),egui::Button::new("Download / import Ukrainian base")).clicked() {
                                    let mut command=Command::new(&self.python);command.arg(self.root.join("scripts/prepare_ukrainian_base.py"));
                                    self.process(command,"Acquiring and importing Ukrainian training base…");
                                }
                                if ui.add_enabled(base.exists(),egui::Button::new("Use Ukrainian base")).clicked() {
                                    self.base_checkpoint=base.display().to_string();self.from_scratch=false;
                                    self.status="Ukrainian base selected; use 22050 Hz Ukrainian recordings".into();
                                }
                            });
                            ui.horizontal(|ui| {
                                let base=self.root.join("voices/vits-vosk-ru-base/base.pt");
                                if ui.add_enabled(!base.exists(),egui::Button::new("Download / import Russian base")).clicked() {
                                    let mut command=Command::new(&self.python);command.arg(self.root.join("scripts/prepare_russian_base.py"));
                                    self.process(command,"Acquiring and importing Russian training base…");
                                }
                                if ui.add_enabled(base.exists(),egui::Button::new("Use Russian base")).clicked() {
                                    self.base_checkpoint=base.display().to_string();self.from_scratch=false;
                                    self.status="Russian base selected; use 22050 Hz Russian recordings".into();
                                }
                            });
                            for (code, label) in [("de", "German"), ("es", "Spanish"), ("fr", "French"), ("it", "Italian"), ("pt", "Portuguese"), ("pl", "Polish"), ("nl", "Dutch")] {
                                let css10 = matches!(code, "de" | "es");
                                let family = if css10 { "css10" } else { "neon" };
                                ui.horizontal(|ui| {
                                    let base = self.root.join(format!("voices/vits-{family}-{code}-base/base.pt"));
                                    if ui.add_enabled(!base.exists(), egui::Button::new(format!("Download / import {label} base"))).clicked() {
                                        let mut command = Command::new(&self.python);
                                        command.arg(self.root.join(if css10 { "scripts/prepare_css10_base.py" } else { "scripts/prepare_european_base.py" })).arg(code);
                                        self.process(command, &format!("Acquiring and importing {label} training base…"));
                                    }
                                    if ui.add_enabled(base.exists(), egui::Button::new(format!("Use {label} base"))).clicked() {
                                        self.base_checkpoint = base.display().to_string();
                                        self.from_scratch = false;
                                        self.status = format!("{label} base selected; use 22050 Hz {label} recordings");
                                    }
                                });
                            }
                            ui.label("European bases: Neon grapheme VITS (CSS10, Common Voice and MAI source families). Publisher declares BSD-3-Clause; full weight notices and recording/consent provenance remain unresolved. Import and synthetic training checks do not establish voice quality or commercial clearance.");
                            ui.label("Russian base: original Vosk training weights, F01 conditioning by default. Install Russian inference assets for its dictionary first. Publisher declares Apache-2.0; dictionary, recording and derived voice rights remain unverified.");
                            ui.label("Ukrainian base: original Piper training weights, lada conditioning by default. Repository declares MIT; model card/data attribution and complete rights lineage remain unverified.");
                            ui.label("LJSpeech base: mirror declares Apache-2.0; dataset declared public domain. Earlier weight mirror lacks a separate license record. Native CMU IPA differs from the original frontend.");
                            if ui.button("Check base and target compatibility").clicked() {
                                self.evaluation.preflight=None;
                                let mut command=Command::new(&self.python);
                                command.arg(self.root.join("training/preflight.py")).arg("--checkpoint").arg(&self.base_checkpoint).arg("--dataset").arg(&self.dataset);
                                self.process(command,"Checking base weights and dataset compatibility…");
                            }
                            self.evaluation.preflight_ui(ui,&self.base_checkpoint,&self.dataset,&self.root);
                        }
                    });
                    path_field(ui, "Dataset manifest", &mut self.dataset, Some("json"));
                    path_field(ui, "Checkpoint folder", &mut self.output, None);
                    ui.label("Python executable (setup requires Python 3.10+)"); ui.text_edit_singleline(&mut self.python);
                    ui.horizontal(|ui| {
                        if ui.button("Set up training environment").clicked() {
                            let mut command = Command::new(&self.python); command.arg(self.root.join("training/setup.py"));
                            self.process(command, "Installing training dependencies…");
                        }
                        if ui.button("Use project environment").clicked() { self.python = self.root.join(if cfg!(windows) { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }).display().to_string(); }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Total steps"); ui.add(egui::DragValue::new(&mut self.steps).range(1..=1000000));
                        ui.label("Batch size"); ui.add(egui::DragValue::new(&mut self.batch).range(1..=64));
                        ui.checkbox(&mut self.cuda, "CUDA"); ui.checkbox(&mut self.resume, "Resume latest checkpoint");
                    });
                    ui.label("CPU works for engineering checks; a usable voice needs real recordings and substantial training. Stop saves at a step boundary on macOS/Linux.");
                    ui.horizontal(|ui| {
                        if ui.button("Check dataset").clicked() { self.validate(); }
                        if ui.button("Start training").clicked() { self.train(); }
                        if ui.button("Listen to training preview").clicked() {
                            let path = Path::new(&self.output).join("preview.wav");
                            match read_audio(&path) { Ok(audio) => { self.preview_audio(audio); }, Err(e) => self.status = e }
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Export and verify ONNX").clicked() {
                            let mut command = Command::new(&self.python);
                            command.arg(self.root.join("training/export_and_verify.py")).arg("--checkpoint").arg(Path::new(&self.output).join("latest.pt")).arg("--output").arg(Path::new(&self.output).join("voice"));
                            self.process(command, "Exporting and checking native inference…");
                        }
                        if ui.button("Evaluate before / after…").clicked() { self.evaluation.open=true; }
                        if ui.button("Test exported voice").clicked() {
                            self.test_voice = TestVoice::Own; self.voice = Path::new(&self.output).join("voice").display().to_string(); self.tab = Tab::Test; self.audio = None; self.effects.clear();
                            *self.engine.lock().unwrap() = None;
                        }
                    });
                    });
                }
            });
            if let Some(action)=self.evaluation.show(ctx,&self.output,self.busy) {
                match action {
                    training_ui::Action::Compare {checkpoint,split,limit} => {
                        let mut command=Command::new(&self.python);
                        command.arg(self.root.join("training/compare.py")).arg("--base").arg(&self.base_checkpoint)
                            .arg("--checkpoint").arg(checkpoint).arg("--dataset").arg(&self.dataset)
                            .arg("--output").arg(Path::new(&self.output).join("evaluations"))
                            .arg("--split").arg(if split==dataset::Split::Test {"test"}else{"validation"}).arg("--limit").arg(limit.to_string());
                        self.process(command,"Exporting and generating held-out native comparisons…");
                    }
                    training_ui::Action::Export {checkpoint,sha256,output} => {
                        let mut command=Command::new(&self.python);
                        command.arg(self.root.join("training/export_and_verify.py")).arg("--checkpoint").arg(checkpoint)
                            .arg("--expected-sha256").arg(sha256).arg("--new-output").arg("--output").arg(output);
                        self.process(command,"Exporting evaluated checkpoint and checking native inference…");
                    }
                    training_ui::Action::Test(voice) => {
                        self.test_voice=TestVoice::Own;self.voice=voice.display().to_string();self.tab=Tab::Test;
                        if let Some(sample)=self.evaluation.result.as_ref().and_then(|r|r.report.samples.first()) {self.text=sample.text.clone();}
                        self.scale=1.0;
                        self.audio=None;self.effects.clear();self.playback.stop();self.evaluation.open=false;
                        *self.engine.lock().unwrap()=None;
                        self.status="Ready to test evaluated voice".into();
                    }
                    training_ui::Action::Keep {checkpoint,sha256,step} => {
                        self.begin("Saving evaluated checkpoint…");
                        let tx=self.tx.clone();
                        thread::spawn(move || {
                            let destination=checkpoint.parent().unwrap_or(Path::new(".")).join("saved-checkpoints").join(format!("step-{step}-{}.pt",&sha256[..12]));
                            let result=local_tts_core::training::snapshot_checkpoint(&checkpoint,&destination,&sha256);
                            if result.is_ok() {let _=tx.send(Event::Log(format!("Saved evaluated checkpoint: {}",destination.display())));}
                            let _=tx.send(Event::Done(result));
                        });
                    }
                    training_ui::Action::Play(root,file) => match local_tts_core::evaluation::read_audio(&root,&file) {
                        Ok(audio) => {self.preview_audio(audio);},
                        Err(e) => self.status=format!("Comparison playback failed: {e}"),
                    },
                }
            }
            if self.tab == Tab::Dataset { return; }
            ui.separator();
            ui.horizontal(|ui| {
                if let Some(cancel) = &self.synthesis_cancel && ui.button("Cancel synthesis").clicked() {
                    cancel.store(true, Ordering::Relaxed);
                    self.status = "Cancelling after the current model chunk…".into();
                }
                if ui.button("Stop playback").clicked() {self.playback.stop();self.status="Playback stopped".into();}
                ui.label(&self.status); if self.busy && self.child.lock().unwrap().is_some() && ui.button("Stop job").clicked() { self.stop(); } });
            egui::ScrollArea::vertical().stick_to_bottom(true).max_height(260.0).show(ui, |ui| {
                for line in &self.logs { ui.monospace(line); }
            });
        });
    }
}
fn read_audio(path: &Path) -> Result<Audio, String> {
    // Preview files are emitted as signed 16-bit mono WAV by the trainer.
    local_tts_core::read_wav(path).map_err(|e| e.to_string())
}
fn main() -> eframe::Result {
    let runtime = runtime::locate()
        .map_err(|e| eframe::Error::AppCreation(Box::new(std::io::Error::other(e))))?;
    if std::env::args().any(|arg| arg == "--check-runtime") {
        println!(
            "{}",
            serde_json::json!({"root": runtime.root, "cli": runtime.cli, "version": env!("CARGO_PKG_VERSION")})
        );
        return Ok(());
    }
    eframe::run_native(
        "Local TTS Studio",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([880.0, 760.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "noto".into(),
                egui::FontData::from_static(include_bytes!("../../../assets/fonts/NotoSans.ttf"))
                    .into(),
            );
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.get_mut(&family).unwrap().push("noto".into());
            }
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(Studio::new(runtime)))
        }),
    )
}

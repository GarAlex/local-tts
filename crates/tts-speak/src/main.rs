use local_tts_core::{Language, SynthesisRequest, Synthesizer};
use std::{
    collections::BTreeMap, io::Read, path::Path, process::ExitCode, sync::atomic::AtomicBool,
};
const BACKENDS: [&str; 10] = [
    "kokoro",
    "vosk",
    "piper",
    "vits",
    "chatterbox",
    "qwen-design",
    "inflect",
    "kitten",
    "math",
    "math-neural",
];
/// Backends whose English CMU frontend accepts `--lexicon`.
const LEXICON_BACKENDS: [&str; 4] = ["kokoro", "inflect", "kitten", "vits"];
const HELP: &str = "tts-speak — offline text to WAV, using local-tts-core
Usage: tts-speak --model FOLDER --output FILE.wav [options]
       tts-speak --list-voices [--backend NAME] [--model FOLDER]
  --backend kokoro|vosk|piper|vits|chatterbox|qwen-design|inflect|kitten|math|math-neural  (default kokoro)
  --language en|ru|uk|de|fr|es|it|pt|pl|nl              (default en)
  --text TEXT | --input UTF8_FILE                      (default: stdin)
  --speaker NAME                                      (defaults: af_heart / F01 / lada / default)
  --duration-scale NUMBER                             (default 1)
  --reference FILE.wav | --profile FOLDER              (Chatterbox only)
  --description TEXT                                  (required for qwen-design)
  --lexicon FILE                                      CMU-format extra words (English kokoro/inflect/kitten/vits)
  --json                                              print a JSON result on stdout
  --list-voices                                       print the backend's speakers/languages as JSON
                                                      (--model required for piper and vits)
  --force                                             replace an existing output
  --help | --version
Models must already be installed. No downloads, GUI, training, or Python runtime.";
fn read_text(reader: impl Read) -> Result<String, String> {
    let mut text = String::new();
    reader
        .take(1_048_577)
        .read_to_string(&mut text)
        .map_err(|e| e.to_string())?;
    if text.len() > 1_048_576 || text.trim().is_empty() || text.contains('\0') {
        return Err("input must be nonempty UTF-8 text, without NUL, at most 1 MiB".into());
    }
    Ok(text)
}
fn run(args: Vec<String>) -> Result<(), String> {
    if args == ["--help"] || args == ["-h"] {
        println!("{HELP}");
        return Ok(());
    }
    if args == ["--version"] {
        println!("tts-speak {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let mut opts = BTreeMap::new();
    let mut flags = std::collections::BTreeSet::new();
    let mut iter = args.into_iter();
    while let Some(key) = iter.next() {
        if ["--force", "--json", "--list-voices"].contains(&key.as_str()) {
            if !flags.insert(key.clone()) {
                return Err(format!("duplicate {key}"));
            }
            continue;
        }
        if ![
            "--model",
            "--output",
            "--backend",
            "--language",
            "--text",
            "--input",
            "--speaker",
            "--duration-scale",
            "--reference",
            "--profile",
            "--description",
            "--lexicon",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}; use --help"));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.starts_with("--") && key != "--text" && key != "--description" {
            return Err(format!("missing value for {key}"));
        }
        if opts.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option {key}"));
        }
    }
    let get = |key: &str| opts.get(key).map(String::as_str);
    let force = flags.contains("--force");
    let backend = get("--backend").unwrap_or("kokoro");
    if !BACKENDS.contains(&backend) {
        return Err("unknown backend".into());
    }
    if flags.contains("--list-voices") {
        if let Some(key) = opts
            .keys()
            .find(|k| !["--backend", "--model"].contains(&k.as_str()))
        {
            return Err(format!("--list-voices does not accept {key}"));
        }
        if force {
            return Err("--list-voices does not accept --force".into());
        }
        println!("{}", list_voices(backend, get("--model").map(Path::new))?);
        return Ok(());
    }
    let model = Path::new(get("--model").ok_or("--model is required")?);
    let output = Path::new(get("--output").ok_or("--output is required")?);
    if output.file_name().is_none() {
        return Err("output must name a WAV file".into());
    }
    if output.exists() && !force {
        return Err("output already exists; use --force to replace it".into());
    }
    let lexicon = get("--lexicon").map(Path::new);
    if lexicon.is_some() && !LEXICON_BACKENDS.contains(&backend) {
        return Err("lexicon requires kokoro, inflect, kitten or vits".into());
    }
    let language: Language = get("--language")
        .unwrap_or("en")
        .parse()
        .map_err(|e: String| e)?;
    let scale: f32 = get("--duration-scale")
        .unwrap_or("1")
        .parse()
        .map_err(|_| "invalid duration scale")?;
    if !scale.is_finite() || !(0.5..=2.0).contains(&scale) {
        return Err("duration scale must be 0.5–2".into());
    }
    if get("--reference").is_some() && get("--profile").is_some() {
        return Err("choose reference or profile".into());
    }
    if backend != "chatterbox" && (get("--reference").is_some() || get("--profile").is_some()) {
        return Err("reference/profile requires chatterbox".into());
    }
    if backend == "qwen-design" {
        if get("--description").is_none_or(|s| s.trim().is_empty()) {
            return Err("qwen-design requires --description".into());
        }
        if get("--speaker").is_some() || get("--duration-scale").is_some() {
            return Err("qwen-design does not accept speaker/duration scale".into());
        }
    } else if get("--description").is_some() {
        return Err("description requires qwen-design".into());
    }
    if backend == "chatterbox" && get("--duration-scale").is_some() {
        return Err("chatterbox does not accept duration scale".into());
    }
    let text = match (get("--text"), get("--input")) {
        (Some(_), Some(_)) => return Err("choose --text or --input".into()),
        (Some(t), None) => read_text(t.as_bytes())?,
        (None, Some(p)) => read_text(std::fs::File::open(p).map_err(|e| e.to_string())?)?,
        (None, None) => read_text(std::io::stdin().lock())?,
    };
    // Exported VITS voices name their own speakers; default to the first.
    let vits_speaker = match (backend, get("--speaker")) {
        ("vits", None) => vits_manifest(model)?.speakers.into_iter().next(),
        _ => None,
    };
    let speaker = get("--speaker")
        .or(vits_speaker.as_deref())
        .unwrap_or(default_speaker(backend));
    let request = SynthesisRequest {
        text: &text,
        language,
        speaker,
        duration_scale: scale,
    };
    let with_lexicon = |added: Result<usize, local_tts_core::kokoro::Error>| match lexicon {
        Some(_) => added.map(|_| ()).map_err(|e| e.to_string()),
        None => Ok(()),
    };
    // Each branch calls the reusable native library; no training subprocesses.
    let audio = match backend {
        "math-neural" => local_tts_core::math_speech::MathStudent::load(model)
            .map_err(|e| e.to_string())?
            .synthesize(&request)
            .map_err(|e| e.to_string())?,
        "math" => local_tts_core::math_speech::MathVoice::load(model)?
            .synthesize(&request)
            .map_err(|e| e.to_string())?,
        "inflect" | "kitten" => {
            let mut engine = local_tts_core::compact_english::CompactEnglish::load(
                model,
                if backend == "inflect" {
                    local_tts_core::compact_english::CompactKind::Inflect
                } else {
                    local_tts_core::compact_english::CompactKind::Kitten
                },
            )
            .map_err(|e| e.to_string())?;
            if let Some(path) = lexicon {
                with_lexicon(engine.add_lexicon(path))?;
            }
            engine.synthesize(&request).map_err(|e| e.to_string())?
        }
        "kokoro" => {
            let mut engine =
                local_tts_core::kokoro::Kokoro::load(model).map_err(|e| e.to_string())?;
            if let Some(path) = lexicon {
                with_lexicon(engine.add_lexicon(path))?;
            }
            engine.synthesize(&request).map_err(|e| e.to_string())?
        }
        "vosk" => local_tts_core::vosk::Vosk::load(model)
            .map_err(|e| e.to_string())?
            .synthesize(&request)
            .map_err(|e| e.to_string())?,
        "piper" => local_tts_core::piper::Piper::load(model)
            .map_err(|e| e.to_string())?
            .synthesize(&request)
            .map_err(|e| e.to_string())?,
        "vits" => {
            let mut engine = local_tts_core::vits::Vits::load(model).map_err(|e| e.to_string())?;
            if let Some(path) = lexicon {
                with_lexicon(engine.add_lexicon(path))?;
            }
            engine.synthesize(&request).map_err(|e| e.to_string())?
        }
        "chatterbox" => {
            let mut engine = if let Some(profile) = get("--profile") {
                local_tts_core::voice_profile::load_engine(model, Path::new(profile))?
            } else {
                local_tts_core::chatterbox::Chatterbox::load_with_reference(
                    model,
                    get("--reference").map(Path::new),
                )
                .map_err(|e| e.to_string())?
            };
            engine.synthesize(&request).map_err(|e| e.to_string())?
        }
        "qwen-design" => local_tts_core::qwen_design::QwenDesign::load(model)
            .map_err(|e| e.to_string())?
            .synthesize_design(
                &text,
                language,
                get("--description").unwrap(),
                &AtomicBool::new(false),
            )
            .map_err(|e| e.to_string())?,
        _ => unreachable!(),
    };
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stage = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    audio.write_wav(stage.path()).map_err(|e| e.to_string())?;
    if force {
        stage.persist(output).map_err(|e| e.to_string())?;
    } else {
        stage.persist_noclobber(output).map_err(|e| e.to_string())?;
    }
    let seconds = audio.samples.len() as f64 / audio.sample_rate as f64;
    if flags.contains("--json") {
        println!(
            "{}",
            serde_json::json!({
                "output": output.display().to_string(),
                "format": "wav",
                "seconds": (seconds * 1000.0).round() / 1000.0,
                "sampleRate": audio.sample_rate,
                "backend": backend,
                "language": language.code(),
                "speaker": speaker,
            })
        );
    }
    eprintln!(
        "Saved {} ({seconds:.2}s, {} Hz)",
        output.display(),
        audio.sample_rate
    );
    Ok(())
}
fn default_speaker(backend: &str) -> &'static str {
    match backend {
        "kokoro" => "af_heart",
        "vosk" => "F01",
        "piper" => "lada",
        _ => "default",
    }
}
fn vits_manifest(model: &Path) -> Result<local_tts_core::voice::VoiceManifest, String> {
    let path = model.join("voice.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}
/// Speakers and languages a backend accepts, as JSON. Only piper and vits
/// read their package; the others are fixed by the backend.
fn list_voices(backend: &str, model: Option<&Path>) -> Result<String, String> {
    const CHATTERBOX: [&str; 8] = ["en", "de", "fr", "es", "it", "pt", "pl", "nl"];
    let needs_model = || model.ok_or(format!("--list-voices for {backend} requires --model"));
    let (voices, languages): (Vec<String>, Vec<String>) = match backend {
        "piper" => {
            let engine =
                local_tts_core::piper::Piper::load(needs_model()?).map_err(|e| e.to_string())?;
            (
                engine.speakers().map(String::from).collect(),
                vec!["uk".into()],
            )
        }
        "vits" => {
            let manifest = vits_manifest(needs_model()?)?;
            (manifest.speakers, vec![manifest.language.code().into()])
        }
        "vosk" => (
            local_tts_core::vosk::SPEAKERS.map(String::from).to_vec(),
            vec!["ru".into()],
        ),
        "kitten" => (
            local_tts_core::compact_english::KITTEN_VOICES
                .map(String::from)
                .to_vec(),
            vec!["en".into()],
        ),
        "chatterbox" => (
            vec!["default".into()],
            CHATTERBOX.map(String::from).to_vec(),
        ),
        "qwen-design" => (
            Vec::new(),
            ["en", "ru", "de", "fr", "es", "it", "pt"]
                .map(String::from)
                .to_vec(),
        ),
        _ => (vec![default_speaker(backend).into()], vec!["en".into()]),
    };
    let default = match backend {
        "qwen-design" => None,
        "vits" => voices.first().map(String::as_str),
        _ => Some(default_speaker(backend)),
    };
    Ok(serde_json::json!({
        "backend": backend,
        "default": default,
        "voices": voices,
        "languages": languages,
        "lexicon": LEXICON_BACKENDS.contains(&backend),
    })
    .to_string())
}
fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tts-speak: {error}");
            ExitCode::FAILURE
        }
    }
}

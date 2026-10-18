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
       tts-speak --catalog [--voices-dir DIR]
       tts-speak --install ID | --remove ID  --voices-dir DIR [--mirror URL]
  --backend kokoro|vosk|piper|vits|chatterbox|qwen-design|inflect|kitten|math|math-neural  (default kokoro)
  --language en|ru|uk|de|fr|es|it|pt|pl|nl              (default en)
  --text TEXT | --input UTF8_FILE                      (default: stdin)
  --speaker NAME                                      (defaults: af_heart / F01 / lada / default)
  --duration-scale NUMBER                             (default 1)
  --reference FILE.wav | --profile FOLDER              (Chatterbox only)
  --description TEXT                                  (required for qwen-design)
  --lexicon FILE                                      CMU-format extra words (English kokoro/inflect/kitten/vits)
  --guess-unknown                                     guess words the dictionary lacks, and list the guesses
  --json                                              print a JSON result on stdout
  --list-voices                                       print the backend's speakers/languages as JSON
                                                      (--model required for piper and vits)
  --catalog                                           print installable voice packages as JSON
  --install ID / --remove ID                          download+verify / delete a package in --voices-dir
  --mirror URL                                        try URL/ID/FILE before the pinned upstream URL
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
        if [
            "--force",
            "--json",
            "--list-voices",
            "--catalog",
            "--guess-unknown",
        ]
        .contains(&key.as_str())
        {
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
            "--install",
            "--remove",
            "--voices-dir",
            "--mirror",
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
    let package_options = ["--install", "--remove", "--voices-dir", "--mirror"];
    if flags.contains("--catalog") || get("--install").is_some() || get("--remove").is_some() {
        if let Some(key) = opts.keys().find(|k| !package_options.contains(&k.as_str())) {
            return Err(format!("package commands do not accept {key}"));
        }
        if flags.len() > 1
            || (flags.contains("--catalog") as usize
                + opts.contains_key("--install") as usize
                + opts.contains_key("--remove") as usize)
                != 1
        {
            return Err("choose one of --catalog, --install or --remove".into());
        }
        return packages(
            get("--install"),
            get("--remove"),
            get("--voices-dir"),
            get("--mirror"),
        );
    }
    if let Some(key) = opts.keys().find(|k| package_options.contains(&k.as_str())) {
        return Err(format!("{key} requires --install or --remove"));
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
    let guess_unknown = flags.contains("--guess-unknown");
    if guess_unknown && !LEXICON_BACKENDS.contains(&backend) {
        return Err("--guess-unknown requires kokoro, inflect, kitten or vits".into());
    }
    let mut guesses: Vec<local_tts_core::english::Guess> = Vec::new();
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
            if guess_unknown {
                engine.set_fallback(true).map_err(|e| e.to_string())?;
            }
            let audio = engine.synthesize(&request).map_err(|e| e.to_string())?;
            guesses = engine.take_guesses();
            audio
        }
        "kokoro" => {
            let mut engine =
                local_tts_core::kokoro::Kokoro::load(model).map_err(|e| e.to_string())?;
            if let Some(path) = lexicon {
                with_lexicon(engine.add_lexicon(path))?;
            }
            if guess_unknown {
                engine.set_fallback(true).map_err(|e| e.to_string())?;
            }
            let audio = engine.synthesize(&request).map_err(|e| e.to_string())?;
            guesses = engine.take_guesses();
            audio
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
            if guess_unknown {
                engine.set_fallback(true).map_err(|e| e.to_string())?;
            }
            let audio = engine.synthesize(&request).map_err(|e| e.to_string())?;
            guesses = engine.take_guesses();
            audio
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
                "guessed": guesses.iter().map(|g| serde_json::json!({
                    "word": g.word, "arpabet": g.arpabet, "method": g.method,
                })).collect::<Vec<_>>(),
            })
        );
    }
    for guess in &guesses {
        // Lexicon-ready: paste the line into --lexicon to keep or correct it.
        eprintln!(
            "guessed ({}): {}  {}",
            guess.method,
            guess.word.to_uppercase(),
            guess.arpabet
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
/// `--catalog`, `--install` and `--remove`: the pinned voice packages.
fn packages(
    install: Option<&str>,
    remove: Option<&str>,
    voices_dir: Option<&str>,
    mirror: Option<&str>,
) -> Result<(), String> {
    use local_tts_core::catalog;
    let dir = voices_dir.map(Path::new);
    if let Some(id) = install {
        let dir = dir.ok_or("--install requires --voices-dir")?;
        let mut shown = 0;
        let path = catalog::install(
            id,
            dir,
            mirror,
            &mut |done, total| {
                let percent = (done * 100).checked_div(total).unwrap_or(0).min(100);
                if percent >= shown + 10 || done == total {
                    shown = percent;
                    eprintln!("{id}: {percent}% of {:.1} MB", total as f64 / 1e6);
                }
            },
            &AtomicBool::new(false),
        )?;
        println!(
            "{}",
            serde_json::json!({ "installed": id, "path": path.display().to_string() })
        );
        return Ok(());
    }
    if let Some(id) = remove {
        let removed = catalog::remove(id, dir.ok_or("--remove requires --voices-dir")?)?;
        println!("{}", serde_json::json!({ "removed": removed, "id": id }));
        return Ok(());
    }
    if mirror.is_some() {
        return Err("--mirror requires --install".into());
    }
    let installed = dir.map(catalog::installed).unwrap_or_default();
    let outdated = dir.map(catalog::outdated).unwrap_or_default();
    let listing: Vec<_> = catalog::catalog()
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "title": p.title,
                "backend": p.backend,
                "languages": p.languages,
                "license": p.license,
                "source": p.source,
                "downloadBytes": p.download_bytes(),
                "installed": installed.contains(&p.id),
                "complete": installed.contains(&p.id) && !outdated.contains(&p.id),
            })
        })
        .collect();
    println!("{}", serde_json::Value::Array(listing));
    Ok(())
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
        // With --model, the voices that package has; without, every pinned one.
        "kokoro" => (
            local_tts_core::kokoro::voices()
                .into_iter()
                .filter(|v| model.is_none_or(|m| m.join(format!("{}.bin", v.name)).is_file()))
                .map(|v| v.name)
                .collect(),
            vec!["en".into()],
        ),
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
    let details: Vec<_> = if backend == "kokoro" {
        local_tts_core::kokoro::voices()
            .into_iter()
            .filter(|v| voices.contains(&v.name))
            .map(|v| serde_json::json!({ "name": v.name, "accent": v.accent, "gender": v.gender, "grade": v.grade }))
            .collect()
    } else {
        Vec::new()
    };
    Ok(serde_json::json!({
        "backend": backend,
        "default": default,
        "voices": voices,
        "details": details,
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

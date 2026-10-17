use local_tts_core::{
    Language, SynthesisRequest, Synthesizer, dataset, kokoro::Kokoro, training,
    voice::VoiceManifest,
};
use std::{env, path::Path, process::ExitCode};

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.as_slice() == ["--version"] {
        println!("local-tts {} (ABI 1)", env!("CARGO_PKG_VERSION"));
        return Ok(true);
    }
    if args.len() == 6 && args[0] == "voice-design" {
        let model = local_tts_core::qwen_design::QwenDesign::load(Path::new(&args[1]))?;
        let audio = model.synthesize_design(
            &args[5],
            args[3].parse()?,
            &args[4],
            &std::sync::atomic::AtomicBool::new(false),
        )?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved designed voice: {} samples at {} Hz",
            audio.samples.len(),
            audio.sample_rate
        );
        return Ok(true);
    }
    if args.len() == 6 && args[0] == "profile-create" {
        local_tts_core::voice_profile::VoiceProfile::create(
            Path::new(&args[1]),
            Path::new(&args[2]),
            &args[3],
            args[4].parse()?,
            args[5].parse()?,
        )
        .map_err(std::io::Error::other)?;
        println!("Saved voice profile: {}", args[1]);
        return Ok(true);
    }
    if args.len() == 5 && args[0] == "profile-synthesize" {
        let folder = Path::new(&args[2]);
        let profile = local_tts_core::voice_profile::VoiceProfile::load(folder)
            .map_err(std::io::Error::other)?;
        let mut engine = local_tts_core::voice_profile::load_engine(Path::new(&args[1]), folder)
            .map_err(std::io::Error::other)?;
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[4],
            language: profile.language,
            speaker: "default",
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[3]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!("Saved profile speech: {}", args[3]);
        return Ok(true);
    }
    if args.len() == 3 && args[0] == "text-prepare" {
        let language: Language = args[1].parse()?;
        println!(
            "{}",
            local_tts_core::text::prepare(&args[2], language).map_err(std::io::Error::other)?
        );
        return Ok(true);
    }
    if (5..=6).contains(&args.len()) && args[0] == "chatterbox-synthesize" {
        let reference = args.get(5).map(Path::new);
        let mut engine = local_tts_core::chatterbox::Chatterbox::load_with_reference(
            Path::new(&args[1]),
            reference,
        )?;
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[4],
            language: args[3].parse()?,
            speaker: "default",
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved {} samples at {} Hz",
            audio.samples.len(),
            audio.sample_rate
        );
        return Ok(true);
    }
    if matches!(args.len(), 6..=8) && args[0] == "audio-process" {
        let input = local_tts_core::read_wav(Path::new(&args[1]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let settings = local_tts_core::effects::VoiceEffects {
            bass_db: args[3].parse()?,
            brightness_db: args[4].parse()?,
            compression: args[5].parse()?,
            pitch_semitones: args.get(6).map_or(Ok(0.0), |x| x.parse())?,
            formant_semitones: args.get(7).map_or(Ok(0.0), |x| x.parse())?,
            ..Default::default()
        };
        let (audio, stats) =
            local_tts_core::effects::process(&input, settings).map_err(std::io::Error::other)?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved {} samples at {} Hz; peak {:.4} → {:.4}, protection gain {:.4}, max compression {:.2} dB",
            audio.samples.len(),
            audio.sample_rate,
            stats.input_peak,
            stats.output_peak,
            stats.peak_gain,
            stats.max_compression_db
        );
        return Ok(true);
    }
    if (args.len() == 4
        && matches!(
            args[0].as_str(),
            "pronunciation-add" | "pronunciation-remove"
        ))
        || (args.len() == 3 && args[0] == "pronunciation-list")
    {
        let language = match args[2].as_str() {
            "ru" => Language::Ru,
            "uk" => Language::Uk,
            _ => return Err("use ru or uk for pronunciation corrections".into()),
        };
        let path = Path::new(&args[1]).join(local_tts_core::pronunciations::FILE_NAME);
        let mut file = local_tts_core::pronunciations::PronunciationFile::open(&path, language)
            .map_err(std::io::Error::other)?;
        if args[0] == "pronunciation-list" {
            for (key, value) in file.data().entries() {
                println!("{key} → {value}");
            }
        } else {
            let mut data = file.data().clone();
            if args[0] == "pronunciation-add" {
                data.insert(&args[3]).map_err(std::io::Error::other)?;
            } else {
                data.remove(&args[3]).map_err(std::io::Error::other)?;
            }
            file.save(data).map_err(std::io::Error::other)?;
            println!("Saved corrections: {}", path.display());
        }
        return Ok(true);
    }
    if args.len() == 5 && args[0] == "vosk-synthesize" {
        let mut engine = local_tts_core::vosk::Vosk::load(Path::new(&args[1]))?;
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[4],
            language: Language::Ru,
            speaker: &args[3],
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved {} Russian samples at {} Hz to {}",
            audio.samples.len(),
            audio.sample_rate,
            args[2]
        );
        return Ok(true);
    }
    if args.len() == 5 && args[0] == "piper-synthesize" {
        let mut engine = local_tts_core::piper::Piper::load(Path::new(&args[1]))?;
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[4],
            language: Language::Uk,
            speaker: &args[3],
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved {} Ukrainian samples at {} Hz to {}",
            audio.samples.len(),
            audio.sample_rate,
            args[2]
        );
        return Ok(true);
    }
    if args.len() == 7 && args[0] == "vits-compare" {
        let split = match args[5].as_str() {
            "validation" => dataset::Split::Validation,
            "test" => dataset::Split::Test,
            _ => return Err("comparison requires validation or test split".into()),
        };
        let report = local_tts_core::evaluation::compare(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            split,
            args[6].parse()?,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .map_err(std::io::Error::other)?;
        println!(
            "Native held-out comparison saved: {} ({} clips)",
            args[4],
            report.samples.len()
        );
        return Ok(true);
    }
    if args.len() == 4 && args[0] == "vits-synthesize" {
        let mut engine = local_tts_core::vits::Vits::load(Path::new(&args[1]))?;
        let speaker = engine.manifest.speakers[0].clone();
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[3],
            language: engine.manifest.language,
            speaker: &speaker,
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!("Saved own-voice audio to {}", args[2]);
        return Ok(true);
    }
    if args.len() == 2 && args[0] == "voice-seal" {
        local_tts_core::integrity::seal_vits(Path::new(&args[1])).map_err(std::io::Error::other)?;
        println!(
            "Established checksum baseline for trusted package: {}",
            args[1]
        );
        return Ok(true);
    }
    if args.len() == 2 && args[0] == "vits-verify" {
        let root = Path::new(&args[1]);
        let mut voice = local_tts_core::vits::Vits::load(root)?;
        println!(
            "Native ONNX parity passed; maximum error {}",
            voice.verify(root)?
        );
        return Ok(true);
    }
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!(
            "local-tts voice-design <model-folder> <output.wav> <language> <description> <text>\nlocal-tts profile-create <new-folder> <reference.wav> <name> <language> <expressiveness>\nlocal-tts profile-synthesize <model-folder> <profile-folder> <output.wav> <text>\nlocal-tts text-prepare <language-code> <text>\nlocal-tts chatterbox-synthesize <voice-folder> <output.wav> <language-code> <text> [reference.wav]\nlocal-tts audio-process <input.wav> <output.wav> <bass-db> <brightness-db> <compression-0-to-1> [pitch-semitones] [formant-semitones]\nlocal-tts pronunciation-add <voice-folder> <ru|uk> <stressed-word>\nlocal-tts pronunciation-remove <voice-folder> <ru|uk> <word>\nlocal-tts pronunciation-list <voice-folder> <ru|uk>\nlocal-tts vosk-synthesize <voice-folder> <output.wav> <speaker> <Russian text>\nlocal-tts piper-synthesize <voice-folder> <output.wav> <speaker> <Ukrainian text>\nlocal-tts dataset-check <dataset.json>\nlocal-tts voice-check <voice.json>\nlocal-tts training-prepare <dataset.json> <prepared.json> [base-training.json [--text-v1]]\nlocal-tts synthesize <model-folder> <output.wav> <text>\nlocal-tts vits-compare <before-voice> <after-voice> <dataset.json> <new-output> <validation|test> <1-12>\nlocal-tts vits-verify <voice-folder>\nlocal-tts voice-seal <trusted-vits-folder>\nlocal-tts vits-synthesize <voice-folder> <output.wav> <text>"
        );
        return Ok(true);
    }
    if (3..=5).contains(&args.len()) && args[0] == "training-prepare" {
        let result = if args.len() >= 4 {
            let base: training::TrainingConfig = serde_json::from_slice(&std::fs::read(&args[3])?)?;
            if args.len() == 5 {
                if args[4] != "--text-v1" {
                    return Err("unknown training preparation option".into());
                }
                training::prepare_from_base_current(Path::new(&args[1]), Path::new(&args[2]), &base)
            } else {
                training::prepare_from_base(Path::new(&args[1]), Path::new(&args[2]), &base)
            }
        } else {
            training::prepare(Path::new(&args[1]), Path::new(&args[2]))
        };
        result.map_err(std::io::Error::other)?;
        println!("Prepared native training tokens: {}", args[2]);
        return Ok(true);
    }
    if args.len() == 4 && args[0] == "synthesize" {
        let mut engine = Kokoro::load(Path::new(&args[1]))?;
        println!("Kokoro variant: {}", engine.variant().label());
        let audio = engine.synthesize(&SynthesisRequest {
            text: &args[3],
            language: Language::En,
            speaker: "af_heart",
            duration_scale: 1.0,
        })?;
        audio
            .write_wav(Path::new(&args[2]))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        println!(
            "Saved {} samples at {} Hz to {}",
            audio.samples.len(),
            audio.sample_rate,
            args[2]
        );
        return Ok(true);
    }
    if args.len() != 2 {
        return Err("usage: local-tts <dataset-check|voice-check> <manifest.json>".into());
    }
    let path = Path::new(&args[1]);
    let root = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    match args[0].as_str() {
        "dataset-check" => {
            let report = dataset::validate(&dataset::read(path)?, root);
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(report.errors.is_empty())
        }
        "voice-check" => {
            let voice: VoiceManifest = serde_json::from_slice(&std::fs::read(path)?)?;
            voice.validate(root).map_err(std::io::Error::other)?;
            println!(
                "Voice package metadata and model path are valid; model graph and licensing are not verified."
            );
            Ok(true)
        }
        _ => Err("unknown command".into()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

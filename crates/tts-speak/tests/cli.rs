use std::io::Write;
use std::process::{Command, Stdio};
fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tts-speak"))
}
#[test]
fn help_errors_and_output_protection_do_not_require_models() {
    let help = cli().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("No downloads"));
    for args in [
        vec!["train"],
        vec!["--model"],
        vec![
            "--model", "missing", "--output", "out.wav", "--text", "a", "--input", "b",
        ],
        vec![
            "--model",
            "missing",
            "--output",
            "out.wav",
            "--text",
            "a",
            "--backend",
            "qwen-design",
        ],
    ] {
        assert!(!cli().args(args).output().unwrap().status.success());
    }
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("existing.wav");
    std::fs::write(&out, b"preserve").unwrap();
    let result = cli()
        .args(["--model", "missing", "--output"])
        .arg(&out)
        .args(["--text", "Hello."])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("already exists"));
    assert_eq!(std::fs::read(out).unwrap(), b"preserve");
}
#[test]
#[ignore = "requires installed Kokoro Q8"]
fn real_english_argument_file_and_stdin() {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/kokoro-en-q8");
    let dir = tempfile::tempdir().unwrap();
    let textfile = dir.path().join("text.txt");
    std::fs::write(&textfile, "Hello world.").unwrap();
    for mode in 0..3 {
        let out = dir.path().join(format!("{mode}.wav"));
        let mut command = cli();
        command.arg("--model").arg(&model).arg("--output").arg(&out);
        if mode == 0 {
            command.args(["--text", "Hello world."]);
        }
        if mode == 1 {
            command.arg("--input").arg(&textfile);
        }
        if mode == 2 {
            command.stdin(Stdio::piped());
        }
        let mut child = command.spawn().unwrap();
        if mode == 2 {
            child
                .stdin
                .take()
                .unwrap()
                .write_all(b"Hello world.")
                .unwrap();
        }
        assert!(child.wait().unwrap().success());
        let audio = local_tts_core::read_wav(&out).unwrap();
        assert_eq!(audio.sample_rate, 24000);
        assert!(audio.samples.len() > 2400);
        assert!(audio.samples.iter().any(|v| v.abs() > 0.001));
    }
}
#[test]
fn list_voices_is_json_and_needs_no_model_for_fixed_backends() {
    let out = cli()
        .args(["--list-voices", "--backend", "vosk"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let listing: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(listing["default"], "F01");
    assert_eq!(listing["languages"], serde_json::json!(["ru"]));
    assert_eq!(listing["voices"].as_array().unwrap().len(), 5);
    for args in [
        vec!["--list-voices", "--backend", "piper"],
        vec!["--list-voices", "--text", "a"],
        vec!["--list-voices", "--list-voices"],
        vec![
            "--backend",
            "vosk",
            "--model",
            "missing",
            "--output",
            "out.wav",
            "--lexicon",
            "x",
            "--text",
            "a",
        ],
    ] {
        assert!(
            !cli().args(&args).output().unwrap().status.success(),
            "{args:?}"
        );
    }
}
#[test]
#[ignore = "requires installed Kokoro Q8"]
fn lexicon_supplies_unknown_words_and_json_reports_the_result() {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/kokoro-en-q8");
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("brand.wav");
    let text = ["--text", "Meet PromoShot."];
    let missing = cli()
        .arg("--model")
        .arg(&model)
        .arg("--output")
        .arg(&out)
        .args(text)
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&missing.stderr).contains("Unknown word 'promoshot'"));
    let lexicon = dir.path().join("brand.dict");
    std::fs::write(&lexicon, "PromoShot  P R OW1 M OW0 SH AA2 T\n").unwrap();
    let result = cli()
        .arg("--model")
        .arg(&model)
        .arg("--output")
        .arg(&out)
        .arg("--lexicon")
        .arg(&lexicon)
        .arg("--json")
        .args(text)
        .output()
        .unwrap();
    assert!(result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["format"], "wav");
    assert_eq!(report["sampleRate"], 24000);
    assert!(report["seconds"].as_f64().unwrap() > 0.3);
    assert_eq!(local_tts_core::read_wav(&out).unwrap().sample_rate, 24000);
}

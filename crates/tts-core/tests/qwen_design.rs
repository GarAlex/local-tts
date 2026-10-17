use local_tts_core::qwen_design::QwenDesign;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[test]
fn pre_cancelled_loading_does_not_access_assets() {
    let result =
        QwenDesign::load_cancellable(Path::new("missing-qwen-assets"), &AtomicBool::new(true));
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("cancelled load succeeded"),
    };
    assert!(error.to_string().contains("cancelled"));
}

#[test]
#[ignore = "requires downloaded pinned Qwen VoiceDesign assets"]
fn cancellation_interrupts_real_package_verification() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../voices/qwen-voice-design");
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let worker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        flag.store(true, Ordering::Relaxed);
    });
    let result = QwenDesign::load_cancellable(&root, &cancel);
    worker.join().unwrap();
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("cancelled load succeeded"),
    };
    assert!(error.to_string().contains("cancelled"));
}

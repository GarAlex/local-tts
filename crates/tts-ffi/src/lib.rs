//! C ABI v1. See include/local_tts.h for pointer lifetimes and concurrency rules.
use local_tts_core::{
    Audio, Language, SynthesisRequest, Synthesizer, kokoro::Kokoro, piper::Piper, vits::Vits,
    vosk::Vosk,
};
use std::{
    ffi::{CStr, CString, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    ptr,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

enum Backend {
    Math(local_tts_core::math_speech::MathVoice),
    MathStudent(local_tts_core::math_speech::MathStudent),
    Compact(Box<local_tts_core::compact_english::CompactEnglish>),
    QwenDesign(Box<local_tts_core::qwen_design::QwenDesign>),
    Kokoro(Kokoro),
    Vits(Vits),
    Piper(Piper),
    Vosk(Vosk),
    Chatterbox(Box<local_tts_core::chatterbox::Chatterbox>),
}
pub struct TtsEngine {
    backend: Mutex<Backend>,
}
pub struct TtsCancel {
    flag: AtomicBool,
}
pub struct TtsAudio {
    audio: Audio,
}

/// Initialize with tts_effects_default. All four fields must be finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TtsEffects {
    pub struct_size: u32,
    pub bass_db: f32,
    pub brightness_db: f32,
    pub compression: f32,
    pub peak_ceiling: f32,
}

/// Additive settings interface; original 20-byte TtsEffects ABI is unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TtsEffectsV2 {
    pub struct_size: u32,
    pub bass_db: f32,
    pub brightness_db: f32,
    pub compression: f32,
    pub peak_ceiling: f32,
    pub pitch_semitones: f32,
    pub formant_semitones: f32,
}

#[unsafe(no_mangle)]
pub extern "C" fn tts_effects_v2_default() -> TtsEffectsV2 {
    let settings = local_tts_core::effects::VoiceEffects::default();
    TtsEffectsV2 {
        struct_size: std::mem::size_of::<TtsEffectsV2>() as u32,
        bass_db: settings.bass_db,
        brightness_db: settings.brightness_db,
        compression: settings.compression,
        peak_ceiling: settings.peak_ceiling,
        pitch_semitones: settings.pitch_semitones,
        formant_semitones: settings.formant_semitones,
    }
}

/// # Safety
/// Same lifetimes/ownership as tts_audio_process. Settings must point to a u32
/// size prefix followed by the full TtsEffectsV2 when the size matches.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_process_v2(
    audio: *const TtsAudio,
    settings: *const TtsEffectsV2,
    output: *mut *mut TtsAudio,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let source = unsafe { audio.as_ref() }.ok_or("null audio")?;
        if settings.is_null() {
            return Err("null effects settings".into());
        }
        if unsafe { settings.cast::<u32>().read() } != std::mem::size_of::<TtsEffectsV2>() as u32 {
            return Err("unsupported effects v2 struct size".into());
        }
        let settings = unsafe { settings.read() };
        let (audio, _) = local_tts_core::effects::process(
            &source.audio,
            local_tts_core::effects::VoiceEffects {
                bass_db: settings.bass_db,
                brightness_db: settings.brightness_db,
                compression: settings.compression,
                peak_ceiling: settings.peak_ceiling,
                pitch_semitones: settings.pitch_semitones,
                formant_semitones: settings.formant_semitones,
            },
        )?;
        unsafe {
            *output = Box::into_raw(Box::new(TtsAudio { audio }));
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn tts_effects_default() -> TtsEffects {
    let settings = local_tts_core::effects::VoiceEffects::default();
    TtsEffects {
        struct_size: std::mem::size_of::<TtsEffects>() as u32,
        bass_db: settings.bass_db,
        brightness_db: settings.brightness_db,
        compression: settings.compression,
        peak_ceiling: settings.peak_ceiling,
    }
}

/// # Safety
/// Audio and settings must remain live and immutable for this call. Settings
/// must point to a u32 size field, followed by the full struct when size matches.
/// Output/error slots must be writable and distinct from the input storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_process(
    audio: *const TtsAudio,
    settings: *const TtsEffects,
    output: *mut *mut TtsAudio,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let source = unsafe { audio.as_ref() }.ok_or("null audio")?;
        if settings.is_null() {
            return Err("null effects settings".into());
        }
        // Read only the prefix before verifying layout; never reference a short struct.
        if unsafe { settings.cast::<u32>().read() } != std::mem::size_of::<TtsEffects>() as u32 {
            return Err("unsupported effects struct size".into());
        }
        let settings = unsafe { settings.read() };
        let (audio, _) = local_tts_core::effects::process(
            &source.audio,
            local_tts_core::effects::VoiceEffects {
                bass_db: settings.bass_db,
                brightness_db: settings.brightness_db,
                compression: settings.compression,
                peak_ceiling: settings.peak_ceiling,
                ..Default::default()
            },
        )?;
        unsafe {
            *output = Box::into_raw(Box::new(TtsAudio { audio }));
        }
        Ok(())
    })
}

fn boundary(error: *mut *mut c_char, f: impl FnOnce() -> Result<(), String>) -> i32 {
    // The caller provides a fresh writable error slot, or null.
    if !error.is_null() {
        unsafe {
            *error = ptr::null_mut();
        }
    }
    let result =
        catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err("internal Rust panic".into()));
    match result {
        Ok(()) => 0,
        Err(message) => {
            if !error.is_null() {
                let message = CString::new(message.replace('\0', " ")).expect("sanitized error");
                unsafe {
                    *error = message.into_raw();
                }
            }
            1
        }
    }
}
unsafe fn text<'a>(value: *const c_char) -> Result<&'a str, String> {
    if value.is_null() {
        return Err("null string argument".into());
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|_| "argument must be UTF-8".into())
}

#[unsafe(no_mangle)]
pub extern "C" fn tts_abi_version() -> u32 {
    1
}

/// # Safety
/// Input must be live NUL-terminated UTF-8. Output/error must be fresh, writable,
/// distinct slots. Free successful output with tts_string_free exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_text_prepare(
    input: *const c_char,
    language: u32,
    output: *mut *mut c_char,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null text output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let language = Language::from_abi(language)?;
        let prepared = local_tts_core::text::prepare(unsafe { text(input)? }, language)?;
        let prepared = CString::new(prepared).map_err(|_| "invalid output text")?;
        unsafe {
            *output = prepared.into_raw();
        }
        Ok(())
    })
}

/// # Safety
/// Free only strings returned by tts_text_prepare, exactly once. Null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe {
            drop(CString::from_raw(value));
        }
    }
}

/// # Safety
/// Strings must be valid NUL-terminated UTF-8; output/error slots must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_load(
    folder: *const c_char,
    backend: u32,
    output: *mut *mut TtsEngine,
    error: *mut *mut c_char,
) -> i32 {
    unsafe { tts_engine_load_with_reference(folder, backend, ptr::null(), output, error) }
}

/// Load an engine with an optional custom speaker reference. The reference is
/// used by the Chatterbox backend only; pass NULL to retain the package default.
/// The reference must be a mono 24 kHz WAV file no longer than 30 seconds.
/// Existing `tts_engine_load` behavior and backend IDs remain unchanged.
///
/// # Safety
/// Strings must be valid NUL-terminated UTF-8 and remain live for the call.
/// Output/error slots must be writable, distinct and fresh.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_load_with_reference(
    folder: *const c_char,
    backend: u32,
    reference: *const c_char,
    output: *mut *mut TtsEngine,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null engine output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let folder = Path::new(unsafe { text(folder)? });
        let reference = if reference.is_null() {
            None
        } else {
            Some(Path::new(unsafe { text(reference)? }))
        };
        if reference.is_some() && backend != 4 {
            return Err("custom references require the Chatterbox backend".into());
        }
        let backend = match backend {
            9 => Backend::MathStudent(
                local_tts_core::math_speech::MathStudent::load(folder)
                    .map_err(|e| e.to_string())?,
            ),
            8 => Backend::Math(local_tts_core::math_speech::MathVoice::load(folder)?),
            6 | 7 => Backend::Compact(Box::new(
                local_tts_core::compact_english::CompactEnglish::load(
                    folder,
                    if backend == 6 {
                        local_tts_core::compact_english::CompactKind::Inflect
                    } else {
                        local_tts_core::compact_english::CompactKind::Kitten
                    },
                )
                .map_err(|e| e.to_string())?,
            )),
            5 => Backend::QwenDesign(Box::new(
                local_tts_core::qwen_design::QwenDesign::load(folder).map_err(|e| e.to_string())?,
            )),
            0 => Backend::Kokoro(Kokoro::load(folder).map_err(|e| e.to_string())?),
            1 => Backend::Vits(Vits::load(folder).map_err(|e| e.to_string())?),
            2 => Backend::Piper(Piper::load(folder).map_err(|e| e.to_string())?),
            3 => Backend::Vosk(Vosk::load(folder).map_err(|e| e.to_string())?),
            4 => Backend::Chatterbox(Box::new(
                local_tts_core::chatterbox::Chatterbox::load_with_reference(folder, reference)
                    .map_err(|e| e.to_string())?,
            )),
            _ => {
                return Err(
                    "unknown backend (0=Kokoro, 1=VITS, 2=Piper Ukrainian, 3=Vosk Russian, 4=Chatterbox multilingual, 5=Qwen VoiceDesign, 6=Inflect Nano, 7=Kitten Nano, 8=English math fragments, 9=English math student)".into(),
                );
            }
        };
        unsafe {
            *output = Box::into_raw(Box::new(TtsEngine {
                backend: Mutex::new(backend),
            }));
        }
        Ok(())
    })
}

/// Load a portable Chatterbox reference profile with separately installed weights.
/// The profile supplies reference audio and expressiveness. Its language is a
/// suggestion; synthesis requests still specify language explicitly.
/// # Safety
/// Strings must be NUL-terminated UTF-8; output/error slots fresh and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_load_profile(
    model_folder: *const c_char,
    profile_folder: *const c_char,
    output: *mut *mut TtsEngine,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null engine output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let model = Path::new(unsafe { text(model_folder)? });
        let profile = Path::new(unsafe { text(profile_folder)? });
        let engine = local_tts_core::voice_profile::load_engine(model, profile)?;
        unsafe {
            *output = Box::into_raw(Box::new(TtsEngine {
                backend: Mutex::new(Backend::Chatterbox(Box::new(engine))),
            }));
        }
        Ok(())
    })
}

/// Configure Chatterbox emotion intensity (0–1, default 0.5).
/// # Safety
/// Engine must remain live for the call; error slot must be fresh and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_set_exaggeration(
    engine: *mut TtsEngine,
    value: f32,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        let engine = unsafe { engine.as_ref() }.ok_or("null engine")?;
        let mut backend = engine
            .backend
            .lock()
            .map_err(|_| "engine poisoned; reload it")?;
        match &mut *backend {
            Backend::Chatterbox(model) => model.set_exaggeration(value).map_err(|e| e.to_string()),
            _ => Err("expressiveness requires the Chatterbox backend".into()),
        }
    })
}

/// # Safety
/// Engine must remain live; error slot must be writable. Reloads are serialized with synthesis.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_reload_pronunciations(
    engine: *mut TtsEngine,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        let engine = unsafe { engine.as_ref() }.ok_or("null engine")?;
        let mut backend = engine
            .backend
            .lock()
            .map_err(|_| "engine poisoned; reload it")?;
        match &mut *backend {
            Backend::Piper(engine) => engine.reload_pronunciations().map_err(|e| e.to_string()),
            Backend::Vosk(engine) => engine.reload_pronunciations().map_err(|e| e.to_string()),
            _ => Err("pronunciation corrections support pretrained Russian/Ukrainian only".into()),
        }
    })
}

/// # Safety
/// Engine must be live; strings and output slots valid. Do not free during calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_synthesize(
    engine: *mut TtsEngine,
    input: *const c_char,
    language: u32,
    speaker: *const c_char,
    duration_scale: f32,
    output: *mut *mut TtsAudio,
    error: *mut *mut c_char,
) -> i32 {
    unsafe {
        tts_synthesize_cancellable(
            engine,
            input,
            language,
            speaker,
            duration_scale,
            ptr::null(),
            output,
            error,
        )
    }
}
/// # Safety
/// Same as tts_synthesize; cancellation handle must remain live until return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_synthesize_cancellable(
    engine: *mut TtsEngine,
    input: *const c_char,
    language: u32,
    speaker: *const c_char,
    duration_scale: f32,
    cancel: *const TtsCancel,
    output: *mut *mut TtsAudio,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null audio output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let engine = unsafe { engine.as_ref() }.ok_or("null engine")?;
        let language = Language::from_abi(language)?;
        let request = SynthesisRequest {
            text: unsafe { text(input)? },
            speaker: unsafe { text(speaker)? },
            language,
            duration_scale,
        };
        let fallback = AtomicBool::new(false);
        let cancel = unsafe { cancel.as_ref() }.map_or(&fallback, |c| &c.flag);
        let mut backend = engine
            .backend
            .lock()
            .map_err(|_| "engine poisoned; reload it")?;
        let audio = match &mut *backend {
            Backend::QwenDesign(_) => {
                return Err(
                    "Qwen VoiceDesign requires tts_synthesize_design with a description".into(),
                );
            }
            Backend::MathStudent(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Math(engine) => {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err("synthesis cancelled".into());
                }
                let result = engine.synthesize(&request);
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err("synthesis cancelled".into());
                }
                result
            }
            Backend::Compact(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Kokoro(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Vits(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Piper(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Vosk(engine) => engine.synthesize_cancellable(&request, cancel),
            Backend::Chatterbox(engine) => engine.synthesize_cancellable(&request, cancel),
        }
        .map_err(|e| e.to_string())?;
        unsafe {
            *output = Box::into_raw(Box::new(TtsAudio { audio }));
        }
        Ok(())
    })
}

/// Generate speech from a natural-language voice description using Qwen VoiceDesign.
/// Text: <=300 characters, description: <=500; short-utterance token/output bounds.
/// # Safety
/// Engine/strings/cancel must remain live; output/error slots fresh and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_synthesize_design(
    engine: *mut TtsEngine,
    input: *const c_char,
    language: u32,
    description: *const c_char,
    cancel: *const TtsCancel,
    output: *mut *mut TtsAudio,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        if output.is_null() {
            return Err("null audio output".into());
        }
        unsafe {
            *output = ptr::null_mut();
        }
        let engine = unsafe { engine.as_ref() }.ok_or("null engine")?;
        let language = Language::from_abi(language)?;
        let input = unsafe { text(input)? };
        let description = unsafe { text(description)? };
        let fallback = AtomicBool::new(false);
        let cancel = unsafe { cancel.as_ref() }.map_or(&fallback, |c| &c.flag);
        let mut backend = engine
            .backend
            .lock()
            .map_err(|_| "engine poisoned; reload it")?;
        let audio = match &mut *backend {
            Backend::QwenDesign(model) => model
                .synthesize_design(input, language, description, cancel)
                .map_err(|e| e.to_string())?,
            _ => return Err("voice design requires the Qwen VoiceDesign backend".into()),
        };
        unsafe {
            *output = Box::into_raw(Box::new(TtsAudio { audio }));
        }
        Ok(())
    })
}

/// # Safety
/// Audio must be live; returned samples remain valid until tts_audio_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_samples(audio: *const TtsAudio) -> *const f32 {
    unsafe { audio.as_ref() }.map_or(ptr::null(), |a| a.audio.samples.as_ptr())
}
/// # Safety
/// Audio must be live or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_length(audio: *const TtsAudio) -> usize {
    unsafe { audio.as_ref() }.map_or(0, |a| a.audio.samples.len())
}
/// # Safety
/// Audio must be live or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_sample_rate(audio: *const TtsAudio) -> u32 {
    unsafe { audio.as_ref() }.map_or(0, |a| a.audio.sample_rate)
}
/// # Safety
/// Audio must be live, strings NUL-terminated UTF-8, error slot writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_write_wav(
    audio: *const TtsAudio,
    path: *const c_char,
    error: *mut *mut c_char,
) -> i32 {
    boundary(error, || {
        let audio = unsafe { audio.as_ref() }.ok_or("null audio")?;
        audio
            .audio
            .write_wav(Path::new(unsafe { text(path)? }))
            .map_err(|e| e.to_string())
    })
}
/// # Safety
/// Free exactly once, after all engine calls complete. Null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_engine_free(engine: *mut TtsEngine) {
    if !engine.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(engine));
        }));
    }
}
/// # Safety
/// Free exactly once, after all sample readers complete. Null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_audio_free(audio: *mut TtsAudio) {
    if !audio.is_null() {
        unsafe {
            drop(Box::from_raw(audio));
        }
    }
}
/// # Safety
/// Pass only an error returned by this library, exactly once. Null is allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_error_free(error: *mut c_char) {
    if !error.is_null() {
        unsafe {
            drop(CString::from_raw(error));
        }
    }
}

/// Allocate one cancellation token per job. Tokens are sticky and never reset.
#[unsafe(no_mangle)]
pub extern "C" fn tts_cancel_new() -> *mut TtsCancel {
    Box::into_raw(Box::new(TtsCancel {
        flag: AtomicBool::new(false),
    }))
}
/// # Safety
/// Token must remain live during this call. Safe to call from another thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_cancel_request(cancel: *const TtsCancel) {
    if let Some(cancel) = unsafe { cancel.as_ref() } {
        cancel.flag.store(true, Ordering::Relaxed);
    }
}
/// # Safety
/// Free exactly once after synthesis and cancellation callers finish. Null allowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tts_cancel_free(cancel: *mut TtsCancel) {
    if !cancel.is_null() {
        unsafe {
            drop(Box::from_raw(cancel));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn custom_reference_requires_supported_backend_and_clears_output() {
        let folder = CString::new("missing-model").unwrap();
        let reference = CString::new("reference.wav").unwrap();
        let mut output = std::ptr::dangling_mut::<TtsEngine>();
        let mut error = ptr::null_mut();
        unsafe {
            assert_eq!(
                tts_engine_load_with_reference(
                    folder.as_ptr(),
                    0,
                    reference.as_ptr(),
                    &mut output,
                    &mut error
                ),
                1
            );
            assert!(output.is_null());
            assert!(
                CStr::from_ptr(error)
                    .to_str()
                    .unwrap()
                    .contains("Chatterbox")
            );
            tts_error_free(error);
        }
    }
    #[test]
    fn french_text_uses_additive_language_id() {
        unsafe {
            let mut output = ptr::null_mut();
            let mut error = ptr::null_mut();
            assert_eq!(
                tts_text_prepare(c"ÉCOLE ET CŒUR".as_ptr(), 4, &mut output, &mut error),
                0
            );
            assert!(error.is_null());
            assert_eq!(CStr::from_ptr(output).to_str().unwrap(), "école et cœur");
            tts_string_free(output);
        }
    }
    #[test]
    fn text_preparation_owns_strings_and_clears_failed_outputs() {
        unsafe {
            let mut output = ptr::null_mut();
            let mut error = ptr::null_mut();
            assert_eq!(
                tts_text_prepare(c"2 kg and $5.20".as_ptr(), 0, &mut output, &mut error),
                0
            );
            assert!(error.is_null());
            assert_eq!(
                CStr::from_ptr(output).to_str().unwrap(),
                "two kilograms and five dollars and twenty cents"
            );
            tts_string_free(output);
            for (input, language) in [
                (c"5".as_ptr(), 99),
                (ptr::null(), 0),
                (c"2026-02-30".as_ptr(), 0),
            ] {
                output = ptr::dangling_mut();
                assert_eq!(
                    tts_text_prepare(input, language, &mut output, &mut error),
                    1
                );
                assert!(output.is_null());
                assert!(!error.is_null());
                tts_error_free(error);
            }
            let invalid_utf8 = [0xff_u8, 0];
            assert_eq!(
                tts_text_prepare(invalid_utf8.as_ptr().cast(), 0, &mut output, &mut error),
                1
            );
            assert!(output.is_null());
            tts_error_free(error);
            assert_eq!(
                tts_text_prepare(c"5".as_ptr(), 0, ptr::null_mut(), ptr::null_mut()),
                1
            );
            tts_string_free(ptr::null_mut());
        }
    }
    #[test]
    fn handles_support_thread_sharing() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TtsEngine>();
        assert_send_sync::<TtsAudio>();
    }
    #[test]
    fn effects_v2_layout_errors_do_not_read_legacy_tail() {
        let source = TtsAudio {
            audio: Audio {
                sample_rate: 24000,
                samples: vec![0.2; 10],
            },
        };
        let old = tts_effects_default();
        let mut output = ptr::dangling_mut();
        let mut error = ptr::null_mut();
        unsafe {
            assert_eq!(
                tts_audio_process_v2(
                    &source,
                    (&old as *const TtsEffects).cast(),
                    &mut output,
                    &mut error
                ),
                1
            );
            assert!(output.is_null());
            tts_error_free(error);
            let settings = tts_effects_v2_default();
            assert_eq!(
                tts_audio_process_v2(&source, &settings, &mut output, &mut error),
                0
            );
            assert!(error.is_null());
            assert_eq!((*output).audio.samples, source.audio.samples);
            tts_audio_free(output);
        }
    }
    #[test]
    fn effects_reject_short_layout_and_clear_failed_output() {
        let source = TtsAudio {
            audio: Audio {
                sample_rate: 24000,
                samples: vec![0.2; 10],
            },
        };
        let size_only = 4_u32;
        let mut output = ptr::dangling_mut();
        let mut error = ptr::null_mut();
        unsafe {
            assert_eq!(
                tts_audio_process(
                    &source,
                    (&size_only as *const u32).cast(),
                    &mut output,
                    &mut error
                ),
                1
            );
            assert!(output.is_null());
            assert!(
                CStr::from_ptr(error)
                    .to_str()
                    .unwrap()
                    .contains("struct size")
            );
            tts_error_free(error);
            let settings = tts_effects_default();
            assert_eq!(
                tts_audio_process(&source, &settings, &mut output, &mut error),
                0
            );
            assert!(error.is_null());
            assert_eq!((*output).audio.samples, source.audio.samples);
            assert_ne!(
                (*output).audio.samples.as_ptr(),
                source.audio.samples.as_ptr()
            );
            tts_audio_free(output);
            let invalid = TtsAudio {
                audio: Audio {
                    sample_rate: 24000,
                    samples: vec![f32::NAN],
                },
            };
            assert_eq!(
                tts_audio_process(&invalid, &settings, &mut output, &mut error),
                1
            );
            assert!(output.is_null());
            tts_error_free(error);
        }
    }
    #[test]
    fn panic_and_invalid_arguments_become_errors() {
        unsafe {
            let mut error = ptr::null_mut();
            assert_eq!(boundary(&mut error, || panic!("fixture")), 1);
            assert_eq!(
                CStr::from_ptr(error).to_str().unwrap(),
                "internal Rust panic"
            );
            tts_error_free(error);
            let invalid_utf8 = [255_u8, 0];
            let mut output = ptr::null_mut();
            assert_eq!(
                tts_engine_load(invalid_utf8.as_ptr().cast(), 0, &mut output, &mut error),
                1
            );
            assert!(output.is_null());
            assert!(CStr::from_ptr(error).to_str().unwrap().contains("UTF-8"));
            tts_error_free(error);
            assert_eq!(
                tts_synthesize(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    1.0,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                1
            );
            assert_eq!(tts_audio_write_wav(ptr::null(), ptr::null(), &mut error), 1);
            tts_error_free(error);
        }
    }
}

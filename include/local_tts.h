#ifndef LOCAL_TTS_H
#define LOCAL_TTS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TtsEngine TtsEngine;
typedef struct TtsAudio TtsAudio;
typedef struct TtsCancel TtsCancel;
enum { TTS_KOKORO = 0, TTS_VITS = 1, TTS_PIPER_UK = 2, TTS_VOSK_RU = 3, TTS_CHATTERBOX = 4, TTS_QWEN_DESIGN = 5, TTS_INFLECT_NANO = 6, TTS_KITTEN_NANO = 7, TTS_MATH_EN = 8, TTS_MATH_STUDENT_EN = 9 };
enum { TTS_ENGLISH = 0, TTS_RUSSIAN = 1, TTS_UKRAINIAN = 2,
       TTS_GERMAN = 3, TTS_FRENCH = 4, TTS_SPANISH = 5, TTS_ITALIAN = 6,
       TTS_PORTUGUESE = 7, TTS_POLISH = 8, TTS_DUTCH = 9 };
/* ABI v1. All strings are NUL-terminated UTF-8. Return 0 on success, 1 on error.
   Use fresh output/error slots for each call. Failed calls set output to NULL.
   Error may be NULL to suppress details; otherwise free the returned error.
   Live engine calls are serialized internally. Independent engines may run in
   parallel. Free handles only after all callers/readers finish; never copy
   ownership. Invalid/dangling pointers cannot be detected by this ABI.
   Synthesis is blocking; host apps should invoke it on a worker thread.
   Only trusted model packages should be loaded. No Python or network needed. */
uint32_t tts_abi_version(void);
/* Model-free text-v1 expansion and limited contextual Cyrillic stress. No saved
   voice corrections are loaded. 10000-character input/output limit. EN dates
   use ISO YYYY-MM-DD; RU/UK also accept DD.MM.YYYY. See docs/LANGUAGE.md for
   supported forms and ambiguity errors. Output is separately owned UTF-8;
   free with tts_string_free. Output/error slots must be distinct and fresh. */
int32_t tts_text_prepare(const char *text, uint32_t language, char **output, char **error);
void tts_string_free(char *text);
/* TTS_KOKORO recognizes pinned FP32, Q8 and Q4 packages from model.onnx.
   Select a variant by passing its folder; backend IDs and ABI version are unchanged. */
/* TTS_CHATTERBOX: pinned multilingual Q4 package, speaker "default",
   duration_scale 1.0; all listed languages except Ukrainian. */
/* TTS_QWEN_DESIGN loads separately pinned INT4 assets. Use tts_synthesize_design. */
int32_t tts_engine_load(const char *folder, uint32_t backend, TtsEngine **output, char **error);
/* Additive Chatterbox zero-shot voice-cloning loader. Pass a mono 24 kHz WAV
   reference (at most 30 seconds) to condition the generated voice, or NULL to
   use the package's default reference. Other backends reject a non-NULL reference. */
int32_t tts_engine_load_with_reference(const char *folder, uint32_t backend,
                                       const char *reference_wav,
                                       TtsEngine **output, char **error);
/* Load a portable reference profile with separately installed Chatterbox weights.
   Reference checksums and model revision are verified. Profile language is a
   suggestion; synthesis still takes an explicit language. Output NULL on failure. */
int32_t tts_engine_load_profile(const char *model_folder, const char *profile_folder,
                               TtsEngine **output, char **error);
/* Chatterbox emotion intensity: finite 0–1, default 0.5. Serialized with
   synthesis. Failed calls retain the previous value. Other backends reject it.
   This adjusts intensity, not a named mood or speaker identity. */
int32_t tts_engine_set_exaggeration(TtsEngine *engine, float value, char **error);
/* RU/UK engines load optional pronunciations.json from the model folder.
   Reload saved corrections without reloading the neural model. Serialized with
   synthesis; a failed reload retains previous corrections and reports an error.
   Other backends reject this operation. Explicit input stress takes precedence. */
int32_t tts_engine_reload_pronunciations(TtsEngine *engine, char **error);
int32_t tts_synthesize(TtsEngine *engine, const char *text, uint32_t language,
                       const char *speaker, float duration_scale, TtsAudio **output, char **error);
/* One sticky token per job. Request cancellation from another thread; free only
   after synthesis and all request calls finish. NULL token disables cancellation.
   Observed between chunks (also between Chatterbox token steps): the current
   model invocation completes. Cancellation
   returns status 1, "synthesis cancelled", and NULL audio; partial audio is discarded.
   Existing ABI v1 entry points retain their behavior. */
TtsCancel *tts_cancel_new(void);
void tts_cancel_request(const TtsCancel *cancel);
void tts_cancel_free(TtsCancel *cancel);
int32_t tts_synthesize_cancellable(TtsEngine *engine, const char *text, uint32_t language,
                                  const char *speaker, float duration_scale, const TtsCancel *cancel,
                                  TtsAudio **output, char **error);
/* Qwen VoiceDesign: describe a voice, then generate a short utterance. Native,
   offline. EN/RU/DE/FR/ES/IT/PT; UK/PL/NL rejected. Max 300 text characters and
   500 description characters, <=240 codec steps. Token-limit failures discard
   partial output. NULL cancellation disables cancellation. Existing ownership
   rules apply. Use a worker thread. Default sampling is stochastic. */
int32_t tts_synthesize_design(TtsEngine *engine, const char *text, uint32_t language,
                              const char *description, const TtsCancel *cancel,
                              TtsAudio **output, char **error);
/* Borrowed immutable mono f32 PCM. Valid until audio handle is freed.
   NULL handles return NULL/zero. No allocator crosses the ABI. */
const float *tts_audio_samples(const TtsAudio *audio);
size_t tts_audio_length(const TtsAudio *audio);
uint32_t tts_audio_sample_rate(const TtsAudio *audio);
int32_t tts_audio_write_wav(const TtsAudio *audio, const char *path, char **error);
/* Whole-clip mono processing, 8–96 kHz, nonempty finite PCM, <= ten minutes.
   Initialize with tts_effects_default. Exact struct_size required in this release.
   Shelves: 180 Hz / 3000 Hz, [-12,12] dB. Compression [0,1], no makeup gain.
   Peak ceiling [0.1,1], default .98: global attenuation if peaks exceed ceiling.
   Neutral settings preserve PCM below ceiling. Output is separately owned;
   source and settings remain immutable and live throughout the blocking call.
   Independent calls may share immutable source PCM. Use a worker for long clips.
   No model/retraining required. Duration and sample rate are preserved. */
typedef struct TtsEffects {
    uint32_t struct_size;
    float bass_db;
    float brightness_db;
    float compression;
    float peak_ceiling;
} TtsEffects;
TtsEffects tts_effects_default(void);
int32_t tts_audio_process(const TtsAudio *audio, const TtsEffects *settings,
                         TtsAudio **output, char **error);
/* Additive pitch/formant interface; ABI v1 and TtsEffects remain unchanged.
   Same audio limits/ownership as above. Pitch [-12,12], formants [-6,6]
   semitones, default zero. Pitch preserves duration and compensates formants;
   formant shifting changes the spectral envelope independently. Approximate
   spectral processing may smear transients. Both zero bypass spectral processing.
   Initialize with tts_effects_v2_default; exact 28-byte struct_size required. */
typedef struct TtsEffectsV2 {
    uint32_t struct_size;
    float bass_db;
    float brightness_db;
    float compression;
    float peak_ceiling;
    float pitch_semitones;
    float formant_semitones;
} TtsEffectsV2;
TtsEffectsV2 tts_effects_v2_default(void);
int32_t tts_audio_process_v2(const TtsAudio *audio, const TtsEffectsV2 *settings,
                            TtsAudio **output, char **error);
void tts_engine_free(TtsEngine *engine);
void tts_audio_free(TtsAudio *audio);
void tts_error_free(char *error);
#ifdef __cplusplus
}
#endif
#endif

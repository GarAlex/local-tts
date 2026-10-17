# Saved reference voice profiles

Select **Multilingual / voice cloning — Chatterbox**, prepare or choose a reference
WAV, set language and Expressiveness, then open **Saved voice profiles**. Enter a
name and choose **Save new profile…**. Choose a new folder name in the save dialog.
Existing profile folders are never overwritten. **Open profile…** restores a
profile in a later Studio session. Loading clears stale generated audio.

A profile contains `profile.json` and a copied `reference.wav`. The JSON records
schema version 1, display name, suggested language, expressiveness, pinned model
revision and reference SHA-256. Moving the whole folder works; the original
recording is no longer needed. Models remain separately installed. Checksums
identify changed files; they are not signatures or proof of speaker rights.
Use recordings you have permission to use. Profile copies contain voice data;
share or store them accordingly. Fine-tuning remains the main own-voice training
workflow. A profile is reference conditioning, not trained weights.

Profiles support the current Chatterbox languages, including English and Russian.
Ukrainian is rejected. The profile language is a suggestion for cross-language
cloning: library callers still choose each synthesis request's language.
Expressiveness is finite 0–1, default 0.5. Model revision mismatches fail explicitly.
Metadata and audio files are bounded; symlink assets and altered references fail
validation. Manually editing a reference requires creating a new profile.

## CLI

```sh
local-tts profile-create my-voice-profile reference.wav 'My voice' en 0.6
local-tts profile-synthesize voices/chatterbox-multilingual my-voice-profile \
  speech.wav 'Hello from my saved voice.'
```

The CLI synthesizes using the profile's suggested language. Reference requirements
are the same as the Chatterbox loader: non-silent finite mono 24 kHz WAV, 0.1–30 s.
Studio preparation offers conversion/trimming; profile creation copies a prepared
reference without changing its PCM.

## Rust and C/C++

Rust: `voice_profile::VoiceProfile::create` / `load` manage packages.
`voice_profile::load_engine(model_folder, profile_folder)` returns a configured
`Chatterbox` that implements `Synthesizer`. Inspect `VoiceProfile::language` if
you want to use the saved suggested language.

C/C++: `tts_engine_load_profile(model_folder, profile_folder, &engine, &error)`
loads the same package. Existing synthesis/audio ownership rules apply. This is
an additive ABI-v1 function; models and Python are not included in the profile.
See `examples/cpp/profile_test.cpp` for a standalone consumer.

## Verification

Package tests cover source removal, folder relocation, changed audio, existing
folder rejection, invalid language/intensity/name/revision and oversized metadata.
CLI and independent C++ consumers generated English speech using a saved profile.
Live Studio profile opening restored English and intensity 0.6; saving preserved
the edited display name and copied the reference. These checks validate storage,
conditioning and integration, not similarity to a new speaker. Current release
archives predate these features; source/development builds contain them.

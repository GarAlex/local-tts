# Native embedding

Rust consumers depend on `local-tts-core`; see `crates/tts-core/examples/speak.rs`.
C and C++ consumers use [local_tts.h](../include/local_tts.h). ABI version 1 exposes
Kokoro English, pretrained Vosk Russian/Piper Ukrainian and exported own VITS voices.
Language support and usable speech quality depend on the loaded voice.
Kokoro FP32/Q8/Q4 packages share backend ID 0; choose the package folder with
`Kokoro::load` or `tts_engine_load`. Model hashes identify the supported variant.
See [quantized packages](KOKORO_VARIANTS.md).

Current text normalization is reusable without a model through
`local_tts_core::text::prepare` and additive ABI-v1 `tts_text_prepare`.
Free its owned UTF-8 output with `tts_string_free`; use separate fresh output/error
slots. It expands supported EN/RU/UK dates, quantities and abbreviations and
applies limited contextual Cyrillic stress, without loading saved voice entries.
Synthesis automatically uses the chosen engine's contract. Old exported VITS
voices retain legacy behavior; new runs record `normalization: text-v1` for shared
training/inference. See [language forms and limits](LANGUAGE.md).

```sh
cargo build --release -p local-tts-ffi
```

Outputs are `target/release/liblocal_tts.dylib` and `liblocal_tts.a` on macOS,
`liblocal_tts.so` and `liblocal_tts.a` on Linux, or `local_tts.dll` plus static/import
libraries on Windows. macOS has been compiled and exercised; Linux/Windows
packaging and consuming builds remain unverified.

Example development build on macOS/Linux:

```sh
cargo build -p local-tts-ffi
c++ -std=c++17 -I include examples/cpp/speak.cpp -L target/debug -llocal_tts \
  -Wl,-rpath,"$PWD/target/debug" -o target/speak-cpp
target/speak-cpp voices/kokoro-en target/cpp-speech.wav
```

The C++ example uses RAII to free handles and errors on every path. Call synthesis
from a host worker thread to keep the UI responsive. One engine serializes its
calls internally; separate engines can run independently. Audio handles own mono
float PCM and can outlive the engine. Sample pointers are borrowed until the audio
handle is freed. Playback/device selection belongs to your host application.

Each call takes fresh output/error slots. Free errors with `tts_error_free`, audio
with `tts_audio_free`, and engines with `tts_engine_free`; do not use the host
allocator. Null free calls are harmless. Never free an engine with active calls,
or audio while sample readers are active. Rust panics in fallible operations are
converted to errors when unwinding is enabled; invalid C pointers, out-of-memory
aborts and native runtime faults cannot be recovered by this interface.

Paths, text and speaker names are NUL-terminated UTF-8. Language constants are
English=0, Russian=1, Ukrainian=2; backend constants are Kokoro=0, VITS=1, Piper Ukrainian=2 and Vosk Russian=3.
See [Cyrillic backend examples](CYRILLIC.md) for speaker names and commands.
For Chatterbox zero-shot voice cloning, use the additive
`tts_engine_load_with_reference` entry point with a permitted mono 24 kHz WAV
reference up to 30 seconds. Passing NULL keeps the packaged default reference;
the reference is encoded once during engine loading. Fine-tuning remains the
recommended path for a stable production voice. Configure model emotion intensity
with `tts_engine_set_exaggeration(engine, value, &error)` (finite 0–1, default 0.5).
Calls are serialized with synthesis; errors retain the previous value. Other
backends reject this operation. Intensity does not select a named mood or age.
For pretrained Russian/Ukrainian engines, an optional `pronunciations.json` in the
voice folder supplies user-authored word stress. After editing it, call
`tts_engine_reload_pronunciations(engine, &error)`: zero means success, a nonzero
result returns an owned error string. Reload is serialized with synthesis, changes
only corrections, and retains the previous engine corrections on failure. A
missing file clears them; other backends return an unsupported-operation error.
The additive function keeps ABI version 1 and existing IDs unchanged. Rust callers
use `reload_pronunciations` or `set_pronunciations`; see
[saved stress format and editing](CYRILLIC.md#saved-word-stress).

The speaker must match the package (`af_heart` for the provided Kokoro voice).
Duration is 0.5–2.0; values greater than one produce slower speech. Synthesis is
blocking; streaming is not implemented. Text requests
automatically split into model-sized chunks at sentence/word boundaries and join
with 120 ms silence. Input and encoded text are limited to 10,000 characters and
output to ten minutes. Words exceeding a chunk limit fail without truncation.

For cancellation, Rust callers use `Kokoro::synthesize_cancellable` or
`Vits::synthesize_cancellable` (also `Vosk`/`Piper`) with a shared `AtomicBool`. C/C++ callers allocate
one `tts_cancel_new` token per job, pass it to `tts_synthesize_cancellable`, and
call `tts_cancel_request` from another thread. Tokens are sticky: allocate a new
one for the next job. Keep the token alive until synthesis and cancellation calls
return, then `tts_cancel_free`. Cancellation is checked between chunks and after
the final chunk. Model loading, mutex waits and a running model invocation finish
before the cancellation error is returned. Partial audio is discarded; the loaded
engine remains reusable. Existing `tts_synthesize` calls remain uncancellable.

Ship the shared library, trusted model folder, header and required notices.
Inference requires neither Python nor a server. Static consumers must also link
platform system libraries required by ONNX Runtime; shared linking is the tested
consumer path. This is a development ABI; retain the version check when embedding.
A relocatable macOS shared-library SDK and CMake consumer are generated by
`scripts/release.py`; see [release process](RELEASE.md). Public release notice
gaps, compatibility guarantees and other-platform verification remain.

Run C++ verification with the installed English assets:

```sh
python3 scripts/test_cpp.py
# Also exercise an exported own voice (language / speaker / supported text):
python3 scripts/test_cpp.py target/gui-training-run/voice 0 engineering-fixture ab
```

## Package integrity

All backends hash model assets before loading. Pretrained Cyrillic model/config
hashes and the Russian dictionary are pinned in the library. Kokoro model/tokenizer/style hashes
are pinned in the library; the CMU pronunciation dictionary is editable and is
not pinned at load time. VITS exports now include `checksums.json`, covering the
model, token inventory, voice metadata, rights, license and parity references.
The loader requires hashes for model, tokens and voice metadata and verifies all
listed files. Hashing uses bounded buffers and rejects files above two gigabytes.

Older trusted VITS packages must be re-exported or explicitly baselined:

```sh
cargo run -p local-tts-cli -- voice-seal /path/to/trusted/voice
```

Sealing trusts current contents and replaces their checksum baseline. It cannot
prove that an old package is authentic or unmodified. Hash manifests detect
accidental changes; an attacker who can replace both files and hashes can bypass
this check. They are not signatures. Keep packages immutable while loading;
verification and subsequent reads are separate operations. Recheck numerical
parity after any intentional model/token change.


Portable reference profiles can be loaded with `tts_engine_load_profile` while
keeping model weights installed separately. See [saved profiles](VOICE_PROFILES.md)
for the package format, Rust/CLI workflows and C++ example. Existing ABI-v1 entry
points are unchanged.

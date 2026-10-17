# Local voice design

Choose **Voice design — Qwen3** in Studio. Download the pinned model once, then
enter a short speech text and a voice description, for example:
“A calm older male voice, low pitch, clear diction, gentle delivery.”
Generate, play and save through the normal Studio audio controls.

This backend creates speech from descriptive instructions without a reference
recording or fine-tuning. Age, mood, accent and timbre requests are probabilistic;
they are not exact sliders or guarantees. Repeated generation is stochastic.
Fine-tuning remains the main path for training your own voice. Chatterbox remains
the reference-cloning backend; its saved profiles do not apply to Qwen design.

Supported project languages: English, Russian, German, French, Spanish, Italian
and Portuguese. Ukrainian, Polish and Dutch are rejected explicitly. Use concise
English voice descriptions initially. Cross-language style adherence needs listening
assessment; model capability does not establish pronunciation or identity quality.

The initial native CPU INT4 package downloads about 5.85 GB. Loading also needs
substantial RAM for embeddings and sessions. CPU generation can be slow. Input is
limited to 300 speech characters and 500 description characters (256 tokens each),
and 240 speech frames. If the model reaches the frame limit without EOS, the
request fails and partial speech is discarded. Cancellation is checked between
graph calls; a running graph finishes before cancellation takes effect. Qwen loading also checks cancellation during 64 KiB checksum reads and between
graph/embedding loads. A single graph load or embedding read still finishes before
the next check. Studio uses this cancellable loader; ordinary load remains available.

## CLI and library

```sh
python3 scripts/download_qwen_design.py
cargo run --release -p local-tts-cli -- voice-design voices/qwen-voice-design target/design.wav en 'A calm adult male voice.' 'Hello.'
```

Rust uses `local_tts_core::qwen_design::QwenDesign::load` and
`synthesize_design(text, language, description, cancel)`.
C/C++ loads backend `TTS_QWEN_DESIGN` (5) through `tts_engine_load`, then calls
`tts_synthesize_design`. Returned PCM ownership and cancellation match the existing
API. Ordinary `tts_synthesize` on this backend returns an explicit error.
All inference stays in process; model loading performs no network calls or Python
execution. Downloads and loads verify pinned hashes.

## Provenance and redistribution

The official Qwen3-TTS VoiceDesign model and WaveKat ONNX export declare Apache-2.0.
Pinned model/export revisions and hashes are in `assets/qwen-design-assets.json`.
Adapted native code retains WaveKat attribution, the full Apache license and
`UPSTREAM.json` under `crates/tts-core/src/qwen_design`. Release notice collection
includes both attribution and license text. Keep these notices when redistributing;
model/data provenance and output quality still require project-specific evaluation.

Sources:
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- https://huggingface.co/wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX
- https://github.com/wavekat/wavekat-tts

## Reuse a designed voice

Save a generated WAV. Switch to **Multilingual / voice cloning — Chatterbox**,
import that WAV as the reference, trim it if needed, and save a portable voice
profile. This captures an audio reference rather than a Qwen random seed: cloning
may change the voice and delivery, and similarity must be assessed by listening.
Fine-tuning remains the main workflow for training your own recorded voice.

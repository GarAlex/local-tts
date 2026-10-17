# Third-party sources and assets

| Component | Source / pinned revision | License / retained record |
|---|---|---|
| VITS architecture modules | jaywalnut310/vits `2e561ba58618d021b5b8323d3765880f7e0ecfdb` | MIT, `training/vendor/vits/LICENSE` |
| VITS modifications | Relative embedding dynamic indexing; portable maximum-path alignment; our trainer/loader/exporter | Modified here; upstream MIT notice retained |
| Kokoro ONNX conversion and voice vectors | onnx-community/Kokoro-82M-v1.0-ONNX `1939ad2a8e416c0acfeecc08a694d14ef25f2231` | Apache-2.0, downloaded model card and license |
| CMU pronunciation dictionary | cmusphinx/cmudict `74790861f652b15e4ac49015a90074ad62a27690` | BSD terms, `assets/licenses/CMUDICT.txt` and downloaded license |
| Kokoro phoneme inventory reference | hexgrad/misaki `EN_PHONES.md` | Apache-2.0 upstream; our ARPAbet mapping is independently implemented |
| Noto Sans | google/fonts revision recorded in `assets/fonts/SOURCE.txt` | SIL Open Font License 1.1, `assets/fonts/OFL.txt` |
| Vosk Russian 0.7 multi | alphacep publisher archive; Hugging Face card `0960eb904b9035a9fdbba2882933a4f5a84c525f` | Card declares Apache-2.0; archive/member SHA in `assets/russian-assets.json`; separate dictionary/data terms not supplied |
| Adapted Russian G2P rules | alphacep/vosk-tts `b492223c8747a24499e4c1ae905800e85ab887b4` | Apache-2.0, `assets/VOSK-LICENSE.txt`; Rust port, explicit-stress/error behavior changed here |
| Piper Ukrainian text voice | rhasspy/piper-voices `c10ece1aade47bb51c153c893d14e5bf8e5b7117` | Repository MIT declaration, supplied voice card retained; dataset card discrepancy documented in `CYRILLIC.md` |
| Ukrainian dataset records | egorsmkv/ukrainian-tts-datasets `953d9c5afe57fad5656124be5049a453a5450a04` | Apache-2.0 README/license retained; card's CC0/OHF link not verified |
| Native inference | ONNX Runtime via ort rc.10 | ONNX Runtime MIT; ort MIT/Apache-2.0 |
| Desktop app | eframe/egui, rfd, rodio (codec features disabled) | MIT/Apache-2.0 families; versions in Cargo.lock |
| Native pitch/formant processing | Signalsmith-Audio/signalsmith-stretch `a670068d9aeb64913331d5cc29337b19a457a7df` (1.3.2) | MIT, `crates/tts-core/native/vendor/signalsmith-stretch/LICENSE.txt` |
| Native spectral/FFT implementation | Signalsmith-Audio/linear `de55e6a50ffcf6f8f43f649692d94691c7025151` | MIT, `crates/tts-core/native/vendor/linear/LICENSE.txt` |

Signalsmith source/notice SHA-256 hashes are recorded in
`crates/tts-core/native/vendor/UPSTREAM.json`; upstream headers are unmodified.
Only the required header code and MIT notices are vendored, not upstream example,
test or documentation programs. Our C++ bridge adds duration-preserving padding,
cropping, deterministic initialization and exception containment. The `cc` build
helper is pinned by Cargo.lock. No optional FFT accelerator library is enabled.
Both upstream notices are included in the development macOS app bundle; library
consumers must retain them when distributing the compiled implementation.

Downloaded model assets live outside source control in `voices/kokoro-en`, `voices/vosk-ru` and `voices/piper-uk`. Cyrillic pins and license/provenance findings are documented in `CYRILLIC.md`. `assets/english-assets.json` pins URLs and expected SHA-256 checksums; the downloader verifies them before replacing files. These checksums record the reviewed artifacts, not an independent authenticity signature. Downloading requires Internet; subsequent synthesis is offline.

`Cargo.lock` pins Rust dependencies. `training/requirements-lock.txt` records the installed training dependencies on this Mac. CUDA deployment needs a suitable PyTorch build and platform-specific verification; no CUDA run has been performed here.

This document records selected components. A release still needs the complete transitive dependency notice bundle and a platform-specific dependency audit; no blanket legal certification is implied.

The native Cyrillic number expander and saved-stress dictionary editor are original
project code. Saved corrections are user-authored local data; no external stress
dictionary or additional model is included. These features add no dependencies.

Kokoro FP32, Q8 (`model_quantized.onnx`) and Q4 (`model_q4.onnx`) use the same
pinned ONNX repository revision and retained Apache-2.0 model card/license.
`assets/kokoro-variants.json` pins individual model hashes. Shared tokenizer,
`af_heart` style and BSD CMU dictionary records remain unchanged. No quantization
library or training dependency is added to deployed applications. See
[Kokoro variants](KOKORO_VARIANTS.md) for mixed precision and testing limits.

Public English training base: `csukuangfj/vits-ljs` revision
`7ac337c834f318e45a34037cb3371cc3929187ff`, with weight/config/inventory hashes in
`assets/vits-ljs-assets.json`. Mirror declares Apache-2.0; originating earlier
mirror lacks a separate license record. LJSpeech publisher declares public-domain
audio/text in the USA. Upstream code MIT and dictionary BSD are separate records.
Imported base exports retain these records and the stated provenance gap; this
candidate is not a completed commercial clearance audit. See [public base](PUBLIC_BASE.md).

Russian training architecture: alphacep/vosk-tts
`b492223c8747a24499e4c1ae905800e85ab887b4`, Apache-2.0, with upstream
MB-iSTFT-VITS2/PQMF MIT and STFT BSD notices retained under
`training/vendor/vosk/`. Source hashes and modifications are recorded in
`UPSTREAM.json`; exports retain the base notices. This profile adds no training
package or native runtime dependency. Public generator/source pins and unresolved
dictionary/data/consent lineage are described in [Russian base](PUBLIC_RU_BASE.md).

The macOS release recipe now collects target Cargo notices, pinned upstream supplements, embedded font terms, ONNX Runtime and Rust standard-library records. Per-product enabled dependency trees correct the earlier four-gap metadata count: the SDK’s 134-package inventory has no missing texts, and Studio’s former gap, dispatch 0.2.0’s missing MIT text, is closed by the macOS main-queue adapter below. Declarations are retained without inventing copyrights. See [release process](RELEASE.md) and generated `licenses/inventory.json`.

Chatterbox multilingual ONNX assets and the reference voice are pinned in
`assets/chatterbox-assets.json`; the publisher declares MIT and the upstream MIT
license is retained with downloaded assets. Tokenizers 0.22.2 uses Apache-2.0;
HTTP/download and onig features are disabled. New permissive Cargo expressions
were checked against packaged license texts. No new missing notice texts were
found. See [French provenance and limits](FRENCH.md).

## Qwen3 descriptive voice design

The optional pinned Qwen3-TTS VoiceDesign INT4 export declares Apache-2.0;
`assets/qwen-design-assets.json` records the official model and ONNX export revisions.
Native inference adapts WaveKat Apache-2.0 code at
`ad3c7d0f927370ab23e1afad176c3cccff51467a`; attribution, full license and adaptation
record are retained in `crates/tts-core/src/qwen_design`. The release notice collector
copies the attribution and license. Models are downloaded separately, not bundled.
See VOICE_DESIGN.md for usage and limits. These declarations do not close upstream
training-data provenance gaps or establish output rights for every use.

## macOS main-queue adapter

The root Cargo patch replaces dispatch 0.2.0 with the original MIT adapter in
crates/dispatch-compat, implementing only Queue::main().exec_sync for the pinned
objc2-foundation 0.2.2 GUI dependency. No dispatch 0.2.0 source is copied. The
adapter uses dispatch2 0.3.1; its upstream license text is already pinned in the
release-notices inventory. The adapter README is copied alongside PROJECT-MIT.txt.
This removes the missing dispatch notice from the enabled Studio dependency tree;
historical release candidates still retain their original gap records.

## Compact English candidates

Inflect Nano v2 and Kitten Nano INT8 model declarations are Apache-2.0, pinned in
assets/compact-english-assets.json. Native host inference is original project
code; vocabulary IDs follow upstream data inventories. Keith Ito's MIT vocabulary
notice and Kitten's Apache license are retained in assets/compact-notices and
included in release notices. CMU dictionary assets retain their BSD license.
Models retain their own downloaded licenses/attribution. See COMPACT_ENGLISH.md;
this native frontend does not reproduce the publishers' eSpeak-ng pipeline.

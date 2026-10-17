# PromoShot integration plan

Goal: offline narration in PromoShot — the sandboxed Mac App Store app and the
headless `promoshot-mcp`/`promo` pair — using this library, with voices the
person downloads in the app.

## Constraints found

- PromoShot's Mac app is sandboxed (`PromoShotMac.entitlements`) and sold through
  the Mac App Store. It cannot spawn `tts-speak` or Python; synthesis must be
  linked in, and model downloads must be native code writing into the app
  container's Application Support.
- PromoShot already links Rust as one static library (`promo-ffi`, built by
  `macapp/Scripts/build-core.sh`, called from Swift via `@_silgen_name`). It has
  no ONNX Runtime, so linking `local-tts-core` adds ORT once, with no version clash.
  Expected cost: roughly 25–30 MB of binary (ORT plus tokenizers), arm64 only today.
- Narration flows through `promo-speech`: `Synth::synthesize(provider, voice,
  text) -> bytes`, receipts keyed by `sha256(provider|voice|text)`, files named
  `narration-{id}.mp3`, providers `openai | elevenlabs | google`, all keyed.
- Downloads here are Python-only (`scripts/download_*.py`); `tts-core` has
  hash verification (`integrity.rs`) but no downloader.

## Where the voices come from

All pins (URL at a fixed commit, SHA-256 per file) are in `assets/*.json`.

| Package | Source | Size | Weights license | Ship? |
|---|---|---|---|---|
| kokoro-en-q8 | Hugging Face `onnx-community/Kokoro-82M-v1.0-ONNX` + CMU dict (GitHub) | 92 MB | Apache-2.0 / BSD | **Default English** |
| kitten-nano-en | HF `KittenML/kitten-tts-nano-0.8-int8` | 33 MB | Apache-2.0 | Yes — 8 small English voices |
| inflect-nano-en | HF `owensong/Inflect-Nano-v2-ONNX` | 19 MB | Apache-2.0 | Optional |
| chatterbox-multilingual | HF `onnx-community/chatterbox-multilingual-ONNX` | 1.4 GB | MIT | Optional — FR/DE/ES/IT/PT/PL/NL + cloning; ~20–30 s per line |
| kokoro-en fp32 / q4 | HF (same repo) | 314 / 295 MB | Apache-2.0 | No — Q8 is the size/quality pick |
| qwen-voice-design | HF `wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX` | 5.4 GB | Apache-2.0 | No — too large for an in-app download |
| vosk-ru | alphacephei.com zip | 236 MB | Apache-2.0 (card) | Hold — dictionary/data terms unverified |
| piper-uk | HF `rhasspy/piper-voices` | 73 MB | MIT repo, CC0 claim | Hold — voice-card claim unverified |

Fetching from Hugging Face at pinned commits works, but a mirror under PromoShot's
control (GitHub Releases, R2/S3) is more reliable for a paid app: same paths,
same hashes, HF as fallback. Model weights are data, not executable code, so
downloading them is consistent with App Store guideline 2.5.2.

## Plan

### 1. In this repo (local-tts)

1. **Catalog + downloader in `tts-core`** behind a `download` feature
   (`ureq`, already in PromoShot's tree): `catalog()` lists packages from the
   pinned `assets/*.json` (id, files, URLs, sizes, license, languages); `install(id,
   root, progress, cancel)` downloads to a temp folder, verifies each SHA-256, and
   renames atomically; `installed(root)`, `remove(id, root)`. Mirror URL first, HF
   second. Studio and the Python scripts can then share it.
2. **More Kokoro voices.** The loader is fixed to `af_heart`. Kokoro ships ~50
   voice style files in the same HF repo; pin a curated set (US/UK, female/male)
   and accept them as speakers. This is the biggest quality-per-MB win.
3. **Engine cache API** for hosts: load once per package, reuse; synthesis stays
   blocking on a worker thread with the existing cancel flag.
4. **Text frontend for marketing copy.** Already done: `--lexicon` / `add_lexicon`.
   Still needed: an opt-in "display text" pass for English — `&` → and, `…` → `.`,
   `/` → a pause, letters+digits (`MP4`, `M2`) spelled out — kept outside the
   versioned text-v1 contract so trained voices are unaffected.
5. **Release target**: arm64 + x86_64 (or arm64-only, matching PromoShot), min
   macOS 15 verified on a clean machine.

### 2. In PromoShot

1. **Dependency**: `promo-speech` gains a `local` feature depending on
   `local-tts-core` (git tag or path), enabled in `promo-ffi` and `promoshot-mcp`.
2. **Provider `local`**, keyless: add it to `voices.rs` (roster = installed packages
   × their speakers) and to `Synth` routing. Voice ids carry the package and its
   pin, e.g. `kokoro-en-q8@1939ad2a/af_heart`, so the receipt re-renders when a
   model changes.
3. **Audio format**: local synthesis returns WAV, so name files by provider
   (`narration-{id}.wav`), or encode AAC in Swift (`AVAssetWriter`) to keep files
   small. `placehold` stays MP3; the extension follows the real file.
4. **Lexicon per project**: `Resources/pronunciations.dict` (CMU format) loaded
   into the engine before each walk; a small editor beside the narration text.
5. **Swift**: `LocalSpeechProvider: SpeechProvider` (`keysURL` nil, `key`
   ignored) calling the existing `promo_speech_start/poll/bytes` path with
   provider `local`. A **Settings ▸ Voices** page: catalog with size and license,
   Download (progress, cancel, resume-by-restart), Delete, disk usage. Models live in
   `~/Library/Containers/com.writea.revoice/Data/Library/Application Support/PromoShot/Voices/<id>`.
   Network access uses the existing `network.client` entitlement.
6. **Headless**: `promoshot-mcp voices list|install|remove` uses the same Rust
   downloader; `promo_voices {"provider":"local"}` and `promo_speak` work with no
   key, so agents get offline narration through MCP.
7. **Notices**: add Kokoro/CMU/Kitten/Chatterbox notices to the app's
   acknowledgements; each package's license files are already in its folder.

### Order (headless first)

PromoShot's headless half (`promoshot/core`: `promo`, `promoshot-mcp`, the
ghcr image, the MCP registry entry) is public, Apache-2.0, and built by CI on
ubuntu + macOS. It cannot depend on a private repository, so this library is
published first as its own MIT repository and consumed as a pinned git
dependency (crates.io later).

1. Publish this repo: library, `tts-speak`, C ABI; CI on macOS + Linux
   (the Linux build is not yet verified); tag `v0.1.0`.
2. Catalog + downloader + Kokoro voice set (this repo).
3. `local` cargo feature in `promo-speech` + `promoshot-mcp voices install|list|remove`;
   `promo_speak`/`promo_voices` keyless for `local`; the Docker image keeps models in
   a mounted `/models` volume rather than baking them in.
4. Mac app: the same feature in `promo-ffi`, Swift provider + Settings ▸ Voices.
5. Display-text pass, AAC encoding, mirror hosting, notices.

## Usable today without the app

`tts-speak` (installed at `~/.cargo/bin/tts-speak`) renders a WAV that
an agent drops into a project's `Resources/` as an ordinary audio resource:

```sh
tts-speak --model voices/kokoro-en-q8 --lexicon brand.dict --json --output "Demo.promo/Resources/vo-1.wav" --text 'Meet PromoShot.'
```

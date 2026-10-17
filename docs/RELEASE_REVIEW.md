# Repository and release review — 2026-10-03

The project is a Rust inference/dataset library, C ABI and egui Studio, with an
optional Python/PyTorch training/export toolkit. It includes native English,
Russian and Ukrainian backends, versioned text preparation and imported VITS
architectures. This review covers the architecture, source entry points, package
integrity, dataset/training flow, ABI, user-data handling, dependencies, existing
verification evidence and release readiness. It is a maintainer review followed
by implementation, not an independent security or legal certification.

## Verdict

The inference and training machinery has substantial engineering coverage, but
that coverage does not establish trained voice quality or public distribution
readiness. A relocatable candidate packaging process now replaces the previous
workspace-only distribution path. Studio release remains gated by one missing upstream notice text, while the
independent SDK notice inventory is complete for its selected dependency tree.
Signing, platform/GUI/hardware checks and chosen model provenance remain.
There are no model weights or Python wheels in the new archives.

## Findings and disposition

| Priority | Location | Finding and action |
|---|---|---|
| High, fixed | `crates/tts-studio/src/main.rs`, `runtime.rs` | Studio embedded and unconditionally opened its compile-time checkout. Release startup now resolves signed bundle resources, installs verified tools into a versioned user workspace and reports actionable errors. Debug source development remains available. |
| High, fixed | `training/launch.py`, `checkpoint_tools.py`, `export_and_verify.py`, `runtime.py` | Training/export looked for `target/debug` or invoked Cargo. A shared resolver now prefers bundled native CLI and permits Cargo fallback only in a source checkout. |
| High, corrected; Studio gap open | `scripts/release_notices.py`, `release/license-policy.json` | The previous metadata inventory included inactive optional packages and overreported release blockers. Enabled per-product trees contain 78 SDK packages with complete notices and 209 Studio packages with one missing MIT text (dispatch 0.2.0). SDK-only packaging/submission is independent; combined submission remains gated. No license attribution was fabricated. |
| High, open | `docs/PUBLIC_BASE.md`, `PUBLIC_RU_BASE.md`, `PUBLIC_UK_BASE.md`, `CYRILLIC.md` | English originating weights and RU/UK dictionary/data/consent lineage have documented gaps. Training one's own voice does not erase base/data terms. Do not infer clearance from successful import or hashes. New binary archives contain no voices. |
| Medium, implemented | `scripts/release.py`, `release_notices.py` | The development bundle had partial notices and no release manifest, relocation test or public signing workflow. New process records source/dependency hashes, collects notices, audits linkage, signs, stages atomically and creates app/SDK/source archives. |
| Medium, open | `crates/tts-studio/src/recording.rs`, `playback.rs`; release matrix | Microphone/device behavior, fresh-machine GUI, Gatekeeper with real Developer ID, Intel, minimum macOS, Linux and Windows need their own validation. Headless checks cannot substitute for these. |
| Medium, open | `training/requirements-lock.txt`, `setup.py` | Training environment is a macOS-tested pip freeze, not a portable wheel lock with hashes. Native archives exclude Python dependencies. Platform-specific CUDA/Windows/Linux training environments require separate locks and verification. |
| Medium, open | `docs/PHASES.md`, `docs/LANGUAGE.md` | Synthetic training/parity and finite audio establish implementation contracts, not speaker similarity/intelligibility. Context rules cover selected phrases, not general stress/case prediction. Real consenting-speaker adaptation and held-out native-speaker listening remain required. |
| Keep | `tts-core`, `tts-ffi`, `include/local_tts.h` | Inference remains in-process and reusable, with explicit ownership, bounded text/audio, cancellation between model calls and versioned frontend metadata. Package SDK exercises the public header with a separate C++ consumer. |
| Keep | `dataset.rs`, `integrity.rs`, `evaluation.rs`, `training/checkpoint_tools.py` | Relative-path validation, bounded package hashing, restricted checkpoint loading, hash-bound evaluated exports and version-preserving resume are useful contracts. Hashes detect corruption and do not authenticate a publisher; keep trusted inputs and immutable packages during loading. |

## Next release work

1. Obtain the missing dispatch 0.2.0 notice (or replace it) for Studio; retain
   separate SDK/App dependency evidence and review native/compiler attributions.
2. Test the candidate app and SDK on a clean supported Mac, including first launch,
   acquisition, fine-tuning, playback, microphone, cancellation and data retention.
3. Establish real speech quality and per-model provenance for any model-bearing
   distribution. Select hardware and a measured performance baseline.
4. Exercise Developer ID signing/notarization with operator credentials, then
   publish the verified artifacts/checksums through an explicitly chosen host.
5. Add native Linux/Windows recipes and training wheel locks; run the same consumer
   and relocation checks there before adding those platforms to the matrix.

## Evidence and limits

The implementation runs regular Rust tests, strict Clippy, release builds and
relocated app-runtime/CLI/C++ checks. Optional release flags additionally test
real Kokoro inference and installed-kit scratch/fine-tuning/export/native parity.
Each successful candidate embeds its exact `smoke-tests.json`, dependency audit,
notice inventory and source manifest. Earlier full EN/RU/UK/model/effects tests
remain recorded in `target/verification*.json` and implementation audit history.
No code signing account, upload, microphone permission or real speaker recording
was used. This review did not exhaustively fuzz every parser, audit external
training data, validate all transitive upstream source, or measure production
voice quality/performance.

## Follow-up: accurate product dependency scope

The previous four-gap count came from the broad Cargo metadata resolution. The
release process now joins an explicit enabled `cargo tree --edges normal,build`
for each product's package roots to exact metadata identities. Tests reject
unknown/ambiguous identities and cover omission of inactive optional packages.
The binary SDK can be built and submitted independently with `--package sdk`.
It includes no Studio executable and passes the same relocation, C++/CMake,
inference and optional installed-kit fine-tuning/export checks. Each product keeps
its tree and inventory. Separate Cargo builds keep each binary's feature scope
aligned with its tree. Source archives include project sources/Cargo manifests,
not the unbuilt external crates' implementations. This corrects the earlier
release blocker classification; it does not change any upstream license terms.

### dispatch gap resolution — 2026-10-06

Current sources replace the old dispatch crate with an original narrow adapter
backed by dispatch2. Synchronous callback borrowing, return values and panic
propagation passed native tests. The enabled Studio tree has 252 external packages
with zero unresolved notice texts. This supersedes the dispatch blocker above for
new builds only; old candidates and their manifests remain unchanged. Commercial
model/data provenance, signing, notarization and manual platform checks remain.

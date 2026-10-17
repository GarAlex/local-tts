# Release packaging

The native macOS recipe builds a relocatable Studio app, C/C++ shared-library SDK
and Rust/source archive. It publishes nothing automatically. Current validation
is on Apple Silicon; Intel macOS must run the same recipe and pass its own tests.
Linux and Windows packages are not yet supported. The app declares macOS 14.0+;
the minimum OS and Intel hardware still need separate-machine verification.

## Build a candidate

Requirements: macOS, Xcode command-line tools, Rust/Cargo with edition-2024 support,
Python 3.9+ for packaging and available Cargo dependencies. Rust 1.96.0 is the
current tested toolchain; the exact compiler and Cargo.lock hash are recorded.
Training verification additionally needs an existing Python 3.12 environment with
`training/requirements-lock.txt`. Inference does not need Python.

```sh
python3 scripts/release.py
# Independent library/CLI SDK (no Studio app):
python3 scripts/release.py --package sdk
# Also exercise a real local voice and the installed training/fine-tuning/export kit:
python3 scripts/release.py --voice voices/kokoro-en --training-python .venv/bin/python
```

Output is `dist/local-tts-<version>-<target>-<source-hash>/` (with `-sdk` for
SDK-only builds). SDK-only mode builds the CLI/library roots, omits the app ZIP
and GUI dependencies, and retains training/export tools and the source archive.
The manifest records exactly which products it contains. The SDK has its own
notice inventory and signing/notarization path.
Combined releases also compile each product separately with exactly its
inventoried roots, preventing Cargo feature unification with GUI dependencies
from changing the SDK's dependency scope.

Combined output contains:

- Studio ZIP and expanded `.app`, with the CLI, training scripts, acquisition tools,
  asset pins, source licenses, fonts and generated dependency notices.
- SDK tarball and expanded directory: shared library, CLI, C header, C++ examples,
  CMake consumer, training kit, documentation and notices.
- Source tarball: allowlisted project sources and vendored code, Cargo.lock and
  license/provenance records. No `.venv`, recordings, datasets, checkpoints, voices,
  build cache or local credentials are copied.
- `release-manifest.json`: version, source hashes, toolchain, target, signing state,
  notice gaps and per-file SHA-256. `SHA256SUMS` covers the archives and reports.
- `smoke-tests.json` and `binary-dependencies.json` retain validation evidence.

When CMake is installed, the relocation suite also configures, builds, installs
and relocates the supplied CMake consumer; the direct C++ compiler check always
runs. With `--voice`, both available consumer paths synthesize through the SDK
shared library.

The build runs packaging failure-path tests, format, workspace tests, strict Clippy and a locked release build.
It checks binary architecture, rejects external non-system dylib dependencies,
sets the SDK install name to `@rpath/liblocal_tts.dylib`, signs locally, verifies
signatures and runs relocated CLI/C++/runtime tests. Final archives are extracted and checked against the staged files/source hashes, then their app signature, runtime bootstrap and CLI are exercised. A source change during the
build aborts packaging. Existing output directories are never overwritten;
failed stages are removed. To rerun unchanged sources with additional checks, select
a fresh output parent with `--output`; signing/notarization uses a separate output
parent so it does not replace the local candidate. Archive timestamps/signatures are not byte-reproducible;
source/dependency selection is pinned and recorded.

To recheck an existing candidate after copying it:

```sh
python3 scripts/verify_release.py /path/to/candidate --extract
```

The verifier is read-only with respect to the candidate. It rejects missing,
extra or changed files and unsafe archive paths/links, then checks extracted
content and executable behavior in a temporary folder. A checksum manifest is
not proof of publisher identity. `public_release_ready` remains false even for a
notarized candidate until the operator records the manual release checks.

Candidates use ad-hoc signing and are for local testing. They are not notarized
public downloads. Hashes detect corruption; they are not publisher signatures.

The multilingual Chatterbox checks are optional because they need the separate
1.56 GB model. They exercise relocated French CLI and French/German/Spanish C++
shared-library synthesis:

```sh
python3 scripts/release.py --voice voices/kokoro-en \
  --multilingual-voice voices/chatterbox-multilingual --training-python .venv/bin/python
```

## Studio installation and data

Move the app out of its ZIP, for example into Applications. On first launch its
verified tools are copied into:

`~/Library/Application Support/Local TTS/runtime-<version>-<payload-hash>/`

Set `LOCAL_TTS_HOME` before launching to choose a different data parent. The app
never needs its original checkout or writes training data inside the application
bundle. The per-release workspace holds downloaded voices, `.venv` and default
checkpoints. Existing datasets/checkpoints/voices can be selected from their old
locations after an upgrade. Workspaces are not silently migrated or deleted.
Reopening the same release preserves user data; changing installed tool files
causes an explicit integrity error. Reinstalling the `.app` alone does not erase
user data. To recover changed tools, move the affected runtime folder aside,
relaunch, then select the old data paths. Keep old training environments for exact
resume compatibility.

Use `Contents/MacOS/tts-studio --check-runtime` for a headless installation check.
This runs the same resource resolution as the GUI but does not open audio devices.

Models and Python wheels are not bundled. Point Studio at an existing voice for
fully offline synthesis, or use acquisition tools with Python and Internet access.
The training setup installs pinned Python requirements into the user workspace;
its wheel terms, platform compatibility and training hardware remain separate
from the native inference package. SDK training tools automatically use `bin/local-tts`;
`LOCAL_TTS_CLI` can select an explicit native executable.

## C/C++ consumption

```sh
cmake -S /path/to/local-tts-sdk -B /tmp/tts-consumer
cmake --build /tmp/tts-consumer
/tmp/tts-consumer/tts-speak /path/to/kokoro-en /tmp/speech.wav
cmake --install /tmp/tts-consumer --prefix /path/to/install
```

Installed consumers use `@loader_path/../lib`. Ship `lib/`, `licenses/` and your
chosen voice's notices with the consumer. ABI 1 uses the ownership rules in
`include/local_tts.h`. This recipe distributes the tested shared library; static
linking is still a source-build option with platform link requirements.

## Notices and public-release gates

`release/license-policy.json` is an explicit allowlist of reviewed Cargo license
expressions. A new expression or unexpected missing notice stops packaging.
`licenses/inventory.json` records the enabled normal/build dependency tree of
each product: Studio plus CLI for the app, CLI plus FFI for the SDK. The exact
`cargo tree` output is retained beside each inventory. Development dependencies
and inactive optional packages are excluded. Ambiguous package identities or an
empty/unparseable tree fail collection. On the tested Apple Silicon build this
is 253 third-party packages for Studio and 134 for the SDK after the multilingual
tokenizer addition (the earlier pre-French candidate had 209 and 78). Registry notice files,
font-specific terms, versioned upstream supplements, ONNX Runtime 1.22 notices,
Rust standard-library attributions and native DSP notices are included. Upstream
supplements have URLs/revisions/SHA-256 in `assets/release-notices/sources.json`.
Do not replace a missing copyright notice with invented author/year information.

The earlier report incorrectly treated all four missing texts in Cargo metadata
as binary-release blockers. block 0.1.6, malloc_buf 0.0.6 and hexf-parse 0.2.1 are
not enabled in the current macOS release trees. dispatch 0.2.0 is enabled only in
Studio's tree; its missing MIT text is also recorded in the
[upstream issue](https://github.com/SSheldon/rust-dispatch/issues/18).
Its published Cargo license/author declaration remains in the app inventory.
The SDK notice inventory has no unresolved texts. SDK-only notarization is thus
independent of the Studio gap; combined notarization still stops on that gap.
Obtain the missing upstream record or replace that dependency before distributing
Studio. Known metadata gaps stay in the policy so enabling one on another target
cannot silently bypass review. Notices for native dependencies are collected
conservatively; inventory completeness is not clearance for arbitrary models/data.

Before a public release also complete: clean-machine GUI/playback/microphone
checks, minimum-OS checks, native-speaker listening, and model-specific provenance
review for every model you choose to distribute. Current archives include no
weights, recordings or derived voices. See RELEASE_REVIEW.md for remaining gaps.

## Signing and notarization

A Developer ID certificate and existing `notarytool` keychain profile are supplied
by the release operator. No private key or password belongs in this repository.
After the selected product’s release checks are closed, SDK-only submission is:

```sh
python3 scripts/release.py --package sdk --output dist/notarized-sdk \
  --identity 'Developer ID Application: Your Name (TEAMID)' \
  --notary-profile local-tts-notary \
  --voice /path/to/kokoro-en --training-python /path/to/training-python
```

Combined Studio/SDK submission additionally requires resolving the Studio notice gap:

```sh
python3 scripts/release.py --output dist/notarized \
  --identity 'Developer ID Application: Your Name (TEAMID)' \
  --notary-profile local-tts-notary \
  --voice /path/to/kokoro-en --training-python /path/to/training-python
```

The recipe signs the selected CLI/library binaries and app (when included), enables hardened runtime for
Developer ID builds, includes microphone entitlement/usage text, verifies the
selected binaries, submits them to Apple and requires Accepted status. When an
app is included, it staples/validates its ticket and checks Gatekeeper before
creating final archives.
The ZIP is recreated after stapling. Standalone SDK binaries rely on Apple's
online notarization ticket lookup; they cannot receive an app staple themselves.
Supplying `--notary-profile` is the explicit network submission action. No signing
identity or notarization credentials were used during local implementation.

Reference workflow: [Apple notarization documentation](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
and [customizing notarization](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).

## Release sequence and rollback

1. Freeze source/version and run required model/language/training regression checks
   with `scripts/verify.py`; preserve the reports with the candidate.
2. Build the candidate above; inspect notice inventory and dependency audit.
3. Verify archive checksums after copying to another location, extract, run the SDK
   consumer and Studio, and test a real voice. Retain clean-machine results.
4. Resolve all public-release gates, sign/notarize, recheck final archive hashes,
   then upload only the reviewed archives and reports to the chosen release host.
5. Keep prior archives/manifests. Rollback means reinstalling the prior app/SDK and
   selecting its retained data/environment. Do not overwrite user checkpoints.

There is no hosted Git repository/CI configuration in this workspace. The script
is suitable for a native macOS CI worker with the stated tools, and makes no
unsupported cross-compilation or automatic publishing claim.

## Voice-design package checks

The optional Qwen check uses the separately downloaded, pinned 5.85 GB package:

```sh
python3 scripts/release.py --design-voice voices/qwen-voice-design
```

It exercises relocated English CLI and Russian C++ descriptive synthesis, PCM
validation, pre-request cancellation and invalid/unsupported requests. Models
remain outside the release. No GUI window is launched by these automated checks.

Candidate `local-tts-0.1.0-aarch64-apple-darwin-a7d56ab7ee1a` (2026-10-06)
passed these checks, archive extraction/hashes, architecture/linkage and ad-hoc
signature verification. It includes descriptive design, reference cloning and
portable profiles. Public distribution still requires the recorded dispatch notice,
Developer ID/notarization and manual platform/quality checks.

Latest candidate: `local-tts-0.1.0-aarch64-apple-darwin-91200d53ff05` supersedes
the preceding candidate's dispatch gap. Both enabled notice inventories are
complete. Packaged GUI launch/normal closure and the full voice-design packaging
recipe passed. Signing is still ad-hoc; public-release manual gates remain.

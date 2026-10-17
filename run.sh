#!/usr/bin/env bash
# Launch the workspace Studio without creating another running copy.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if command -v pgrep >/dev/null 2>&1 && pgrep -x tts-studio >/dev/null; then
    echo "Local TTS Studio is already running. Close it before rebuilding."
    exit 0
fi

if [[ "$(uname -s)" == Darwin ]]; then
    python3 scripts/bundle_macos.py
    open "${project_dir}/target/Local TTS Studio.app"
else
    exec cargo run --locked -p tts-studio
fi

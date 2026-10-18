#!/bin/sh
# Rebuild assets/nemo-en-tn/ from NeMo's English TN grammars, in a pinned
# x86-64 container (pynini publishes no Linux ARM wheel). Needs Docker.
#   training/nemo/build_grammars.sh
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
out="$root/assets/nemo-en-tn"
mkdir -p "$out"
image=${NEMO_IMAGE:-python:3.11-slim}
docker run --rm --platform linux/amd64 \
  -v "$out:/out" -v "$root/training/nemo:/build:ro" "$image" sh -c '
    set -e
    if ! python -c "import nemo_text_processing" 2>/dev/null; then
      apt-get update -qq && apt-get install -y -qq gcc g++ >/dev/null
      pip install --progress-bar off nemo_text_processing==1.2.0 pynini==2.1.6.post1
    fi
    python /build/export_grammars.py'

#!/usr/bin/env bash
# Stage the two model directories the Dockerfile bakes in.
#   models/original  a snapshot of google/embeddinggemma-2 (1.5 GB, all towers; loaded text only)
#   models/repack    repack/repack_text_w8a8_emb4.py run on that snapshot (203 MiB)
# Set REPACK to an existing build directory to copy it instead of rebuilding.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
rm -rf models/repack models/original
snap=$(python3 -c "from huggingface_hub import snapshot_download as s; print(s('google/embeddinggemma-2'))")
cp -rlL "$snap" models/original  # hard links into the HF cache, no second copy
if [ -n "${REPACK:-}" ]; then
  cp -r "$REPACK" models/repack
else
  python3 repack/repack_text_w8a8_emb4.py models/original models/repack
fi
du -sh models/*

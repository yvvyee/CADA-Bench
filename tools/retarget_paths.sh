#!/usr/bin/env bash
# Retarget the hard-coded data root in all CADA-Bench scripts.
#
# The experiment scripts were written with an absolute data root of
# /home/yvvyee/data, which holds the base models and the LLaVA-CC3M corpus.
# Run this once to point every script at your own data root.
#
# Usage:
#   bash tools/retarget_paths.sh /path/to/your/data_root
#
# After retargeting, the data root must contain:
#   Qwen2-VL-7B-Instruct/         (HuggingFace snapshot)
#   Qwen2.5-VL-7B-Instruct/       (HuggingFace snapshot, for cross-backbone runs)
#   LLaVA-CC3M-Pretrain-595K/     (chat.json + images/, see data/DATA.md)
# plus the benchmark JSON artifacts from data/ (copy or symlink them in).

set -euo pipefail
OLD="/home/yvvyee/data"
NEW="${1:?usage: retarget_paths.sh /path/to/data_root}"
NEW="${NEW%/}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
count=0
# Rewrite the hard-coded prefix in both the scripts (src/**/*.py) and the
# committed data artifacts (data/*.json), which embed absolute image paths.
while IFS= read -r -d '' f; do
  if grep -q "$OLD" "$f"; then
    sed -i "s#${OLD}#${NEW}#g" "$f"
    count=$((count+1))
    echo "retargeted: ${f#$ROOT/}"
  fi
done < <( { find "$ROOT/src" -name '*.py' -print0; find "$ROOT/data" -name '*.json' -print0; } )
echo "done: $count file(s) now point at $NEW"

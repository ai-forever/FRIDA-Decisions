#!/usr/bin/env bash
# Run inside an existing vLLM container with the repository and model mounted.
#   bash examples/test_vllm_offline_container.sh --model /models/joint-engine
#   bash examples/test_vllm_offline_container.sh --model /models/joint-engine \
#       --task embed --texts 'What is semantic search?' 'Another query'
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'HELP'
Usage: test_vllm_offline_container.sh --model PATH [--task embed|token_embed|plugin] [--texts TEXT ...]

Install the local plugin into the container's existing Python environment,
verify plugin discovery, then test all three offline variants in separate processes.
Use --task to select a single variant. No server or uv needed.
Arguments are forwarded to examples/vllm_offline.py.

Requires a materialized Sentence Transformers model from joint-st-convert.
PYTHON selects the container's Python executable (default: python3).
HELP
}

if (($# == 0)); then usage >&2; exit 2; fi
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then usage; exit 0; fi
done

# Validate CLI arguments before installing anything or importing GPU libraries.
"$PYTHON" - "$REPO_ROOT/examples" "$@" <<'PY'
import sys
sys.path.insert(0, sys.argv.pop(1))
from _common import arguments
arguments('Offline joint vLLM inference', pooling_tasks=True)
PY

"$PYTHON" -c 'import setuptools, torch, safetensors, vllm; print("vLLM", vllm.__version__, "torch", torch.__version__)'
"$PYTHON" -m pip install --no-deps --no-build-isolation \
    -e "$REPO_ROOT/core" -e "$REPO_ROOT/vllm-integration"
export VLLM_PLUGINS="${VLLM_PLUGINS:+${VLLM_PLUGINS},}joint_embeddings"

"$PYTHON" "$REPO_ROOT/examples/vllm_container_smoke.py" check
for arg in "$@"; do
    if [[ "$arg" == --task || "$arg" == --task=* || "$arg" == --data-envelope ]]; then
        exec "$PYTHON" "$REPO_ROOT/examples/vllm_offline.py" "$@"
    fi
done
for task in embed token_embed plugin; do
    echo "Testing $task"
    "$PYTHON" "$REPO_ROOT/examples/vllm_offline.py" "$@" --task "$task"
done

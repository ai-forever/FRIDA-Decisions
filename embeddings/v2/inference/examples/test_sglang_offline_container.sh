#!/usr/bin/env bash
# Run inside an existing SGLang container with the repository and model mounted.
#   bash examples/test_sglang_offline_container.sh --model /models/joint-engine
#   bash examples/test_sglang_offline_container.sh --model /models/joint-engine \
#       --role query --texts 'What is semantic search?' 'Another query'
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'HELP'
Usage: test_sglang_offline_container.sh --model PATH [--role query|document] [--texts TEXT ...]

Install the local plugin into the container's existing Python environment,
verify plugin discovery, then run offline joint inference. No server or uv needed.
Arguments are forwarded to examples/sglang_offline.py.

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
arguments('Offline joint SGLang inference')
PY

"$PYTHON" -c 'import setuptools, torch, safetensors, sglang; print("SGLang", sglang.__version__, "torch", torch.__version__)'
"$PYTHON" -m pip install --no-deps --no-build-isolation \
    -e "$REPO_ROOT/core" -e "$REPO_ROOT/sglang-integration"
export SGLANG_EXTERNAL_MODEL_PACKAGE=inference_plugins_sglang.models
export SGLANG_EXTERNAL_MM_MODEL_ARCH=Qwen35JointEmbedding
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=inference_plugins_sglang.processors

"$PYTHON" "$REPO_ROOT/examples/sglang_container_smoke.py" check
exec "$PYTHON" "$REPO_ROOT/examples/sglang_offline.py" "$@"

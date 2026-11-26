#!/usr/bin/env bash
# Run inside an existing vLLM container with this repository mounted.
#   bash examples/test_vllm_container.sh --check-only
#   bash examples/test_vllm_container.sh --model /models/joint
# Uses preinstalled Python/vLLM/PyTorch; does not create a venv or upgrade them.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VLLM="${VLLM:-vllm}"
MODEL=""
PORT=18000
TIMEOUT=600
CHECK_ONLY=0
LOG_DIR=""
SERVER_PID=""

usage() {
    cat <<'HELP'
Usage: test_vllm_container.sh [options]
  --check-only       Install plugins; test discovery, imports, and synthetic heads
  --model PATH       Materialized joint Sentence Transformers model for GPU tests
  --port PORT        HTTP smoke-test port (default: 18000)
  --timeout SECONDS  Server startup timeout (default: 600)
  --log-dir PATH     Preserve logs/results here (default: new directory in /tmp)
  --help            Show this help
Environment: PYTHON (default: python3) and VLLM (default: vllm) must use the same environment.

Full mode tests dense-only, multivector-only and combined offline inference, starts its own localhost server, tests HTTP
text/image batches, changed-image sensitivity, concurrency and parity,
then stops only the server it started.
Requires the full Sentence Transformers export from joint-st-convert.
Use --check-only if the engine export is unavailable. This does not test GPU serving.
HELP
}

while (($#)); do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --model|--port|--timeout|--log-dir)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --model) MODEL="$2" ;;
                --port) PORT="$2" ;;
                --timeout) TIMEOUT="$2" ;;
                --log-dir) LOG_DIR="$2" ;;
            esac
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    echo "--port must be between 1 and 65535" >&2; exit 2
fi
if [[ ! "$TIMEOUT" =~ ^[0-9]+$ ]] || ((TIMEOUT < 1)); then
    echo "--timeout must be a positive integer" >&2; exit 2
fi
if (( ! CHECK_ONLY )) && [[ -z "$MODEL" ]]; then
    usage >&2; exit 2
fi

if [[ -z "$LOG_DIR" ]]; then LOG_DIR="$(mktemp -d /tmp/joint-vllm-smoke.XXXXXX)"; fi
mkdir -p -- "$LOG_DIR"
LOG_DIR="$(cd -- "$LOG_DIR" && pwd)"
echo "Logs and results: $LOG_DIR"
cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if ((status)); then
        echo "Smoke test failed. Logs: $LOG_DIR" >&2
        if [[ -f "$LOG_DIR/server.log" ]]; then tail -n 60 "$LOG_DIR/server.log" >&2; fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Avoid resolving dependencies against a different torch/CUDA build or fetching
# build requirements. setuptools and pip must already be in the container.
"$PYTHON" -c 'import setuptools, torch, safetensors, vllm; print("vLLM", vllm.__version__, "torch", torch.__version__)'
"$PYTHON" -m pip install --no-deps --no-build-isolation -e "$REPO_ROOT/core" -e "$REPO_ROOT/vllm-integration"
# An explicit allowlist must include this package's general model plugin.
export VLLM_PLUGINS="${VLLM_PLUGINS:+${VLLM_PLUGINS},}joint_embeddings"
export PYTHONPATH="$REPO_ROOT/examples${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$REPO_ROOT/examples/vllm_container_smoke.py" check | tee "$LOG_DIR/check.log"
if (( CHECK_ONLY )); then
    echo "PASS: registration and synthetic heads (no real checkpoint or GPU inference)."
    exit 0
fi
MODEL="$(cd -- "$MODEL" && pwd)"
# Separate processes release each engine's GPU memory before loading the next.
for task in embed token_embed plugin; do
    "$PYTHON" "$REPO_ROOT/examples/vllm_container_smoke.py" offline \
        --task "$task" --model "$MODEL" --result "$LOG_DIR/offline.json" \
        | tee "$LOG_DIR/offline.$task.log"
done

# Refuse to test against a preexisting server on the same port.
"$PYTHON" - "$PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
"$VLLM" serve "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --served-model-name joint-model \
    --runner pooling \
    --hf-overrides '{"architectures":["Qwen35JointEmbedding"]}' \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --max-model-len 512 --max-num-batched-tokens 512 --enforce-eager \
    >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

"$PYTHON" "$REPO_ROOT/examples/vllm_container_smoke.py" online \
    --model "$MODEL" --result "$LOG_DIR/offline.json" \
    --url "http://127.0.0.1:$PORT" --timeout "$TIMEOUT" --server-pid "$SERVER_PID" \
    | tee "$LOG_DIR/online.log"
echo "PASS: offline and online text/visual joint embeddings. Results: $LOG_DIR"

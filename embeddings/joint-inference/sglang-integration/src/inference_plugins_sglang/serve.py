"""Launch SGLang with JSON-safe /encode output for joint embedding tensors.

Use the same arguments as python -m sglang.launch_server. Supports the default
single HTTP/tokenizer worker used by the supplied smoke tests.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    # Additional tokenizer workers can re-import the upstream app without this
    # process's route wrapper. Fail explicitly rather than serve corrupted JSON.
    """Parse command-line arguments and run the requested operation."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--tokenizer-worker-num":
            if i + 1 >= len(sys.argv) or sys.argv[i + 1] != "1":
                raise SystemExit("Joint HTTP wrapper requires --tokenizer-worker-num 1")
        elif arg.startswith("--tokenizer-worker-num=") and arg.split("=", 1)[1] != "1":
            raise SystemExit("Joint HTTP wrapper requires --tokenizer-worker-num 1")
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "inference_plugins_sglang.models"
    os.environ["SGLANG_EXTERNAL_MM_MODEL_ARCH"] = "Qwen35JointEmbedding"
    os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = "inference_plugins_sglang.processors"
    from sglang.srt.entrypoints.http_server import app

    from inference_plugins_sglang.http import install_encode_serialization

    install_encode_serialization(app)
    # Delegate lifecycle, process cleanup and CLI parsing to SGLang.
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()

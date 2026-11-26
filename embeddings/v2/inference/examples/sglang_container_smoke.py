"""Helpers for test_sglang_container.sh, using the installed SGLang runtime."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    engine_contract,
    post_json,
    sglang_outputs,
)

if TYPE_CHECKING:
    from inference_plugins.contract import Contract
    from inference_plugins.types import JointEmbedding, PathLike

MAX_CONTEXT_LENGTH = 512

TEXTS = ["What is semantic search?", "Search using meaning, rather than exact words."]


def check() -> None:
    """Check runtime registration, output layout, and embedding head parity."""
    import torch
    from inference_plugins.contract import Contract
    from inference_plugins.heads import JointHeads
    from inference_plugins_sglang.models.qwen35_joint import Qwen35JointEmbedding
    from sglang.srt.layers.pooler import EmbeddingPoolerOutput
    from sglang.srt.managers.io_struct import EmbeddingReqInput
    from sglang.srt.models.registry import ModelRegistry

    cls, _ = ModelRegistry.resolve_model_cls(["Qwen35JointEmbedding"])
    assert cls is Qwen35JointEmbedding  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
    request = EmbeddingReqInput(input_ids=[[1, 2]], return_pooled_hidden_states=True)
    assert request.return_pooled_hidden_states  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
    torch.manual_seed(0)
    c = Contract()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(c.multivector_dim, c.dense_dim)
    hidden = torch.randn(2, c.dense_dim, device=device)
    heads = JointHeads(c, weight, device=device)
    dense, tokens = heads(hidden, [1, 2])
    native = EmbeddingPoolerOutput(embeddings=[tokens], pooled_hidden_states=hidden[-1:])
    items = sglang_outputs(
        {"embedding": native.embeddings[0], "pooled_hidden_state": native.pooled_hidden_states[0]},
        c,
    )
    torch.testing.assert_close(torch.tensor(items[0]["dense"]), dense.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        tokens, torch.nn.functional.normalize(hidden @ weight.to(device).T, dim=-1)
    )
    print(  # noqa: T201 - CLI output is intended for the user.
        "PASS: external model discovery, native request/output APIs, device placement and head parity"
    )


def inputs(model: PathLike) -> tuple[Path, Contract, list[list[int]], list[int]]:
    """Prepare smoke inputs and expected token lengths from the model contract."""
    root, c, tokenizer = engine_contract(model)
    ids = c.tokenize(tokenizer, TEXTS, "document")
    lengths = [sum(token not in c.skip_token_ids for token in row) for row in ids]
    if any(length == 0 for length in lengths):
        raise ValueError("Smoke texts must retain at least one multivector token")
    return root, c, ids, lengths


def validate(items: list[JointEmbedding], lengths: list[int], contract: Contract) -> None:
    """Validate embedding outputs against their reference values and contract."""
    import torch

    if not isinstance(items, list) or len(items) != len(lengths):
        raise ValueError("Expected one result per input")
    for item, length in zip(items, lengths, strict=True):
        dense, tokens = torch.tensor(item["dense"]), torch.tensor(item["multivector"])
        if dense.shape != (contract.dense_dim,) or tokens.shape != (
            length,
            contract.multivector_dim,
        ):
            raise ValueError(f"Unexpected output shapes: {dense.shape}, {tokens.shape}")
        if not torch.isfinite(dense).all() or not torch.isfinite(tokens).all():
            raise ValueError("Nonfinite embedding")
        if contract.normalize_dense:
            torch.testing.assert_close(dense.norm(), torch.tensor(1.0), atol=0.02, rtol=0.02)
        if contract.normalize_multivector:
            torch.testing.assert_close(
                tokens.norm(dim=-1), torch.ones(length), atol=0.02, rtol=0.02
            )


def offline(args: argparse.Namespace) -> None:
    """Run offline inference and save validated outputs for online comparison."""
    from sglang import Engine
    from sglang.srt.managers.io_struct import EmbeddingReqInput

    root, c, ids, lengths = inputs(args.model)
    engine = Engine(
        model_path=str(root),
        json_model_override_args='{"architectures":["Qwen35JointEmbedding"]}',
        is_embedding=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        context_length=512,
    )
    try:
        request = EmbeddingReqInput(input_ids=ids, return_pooled_hidden_states=True)
        generator = engine.tokenizer_manager.generate_request(request, None)
        try:
            response = engine.loop.run_until_complete(anext(generator))
        finally:
            engine.loop.run_until_complete(generator.aclose())
        items = sglang_outputs(response, c)
        validate(items, lengths, c)
        Path(args.result).write_text(json.dumps(items), encoding="utf-8")
        print("PASS: offline heterogeneous batch, dimensions, finite values and normalization")  # noqa: T201 - CLI output is intended for the user.
    finally:
        engine.shutdown()


def online(args: argparse.Namespace) -> None:
    """Check server readiness and compare online outputs with offline results."""
    import torch

    _, c, ids, lengths = inputs(args.model)
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            os.kill(args.server_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("SGLang server exited; inspect server.log") from exc
        try:
            with urlopen(args.url + "/health", timeout=2) as response:  # noqa: S310 - Explicit user-supplied inference server URL.
                if response.status == HTTPStatus.OK:
                    break
        except (URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("SGLang server startup timeout; inspect server.log")
        time.sleep(1)
    expected = json.loads(Path(args.result).read_text(encoding="utf-8"))

    def request(token_ids: list[list[int]], label: str) -> list[JointEmbedding]:
        raw = post_json(
            args.url + "/encode",
            {
                "input_ids": token_ids,
                "return_pooled_hidden_states": True,
            },
        )
        # Preserve the server's exact JSON before validating or transforming it.
        # This distinguishes transport/serialization failures from model output.
        path = Path(args.result).parent / f"online-{label}-response.json"
        path.write_text(json.dumps(raw, indent=2))
        try:
            return sglang_outputs(raw, c)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{exc} Raw HTTP response saved to {path}") from exc

    def compare(actual: list[JointEmbedding], reference: list[JointEmbedding]) -> None:
        """Compare dense and token embeddings against reference outputs."""
        for a, b in zip(actual, reference, strict=True):
            for field in ("dense", "multivector"):
                torch.testing.assert_close(
                    torch.tensor(a[field]), torch.tensor(b[field]), atol=0.02, rtol=0.02
                )

    batch = request(ids, "batch")
    validate(batch, lengths, c)
    compare(batch, expected)
    with ThreadPoolExecutor(max_workers=2) as executor:
        singles = list(
            executor.map(lambda pair: request([pair[1]], f"single-{pair[0]}"), enumerate(ids))
        )
    for i, single in enumerate(singles):
        validate(single, [lengths[i]], c)
        compare(single, [expected[i]])
    print("PASS: native HTTP serialization, batching, concurrency and offline/online parity")  # noqa: T201 - CLI output is intended for the user.


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "offline", "online"])
    parser.add_argument("--model")
    parser.add_argument("--result")
    parser.add_argument("--url")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "inference_plugins_sglang.models"
    os.environ["SGLANG_EXTERNAL_MM_MODEL_ARCH"] = "Qwen35JointEmbedding"
    os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = "inference_plugins_sglang.processors"
    {"check": check, "offline": lambda: offline(args), "online": lambda: online(args)}[args.mode]()


if __name__ == "__main__":
    main()

"""Helpers for test_vllm_container.sh; run that script inside your vLLM container."""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.request import urlopen

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    engine_contract,
    post_json,
)
from _visual import (  # noqa: PLC2701 - Shared visual smoke checks.
    image_data_url,
    offline_visual_inputs,
    validate_visual_outputs,
    visual_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from inference_plugins.contract import Contract
    from inference_plugins.types import JointEmbedding, PathLike
    from vllm import LLM

MAX_CONTEXT_LENGTH = 512

TEXTS = ["What is semantic search?", "Search using meaning, rather than exact words."]


def check() -> None:
    """Check runtime registration, output layout, and embedding head parity."""
    import torch
    from inference_plugins.contract import Contract
    from inference_plugins.heads import JointHeads, unpack
    from inference_plugins_vllm.io import JointIOProcessor
    from inference_plugins_vllm.model import DensePooler, JointPooler, TokenPooler
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins
    from vllm.utils.import_utils import resolve_obj_by_qualname

    for group in ("vllm.general_plugins", "vllm.io_processor_plugins", "vllm.endpoint_plugins"):
        entries = [
            e for e in importlib.metadata.entry_points(group=group) if e.name == "joint_embeddings"
        ]
        if len(entries) != 1:
            raise RuntimeError(f"Expected one joint_embeddings entry in {group}, got {entries}")
        if group == "vllm.endpoint_plugins":
            endpoint = entries[0].load()()
            if endpoint.required_tasks != ("embed", "token_embed"):
                raise RuntimeError("Unexpected joint pooling endpoint task requirements")
        if group == "vllm.io_processor_plugins":
            assert resolve_obj_by_qualname(entries[0].load()()) is JointIOProcessor  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
    load_general_plugins()
    assert "Qwen35JointEmbedding" in ModelRegistry.get_supported_archs()  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
    assert JointIOProcessor is not None  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
    torch.manual_seed(0)
    c = Contract()
    weight = torch.randn(c.multivector_dim, c.dense_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = torch.randn(5, c.dense_dim, device=device)
    from vllm.config import SchedulerConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(
        VllmConfig(scheduler_config=SchedulerConfig(enable_chunked_prefill=False))
    ):
        pooler = JointPooler(
            TokenPooler(c, JointHeads(c, weight, device=device).projection), DensePooler(c)
        )
    cursor = SimpleNamespace(
        num_scheduled_tokens_cpu=torch.tensor([2, 3]),
        last_token_indices_gpu=torch.tensor([1, 4], device=device),
        get_finished_mask=lambda: [True, True],
    )
    metadata = SimpleNamespace(
        prompt_lens=torch.tensor([2, 3]),
        tasks=["token_embed", "token_embed"],
        pooling_params=[
            SimpleNamespace(
                dimensions=None,
                use_activation=False,
                late_chunking_params=None,
                extra_kwargs={"joint_embeddings": True},
            )
            for _ in range(2)
        ],
        get_pooling_cursor=lambda: cursor,
        get_prompt_token_ids_cpu=lambda: [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
    )
    packed = pooler(hidden, metadata)
    for h, tensor in zip(hidden.split([2, 3]), packed, strict=True):
        item = unpack(tensor, c)
        torch.testing.assert_close(
            torch.tensor(item["dense"], device=device), torch.nn.functional.normalize(h[-1], dim=-1)
        )
        torch.testing.assert_close(
            torch.tensor(item["multivector"], device=device),
            torch.nn.functional.normalize(h @ weight.to(device).T, dim=-1),
        )
    print(  # noqa: T201 - CLI output is intended for the user.
        "PASS: installed entry points, fresh-process model discovery, engine API imports, joint pooler parity"
    )


def validate(
    items: Sequence[Mapping[str, Any]],
    lengths: list[int],
    contract: Contract,
    *,
    heads: tuple[str, ...] = ("dense", "multivector"),
) -> None:
    """Check that precisely the requested heads have valid shapes and values."""
    import torch

    if len(items) != len(lengths):
        raise ValueError("Expected one structured result per input")
    for item, length in zip(items, lengths, strict=True):
        if set(item) != set(heads):
            raise ValueError(f"Expected only these embedding fields: {heads}")
        for field in heads:
            tensor = torch.tensor(item[field])
            shape = (
                (contract.dense_dim,) if field == "dense" else (length, contract.multivector_dim)
            )
            if tensor.shape != shape:
                raise ValueError(f"Unexpected {field} shape: {tensor.shape}; expected {shape}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite {field} output")
            normalize = (
                contract.normalize_dense if field == "dense" else contract.normalize_multivector
            )
            if normalize:
                norms = tensor.norm(dim=-1)
                torch.testing.assert_close(norms, torch.ones_like(norms), atol=0.02, rtol=0.02)


def inputs(model: PathLike) -> tuple[Path, Contract, list[int]]:
    """Prepare smoke inputs and expected token lengths from the model contract."""
    root, c, tokenizer = engine_contract(model)
    ids = c.tokenize(tokenizer, TEXTS, "document")
    lengths = [sum(token not in c.skip_token_ids for token in row) for row in ids]
    if any(length == 0 for length in lengths):
        raise ValueError("Smoke texts must retain at least one multivector token")
    return root, c, lengths


def check_pil(llm: LLM, prompt: dict[str, Any], reference: JointEmbedding) -> None:
    """Compare a direct PIL input with the equivalent data-URL image output."""
    import torch
    from inference_plugins_vllm.offline import encode_joint
    from PIL import Image

    png = base64.b64decode(image_data_url().split(",", 1)[1])
    with Image.open(BytesIO(png)) as image:
        pil_prompt = {**prompt, "multi_modal_data": {"image": image}}
        result = encode_joint(llm, [pil_prompt])[0]
    for field in ("dense", "multivector"):
        torch.testing.assert_close(
            torch.tensor(result[field]), torch.tensor(reference[field]), atol=0.02, rtol=0.02
        )


def offline(args: argparse.Namespace) -> None:
    """Run offline inference and save validated outputs for online comparison."""
    from inference_plugins_vllm.offline import encode_joint
    from vllm import LLM
    from vllm.config import PoolerConfig

    root, c, lengths = inputs(args.model)
    _, _, tokenizer = engine_contract(args.model)
    token_ids = c.tokenize(tokenizer, TEXTS, "document")
    # Match server limits; preflight before allocating GPU memory.
    if max(map(len, token_ids)) > MAX_CONTEXT_LENGTH:
        raise ValueError("Smoke inputs exceed the 512-token test runtime")
    llm = LLM(
        model=str(root),
        runner="pooling",
        hf_overrides={"architectures": ["Qwen35JointEmbedding"]},
        pooler_config=PoolerConfig(task="embed" if args.task == "embed" else "token_embed"),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=512,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    import torch

    prompts = [{"prompt_token_ids": row} for row in token_ids]
    if args.task != "plugin":
        field = "dense" if args.task == "embed" else "multivector"
        outputs = llm.encode(prompts, pooling_task=args.task)
        standalone = [{field: output.outputs.data.tolist()} for output in outputs]
        validate(standalone, lengths, c, heads=(field,))
        Path(args.result).with_suffix(f".{args.task}.json").write_text(
            json.dumps(standalone), encoding="utf-8"
        )
        print(f"PASS: offline {args.task} only, dimensions, finite values and normalization")  # noqa: T201 - Smoke check report.
        return
    items = encode_joint(llm, prompts)
    validate(items, lengths, c)
    for task, field in (("embed", "dense"), ("token_embed", "multivector")):
        reference = json.loads(
            Path(args.result).with_suffix(f".{task}.json").read_text(encoding="utf-8")
        )
        for single, joint in zip(reference, items, strict=True):
            joint_output: Mapping[str, Any] = joint
            torch.testing.assert_close(
                torch.tensor(single[field]), torch.tensor(joint_output[field]), atol=0.02, rtol=0.02
            )
    print("PASS: offline combined heads match separate dense and multivector inference")  # noqa: T201 - Smoke check report.
    visual_prompts = offline_visual_inputs(tokenizer)
    visual = encode_joint(llm, visual_prompts)
    check_pil(llm, visual_prompts[1], visual[1])
    print(json.dumps(validate_visual_outputs(visual), indent=2))  # noqa: T201 - Smoke check report.
    Path(args.result).with_suffix(".visual.json").write_text(json.dumps(visual), encoding="utf-8")
    Path(args.result).write_text(json.dumps(items), encoding="utf-8")
    print("PASS: offline text/PIL/visual batches, image sensitivity, dimensions and finite values")  # noqa: T201 - CLI output is intended for the user.


def online(args: argparse.Namespace) -> None:
    """Check server readiness and compare online outputs with offline results."""
    import torch

    _, c, lengths = inputs(args.model)
    _, _, tokenizer = engine_contract(args.model)
    token_ids = c.tokenize(tokenizer, TEXTS, "document")
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            os.kill(args.server_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("vLLM server exited; inspect server.log") from exc
        try:
            with urlopen(args.url + "/health", timeout=2) as response:  # noqa: S310 - Explicit user-supplied inference server URL.
                if response.status == HTTPStatus.OK:
                    break
        except (URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("vLLM server startup timeout; inspect server.log")
        time.sleep(1)
    expected = json.loads(Path(args.result).read_text(encoding="utf-8"))

    def request(texts: list[str]) -> list[JointEmbedding]:
        return post_json(
            args.url + "/pooling",
            {
                "model": "joint-model",
                "task": "plugin",
                "input": c.tokenize(tokenizer, texts, "document"),
                "add_special_tokens": False,
            },
        )["data"]

    def compare(actual: list[JointEmbedding], reference: list[JointEmbedding]) -> None:
        """Compare dense and token embeddings against reference outputs."""
        for a, b in zip(actual, reference, strict=True):
            for field in ("dense", "multivector"):
                torch.testing.assert_close(
                    torch.tensor(a[field]), torch.tensor(b[field]), atol=0.02, rtol=0.02
                )

    batch = request(TEXTS)
    validate(batch, lengths, c)
    compare(batch, expected)
    compare(
        post_json(
            args.url + "/pooling",
            {
                "model": "joint-model",
                "task": "plugin",
                "data": {"input": token_ids, "add_special_tokens": False},
            },
        )["data"],
        expected,
    )
    for task, field in (("embed", "dense"), ("token_embed", "multivector")):
        for encoding in ("float", "base64"):
            separate = post_json(
                args.url + "/pooling",
                {
                    "model": "joint-model",
                    "task": task,
                    "input": token_ids,
                    "add_special_tokens": False,
                    "encoding_format": encoding,
                    "embed_dtype": "float32",
                    "endianness": "little",
                },
            )["data"]
            assert len(separate) == len(expected)  # noqa: S101 - Smoke check assertions deliberately fail on runtime incompatibility.
            for item, joint in zip(separate, expected, strict=True):
                reference = torch.tensor(joint[field])
                if encoding == "base64":
                    actual = torch.tensor(
                        [
                            x[0]
                            for x in struct.iter_unpack(
                                "<f", base64.b64decode(item["data"], validate=True)
                            )
                        ]
                    )
                    actual = actual.reshape(reference.shape)
                else:
                    actual = torch.tensor(item["data"])
                torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
    dense_api = post_json(
        args.url + "/v1/embeddings", {"model": "joint-model", "input": token_ids}
    )["data"]
    for item, joint in zip(dense_api, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(item["embedding"]), torch.tensor(joint["dense"]), atol=0.02, rtol=0.02
        )
    visual = [
        post_json(
            args.url + "/pooling",
            {
                "model": "joint-model",
                "task": "plugin",
                "messages": messages,
                "add_generation_prompt": False,
            },
        )["data"][0]
        for messages in visual_inputs()
    ]
    print(json.dumps(validate_visual_outputs(visual), indent=2))  # noqa: T201 - Smoke check report.
    visual_reference = json.loads(
        Path(args.result).with_suffix(".visual.json").read_text(encoding="utf-8")
    )
    compare(visual, visual_reference)

    # Concurrent single-input requests must retain request isolation and order.
    with ThreadPoolExecutor(max_workers=2) as executor:
        singles = list(executor.map(lambda text: request([text]), TEXTS))
    for i, single in enumerate(singles):
        validate(single, [lengths[i]], c)
        compare(single, [expected[i]])
    print(  # noqa: T201 - Smoke check report.
        "PASS: text/visual HTTP, image sensitivity, concurrency and offline/online parity (atol=rtol=0.02)"
    )


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "offline", "online"])
    parser.add_argument("--task", choices=["embed", "token_embed", "plugin"], default="plugin")
    parser.add_argument("--model")
    parser.add_argument("--result")
    parser.add_argument("--url")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    if args.mode == "check":
        check()
    elif args.mode == "offline":
        offline(args)
    else:
        online(args)


if __name__ == "__main__":
    main()

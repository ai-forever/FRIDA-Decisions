"""Run one vLLM variant locally; use --task embed, token_embed or plugin.

    python examples/vllm_offline.py --model /models/st-model --task embed
    python examples/vllm_offline.py --model /models/st-model --task token_embed
    python examples/vllm_offline.py --model /models/st-model --task plugin

The tasks return dense only, multivectors only, or both heads, respectively.
"""

from __future__ import annotations

from _common import (  # noqa: PLC2701 - Shared helper for the sibling example scripts.
    arguments,
    display,
    engine_contract,
)


def main() -> None:
    """Parse command-line arguments and run the requested operation."""
    args = arguments(__doc__, pooling_tasks=True)
    if args.data_envelope and args.task != "plugin":
        raise ValueError("--data-envelope requires --task plugin")
    root, contract, _ = engine_contract(args.model)
    from vllm import LLM
    from vllm.config import PoolerConfig

    llm = LLM(
        model=str(root),
        runner="pooling",
        hf_overrides={"architectures": ["Qwen35JointEmbedding"]},
        pooler_config=PoolerConfig(task="token_embed" if args.task == "plugin" else args.task),
        io_processor_plugin="frida_embeddings_v2" if args.data_envelope else None,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=contract.max_length,
        max_num_batched_tokens=contract.max_length,
        enforce_eager=True,
    )
    if args.data_envelope:
        outputs = llm.encode({"data": {"input": args.texts}}, pooling_task="plugin")
        display(outputs[0].outputs)
    elif args.task == "plugin":
        from inference_plugins_vllm.offline import encode_joint

        display(encode_joint(llm, args.texts))
    else:
        outputs = llm.encode(args.texts, pooling_task=args.task)
        field = "dense" if args.task == "embed" else "multivector"
        display([{field: item.outputs.data.tolist()} for item in outputs])


if __name__ == "__main__":
    main()

r"""Encode PIL images offline with any of the three vLLM tasks.

    python examples/vllm_multimodal_offline.py --model /path/to/merged-st-model \\
        --image page.png --task plugin --text 'Describe this page'

Use --task embed or --task token_embed for a single head. The example runs an
image-only input and an image+text input, retaining PIL images in memory. Increase
--max-model-len if the image patches exceed the saved text context length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import display, engine_contract  # noqa: PLC2701 - Shared sibling helper.
from PIL import Image


def main() -> None:
    """Run image-only and captioned PIL inputs without HTTP or base64 encoding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", default="Describe this page")
    parser.add_argument("--task", choices=["embed", "token_embed", "plugin"], default="plugin")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--data-envelope", action="store_true", help="Use the official plugin data envelope"
    )
    args = parser.parse_args()
    if args.data_envelope and args.task != "plugin":
        parser.error("--data-envelope requires --task plugin")
    from vllm import LLM
    from vllm.config import PoolerConfig

    root, contract, tokenizer = engine_contract(args.model)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    prompts = []
    for caption in ("", args.text):
        content = [{"type": "image"}]
        if caption:
            content.append({"type": "text", "text": caption})
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False
        )
        prompts.append({"prompt": formatted, "multi_modal_data": {"image": image}})
    llm = LLM(
        model=str(root),
        runner="pooling",
        hf_overrides={"architectures": ["Qwen35JointEmbedding"]},
        pooler_config=PoolerConfig(task="token_embed" if args.task == "plugin" else args.task),
        io_processor_plugin="joint_embeddings" if args.data_envelope else None,
        enable_prefix_caching=False,
        enforce_eager=True,
        max_model_len=args.max_model_len or contract.max_length,
    )
    if args.data_envelope:
        outputs = llm.encode({"data": {"input": prompts}}, pooling_task="plugin")
        display(outputs[0].outputs)
    elif args.task == "plugin":
        from inference_plugins_vllm.offline import encode_joint

        display(encode_joint(llm, prompts))
    else:
        outputs = llm.encode(prompts, pooling_task=args.task)
        print(json.dumps([item.outputs.data.tolist() for item in outputs]))  # noqa: T201 - Example output.


if __name__ == "__main__":
    main()

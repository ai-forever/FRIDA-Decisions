"""Combined embedding output; single heads use native vLLM tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from inference_plugins.heads import unpack_tensors
from inference_plugins.types import DisplayEmbedding
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.renderers.inputs.preprocess import prompt_to_seq

from .checkpoint import load_checkpoint
from .offline import joint_pooling_params

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from inference_plugins.types import TensorOutput
    from vllm.config import VllmConfig
    from vllm.inputs import PromptType
    from vllm.outputs import PoolingRequestOutput
    from vllm.pooling_params import PoolingParams
    from vllm.renderers import BaseRenderer


class JointIOProcessor(IOProcessor[dict[str, Any], list[DisplayEmbedding]]):
    """Return both heads through the plugin endpoint."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer) -> None:
        super().__init__(vllm_config, renderer)
        self.contract = load_checkpoint(vllm_config).contract

    def parse_data(self, data: object) -> dict[str, Any]:  # noqa: PLR6301 - Engine interface.
        """Accept native input or messages inside the official plugin envelope."""
        if not isinstance(data, dict) or ("input" in data) == ("messages" in data):
            raise ValueError("data must contain exactly one of input or messages")
        if {"embed_task", "role"} & data.keys():
            raise ValueError(
                "embed_task and role are unsupported; plugin always returns both heads"
            )
        if data.get("input", data.get("messages")) is None:
            raise ValueError("data input/messages cannot be null")
        return data

    def pre_process(
        self, prompt: dict[str, Any], request_id: str | None = None, **kwargs: Any
    ) -> Sequence[PromptType]:
        """Pass offline native prompts to vLLM without rendering or tokenization."""
        data = self.parse_data(prompt)
        if "messages" in data:
            raise ValueError(
                "Offline encode requires native prompts in data.input; use messages over HTTP"
            )
        if set(data) != {"input"}:
            raise ValueError("Offline data accepts only input; pass options through LLM.encode")
        return cast("Sequence[PromptType]", prompt_to_seq(data["input"]))

    def merge_pooling_params(  # noqa: PLR6301 - Engine interface.
        self, params: PoolingParams | None = None
    ) -> PoolingParams:
        """Choose native V2 execution while keeping the HTTP task plugin."""
        return joint_pooling_params(params)

    def tensor_outputs(
        self, model_output: Sequence[PoolingRequestOutput]
    ) -> Iterator[TensorOutput]:
        """Decode both heads without materializing Python lists."""
        for item in model_output:
            heads = unpack_tensors(item.outputs.data, self.contract)
            yield {"dense": heads["dense"], "multivector": heads["multivector"]}

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs: Any,
    ) -> list[DisplayEmbedding]:
        """Return both named embedding fields."""
        return [
            {"dense": heads["dense"].tolist(), "multivector": heads["multivector"].tolist()}
            for heads in self.tensor_outputs(model_output)
        ]

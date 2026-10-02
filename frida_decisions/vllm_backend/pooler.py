"""Span-mean readout: one margin per option of every prompt.

The pooler gets the hidden states of the tokens computed in this step, flat
across requests, plus each prompt's ids. For each prompt it finds the options
from the ids, averages each option's hidden states in fp32 and applies the head
-- the readout of `Judge`. The output for a prompt is a 1-D tensor of K
margins, in option order, no activation.

Under a prefix-cache hit the first `cached` tokens of a prompt were not computed
in this step and have no hidden states here. The computed part is parsed with
the model's own parser (`rows.check_prompt`), so what is read out is what the
model computed.

A prompt that does not parse, whose state hash is not the hash of its state, or
whose computed part the model had to isolate, gets a single NaN instead of
margins: raising here would take the whole engine down for one bad request, and
NaN cannot be mistaken for an answer (`aggregate` rejects it). vLLM's own warm-up
prompts land here by design.
"""
from __future__ import annotations

from collections.abc import Set

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .kernel import pool_margins, readout_plan
from .rows import PREFIX

logger = init_logger(f"vllm.plugins.{__name__}")    # vLLM configures only "vllm.*"


class FridaPooler(Pooler):
    def __init__(self, head: nn.Linear, vocab: int):
        super().__init__()
        self.head = head
        self.vocab = vocab

    def get_supported_tasks(self) -> Set[PoolingTask]:
        # vLLM's default model runner serves no "plugin" task; the IO processor sends classify.
        return {"classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor,
                pooling_metadata: PoolingMetadata) -> list[torch.Tensor | None]:
        cursor = pooling_metadata.get_pooling_cursor()
        prompts = pooling_metadata.get_prompt_token_ids_cpu()
        first = cursor.first_token_indices_gpu.tolist()
        scheduled = cursor.num_scheduled_tokens_cpu.tolist()
        prompt_lens = pooling_metadata.prompt_lens.tolist()
        finished = cursor.get_finished_mask()

        done = [i for i in range(len(prompts)) if finished[i]]   # all, without chunked prefill
        plan = readout_plan([prompts[i][:prompt_lens[i]].tolist() for i in done],
                            [scheduled[i] for i in done], [first[i] for i in done], self.vocab)
        for j, error in plan[4].items():
            ids = prompts[done[j]][:prompt_lens[done[j]]]
            if prompt_lens[done[j]] > 2 * PREFIX and bool(ids.any()):   # not vLLM's warm-up rows
                logger.warning("FRIDA-Decisions pooler: prompt rejected: %s", error)
        margins = iter(pool_margins(hidden_states, self.head, plan))
        return [next(margins) if finished[i] else None for i in range(len(prompts))]

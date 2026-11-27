"""The three seams between a compiled request and a number.

    Compiled --Renderer--> ScoringPlan --Executor(backbone)--> hidden --Head--> scores

* A **Renderer** owns every token decision: segment ids, separators, caps,
  positions, where the head reads. It stores `name` / `version` in the
  checkpoint's format file, because changing any of it changes scores.
* An **Executor** owns the physical layout and the backbone: row splitting,
  masks, position encoding (T5 buckets, RoPE ids), caches. It never changes
  which tokens a segment has.
* A **Head** turns hidden states at the read positions into one scalar per slot.

Implementations: renderers `render.t5_packed.T5PackedRenderer` (FRIDA, layouts
1/2) and `render.causal_scalar.CausalScalarRenderer` (Qwen-style causal LMs);
executors `exec.t5_dense.T5DenseExecutor`, `exec.t5_cached.T5StateCacheExecutor`,
`exec.causal.CausalPerOptionExecutor` and `exec.causal.CausalTreeExecutor`; one
head, `head.LinearReadoutHead`, for all of them. A linear
label-readout prompt for causal LMs (`render.label_prompt.LabelPromptRenderer`)
is a different shape -- one prompt per question, scores are label logits -- and
does not go through `ScoringPlan`.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from frida_decisions.v2.model.layout import ScoringPlan
from frida_decisions.v2.protocol import Compiled, Task, compiled_from_tasks


@runtime_checkable
class Renderer(Protocol):
    name: str
    version: str

    def render(self, request: Compiled | Sequence[Task]) -> ScoringPlan: ...


@runtime_checkable
class Executor(Protocol):
    def score(self, plans: Sequence[ScoringPlan]) -> list[Any]:
        """One 1-D tensor of scores per plan, in slot order."""
        ...

    def cost(self, plans: Sequence[ScoringPlan]) -> float:
        """What the forward would cost in the executor's own unit (dense T5: rows x L^2)."""
        ...


@runtime_checkable
class Head(Protocol):
    def __call__(self, hidden: Any, readout: Any) -> Any:
        """`hidden` (B, L, D), `readout` -> (n_slots,)."""
        ...


class TaskList(list):
    """A flat task list (what `compile_request` returns, and what the bench
    arms iterate) that also carries its structured form, question types
    included, for a renderer that needs them."""
    structured: Compiled | None = None


def as_compiled(request: Compiled | Sequence[Task]) -> Compiled:
    """Accept both today's flat task list and the structured form."""
    if isinstance(request, Compiled):
        return request
    if getattr(request, "structured", None) is not None:
        return request.structured
    return compiled_from_tasks(list(request))

"""A request to backend rows, and the rows' margins back to a response.

Shared by the vLLM server's IO processor (`server.py`) and the in-process
`VllmJudge` (`engine.py`), so both answer exactly like `Judge`. No torch and no
vLLM here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DecisionsConfig
from ..constants import PRODUCT_NAME
from ..packing import TextEncoder
from ..protocol import Calibration, Request, RequestError, aggregate, compile_request, parse_request
from .rows import Row, build_rows, tokenize_rows


@dataclass
class Prepared:
    request: Request
    candidates: list
    rows: list[Row]
    state_tokens: int
    state_truncated: bool
    special_split: list[str] = field(default_factory=list)


def prepare(request, text: TextEncoder, config: DecisionsConfig, state_max: int,
            vocab: int, max_row_tokens: int | None = None) -> Prepared:
    """Validate, compile and lay out one request.

    `request` is the JSON object `Judge` takes, or an already parsed `Request`.
    Raises `RequestError` for an invalid request, and for a row longer than
    `max_row_tokens` (the server's `--max-model-len`)."""
    parsed = request if isinstance(request, Request) else parse_request(request)
    candidates = compile_request(parsed, config.instruction_suffixes, config.yes_no_default_criteria)
    tok, split = tokenize_rows(candidates, text, config, state_max)
    rows = build_rows(tok, config, vocab)
    if max_row_tokens is not None:
        longest = max(len(r.ids) for r in rows)
        if longest > max_row_tokens:
            raise RequestError(f"a row of this request is {longest} tokens, over the server's "
                               f"limit of {max_row_tokens}; raise --max-model-len or lower the "
                               "state cut", field="state")
    return Prepared(parsed, candidates, rows, len(tok.state),
                    text.count(candidates[0].state) > len(tok.state), split)


def respond(prepared: Prepared, row_margins: list[list[float]], cached_tokens: int,
            prompt_tokens: int, calibration: Calibration | None = None,
            extra_usage: dict | None = None) -> dict:
    """The `Judge` response from each row's margins (row order, option order within a row)."""
    candidates = prepared.candidates
    if len(row_margins) != len(prepared.rows):
        raise RuntimeError(f"{len(prepared.rows)} rows sent, {len(row_margins)} came back")
    margins = [float("nan")] * len(candidates)
    for row, values in zip(prepared.rows, row_margins):
        if len(values) != len(row.candidates) or any(v != v for v in values):
            raise RuntimeError(f"a row of {len(row.candidates)} options returned {len(values)} "
                               "values: the model rejected the row (see the server log)")
        for index, value in zip(row.candidates, values):
            margins[index] = value
    answers = aggregate(prepared.request, candidates, margins, calibration)
    by_question: dict[str, dict[str, float]] = {}
    for cand, margin in zip(candidates, margins):
        by_question.setdefault(cand.question_id, {})[cand.option_id] = margin
    return {"model": PRODUCT_NAME,
            "answers": answers,
            "margins": by_question,
            "usage": {"state_tokens": prepared.state_tokens,
                      "state_truncated": prepared.state_truncated,
                      "questions": len(prepared.request.questions),
                      "candidates": len(candidates),
                      "rows": len(prepared.rows),
                      "prompt_tokens": prompt_tokens,
                      "cached_tokens": cached_tokens,
                      "special_tokens_split": prepared.special_split,
                      "backend": "vllm",
                      **(extra_usage or {})}}

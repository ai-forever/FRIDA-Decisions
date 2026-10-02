"""What every backend shares: model folder resolution, request handling, responses."""
from __future__ import annotations

import time
from pathlib import Path

from .config import DecisionsConfig
from .constants import PRODUCT_NAME
from .packing import TextEncoder, TokenizedRequest, tokenize
from .protocol import Calibration, aggregate, compile_request, parse_request


def resolve_model_dir(path_or_repo: str | Path, allow_patterns: list[str] | None = None,
                      revision: str | None = None) -> Path:
    """A local folder as is, otherwise a Hugging Face repo id downloaded to the hub cache."""
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), allow_patterns=allow_patterns,
                                  revision=revision))


class BaseJudge:
    """Request in, answers out. Subclasses implement `_score`."""

    backend = "base"

    def __init__(self, folder: Path, state_max: int | None = None):
        self.folder = Path(folder)
        self.config = DecisionsConfig.load(self.folder)
        self.text = TextEncoder(self.folder / "tokenizer.json")
        self.state_max = self.config.state_max_tokens if state_max is None else int(state_max)
        if self.state_max < 1:
            raise ValueError("state_max must be positive")

    # ----------------------------------------------------------- to implement
    def _score(self, requests: list[TokenizedRequest]) -> tuple[list[list[float]], dict]:
        """One list of margins per request, plus usage counters for the call."""
        raise NotImplementedError

    # ----------------------------------------------------------- public API
    def compile(self, request: dict):
        """Validate and compile a request: `(parsed, candidates, tokenized)`."""
        parsed = parse_request(request)
        candidates = compile_request(parsed, self.config.instruction_suffixes,
                                     self.config.yes_no_default_criteria)
        return parsed, candidates, tokenize(candidates, self.text, self.config, self.state_max)

    def __call__(self, request: dict, calibration: Calibration | None = None) -> dict:
        return self.judge(request, calibration)

    def judge(self, request: dict, calibration: Calibration | None = None) -> dict:
        """Answer every question of one request."""
        return self.judge_batch([request], calibration)[0]

    def judge_batch(self, requests: list[dict], calibration: Calibration | None = None) -> list[dict]:
        """Answer several requests. Their packed rows share encoder forwards."""
        if not requests:
            return []
        compiled = [self.compile(r) for r in requests]
        started = time.perf_counter()
        margins, usage = self._score([tok for _, _, tok in compiled])
        elapsed = (time.perf_counter() - started) * 1000.0
        out = []
        for (parsed, candidates, tok), m in zip(compiled, margins):
            answers = aggregate(parsed, candidates, m, calibration)
            by_question: dict = {}
            for cand, value in zip(candidates, m):
                by_question.setdefault(cand.question_id, {})[cand.option_id] = value
            out.append({
                "model": PRODUCT_NAME,
                "answers": answers,
                "margins": by_question,
                "usage": {
                    "state_tokens": len(tok.state),
                    "state_truncated": self.text.count(candidates[0].state) > len(tok.state),
                    "questions": len(parsed.questions),
                    "candidates": len(candidates),
                    **usage,
                    "backend": self.backend,
                    "milliseconds": round(elapsed, 2),   # wall time of the whole call
                },
            })
        return out

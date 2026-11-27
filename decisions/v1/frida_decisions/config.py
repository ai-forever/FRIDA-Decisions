"""`decisions_config.json`: the packing limits and texts that travel with the weights.

The encoder weights alone do not define the model. How long a state may be,
where a candidate is cut, how many options share one packed row and which
instruction text is appended to each question type are all part of what the
head was trained against, so they are stored next to the weights and read by
every backend from the same file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .constants import DECISIONS_CONFIG_FILE, NO_LABEL, YES_LABEL, QuestionType

# Appended to every question's instructions, one text per question type. The
# model was trained with exactly these strings; changing them changes scores.
DEFAULT_INSTRUCTION_SUFFIXES = {
    QuestionType.CHOICE: (
        "Оцени, является ли вариант, описанный в кандидате, "
        "подходящим ответом на вопрос выше, с учётом сведений в запросе. "
        "Суди о варианте по его описанию, а не по одной лишь тематической близости."
    ),
    QuestionType.SCORE: (
        "Оцени, точно ли уровень, описанный в кандидате, "
        "характеризует запрос в рамках оценки выше. "
        "Суди об уровне по его описанию, а не по одной лишь тематической близости."
    ),
    QuestionType.YES_NO: (
        "Оцени, верно ли кандидат описывает ответ на вопрос выше, "
        "с учётом сведений в запросе. "
        "Отвечай утвердительно, если этот вариант выполняется, иначе отрицательно."
    ),
    QuestionType.RANKING: (
        "Оцени, насколько кандидат подходит под вопрос выше с учётом сведений "
        "в запросе. Кандидат — это сам материал, а не описание варианта: "
        "суди по его содержанию, а не по одной лишь тематической близости."
    ),
}

# A yes/no question without `criteria` is scored as a two-option question
# against these two texts.
DEFAULT_YES_NO_CRITERIA = {
    YES_LABEL: "The answer to the question is yes.",
    NO_LABEL: "The answer to the question is no.",
}


@dataclass
class DecisionsConfig:
    # Token caps per segment. A segment longer than its cap is cut from the right.
    state_max_tokens: int = 512          # the trained range; longer states are cut
    instruction_max_tokens: int = 96     # question text + type suffix
    option_max_tokens: int = 256         # one option, including the closing EOS token
    # Row splitting. A packed row is closed when it holds this many options, or
    # when the next option would take it past `max_row_tokens`. Both splits are
    # exact: options never attend to each other, so the row an option travels
    # in does not change its score.
    max_options_per_row: int = 16
    max_row_tokens: int = 1024
    # Packed widths are rounded up to a multiple of this (fused attention kernels
    # want it). Padding is masked out and never read.
    align: int = 8
    eos_token_id: int = 2                # closes every option
    pad_token_id: int = 0
    tokenizer_max_length: int = 1024
    # T5 relative-position bucketing, copied from the encoder config.
    num_buckets: int = 32
    max_distance: int = 128
    instruction_suffixes: dict = field(default_factory=lambda: dict(DEFAULT_INSTRUCTION_SUFFIXES))
    yes_no_default_criteria: dict = field(default_factory=lambda: dict(DEFAULT_YES_NO_CRITERIA))
    # Format version of this file.
    format_version: int = 1

    @classmethod
    def load(cls, folder: str | Path) -> "DecisionsConfig":
        path = Path(folder) / DECISIONS_CONFIG_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, folder: str | Path) -> Path:
        path = Path(folder) / DECISIONS_CONFIG_FILE
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return path

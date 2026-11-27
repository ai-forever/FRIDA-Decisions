"""Segments for a causal LM backbone with one scalar per option, read
at the option's closing separator.

The renderer uses the backbone's own special tokens. With the Qwen defaults:

    <|im_start|> state <|im_end|>  question <|im_end|>  option <|im_end|>
                                                               ^ readout

Each segment is tokenized on its own (no BPE merge across a boundary, no chat
template); positions are contiguous along the path state -> question -> option,
so every option of a question sits at the same positions right after the
question -- exactly what a single `state + question + option` sequence gives a
RoPE model. The tree (`ScoringPlan`) is the same structure as the T5 path, so
the same head, the same `plan.group` and the same losses apply.

Segment caps are optional: `None` keeps the text whole (the data are already
clipped in characters); a cap cuts from the right, separators excluded.

`state_format` (recorded in the checkpoint's format record):

* "json" (default, renderer version "1"): a non-string state is the protocol's
  compact sorted JSON, and text goes to the tokenizer as written;
* "messages_v1" (renderer version "3"; "2" is the decision loop's): a state
  with `messages` or `sessions` -- given as a dict or as its JSON text -- is
  `protocol.session_text`, and every segment's text is `protocol.neutralise`d,
  so a `<|im_end|>` written in a state, question or option never becomes the
  separator. An old loader refuses the version instead of reading text the
  model was not trained on.

`type_text` (recorded too): what follows a question's instructions. "adapter" is
the compiled per-type prose, `protocol.ADAPTERS_RU`; "none" drops it, with the
blank line before it. The drop is
here, not in the compile step, so a checkpoint's own format record decides it on
every path, and a record without the field reads as "adapter".
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

from frida_decisions.v2.model.layout import ScoringPlan, Segment
from frida_decisions.v2.model.render.base import as_compiled
from frida_decisions.v2.protocol import (ADAPTERS_RU, Compiled, Task, content_text, neutralise,
                                          upgrade_state)

CAUSAL_SCALAR = "causal_scalar"
CAUSAL_SCALAR_VERSION = "1"
STATE_FORMATS = {"json": CAUSAL_SCALAR_VERSION, "messages_v1": "3"}
OPTION_CONTEXTS = ("none", "list")
TYPE_TEXTS = ("adapter", "none")
# Renderer version per (state_format, option_context, type_text): a reader that
# does not know a combination refuses the checkpoint instead of rendering it
# another way. Versions 6 through 9 are reserved.
VERSIONS = {("json", "none", "adapter"): CAUSAL_SCALAR_VERSION,
            ("messages_v1", "none", "adapter"): "3",
            ("json", "list", "adapter"): "4", ("messages_v1", "list", "adapter"): "5",
            ("json", "none", "none"): "10", ("messages_v1", "none", "none"): "11",
            ("json", "list", "none"): "12", ("messages_v1", "list", "none"): "13"}
OPTION_LIST_HEADER = "Варианты:"


@dataclass(frozen=True)
class CausalScalarConfig:
    """Token strings of the separators, segment caps and row limits."""
    state_open: tuple[str, ...] = ("<|im_start|>",)
    state_close: tuple[str, ...] = ("<|im_end|>",)
    question_close: tuple[str, ...] = ("<|im_end|>",)
    option_close: tuple[str, ...] = ("<|im_end|>",)   # its last token is the readout
    state_max_tokens: int | None = None
    instruction_max_tokens: int | None = None
    option_max_tokens: int | None = None
    readout: str = "last"
    # Only for executors that pack several options into one row (tree mask).
    max_options_per_row: int = 16
    max_row_tokens: int | None = None
    # Per-option executor: micro-batch so that rows x padded length stays under
    # this many tokens; None = one forward for the whole batch.
    max_batch_tokens: int | None = None
    # Packed executor: the length of a packed sequence, and the row cap; and
    # FlexAttention's `kernel_options` (None: `causal_packed.FLEX_KERNEL_OPTIONS`,
    # {}: torch's own choice).
    pack_tokens: int = 16384
    flex_kernel_options: dict | None = None
    # How a state is turned into text (module doc): "json" or "messages_v1".
    state_format: str = "json"
    # "list": a choice question of at most `option_context_max` options carries
    # the full option list after its instruction, so every option is scored
    # having seen its alternatives (the question segment is every option's
    # prefix, so executors stay exact). "none": the question alone.
    option_context: str = "none"
    option_context_max: int = 10
    # What follows a question's instructions (module doc): "adapter" or "none".
    type_text: str = "adapter"

    @property
    def version(self) -> str:
        """The renderer version this config writes into a format record."""
        return VERSIONS[(self.state_format, self.option_context, self.type_text)]

    def replace(self, **changes) -> "CausalScalarConfig":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CausalScalarConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        data = {k: (tuple(v) if isinstance(v, list) else v)
                for k, v in data.items() if k in known}
        return cls(**data)


# Separator sets by chat format: ChatML (Qwen, the default) and GigaChat3, whose
# messages end with `<|message_sep|>\n\n` (the newlines are part of the token).
SEPARATORS = {
    "chatml": {},
    "gigachat": {"state_open": ("<s>",), "state_close": ("<|message_sep|>\n\n",),
                 "question_close": ("<|message_sep|>\n\n",),
                 "option_close": ("<|message_sep|>\n\n",)},
}


class CausalScalarRenderer:
    name = CAUSAL_SCALAR
    version = CAUSAL_SCALAR_VERSION

    def __init__(self, tokenizer, config: CausalScalarConfig | None = None):
        self.tokenizer = tokenizer
        self._serving_backend = None
        self.config = config or CausalScalarConfig()
        if self.config.readout not in ("last", "mean"):
            raise ValueError(f"readout {self.config.readout!r}")
        if self.config.option_context not in OPTION_CONTEXTS:
            raise ValueError(f"option_context {self.config.option_context!r} is not one of "
                             f"{OPTION_CONTEXTS}")
        if self.config.type_text not in TYPE_TEXTS:
            raise ValueError(f"type_text {self.config.type_text!r} is not one of {TYPE_TEXTS}")
        if self.config.state_format not in STATE_FORMATS:
            raise ValueError(f"state_format {self.config.state_format!r} is not one of "
                             f"{sorted(STATE_FORMATS)}")
        c = self.config
        self.state_open = self._special(c.state_open)
        self.state_close = self._special(c.state_close)
        self.question_close = self._special(c.question_close)
        self.option_close = self._special(c.option_close)
        # A VL backbone's `model/vision.VisionInputs` (set by the encoder built
        # with `vision=True`); None: the renderer reads text only.
        self.media = None

    def _special(self, tokens: Sequence[str]) -> list[int]:
        """Ids of separator tokens; each must be one known token of the vocabulary."""
        ids = []
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for token in tokens:
            i = self.tokenizer.convert_tokens_to_ids(token)
            if i is None or (unk is not None and i == unk and token != self.tokenizer.unk_token):
                raise ValueError(f"separator {token!r} is not a token of this tokenizer")
            ids.append(int(i))
        return ids

    @property
    def separator_ids(self) -> list[int]:
        """Distinct separator ids: the embedding rows worth training (`train_tokens`)."""
        out = []
        for i in self.state_open + self.state_close + self.question_close + self.option_close:
            if i not in out:
                out.append(i)
        return out

    def encode(self, text: str, cap: int | None) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids if cap is None else ids[:cap]

    def state_ids(self, text: str) -> list[int]:
        return [*self.state_open, *self.encode(text, self.config.state_max_tokens),
                *self.state_close]

    def question_ids(self, text: str) -> list[int]:
        return [*self.encode(text, self.config.instruction_max_tokens), *self.question_close]

    def option_ids(self, text: str) -> list[int]:
        return [*self.encode(text, self.config.option_max_tokens), *self.option_close]

    def text(self, content) -> str:
        """A segment's text under this config's `state_format`."""
        text = content_text(content)
        return text if self.config.state_format == "json" else neutralise(text)

    def instruction_text(self, question) -> str:
        """The compiled instruction, with type_text "none" without its adapter.
        The adapter is found by its text, so a bare flat task list (no question
        types) renders the same as a `Compiled`."""
        text = self.text(question.instruction)
        if self.config.type_text == "adapter":
            return text
        for adapter in ADAPTERS_RU.values():
            if text.endswith("\n\n" + adapter):
                return text[:-len(adapter) - 2]
        raise ValueError(f"type_text 'none': question {question.question_id!r} does not end "
                         "with an ADAPTERS_RU text, so there is nothing to drop")

    def question_text(self, question) -> str:
        """The instruction, and with option_context "list" the options of a
        choice question of at most option_context_max of them, one per line,
        in the order they are scored."""
        text = self.instruction_text(question)
        if self.config.option_context == "list" and question.qtype is None:
            raise ValueError("option_context 'list' needs question types: render a Compiled "
                             "(compile_structured) or a TaskList, not a bare task list")
        if (self.config.option_context == "list" and question.qtype == "choice"
                and len(question.options) <= self.config.option_context_max):
            listed = "\n".join(f"- {self.text(o.content)}" for o in question.options)
            text = f"{text}\n\n{OPTION_LIST_HEADER}\n{listed}"
        return text

    def state_text(self, content) -> str:
        if self.config.state_format == "json":
            return content_text(content)
        return neutralise(upgrade_state(content_text(content)))

    def enable_serving_tokenization(self):
        """Allow the verified Qwen tokenizer to emit only the token IDs consumed by the model."""
        from tokenizers import __version__ as tokenizers_version
        from transformers import __version__ as transformers_version

        tokenizer = self.tokenizer
        if (transformers_version == "5.19.0" and tokenizers_version == "0.23.3"
                and type(tokenizer).__module__ == "transformers.models.qwen2.tokenization_qwen2"
                and type(tokenizer).__name__ == "Qwen2Tokenizer" and tokenizer.is_fast):
            self._serving_backend = tokenizer.backend_tokenizer

    def _batch_suffix_ids(self, compiled, state=None):
        """Tokenize the text segments of one request with a single tokenizer call."""
        texts, limits, closers = [], [], []
        if state is not None:
            texts.append(state)
            limits.append(self.config.state_max_tokens)
            closers.append([])
        for question in compiled.questions:
            texts.append(self.question_text(question))
            limits.append(self.config.instruction_max_tokens)
            closers.append(self.question_close)
            for option in question.options:
                texts.append(self.text(option.content))
                limits.append(self.config.option_max_tokens)
                closers.append(self.option_close)
        if not texts:
            return
        if state is not None:
            encodings = self._serving_backend.encode_batch_fast(texts, add_special_tokens=False)
            encoded = (entry.ids for entry in encodings)
        else:
            encoded = self.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False,
                                     return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        for ids, limit, close in zip(encoded, limits, closers, strict=True):
            if limit is not None:
                ids = ids[:limit]
            yield ids + close if close else ids


    def render(self, request: Compiled | Sequence[Task], images=None, *, batch_suffix=False) -> ScoringPlan:
        """The request's tree. `images` (paths or PIL images; needs `media`) go
        at the start of the state, before its text; the state's positions are
        then the model's M-RoPE rows (3, s) and the text after it continues
        from their max + 1. Without images, nothing here differs from a text
        renderer."""
        compiled = as_compiled(request)
        prepared_state = None
        suffix = None
        backend = self._serving_backend
        if (batch_suffix and backend is not None and backend.padding is None and backend.truncation is None
                and backend.encode_special_tokens == self.tokenizer.split_special_tokens):
            suffix = self._batch_suffix_ids(compiled, self.state_text(compiled.state))
            prepared_state = next(suffix)
        mm = {}
        if images:
            if self.media is None:
                raise ValueError("images given to a text-only model: build it with vision=True")
            mm = self.media.prepare(images)
            text = (prepared_state if prepared_state is not None else
                    self.encode(self.state_text(compiled.state), self.config.state_max_tokens))
            blocks = [t for grid in mm["image_grid_thw"] for t in self.media.block(grid)]
            state = [*self.state_open, *blocks, *text, *self.state_close]
            state_pos = self.media.positions(state, mm["image_grid_thw"])
            s = int(state_pos.max()) + 1
        else:
            state = ([*self.state_open, *prepared_state, *self.state_close] if prepared_state is not None
                     else self.state_ids(self.state_text(compiled.state)))
            s = len(state)
            state_pos = range(s)
        segments = [Segment(None, state, state_pos, "state")]
        if suffix is None and batch_suffix:
            suffix = self._batch_suffix_ids(compiled)
        group, keys, slot = [], [], 0
        for qi, question in enumerate(compiled.questions):
            q_ids = self.question_ids(self.question_text(question)) if suffix is None else next(suffix)
            q_index = len(segments)
            segments.append(Segment(0, q_ids, range(s, s + len(q_ids)), "question"))
            offset = s + len(q_ids)
            for option in question.options:
                o_ids = self.option_ids(self.text(option.content)) if suffix is None else next(suffix)
                segments.append(Segment(q_index, o_ids, range(offset, offset + len(o_ids)),
                                        "option", readout=(slot, self.config.readout)))
                group.append(qi)
                keys.append((question.question_id, option.option_id))
                slot += 1
        return ScoringPlan(segments, slot, group, len(compiled.questions), keys, mm=mm)

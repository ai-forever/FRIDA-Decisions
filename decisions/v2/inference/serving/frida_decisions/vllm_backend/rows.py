"""One packed row as a flat list of token ids, and back.

vLLM hands a model nothing but token ids and positions, so the structure that
`packing.layout_rows` keeps in Python lists -- where the state ends, which
tokens are a question, which are an option and of which question -- has to
travel inside the ids. It travels as marker tokens from FRIDA's own vocabulary:

    <pad> H{4}  state  N{4} <pad>  ( <s> question  ( <mask> option </s> )+ )+

* `<pad>` H{4} -- four ids derived from sha256 of the state ids. They put the
  identity of the *whole* state into the first prefix-cache block, so a cached
  block of state is reused only by a row with the same complete state (a state
  token's keys depend on the tokens after it, which vLLM's causal block hash
  does not cover). A request-level `cache_salt` would do the same, but vLLM
  applies that one over every prompt, so it stays free for tenant isolation.
  The pooler recomputes the hash and refuses a row where it does not match.
* N{4} `<pad>` -- a nonce, fresh per row, *directly* after the state, then the
  separator. vLLM reuses whole 16-token blocks whose ids match, so a hit ends at
  the first block holding a nonce id: never past the state, short of a nonce id
  repeating, and then only into the nonce (handled below). Questions and options
  are never reused, so a retry of the same request still computes every token
  the pooler reads. (A constant separator *before* the nonce would end a
  matching block one token past the state whenever the state length is 10 mod
  16.)
* `<s>` (1) opens a question, `<mask>` (4) opens an option. The option's own
  trailing `</s>` (2) is the EOS every candidate ends with: content, read out
  like the rest of the option.

Every marker -- both `<pad>`, the hash and nonce ids, `<s>`, `<mask>` -- attends
to itself only and nothing attends to it, so for every other token it is an
exact no-op, wherever it was computed. Positions are those of `layout_rows`:
state `0..S-1`, question `S..S+I-1`, every option restarting at `S+I`,
recomputed from the ids.

Content must not contain a marker id. `TextEncoder` turns a literal "<s>" in
the text into id 1, so `tokenize_rows` re-tokenizes such a segment with special
tokens split into plain text and records it. That is the one place this backend
can differ from `Judge`, which would feed id 1 to the model.

No torch and no vLLM in this file.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from ..config import DecisionsConfig
from ..packing import TextEncoder, TokenizedRequest, layout_rows, tokenize

PAD, BOS, EOS, MASK = 0, 1, 2, 4
MARKERS = (PAD, BOS, MASK)
CONTENT_BREAKERS = frozenset(MARKERS + (EOS,))   # ids content may not contain
HASH_LEN = 4
NONCE_LEN = 4
PREFIX = 1 + HASH_LEN              # `<pad> H{4}` before the state
ID_LOW = 5                         # hash and nonce ids are drawn from [ID_LOW, vocab)

# token kinds
STATE, QUESTION, OPTION, MARK = 0, 1, 2, 3


class LayoutError(ValueError):
    """A row whose ids do not follow the grammar above."""


def state_hash(state: list[int], vocab: int) -> list[int]:
    """HASH_LEN ids in [ID_LOW, vocab) from sha256 of the state ids."""
    digest = hashlib.sha256(b"frida-decisions-state:" + ",".join(map(str, state)).encode()).digest()
    span = vocab - ID_LOW
    return [ID_LOW + int.from_bytes(digest[8 * i:8 * i + 8], "little") % span
            for i in range(HASH_LEN)]


def random_nonce(vocab: int):
    def draw():
        span = vocab - ID_LOW
        return [ID_LOW + int.from_bytes(os.urandom(8), "little") % span
                for _ in range(NONCE_LEN)]
    return draw


# --------------------------------------------------------------------- building
def tokenize_rows(candidates, text: TextEncoder, config: DecisionsConfig,
                  state_max: int | None = None) -> tuple[TokenizedRequest, list[str]]:
    """`packing.tokenize`, minus marker ids inside content.

    Returns the tokenized request and the segments that had to be re-tokenized
    with special tokens split (empty for ordinary text)."""
    state_max = config.state_max_tokens if state_max is None else state_max
    req = tokenize(candidates, text, config, state_max)
    split: list[str] = []

    def clean(ids: list[int], source: str, cap: int, what: str) -> list[int]:
        if CONTENT_BREAKERS.isdisjoint(ids):
            return ids
        split.append(what)
        ids = _plain(text).encode(source, cap)
        if not CONTENT_BREAKERS.isdisjoint(ids):
            raise LayoutError(f"{what}: the tokenizer emits a marker id even with "
                              "special tokens split")
        return ids

    state = clean(req.state, candidates[0].state, state_max, "state")
    instructions: dict[str, str] = {}
    for cand in candidates:
        instructions.setdefault(cand.question_id, cand.instruction)
    questions = [clean(ids, instruction, config.instruction_max_tokens, f"question {qid}")
                 for ids, (qid, instruction) in zip(req.questions, instructions.items())]
    options = []
    for (qi, body), cand in zip(req.options, candidates):
        content = clean(body[:-1], cand.text, config.option_max_tokens - 1,
                        f"option {cand.question_id}/{cand.option_id}")
        options.append((qi, content + [body[-1]]))
    return TokenizedRequest(state, questions, options), split


def _plain(text: TextEncoder) -> TextEncoder:
    """A copy of `text` that tokenizes special-token strings as plain text (cached on it)."""
    plain = getattr(text, "_plain", None)
    if plain is None:
        plain = TextEncoder.__new__(TextEncoder)
        plain.tokenizer = text.tokenizer.__class__.from_str(text.tokenizer.to_str())
        plain.tokenizer.no_truncation()
        plain.tokenizer.no_padding()
        plain.tokenizer.encode_special_tokens = True
        text._plain = plain
    return plain


@dataclass
class Row:
    ids: list[int]
    candidates: list[int]          # candidate index of every option, in row order


def build_rows(req: TokenizedRequest, config: DecisionsConfig, vocab: int,
               nonce=None) -> list[Row]:
    """The rows `packing.build_rows` lays out for one request, as marker-delimited ids.

    The split is `layout_rows`' own, so a candidate travels in the same row as
    with `Judge` and `OnnxJudge`. `nonce()` returns NONCE_LEN ids; `None` draws
    them from `os.urandom`."""
    if not req.options:
        raise LayoutError("no candidates")
    nonce = nonce or random_nonce(vocab)
    head = [PAD] + state_hash(req.state, vocab) + list(req.state)
    rows, index = [], 0
    for layout in layout_rows(req, len(req.state), config.max_options_per_row,
                              config.max_row_tokens):
        ids = head + nonce() + [PAD]
        segments = sorted([(start, end, BOS) for start, end in layout.questions] +
                          [(start, end, MASK) for start, end, _, _ in layout.options])
        for start, end, marker in segments:
            ids += [marker] + layout.ids[start:end]
        n = len(layout.options)
        rows.append(Row(ids, list(range(index, index + n))))
        index += n
    return rows


# ---------------------------------------------------------------------- parsing
@dataclass
class Parsed:
    """The layout of the computed part of one row.

    The row's first `cached` tokens are not here: an earlier step computed them
    and their keys are read from the KV cache. They can only be the `<pad> H{4}`
    prefix (the first `cached_marks` of them), state tokens, and -- if a nonce id
    happened to repeat a cached row's -- the first `cached_tail` nonce ids. `valid`
    is False for ids that do not follow the grammar (vLLM's own warm-up rows, a
    broken client): the model then computes every token in isolation and the
    pooler returns NaN for the row, instead of the engine dying on one request.
    """
    cached: int
    cached_marks: int
    state_len: int
    kind: list[int]                # per computed token: STATE / QUESTION / OPTION / MARK
    seg: list[int]                 # block id; tokens attend within their block ...
    parent: list[int]              # ... and an option also to its question's block (-1: none)
    pos: list[int]                 # positions in the `layout_rows` layout
    options: list[tuple[int, int]] # [start, end) of every option's content, computed-token indices
    hash_ids: list[int] = field(default_factory=list)   # H{4}, if computed in this step
    state_ids: list[int] = field(default_factory=list)  # state tokens computed in this step
    cached_tail: int = 0           # nonce ids inside the cache hit
    valid: bool = True
    error: str = ""


def parse_row(ids: list[int], cached: int = 0) -> Parsed:
    """Read the grammar off the ids computed in this step, which begin at prompt index `cached`."""
    n = len(ids)
    kind, seg, parent, pos = [MARK] * n, [0] * n, [-1] * n, [0] * n
    options: list[tuple[int, int]] = []
    next_seg = 1
    cached_marks = min(cached, PREFIX)

    def marker(j):
        nonlocal next_seg
        kind[j], seg[j], pos[j] = MARK, next_seg, 0
        next_seg += 1

    i = 0
    # <pad> H{4}: whatever part of it was not cached
    if cached == 0 and (n == 0 or ids[0] != PAD):
        raise LayoutError("a row starts with <pad> and the state hash")
    for j in range(cached, PREFIX):
        if i >= n or (j > 0 and ids[i] in CONTENT_BREAKERS):
            raise LayoutError("state hash cut short")
        marker(i)
        i += 1
    hash_ids = ids[1:PREFIX] if cached == 0 else []
    # state N{4} <pad>: the separator is the first <pad>; the NONCE_LEN ids before
    # it are the nonce -- or fewer, when the cache hit took the first ones.
    sep = next((k for k in range(i, n) if ids[k] == PAD), None)
    if sep is None:
        raise LayoutError("no nonce and <pad> after the state")
    in_step = min(sep - i, NONCE_LEN)
    tail = NONCE_LEN - in_step
    if tail and cached < PREFIX + tail:
        raise LayoutError("nonce cut short")
    state_end = sep - in_step
    for k in range(i, state_end):
        if ids[k] in CONTENT_BREAKERS:
            raise LayoutError(f"marker {ids[k]} inside the state at {cached + k}")
        kind[k], seg[k], pos[k] = STATE, 0, cached + k - PREFIX
    state_ids = ids[i:state_end]
    state_len = cached + state_end - PREFIX - tail
    for k in range(state_end, sep):
        if ids[k] in CONTENT_BREAKERS:
            raise LayoutError(f"marker {ids[k]} inside the nonce at {cached + k}")
        marker(k)
    marker(sep)
    i = sep + 1
    if i >= n or ids[i] != BOS:
        raise LayoutError("a row must contain at least one <s> question")

    while i < n:
        if ids[i] != BOS:
            raise LayoutError(f"expected <s> at {cached + i}, got {ids[i]}")
        marker(i)
        i += 1
        q_seg = next_seg
        next_seg += 1
        q_len = 0
        while i < n and ids[i] not in CONTENT_BREAKERS:
            kind[i], seg[i], pos[i] = QUESTION, q_seg, state_len + q_len
            q_len += 1
            i += 1
        if i >= n or ids[i] != MASK:
            raise LayoutError(f"question at {cached + i} has no option")
        while i < n and ids[i] == MASK:
            marker(i)
            i += 1
            o_seg = next_seg
            next_seg += 1
            start, o_len = i, 0
            while i < n and ids[i] not in MARKERS:
                kind[i], seg[i], parent[i] = OPTION, o_seg, q_seg
                pos[i] = state_len + q_len + o_len
                o_len += 1
                i += 1
                if ids[i - 1] == EOS:
                    break
            if o_len == 0 or ids[i - 1] != EOS:
                raise LayoutError(f"option at {cached + start} does not end with </s>")
            options.append((start, i))
    return Parsed(cached, cached_marks, state_len, kind, seg, parent, pos, options,
                  hash_ids, state_ids, tail)


def parse_or_isolate(ids: list[int], cached: int = 0) -> Parsed:
    """`parse_row`, or for ids that break the grammar a layout where every token
    sees only itself and no cached key is read."""
    try:
        return parse_row(ids, cached)
    except LayoutError as error:
        n = len(ids)
        return Parsed(0, 0, 0, [MARK] * n, list(range(1, n + 1)), [-1] * n, [0] * n, [],
                      valid=False, error=str(error))


def isolated_state(n: int) -> Parsed:
    """A bare state of n tokens, full attention: the shape of vLLM's profiling runs."""
    return Parsed(0, 0, n, [STATE] * n, [0] * n, [-1] * n, list(range(n)), [])


def check_prompt(prompt: list[int], cached: int, vocab: int) -> tuple[Parsed | None, str]:
    """For the pooler: the computed part of a prompt parsed exactly as the model
    parsed it, after checking the whole prompt -- grammar, and that its state hash
    is the hash of its state. (None, reason) if either fails."""
    try:
        whole = parse_row(prompt, 0)
    except LayoutError as error:
        return None, str(error)
    if whole.hash_ids != state_hash(whole.state_ids, vocab):
        return None, "state hash does not match the state"
    part = parse_or_isolate(prompt[cached:], cached)
    if not part.valid:
        return None, f"computed part after a {cached}-token cache hit: {part.error}"
    if part.state_len != whole.state_len or len(part.options) != len(whole.options):
        return None, "cache hit disagrees with the prompt"
    return part, ""

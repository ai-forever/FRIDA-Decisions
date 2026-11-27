"""A request's tree as one vLLM prompt of token ids, and back.

vLLM hands a model token ids and nothing else per request, so the tree travels
inside them, as in fd's `vllm_backend/rows.py`. fd marks segments with marker
tokens of its own vocabulary and scans for them. A Qwen tokenizer's specials occur
in content, so here a header states the lengths and nothing is scanned:

    MAGIC  flags  n(tokens)  n(questions)  n(state)  [n(base)]
    ( n(question)  n(options)  n(option)+ )+
    state  ( question  option+ )+

`n(v)` is two ids, base 2^15, so that a header id stays inside the vocabulary (vLLM
validates ids) and a state can hold millions of tokens. The tokens follow in
`plan.segments` order, which is the renderer's, and each is computed once.

Positions are not sent for text: they are contiguous along state -> question -> option,
which is what `CausalScalarRenderer.render` writes. A state with images (`FLAG_IMAGES`)
holds the model's image tokens in its ids and has M-RoPE rows `(3, n)` for positions;
the text after it starts at `base`, their max + 1, which the header carries. The pixels,
the patch grid and those rows travel beside the ids (`server.media_params`, vLLM's
pooling params), and `decode` takes them back as `media`.

No torch and no vLLM in this file.
"""
from __future__ import annotations

from frida_decisions.v2.model.layout import ScoringPlan, Segment

MAGIC = 30011
BASE = 1 << 15
FLAG_MEAN = 1                      # the head reads an option's mean instead of its last token
FLAG_IMAGES = 2                    # a state with images: media beside the ids, n(base) in the header
HEAD = 8                           # MAGIC, flags, and three two-id numbers


class RowError(ValueError):
    """A plan this format cannot carry, or ids that do not follow it."""


def _put(out: list[int], value: int) -> None:
    out.append(value >> 15)
    out.append(value & (BASE - 1))


def _contiguous(positions, start, stop):
    expected = range(start, stop)
    return positions == expected if isinstance(positions, range) else list(positions) == list(expected)


def encode(plan: ScoringPlan) -> list[int]:
    """The prompt of one plan. Raises `RowError` if it is not state -> question -> option
    with contiguous positions after the state, in the order `decode` rebuilds."""
    segments = plan.segments
    if not segments or segments[0].parent is not None or segments[0].kind != "state":
        raise RowError("the first segment must be the state")
    state = segments[0]
    images = bool(plan.mm)
    if images:
        if tuple(getattr(state.pos, "shape", ())) != (3, len(state.ids)):
            raise RowError("a state with images needs (3, n) positions")
        s = int(state.pos.max()) + 1
    else:
        if not _contiguous(state.pos, 0, len(state.ids)):
            raise RowError("the state's positions must count from 0")
        s = len(state.ids)
    mode = None
    header: list[int] = []
    questions = 0
    slot = 0
    i, n = 1, len(segments)
    tokens = list(state.ids)
    while i < n:
        question = segments[i]
        if question.kind != "question" or question.parent != 0:
            raise RowError(f"segment {i}: a question under the state expected")
        q = len(question.ids)
        if not _contiguous(question.pos, s, s + q):
            raise RowError(f"segment {i}: positions must follow the state's")
        tokens += question.ids
        q_index, i = i, i + 1
        lengths = []
        while i < n and segments[i].parent == q_index:
            option = segments[i]
            if option.kind != "option" or option.readout is None or option.readout[0] != slot:
                raise RowError(f"segment {i}: an option read at slot {slot} expected")
            if mode is None:
                mode = option.readout[1]
            elif mode != option.readout[1]:
                raise RowError("one readout per request")
            if not _contiguous(option.pos, s + q, s + q + len(option.ids)):
                raise RowError(f"segment {i}: positions must follow the question's")
            tokens += option.ids
            lengths.append(len(option.ids))
            slot += 1
            i += 1
        if not lengths:
            raise RowError(f"question at segment {q_index} has no option")
        _put(header, q)
        _put(header, len(lengths))
        for length in lengths:
            _put(header, length)
        questions += 1
    if not questions:
        raise RowError("a request needs a question")
    if slot != plan.n_slots:
        raise RowError(f"{slot} options for {plan.n_slots} slots")
    first = [MAGIC, (FLAG_MEAN if mode == "mean" else 0) | (FLAG_IMAGES if images else 0)]
    _put(first, len(tokens))
    _put(first, questions)
    _put(first, len(state.ids))
    if images:
        _put(first, s)
    return first + header + tokens


def decode(ids: list[int], media: dict | None = None) -> ScoringPlan:
    """The plan of one whole prompt. Raises `RowError` for anything else, such as the
    rows of vLLM's own warm-up. A prompt with images needs `media`: `pixel_values`,
    `image_grid_thw` and the state's `positions` `(3, n)`, which become `plan.mm` and the
    state's positions."""
    size = len(ids)
    if size < HEAD or ids[0] != MAGIC:
        raise RowError("no header")
    flags = ids[1]
    if flags & ~(FLAG_MEAN | FLAG_IMAGES):
        raise RowError(f"flags {flags}")
    mode = "mean" if flags & FLAG_MEAN else "last"
    images = bool(flags & FLAG_IMAGES)
    total = ids[2] << 15 | ids[3]
    questions = ids[4] << 15 | ids[5]
    state_len = ids[6] << 15 | ids[7]
    cursor = HEAD
    s = state_len
    if images:
        if size < HEAD + 2:
            raise RowError("no header")
        s = ids[8] << 15 | ids[9]
        cursor += 2
        if media is None or "positions" not in media:
            raise RowError("a prompt with images came without its media")
        if tuple(media["positions"].shape) != (3, state_len) or int(media["positions"].max()) + 1 != s:
            raise RowError("the media's positions do not match the state")
    elif media:
        raise RowError("media for a prompt without images")
    at = size - total                              # the first token, and past the header's end
    if at < cursor or not state_len or not questions or state_len > total:
        raise RowError("bad lengths")
    token = at + state_len
    state_pos = media["positions"] if images else range(state_len)
    segments = [Segment(None, ids[at:token], state_pos, "state")]
    group: list[int] = []
    slot = 0
    for g in range(questions):
        if cursor + 4 > at:
            raise RowError("header cut short")
        q = ids[cursor] << 15 | ids[cursor + 1]
        options = ids[cursor + 2] << 15 | ids[cursor + 3]
        cursor += 4
        if cursor + 2 * options > at or not options:
            raise RowError("header cut short")
        q_index = len(segments)
        end = token + q
        segments.append(Segment(0, ids[token:end], range(s, s + q), "question"))
        token = end
        offset = s + q
        for _ in range(options):
            n = ids[cursor] << 15 | ids[cursor + 1]
            cursor += 2
            end = token + n
            segments.append(Segment(q_index, ids[token:end], range(offset, offset + n), "option",
                                    readout=(slot, mode)))
            token = end
            group.append(g)
            slot += 1
    if cursor != at or token != size:
        raise RowError("lengths do not add up")
    mm = {"pixel_values": media["pixel_values"], "image_grid_thw": media["image_grid_thw"]} if images else {}
    if images and "rgb8_table" in media:
        mm["rgb8_table"] = media["rgb8_table"]
    return ScoringPlan(segments, slot, group, questions, mm=mm)

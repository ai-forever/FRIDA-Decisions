"""The Jev (System One) contract: Choice / Score / Noul, compiled to reranker calls.

The wire format is KaLM-Jev's, field for field, so that their `examples/*.json`
run against us unchanged and their published `results/*.json` can be diffed
against ours. What differs is underneath: KaLM-Reranker is an encoder-decoder
that reads a candidate in the encoder, the state in the decoder, and scores by
the gap between the `yes` and `no` logits at the last decoder position. FRIDA is
`T5EncoderModel` -- no decoder, no LM head, no yes/no readout. Our margin comes
from a scoring token appended to the candidate, through a linear head.

Three consequences, all of them visible from here:

1. **CLS cannot be the readout.** In the cached arrangement the state's key/value
   pairs are computed once and reused for every candidate, so position 0 -- which
   lives inside the state -- is byte-identical across candidates. A score read
   there would be the same number K times. `SCORE_TOKEN` marks where to read
   instead, at the end of each candidate.

2. **`noul` without `criteria` has no direct analogue.** KaLM scores the state as
   its own sole Document and converts the native yes/no margin with a sigmoid.
   With no candidate there is nothing to read a scalar from, so we desugar the
   no-criteria path into the two-candidate path using their own default texts.
   The API is identical; the cost is two passes instead of one, and our numbers
   on that branch are not expected to match theirs. This is the only deliberate
   departure from template `v3-noul-state-document`.

3. **The adapters are ours.** Adapter text is internal to the model, not part of
   the wire format, so switching it costs no compatibility. FRIDA is a Russian
   model and gets Russian adapters by default; `ADAPTERS_EN` holds KaLM's
   English originals for the compatibility runner, where matching their prompt
   matters more than playing to the model's strength.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from frida_decisions.v2.model.hf_v2.decision_confidence import confidence

TEMPLATE_VERSION = "v1-frida-candidate-readout"

# Marks the position the scalar is read from. It closes the candidate block, so
# under the block mask it has seen the whole state and the whole candidate and
# nothing of the other candidates.
SCORE_TOKEN = "<score>"

ADAPTERS_RU = {
    "choice": (
        "Оцени, является ли вариант, описанный в кандидате, "
        "подходящим ответом на вопрос выше, с учётом сведений в запросе. "
        "Суди о варианте по его описанию, а не по одной лишь тематической близости."
    ),
    "score": (
        "Оцени, точно ли уровень, описанный в кандидате, "
        "характеризует запрос в рамках оценки выше. "
        "Суди об уровне по его описанию, а не по одной лишь тематической близости."
    ),
    "noul": (
        "Оцени, верно ли кандидат описывает ответ на вопрос выше, "
        "с учётом сведений в запросе. "
        "Отвечай утвердительно, если этот вариант выполняется, иначе отрицательно."
    ),
    "ranking": (
        "Оцени, насколько кандидат подходит под вопрос выше с учётом сведений "
        "в запросе. Кандидат — это сам материал, а не описание варианта: "
        "суди по его содержанию, а не по одной лишь тематической близости."
    ),
}

ADAPTERS_EN = {
    "choice": (
        "Evaluate whether the candidate option described in the Document "
        "is an appropriate answer to the question above, "
        "given the information in the Query. "
        "Judge the candidate by its stated description, "
        "not merely by topical relevance."
    ),
    "score": (
        "Evaluate whether the level described in the Document "
        "accurately characterizes the Query with respect to the "
        "assessment above. "
        "Judge this level by its stated criteria, "
        "not merely by topical relevance."
    ),
    "noul": (
        "Given the Query, evaluate whether the candidate criterion in the "
        "Document correctly describes the answer to the question above. "
        "Answer yes if this candidate criterion is satisfied, otherwise no."
    ),
    # Ours, not KaLM's: KaLM-Jev has no ranking type.
    "ranking": (
        "Given the Query, evaluate how well the candidate in the Document "
        "satisfies the question above. The candidate is the material itself, "
        "not a description of an option: judge it by its content, "
        "not merely by topical relevance."
    ),
}

# KaLM's own defaults, kept verbatim: the no-criteria path desugars into these
# rather than into a translation of them, so a request that omits criteria is
# scored against the same text on both sides.
NOUL_DEFAULT_CRITERIA = {
    "true": "The answer to the question is yes.",
    "false": "The answer to the question is no.",
}

# `ranking` is ours; the first three are KaLM-Jev's wire format field for field.
#
# Why a fourth type rather than `choice` with long options (owner, 2026-09-25).
# A `choice` option is a CRITERION -- a label and its description, rendered as
# `"<id>: <description>"`, one of which is the answer. A ranking candidate is
# CONTENT -- a passage, an agent action, a document -- and the question is a
# criterion to order them by. The difference shows in three places:
#
# * the candidate is rendered bare: an id like `doc-4412` is noise to the
#   model, where a choice id like `refund` is half the meaning;
# * the answer is an order, not a pick -- `ranking` lists every id best first
#   and `scores` keeps the margins, so a caller can take top-k or cut;
# * the gold may name several right candidates, and the training loss is the
#   multi-positive listwise one rather than a single-target softmax.
#
# The layout is the critic's: context in the state, candidates in the
# options. It is the opposite of the pair-in-state layout TypeSafe's RAG
# cookbook uses, which asks four yes/no questions per (query, passage) pair --
# right for classifying a passage, and K requests with no competition between
# candidates when the job is to order them. Packing is identical to `choice`
# (options never see each other, positions restart per option), so a
# candidate's score does not depend on which others arrived with it.
TYPES = ("choice", "score", "noul", "ranking")


class ProtocolError(Exception):
    """Carries the wire error shape: type, message, and the offending field."""

    def __init__(self, status, message, field=None, kind="invalid_request"):
        super().__init__(message)
        self.status, self.message, self.field, self.kind = status, message, field, kind

    def payload(self):
        error = {"type": self.kind, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}


def render_content(value):
    """Strings pass through; everything else becomes compact, sorted JSON.

    Sorted keys and no whitespace make the serialisation canonical, which is
    what lets an encoded state or candidate be looked up in a cache by its text.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------- sessions
# One serialisation of dialogues and agent traces inside a state, the same for
# every backbone and never a model's own chat template:
#
#     state = {"policy": ..., "tools": [...], "messages": [...]}
#     state = {"sessions": [{"id": "1", "messages": [...]}, ...]}
#     message = {"role": "system"|"user"|"assistant"|"tool", "content": str|null,
#                "name": str?, "tool_calls": [{"name": str, "arguments": object}]?}
#
# OpenAI's tool-call form ({"type": "function", "function": {"name", "arguments":
# "<json>"}}) is read as well. The text: context fields first (CONTEXT_FIELDS in
# that order, then the others), then one numbered header per message --
# "[3] user", "[4] assistant (WebSurfer)", "[6] tool search_orders" -- its
# content on the following lines as written, an assistant's tool calls as
# "→ name({...})" lines, and "=== session 2 ===" before each session. The
# numbers run across sessions, so a question can name "шаг [7]".
#
# Content cannot forge that structure: a content line that starts like a header,
# a session line or a call line gets a "¦" in front, and "<|x|>" (a chat-format
# special token written as text) becomes "<¦x¦>".

SESSION_KEYS = ("messages", "sessions")
CONTEXT_FIELDS = ("policy", "goal", "task", "tools")
_SPECIAL_TOKEN = re.compile(r"<\|([^|<>\s]{1,64})\|>")
_STRUCTURAL_LINE = re.compile(r"^(?=\[\d+\] |=== |→ )", re.MULTILINE)


def neutralise(text: str) -> str:
    """`<|x|>` written as text becomes `<¦x¦>`: never a special token of the model."""
    return _SPECIAL_TOKEN.sub(r"<¦\1¦>", text)


def is_session(value) -> bool:
    return isinstance(value, dict) and any(isinstance(value.get(k), list) for k in SESSION_KEYS)


def _content(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):                     # OpenAI content parts
        parts = []
        for part in value:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                parts.append(f"[{part.get('type', 'part')}]")
            else:
                parts.append(str(part))
        value = "\n".join(parts)
    elif not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return _STRUCTURAL_LINE.sub("¦", neutralise(value))


def _tool_call(call: dict) -> str:
    if isinstance(call.get("function"), dict):      # OpenAI: arguments are a JSON string
        name = call["function"].get("name", "")
        args = call["function"].get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
    else:
        name, args = call.get("name", ""), call.get("arguments")
    rendered = args if isinstance(args, str) else json.dumps(
        args if args is not None else {}, ensure_ascii=False, separators=(", ", ": "))
    return f"→ {neutralise(str(name))}({neutralise(rendered)})"


def session_text(state: dict) -> str:
    """A state with `messages` or `sessions` as text (the format above)."""
    blocks = []
    keys = [k for k in CONTEXT_FIELDS if k in state]
    keys += [k for k in state if k not in CONTEXT_FIELDS and k not in SESSION_KEYS]
    for key in keys:
        value = state[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        blocks.append(f"{neutralise(str(key))}:\n{_content(text)}")
    n = 0

    def messages(items):
        nonlocal n
        for m in items:
            n += 1
            role = str(m.get("role", "user"))
            head = f"[{n}] {role}"
            if role == "tool":
                head += f" {neutralise(str(m.get('name', '')))}".rstrip()
            elif m.get("name"):
                head += f" ({neutralise(str(m['name']))})"
            lines = [head]
            body = _content(m.get("content"))
            if body:
                lines.append(body)
            lines += [_tool_call(c) for c in m.get("tool_calls") or []]
            blocks.append("\n".join(lines))

    for i, session in enumerate(state.get("sessions") or []):
        head = [f"=== session {neutralise(str(session.get('id', i + 1)))} ==="]
        for key, value in session.items():          # e.g. its date
            if key not in ("id", "messages"):
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                head.append(f"{neutralise(str(key))}: {_content(text)}")
        blocks.append("\n".join(head))
        messages(session.get("messages") or [])
    messages(state.get("messages") or [])
    return "\n\n".join(blocks)


def upgrade_state(state):
    """The state as a session renderer reads it: a session dict (or its JSON
    text) becomes `session_text`, anything else is left as it is. A bench can
    pass the result to any model as a plain string state."""
    if isinstance(state, str) and state.startswith("{"):
        try:
            parsed = json.loads(state)
        except ValueError:
            return state
        return session_text(parsed) if is_session(parsed) else state
    return session_text(state) if is_session(state) else state


@dataclass(frozen=True)
class Question:
    type: str
    instructions: object
    criteria: object = None


@dataclass(frozen=True)
class Request:
    state: object
    questions: dict


@dataclass(frozen=True)
class Task:
    """One scalar to produce: the state judged against one candidate."""
    question_id: str
    option_id: str
    instruction: str
    state: str
    candidate: str


def parse_request(raw):
    """Validate a decoded request body and reject everything Jev rejects."""
    if not isinstance(raw, dict):
        raise ProtocolError(400, "Request must be an object")
    unknown = set(raw) - {"state", "questions", "model"}
    if unknown:
        raise ProtocolError(400, f"Unknown field: {sorted(unknown)[0]}", sorted(unknown)[0])
    if "state" not in raw:
        raise ProtocolError(400, "Missing state", "state")
    if not isinstance(raw.get("questions"), dict) or not raw["questions"]:
        raise ProtocolError(400, "questions must be a non-empty object", "questions")

    questions = {}
    for qid, q in raw["questions"].items():
        where = f"questions.{qid}"
        if not qid:
            raise ProtocolError(400, "Question id must not be empty", "questions")
        if not isinstance(q, dict):
            raise ProtocolError(400, "Question must be an object", where)
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            raise ProtocolError(400, f"Unknown field: {sorted(extra)[0]}",
                           f"{where}.{sorted(extra)[0]}")
        if q.get("type") not in TYPES:
            raise ProtocolError(400, f"type must be one of {TYPES}", f"{where}.type")
        if "instructions" not in q:
            raise ProtocolError(400, "Missing instructions", f"{where}.instructions")
        questions[qid] = Question(q["type"], q["instructions"],
                                  _criteria(q.get("criteria"), q["type"], where))
    return Request(raw["state"], questions)


def _criteria(criteria, qtype, where):
    field = f"{where}.criteria"
    if qtype == "ranking":
        # An object keeps the caller's ids (docids, action ids) on the way back;
        # a bare list is accepted and numbered from zero.
        if isinstance(criteria, list):
            criteria = {str(i): value for i, value in enumerate(criteria)}
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ProtocolError(400, "ranking needs at least two candidates", field)
        if any(not key for key in criteria):
            raise ProtocolError(400, "Candidate id must not be empty", field)
        if any(value is None for value in criteria.values()):
            raise ProtocolError(400, "Candidate must not be null", field)
        return dict(criteria)
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ProtocolError(400, "choice needs an object of at least two options", field)
        if any(not key for key in criteria):
            raise ProtocolError(400, "Option id must not be empty", field)
        return dict(criteria)
    if qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ProtocolError(400, "score needs a list of at least two levels", field)
        if any(value is None for value in criteria):
            raise ProtocolError(400, "Level description must not be null", field)
        return list(criteria)
    if criteria is None:
        return None
    if not isinstance(criteria, dict) or not ({"true", "false"} & set(criteria)):
        raise ProtocolError(400, "noul criteria need at least one of true / false", field)
    if set(criteria) - {"true", "false"}:
        raise ProtocolError(400, "noul criteria accept only true and false", field)
    if any(value is None for value in criteria.values()):
        raise ProtocolError(400, "noul criterion must not be null", field)
    return dict(criteria)


# ------------------------------------------------------------------ content
# What the model reads, before any renderer turns it into tokens. Text only for
# now; a media part (image / audio / video, with a sha256 cache key) is the next
# member of `Part` and will arrive through a new wire field, never through
# `state` (see docs/design/BACKBONE_AND_MODALITIES.md).
@dataclass(frozen=True)
class TextPart:
    text: str


Part = TextPart                       # later: TextPart | MediaPart
Content = tuple                       # tuple[Part, ...]


def to_content(value) -> Content:
    """Any wire value as content: one text part, byte-identical to `render_content`."""
    return (TextPart(render_content(value)),)


def content_text(content: Content) -> str:
    """The text of a text-only content. Raises on a part a text renderer cannot read."""
    out = []
    for part in content:
        if not isinstance(part, TextPart):
            raise TypeError(f"{type(part).__name__} is not text; this renderer reads text only")
        out.append(part.text)
    return "".join(out)


@dataclass(frozen=True)
class Option:
    """One candidate of a question, as the model reads it."""
    option_id: str
    content: Content


@dataclass(frozen=True)
class CQuestion:
    """One compiled question.

    `instructions` is the caller's text alone; `adapter` is the per-type text the
    T5 path appends to it (`None` when the question was rebuilt from flat tasks,
    whose instruction already has it). `instruction` is what today's model reads.
    """
    question_id: str
    qtype: str | None
    instructions: Content
    adapter: str | None
    options: tuple

    @property
    def instruction(self) -> Content:
        if self.adapter is None:
            return self.instructions
        return (TextPart(content_text(self.instructions) + "\n\n" + self.adapter),)


@dataclass(frozen=True)
class Compiled:
    """A request after validation: one state, questions in order, options in order.

    `flat()` is the margin order: one `Task` per option, question by question --
    exactly what `compile_request` returns and `aggregate` consumes.
    """
    state: Content
    questions: tuple

    def flat(self) -> list[Task]:
        state = content_text(self.state)
        return [Task(q.question_id, o.option_id, content_text(q.instruction), state,
                     content_text(o.content))
                for q in self.questions for o in q.options]

    @property
    def n_options(self) -> int:
        return sum(len(q.options) for q in self.questions)


def compile_structured(request, adapters=None) -> Compiled:
    """Compile a parsed request (from `parse_request`) into `Compiled`.

    Question order and candidate order are preserved. Question ids never reach
    the model; they only delimit questions (a question boundary is a change of
    `question_id`, never of instruction text).
    """
    adapters = adapters or ADAPTERS_RU
    questions = []
    for qid, question in request.questions.items():
        if question.type == "choice":
            candidates = [(key, key if value is None else key + ": " + render_content(value))
                          for key, value in question.criteria.items()]
        elif question.type == "ranking":
            candidates = [(key, render_content(value))
                          for key, value in question.criteria.items()]
        elif question.type == "score":
            candidates = [(str(i), render_content(value))
                          for i, value in enumerate(question.criteria)]
        else:
            # Both noul paths end up here: polarity lives in option_id, never in
            # the candidate text, so the two encodings stay comparable.
            supplied = question.criteria or {}
            candidates = [(key, render_content(supplied.get(key, NOUL_DEFAULT_CRITERIA[key])))
                          for key in ("true", "false")]
        questions.append(CQuestion(
            qid, question.type, to_content(question.instructions), adapters[question.type],
            tuple(Option(key, (TextPart(text),)) for key, text in candidates)))
    return Compiled(to_content(request.state), tuple(questions))


def compiled_from_tasks(tasks: list[Task]) -> Compiled:
    """Rebuild the structure of a flat task list (the inverse of `Compiled.flat`).

    A new question starts wherever `question_id` changes. The state is the first
    task's, as it always was for packing. Question types are not in a `Task`, so
    they come back as `None`.
    """
    if not tasks:
        raise ValueError("a request compiles to at least one task")
    questions, current, options = [], None, []

    def close():
        if current is not None:
            questions.append(CQuestion(current.question_id, None, (TextPart(current.instruction),),
                                       None, tuple(options)))

    for task in tasks:
        if current is None or task.question_id != current.question_id:
            close()
            current, options = task, []
        options.append(Option(task.option_id, (TextPart(task.candidate),)))
    close()
    return Compiled((TextPart(tasks[0].state),), tuple(questions))


def compile_request(request, adapters=None) -> list[Task]:
    """Flatten a request into the list of scalars the model has to produce.

    Kept as the list every existing caller expects; `compile_structured` is the
    same compilation with the question / option structure kept.
    """
    return compile_structured(request, adapters).flat()


@dataclass(frozen=True)
class Calibration:
    choice_temperature: float = 1.0
    score_temperature: float = 1.0
    noul_a: float = 1.0
    noul_b: float = 0.0

    def __post_init__(self):
        values = (self.choice_temperature, self.score_temperature, self.noul_a, self.noul_b)
        if not all(math.isfinite(x) for x in values) or min(values[:3]) <= 0:
            raise ValueError("Temperatures and noul_a must be positive and finite")

    @property
    def status(self):
        return "none" if self == Calibration() else "manual"


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def softmax(values, temperature=1.0):
    top = max(values)
    weights = [math.exp((x - top) / temperature) for x in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def aggregate(request, tasks, margins, calibration=None):
    """Turn one scalar per task into one answer per question."""
    calibration = calibration or Calibration()
    if len(tasks) != len(margins) or not all(math.isfinite(x) for x in margins):
        raise RuntimeError("Backend returned invalid margins")

    grouped = {qid: [] for qid in request.questions}
    for task, margin in zip(tasks, margins):
        grouped[task.question_id].append((task.option_id, margin))

    answers = {}
    for qid, question in request.questions.items():
        candidates = grouped[qid]
        if question.type == "noul":
            by_label = dict(candidates)
            if len(candidates) != 2 or set(by_label) != {"true", "false"}:
                raise RuntimeError("noul requires exactly one true and one false margin")
            # At a=1, b=0 this is softmax([z_true, z_false])[0].
            margin = by_label["true"] - by_label["false"]
            answers[qid] = {"type": "noul",
                            "noul": sigmoid(calibration.noul_a * margin + calibration.noul_b)}
            continue

        temperature = (calibration.score_temperature if question.type == "score"
                       else calibration.choice_temperature)
        p = softmax([m for _, m in candidates], temperature)
        keys = (list(question.criteria) if question.type in ("choice", "ranking")
                else [str(i) for i in range(len(p))])
        if question.type == "ranking":
            # Best first; equal margins keep the caller's order, so the answer
            # is deterministic. `probabilities` is the share of one unit among
            # these candidates -- NOT the probability that a candidate is
            # relevant, which a softmax cannot say when several are.
            order = sorted(range(len(p)), key=lambda i: (-candidates[i][1], i))
            answers[qid] = {"type": "ranking",
                            "ranking": [keys[i] for i in order],
                            "scores": {keys[i]: candidates[i][1] for i in range(len(p))},
                            "probabilities": dict(zip(keys, p)),
                            "confidence": confidence(p)}
            continue
        answer = {"type": question.type,
                  "probabilities": dict(zip(keys, p)),
                  "confidence": confidence(p)}
        if question.type == "choice":
            answer["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
        else:
            answer["score"] = math.fsum(i * value for i, value in enumerate(p))
            answer["legend"] = {str(i): render_content(v)
                                for i, v in enumerate(question.criteria)}
        answers[qid] = answer
    return answers

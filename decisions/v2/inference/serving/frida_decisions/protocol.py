"""The request format: validation, compilation to candidates, aggregation to answers.

A request is a `state` (the text, or any JSON value, being judged) and a set of
named `questions`. Each question is compiled into a list of candidates; the
model produces one scalar margin per candidate, and the margins of a question
are turned into its answer here. Nothing in this module touches a model, so
every backend shares it.

    {"state": "...",
     "questions": {
        "topic":   {"type": "choice",  "instructions": "...", "criteria": {"id": "description", ...}},
        "urgency": {"type": "score",   "instructions": "...", "criteria": ["level 0", "level 1", ...]},
        "refund":  {"type": "noul",    "instructions": "..."},              # yes/no
        "best":    {"type": "ranking", "instructions": "...", "criteria": {"doc-1": "passage", ...}}}}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .config import DEFAULT_INSTRUCTION_SUFFIXES, DEFAULT_YES_NO_CRITERIA
from .constants import NO_LABEL, YES_LABEL, QuestionType

CHOICE, SCORE, YES_NO, RANKING = (QuestionType.CHOICE, QuestionType.SCORE,
                                  QuestionType.YES_NO, QuestionType.RANKING)


class RequestError(ValueError):
    """An invalid request. Carries an HTTP-style status and the offending field."""

    def __init__(self, message: str, field: str | None = None, status: int = 400,
                 kind: str = "invalid_request"):
        super().__init__(message)
        self.message, self.field, self.status, self.kind = message, field, status, kind

    def payload(self) -> dict:
        error = {"type": self.kind, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}


def render_content(value) -> str:
    """Strings pass through; any other JSON value becomes compact, key-sorted JSON.

    The serialisation is canonical, so the same object always renders to the
    same text (and therefore to the same tokens and the same cache key).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Question:
    type: str
    instructions: object
    criteria: object = None


@dataclass(frozen=True)
class Request:
    state: object
    questions: dict          # question id -> Question, in request order


@dataclass(frozen=True)
class Candidate:
    """One scalar to produce: the state judged against one option of one question."""
    question_id: str
    option_id: str
    instruction: str         # question instructions + the type's suffix
    state: str
    text: str                # the option as the model reads it


# ----------------------------------------------------------------- parsing
def parse_request(raw) -> Request:
    """Validate a decoded request body. Raises `RequestError` on anything malformed."""
    if not isinstance(raw, dict):
        raise RequestError("Request must be an object")
    unknown = set(raw) - {"state", "questions", "model"}
    if unknown:
        name = sorted(unknown)[0]
        raise RequestError(f"Unknown field: {name}", name)
    if "state" not in raw:
        raise RequestError("Missing state", "state")
    if not isinstance(raw.get("questions"), dict) or not raw["questions"]:
        raise RequestError("questions must be a non-empty object", "questions")

    questions = {}
    for qid, q in raw["questions"].items():
        where = f"questions.{qid}"
        if not qid:
            raise RequestError("Question id must not be empty", "questions")
        if not isinstance(q, dict):
            raise RequestError("Question must be an object", where)
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            name = sorted(extra)[0]
            raise RequestError(f"Unknown field: {name}", f"{where}.{name}")
        if q.get("type") not in QuestionType.ALL:
            raise RequestError(f"type must be one of {QuestionType.ALL}", f"{where}.type")
        if "instructions" not in q:
            raise RequestError("Missing instructions", f"{where}.instructions")
        questions[qid] = Question(q["type"], q["instructions"],
                                  _criteria(q.get("criteria"), q["type"], where))
    return Request(raw["state"], questions)


def _criteria(criteria, qtype: str, where: str):
    field = f"{where}.criteria"
    if qtype == RANKING:
        # An object keeps the caller's ids (document ids, action ids) on the way
        # back; a bare list is accepted and numbered from zero.
        if isinstance(criteria, list):
            criteria = {str(i): value for i, value in enumerate(criteria)}
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise RequestError("ranking needs at least two candidates", field)
        if any(not key for key in criteria):
            raise RequestError("Candidate id must not be empty", field)
        if any(value is None for value in criteria.values()):
            raise RequestError("Candidate must not be null", field)
        return dict(criteria)
    if qtype == CHOICE:
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise RequestError("choice needs an object of at least two options", field)
        if any(not key for key in criteria):
            raise RequestError("Option id must not be empty", field)
        return dict(criteria)
    if qtype == SCORE:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise RequestError("score needs a list of at least two levels", field)
        if any(value is None for value in criteria):
            raise RequestError("Level description must not be null", field)
        return list(criteria)
    # yes/no: criteria are optional
    if criteria is None:
        return None
    labels = {YES_LABEL, NO_LABEL}
    if not isinstance(criteria, dict) or not (labels & set(criteria)):
        raise RequestError(f"{qtype} criteria need at least one of {YES_LABEL} / {NO_LABEL}", field)
    if set(criteria) - labels:
        raise RequestError(f"{qtype} criteria accept only {YES_LABEL} and {NO_LABEL}", field)
    if any(value is None for value in criteria.values()):
        raise RequestError(f"{qtype} criterion must not be null", field)
    return dict(criteria)


# ----------------------------------------------------------------- compiling
def compile_request(request: Request, suffixes: dict | None = None,
                    yes_no_defaults: dict | None = None) -> list[Candidate]:
    """Flatten a request into the candidates the model has to score.

    Question order and option order are preserved. Question ids never reach the
    model: only the instruction, the state and one option text do.
    """
    suffixes = suffixes or DEFAULT_INSTRUCTION_SUFFIXES
    yes_no_defaults = yes_no_defaults or DEFAULT_YES_NO_CRITERIA
    state = render_content(request.state)
    out = []
    for qid, question in request.questions.items():
        instruction = render_content(question.instructions) + "\n\n" + suffixes[question.type]
        if question.type == CHOICE:
            # A choice option is a label plus its description.
            options = [(key, key if value is None else key + ": " + render_content(value))
                       for key, value in question.criteria.items()]
        elif question.type == RANKING:
            # A ranking candidate is content; its id is not shown to the model.
            options = [(key, render_content(value)) for key, value in question.criteria.items()]
        elif question.type == SCORE:
            options = [(str(i), render_content(value)) for i, value in enumerate(question.criteria)]
        else:
            # yes/no is scored as two options; the polarity lives in the option
            # id, never in the text, so both sides are encoded the same way.
            supplied = question.criteria or {}
            options = [(key, render_content(supplied.get(key, yes_no_defaults[key])))
                       for key in (YES_LABEL, NO_LABEL)]
        out.extend(Candidate(qid, key, instruction, state, text) for key, text in options)
    return out


# ----------------------------------------------------------------- aggregation
@dataclass(frozen=True)
class Calibration:
    """Optional post-hoc calibration. The defaults leave the model's margins as they are."""
    choice_temperature: float = 1.0
    score_temperature: float = 1.0
    yes_no_scale: float = 1.0
    yes_no_bias: float = 0.0

    def __post_init__(self):
        values = (self.choice_temperature, self.score_temperature,
                  self.yes_no_scale, self.yes_no_bias)
        if not all(math.isfinite(x) for x in values) or min(values[:3]) <= 0:
            raise ValueError("Temperatures and yes_no_scale must be positive and finite")


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def softmax(values, temperature: float = 1.0) -> list[float]:
    top = max(values)
    weights = [math.exp((x - top) / temperature) for x in values]
    total = math.fsum(weights)
    return [w / total for w in weights]


def confidence(probabilities) -> float:
    """One minus the normalised entropy: 1 when certain, 0 when uniform."""
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))


def aggregate(request: Request, candidates: list[Candidate], margins: list[float],
              calibration: Calibration | None = None) -> dict:
    """Turn one margin per candidate into one answer per question."""
    calibration = calibration or Calibration()
    if len(candidates) != len(margins) or not all(math.isfinite(x) for x in margins):
        raise RuntimeError("Backend returned invalid margins")

    grouped = {qid: [] for qid in request.questions}
    for cand, margin in zip(candidates, margins):
        grouped[cand.question_id].append((cand.option_id, margin))

    answers = {}
    for qid, question in request.questions.items():
        scored = grouped[qid]
        if question.type == YES_NO:
            by_label = dict(scored)
            # With the default calibration this is softmax([m_yes, m_no])[0].
            margin = by_label[YES_LABEL] - by_label[NO_LABEL]
            answers[qid] = {"type": YES_NO,
                            YES_NO: sigmoid(calibration.yes_no_scale * margin + calibration.yes_no_bias)}
            continue

        temperature = (calibration.score_temperature if question.type == SCORE
                       else calibration.choice_temperature)
        p = softmax([m for _, m in scored], temperature)
        keys = (list(question.criteria) if question.type in (CHOICE, RANKING)
                else [str(i) for i in range(len(p))])
        if question.type == RANKING:
            # Best first; equal margins are ordered by key (see `best_key`).
            # `probabilities` is a share of one unit among these candidates, not
            # the probability that a candidate is relevant (several may be).
            order = sorted(range(len(p)), key=lambda i: (-scored[i][1], keys[i]))
            answers[qid] = {"type": RANKING,
                            RANKING: [keys[i] for i in order],
                            "scores": {keys[i]: scored[i][1] for i in range(len(p))},
                            "probabilities": dict(zip(keys, p)),
                            "confidence": confidence(p)}
            continue
        answer = {"type": question.type,
                  "probabilities": dict(zip(keys, p)),
                  "confidence": confidence(p)}
        if question.type == CHOICE:
            answer[CHOICE] = best_key(keys, [m for _, m in scored])
        else:
            # The expected level under the distribution, on the 0..N-1 scale.
            answer[SCORE] = math.fsum(i * value for i, value in enumerate(p))
            answer["legend"] = {str(i): render_content(v) for i, v in enumerate(question.criteria)}
        answers[qid] = answer
    return answers


def best_key(keys, values) -> str:
    """The key with the largest value; an exact tie goes to the lexicographically
    smallest key (compared as strings), so a decision never depends on the order
    in which the options were listed."""
    top = max(values)
    return min(str(k) for k, v in zip(keys, values) if v == top)


def decision(answer: dict):
    """The headline of an answer, without the probabilities around it.

    choice -> the chosen id; yes/no -> bool (exactly 0.5 counts as no);
    score -> the most probable level as a string key; ranking -> the best
    candidate's id. Exact ties go to the smallest key (`best_key`).
    """
    kind = answer["type"]
    if kind == CHOICE:
        return answer[CHOICE]
    if kind == YES_NO:
        return answer[YES_NO] > 0.5
    if kind == RANKING:
        return answer[RANKING][0]
    probabilities = answer["probabilities"]
    return best_key(list(probabilities), list(probabilities.values()))

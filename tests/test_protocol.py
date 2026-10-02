"""Request validation, compilation and aggregation (no model needed)."""
from __future__ import annotations

import math

import pytest

from frida_decisions import QuestionType, RequestError, aggregate, compile_request, parse_request
from frida_decisions.protocol import decision

YES_NO = QuestionType.YES_NO


def test_rejects_malformed_requests():
    bad = [
        [],
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": 2}}}},
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "unknown", "instructions": "x"}}},
        {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": 1}}}},
        {"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}},
        {"state": "s", "questions": {"q": {"type": YES_NO, "instructions": "x", "criteria": {"maybe": "m"}}}},
        {"state": "s", "questions": {"q": {"type": "ranking", "instructions": "x", "criteria": ["only"]}}},
        {"state": "s", "extra": 1, "questions": {"q": {"type": YES_NO, "instructions": "x"}}},
    ]
    for raw in bad:
        with pytest.raises(RequestError):
            parse_request(raw)


def test_error_payload_names_the_field():
    with pytest.raises(RequestError) as info:
        parse_request({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x"}}})
    assert info.value.payload()["error"]["field"] == "questions.q.criteria"


def test_compile_order_and_rendering():
    request = parse_request({"state": {"b": 1, "a": "й"}, "questions": {
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": "first", "y": None}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        "n": {"type": YES_NO, "instructions": "yes?", "criteria": {"true": "it is"}},
        "r": {"type": "ranking", "instructions": "order", "criteria": ["p0", {"k": 2}]},
    }})
    cands = compile_request(request)
    assert [(c.question_id, c.option_id) for c in cands] == [
        ("c", "x"), ("c", "y"), ("s", "0"), ("s", "1"), ("n", "true"), ("n", "false"),
        ("r", "0"), ("r", "1")]
    assert cands[0].state == '{"a":"й","b":1}'
    assert cands[0].text == "x: first" and cands[1].text == "y"
    assert cands[4].text == "it is" and cands[5].text == "The answer to the question is no."
    assert cands[7].text == '{"k":2}'
    assert cands[0].instruction.startswith("pick\n\n")


def test_aggregate_shapes():
    request = parse_request({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}},
        "s": {"type": "score", "instructions": "i", "criteria": ["0", "1", "2"]},
        "n": {"type": YES_NO, "instructions": "i"},
        "r": {"type": "ranking", "instructions": "i", "criteria": {"d1": "x", "d2": "y", "d3": "z"}},
    }})
    cands = compile_request(request)
    margins = [0.0, 2.0, 5.0, 0.0, 0.0, 1.0, -1.0, 0.5, 3.0, 0.5]
    answers = aggregate(request, cands, margins)
    assert decision(answers["c"]) == "b"
    assert decision(answers["s"]) == "0" and 0 < answers["s"]["score"] < 1
    assert math.isclose(answers["n"][YES_NO], 1 / (1 + math.exp(-2.0)))   # true 1.0 vs false -1.0
    assert decision(answers["n"]) is True
    assert answers["r"]["ranking"] == ["d2", "d1", "d3"]       # ties keep the caller's order
    assert math.isclose(sum(answers["r"]["probabilities"].values()), 1.0)

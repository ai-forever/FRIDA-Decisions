"""Candidate cardinality and whole-request length for the FRIDA scalar head."""
import copy
import json

from transformers.decision import DecisionConfig


MAX_REQUEST_TOKENS = 32768


def prepare_scalar_inputs(config, request, tokenizer):
    """Supply positional labels for upstream validation without mutating a shared pipeline."""
    if isinstance(request, str):
        request = json.loads(request)
    questions = request.get("questions") if isinstance(request, dict) else None
    labels = dict(config.labels)
    if isinstance(questions, dict):
        for name, question in questions.items():
            if not isinstance(question, dict):
                continue
            kind = question.get("type")
            if kind not in ("choice", "score"):
                continue
            criteria = question.get("criteria")
            if not isinstance(criteria, dict if kind == "choice" else list):
                continue  # The Transformers schema reports invalid field types.
            if len(criteria) < 2:
                raise ValueError(f"Question {name!r}: {kind} requires at least two candidates")
            # Upstream defaults to A..Z for a language-model head. This head scores
            # the closing token of each candidate and never renders the labels.
            if len(labels.get(kind, ())) < len(criteria):
                labels[kind] = [str(i) for i in range(len(criteria))]
    local = copy.copy(config)
    local.labels = labels
    return DecisionConfig.prepare_inputs(local, request, tokenizer)


def check_request_length(inputs, questions, merge_size):
    """Measure the same complete tree as vLLM's rows.encode before any model forward."""
    candidates = sum(len(q.options) for q in questions)
    # The wire has eight prefix IDs, two two-ID fields per question and one
    # two-ID length per candidate. input_ids already includes all text separators.
    length = inputs["input_ids"].shape[-1] + 8 + 4 * len(questions) + 2 * candidates
    grids = inputs.get("image_grid_thw")
    if grids is not None and len(grids):
        # Images add one shared two-ID position field, two wrappers per image,
        # and one token per spatially merged patch. The grid still includes time.
        patches = grids.prod(dim=-1) // (merge_size * merge_size)
        length += 2 + 2 * len(grids) + int(patches.sum())
    if length > MAX_REQUEST_TOKENS:
        raise ValueError(f"Request is {length} tokens, over the limit of {MAX_REQUEST_TOKENS}")
    return length

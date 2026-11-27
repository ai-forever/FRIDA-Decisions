"""Branch a Qwen hybrid cache inside a single decision request."""
import copy

import torch
from transformers.cache_utils import DynamicLayer


def fork_cache(cache):
    # Exact DynamicLayer appends with torch.cat, leaving the prefix tensors intact.
    # Recurrent and convolution states are updated in place and need private storage.
    shared = {}
    for layer in cache.layers:
        if type(layer) is DynamicLayer:
            for tensor in (layer.keys, layer.values):
                shared[id(tensor)] = tensor
    return copy.deepcopy(cache, shared)


def score_shared_state(backbone, head, state, groups, embeds, positions):
    """Prefill the state once and the question once per group, then score its leaves."""
    def tail_positions(start, length):
        if positions is None:
            return None
        offsets = torch.arange(length, device=state.device) + start
        return offsets.expand(3, 1, -1)

    state_inputs = {"input_ids": state[None]} if embeds is None else {"inputs_embeds": embeds}
    root_positions = positions
    next_position = None if positions is None else positions.amax() + 1
    if len(groups) == 1:
        # One prefill call suffices if every option shares the same question.
        question = groups[0][0]
        if embeds is None:
            state_inputs = {"input_ids": torch.cat((state, question))[None]}
        else:
            question_embeds = backbone.get_input_embeddings()(question[None])
            state_inputs = {"inputs_embeds": torch.cat((embeds, question_embeds), dim=1)}
            root_positions = torch.cat((positions, tail_positions(next_position, len(question))), dim=-1)

    root = backbone(**state_inputs, position_ids=root_positions, use_cache=True).past_key_values
    scores = []
    for q_index, (question, options) in enumerate(groups):
        # The final child can consume its parent directly.
        parent = root if q_index == len(groups) - 1 else fork_cache(root)
        if len(groups) > 1:
            parent = backbone(input_ids=question[None], past_key_values=parent,
                              position_ids=tail_positions(next_position, len(question)),
                              use_cache=True).past_key_values
        start = None if positions is None else next_position + len(question)
        for i, option in enumerate(options):
            cache = parent if i == len(options) - 1 else fork_cache(parent)
            hidden = backbone(input_ids=option[None], past_key_values=cache,
                              position_ids=tail_positions(start, len(option)), use_cache=True).last_hidden_state
            scores.append(head(hidden[0, -1].float()).reshape(()))
    return torch.stack(scores)

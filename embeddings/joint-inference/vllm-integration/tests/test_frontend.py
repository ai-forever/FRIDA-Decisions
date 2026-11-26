"""Combined output selection and migration to native inputs."""
from types import SimpleNamespace

import torch
from inference_plugins.heads import pack


def outputs(processor, params):
    dense, tokens = torch.tensor([1., 2., 3., 4.]), torch.ones(2, 2)
    if params.task == 'embed': data = dense
    elif params.extra_kwargs['joint_embeddings']: data = pack(dense, tokens, processor.contract)
    else: data = tokens
    return [SimpleNamespace(outputs=SimpleNamespace(data=data))]


def test_plugin_execution_returns_both_heads(processor):
    params = processor.merge_pooling_params()
    assert params.task == 'token_embed'
    assert params.extra_kwargs['joint_embeddings'] is True
    assert set(processor.post_process(outputs(processor, params))[0]) == {'dense', 'multivector'}



def test_data_envelope_accepts_native_inputs_and_rejects_removed_selectors(processor):
    import pytest
    for value, expected in [('text', ['text']), ([1, 2], [[1, 2]]), (['first', 'second'], ['first', 'second'])]:
        parsed = processor.parse_data({'input': value})
        assert processor.pre_process(parsed) == expected
    for data in (None, {}, {'input': None}, {'input': [], 'messages': []},
                 {'input': 'text', 'embed_task': 'dense'}, {'input': 'text', 'role': 'query'}):
        with pytest.raises(ValueError): processor.parse_data(data)
    with pytest.raises(ValueError, match='use messages over HTTP'):
        processor.pre_process(processor.parse_data({'messages': [{'role': 'user', 'content': 'hello'}]}))

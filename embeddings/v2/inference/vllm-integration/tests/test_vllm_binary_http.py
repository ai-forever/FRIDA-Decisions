"""Joint IO views and native multimodal request examples."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from inference_plugins.contract import Contract

CONTRACT = Contract(dense_dim=4, multivector_dim=2)


def batch() -> list[SimpleNamespace]:
    return [SimpleNamespace(outputs=SimpleNamespace(data=torch.arange((2 + n) * 2, dtype=torch.bfloat16).reshape(-1, 2))) for n in (0, 1, 3)]


def test_real_io_processor_tensor_views_match_json_without_engine(processor) -> None:
    Params = type(processor.merge_pooling_params())
    assert processor.parse_data({'input': ['hello']}) == {'input': ['hello']}
    merged = processor.merge_pooling_params()
    assert merged.task == "token_embed"
    assert merged.extra_kwargs == {"joint_embeddings": True}
    assert merged.skip_reading_prefix_cache and merged.use_activation is False
    original = Params(extra_kwargs={"another_option": 7})
    merged = processor.merge_pooling_params(original)
    assert merged.extra_kwargs == {"another_option": 7, "joint_embeddings": True}
    assert original.extra_kwargs == {"another_option": 7}
    with pytest.raises(ValueError, match="normalization"):
        processor.merge_pooling_params(Params(use_activation=True))
    with pytest.raises(ValueError, match="dimensions"):
        processor.merge_pooling_params(Params(dimensions=2))
    outputs = batch()
    tensors = list(processor.tensor_outputs(outputs))
    lists = processor.post_process(outputs)
    for original, tensor, item in zip(outputs, tensors, lists):
        assert tensor["dense"].data_ptr() == original.outputs.data.data_ptr()
        assert tensor["dense"].tolist() == item["dense"]
        assert tensor["multivector"].tolist() == item["multivector"]


@pytest.mark.parametrize('task,data_envelope', [('embed', False), ('token_embed', False), ('plugin', False), ('plugin', True)])
@pytest.mark.parametrize('encoding_format', ['float', 'base64'])
def test_multimodal_client_sends_native_messages(tmp_path, monkeypatch, task, encoding_format, data_envelope):
    import importlib.util
    import sys
    from pathlib import Path
    example = Path(__file__).resolve().parents[2] / 'examples/vllm_multimodal_online.py'
    monkeypatch.syspath_prepend(str(example.parent))
    spec = importlib.util.spec_from_file_location('image_client_test', example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'image.png'
    path.write_bytes(b'png')
    def post(url, payload):
        assert url == 'http://localhost:8000/pooling'
        assert payload['task'] == task
        assert payload['encoding_format'] == encoding_format
        assert 'embed_task' not in payload.get('data', {})
        content = payload['data'] if data_envelope else payload
        messages = content['messages']
        assert content['add_generation_prompt'] is True
        assert messages[0]['content'][0]['image_url']['url'] == 'data:image/png;base64,cG5n'
        assert messages[0]['content'][1] == {'type': 'text', 'text': 'Describe'}
        return {'data': [{'dense': [1], 'multivector': [[1]]}] if task == 'plugin' else [{'data': [1]}]}
    monkeypatch.setattr(module, 'post_json', post)
    monkeypatch.setattr(module, 'display', lambda value: None)
    monkeypatch.setattr(sys, 'argv', ['example', '--url', 'http://localhost:8000', '--image', str(path), '--text', 'Describe', '--task', task, '--encoding-format', encoding_format] + (['--data-envelope'] if data_envelope else []))
    module.main()

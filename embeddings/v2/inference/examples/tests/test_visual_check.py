"""Visual smoke checks must reject a backend that ignores image pixels."""

import base64
from copy import deepcopy
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

from inference_plugins.types import JointEmbedding

import pytest

EXAMPLES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES))
from _visual import image_data_url, validate_visual_outputs, visual_inputs


def outputs() -> list[JointEmbedding]:
    items: list[JointEmbedding] = [
        {'dense': [1., 0.], 'multivector': [[1., 0.]] * rows}
        for rows in (2, 4, 5)
    ]
    items.append({'dense': [0., 1.], 'multivector': [[0., 1.]] * 5})
    return items


def test_fixtures_are_valid_same_size_images_with_different_pixels() -> None:
    image_module = pytest.importorskip('PIL.Image')
    first, second = [image_module.open(io.BytesIO(base64.b64decode(image_data_url(changed=value).split(',')[1]))) for value in (False, True)]
    assert first.size == second.size == (256, 256)
    assert first.mode == second.mode == 'RGB'
    assert first.tobytes() != second.tobytes()
    batch = visual_inputs()
    assert len(batch) == 4
    assert len(batch[1][0]['content']) == 1  # Image-only is a separate test case.
    assert batch[2][0]['content'][1] == batch[3][0]['content'][1]


def test_visual_report_and_ignored_image_detection() -> None:
    items = outputs()
    report = validate_visual_outputs(items)
    assert report['image_change_max_abs_delta'] == {'dense': 1., 'multivector': 1.}
    items[-1] = deepcopy(items[-2])
    with pytest.raises(ValueError, match='Changing image pixels'):
        validate_visual_outputs(items)


@pytest.mark.parametrize('head', ['dense', 'multivector'])
def test_each_head_must_respond_to_pixels(head: Literal["dense", "multivector"]) -> None:
    items = outputs()
    if head == 'dense':
        items[-1]['dense'] = deepcopy(items[-2]['dense'])
    else:
        items[-1]['multivector'] = deepcopy(items[-2]['multivector'])
    with pytest.raises(ValueError, match='both heads'):
        validate_visual_outputs(items)


def test_visual_check_rejects_nonfinite_and_missing_visual_tokens() -> None:
    items = outputs()
    items[1]['dense'][0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        validate_visual_outputs(items)
    items = outputs()
    items[0]['multivector'] = [[1., 0.]] * 5
    with pytest.raises(ValueError, match='visual-token rows'):
        validate_visual_outputs(items)


@pytest.mark.parametrize('online', [False, True])
def test_visual_example_offline_and_online(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], online: bool) -> None:
    spec = importlib.util.spec_from_file_location('visual_example_test', EXAMPLES / 'vllm_visual_check.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[dict[str, Any]] = []
    def post(url: str, payload: dict[str, Any]) -> dict[str, list[JointEmbedding]]:
        assert url == 'http://localhost:8000/pooling' and payload['task'] == 'plugin'
        assert 'data' not in payload and payload['add_generation_prompt'] is True
        assert payload['messages'][0]['role'] == 'user'
        result = outputs()[len(calls)]
        calls.append(payload)
        return {'data': [result]}
    class Tokenizer:
        def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return 'formatted prompt'
    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs['pooler_config'].task == 'token_embed'
            assert kwargs['enable_prefix_caching'] is False
    def encode_joint(llm: LLM, prompts: list[dict[str, Any]]) -> list[JointEmbedding]:
        assert isinstance(llm, LLM) and len(prompts) == 4
        assert prompts[0] == {'prompt': 'formatted prompt'}
        assert prompts[1]['multi_modal_data']['image'][0].size == (256, 256)
        return outputs()
    engine = ModuleType('vllm')
    setattr(engine, "LLM", LLM)
    config_module = ModuleType("vllm.config")
    monkeypatch.setattr(config_module, "PoolerConfig", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    helper = ModuleType("inference_plugins_vllm.offline")
    monkeypatch.setattr(helper, "encode_joint", encode_joint, raising=False)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, 'vllm', engine)
    monkeypatch.setattr(module, 'engine_contract', lambda _: (tmp_path, SimpleNamespace(max_length=512), Tokenizer()))
    monkeypatch.setattr(module, 'post_json', post)
    saved = tmp_path / 'visual.json'
    args = ['example', '--model', 'joint-model', '--result', str(saved)]
    if online: args += ['--url', 'http://localhost:8000']
    monkeypatch.setattr(sys, 'argv', args)
    module.main()
    assert saved.exists()
    assert 'PASS: image-only, image+text' in capsys.readouterr().out

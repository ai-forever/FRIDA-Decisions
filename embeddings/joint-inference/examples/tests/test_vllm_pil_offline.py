"""The PIL example must route all tasks without converting images to URLs."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from PIL import Image
import pytest
import torch

from inference_plugins.contract import Contract


@pytest.mark.parametrize('task', ['embed', 'token_embed', 'plugin'])
def test_pil_example_uses_native_images_for_each_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task: str
) -> None:
    path = Path(__file__).resolve().parents[1] / 'vllm_multimodal_offline.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('pil_example', path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    image_path = tmp_path / 'page.png'
    Image.new('RGB', (16, 16), 'red').save(image_path)
    seen: list[Image.Image] = []
    class Tokenizer:
        eos_token_id: int | None = 99
        eos_token: str | None = '<eos>'
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
            return {'input_ids': [[1]] * len(texts)}
        def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> str:
            assert conversation[0]['content'][0] == {'type': 'image'}
            assert kwargs == {'tokenize': False, 'add_generation_prompt': False}
            return '<image>'
    def check_prompts(prompts: list[dict[str, Any]]) -> None:
        assert isinstance(prompts, list) and len(prompts) == 2
        for prompt in prompts:
            image = prompt['multi_modal_data']['image']
            assert isinstance(image, Image.Image) and image.size == (16, 16)
            seen.append(image)
        assert seen[0] is seen[1]

    class LLM:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs['pooler_config'].task == ('token_embed' if task == 'plugin' else task)
            assert kwargs['io_processor_plugin'] is None

        def encode(self, prompts: Any, pooling_task: str) -> list[Any]:
            assert pooling_task == task and task != 'plugin'
            check_prompts(prompts)
            return [SimpleNamespace(outputs=SimpleNamespace(data=torch.ones(1)))] * 2

    def encode_joint(llm: LLM, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert isinstance(llm, LLM) and task == 'plugin'
        check_prompts(prompts)
        return [{'dense': [1.], 'multivector': [[1.]]}] * 2
    vllm = ModuleType('vllm')
    setattr(vllm, 'LLM', LLM)
    config_module = ModuleType("vllm.config")
    monkeypatch.setattr(config_module, "PoolerConfig", SimpleNamespace, raising=False)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    helper = ModuleType("inference_plugins_vllm.offline")
    monkeypatch.setattr(helper, "encode_joint", encode_joint, raising=False)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setattr(example, 'engine_contract', lambda model: (tmp_path, Contract(), Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['example', '--model', str(tmp_path), '--image', str(image_path), '--task', task])
    example.main()
    assert len(seen) == 2 and seen[0] is seen[1]

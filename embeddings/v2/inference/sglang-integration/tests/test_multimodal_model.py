"""The joint wrapper delegates image/language execution to the native model."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def test_native_multimodal_forward_and_st_projection(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = root / 'tests/fixtures/st'
    modules = json.loads((source / 'modules.json').read_text())
    for name in ('modules.json', 'sentence_bert_config.json', 'config_sentence_transformers.json', 'tokenizer_config.json'):
        shutil.copy2(source / name, tmp_path / name)
    for entry in modules[1:]:
        folder = tmp_path / entry['path']
        folder.mkdir()
        settings = json.loads((source / entry['path'] / 'config.json').read_text())
        if entry['type'].endswith('.Pooling'): settings['embedding_dimension'] = 8
        if entry['type'].endswith('.Dense'): settings.update(in_features=8, out_features=2)
        (folder / 'config.json').write_text(json.dumps(settings))
    weight = torch.randn(2, 8)
    save_file({'linear.weight': weight}, tmp_path / modules[2]['path'] / 'model.safetensors')
    native_calls = []

    class NativeModel(torch.nn.Module):
        def __init__(self, config, quant_config, prefix):
            super().__init__()
            self.visual = torch.nn.Linear(3, 8, bias=False)
        def forward(self, input_ids, positions, forward_batch, get_embedding=False, **kwargs):
            native_calls.append(forward_batch)
            assert get_embedding
            hidden = forward_batch.hidden + self.visual(forward_batch.image)
            return self.pooler(hidden, forward_batch)
        def load_weights(self, weights):
            return {name for name, tensor in weights}

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    stub('sglang.srt.models.qwen3_5', Qwen3_5ForConditionalGeneration=NativeModel)
    stub('sglang.srt.layers.pooler', EmbeddingPoolerOutput=lambda **kwargs: SimpleNamespace(**kwargs))
    stub('sglang.srt.runtime_context', get_model=lambda: SimpleNamespace(model_path=str(tmp_path)))
    path = root / 'sglang-integration/src/inference_plugins_sglang/models/qwen35_joint.py'
    spec = importlib.util.spec_from_file_location('inference_plugins_sglang.models.qwen35_joint', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    model = module.Qwen35JointEmbedding(SimpleNamespace())
    batch = SimpleNamespace(input_ids=torch.arange(3), extend_seq_lens_cpu=[3], extend_prefix_lens_cpu=[0], seq_lens=torch.tensor([3]), return_pooled_hidden_states=True, hidden=torch.randn(3, 8), image=torch.ones(1, 3))
    result = model(batch.input_ids, torch.arange(3), batch)
    hidden = batch.hidden + model.visual(batch.image)
    assert native_calls == [batch]
    torch.testing.assert_close(result.pooled_hidden_states[0], hidden[-1])
    torch.testing.assert_close(result.embeddings[0], torch.nn.functional.normalize(hidden @ weight.T, dim=-1))
    with pytest.raises(ValueError, match='only supports embeddings'):
        model(batch.input_ids, torch.arange(3), batch, get_embedding=False)
    assert model.load_weights([('model.visual.weight', torch.ones(1))]) == {'model.visual.weight'}
    stub('sglang.srt.multimodal.processors.qwen_vl', QwenVLImageProcessor=type('NativeProcessor', (), {}))
    processor_path = root / 'sglang-integration/src/inference_plugins_sglang/processors/qwen35_joint.py'
    spec = importlib.util.spec_from_file_location('test_joint_mm_processor', processor_path)
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    assert processor.JointImageProcessor.models == [module.Qwen35JointEmbedding]

"""CPU IO fixtures substituting only unavailable vLLM interfaces."""
from copy import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from inference_plugins.contract import Contract


@pytest.fixture
def processor(monkeypatch):
    class Base:
        def __class_getitem__(cls, item): return cls
    class Params:
        def __init__(self, **kwargs):
            self.task = None
            self.dimensions = self.use_activation = self.extra_kwargs = None
            self.__dict__.update(kwargs)
        def clone(self): return copy(self)
    def prompt_to_seq(value):
        if isinstance(value, (str, dict, bytes)) or (
            isinstance(value, list) and value and all(isinstance(x, int) for x in value)
        ):
            return [value]
        return value
    for name, attrs in {
        'vllm.plugins.io_processors.interface': {'IOProcessor': Base},
        'vllm.pooling_params': {'PoolingParams': Params},
        'vllm.renderers.inputs.preprocess': {'prompt_to_seq': prompt_to_seq},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('inference_plugins_vllm.io_modes_test', Path(__file__).resolve().parents[1] / 'src/inference_plugins_vllm/io.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = object.__new__(module.JointIOProcessor)
    instance.contract = Contract(dense_dim=4, multivector_dim=2)
    instance.vllm_config = SimpleNamespace(model_config=None)
    return instance

"""Exercise native rendering methods from the checkout with the joint adapter."""
from __future__ import annotations

import ast
from collections.abc import Sequence
from copy import copy
from pathlib import Path
import os
from types import SimpleNamespace

import pytest
import torch


class Params:
    def __init__(self, **kwargs):
        self.task = self.dimensions = self.use_activation = self.extra_kwargs = None
        self.__dict__.update(kwargs)
    def clone(self): return copy(self)


class ValidationError(ValueError):
    def __init__(self, message, **kwargs): super().__init__(message)


class Completion:
    def __init__(self, input): self.input = input
    @classmethod
    def model_validate(cls, payload):
        obj = cls(payload['input'] if cls is Completion else payload['messages'])
        obj.__dict__.update(payload)
        return obj
    def to_pooling_params(self): return Params(task='plugin')
    def build_tok_params(self, *args): return 'completion-options'


class Chat:
    chat_template = chat_template_kwargs = None
    def __init__(self, messages): self.messages = messages
    @classmethod
    def model_validate(cls, payload):
        obj = cls(payload['input'] if cls is Completion else payload['messages'])
        obj.__dict__.update(payload)
        return obj
    def to_pooling_params(self): return Params(task='plugin')
    def build_tok_params(self, *args): return 'chat-options'
    def build_chat_params(self, *args): return self
    def with_defaults(self, *args, **kwargs): return self


class PluginRequest(SimpleNamespace):
    def model_dump(self, exclude, exclude_unset):
        return {key: value for key, value in vars(self).items() if key not in exclude}
class PluginContext(SimpleNamespace): pass
class OfflineContext(SimpleNamespace):
    def build_tok_params(self, options): return self.tokenization_kwargs


@pytest.fixture
def adapter(processor):
    root = Path(os.environ.get('VLLM_SOURCE_DIR', Path(__file__).resolve().parents[3] / 'vllm')) / 'vllm'
    if not root.exists(): pytest.skip('Set VLLM_SOURCE_DIR to the local vLLM checkout')
    ns = dict(Sequence=Sequence, PoolingParams=Params, VLLMValidationError=ValidationError,
        TextPrompt=dict, TokensPrompt=dict,
        is_list_of=lambda values, cls: all(isinstance(x, cls) for x in values),
        PoolingChatLikeRequest=Chat, PoolingCompletionLikeRequest=Completion,
        OfflineEncodeInputsContext=OfflineContext,
        EncodeCMPLRenderParams=dict, EncodeChatRenderParams=dict,
        is_mistral_tokenizer=lambda _: False,
        merge_kwargs=lambda left, right: {**(left or {}), **(right or {})})
    def execute(nodes, path):
        nodes = [node for node in nodes if not (isinstance(node, ast.ImportFrom) and node.module == '__future__')]
        tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    path = root / 'renderers/inputs/preprocess.py'
    nodes = ast.parse(path.read_text()).body
    execute([node for node in nodes if isinstance(node, ast.FunctionDef) and not node.decorator_list and node.name in {
        'prompt_to_seq', '_validate_prompt_dict', 'parse_dec_only_prompt', 'parse_model_prompt'}], path)
    path = root / 'entrypoints/pooling/base/io_processor.py'
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'PoolingIOProcessor')
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        'create_pooling_params', 'get_request_factory_online', 'get_request_factory_offline',
        '_params_to_seq', '_lora_request_to_seq', '_priority_to_seq', '_validate_chat_template'}]
    execute([cls], path)
    plugin_path = root / 'entrypoints/pooling/pooling/io_processor.py'
    plugin_cls = next(node for node in ast.parse(plugin_path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'PluginWithIOProcessorPlugins')
    plugin_cls.body = [node for node in plugin_cls.body if isinstance(node, ast.FunctionDef) and node.name == 'get_request_factory_offline']
    ns.update(IOProcessorRequest=PluginRequest, OfflinePluginInputsContext=PluginContext,
              PoolingChatRequest=Chat, PoolingCompletionRequest=Completion)
    execute([plugin_cls], plugin_path)
    path = Path(__file__).resolve().parents[1] / 'src/inference_plugins_vllm/native.py'
    nodes = ast.parse(path.read_text()).body
    execute([node for node in nodes if not isinstance(node, ast.ImportFrom) or not (node.module or '').startswith('vllm.')], path)
    result = ns['JointInputProcessor'](io_processor=processor)
    official = ns['PluginWithIOProcessorPlugins']()
    result.official = official
    result.unwrap_data_request = ns['unwrap_data_request']
    result.io_processor = processor
    result.model_config = SimpleNamespace(is_encoder_decoder=False, multimodal_config=None)
    result.renderer = SimpleNamespace(default_cmpl_tok_params=None, default_chat_tok_params=None, tokenizer=None)
    result.chat_template = result.template_kwargs = result.tool_dicts = None
    result.chat_template_content_format = 'string'
    result.trust_request_chat_template = False
    official.__dict__.update(result.__dict__)
    return result


@pytest.mark.parametrize('native_request', [Completion([[1, 2], [3]]), Completion(['first', 'second']),
    Chat([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}])])
def test_native_http_requests_reuse_renderer_and_joint_params(adapter, native_request):
    params = adapter.create_pooling_params(native_request)
    assert params.task == 'token_embed' and params.extra_kwargs == {'joint_embeddings': True}
    ctx = SimpleNamespace(request=native_request, pooling_params=params, lora_request=None, priorities=None,
        prompt_extras={'mm_processor_kwargs': {'max_pixels': 128}})
    rendered = adapter.get_request_factory_online(ctx)
    assert len(rendered) == (1 if isinstance(native_request, Chat) else 2)
    for item in rendered:
        assert item['params'] is params
        assert item['prompt_extras'] is ctx.prompt_extras
        if isinstance(native_request, Chat): assert item['conversations'] is native_request.messages
    if isinstance(native_request, Completion) and isinstance(native_request.input[0], list):
        assert rendered[0]['prompts']['prompt_token_ids'] == [1, 2]


@pytest.mark.parametrize('data', [{'input': ['text']}, {'input': [[1, 2], [3]]}, {'messages': [{'role': 'user', 'content': 'hello'}]}])
def test_http_data_is_normalized_with_native_request_models(adapter, data):
    envelope = PluginRequest(data=data, model='served-model', encoding_format='base64', request_id='r1')
    request = adapter.unwrap_data_request(envelope, adapter.io_processor.parse_data(data))
    assert isinstance(request, Chat if 'messages' in data else Completion)
    assert request.task == 'plugin' and request.model == 'served-model'
    assert request.encoding_format == 'base64' and request.request_id == 'r1'
    ctx = SimpleNamespace(request=request, pooling_params=adapter.create_pooling_params(request),
        lora_request=None, priorities=None, prompt_extras=None)
    assert len(adapter.get_request_factory_online(ctx)) >= 1
    assert envelope.data is data


def test_offline_official_envelope_uses_native_plugin_processor(adapter):
    image = torch.ones(3, 8, 8)
    prompt = {'prompt': '<image>', 'multi_modal_data': {'image': image}}
    params = Params(extra_kwargs={'custom': 1})
    ctx = PluginContext(prompts={'data': {'input': [prompt, [1, 2]]}}, pooling_params=params,
        tokenization_kwargs={'add_special_tokens': False}, lora_request=None, priorities=None)
    factory, count = adapter.official.get_request_factory_offline(ctx)
    rows = list(factory())
    assert count == 2
    assert rows[0]['prompts'] is prompt
    assert rows[0]['prompts']['multi_modal_data']['image'] is image
    assert rows[1]['prompts']['prompt_token_ids'] == [1, 2]
    assert all(row['params'].extra_kwargs == {'custom': 1, 'joint_embeddings': True} for row in rows)
    assert params.task is None and params.extra_kwargs == {'custom': 1}

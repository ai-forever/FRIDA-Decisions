"""Exercise plugin state and dispatch without starting a model worker."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel


@pytest.fixture
def endpoint(monkeypatch):
    class StockServing:
        def __init__(self, engine_client, models, *, supported_tasks, request_logger, chat_template_config):
            self.engine_client, self.models = engine_client, models
            self.renderer = engine_client.renderer
            self.vllm_config = engine_client.vllm_config
            self.model_config = models.model_config
            self.supported_tasks, self.request_logger = supported_tasks, request_logger
            self.chat_template_config = chat_template_config
            self.io_processors = {"embed": object()}

        def get_io_processor(self, request):
            return self.io_processors[request.task or "embed"]

        async def __call__(self, request, raw_request):
            return request, raw_request

        def _build_response(self, ctx):
            return 'native response'

    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    install("vllm.entrypoints.pooling.embed.io_processor", EmbedIOProcessor=Mock(), TokenEmbedIOProcessor=Mock())
    install("vllm.entrypoints.pooling.pooling.api_router", pooling=lambda raw: raw.app.state.serving_pooling)
    install("vllm.entrypoints.pooling.pooling.protocol", IOProcessorRequest=type('Envelope', (), {}), IOProcessorResponse=SimpleNamespace, PoolingRequest=dict)
    install("vllm.entrypoints.pooling.pooling.serving", ServingPooling=StockServing)
    install("vllm.entrypoints.serve.engine.protocol", ErrorResponse=BaseModel)
    install("vllm.entrypoints.serve.utils.api_utils", create_json_response=lambda obj: obj,
            with_cancellation=lambda f: f, load_aware_call=lambda f: f, validate_json_request=lambda: None)
    install("vllm.exceptions", VLLMValidationError=lambda message, parameter: ValueError(message))
    install("inference_plugins_vllm.encoding", encoded_response=Mock())
    install("inference_plugins_vllm.native", JointInputProcessor=Mock(), unwrap_data_request=Mock())
    install("inference_plugins_vllm.io", JointIOProcessor=Mock())
    path = Path(__file__).resolve().parents[1] / 'src/inference_plugins_vllm/endpoint.py'
    spec = importlib.util.spec_from_file_location('inference_plugins_vllm.endpoint_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_is_app_local_and_preserves_native_defaults(endpoint):
    plugin = endpoint.JointPoolingEndpoint()
    app = Mock()
    plugin.attach_router(app)
    app.include_router.assert_called_once_with(endpoint.router)
    engine = SimpleNamespace(renderer=object(), vllm_config=object())
    model_config = SimpleNamespace(architecture='Other')
    stock = endpoint.ServingPooling(engine, SimpleNamespace(model_config=model_config),
        supported_tasks=('embed', 'token_embed'), request_logger=None, chat_template_config=object())
    state = SimpleNamespace(serving_pooling=stock)
    asyncio.run(plugin.init_state(engine, state, None))
    assert state.serving_pooling is stock and not hasattr(state, 'joint_serving_pooling')
    raw_request = SimpleNamespace(app=SimpleNamespace(state=state))
    req = SimpleNamespace(task='embed')
    assert asyncio.run(endpoint.joint_pooling(req, raw_request)) == (req, raw_request)
    model_config.architecture = 'Qwen35JointEmbedding'
    asyncio.run(plugin.init_state(engine, state, None))
    assert state.serving_pooling is stock
    handler = state.joint_serving_pooling
    assert type(handler) is endpoint.JointServingPooling
    assert set(stock.io_processors) == {'embed'}
    for task in [None, 'embed', 'token_embed', 'plugin']:
        req = SimpleNamespace(task=task, dimensions=None)
        assert handler.get_io_processor(req) is handler.io_processors[task or 'embed']
    with pytest.raises(ValueError, match='dimensions'):
        handler.get_io_processor(SimpleNamespace(task='token_embed', dimensions=2))

    envelope = endpoint.IOProcessorRequest()
    envelope.data = {'input': 'q'}
    actual, raw = asyncio.run(endpoint.joint_pooling(envelope, raw_request))
    handler.joint.parse_data.assert_called_once_with(envelope.data)
    endpoint.unwrap_data_request.assert_called_once_with(envelope, handler.joint.parse_data.return_value)
    assert actual is endpoint.unwrap_data_request.return_value and raw is raw_request

    ctx = SimpleNamespace(request=SimpleNamespace(task='embed'), final_res_batch=[], request_id="id", raw_request=None)
    assert handler._build_response(ctx) == 'native response'
    ctx.request.task = 'plugin'
    ctx.request.encoding_format = 'float'
    endpoint.joint_response = Mock()
    assert handler._build_response(ctx) is endpoint.joint_response.return_value
    endpoint.joint_response.assert_called_with(handler.joint, ctx, binary=False)
    ctx.raw_request = SimpleNamespace(headers={'accept': endpoint.MEDIA_TYPE})
    handler._build_response(ctx)
    endpoint.joint_response.assert_called_with(handler.joint, ctx, binary=True)
    ctx.raw_request = None
    for encoding in ('base64', 'bytes', 'bytes_only'):
        ctx.request.encoding_format = encoding
        assert handler._build_response(ctx) is endpoint.encoded_response.return_value
        endpoint.encoded_response.assert_called_with(handler.joint, ctx)


def test_endpoint_registers_post_pooling_and_forwards_the_http_request(endpoint):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    app = FastAPI()
    endpoint.JointPoolingEndpoint.attach_router(app)

    async def serving(request, raw_request):
        assert raw_request.url.path == '/pooling'
        return JSONResponse({'data': request['input']})

    app.state.joint_serving_pooling = serving
    with TestClient(app) as client:
        response = client.post('/pooling', json={'task': 'embed', 'input': ['first', 'second']})
        assert response.status_code == 200
        assert response.json() == {'data': ['first', 'second']}
        assert client.get('/pooling').status_code == 405

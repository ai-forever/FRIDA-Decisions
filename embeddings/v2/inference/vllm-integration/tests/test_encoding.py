"""Round-trip the real vLLM tensor codec and our named-head response hook."""
from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def encoding(monkeypatch):
    root = Path(os.environ.get('VLLM_SOURCE_DIR', Path(__file__).resolve().parents[3] / 'vllm')) / 'vllm'
    if not root.exists(): pytest.skip('Set VLLM_SOURCE_DIR to the local vLLM checkout')
    monkeypatch.setitem(sys.modules, 'pybase64', base64)
    path = root / 'utils/serial_utils.py'
    spec = importlib.util.spec_from_file_location('vllm.utils.serial_utils', path)
    serial = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, serial)
    spec.loader.exec_module(serial)
    # Use the native usage helpers too, without importing the GPU runtime.
    nodes = ast.parse((root / 'entrypoints/pooling/utils.py').read_text()).body
    nodes = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in {'get_pooling_usage', 'get_pooling_usage_payload', 'encode_pooling_output_base64', 'encode_pooling_bytes', 'build_pooling_bytes_streaming_response'}]
    utils = ModuleType('vllm.entrypoints.pooling.utils')
    from starlette.responses import StreamingResponse
    utils.__dict__.update(UsageInfo=SimpleNamespace, pybase64=base64, tensor2binary=serial.tensor2binary, StreamingResponse=StreamingResponse, json=json)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<native usage>', 'exec'), utils.__dict__)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    path = Path(__file__).resolve().parents[1] / 'src/inference_plugins_vllm/encoding.py'
    spec = importlib.util.spec_from_file_location('inference_plugins_vllm.encoding_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Processor:
    def tensor_outputs(self, outputs):
        for item in outputs:
            yield item.heads


def context(fmt, dtype='float32', endian='little'):
    # Include noncontiguous tensors, gradients and an empty multivector matrix.
    outputs = [SimpleNamespace(prompt_token_ids=[1, 2], outputs=SimpleNamespace(data=torch.empty(0)), heads={
        'dense': torch.arange(8., requires_grad=True)[::2],
        'multivector': torch.arange(n * 4.).reshape(n, 4)[:, ::2],
    }) for n in (0, 1, 3)]
    return SimpleNamespace(request=SimpleNamespace(task='plugin', encoding_format=fmt,
        embed_dtype=dtype, endianness=endian), final_res_batch=outputs,
        request_id='joint-test', created_time=123, model_name='joint-model')


def unpack(data, info):
    dtype = np.dtype(info['embed_dtype']).newbyteorder('<' if info['endianness'] == 'little' else '>')
    return np.frombuffer(data, dtype=dtype).reshape(info['shape'])


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('endian', ['little', 'big', 'native'])
def test_base64_named_heads_roundtrip(encoding, dtype, endian):
    ctx = context('base64', dtype, endian)
    body = json.loads(encoding.encoded_response(Processor(), ctx).body)
    assert body['request_id'] == 'joint-test' and body['created_at'] == 123
    for item, original in zip(body['data'], ctx.final_res_batch, strict=True):
        assert set(item) == {'dense', 'multivector'}
        for name, descriptor in item.items():
            assert descriptor['endianness'] == (sys.byteorder if endian == 'native' else endian)
            actual = unpack(base64.b64decode(descriptor['data'], validate=True), descriptor)
            np.testing.assert_array_equal(actual, original.heads[name].detach().numpy())


@pytest.mark.parametrize('fmt', ['bytes', 'bytes_only'])
@pytest.mark.parametrize('endian', ['big', 'little'])
def test_bytes_offsets_and_usage_match_base64(encoding, fmt, endian):
    ctx = context(fmt, endian=endian)
    response = encoding.encoded_response(Processor(), ctx)
    async def read(): return b''.join([chunk async for chunk in response.body_iterator])
    raw = asyncio.run(read())
    assert response.media_type == 'application/octet-stream'
    # bytes_only carries the same bytes without the metadata header.
    ctx.request.encoding_format = 'bytes'
    metadata = json.loads(encoding.encoded_response(Processor(), ctx).headers['metadata'])
    assert ('metadata' in response.headers) == (fmt == 'bytes')
    assert metadata['usage'] == {'prompt_tokens': 6, 'total_tokens': 6}
    offset = 0
    for index, descriptor in enumerate(metadata['data']):
        assert descriptor['index'] == index
        assert descriptor['input_index'] == index // 2
        assert descriptor['head'] == ('dense' if index % 2 == 0 else 'multivector')
        assert descriptor['start'] == offset
        offset = descriptor['end']
        actual = unpack(raw[descriptor['start']:offset], descriptor)
        expected = ctx.final_res_batch[index // 2].heads[descriptor['head']].detach().numpy()
        np.testing.assert_array_equal(actual, expected)
    assert offset == len(raw)


def test_bfloat16_uses_native_bit_representation(encoding):
    ctx = context('base64', 'bfloat16', 'little')
    item = json.loads(encoding.encoded_response(Processor(), ctx).body)['data'][0]['dense']
    assert item['embed_dtype'] == 'bfloat16'
    expected = ctx.final_res_batch[0].heads['dense'].detach().to(torch.bfloat16).view(torch.int16).numpy().astype('<i2').tobytes()
    assert base64.b64decode(item['data']) == expected


@pytest.mark.parametrize('fmt', ['base64', 'bytes', 'bytes_only'])
def test_encoded_http_response_uses_native_transport(encoding, fmt):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    @app.post('/pooling')
    def endpoint(body: dict):
        return encoding.encoded_response(Processor(), context(body['encoding_format']))
    with TestClient(app) as client:
        response = client.post('/pooling', json={'task': 'plugin', 'encoding_format': fmt})
    assert response.status_code == 200
    if fmt == 'base64':
        descriptor = response.json()['data'][0]['dense']
        np.testing.assert_array_equal(unpack(base64.b64decode(descriptor['data']), descriptor), [0., 2., 4., 6.])
    else:
        assert response.headers['content-type'] == 'application/octet-stream'
        assert ('metadata' in response.headers) == (fmt == 'bytes')
        if fmt == 'bytes':
            metadata = response.headers['metadata']
        else:
            metadata = encoding.encoded_response(Processor(), context('bytes')).headers['metadata']
        layout = json.loads(metadata)['data']
        for part in layout:
            actual = unpack(response.content[part['start']:part['end']], part)
            assert list(actual.shape) == part['shape']

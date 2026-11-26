"""Only native encoding formats and launcher aliases remain."""

import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import sys
import tomllib
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('fmt', ['float', 'base64', 'bytes', 'bytes_only'])
def test_client_requests_native_formats(monkeypatch, capsys, fmt):
    path = Path(__file__).resolve().parents[2] / 'examples/vllm_binary_online.py'
    spec = importlib.util.spec_from_file_location('native_binary_client', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Response(io.BytesIO):
        def __init__(self, body, content_type, metadata=None):
            super().__init__(body)
            self.headers = Message()
            self.headers['Content-Type'] = content_type
            if metadata is not None: self.headers['metadata'] = json.dumps(metadata)
    media_type = 'application/octet-stream' if fmt.startswith('bytes') else 'application/json'
    def respond(request, **kwargs):
        payload = json.loads(request.data)
        assert payload['task'] == 'plugin' and payload['encoding_format'] == fmt
        assert payload['input'] == ['one'] and 'data' not in payload
        assert payload['embed_dtype'] == 'float32' and payload['endianness'] == 'little'
        assert request.headers['Accept'] == media_type
        if fmt.startswith('bytes'):
            return Response(b'raw', media_type, {'data': []} if fmt == 'bytes' else None)
        dense, tokens = [1., 2.], [[1., 2.]]
        if fmt == 'base64':
            dense, tokens = {'data': 'encoded', 'shape': [2]}, {'data': 'encoded', 'shape': [1, 2]}
        return Response(json.dumps({'data': [{'dense': dense, 'multivector': tokens}]}).encode(), media_type)
    monkeypatch.setattr(module, 'urlopen', respond)
    monkeypatch.setattr(sys, 'argv', ['example', '--texts', 'one', '--encoding-format', fmt])
    module.main()
    assert f'Transport: {media_type}' in capsys.readouterr().out
    monkeypatch.setattr(module, 'urlopen', lambda *args, **kwargs: Response(b'', 'text/plain'))
    with pytest.raises(RuntimeError, match='Requested'):
        module.main()


def test_launcher_aliases_delegate_to_stock_cli(monkeypatch):
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    scripts = config['project']['scripts']
    assert scripts['frida-joint-serve'] == scripts['joint-vllm'] == 'inference_plugins_vllm.serve:main'
    from inference_plugins_vllm import serve
    module = ModuleType('vllm.entrypoints.cli.main')
    module.main = Mock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    serve.main()
    module.main.assert_called_once_with()

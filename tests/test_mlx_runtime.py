"""Optional dependencies and a real request in a separate MLX-only environment."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_import_and_missing_mlx_error():
    code = '''
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mlx', 'torch', 'onnxruntime'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
import frida_decisions
assert not any(n.startswith(('mlx', 'torch', 'onnxruntime')) for n in sys.modules)
try:
    from frida_decisions import MlxJudge
except ImportError as error:
    assert "pip install 'frida-decisions[mlx]'" in str(error)
else:
    raise AssertionError('Missing MLX must raise an installation error')
'''
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_clean_mlx_only_real_inference(model_dir):
    python = os.environ.get('FD_MLX_ONLY_PYTHON')
    if not python:
        pytest.skip('Set FD_MLX_ONLY_PYTHON to the clean MLX-only interpreter')
    result = subprocess.run([python, 'examples/mlx_quickstart.py', '--model', str(model_dir),
                             '--assert-no-torch'], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=120)
    import json
    response = json.loads(result.stdout)
    assert response['usage']['backend'] == 'mlx'
    assert response['answers']['intent']['choice'] == 'port'

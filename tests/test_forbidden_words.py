"""(d) The repository must not mention names that do not belong to it.

The words are stored reversed so this file does not match itself. The model
card is exempt: its benchmark table names the systems it was compared with.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [w[::-1] for w in ("vej", "efasepyt", "mlak", "vejur")]
PATTERN = re.compile("|".join(FORBIDDEN), re.IGNORECASE)
EXEMPT = {"model_card/README.md"}


def repository_files() -> list[Path]:
    """Files that would be committed: tracked plus untracked-but-not-ignored."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", "_export", "__pycache__", ".pytest_cache"}
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts)]


def test_no_forbidden_words_in_repository():
    hits = []
    for path in repository_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        if PATTERN.search(rel):
            hits.append(f"{rel} (file name)")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{rel}:{number}")
    assert not hits, hits


def test_no_forbidden_words_in_exported_configs():
    folder = ROOT / "_export" / "FRIDA-Decisions"
    if not folder.exists():
        pytest.skip("no export")
    for path in folder.glob("*.json"):
        if path.name == "tokenizer.json":          # the base model's vocabulary, not ours
            continue
        text = path.read_text(encoding="utf-8")
        assert not PATTERN.search(text), path.name
        json.loads(text)


def test_onnx_graph_has_no_local_strings():
    """Node names, metadata and doc strings of the shipped graph (not the weights)."""
    onnx = pytest.importorskip("onnx")
    path = ROOT / "_export" / "FRIDA-Decisions" / "onnx" / "model_int8_pertoken.onnx"
    if not path.exists():
        pytest.skip("no ONNX export")
    model = onnx.load(str(path), load_external_data=False)
    strings = [model.doc_string, model.producer_name, model.graph.name, model.graph.doc_string]
    strings += [p.key + p.value for p in model.metadata_props]
    for node in model.graph.node:
        strings += [node.name, node.doc_string, *node.input, *node.output]
        strings += [p.key + p.value for p in node.metadata_props]
    strings += [i.name for i in model.graph.initializer]
    local = re.compile(PATTERN.pattern + r"|users[\\/]|\.py\b", re.IGNORECASE)
    hits = sorted({s for s in strings if s and local.search(s)})
    assert not hits, hits[:20]

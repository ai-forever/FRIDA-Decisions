"""(d) The repository must not mention names that do not belong to it.

The list of names is not stored in the repository: set `FD_FORBIDDEN_WORDS`
to a comma-separated list before a release check, otherwise the word checks
are skipped. The model card is exempt: its benchmark table names the systems
it was compared with.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXEMPT = {"model_card/README.md"}
LOCAL = r"users[\\/]|\.py\b"                 # local paths and source file names


def forbidden_words() -> list[str]:
    raw = os.environ.get("FD_FORBIDDEN_WORDS", "")
    return [w.strip() for w in raw.split(",") if w.strip()]


def forbidden_pattern() -> re.Pattern:
    words = forbidden_words()
    if not words:
        pytest.skip("FD_FORBIDDEN_WORDS is not set")
    return re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)


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
    pattern = forbidden_pattern()
    hits = []
    for path in repository_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        if pattern.search(rel):
            hits.append(f"{rel} (file name)")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{number}")
    assert not hits, hits


def test_no_forbidden_words_in_exported_configs():
    pattern = forbidden_pattern()
    folder = ROOT / "_export" / "FRIDA-Decisions"
    if not folder.exists():
        pytest.skip("no export")
    for path in folder.glob("*.json"):
        if path.name == "tokenizer.json":          # the base model's vocabulary, not ours
            continue
        text = path.read_text(encoding="utf-8")
        assert not pattern.search(text), path.name
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
    words = "|".join(re.escape(w) for w in forbidden_words())
    local = re.compile(LOCAL + ("|" + words if words else ""), re.IGNORECASE)
    hits = sorted({s for s in strings if s and local.search(s)})
    assert not hits, hits[:20]

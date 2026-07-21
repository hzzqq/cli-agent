"""cli-agent 检索逻辑单元测试（无外部依赖，纯 Python 可跑）。

运行：pytest cli-agent/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path，便于直接 import 模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from index_store import INDEX_FILE, IndexEntry, save_index  # noqa: E402
from retriever import MAX_CONTEXT_CHARS, build_context, retrieve  # noqa: E402


@pytest.fixture
def sample_index(tmp_path, monkeypatch):
    entries = [
        IndexEntry(path="a.py", size=10, snippet="def foo(): pass\ndef bar(): return 1"),
        IndexEntry(path="b.py", size=20, snippet="class Foo: pass\nclass Bar: pass"),
        IndexEntry(path="c.md", size=5, snippet="foo bar baz qux"),
    ]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_retrieve_returns_top_k(sample_index):
    # "function" 不在任何片段，"foo" 在三个文件均出现；
    # c.md 因文档短、TF 归一化更高，应为最相关；取 top-2 应含 c.md
    hits = retrieve("foo function", top_k=2)
    assert 1 <= len(hits) <= 2
    paths = [e.path for e in hits]
    assert "c.md" in paths


def test_retrieve_favors_relevant_file(sample_index):
    # "Foo" 仅出现在 b.py（class Foo），应被优先召回
    hits = retrieve("class Foo", top_k=1)
    assert hits and hits[0].path == "b.py"


def test_retrieve_empty_question(sample_index):
    assert retrieve("", top_k=3) == []


def test_retrieve_no_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert retrieve("anything") == []


def test_build_context_truncates(sample_index):
    text, paths = build_context("foo", top_k=10)
    assert text
    # 受 MAX_CONTEXT_CHARS 约束（留少量余量容忍行尾换行）
    assert len(text) <= MAX_CONTEXT_CHARS + 200
    assert len(paths) >= 1


def test_build_context_no_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text, paths = build_context("foo", top_k=5)
    assert text == ""
    assert paths == []

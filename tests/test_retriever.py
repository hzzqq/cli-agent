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

from index_store import INDEX_FILE, IndexEntry, save_index, load_index, search_index, prune_missing  # noqa: E402
from retriever import MAX_CONTEXT_CHARS, _tokenize, build_context, retrieve, retrieve_scored, explain_retrieval  # noqa: E402


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


@pytest.fixture
def cn_index(tmp_path, monkeypatch):
    entries = [
        IndexEntry(path="模型训练.py", size=30, snippet="深度学习模型训练需要大量标注数据"),
        IndexEntry(path="数据清洗.py", size=30, snippet="数据清洗与特征工程是预处理步骤"),
        IndexEntry(path="readme.md", size=10, snippet="本项目用于模型训练流水线"),
    ]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_chinese_tokenize_bigram():
    toks = _tokenize("深度学习")
    assert "深度" in toks
    assert "学习" in toks
    assert "深度学习" in toks


def test_chinese_phrase_recall(cn_index):
    # 「标注数据」只出现在 模型训练.py，且 query 非整词相等
    hits = retrieve("如何准备标注数据", top_k=1)
    assert hits and hits[0].path == "模型训练.py"


def test_build_context_caps_single_block(tmp_path, monkeypatch):
    """R1 新需求验证：巨型文件片段被截断，避免独占上下文挤出其它相关文件。"""
    from index_store import IndexEntry, save_index
    from retriever import MAX_BLOCK_CHARS

    huge = "x" * (MAX_BLOCK_CHARS * 4)  # 远超单块上限
    entries = [
        IndexEntry(path="huge.py", size=9999, snippet=huge),
        IndexEntry(path="small.py", size=10, snippet="foo bar function"),
    ]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    text, paths = build_context("foo function", top_k=10)
    # 巨型文件片段被截断并标注
    assert "已截断" in text
    # 截断后单块不撑爆整体上下文上限
    assert len(text) <= MAX_BLOCK_CHARS + 500
    # 另一相关文件仍被纳入，未被巨型文件挤出
    assert "small.py" in paths


def test_chinese_query_hits_partial(cn_index):
    # query「数据」应召回含「数据」的文件（标注数据 / 数据清洗）
    hits = retrieve("数据处理", top_k=3)
    paths = [e.path for e in hits]
    assert "数据清洗.py" in paths


def test_retrieve_min_score_filters_weak(tmp_path, monkeypatch):
    # strong.py 中 foo 出现 4 次（强相关），weak.py 仅 1 次（弱相关）
    # min_score 设到两者之间，只应保留强相关文件
    entries = [
        IndexEntry(path="strong.py", size=10, snippet="foo foo foo foo"),
        IndexEntry(path="weak.py", size=10, snippet="foo bar"),
    ]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    hits = retrieve("foo", top_k=5, min_score=0.4)
    paths = [e.path for e in hits]
    assert paths == ["strong.py"]  # 仅强相关
    assert "weak.py" not in paths


def test_build_context_labels_score(sample_index):
    text, paths = build_context("foo", top_k=5)
    assert "相关度" in text  # 可观测性：上下文标注相关度
    assert "大小:" in text


def test_search_index_finds_match(tmp_path):
    entries = [IndexEntry(path="foo.py", size=10, snippet="def hello(): pass")]
    save_index(entries, str(tmp_path))
    hits = search_index("hello", str(tmp_path / INDEX_FILE))
    assert len(hits) == 1 and hits[0].path == "foo.py"


def test_search_index_no_keyword(tmp_path):
    entries = [IndexEntry(path="foo.py", size=10, snippet="def hello(): pass")]
    save_index(entries, str(tmp_path))
    assert search_index("", str(tmp_path / INDEX_FILE)) == []


def test_query_term_frequency_boosts(tmp_path, monkeypatch):
    """R2 隐性打分缺陷验证：重复查询词应加权（修复 set 丢弃词频）。"""
    entries = [
        IndexEntry(path="py.py", size=30, snippet="python python python python"),
        IndexEntry(path="jv.py", size=10, snippet="java"),
    ]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    s_once = retrieve_scored("python", top_k=5)[0][1]
    s_twice = retrieve_scored("python python", top_k=5)[0][1]
    assert s_twice > s_once  # 「python python」比「python」更强调，得分应更高


def test_explain_retrieval_returns_terms(tmp_path, monkeypatch):
    """R1 新需求验证：explain_retrieval 返回命中文件与匹配关键词（透明性）。"""
    entries = [IndexEntry(path="a.py", size=10, snippet="def foo(): pass")]
    save_index(entries, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    expl = explain_retrieval("foo", top_k=5)
    assert expl and expl[0]["path"] == "a.py"
    assert "foo" in expl[0]["terms"]


def test_explain_retrieval_empty_on_no_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert explain_retrieval("anything") == []


def test_prune_missing_removes_gone_file(tmp_path):
    p = tmp_path / "gone.py"
    p.write_text("x = 1")
    keep = tmp_path / "keep.py"
    keep.write_text("y = 2")
    entries = [IndexEntry(path=str(p), size=4, snippet="x = 1"),
               IndexEntry(path=str(keep), size=3, snippet="y = 2")]
    save_index(entries, str(tmp_path))
    p.unlink()  # 模拟文件被删除
    removed = prune_missing(str(tmp_path), str(tmp_path / INDEX_FILE))
    assert removed == 1
    remaining = load_index(str(tmp_path / INDEX_FILE))
    assert [e.path for e in remaining] == [str(keep)]


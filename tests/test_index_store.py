"""cli-agent 索引构建测试（无需 LLM / 网络）。

运行：pytest cli-agent/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from index_store import build_index, INDEX_FILE  # noqa: E402
from typer.testing import CliRunner  # noqa: E402
import agent  # noqa: E402

runner = CliRunner()


def test_build_index_skips_too_large(tmp_path):
    """R1 新需求：--max-size 应跳过超过字节上限的文件。"""
    (tmp_path / "small.py").write_text("def f(): pass")
    (tmp_path / "huge.json").write_bytes(b'{"x":1}' * 200)  # ~1400 字节
    entries, skipped = build_index(str(tmp_path), max_size=1000)
    paths = {e.path for e in entries}
    assert str(tmp_path / "small.py") in paths
    assert str(tmp_path / "huge.json") not in paths
    big = [s for s in skipped if s["reason"] == "too_large"]
    assert big and big[0]["path"].endswith("huge.json")


def test_build_index_reports_unsupported_ext(tmp_path):
    """R2 隐性可观测性：非文本扩展名应被公示为 skipped，而非静默丢弃。"""
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.xyz").write_text("binary?")
    entries, skipped = build_index(str(tmp_path))
    assert len(entries) == 1 and entries[0].path.endswith("a.py")
    reasons = {s["reason"] for s in skipped}
    assert "unsupported_ext" in reasons


def test_build_index_default_includes_large_text(tmp_path):
    """默认不限制大小：文本扩展名文件即使较大也应被索引（向后兼容）。"""
    (tmp_path / "big.py").write_bytes(b"# line\n" * 5000)  # 远超 1000
    entries, skipped = build_index(str(tmp_path))  # 无 max_size
    assert len(entries) == 1
    assert all(s["reason"] != "too_large" for s in skipped)


def test_build_index_ext_filter(tmp_path):
    """--ext 白名单：不在白名单内的文本扩展名也应计入 skipped。"""
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# title")
    entries, skipped = build_index(str(tmp_path), exts={".py"})
    assert {e.path for e in entries} == {str(tmp_path / "a.py")}
    assert any(s["reason"] == "ext_filter" and s["path"].endswith("b.md") for s in skipped)


def test_index_command_reports_skipped(tmp_path):
    """端到端：CLI 应公示跳过的文件数量与原因。"""
    (tmp_path / "small.py").write_text("def f(): pass")
    (tmp_path / "huge.json").write_bytes(b'{"x":1}' * 200)
    r = runner.invoke(
        agent.app, ["index", str(tmp_path), "--root", str(tmp_path), "--max-size", "1000"]
    )
    assert r.exit_code == 0
    assert "已索引 1 个文件" in r.stdout
    assert "跳过 1 个文件" in r.stdout
    assert "too_large" in r.stdout


def test_build_index_incremental_reuses_unchanged(tmp_path, monkeypatch):
    """R1 新需求：增量模式复用未变更文件，且变更文件(mtime)被重新读取。"""
    import os as _os
    import time

    f = tmp_path / "a.py"
    f.write_text("VERSION = 1")
    e1, _ = build_index(str(tmp_path))
    assert len(e1) == 1
    old_snippet = e1[0].snippet
    mtime_before = e1[0].mtime

    # 未变更：复用旧条目（snippet 完全一致，不需要重读）
    e2, _ = build_index(str(tmp_path), prev=e1)
    assert len(e2) == 1
    assert e2[0].snippet == old_snippet
    assert e2[0].mtime == mtime_before

    # 修改文件内容，再把 mtime 推到明显不同的值：应重新读取（新内容进入 snippet）
    f.write_text("VERSION = 2  # changed")
    _os.utime(f, (mtime_before + 100, mtime_before + 100))
    e3, _ = build_index(str(tmp_path), prev=e1)
    assert len(e3) == 1
    assert e3[0].snippet != old_snippet
    assert "VERSION = 2" in e3[0].snippet


def test_index_command_incremental_flag(tmp_path):
    """CLI --incremental 应进入增量模式，复用旧索引。"""
    (tmp_path / "a.py").write_text("x = 1")
    r1 = runner.invoke(agent.app, ["index", str(tmp_path), "--root", str(tmp_path)])
    assert r1.exit_code == 0
    r2 = runner.invoke(
        agent.app, ["index", str(tmp_path), "--root", str(tmp_path), "--incremental"]
    )
    assert r2.exit_code == 0
    assert "增量模式" in r2.stdout


def test_index_stats_returns_metadata_and_indexed_at(tmp_path):
    """R2 验证：index_stats 单次读取即返回统计与建立时间（无冗余 I/O）。"""
    from index_store import save_index, index_stats, IndexEntry
    save_index([
        IndexEntry(path="a.py", size=10, snippet="x=1"),
        IndexEntry(path="b.md", size=20, snippet="y"),
    ], str(tmp_path))
    s = index_stats(str(tmp_path / INDEX_FILE))
    assert s is not None
    assert s["file_count"] == 2
    assert s["total_bytes"] == 30
    assert s["indexed_at"]  # 单次读取即拿到建立时间，不依赖第二次单独 open
    exts = dict(s["top_extensions"])
    assert exts.get(".py") == 1 and exts.get(".md") == 1


def test_index_stats_missing_file_returns_none(tmp_path):
    """索引文件不存在时 index_stats 返回 None（供 CLI 友好提示）。"""
    from index_store import index_stats
    assert index_stats(str(tmp_path / "nope.json")) is None


def test_build_index_exclude_by_relpath(tmp_path):
    """R1 新需求：--exclude 按相对路径 fnmatch 忽略文件（如 'tests/*'）。"""
    (tmp_path / "a.py").write_text("x = 1")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_a.py").write_text("def test_a(): pass")
    entries, skipped = build_index(str(tmp_path), exclude=["tests/*"])
    paths = {e.path for e in entries}
    assert str(tmp_path / "a.py") in paths
    assert str(tests_dir / "test_a.py") not in paths
    ex = [s for s in skipped if s["reason"] == "excluded"]
    assert ex and ex[0]["path"].endswith("test_a.py")


def test_build_index_exclude_by_basename(tmp_path):
    """R1 新需求：--exclude 也支持按文件名模式忽略（如 '*.min.js'）。"""
    (tmp_path / "app.js").write_text("console.log(1)")
    (tmp_path / "app.min.js").write_text("// minified")
    entries, skipped = build_index(str(tmp_path), exclude=["*.min.js"])
    paths = {e.path for e in entries}
    assert str(tmp_path / "app.js") in paths
    assert str(tmp_path / "app.min.js") not in paths
    assert any(s["reason"] == "excluded" and s["path"].endswith("app.min.js") for s in skipped)


def test_index_command_exclude_flag(tmp_path):
    """端到端：CLI --exclude 应公示忽略模式并跳过匹配文件。"""
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "vendor.py").write_text("y = 2")
    r = runner.invoke(
        agent.app,
        ["index", str(tmp_path), "--root", str(tmp_path), "--exclude", "vendor.py"],
    )
    assert r.exit_code == 0
    assert "已索引 1 个文件" in r.stdout
    assert "excluded" in r.stdout


def test_prune_missing_respects_root_cwd(tmp_path, monkeypatch):
    """R2 修复验证：从非 root 的 cwd 运行 prune，相对路径应按 root 解析，
    不得误删仍存在的文件（旧实现对相对 path 直接用当前 cwd 判存在）。"""
    from index_store import (
        save_index,
        prune_missing,
        IndexEntry,
        load_index,
        INDEX_FILE,
    )

    (tmp_path / "keep.py").write_text("x = 1")
    (tmp_path / "gone.py").write_text("y = 2")
    save_index(
        [
            IndexEntry(path="keep.py", size=6, snippet="x=1"),
            IndexEntry(path="gone.py", size=5, snippet="y=2"),
        ],
        str(tmp_path),
    )
    # 删除 gone.py 制造「已删除」条目
    (tmp_path / "gone.py").unlink()

    # 切换到与 root 不同的工作目录，再执行 prune
    other = tmp_path / "other_dir"
    other.mkdir()
    monkeypatch.chdir(other)

    removed = prune_missing(str(tmp_path), str(tmp_path / INDEX_FILE))
    assert removed == 1  # 只移除真正删除的 gone.py

    # 切回 root 验证 prune 后 keep.py 仍保留、gone.py 被删除
    monkeypatch.chdir(tmp_path)
    entries = load_index(str(tmp_path / INDEX_FILE))
    kept_paths = {e.path for e in entries}
    assert "keep.py" in kept_paths
    assert "gone.py" not in kept_paths


def test_load_index_skips_malformed_entries(tmp_path):
    """R2 验证：单条条目损坏（如缺 snippet）不应拖垮整个索引，只跳过坏条目。

    旧实现用整体列表推导 IndexEntry(**item)，任一坏条目触发 TypeError 被
    外层捕获后整个索引返回 []，导致 _require_index 误报「未发现索引文件」、
    ask 直接中止。现改为逐条构造、跳过坏条目，保留可用部分。
    """
    import json as _j
    from index_store import load_index

    payload = {
        "indexed_at": "t",
        "files": [
            {"path": "a.py", "size": 10, "snippet": "x"},
            {"path": "b.py", "size": 5},  # 缺 snippet -> TypeError，应跳过
        ],
    }
    p = tmp_path / INDEX_FILE
    p.write_text(_j.dumps(payload), encoding="utf-8")
    entries = load_index(str(p))
    assert len(entries) == 1
    assert entries[0].path == "a.py"

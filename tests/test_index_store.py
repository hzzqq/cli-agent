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

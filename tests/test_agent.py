"""cli-agent CLI 行为测试（使用 typer CliRunner，无需 LLM / 索引文件）。

运行：pytest cli-agent/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def test_context_command_shows_refs(monkeypatch):
    """R1 新需求验证：context 命令只展示检索结果，不调用 LLM。"""
    monkeypatch.setattr(
        agent, "build_context", lambda q, top_k=5, min_score=0.0: ("这是一段上下文", ["a.py", "b.py"])
    )
    r = runner.invoke(agent.app, ["context", "这个函数是做什么的"])
    assert r.exit_code == 0
    assert "a.py" in r.stdout and "b.py" in r.stdout
    assert "2 个文件" in r.stdout


def test_context_command_no_hit(monkeypatch):
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0: ("", []))
    r = runner.invoke(agent.app, ["context", "无关问题"])
    assert r.exit_code == 1
    assert "未检索到" in r.stdout


def test_ask_handles_llm_error_gracefully(monkeypatch):
    """R2 隐式问题验证：LLM 调用失败应被友好捕获，而非抛出裸栈。"""
    monkeypatch.setenv("MOCK_LLM", "0")  # 强制真实分支
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0: ("ctx", ["a.py"]))

    def boom(self, question, context_files, context_text):
        raise agent.LLMError("模拟网络错误")

    monkeypatch.setattr(agent.LLMClient, "answer", boom)
    r = runner.invoke(agent.app, ["ask", "问题"])
    assert r.exit_code == 1
    assert "调用 LLM 失败" in (r.stderr or r.stdout)


def test_ask_mock_path_succeeds(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")  # 走 mock 分支
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0: ("ctx", ["a.py"]))
    r = runner.invoke(agent.app, ["ask", "问题"])
    assert r.exit_code == 0
    assert "MOCK" in r.stdout


def test_ask_min_score_passed_to_build_context(monkeypatch):
    """R1 新需求验证：--min-score 选项应透传给 build_context。"""
    captured = {}

    def fake_build(question, top_k=5, min_score=0.0):
        captured["min_score"] = min_score
        return "ctx", ["a.py"]

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", fake_build)
    r = runner.invoke(agent.app, ["ask", "问题", "--min-score", "0.5"])
    assert r.exit_code == 0
    assert captured.get("min_score") == 0.5

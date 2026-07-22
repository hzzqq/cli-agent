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

    def boom(self, messages, context_files=None, system_prompt=None):
        raise agent.LLMError("模拟网络错误")

    monkeypatch.setattr(agent.LLMClient, "complete", boom)
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


def test_ask_passes_system_prompt(monkeypatch):
    """R1 新需求验证：--system-prompt 应透传给 LLMClient.complete。"""
    captured = {}

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        captured["system_prompt"] = system_prompt
        return "答案"

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0: ("ctx", ["a.py"]))
    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    r = runner.invoke(agent.app, ["ask", "问题", "--system-prompt", "你是严谨的助手"])
    assert r.exit_code == 0
    assert captured.get("system_prompt") == "你是严谨的助手"


def test_ask_system_prompt_file(monkeypatch, tmp_path):
    """R1 新需求验证：--system-prompt-file 从文件读取并透传。"""
    sp_file = tmp_path / "sp.txt"
    sp_file.write_text("来自文件的提示")
    captured = {}

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        captured["system_prompt"] = system_prompt
        return "答案"

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0: ("ctx", ["a.py"]))
    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    r = runner.invoke(agent.app, ["ask", "问题", "--system-prompt-file", str(sp_file)])
    assert r.exit_code == 0
    assert captured.get("system_prompt") == "来自文件的提示"


def test_chat_is_multiturn(monkeypatch):
    """R1 新需求验证：chat 把历史传入 LLM，实现真正的多轮记忆。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    captured = {}

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        captured["messages"] = messages
        return "答案"

    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    # 模拟两轮对话：第二轮应带上第一轮的 user+assistant 历史
    r = runner.invoke(agent.app, ["chat"], input="第一轮\n第二轮\nexit\n")
    assert r.exit_code == 0
    msgs = captured["messages"]
    roles = [m["role"] for m in msgs]
    # 第二轮请求：system + 第一轮user + 第一轮assistant + 第二轮user
    assert roles.count("user") == 2
    assert "assistant" in roles


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


def test_search_command(monkeypatch, tmp_path):
    """R1 新需求验证：search 命令在索引中按关键词定位文件。"""
    from index_store import IndexEntry, save_index

    save_index([IndexEntry(path="foo.py", size=10, snippet="def hello(): pass")], str(tmp_path))
    r = runner.invoke(agent.app, ["search", "hello", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "foo.py" in r.stdout


def test_search_command_no_hit(tmp_path):
    from index_store import IndexEntry, save_index

    save_index([IndexEntry(path="foo.py", size=10, snippet="def hello(): pass")], str(tmp_path))
    r = runner.invoke(agent.app, ["search", "nomatch", "--root", str(tmp_path)])
    assert r.exit_code == 1
    assert "未找到" in r.stdout


def test_ask_reads_from_stdin(monkeypatch):
    """R1 新需求验证：省略参数时从管道(stdin)读取问题。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    captured = {}

    def fake(q, top_k=5, min_score=0.0):
        captured["q"] = q
        return "ctx", ["a.py"]

    monkeypatch.setattr(agent, "build_context", fake)
    r = runner.invoke(agent.app, ["ask"], input="来自管道的问题\n")
    assert r.exit_code == 0
    assert captured.get("q") == "来自管道的问题"
    assert "MOCK" in r.stdout


def test_ask_empty_question_errors(monkeypatch):
    """R2 隐式问题验证：空问题不应无效调用 LLM，应友好报错。"""
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    r = runner.invoke(agent.app, ["ask"], input="")
    assert r.exit_code == 1
    assert "不能为空" in (r.stderr or r.stdout)


def test_search_command_shows_excerpt(tmp_path):
    """R2 可观测性：search 应展示命中关键词的上下文片段，而非仅路径。"""
    from index_store import IndexEntry, save_index

    save_index(
        [IndexEntry(path="foo.py", size=10, snippet="def hello_world(): pass")],
        str(tmp_path),
    )
    r = runner.invoke(agent.app, ["search", "hello", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "foo.py" in r.stdout
    assert "hello_world" in r.stdout  # 片段中应含命中词


def test_prune_command_removes_stale(tmp_path):
    """R1 新需求验证：prune 命令清理已删除文件的陈旧索引条目。"""
    from index_store import IndexEntry, save_index

    # 索引里记录一个已不存在的文件
    save_index(
        [IndexEntry(path=str(tmp_path / "gone.py"), size=10, snippet="x")],
        str(tmp_path),
    )
    r = runner.invoke(agent.app, ["prune", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "已移除 1 个" in r.stdout


def test_prune_command_clean(tmp_path):
    """索引无陈旧条目时，prune 应报告无需清理。"""
    from index_store import IndexEntry, save_index

    keep = tmp_path / "keep.py"
    keep.write_text("x = 1")
    save_index([IndexEntry(path=str(keep), size=5, snippet="x = 1")], str(tmp_path))
    r = runner.invoke(agent.app, ["prune", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "无需清理" in r.stdout

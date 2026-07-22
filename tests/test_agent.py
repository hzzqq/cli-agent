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
import retriever  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def test_context_command_shows_refs(monkeypatch):
    """R1 新需求验证：context 命令只展示检索结果，不调用 LLM。"""
    monkeypatch.setattr(
        agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("这是一段上下文", ["a.py", "b.py"])
    )
    r = runner.invoke(agent.app, ["context", "这个函数是做什么的"])
    assert r.exit_code == 0
    assert "a.py" in r.stdout and "b.py" in r.stdout
    assert "2 个文件" in r.stdout


def test_context_command_no_hit(monkeypatch):
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("", []))
    r = runner.invoke(agent.app, ["context", "无关问题"])
    assert r.exit_code == 1
    assert "未检索到" in r.stdout


def test_ask_handles_llm_error_gracefully(monkeypatch):
    """R2 隐式问题验证：LLM 调用失败应被友好捕获，而非抛出裸栈。"""
    monkeypatch.setenv("MOCK_LLM", "0")  # 强制真实分支
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("ctx", ["a.py"]))

    def boom(self, messages, context_files=None, system_prompt=None):
        raise agent.LLMError("模拟网络错误")

    monkeypatch.setattr(agent.LLMClient, "complete", boom)
    r = runner.invoke(agent.app, ["ask", "问题"])
    assert r.exit_code == 1
    assert "调用 LLM 失败" in (r.stderr or r.stdout)


def test_ask_mock_path_succeeds(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")  # 走 mock 分支
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("ctx", ["a.py"]))
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
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("ctx", ["a.py"]))
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
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("ctx", ["a.py"]))
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

    def fake_build(question, top_k=5, min_score=0.0, max_context_chars=6000):
        captured["min_score"] = min_score
        return "ctx", ["a.py"]

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", fake_build)
    r = runner.invoke(agent.app, ["ask", "问题", "--min-score", "0.5"])
    assert r.exit_code == 0
    assert captured.get("min_score") == 0.5


def test_ask_max_context_chars_passed(monkeypatch):
    """R1 新需求验证：--max-context-chars 选项应透传给 build_context。"""
    captured = {}

    def fake_build(question, top_k=5, min_score=0.0, max_context_chars=6000):
        captured["max_context_chars"] = max_context_chars
        return "ctx", ["a.py"]

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", fake_build)
    r = runner.invoke(agent.app, ["ask", "问题", "--max-context-chars", "1234"])
    assert r.exit_code == 0
    assert captured.get("max_context_chars") == 1234


def test_ask_explain_invokes_explanation(monkeypatch):
    """R1 新需求验证：--explain 打印检索解释（命中文件+相关度+命中词）。"""
    captured = {}

    def fake_build(question, top_k=5, min_score=0.0, max_context_chars=6000):
        return "ctx", ["a.py"]

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", fake_build)

    def fake_explain(q, top_k=5, min_score=0.0):
        captured["q"] = q
        return [{"path": "a.py", "score": 0.9, "terms": ["foo"]}]

    monkeypatch.setattr(retriever, "explain_retrieval", fake_explain)
    r = runner.invoke(agent.app, ["ask", "foo问题", "--explain"])
    assert r.exit_code == 0
    assert captured.get("q") == "foo问题"
    assert "检索解释" in r.stdout
    assert "a.py" in r.stdout


def test_explain_command_standalone(monkeypatch):
    """R1 新需求验证：explain 独立命令展示「为什么召回这些文件」，不调用 LLM。"""
    captured = {}

    def fake_explain(q, top_k=5, min_score=0.0):
        captured["q"] = q
        captured["top_k"] = top_k
        return [{"path": "a.py", "score": 0.9, "terms": ["foo"]},
                {"path": "b.py", "score": 0.3, "terms": ["bar"]}]

    monkeypatch.setattr(agent, "load_index", lambda *a, **k: [1])
    monkeypatch.setattr(retriever, "explain_retrieval", fake_explain)
    r = runner.invoke(agent.app, ["explain", "foo问题", "--top-k", "3"])
    assert r.exit_code == 0
    assert captured.get("q") == "foo问题"
    assert captured.get("top_k") == 3
    assert "共召回 2 个文件" in r.stdout
    assert "a.py" in r.stdout and "b.py" in r.stdout
    assert "foo" in r.stdout  # 命中词展示


def test_explain_command_no_hit(monkeypatch):
    """R2 验证：检索无命中时 explain 应友好退出（exit 1）而非空输出。"""
    monkeypatch.setattr(agent, "load_index", lambda *a, **k: [1])
    monkeypatch.setattr(retriever, "explain_retrieval", lambda *a, **k: [])
    r = runner.invoke(agent.app, ["explain", "不相关问题"])
    assert r.exit_code == 1
    assert "未检索到" in r.stdout


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

    def fake(q, top_k=5, min_score=0.0, max_context_chars=6000):
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


def test_ask_no_context_bypasses_index_requirement(monkeypatch):
    """R2 隐性问题验证：--no-context 下不应强制要求建索引，纯问题也能提问。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    # 关键：load_index 返回空（无索引），但 --no-context 应跳过该检查
    monkeypatch.setattr(agent, "load_index", lambda *a, **k: [])
    r = runner.invoke(agent.app, ["ask", "什么是闭包", "--no-context"])
    assert r.exit_code == 0
    assert "MOCK" in r.stdout
    assert "未使用仓库上下文" in r.stdout


def test_ask_no_context_still_requires_question(monkeypatch):
    """--no-context 不绕过「问题为空」的校验。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda *a, **k: [])
    r = runner.invoke(agent.app, ["ask", "--no-context"], input="")
    assert r.exit_code == 1
    assert "不能为空" in (r.stderr or r.stdout)


def test_ask_reads_question_from_file(monkeypatch, tmp_path):
    """R1 新需求验证：--file 从文件读取问题（优先于位置参数）。"""
    qfile = tmp_path / "q.txt"
    qfile.write_text("请解释依赖注入的实现细节")
    captured = {}

    def fake(q, top_k=5, min_score=0.0, max_context_chars=6000):
        captured["q"] = q
        return "ctx", ["a.py"]

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", fake)
    r = runner.invoke(agent.app, ["ask", "--file", str(qfile)])
    assert r.exit_code == 0
    assert captured.get("q") == "请解释依赖注入的实现细节"


def test_ask_save_writes_answer_file(monkeypatch, tmp_path):
    """R1 新需求验证：--save 把答案写入文件。"""
    out = tmp_path / "answer.md"
    captured = {}

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        captured["answer"] = "这是答案"
        return "这是答案"

    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("ctx", ["a.py"]))
    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    r = runner.invoke(agent.app, ["ask", "问题", "--save", str(out)])
    assert r.exit_code == 0
    assert out.read_text(encoding="utf-8") == "这是答案"


def test_config_command_shows_effective_settings(monkeypatch):
    """R1 新需求验证：config 命令展示生效的 LLM 配置。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    r = runner.invoke(agent.app, ["config"])
    assert r.exit_code == 0
    assert "当前 LLM 配置" in r.stdout
    assert "base_url" in r.stdout
    assert "model" in r.stdout


def test_require_index_reports_corrupt_not_missing(tmp_path, monkeypatch):
    """R2 隐性问题验证：索引文件存在但损坏时，应提示「无法解析」而非「未发现」。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".cliagent_index.json").write_text("{ 这不是合法 json")
    r = runner.invoke(agent.app, ["ask", "问题"])
    assert r.exit_code == 1
    assert "无法解析" in (r.stderr or r.stdout)
    assert "尚未发现" not in (r.stderr or r.stdout)


def test_files_command_lists_indexed(tmp_path):
    """R1 新需求验证：files 命令列出索引中的文件与大小。"""
    from index_store import IndexEntry, save_index
    save_index([
        IndexEntry(path="foo.py", size=12, snippet="x=1"),
        IndexEntry(path="bar.md", size=8, snippet="# t"),
    ], str(tmp_path))
    r = runner.invoke(agent.app, ["files", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "foo.py" in r.stdout
    assert "bar.md" in r.stdout
    assert "2 个文件" in r.stdout


def test_files_command_empty_errors(tmp_path):
    """索引为空/不存在时，files 命令应友好报错。"""
    r = runner.invoke(agent.app, ["files", "--root", str(tmp_path)])
    assert r.exit_code == 1
    assert "索引为空或不存在" in (r.stderr or r.stdout)


def test_clear_command_preview_only(tmp_path):
    """R1 新需求验证：clear 默认仅预览，不真正删除（防误删）。"""
    from index_store import IndexEntry, save_index
    save_index([IndexEntry(path="a.py", size=1, snippet="x")], str(tmp_path))
    idx = tmp_path / ".cliagent_index.json"
    assert idx.exists()
    r = runner.invoke(agent.app, ["clear", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "将删除" in r.stdout
    assert idx.exists()  # 未真正删除


def test_clear_command_with_yes_removes(tmp_path):
    """R1 新需求验证：clear --yes 真正删除索引文件。"""
    from index_store import IndexEntry, save_index
    save_index([IndexEntry(path="a.py", size=1, snippet="x")], str(tmp_path))
    idx = tmp_path / ".cliagent_index.json"
    r = runner.invoke(agent.app, ["clear", "--root", str(tmp_path), "--yes"])
    assert r.exit_code == 0
    assert not idx.exists()


def test_clear_command_no_file(tmp_path):
    """无索引文件时，clear 应友好提示而非报错。"""
    r = runner.invoke(agent.app, ["clear", "--root", str(tmp_path)])
    assert r.exit_code == 0
    assert "没有可删除" in r.stdout


def test_index_incremental_no_old_index_message(tmp_path):
    """R2 隐性可观测性验证：无旧索引时 --incremental 应明确提示将全量重建。"""
    (tmp_path / "a.py").write_text("x = 1")
    r = runner.invoke(
        agent.app, ["index", str(tmp_path), "--root", str(tmp_path), "--incremental"]
    )
    assert r.exit_code == 0
    assert "全量重建" in r.stdout


def test_ask_verbose_prints_retrieval_stats(monkeypatch):
    """R1 新需求验证：--verbose 打印检索概况。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("这是一段上下文内容", ["a.py", "b.py"]))

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        return "答案"

    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    r = runner.invoke(agent.app, ["ask", "问题", "--verbose"])
    assert r.exit_code == 0
    assert "检索" in r.stdout
    assert "2 个文件" in r.stdout
    assert "字符" in r.stdout


def test_ask_no_hits_warns_ungrounded(monkeypatch):
    """R2 隐性问题验证：检索无命中时应告警，提示答案未接地。"""
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(agent, "load_index", lambda: {"x": 1})
    monkeypatch.setattr(agent, "build_context", lambda q, top_k=5, min_score=0.0, max_context_chars=6000: ("", []))

    def fake_complete(self, messages, context_files=None, system_prompt=None):
        return "答案"

    monkeypatch.setattr(agent.LLMClient, "complete", fake_complete)
    r = runner.invoke(agent.app, ["ask", "问题"])
    assert r.exit_code == 0
    assert "未检索到相关文件" in (r.stderr or r.stdout)


def test_session_roundtrip(tmp_path):
    """R1 新能力验证：save_session / load_session 可持久化并还原会话历史。"""
    import json as _j

    p = tmp_path / "s.json"
    hist = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    agent.save_session(str(p), hist)
    assert p.exists()
    assert agent.load_session(str(p)) == hist


def test_load_session_skips_malformed(tmp_path):
    """R1 验证：load_session 过滤畸形消息，只保留合法 {role,content} 项。"""
    import json as _j

    p = tmp_path / "s.json"
    p.write_text(_j.dumps([
        {"role": "user", "content": "ok"},
        {"role": "weird"},        # 缺 content
        "not_a_dict",             # 非对象
    ]), encoding="utf-8")
    assert agent.load_session(str(p)) == [{"role": "user", "content": "ok"}]


def test_chat_session_persists_history(tmp_path, monkeypatch):
    """R1 验证：chat --session 在每轮后落盘历史，支持跨重启续聊。"""
    import json as _j
    from index_store import IndexEntry

    sess = tmp_path / "sess.json"
    monkeypatch.setenv("MOCK_LLM", "1")
    # 让 _require_index 通过（不要求真实索引）
    monkeypatch.setattr(
        agent, "load_index",
        lambda *a, **k: [IndexEntry(path="a.py", size=1, snippet="x")],
    )
    r = runner.invoke(
        agent.app, ["chat", "--session", str(sess)], input="你好\nexit\n"
    )
    assert r.exit_code == 0
    assert sess.exists()
    data = _j.loads(sess.read_text(encoding="utf-8"))
    assert any(
        m.get("role") == "user" and "你好" in m.get("content", "") for m in data
    )


def test_version_command():
    """R1 新需求验证：version 命令打印版本号。"""
    r = runner.invoke(agent.app, ["version"])
    assert r.exit_code == 0
    assert agent.VERSION in r.stdout


def test_save_session_creates_parent_dir(tmp_path):
    """R2 隐性问题验证：会话落盘时父目录不存在应自动创建（而非静默丢历史）。"""
    import json as _j

    sess = tmp_path / "nested" / "deep" / "sess.json"
    agent.save_session(str(sess), [{"role": "user", "content": "hi"}])
    assert sess.exists()
    data = _j.loads(sess.read_text(encoding="utf-8"))
    assert data[0]["content"] == "hi"


def test_save_session_warns_on_failure(tmp_path, capsys):
    """R2 验证：即便保存因权限/路径问题失败，也应告警而不静默吞掉。"""
    # 用一个非目录文件充当「父路径」，制造不可写场景
    bad = tmp_path / "blocker"
    bad.write_text("x")
    sess = bad / "sess.json"  # 父路径是文件，无法 makedirs
    agent.save_session(str(sess), [{"role": "user", "content": "hi"}])
    # 文件未创建（保存确实失败），但应有警告输出到 stderr
    captured = capsys.readouterr()
    assert "会话历史保存失败" in (captured.err or captured.out)


def test_related_command_finds_similar(tmp_path, monkeypatch):
    """R1 新需求验证：related 命令基于内容找出最相似索引文件，并排除自身。"""
    from index_store import IndexEntry, save_index

    save_index([
        IndexEntry(path="target.py", size=10, snippet="def alpha(): pass"),
        IndexEntry(path="sibling.py", size=10, snippet="def alpha(): return 1"),
        IndexEntry(path="other.py", size=10, snippet="zzz qqq vvv"),
    ], str(tmp_path))
    (tmp_path / "target.py").write_text("def alpha(): pass", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(agent.app, ["related", "target.py"])
    assert r.exit_code == 0, r.stdout
    assert "sibling.py" in r.stdout
    # 自身不应作为相似结果出现：输出里 target.py 仅出现在标题行（被查文件），
    # 不应出现在结果列表（分数 路径）中。
    assert r.stdout.count("target.py") == 1


def test_related_command_missing_file(tmp_path, monkeypatch):
    from index_store import IndexEntry, save_index

    save_index([IndexEntry(path="a.py", size=10, snippet="x")], str(tmp_path))
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(agent.app, ["related", "nope.py"])
    assert r.exit_code == 1
    assert "文件不存在" in r.stdout


def test_chat_save_writes_transcript(tmp_path, monkeypatch):
    """R1 新需求验证：chat --save 在退出时把整段对话转录为 Markdown 落盘。"""
    from index_store import IndexEntry

    transcript = tmp_path / "transcript.md"
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(
        agent, "load_index",
        lambda *a, **k: [IndexEntry(path="a.py", size=1, snippet="x")],
    )
    r = runner.invoke(
        agent.app, ["chat", "--save", str(transcript), "--no-context"],
        input="第一轮\n第二轮\nexit\n",
    )
    assert r.exit_code == 0, r.stdout
    assert transcript.exists()
    text = transcript.read_text(encoding="utf-8")
    assert "# cli-agent 对话转录" in text
    assert "第一轮" in text
    assert "第二轮" in text


def test_index_missing_path_fails_fast(tmp_path):
    """R2 隐性可观测性验证：索引目标路径不存在时应快速失败（而非静默建空索引）。"""
    missing = tmp_path / "does_not_exist_dir"
    r = runner.invoke(agent.app, ["index", str(missing)])
    assert r.exit_code == 1
    assert "不存在或不是目录" in (r.stderr or r.stdout)


def test_build_config_reads_file(tmp_path, monkeypatch):
    """R1 新需求验证：_build_config 在 CLI 缺省时回落到持久化配置文件。"""
    import json as _j

    cfg_path = tmp_path / agent.CONFIG_FILE
    cfg_path.write_text(_j.dumps({"model": "saved-model", "base_url": "http://saved"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = agent._build_config(None, None, None, root=str(tmp_path))
    assert cfg is not None
    assert cfg.model == "saved-model"
    assert cfg.base_url == "http://saved"


def test_build_config_cli_overrides_file(tmp_path, monkeypatch):
    """CLI flag 应优先于配置文件（优先级：CLI > 文件 > 环境变量）。"""
    import json as _j

    cfg_path = tmp_path / agent.CONFIG_FILE
    cfg_path.write_text(_j.dumps({"model": "file-model"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = agent._build_config("cli-model", None, None, root=str(tmp_path))
    assert cfg.model == "cli-model"


def test_load_file_config_whitelist_rejects_unknown_keys(tmp_path, monkeypatch):
    """R2 类型安全验证：配置文件中的任意/畸形键不应被注入为 LLM 配置。"""
    import json as _j

    cfg_path = tmp_path / agent.CONFIG_FILE
    cfg_path.write_text(_j.dumps({
        "model": "ok",
        "evil_key": "pwned",        # 非白名单键，应被忽略
        "base_url": 12345,           # 非字符串值，应被忽略
        "api_key": None,             # null，应被忽略
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded = agent._load_file_config(str(tmp_path))
    assert loaded == {"model": "ok"}  # 只保留白名单内的非空字符串


def test_config_save_writes_file(tmp_path, monkeypatch):
    """R1 新需求验证：config --save 把生效配置持久化到 .cliagent_config.json。"""
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(agent.app, ["config", "--model", "m1", "--base-url", "http://b", "--save"])
    assert r.exit_code == 0
    assert "已保存" in r.stdout
    cfg_path = tmp_path / agent.CONFIG_FILE
    assert cfg_path.exists()
    import json as _j
    saved = _j.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["model"] == "m1"
    assert saved["base_url"] == "http://b"


def test_search_command_json_output(tmp_path):
    """R1 验证：search --json 以 JSON 数组输出命中文件（便于脚本消费）。"""
    from index_store import IndexEntry, save_index

    save_index([IndexEntry(path="foo.py", size=10, snippet="def hello(): pass")], str(tmp_path))
    r = runner.invoke(agent.app, ["search", "hello", "--root", str(tmp_path), "--json"])
    assert r.exit_code == 0
    import json as _j
    data = _j.loads(r.stdout)
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["path"] == "foo.py"


def test_files_command_json_output(tmp_path):
    """R1 验证：files --json 以 JSON 数组输出索引文件与大小。"""
    from index_store import IndexEntry, save_index

    save_index([IndexEntry(path="a.py", size=12, snippet="x")], str(tmp_path))
    r = runner.invoke(agent.app, ["files", "--root", str(tmp_path), "--json"])
    assert r.exit_code == 0
    import json as _j
    data = _j.loads(r.stdout)
    assert data[0]["path"] == "a.py"
    assert data[0]["size"] == 12


def test_related_command_json_output(tmp_path, monkeypatch):
    """R1 验证：related --json 以 JSON 数组输出相似文件与分数。"""
    from index_store import IndexEntry, save_index

    # 构造索引：target.py 与 other.py 内容高度重叠；related 要求目标文件真实存在
    (tmp_path / "target.py").write_text("def add(a, b): return a + b", encoding="utf-8")
    save_index([
        IndexEntry(path="target.py", size=20, snippet="def add(a, b): return a + b"),
        IndexEntry(path="other.py", size=20, snippet="def add(a, b): return a + b"),
    ], str(tmp_path))
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(agent.app, ["related", "target.py", "--json"])
    assert r.exit_code == 0
    import json as _j
    data = _j.loads(r.stdout)
    assert isinstance(data, list) and len(data) >= 1
    assert data[0]["path"] != "target.py"  # 排除自身
    assert "score" in data[0]

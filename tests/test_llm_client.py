"""cli-agent LLM 客户端单元测试（mock/真实路径均覆盖，无真实网络）。

运行：pytest cli-agent/tests
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LLMClient, LLMConfig, LLMError, _compose_messages  # noqa: E402


def test_stream_complete_mock_yields_full_text():
    """R1 新需求验证：mock 下 stream_complete 逐字符产出且拼接==完整答案。"""
    cli = LLMClient(LLMConfig(mock=True))
    out = "".join(cli.stream_complete(
        [{"role": "user", "content": "q"}], context_files=["a.py"]
    ))
    assert "MOCK" in out


def test_stream_complete_validates_empty():
    cli = LLMClient(LLMConfig(mock=True))
    with pytest.raises(LLMError):
        list(cli.stream_complete([]))


def test_stream_complete_real_path(monkeypatch):
    """真实路径：stream=True 应逐 token 产出来自 fake 客户端的片段。"""
    captured = {}

    class _FakeDelta:
        def __init__(self, c): self.content = c

    class _FakeChunk:
        def __init__(self, c):
            self.choices = [SimpleNamespace(delta=_FakeDelta(c))]

    class _FakeStream:
        def __init__(self): self._parts = ["你", "好", "世界"]
        def __iter__(self):
            for p in self._parts:
                yield _FakeChunk(p)

    class _FakeCompletions:
        def create(self, **kw):
            captured["stream"] = kw.get("stream")
            return _FakeStream()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    cli = LLMClient(LLMConfig(mock=False))
    monkeypatch.setattr(cli, "_get_client", lambda: _FakeClient())
    pieces = list(cli.stream_complete([{"role": "user", "content": "hi"}]))
    assert pieces == ["你", "好", "世界"]
    assert captured["stream"] is True


def test_stream_complete_retries_on_transient(monkeypatch):
    """R1/R2 验证：建立流遇瞬态错误时按重试次数恢复，最终成功产出。"""
    calls = {"n": 0}

    class _FakeDelta:
        def __init__(self, c): self.content = c

    class _FakeChunk:
        def __init__(self, c):
            self.choices = [SimpleNamespace(delta=_FakeDelta(c))]

    class _FakeStream:
        def __init__(self): self._parts = ["ok"]
        def __iter__(self):
            for p in self._parts:
                yield _FakeChunk(p)

    class _FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:  # 前两次模拟瞬态失败
                raise TimeoutError("connection timeout")
            return _FakeStream()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _FakeClient())
    cli = LLMClient(LLMConfig(mock=False, retries=3, backoff=0))
    out = list(cli.stream_complete([{"role": "user", "content": "q"}]))
    assert out == ["ok"]
    assert cli.last_attempts == 3
    assert cli.last_error is None


def test_stream_complete_no_retry_on_non_transient(monkeypatch):
    """R2 验证：建立流遇非瞬态错误立即放弃（不重试），并包装为 LLMError。"""
    calls = {"n": 0}

    class _FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            raise ValueError("auth failed")  # 非瞬态

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _FakeClient())
    cli = LLMClient(LLMConfig(mock=False, retries=3, backoff=0))
    with pytest.raises(LLMError):
        list(cli.stream_complete([{"role": "user", "content": "q"}]))
    assert cli.last_attempts == 1  # 未重试


class _FakeCompletions:
    def __init__(self, content="ok", usage=None, raise_exc=None):
        self._content = content
        self._usage = usage or SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        self._raise = raise_exc

    def create(self, **kwargs):
        if self._raise is not None:
            raise self._raise
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=self._usage,
        )


class _FakeChat:
    def __init__(self, **kw):
        self.completions = _FakeCompletions(**kw)


class _FakeClient:
    def __init__(self, **kw):
        self.chat = _FakeChat(**kw)


@pytest.fixture
def fake_client(monkeypatch):
    def _make(content="hello", usage=None, raise_exc=None):
        fc = _FakeClient(content=content, usage=usage, raise_exc=raise_exc)
        monkeypatch.setattr(LLMClient, "_get_client", lambda self: fc)
        return fc

    return _make


def test_estimate_tokens_cjk():
    assert LLMClient.estimate_tokens("你好世界") == 4
    assert LLMClient.estimate_tokens("abcde") == 2  # 5 个非 CJK 字符 -> (5+3)//4 = 2


def test_estimate_tokens_empty():
    assert LLMClient.estimate_tokens("") == 0


def test_validate_messages_ok():
    LLMClient._validate_messages(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    )


def test_validate_messages_bad_role():
    with pytest.raises(LLMError):
        LLMClient._validate_messages([{"role": "bot", "content": "x"}])


def test_validate_messages_missing_content():
    with pytest.raises(LLMError):
        LLMClient._validate_messages([{"role": "user", "content": 123}])


def test_complete_mock_sets_observability():
    c = LLMClient(LLMConfig(mock=True))
    out = c.complete([{"role": "user", "content": "如何配置?"}])
    assert "MOCK" in out
    assert c.last_attempts == 1
    assert c.last_error is None
    assert c.last_usage is not None and c.last_usage["prompt_tokens"] >= 0


def test_complete_real_fake(fake_client):
    fake_client(content="answer-42")
    c = LLMClient(LLMConfig(mock=False))
    out = c.complete([{"role": "user", "content": "q"}])
    assert out == "answer-42"
    assert c.last_usage["total_tokens"] == 2


def test_complete_empty_messages_raises():
    c = LLMClient(LLMConfig(mock=True))
    with pytest.raises(LLMError):
        c.complete([])


def test_retries_on_transient(fake_client):
    fake_client(raise_exc=TimeoutError("connection timeout"))
    c = LLMClient(LLMConfig(mock=False, retries=2, backoff=0))
    with pytest.raises(LLMError):
        c.complete([{"role": "user", "content": "q"}])
    assert c.last_attempts == 3  # 首次 + 2 次重试


def test_no_retry_on_non_transient(fake_client):
    fake_client(raise_exc=ValueError("auth failed"))  # 非瞬态关键词，立即放弃
    c = LLMClient(LLMConfig(mock=False, retries=3, backoff=0))
    with pytest.raises(LLMError):
        c.complete([{"role": "user", "content": "q"}])
    assert c.last_attempts == 1


def test_health_mock():
    c = LLMClient(LLMConfig(mock=True))
    h = c.health()
    assert h["ok"] is True and h["mock"] is True and h["attempts"] == 0


def test_health_real_ok(fake_client):
    fake_client(content="OK")
    c = LLMClient(LLMConfig(mock=False))
    h = c.health()
    assert h["ok"] is True and h["mock"] is False and h["error"] is None


def test_health_real_fail(fake_client):
    fake_client(raise_exc=RuntimeError("503 service unavailable"))
    c = LLMClient(LLMConfig(mock=False, retries=1, backoff=0))
    h = c.health()
    assert h["ok"] is False and h["error"] is not None


def test_trim_messages_under_budget():
    """R2 隐性问题验证：预算充裕时不应裁剪任何消息。"""
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    out, trimmed = LLMClient.trim_messages_to_budget(msgs, 1000)
    assert out == msgs and trimmed is False


def test_trim_messages_over_budget_keeps_last():
    """R2 验证：超出预算时丢弃最旧消息，但保留最后一条对话消息。"""
    msgs = [{"role": "user", "content": "x" * 5000} for _ in range(4)]
    out, trimmed = LLMClient.trim_messages_to_budget(msgs, 10)
    assert trimmed is True
    assert len(out) == 1
    assert out[0]["content"] == "x" * 5000


def test_trim_messages_preserves_system():
    """R2 验证：system 提示始终优先保留，即便其自身已占满预算。"""
    msgs = [{"role": "system", "content": "sys" * 5000}] + [
        {"role": "user", "content": "x" * 5000} for _ in range(3)
    ]
    out, trimmed = LLMClient.trim_messages_to_budget(msgs, 10)
    assert out[0]["role"] == "system"
    assert any(m["role"] == "system" for m in out)


def test_complete_trims_oversized_history(monkeypatch):
    """R2 隐性问题验证：超大历史在发送前被裁剪到预算内，避免上下文溢出 400。"""
    captured = {}

    class _Compl:
        def create(self, **kw):
            captured["messages"] = kw["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class _Chat:
        completions = _Compl()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client())
    c = LLMClient(LLMConfig(mock=False, max_context_tokens=10))
    big = [{"role": "user", "content": "x" * 2000} for _ in range(5)]
    out = c.complete(big)
    assert out == "ok"
    msgs = captured["messages"]
    # system 提示 + 仅剩的最后一条 user 消息（其余被裁掉）
    assert msgs[0]["role"] == "system"
    assert len(msgs) == 2


def test_complete_passes_top_p(monkeypatch):
    """R1/R2 验证：top_p 生成参数须真正透传到 SDK 调用（此前被静默忽略）。"""
    captured = {}

    class _C:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class _Chat:
        completions = _C()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client())
    c = LLMClient(LLMConfig(mock=False, top_p=0.9))
    c.complete([{"role": "user", "content": "q"}])
    assert captured.get("top_p") == 0.9


def test_default_top_p_applied(monkeypatch):
    """默认 top_p 仍应随调用下发（保持与 temperature 一致的行为）。"""
    captured = {}

    class _C:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class _Chat:
        completions = _C()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client())
    c = LLMClient(LLMConfig(mock=False))
    c.complete([{"role": "user", "content": "q"}])
    assert captured.get("top_p") == 1.0


def test_compose_messages_no_system_prepends_default():
    """R2 验证：无 system 时前置默认系统提示。"""
    out = _compose_messages([{"role": "user", "content": "q"}], None)
    assert out[0]["role"] == "system"
    assert len(out) == 2


def test_compose_messages_keeps_caller_system():
    """R2 验证：调用方已带 system 消息时保留之，不重复前置。"""
    msgs = [{"role": "system", "content": "mine"}, {"role": "user", "content": "q"}]
    out = _compose_messages(msgs, None)
    assert out[0]["content"] == "mine"
    assert len(out) == 2


def test_compose_messages_system_prompt_overrides():
    """R2 验证：显式传 system_prompt 时覆盖调用方自带 system。"""
    msgs = [{"role": "system", "content": "mine"}, {"role": "user", "content": "q"}]
    out = _compose_messages(msgs, "override")
    assert out[0]["content"] == "override"
    assert len(out) == 2


def test_compose_messages_explicit_system_prompt_prepends():
    out = _compose_messages([{"role": "user", "content": "q"}], "custom")
    assert out[0] == {"role": "system", "content": "custom"}


def test_complete_existing_system_not_duplicated(monkeypatch):
    """R2 验证：调用方已带 system 消息时，complete 不生成「双 system 块」。"""
    captured = {}

    class _C:
        def create(self, **kw):
            captured["messages"] = kw["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class _Chat:
        completions = _C()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client())
    c = LLMClient(LLMConfig(mock=False))
    c.complete([{"role": "system", "content": "caller"}, {"role": "user", "content": "q"}])
    msgs = captured["messages"]
    assert sum(1 for m in msgs if m["role"] == "system") == 1
    assert msgs[0]["content"] == "caller"


def test_list_models_mock():
    """R1 验证：mock 模式返回 [当前模型]。"""
    c = LLMClient(LLMConfig(mock=True))
    assert c.list_models() == [c.config.model]


def test_list_models_real(fake_client, monkeypatch):
    """R1 验证：真实模式从 /models 端点取回 id 列表。"""

    class _Model:
        def __init__(self, i):
            self.id = i

    class _Models:
        def list(self):
            return SimpleNamespace(data=[_Model("a"), _Model("b")])

    class _Client2:
        models = _Models()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client2())
    c = LLMClient(LLMConfig(mock=False))
    assert c.list_models() == ["a", "b"]


def test_list_models_real_error(monkeypatch):
    """R1 验证：端点异常被统一包装为 LLMError。"""

    class _Bad:
        def list(self):
            raise RuntimeError("500 boom")

    class _Client3:
        models = _Bad()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client3())
    c = LLMClient(LLMConfig(mock=False))
    with pytest.raises(LLMError):
        c.list_models()


def test_build_sampling_params_includes_penalties():
    """R1 验证：采样参数构造器汇总 temperature/top_p/frequency/presence。"""
    c = LLMClient(LLMConfig(mock=False, temperature=0.5, top_p=0.9,
                             frequency_penalty=0.3, presence_penalty=0.2))
    p = c._build_sampling_params()
    assert p["temperature"] == 0.5
    assert p["top_p"] == 0.9
    assert p["frequency_penalty"] == 0.3
    assert p["presence_penalty"] == 0.2
    assert p["max_tokens"] == 1024


def test_complete_passes_penalties(monkeypatch):
    """R1/R2 验证：frequency/presence 惩罚须真正透传到 SDK 调用。"""
    captured = {}

    class _C:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    class _Chat:
        completions = _C()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _Client())
    c = LLMClient(LLMConfig(mock=False, frequency_penalty=0.4, presence_penalty=0.25))
    c.complete([{"role": "user", "content": "q"}])
    assert captured.get("frequency_penalty") == 0.4
    assert captured.get("presence_penalty") == 0.25


def test_stream_complete_passes_penalties(monkeypatch):
    """R1/R2 验证：流式路径与完整路径共用同一套采样参数（DRY，不分叉）。"""
    captured = {}

    class _FakeDelta:
        def __init__(self, c): self.content = c

    class _FakeChunk:
        def __init__(self, c):
            self.choices = [SimpleNamespace(delta=_FakeDelta(c))]

    class _FakeStream:
        def __init__(self): self._parts = ["ok"]
        def __iter__(self):
            for p in self._parts:
                yield _FakeChunk(p)

    class _FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            return _FakeStream()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _FakeClient())
    c = LLMClient(LLMConfig(mock=False, frequency_penalty=0.4, presence_penalty=0.25))
    list(c.stream_complete([{"role": "user", "content": "hi"}]))
    assert captured.get("frequency_penalty") == 0.4
    assert captured.get("presence_penalty") == 0.25
    assert captured.get("stream") is True


def test_health_mock_does_not_pollute_observability(monkeypatch):
    """R2 验证：mock 模式下 health() 不触网，也不改写真实问答的可观测字段。"""
    client = LLMClient(LLMConfig(mock=True))
    client.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    client.last_attempts = 2
    client.last_error = "boom"
    health = client.health()
    assert health["ok"] is True
    assert client.last_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert client.last_attempts == 2
    assert client.last_error == "boom"


def test_health_real_restores_observable_state(monkeypatch):
    """R2 验证：真实探针会改写 last_* 字段，但 health() 必须在返回前还原为探针前的值。"""
    client = LLMClient(LLMConfig(mock=False))
    # 预置一次真实问答后的可观测字段
    client.last_usage = {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    client.last_attempts = 1
    client.last_error = None

    def fake_complete(messages, context_files=None, system_prompt=None):
        # 注意：被 monkeypatch 成实例属性后调用时不会再传入 self
        client.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        client.last_attempts = 3
        client.last_error = "probe-failed"
        return "OK"

    monkeypatch.setattr(client, "complete", fake_complete)
    health = client.health()
    assert health["ok"] is True
    # 探针后的可观测字段必须还原为真实问答的值，而非被探针覆盖
    assert client.last_usage == {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    assert client.last_attempts == 1
    assert client.last_error is None


def test_format_usage_empty():
    # usage 缺失返回空串，调用方据此跳过展示（不污染答案）
    from llm_client import format_usage

    assert format_usage(None) == ""
    assert format_usage({}) == ""
    # 全 0 的用量是合法值（接口返回了但恰好为 0），不应被当作缺失而吞掉
    assert format_usage({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) == "token 总 0（提示 0 / 补全 0）"


def test_format_usage_basic():
    from llm_client import format_usage

    line = format_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="gpt-4o",
    )
    assert "token 总 15" in line
    assert "提示 10" in line
    assert "补全 5" in line
    assert "模型 gpt-4o" in line


def test_format_usage_cost_estimate():
    from llm_client import format_usage

    line = format_usage(
        {"prompt_tokens": 1000, "completion_tokens": 2000, "total_tokens": 3000},
        price_prompt=0.01, price_completion=0.03,
    )
    # 1000*0.01/1000 + 2000*0.03/1000 = 0.01 + 0.06 = 0.07
    assert "$0.0700" in line


def test_stream_complete_captures_usage(monkeypatch):
    # R2 修复验证：真实流式路径（mock=False）应捕获尾部块 token 用量，
    # 此前流式（ask/chat 默认）下 last_usage 恒为 None，用量完全不可见。
    from llm_client import LLMClient, LLMConfig

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2
        total_tokens = 5

    class _Chunk:
        def __init__(self, content=None, usage=None):
            self.choices = (
                [type("_C", (), {"delta": type("_D", (), {"content": content})()})()]
                if content is not None else []
            )
            self.usage = usage

    class _Stream:
        def __init__(self):
            self._n = 0
        def __iter__(self):
            return self
        def __next__(self):
            self._n += 1
            if self._n == 1:
                return _Chunk(content="你好")
            if self._n == 2:
                return _Chunk(usage=_Usage())
            raise StopIteration

    class _Completions:
        def create(self, **kwargs):
            return _Stream()
    class _Chat:
        def __init__(self):
            self.completions = _Completions()
    class _FakeClient:
        def __init__(self):
            self.chat = _Chat()

    cfg = LLMConfig(mock=False, model="m", base_url="http://x", api_key="k")
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "_get_client", lambda: _FakeClient())
    out = "".join(client.stream_complete([{"role": "user", "content": "hi"}]))
    assert out == "你好"
    assert client.last_usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_complete_empty_choices_raises_llm_error(monkeypatch):
    """R2 修复（c166）：200 + 空 choices（内容过滤/上游异常）应统一包装为
    LLMError，而非在守卫之外裸抛 IndexError。"""
    from llm_client import LLMClient, LLMError

    # 显式强制真实分支：统一入口下 openwebui 测试会进程级设置 MOCK_LLM=1，
    # 否则 complete 走 mock 短路、测不到空 choices 路径。
    monkeypatch.setenv("MOCK_LLM", "0")

    class _EmptyChoicesResp:
        choices = []

    class _FakeCompletions:
        def create(self, **kwargs):
            return _EmptyChoicesResp()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    monkeypatch.setattr(LLMClient, "_get_client", lambda self: _FakeClient())
    client = LLMClient()
    with pytest.raises(LLMError, match="空 choices"):
        client.complete([{"role": "user", "content": "问题"}])

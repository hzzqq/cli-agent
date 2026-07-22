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

from llm_client import LLMClient, LLMConfig, LLMError  # noqa: E402


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

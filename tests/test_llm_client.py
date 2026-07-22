"""cli-agent LLM 客户端单元测试（无真实网络调用，使用 fake OpenAI 客户端）。

通过 monkeypatch LLMClient._get_client 注入 fake，覆盖：
- mock 模式（无上下文 / 有上下文）
- 真实模式的 prompt 构造与返回内容
- 可观测性：token 用量写入 last_usage
- 健壮性：底层异常统一包装为 LLMError（隐含错误处理覆盖）
- 配置项：timeout / max_tokens / temperature 影响请求参数
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LLMClient, LLMConfig, LLMError  # noqa: E402


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeResp:
    def __init__(self, content, usage):
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _FakeCompletions:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


class _FakeChat:
    def __init__(self, resp):
        self.completions = _FakeCompletions(resp)


class _FakeClient:
    def __init__(self, resp):
        self.chat = _FakeChat(resp)


@pytest.fixture
def fake_ok_client():
    """返回一个会成功返回的 fake 客户端工厂，并暴露调用记录。"""
    resp = _FakeResp("这是答案", _FakeUsage(12, 34))
    client = _FakeClient(resp)
    return client


def test_mock_answer_no_context():
    client = LLMClient(LLMConfig(mock=True))
    out = client.answer("什么是 X", [], "")
    assert "MOCK" in out
    assert "未检索到" in out


def test_mock_answer_with_context():
    client = LLMClient(LLMConfig(mock=True))
    out = client.answer("什么是 X", ["a.py", "b.py"], "ctx")
    assert "a.py" in out and "b.py" in out


def test_real_answer_returns_content(monkeypatch, fake_ok_client):
    client = LLMClient(LLMConfig(mock=False))
    monkeypatch.setattr(client, "_get_client", lambda: fake_ok_client)
    out = client.answer("问题", ["a.py"], "一些上下文")
    assert out == "这是答案"
    # prompt 应包含上下文文本与文件数
    call = fake_ok_client.chat.completions.calls[0]
    user_msg = call["messages"][1]["content"]
    assert "一些上下文" in user_msg
    assert "1 个文件" in user_msg


def test_real_answer_records_usage(monkeypatch, fake_ok_client):
    client = LLMClient(LLMConfig(mock=False))
    monkeypatch.setattr(client, "_get_client", lambda: fake_ok_client)
    client.answer("问题", ["a.py"], "ctx")
    assert client.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }


def test_real_answer_passes_config_params(monkeypatch, fake_ok_client):
    cfg = LLMConfig(mock=False, model="custom", temperature=0.7, max_tokens=256)
    client = LLMClient(cfg)
    monkeypatch.setattr(client, "_get_client", lambda: fake_ok_client)
    client.answer("问题", ["a.py"], "ctx")
    call = fake_ok_client.chat.completions.calls[0]
    assert call["model"] == "custom"
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 256


def test_real_answer_wraps_error(monkeypatch):
    class _BoomClient:
        @property
        def chat(self):
            raise RuntimeError("connection refused")

    client = LLMClient(LLMConfig(mock=False))
    monkeypatch.setattr(client, "_get_client", lambda: _BoomClient())
    with pytest.raises(LLMError) as exc:
        client.answer("问题", ["a.py"], "ctx")
    assert "LLM 调用失败" in str(exc.value)


def test_missing_openai_dependency_wrapped(monkeypatch):
    """初始化时若 openai 不可用，应抛出 LLMError 而非裸 ImportError。"""
    client = LLMClient(LLMConfig(mock=False))
    # 让 _get_client 走到 import 分支，但屏蔽 openai 模块
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LLMError) as exc:
        client._get_client()
    assert "openai" in str(exc.value)

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


def test_complete_multiturn_mock():
    """R1 新需求验证：complete 接受多轮 messages，mock 下取最后一条 user 作为问题。"""
    client = LLMClient(LLMConfig(mock=True))
    out = client.complete([
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "答1"},
        {"role": "user", "content": "第二问"},
    ], context_files=["a.py"])
    assert "MOCK" in out
    assert "a.py" in out


def test_complete_empty_messages_raises():
    """R2 隐性健壮性：空消息列表应提前拦截并包装为 LLMError。"""
    client = LLMClient(LLMConfig(mock=True))
    with pytest.raises(LLMError):
        client.complete([])


def test_complete_passes_custom_system_prompt(monkeypatch, fake_ok_client):
    """R1 灵活性验证：自定义 system_prompt 应出现在请求消息首位。"""
    client = LLMClient(LLMConfig(mock=False))
    monkeypatch.setattr(client, "_get_client", lambda: fake_ok_client)
    client.complete(
        [{"role": "user", "content": "hi"}],
        system_prompt="你是测试助手",
    )
    call = fake_ok_client.chat.completions.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"] == "你是测试助手"
    assert call["messages"][1]["content"] == "hi"


class _FlakyCompletions:
    """前 fail_times 次调用抛瞬态错误，之后返回成功，用于验证重试。"""
    def __init__(self, resp, fail_times, exc_msg="connection refused"):
        self._resp = resp
        self.fail_times = fail_times
        self.exc_msg = exc_msg
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.exc_msg)
        return self._resp


class _FlakyChat:
    def __init__(self, resp, fail_times, exc_msg="connection refused"):
        self.completions = _FlakyCompletions(resp, fail_times, exc_msg)


class _FlakyClient:
    def __init__(self, resp, fail_times, exc_msg="connection refused"):
        self.chat = _FlakyChat(resp, fail_times, exc_msg)


def test_retry_then_success(monkeypatch):
    """R1 新需求验证：瞬态错误应自动重试，直到成功。"""
    resp = _FakeResp("重试后成功", _FakeUsage(1, 1))
    client = LLMClient(LLMConfig(mock=False, retries=2, backoff=0.0))
    monkeypatch.setattr(client, "_get_client", lambda: _FlakyClient(resp, fail_times=2))
    out = client.answer("问题", ["a.py"], "ctx")
    assert out == "重试后成功"
    assert client.last_attempts == 3          # 首次 + 重试 2 次
    assert client.last_error is None


def test_retry_exhausted_raises(monkeypatch):
    """R2 韧性验证：重试耗尽后统一包装为 LLMError，并记录尝试次数。"""
    client = LLMClient(LLMConfig(mock=False, retries=1, backoff=0.0))
    monkeypatch.setattr(
        client, "_get_client",
        lambda: _FlakyClient(_FakeResp("x", _FakeUsage(1, 1)), fail_times=99, exc_msg="connection timed out"),
    )
    with pytest.raises(LLMError) as exc:
        client.answer("问题", ["a.py"], "ctx")
    assert "已重试 1 次" in str(exc.value)
    assert client.last_attempts == 2


def test_non_transient_no_retry(monkeypatch):
    """R2 隐性正确性：鉴权等「非瞬态」错误不应重试，避免无效等待。"""
    client = LLMClient(LLMConfig(mock=False, retries=3, backoff=0.0))
    monkeypatch.setattr(
        client, "_get_client",
        lambda: _FlakyClient(_FakeResp("x", _FakeUsage(1, 1)), fail_times=99, exc_msg="401 unauthorized"),
    )
    with pytest.raises(LLMError) as exc:
        client.answer("问题", ["a.py"], "ctx")
    assert "已重试" not in str(exc.value)   # 非瞬态，未进入重试
    assert client.last_attempts == 1


def test_estimate_tokens_cjk_and_latin():
    """R1 新需求验证：中文按字计、拉丁按 ~4 字符/token 估算。"""
    assert LLMClient.estimate_tokens("你好世界") == 4          # 4 个 CJK 字
    assert LLMClient.estimate_tokens("hello world") == 3       # 11 字符 /4 上取整
    assert LLMClient.estimate_tokens("") == 0
    assert LLMClient.estimate_tokens("a" * 40) == 10           # 40/4


def test_validate_messages_rejects_bad_role_and_missing_content():
    """R2 隐性健壮性：畸形 messages 应在调用前被拦截为 LLMError，而非触发 SDK 5xx。"""
    with pytest.raises(LLMError):
        LLMClient._validate_messages([{"role": "god", "content": "x"}])
    with pytest.raises(LLMError):
        LLMClient._validate_messages([{"role": "user"}])  # 缺 content
    with pytest.raises(LLMError):
        LLMClient._validate_messages(["not-a-dict"])


def test_complete_validates_before_mock():
    """complete 在 mock 路径前也校验消息结构。"""
    client = LLMClient(LLMConfig(mock=True))
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": 123}])  # content 非字符串

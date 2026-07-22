"""OpenAI 兼容的 LLM 客户端，支持真实调用与 MOCK 降级模式。

默认指向本地 Ollama（http://localhost:11434/v1），可通过环境变量覆盖：
- OPENAI_BASE_URL：API 地址
- OPENAI_API_KEY：API Key
- OPENAI_MODEL：模型名
- MOCK_LLM=1：不调用真实接口，返回基于检索结果的 stub 答案
- LLM_TIMEOUT：请求超时（秒，默认 30）
- LLM_MAX_TOKENS：最大生成 token（默认 1024）
- LLM_TEMPERATURE：采样温度（默认 0.2）

设计要点（供 self-driving 循环审计）：
- 所有真实调用错误统一包装为 LLMError，避免把底层 SDK 异常直接抛给用户。
- 每次成功调用后把 token 用量写入 self.last_usage，作为可观测性基线。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:latest"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RETRIES = 1           # 瞬态错误的重试次数（不含首次）
DEFAULT_BACKOFF = 0.2         # 指数退避基延迟（秒）

# 判定为「值得重试」的瞬态错误关键词（网络抖动 / 限流 / 5xx）
_TRANSIENT_KEYWORDS = (
    "timeout", "timed out", "connection", "reset by peer", "broken pipe",
    "429", "rate limit", "too many requests", "503", "502", "500",
    "temporary", "try again", "econnrefused", "etimedout",
)


class LLMError(RuntimeError):
    """LLM 调用 / 配置相关的统一异常，包装底层 SDK 异常，便于上层友好处理。"""


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    mock: bool = field(default_factory=lambda: os.getenv("MOCK_LLM", "0") in ("1", "true", "True"))
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))))
    retries: int = field(default_factory=lambda: int(os.getenv("LLM_RETRIES", str(DEFAULT_RETRIES))))
    backoff: float = field(default_factory=lambda: float(os.getenv("LLM_BACKOFF", str(DEFAULT_BACKOFF))))


class LLMClient:
    """对 OpenAI 兼容 SDK 的轻封装。

    当 config.mock 为 True 时，调用 answer() 不会真正请求网络，
    而是根据传入的上下文文件列表构造一个 stub 答案，方便无 key 演示。
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self.last_usage: Optional[dict] = None  # 可观测性：最近一次 token 用量
        self.last_attempts: int = 0             # 可观测性：最近一次调用实际尝试次数
        self.last_error: Optional[str] = None   # 可观测性：最近一次失败原因（成功则为 None）

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算 token 数（供上下文预算评估 / 可观测性）。

        中文/日文等 CJK 字符按「每字 1 token」估算（与主流 tokenizer 接近）；
        其它字符按「每 4 字符 ≈ 1 token」估算。结果向上取整，至少为 0。
        """
        if not text:
            return 0
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        others = len(text) - cjk
        return cjk + (others + 3) // 4

    @staticmethod
    def _validate_messages(messages: List[dict]) -> None:
        """校验 messages 结构，避免把畸形消息直接丢给 SDK 触发 5xx。

        非法（缺 role/content、role 不在白名单、非 dict）一律包装为 LLMError。
        """
        valid_roles = {"system", "user", "assistant"}
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                raise LLMError(f"消息 #{i} 不是对象：{type(m).__name__}")
            role = m.get("role")
            if role not in valid_roles:
                raise LLMError(f"消息 #{i} 的 role 非法：{role!r}（应为 system/user/assistant）")
            if not isinstance(m.get("content"), str):
                raise LLMError(f"消息 #{i}（role={role}）缺少字符串 content")

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # 隐性问题：依赖缺失应给出明确错误而非裸 ImportError
                raise LLMError("未安装 openai 依赖，请先 `pip install openai`") from exc
            try:
                self._client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            except Exception as exc:  # 初始化失败（地址/密钥/网络）也统一包装
                raise LLMError(f"初始化 LLM 客户端失败：{exc}") from exc
        return self._client

    def complete(
        self,
        messages: List[dict],
        context_files: List[str] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """底层多轮问答入口：直接接收已构造好的 messages 列表。

        messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
        context_files: 仅用于 mock 答案的「参考文件」展示
        system_prompt: 自定义系统提示（覆盖默认；不传则用仓库助手默认提示）

        成功返回助手文本；失败抛出 LLMError（绝不裸抛 SDK 异常）。
        """
        if not messages:
            # 隐性健壮性：空消息列表既无意义也可能让部分 SDK 报错，提前拦截
            raise LLMError("消息列表为空，无法调用 LLM")

        # 隐性健壮性：畸形消息（非法 role / 缺 content）提前拦截，避免触发 SDK 5xx
        self._validate_messages(messages)

        if self.config.mock:
            question = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    question = m.get("content", "")
                    break
            # 可观测性一致性：mock 模式也更新探针字段，避免调用方无法区分 mock/真实
            self.last_attempts = 1
            self.last_error = None
            est = self.estimate_tokens(question)
            self.last_usage = {"prompt_tokens": est, "completion_tokens": 0, "total_tokens": est}
            return self._mock_answer(question, context_files or [])

        sys_prompt = system_prompt or (
            "你是一个帮助理解代码仓库的助手。请基于下面提供的仓库上下文片段，"
            "用中文准确、简洁地回答用户的问题。如果上下文不足以回答，请如实说明。"
        )
        full = [{"role": "system", "content": sys_prompt}] + list(messages)

        client = self._get_client()
        message = ""
        last_exc: Optional[Exception] = None
        max_attempts = max(1, self.config.retries + 1)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                resp = client.chat.completions.create(
                    model=self.config.model,
                    messages=full,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as exc:  # 隐性问题：网络/限流/鉴权错误需被捕获并包装
                last_exc = exc
                self.last_attempts = attempts
                self.last_error = str(exc)
                # 仅对「瞬态」错误重试；非瞬态（鉴权/参数）立即放弃，避免无效重试
                if attempts >= max_attempts or not self._is_transient(exc):
                    break
                # 指数退避后重试，提升对网络抖动/限流的韧性
                time.sleep(self.config.backoff * (2 ** (attempts - 1)))
                continue
            # 成功路径
            self.last_attempts = attempts
            self.last_error = None
            message = resp.choices[0].message.content or ""
            # 可观测性：记录 token 用量（部分兼容接口可能不返回 usage）
            try:
                self.last_usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            except AttributeError:
                self.last_usage = None
            return message

        # 重试耗尽或全部为非瞬态错误：统一包装为 LLMError
        retried = attempts - 1
        suffix = f"（已重试 {retried} 次）" if retried > 0 else ""
        raise LLMError(f"LLM 调用失败{suffix}：{last_exc}") from last_exc

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """判定异常是否为可重试的瞬态错误（网络抖动 / 限流 / 5xx）。"""
        return any(k in str(exc).lower() for k in _TRANSIENT_KEYWORDS)

    def answer(
        self,
        question: str,
        context_files: List[str],
        context_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """基于检索到的上下文作答（单轮封装，内部走 complete 多轮入口）。

        context_files: 命中的文件路径列表（用于参考文件展示 / mock 答案）
        context_text: 拼好的上下文片段文本
        system_prompt: 可覆盖默认系统提示
        成功返回文本；失败抛出 LLMError（绝不裸抛 SDK 异常）。
        """
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"=== 仓库上下文（来自 {len(context_files)} 个文件）===\n{context_text}\n"
            "=== 上下文结束 ===\n请回答上面的问题。"
        )
        return self.complete(
            [{"role": "user", "content": user_prompt}],
            context_files=context_files,
            system_prompt=system_prompt,
        )

    @staticmethod
    def _mock_answer(question: str, context_files: List[str]) -> str:
        if not context_files:
            return (
                "（MOCK 模式）未检索到相关文件。请先运行 `index <path>` 建立索引，"
                "或确认问题与仓库内容相关。"
            )
        file_list = "\n".join(f"  - {f}" for f in context_files)
        return (
            f"（MOCK 模式）根据检索结果，关于「{question}」的相关内容可能分布在以下文件里：\n"
            f"{file_list}\n\n"
            "（这是离线 stub 答案；配置 OPENAI_BASE_URL / OPENAI_API_KEY 后即可获得真实 LLM 回答。）"
        )

    def health(self) -> dict:
        """探针：检测 LLM 端点可用性，返回结构化状态字典（可观测性 / 新能力）。

        返回字段：
          ok:      端点是否可用（mock 模式恒为 True）
          mock:    是否处于 mock 模式
          model:   目标模型名
          attempts:真实调用实际尝试次数（mock 为 0）
          error:   失败原因（成功为 None）
        mock 模式不触网，直接返回可用；真实模式发一次最小请求探测。
        """
        if self.config.mock:
            return {"ok": True, "mock": True, "model": self.config.model, "attempts": 0, "error": None}
        try:
            resp = self.complete(
                [{"role": "user", "content": "ping"}],
                system_prompt="你是一个健康检查探针，只回复 OK 两个字母。",
            )
            return {
                "ok": bool(resp),
                "mock": False,
                "model": self.config.model,
                "attempts": self.last_attempts,
                "error": None,
            }
        except LLMError as exc:
            return {
                "ok": False,
                "mock": False,
                "model": self.config.model,
                "attempts": self.last_attempts,
                "error": str(exc),
            }

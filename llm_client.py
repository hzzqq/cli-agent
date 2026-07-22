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
from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:latest"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2


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


class LLMClient:
    """对 OpenAI 兼容 SDK 的轻封装。

    当 config.mock 为 True 时，调用 answer() 不会真正请求网络，
    而是根据传入的上下文文件列表构造一个 stub 答案，方便无 key 演示。
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        self.last_usage: Optional[dict] = None  # 可观测性：最近一次 token 用量

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

    def answer(self, question: str, context_files: List[str], context_text: str) -> str:
        """基于检索到的上下文作答。

        context_files: 命中的文件路径列表（用于参考文件展示 / mock 答案）
        context_text: 拼好的上下文片段文本
        成功返回文本；失败抛出 LLMError（绝不裸抛 SDK 异常）。
        """
        if self.config.mock:
            return self._mock_answer(question, context_files)

        system_prompt = (
            "你是一个帮助理解代码仓库的助手。请基于下面提供的仓库上下文片段，"
            "用中文准确、简洁地回答用户的问题。如果上下文不足以回答，请如实说明。"
        )
        user_prompt = (
            f"用户问题：{question}\n\n"
            f"=== 仓库上下文（来自 {len(context_files)} 个文件）===\n{context_text}\n"
            "=== 上下文结束 ===\n请回答上面的问题。"
        )

        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:  # 隐性问题：网络/限流/鉴权错误需被捕获并包装
            raise LLMError(f"LLM 调用失败：{exc}") from exc

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

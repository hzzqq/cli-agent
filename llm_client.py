"""OpenAI 兼容的 LLM 客户端，支持真实调用与 MOCK 降级模式。

默认指向本地 Ollama（http://localhost:11434/v1），可通过环境变量覆盖：
- OPENAI_BASE_URL：API 地址
- OPENAI_API_KEY：API Key
- MOCK_LLM=1：不调用真实接口，返回基于检索结果的 stub 答案
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:latest"


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    mock: bool = field(default_factory=lambda: os.getenv("MOCK_LLM", "0") in ("1", "true", "True"))


class LLMClient:
    """对 OpenAI 兼容 SDK 的轻封装。

    当 config.mock 为 True 时，调用 answer() 不会真正请求网络，
    而是根据传入的上下文文件列表构造一个 stub 答案，方便无 key 演示。
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )
        return self._client

    def answer(self, question: str, context_files: List[str], context_text: str) -> str:
        """基于检索到的上下文作答。

        context_files: 命中的文件路径列表（用于参考文件展示 / mock 答案）
        context_text: 拼好的上下文片段文本
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
        resp = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

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

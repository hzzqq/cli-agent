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
DEFAULT_TOP_P = 1.0              # nucleus 采样阈值（1.0 = 关闭，等价于贪心）
DEFAULT_FREQUENCY_PENALTY = 0.0  # 频率惩罚（抑制重复词，0 = 关闭）
DEFAULT_PRESENCE_PENALTY = 0.0   # 存在惩罚（鼓励新话题，0 = 关闭）
DEFAULT_RETRIES = 1           # 瞬态错误的重试次数（不含首次）
DEFAULT_BACKOFF = 0.2         # 指数退避基延迟（秒）
DEFAULT_MAX_CONTEXT_TOKENS = 12000  # 发送给模型的历史 token 预算上限（防上下文溢出）


def format_usage(usage: "Optional[dict]", model: "str | None" = None,
                 price_prompt: "float | None" = None,
                 price_completion: "float | None" = None) -> str:
    """把 token 用量格式化为一行可读摘要（供 CLI 把真实消耗呈现给用户）。

    纯函数、可单测。usage 缺省（如接口未返回用量）时返回空串，调用方据此跳过。
    传入 price_prompt / price_completion（每 1k token 价格）时额外给出成本估算，
    便于成本管控（模型定价各异，未提供则不显示金额）。
    """
    if not usage:
        return ""
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tot = int(usage.get("total_tokens", 0) or (pt + ct))
    parts = [f"token 总 {tot}（提示 {pt} / 补全 {ct}）"]
    if model:
        parts.append(f"模型 {model}")
    if price_prompt is not None or price_completion is not None:
        cost = (pt / 1000.0) * (price_prompt or 0.0) + (ct / 1000.0) * (price_completion or 0.0)
        parts.append(f"≈ ${cost:.4f}")
    return " · ".join(parts)


# 默认系统提示（R2 修复重复 system 块时复用，避免多处硬编码同一字符串）
DEFAULT_SYSTEM_PROMPT = (
    "你是一个帮助理解代码仓库的助手。请基于下面提供的仓库上下文片段，"
    "用中文准确、简洁地回答用户的问题。如果上下文不足以回答，请如实说明。"
)

# 判定为「值得重试」的瞬态错误关键词（网络抖动 / 限流 / 5xx）
_TRANSIENT_KEYWORDS = (
    "timeout", "timed out", "connection", "reset by peer", "broken pipe",
    "429", "rate limit", "too many requests", "503", "502", "500",
    "temporary", "try again", "econnrefused", "etimedout",
)


def _compose_messages(messages, system_prompt):
    """构造最终发给模型的 messages 列表。

    R2 修复（隐性缺陷）：原先无条件前置 system 提示，若调用方已在
    messages[0] 提供 system 消息，会产生「双 system 块」——既挤占 token，
    又让模型对重复系统提示的行为不稳定。这里去重：
      - 调用方已带 system 且未显式传 system_prompt → 保留调用方的 system；
      - 调用方已带 system 且显式传 system_prompt → 以参数覆盖调用方 system；
      - 调用方未带 system → 前置 system_prompt 或默认提示。
    """
    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        if system_prompt is not None:
            msgs[0] = {"role": "system", "content": system_prompt}
        return msgs
    return [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}] + msgs


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
    top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", str(DEFAULT_TOP_P))))
    frequency_penalty: float = field(default_factory=lambda: float(os.getenv("LLM_FREQUENCY_PENALTY", str(DEFAULT_FREQUENCY_PENALTY))))
    presence_penalty: float = field(default_factory=lambda: float(os.getenv("LLM_PRESENCE_PENALTY", str(DEFAULT_PRESENCE_PENALTY))))
    retries: int = field(default_factory=lambda: int(os.getenv("LLM_RETRIES", str(DEFAULT_RETRIES))))
    backoff: float = field(default_factory=lambda: float(os.getenv("LLM_BACKOFF", str(DEFAULT_BACKOFF))))
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_CONTEXT_TOKENS", str(DEFAULT_MAX_CONTEXT_TOKENS)))
    )


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
    def trim_messages_to_budget(messages: List[dict], max_tokens: int) -> "tuple[List[dict], bool]":
        """把消息列表裁剪到 token 预算内，避免超大历史触发模型上下文溢出（400 错误）。

        隐性问题：chat 多轮累积后，拼上 system 提示的 messages 可能远超模型
        context window，直接抛 400 且错误信息晦涩、用户无从定位。这里在发送前
        按预算裁剪——始终保留 system 消息（系统提示优先级最高），超出时从最旧的
        user/assistant 消息开始丢弃，直到满足预算或仅剩最后一条非 system 消息
        （保证至少有一条用户/助手内容可被模型消费）。

        返回 (裁剪后的列表, 是否发生过裁剪)。max_tokens<=0 视为不限制。
        """
        if max_tokens <= 0:
            return list(messages), False
        msgs = list(messages)
        est = sum(LLMClient.estimate_tokens(m.get("content", "") or "") for m in msgs)
        if est <= max_tokens:
            return msgs, False
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        rest = [m for m in msgs if m.get("role") != "system"]
        while len(rest) > 1:
            est = sum(LLMClient.estimate_tokens(m.get("content", "") or "") for m in system_msgs + rest)
            if est <= max_tokens:
                break
            rest.pop(0)  # 丢弃最旧的对话消息，保留更近的上下文
        return system_msgs + rest, True

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

        full = _compose_messages(messages, system_prompt)
        # R2 隐性问题修复：超大历史会撑爆上下文窗口导致晦涩的 400 错误，
        # 发送前按 token 预算裁剪（保留 system 与最近的对话）。
        full, _trimmed = LLMClient.trim_messages_to_budget(full, self.config.max_context_tokens)

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
                    **self._build_sampling_params(),
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

    def stream_complete(
        self,
        messages: List[dict],
        context_files: List[str] | None = None,
        system_prompt: str | None = None,
    ):
        """流式多轮问答入口：逐 token 产出（generator），供 CLI 实时打印。

        与 complete 的区别：complete 聚合后整体返回字符串；stream_complete
        逐个 yield 片段，便于「边生成边显示」的交互体验（R1 新能力）。
        失败时同样统一抛 LLMError（绝不裸抛 SDK 异常）。
        """
        if not messages:
            raise LLMError("消息列表为空，无法调用 LLM")
        self._validate_messages(messages)

        if self.config.mock:
            # 复用 complete（含其对 complete 的 mock 与全部可观测字段），
            # 仅把结果逐字符 yield 出去；真实后端才走真正的增量流。
            # 这样测试对 complete 的 mock 也能作用于流式路径。
            ans = self.complete(
                messages, context_files=context_files, system_prompt=system_prompt
            )
            for ch in ans:
                yield ch
            return

        full = _compose_messages(messages, system_prompt)
        # R2 隐性问题修复：与 complete 一致，流式发送前也按 token 预算裁剪，
        # 避免超大历史触发模型上下文溢出。
        full, _trimmed = LLMClient.trim_messages_to_budget(full, self.config.max_context_tokens)
        client = self._get_client()
        # R1/R2 修复（隐性韧性不一致）：原 stream_complete 在建立流时若遇
        # 网络抖动/限流/5xx，会直接抛错让「边生成边显示」中断；而 complete
        # 早已具备「瞬态错误重试 + 指数退避」。这里把流式路径补齐到同一套
        # 重试逻辑——仅对 create（建流）阶段的瞬态错误重试；一旦流已开始产出
        # 数据则不再重试（避免重复吐字）。重试耗尽或非瞬态错误统一包装为 LLMError。
        last_exc: Optional[Exception] = None
        max_attempts = max(1, self.config.retries + 1)
        attempts = 0
        stream = None
        while attempts < max_attempts:
            attempts += 1
            try:
                stream = client.chat.completions.create(
                    model=self.config.model,
                    messages=full,
                    stream=True,
                    stream_options={"include_usage": True},
                    **self._build_sampling_params(),
                )
                self.last_attempts = attempts
                self.last_error = None
                break
            except Exception as exc:  # 建流失败：判定是否可重试
                last_exc = exc
                self.last_attempts = attempts
                self.last_error = str(exc)
                if attempts >= max_attempts or not self._is_transient(exc):
                    break
                time.sleep(self.config.backoff * (2 ** (attempts - 1)))
                continue
        if stream is None:  # 重试耗尽或全部非瞬态
            retried = attempts - 1
            suffix = f"（已重试 {retried} 次）" if retried > 0 else ""
            raise LLMError(f"LLM 流式调用失败{suffix}：{last_exc}") from last_exc
        try:
            for chunk in stream:
                # R2 修复（隐性可观测性缺口）：流式路径此前不捕获 token 用量，
                # 导致 last_usage 在流式（ask/chat 默认）下恒为 None，用量/成本
                # 完全不可见。OpenAI 兼容流在尾部块（choices 为空）携带 usage，
                # 这里显式捕获并归一化为与 complete 一致的字段口径。
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    self.last_usage = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
        except Exception as exc:  # 流中途失败也统一包装
            raise LLMError(f"LLM 流式调用失败：{exc}") from exc

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """判定异常是否为可重试的瞬态错误（网络抖动 / 限流 / 5xx）。"""
        return any(k in str(exc).lower() for k in _TRANSIENT_KEYWORDS)

    def _build_sampling_params(self) -> dict:
        """构造统一的采样参数 dict，供 complete / stream_complete 共用。

        R2/R3（一致性 + DRY）：原先 top_p/temperature/max_tokens 由两条生成
        链各自拼 kwargs，极易出现「一处新增参数、另一处漏加」的分叉（c112 的
        regenerate 就曾与 chat 参数脱节）。集中到一个构造器，保证口径永远一致；
        None 值不发送，便于「不配置即走模型默认」。
        """
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        return {k: v for k, v in params.items() if v is not None}

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
        # R2 修复（隐性可观测性污染）：health() 内部复用 complete() 探测，而
        # complete() 会覆写 last_usage / last_attempts / last_error 三个可观测字段。
        # 若用户在「一次真实 ask」之后调用 `config --check`（触发 health），探针结果
        # 会覆盖真实答案的统计，导致下游读取到错误的 token 用量/尝试次数。这里在
        # 探测前快照、探测后还原，保证 health() 不影响调用方对真实问答的可观测性。
        snap = (self.last_usage, self.last_attempts, self.last_error)
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
        finally:
            self.last_usage, self.last_attempts, self.last_error = snap

    def list_models(self) -> List[str]:
        """列出端点可用的模型 id（R1 新能力，供 CLI `models` 命令 / 前端模型下拉复用）。

        mock 模式不触网，直接返回 [当前模型]；真实模式查询 /models 端点；
        失败统一包装为 LLMError（绝不裸抛 SDK 异常）。
        """
        if self.config.mock:
            return [self.config.model]
        try:
            client = self._get_client()
            resp = client.models.list()
            return [m.id for m in getattr(resp, "data", [])]
        except Exception as exc:
            raise LLMError(f"获取模型列表失败：{exc}") from exc

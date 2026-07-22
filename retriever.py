"""检索逻辑：基于关键词在索引里召回最相关的 top-K 文件。

这是一个轻量级 BM25 风格的实现一个简化版：
- 对问题与文件 snippet 做 token 化（小写、按空白与标点切分）。
- 用词频（TF）与逆文档频率（IDF）计算相似度，取分数最高的 K 个文件。
无需外部依赖，足以支撑 MVP 的"相关文件召回"。
"""

from __future__ import annotations

import math
import re
from typing import List

from index_store import IndexEntry, load_index


_TOOKEN_RE = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")
# CJK 字符区间（含常用汉字）
_CJK_MIN, _CJK_MAX = "\u4e00", "\u9fff"


def _is_cjk(ch: str) -> bool:
    return _CJK_MIN <= ch <= _CJK_MAX


def _tokenize(text: str) -> List[str]:
    """分词：英文/数字按原词，中文做「整词 + 单字 + 相邻二元」展开。

    原实现把「关于foo的说明」当成一个整 token，导致 query「关于」
    无法命中（精确 token 匹配失效）。这里对含 CJK 的 token 额外生成
    单字与相邻 bigram，使中文短语的部分匹配也能召回。
    """
    raw = [t.lower() for t in _TOOKEN_RE.findall(text or "")]
    out: List[str] = []
    for tok in raw:
        out.append(tok)
        cjk_chars = [ch for ch in tok if _is_cjk(ch)]
        if len(cjk_chars) >= 2:
            out.extend(cjk_chars)  # 单字
            for i in range(len(cjk_chars) - 1):
                out.append(cjk_chars[i] + cjk_chars[i + 1])  # 相邻二元
    return out


def _build_documents(entries: List[IndexEntry]):
    """把每个索引条目拼成一篇文档（路径 + snippet），返回 (entry, tokens)。"""
    docs = []
    for e in entries:
        text = f"{e.path}\n{e.snippet}"
        docs.append((e, _tokenize(text)))
    return docs


def _idf(term: str, docs_tokens: List[List[str]]) -> float:
    df = sum(1 for toks in docs_tokens if term in toks)
    if df == 0:
        return 0.0
    return math.log((len(docs_tokens) + 1) / (df + 1)) + 1.0


def retrieve_scored(question: str, top_k: int = 5, min_score: float = 0.0):
    """根据问题召回 top-K 相关文件，并返回每项的相关度分数。

    返回 [(entry, score), ...] 按分数降序。min_score 用于过滤弱相关命中
    （隐性问题：无阈值会把噪声文件一并送入 LLM 上下文，拉低回答质量）。
    """
    entries = load_index()
    if not entries:
        return []

    docs = _build_documents(entries)
    docs_tokens = [toks for _, toks in docs]
    idf_cache = {}

    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    scored = []
    for entry, toks in docs:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for qt in set(q_tokens):
            if qt not in tf:
                continue
            if qt not in idf_cache:
                idf_cache[qt] = _idf(qt, docs_tokens)
            # 简单 TF-IDF
            score += (tf[qt] / max(len(toks), 1)) * idf_cache[qt]
        if score >= min_score:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(e, s) for s, e in scored[:top_k]]


def retrieve(question: str, top_k: int = 5, min_score: float = 0.0) -> List[IndexEntry]:
    """根据问题召回 top-K 相关文件。找不到索引时返回空列表。"""
    return [e for e, _ in retrieve_scored(question, top_k=top_k, min_score=min_score)]


# 上下文拼接上限（字符），防止超大仓库把所有片段塞进 prompt 撑爆 LLM 上下文
MAX_CONTEXT_CHARS = 6000


def build_context(question: str, top_k: int = 5, min_score: float = 0.0) -> tuple[str, List[str]]:
    """返回 (拼接好的上下文文本, 命中的文件路径列表)。

    拼接受到 MAX_CONTEXT_CHARS 上限约束：一旦累计超过上限且已有内容，
    则停止追加后续文件（保证最相关的文件优先进入上下文）。
    每个块头部标注相关度分数，提升上下文可读性与可观测性。
    """
    hits = retrieve_scored(question, top_k=top_k, min_score=min_score)
    paths = [e.path for e, _ in hits]
    if not hits:
        return "", []
    parts = []
    total = 0
    for e, score in hits:
        block = (
            f"--- 文件: {e.path} (大小: {e.size} 字节, 相关度: {score:.2f}) ---\n"
            f"{e.snippet}"
        )
        # 已有内容且加入会超上限时，停止（保留高相关片段）
        if parts and total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts), paths

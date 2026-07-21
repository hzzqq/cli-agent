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


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


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


def retrieve(question: str, top_k: int = 5) -> List[IndexEntry]:
    """根据问题召回 top-K 相关文件。找不到索引时返回空列表。"""
    entries = load_index()
    if not entries:
        return []

    docs = _build_documents(entries)
    docs_tokens = [toks for _, toks in docs]
    idf_cache = {}

    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    scores = []
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
        if score > 0:
            scores.append((score, entry))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scores[:top_k]]


# 上下文拼接上限（字符），防止超大仓库把所有片段塞进 prompt 撑爆 LLM 上下文
MAX_CONTEXT_CHARS = 6000


def build_context(question: str, top_k: int = 5) -> tuple[str, List[str]]:
    """返回 (拼接好的上下文文本, 命中的文件路径列表)。

    拼接受到 MAX_CONTEXT_CHARS 上限约束：一旦累计超过上限且已有内容，
    则停止追加后续文件（保证最相关的文件优先进入上下文）。
    """
    hits = retrieve(question, top_k=top_k)
    paths = [e.path for e in hits]
    if not hits:
        return "", []
    parts = []
    total = 0
    for e in hits:
        block = f"--- 文件: {e.path} (大小: {e.size} 字节) ---\n{e.snippet}"
        # 已有内容且加入会超上限时，停止（保留高相关片段）
        if parts and total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts), paths

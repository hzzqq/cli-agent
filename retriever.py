"""检索逻辑：基于关键词在索引里召回最相关的 top-K 文件。

这是一个轻量级 BM25 风格的实现一个简化版：
- 对问题与文件 snippet 做 token 化（小写、按空白与标点切分）。
- 用词频（TF）与逆文档频率（IDF）计算相似度，取分数最高的 K 个文件。
无需外部依赖，足以支撑 MVP 的"相关文件召回"。
"""

from __future__ import annotations

import math
import re
from collections import Counter
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
    查询侧使用词频（Counter）加权：重复同一关键词会使相关文件得分更高，
    修复此前用 set(q_tokens) 丢弃查询词频、导致「python python」与
    「python」权重相同的隐性打分缺陷。
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
    q_freq = Counter(q_tokens)  # 查询侧词频：修复 set 丢弃词频的隐性打分缺陷

    scored = []
    for entry, toks in docs:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for qt, qf in q_freq.items():
            if qt not in tf:
                continue
            if qt not in idf_cache:
                idf_cache[qt] = _idf(qt, docs_tokens)
            # TF-IDF（含查询侧词频 qf）：重复提问同一关键词时该文件得分更高
            score += qf * (tf[qt] / max(len(toks), 1)) * idf_cache[qt]
        if score >= min_score:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(e, s) for s, e in scored[:top_k]]


def retrieve(question: str, top_k: int = 5, min_score: float = 0.0) -> List[IndexEntry]:
    """根据问题召回 top-K 相关文件。找不到索引时返回空列表。"""
    return [e for e, _ in retrieve_scored(question, top_k=top_k, min_score=min_score)]


def explain_retrieval(question: str, top_k: int = 5, min_score: float = 0.0) -> List[dict]:
    """返回每篇命中文件的匹配关键词与分数，便于排查检索质量（透明性 / 可观测）。

    返回 [{"path": str, "score": float, "terms": List[str]}, ...]，
    供 context 命令或外部工具展示「为什么召回了这些文件」。
    """
    hits = retrieve_scored(question, top_k=top_k, min_score=min_score)
    q_set = set(_tokenize(question))
    out: List[dict] = []
    for e, score in hits:
        doc_tokens = set(_tokenize(f"{e.path}\n{e.snippet}"))
        out.append({
            "path": e.path,
            "score": score,
            "terms": sorted(q_set & doc_tokens),
        })
    return out


# 上下文拼接上限（字符），防止超大仓库把所有片段塞进 prompt 撑爆 LLM 上下文
MAX_CONTEXT_CHARS = 6000
# 单个文件片段上限（字符）：防止某个巨型文件独占上下文、挤掉其它相关文件
MAX_BLOCK_CHARS = 2000


def build_context(
    question: str,
    top_k: int = 5,
    min_score: float = 0.0,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_block_chars: int = MAX_BLOCK_CHARS,
) -> tuple[str, List[str]]:
    """返回 (拼接好的上下文文本, 实际进入上下文的文件路径列表)。

    拼接受到 max_context_chars 上限约束：一旦累计超过上限且已有内容，
    则停止追加后续文件（保证最相关的文件优先进入上下文）。
    每个块头部标注相关度分数，提升上下文可读性与可观测性。

    R1 新增：max_context_chars / max_block_chars 可被调用方覆盖，
    使 ask/context/chat 的 --max-context-chars 选项可调节上下文预算。

    R2 修复（隐性可观测性缺陷）：此前 paths 返回的是「全部命中文件」，
    但部分文件因预算上限被丢弃、并未真正进入上下文，导致 ask 的
    「参考文件」列表与 --json 的 references、--verbose 命中数都虚高。
    现改为只返回实际被加入上下文的文件路径（included_paths）。
    """
    hits = retrieve_scored(question, top_k=top_k, min_score=min_score)
    if not hits:
        return "", []
    parts = []
    total = 0
    included_paths: List[str] = []
    for e, score in hits:
        snippet = e.snippet
        truncated = False
        if len(snippet) > max_block_chars:
            snippet = snippet[:max_block_chars]
            truncated = True
        block = (
            f"--- 文件: {e.path} (大小: {e.size} 字节, 相关度: {score:.2f}) ---\n"
            f"{snippet}"
        )
        if truncated:
            block += "\n...(该文件片段过长，已截断)"
        # 已有内容且加入会超上限时，停止（保留高相关片段）
        if parts and total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)
        included_paths.append(e.path)
    return "\n\n".join(parts), included_paths

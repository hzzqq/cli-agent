"""索引的构建与持久化。

索引结构存到本地 `.cliagent_index.json`：
- files: 每个文件的 path / size / snippet（头部若干行）
- indexed_at: 建立时间

支持增量安全：未索引时 load_index() 返回 []。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List


INDEX_FILE = ".cliagent_index.json"

# 视为文本、参与索引的扩展名
TEXT_EXTS = {
    ".py", ".pyw", ".md", ".txt", ".rst", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".html",
    ".css", ".sh", ".bash", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".rs", ".rb", ".php", ".sql", ".xml",
}

# 跳过的目录
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode", "dist", "build"}

# 单文件最大读取字节（防止超大文件），超过则只截取头部
MAX_FILE_BYTES = 200_000
SNIPPET_LINES = 40


@dataclass
class IndexEntry:
    path: str
    size: int
    snippet: str


def _iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # 就地修改 dirnames 以跳过无需遍历的目录
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in TEXT_EXTS:
                yield os.path.join(dirpath, fname)


def _read_snippet(path: str) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if size > MAX_FILE_BYTES:
                # 只读取前若干行
                lines = []
                for i, line in enumerate(f):
                    if i >= SNIPPET_LINES:
                        break
                    lines.append(line.rstrip("\n"))
                lines.append(f"... (文件较大 {size} 字节，仅索引前 {SNIPPET_LINES} 行)")
                return "\n".join(lines)
            # 小文件读取前 SNIPPET_LINES 行即可，避免把整个大文件塞进 prompt
            lines = []
            for i, line in enumerate(f):
                if i >= SNIPPET_LINES:
                    break
                lines.append(line.rstrip("\n"))
            return "\n".join(lines)
    except (OSError, UnicodeDecodeError):
        return ""


def build_index(root: str, exts: "set[str] | None" = None) -> List[IndexEntry]:
    """遍历 root，建立索引并返回条目列表。

    exts：可选扩展名白名单（小写，含点，如 {'.py', '.md'}）。提供时只索引这些类型。
    """
    entries: List[IndexEntry] = []
    for path in _iter_files(root):
        if exts is not None:
            ext = os.path.splitext(path)[1].lower()
            if ext not in exts:
                continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        snippet = _read_snippet(path)
        entries.append(IndexEntry(path=path, size=size, snippet=snippet))
    return entries


def save_index(entries: List[IndexEntry], root: str = ".") -> str:
    out_path = os.path.join(root, INDEX_FILE)
    payload = {
        "indexed_at": _dt.datetime.now().isoformat(),
        "files": [asdict(e) for e in entries],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def load_index(index_path: str = INDEX_FILE) -> List[IndexEntry]:
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [IndexEntry(**item) for item in data.get("files", [])]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def search_index(keyword: str, index_path: str = INDEX_FILE) -> List[IndexEntry]:
    """在索引的 snippet / 路径中按关键词（大小写不敏感）检索。

    用于「不调用 LLM」时快速定位相关文件，是对 retriever 的轻量补充。
    """
    if not keyword:
        return []
    kw = keyword.lower()
    hits = []
    for e in load_index(index_path):
        if kw in (e.snippet or "").lower() or kw in e.path.lower():
            hits.append(e)
    return hits


def prune_missing(root: str, index_path: str = INDEX_FILE) -> int:
    """移除索引中已不存在的文件条目（陈旧索引清理），返回被移除数量。

    就地重写索引；无匹配时不动文件。
    """
    entries = load_index(index_path)
    if not entries:
        return 0
    kept = [e for e in entries if os.path.exists(e.path)]
    removed = len(entries) - len(kept)
    if removed:
        out_dir = os.path.dirname(os.path.abspath(index_path)) or "."
        save_index(kept, out_dir)
    return removed


def index_stats(index_path: str = INDEX_FILE) -> "Dict | None":
    """返回索引统计信息；无索引时返回 None（供 CLI 友好提示）。"""
    entries = load_index(index_path)
    if not entries:
        return None
    total = sum(e.size for e in entries)
    ext_counter: Counter = Counter()
    for e in entries:
        ext = os.path.splitext(e.path)[1].lower() or "(无扩展名)"
        ext_counter[ext] += 1
    indexed_at = ""
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            indexed_at = json.load(f).get("indexed_at", "")
    except Exception:
        pass
    return {
        "file_count": len(entries),
        "total_bytes": total,
        "top_extensions": sorted(ext_counter.items(), key=lambda x: -x[1]),
        "indexed_at": indexed_at,
    }

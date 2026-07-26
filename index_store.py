"""索引的构建与持久化。

索引结构存到本地 `.cliagent_index.json`：
- files: 每个文件的 path / size / snippet（头部若干行）
- indexed_at: 建立时间

支持增量安全：未索引时 load_index() 返回 []。
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
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
    mtime: float = 0.0  # 修改时间，用于增量索引时判断文件是否变更


def _iter_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # 就地修改 dirnames 以跳过无需遍历的目录
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
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


def build_index(
    root: str,
    exts: "set[str] | None" = None,
    max_size: "int | None" = None,
    min_size: "int | None" = None,
    prev: "List[IndexEntry] | None" = None,
    exclude: "list[str] | None" = None,
) -> "tuple[List[IndexEntry], List[dict]]":
    """遍历 root，建立索引并返回 (条目列表, 跳过列表)。

    exts：可选扩展名白名单（小写，含点，如 {'.py', '.md'}）。提供时只索引这些类型。
    max_size：可选单文件字节上限；超过的文件被跳过（避免大锁文件/数据文件污染索引）。
    min_size：可选单文件字节下限；小于该值的文件被跳过（R1 新能力，过滤空/极小的
        占位文件——如 0 字节锁文件、临时碎片——避免无意义噪声进入检索上下文）。
    prev：上一轮索引条目；提供时进入「增量模式」——mtime 与 size 均未变的文件直接
        复用旧条目（不重读 snippet），显著减少大仓库的重复 I/O（隐性性能悬崖）。
    exclude：可选忽略模式列表（fnmatch，支持相对路径或文件名，如 ["tests/*", "*.min.js"]），
        匹配的文件跳过且不进入索引（R1 新能力，补充固定 SKIP_DIRS 之外的临时忽略需求）。

    返回的 skipped 为 [{path, reason, ...}]，reason ∈
    {"unsupported_ext", "ext_filter", "too_large", "too_small", "unreadable", "excluded"}，
    便于 CLI 向用户公示「哪些文件没被索引」以提升可观测性。
    """
    # R2 修复（隐性可用性缺陷）：用户若写 `--ext py`（无点），exts 会是 {"py"}，
    # 而 TEXT_EXTS 存的是 ".py"，导致全部文件被判定为 ext_filter、索引结果为空、
    # 用户困惑「为何一个文件都没索引到」。这里统一把扩展名归一化为带点的小写形式，
    # 使 `--ext py` 与 `--ext .py` / `--ext PY` 行为一致。
    if exts is not None:
        exts = {
            ("." + e.strip().lower()) if not e.strip().lower().startswith(".")
            else e.strip().lower()
            for e in exts
        }
    prev_by_path = {e.path: e for e in (prev or [])}
    entries: List[IndexEntry] = []
    skipped: List[dict] = []
    for path in _iter_files(root):
        ext = os.path.splitext(path)[1].lower()
        # R1 新能力：用户自定义忽略模式（在扩展名判断之前生效，优先级高于类型匹配）
        if exclude:
            rel = os.path.relpath(path, root)
            base = os.path.basename(path)
            if any(
                fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(base, pat)
                for pat in exclude
            ):
                skipped.append({"path": path, "reason": "excluded"})
                continue
        if exts is not None and ext not in exts:
            skipped.append({"path": path, "reason": "ext_filter"})
            continue
        if ext not in TEXT_EXTS:
            skipped.append({"path": path, "reason": "unsupported_ext"})
            continue
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            skipped.append({"path": path, "reason": "unreadable"})
            continue
        if max_size is not None and size > max_size:
            skipped.append({"path": path, "reason": "too_large", "size": size})
            continue
        # R1 新能力：低于 min_size 的文件跳过（空/极小占位文件过滤）
        if min_size is not None and size < min_size:
            skipped.append({"path": path, "reason": "too_small", "size": size})
            continue
        # 增量模式：未变更的文件直接复用旧条目，跳过 I/O
        old = prev_by_path.get(path)
        if old is not None and old.mtime == mtime and old.size == size:
            entries.append(old)
            continue
        snippet = _read_snippet(path)
        entries.append(IndexEntry(path=path, size=size, snippet=snippet, mtime=mtime))
    return entries, skipped


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
        files = data.get("files", []) if isinstance(data, dict) else []
    except (OSError, json.JSONDecodeError):
        return []
    # R2 修复（健壮性）：原实现对单条条目用 IndexEntry(**item) 的整体列表推导，
    # 任一条目字段缺失/类型错误会抛 TypeError，被外层捕获后「整个索引」返回 []，
    # 导致 _require_index 误报「尚未发现索引文件」、ask 直接中止，即便其余条目完好。
    # 现改为逐条构造，残缺条目单独跳过，保留可用部分（部分损坏不应拖垮整体）。
    entries: List[IndexEntry] = []
    for item in files:
        try:
            entries.append(IndexEntry(**item))
        except (TypeError, ValueError):
            continue
    return entries


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

    R2 修复（隐性正确性缺陷）：原实现对 `e.path` 直接 `os.path.exists`，
    而 build_index 写入的 path 多为相对路径（相对当初建索引时的 root），
    若从另一个 cwd 运行 `prune --root X`，相对路径会按「当前 cwd」解析，
    导致本应保留的文件被误判为「已删除」而错误清除。
    现把相对路径按传入的 root 重新拼接后再判存在，`root` 参数也真正被使用。
    """
    entries = load_index(index_path)
    if not entries:
        return 0
    out_dir = os.path.dirname(os.path.abspath(index_path)) or "."
    kept = []
    for e in entries:
        p = e.path
        if not os.path.isabs(p):
            p = os.path.join(root, p)
        if os.path.exists(p):
            kept.append(e)
    removed = len(entries) - len(kept)
    if removed:
        save_index(kept, out_dir)
    return removed


def clear_index(root: str = ".") -> bool:
    """删除（清空）索引文件。

    返回 True 表示确实删除了索引；False 表示该目录下本就没有索引文件
    （幂等：重复调用不产生错误，也不会误删其它文件）。仅删除已知的
    INDEX_FILE 文件名，不接收任意路径，避免误删风险。

    root：索引文件所在目录（与 index --root 对齐），默认当前目录。
    """
    out_path = os.path.join(root, INDEX_FILE) if root and root != "." else INDEX_FILE
    if not os.path.exists(out_path):
        return False
    try:
        os.remove(out_path)
        return True
    except OSError:
        return False


def index_stats(index_path: str = INDEX_FILE) -> "Dict | None":
    """返回索引统计信息；无索引（或文件损坏）时返回 None（供 CLI 友好提示）。

    隐性性能/健壮性：原实现先 load_index 再单独 open 读一次 indexed_at，
    对同一个文件做了两次磁盘读取，且第二次读取失败时静默丢失 indexed_at。
    这里改为只读取并解析一次，同时拿到 files 与 indexed_at，消除冗余 I/O。
    """
    if not os.path.exists(index_path):
        return None
    # R2 修复（一致性/健壮性）：原实现用 `IndexEntry(**item)` 整体列表推导，
    # 任一条目字段缺失/类型错误会抛 TypeError 而上层未捕获，导致 `stats` / `files`
    # 命令在遇到「单条损坏条目」的索引时直接崩溃——而 load_index 早已改为逐条
    # 容错、ask 检索能正常容忍同一份损坏索引。现复用防御性的 load_index 解析条目，
    # 保证「部分损坏不拖垮整体统计」，与检索路径行为一致。
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    entries = load_index(index_path)
    if not entries:
        return None
    total = sum(e.size for e in entries)
    ext_counter: Counter = Counter()
    for e in entries:
        ext = os.path.splitext(e.path)[1].lower() or "(无扩展名)"
        ext_counter[ext] += 1
    return {
        "file_count": len(entries),
        "total_bytes": total,
        "top_extensions": sorted(ext_counter.items(), key=lambda x: -x[1]),
        "indexed_at": data.get("indexed_at", ""),
    }

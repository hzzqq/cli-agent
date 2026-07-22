"""cli-agent：一个垂直代码库问答 CLI 智能体（MVP）。

用法示例：
  python agent.py index .                      # 对当前目录建索引
  python agent.py index . --ext .py,.md        # 只索引 .py 与 .md
  python agent.py ask "这个仓库是做什么的"      # 单轮提问
  python agent.py ask "..." --json             # 以 JSON 输出（便于脚本解析）
  python agent.py ask "..." --model qwen2 --base-url http://x/v1
  python agent.py chat                          # 进入多轮对话

环境变量：
  OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL  覆盖 LLM 接入
  MOCK_LLM=1                                       离线 stub 模式，无需 API key
"""

from __future__ import annotations

import json as _json
import os
import sys
from typing import Optional

import typer

from index_store import build_index, save_index, load_index, INDEX_FILE
from retriever import build_context
from llm_client import LLMClient, LLMConfig, LLMError


app = typer.Typer(
    help="垂直代码库问答 CLI 智能体：index 建索引，ask 单轮问答，chat 多轮对话。",
    no_args_is_help=True,
)


def _require_index() -> bool:
    """检查是否已建索引，未建则打印友好提示并返回 False。"""
    if not load_index():
        typer.echo(
            f"⚠️  尚未发现索引文件（{INDEX_FILE}）。\n"
            "请先运行：python agent.py index <目录>\n"
            "例如：python agent.py index ."
        )
        return False
    return True


def _build_config(
    model: Optional[str], base_url: Optional[str], api_key: Optional[str]
) -> Optional[LLMConfig]:
    """根据 CLI 覆盖项构造 LLMConfig；无覆盖则返回 None（沿用环境变量默认值）。"""
    overrides = {}
    if model:
        overrides["model"] = model
    if base_url:
        overrides["base_url"] = base_url
    if api_key:
        overrides["api_key"] = api_key
    if not overrides:
        return None
    cfg = LLMConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _do_ask(
    question: str,
    top_k: int,
    config: Optional[LLMConfig] = None,
    as_json: bool = False,
    min_score: float = 0.0,
):
    context_text, paths = build_context(question, top_k=top_k, min_score=min_score)
    client = LLMClient(config)
    try:
        answer = client.answer(question, paths, context_text)
    except LLMError as exc:  # 隐性问题：未捕获则向用户抛出裸栈
        typer.echo(f"⚠️ 调用 LLM 失败：{exc}", err=True)
        raise typer.Exit(code=1)
    if as_json:
        out = {"question": question, "answer": answer, "references": paths}
        typer.echo(_json.dumps(out, ensure_ascii=False, indent=2))
    else:
        typer.echo(answer)
        if paths:
            typer.echo("\n📚 参考文件：")
            for p in paths:
                typer.echo(f"  - {p}")


@app.command()
def index(
    path: str = typer.Argument(".", help="要索引的目录路径，默认当前目录"),
    root: str = typer.Option(".", "--root", help="索引文件存放目录，默认当前目录"),
    ext: Optional[str] = typer.Option(
        None, "--ext", help="只索引指定扩展名，逗号分隔，如 .py,.md"
    ),
):
    """递归遍历目录，建立文本文件索引。"""
    exts = None
    if ext:
        exts = {e.strip().lower() for e in ext.split(",") if e.strip()}
        typer.echo(f"🔍 扩展名过滤：{', '.join(sorted(exts))}")
    typer.echo(f"🔍 正在索引目录：{path}")
    entries = build_index(path, exts=exts)
    out = save_index(entries, root)
    typer.echo(f"✅ 已索引 {len(entries)} 个文件，索引保存到 {out}")


@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="要问的问题，用引号包裹；省略则从标准输入(管道)读取"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
    model: Optional[str] = typer.Option(None, "--model", help="指定模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="指定 API 地址"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="指定 API Key"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 格式输出"),
):
    """基于索引检索相关文件并调用 LLM 作答。"""
    # 省略位置参数时，尝试从标准输入读取（管道场景）
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    if not question:
        typer.echo("⚠️ 问题不能为空（可传入参数，或用管道提供：echo '问题' | python agent.py ask）", err=True)
        raise typer.Exit(code=1)
    if not _require_index():
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key)
    _do_ask(question, top_k, config=cfg, as_json=as_json, min_score=min_score)


@app.command()
def context(
    question: str = typer.Argument(..., help="要检索的问题，用引号包裹"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
):
    """仅展示检索到的上下文与参考文件（不调用 LLM）。

    便于排查检索质量、核对参考来源，或在不想消耗 LLM 额度时预览。
    """
    text, paths = build_context(question, top_k=top_k, min_score=min_score)
    if not paths:
        typer.echo("🔎 未检索到相关文件，请确认索引已建立且问题与仓库内容相关。")
        raise typer.Exit(code=1)
    typer.echo(f"🔎 召回 {len(paths)} 个文件，上下文长度 {len(text)} 字符：")
    for p in paths:
        typer.echo(f"  - {p}")
    if text:
        typer.echo("\n--- 上下文预览 ---")
        typer.echo(text[:2000])


@app.command()
def search(
    keyword: str = typer.Argument(..., help="在索引内容中检索的关键词"),
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
):
    """在本仓库索引中按关键词检索（不调用 LLM），快速定位相关文件。"""
    from index_store import INDEX_FILE, search_index

    path = os.path.join(root, INDEX_FILE)
    hits = search_index(keyword, path)
    if not hits:
        typer.echo("未找到匹配的文件。")
        raise typer.Exit(code=1)
    typer.echo(f"🔍 命中 {len(hits)} 个文件：")
    for e in hits:
        typer.echo(f"  - {e.path}")


@app.command()
def stats(
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
):
    """展示当前索引的统计信息（文件数、总大小、扩展名分布、建立时间）。"""
    from index_store import INDEX_FILE, index_stats

    path = os.path.join(root, INDEX_FILE)
    s = index_stats(path)
    if not s:
        typer.echo(f"⚠️  未发现索引文件（{path}）。请先运行：python agent.py index <目录>")
        raise typer.Exit(code=1)
    typer.echo(f"📊 索引统计（{path}）")
    typer.echo(f"  文件数：{s['file_count']}")
    typer.echo(f"  总大小：{s['total_bytes'] / 1024:.1f} KB")
    typer.echo(f"  建立时间：{s['indexed_at']}")
    typer.echo("  扩展名分布：")
    for ext, cnt in s["top_extensions"][:10]:
        typer.echo(f"    {ext}: {cnt}")


@app.command()
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="每轮召回的相关文件数量"),
    model: Optional[str] = typer.Option(None, "--model", help="指定模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="指定 API 地址"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="指定 API Key"),
):
    """进入交互式多轮对话，每轮都带上检索到的上下文。输入 exit/quit 退出。"""
    if not _require_index():
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key)
    typer.echo("💬 进入对话模式（输入 exit 或 quit 退出）：")
    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n👋 再见。")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            typer.echo("👋 再见。")
            break
        _do_ask(question, top_k, config=cfg)
        typer.echo("")


def main():
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()

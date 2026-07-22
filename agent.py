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
from pathlib import Path
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
    """检查是否已建索引，未建则打印友好提示并返回 False。

    隐性问题：原先索引文件存在但已损坏（无法解析）时，load_index 静默返回
    []，导致提示误报「尚未发现索引文件」，误导用户以为是没建索引。这里
    通过 os.path.exists 区分「未建」与「已损坏」，给出准确诊断。
    """
    if not load_index():
        if os.path.exists(INDEX_FILE):
            typer.echo(
                f"⚠️  索引文件（{INDEX_FILE}）存在但无法解析（可能已损坏）。\n"
                "请重新运行：python agent.py index <目录>"
            )
        else:
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


def _resolve_system_prompt(
    inline: Optional[str], file_path: Optional[str]
) -> "Optional[str]":
    """解析系统提示来源：文件优先，其次内联文本，均无则返回 None（沿用默认）。"""
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            typer.echo(f"⚠️ 无法读取系统提示文件：{exc}", err=True)
            raise typer.Exit(code=1)
    return inline or None


def _do_ask(
    question: str,
    top_k: int,
    config: Optional[LLMConfig] = None,
    as_json: bool = False,
    min_score: float = 0.0,
    history: "list[dict] | None" = None,
    system_prompt: Optional[str] = None,
    no_context: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = False,
    max_context_chars: int = 6000,
    stream: bool = True,
):
    # 隐性问题：--no-context 下不应再强制检索，否则会为「纯通用问题」无谓加载索引
    if no_context:
        context_text, paths = "", []
    else:
        context_text, paths = build_context(
            question, top_k=top_k, min_score=min_score, max_context_chars=max_context_chars
        )
    # R1 可观测性：--verbose 展示检索概况，便于排查召回质量
    if verbose:
        n_chars = len(context_text)
        est = LLMClient.estimate_tokens(context_text) if context_text else 0
        typer.echo(f"🔎 检索：命中 {len(paths)} 个文件，上下文 {n_chars} 字符（约 {est} token）")
    # 隐性问题：检索无命中时 LLM 仍正常作答，但用户无从得知答案「未接地」，
    # 这里显式告警，避免把纯模型臆测误认为基于仓库的回答。
    if not no_context and not paths:
        typer.echo(
            "⚠️ 未检索到相关文件，将仅凭 LLM 已有知识作答（可尝试调整问题或重新运行 index）",
            err=True,
        )
    client = LLMClient(config)
    # 多轮时把历史 + 当前问题（含检索上下文）组装成 messages 传给 complete，
    # 使 chat 真正具备「多轮记忆」，而非每轮只看当前问题（隐性正确性缺陷）。
    user_content = (
        f"用户问题：{question}\n\n"
        f"=== 仓库上下文（来自 {len(paths)} 个文件）===\n{context_text}\n"
        "=== 上下文结束 ===\n请回答上面的问题。"
    )
    messages = list(history or []) + [{"role": "user", "content": user_content}]
    try:
        # --json 输出需整体 JSON，不能逐字打印污染结果，强制非流式
        do_stream = stream and not as_json
        if do_stream:
            # R1 新能力：流式逐 token 打印（边生成边显示），提升交互体感
            answer = ""
            for piece in client.stream_complete(
                messages, context_files=paths, system_prompt=system_prompt
            ):
                answer += piece
                typer.echo(piece, nl=False)
            typer.echo("")  # 换行，避免与后续「参考文件」粘连
        else:
            # 自定义系统提示透传给 complete（CLI 此前未暴露该能力，提示工程不可控）
            answer = client.complete(messages, context_files=paths, system_prompt=system_prompt)
    except LLMError as exc:  # 隐性问题：未捕获则向用户抛出裸栈
        typer.echo(f"⚠️ 调用 LLM 失败：{exc}", err=True)
        raise typer.Exit(code=1)
    if as_json:
        out = {"question": question, "answer": answer, "references": paths, "no_context": no_context}
        typer.echo(_json.dumps(out, ensure_ascii=False, indent=2))
    else:
        typer.echo(answer)
        if paths:
            typer.echo("\n📚 参考文件：")
            for p in paths:
                typer.echo(f"  - {p}")
        elif no_context:
            typer.echo("\nℹ️  未使用仓库上下文（--no-context）")
    # R1 新能力：把答案落盘，便于脚本化消费与归档
    if save_path:
        try:
            Path(save_path).write_text(answer, encoding="utf-8")
            typer.echo(f"\n💾 答案已保存至：{save_path}")
        except OSError as exc:
            typer.echo(f"⚠️ 无法写入答案文件：{exc}", err=True)
            raise typer.Exit(code=1)
    return answer


@app.command()
def index(
    path: str = typer.Argument(".", help="要索引的目录路径，默认当前目录"),
    root: str = typer.Option(".", "--root", help="索引文件存放目录，默认当前目录"),
    ext: Optional[str] = typer.Option(
        None, "--ext", help="只索引指定扩展名，逗号分隔，如 .py,.md"
    ),
    max_size: Optional[int] = typer.Option(
        None, "--max-size", help="跳过超过此字节数的文件（避免大锁文件/数据文件污染索引）"
    ),
    incremental: bool = typer.Option(
        False, "--incremental", help="增量重建：未变更（mtime/size 不变）的文件直接复用旧索引，省去重复 I/O"
    ),
):
    """递归遍历目录，建立文本文件索引。"""
    exts = None
    if ext:
        exts = {e.strip().lower() for e in ext.split(",") if e.strip()}
        typer.echo(f"🔍 扩展名过滤：{', '.join(sorted(exts))}")
    typer.echo(f"🔍 正在索引目录：{path}")
    prev = load_index(os.path.join(root, INDEX_FILE)) if incremental else None
    if incremental:
        # 隐性可观测性：原实现在「无旧索引」时静默不提示，用户误以为增量生效；
        # 这里显式区分「复用」与「全量重建」，避免误导。
        if prev:
            typer.echo(f"♻️  增量模式：载入旧索引 {len(prev)} 条，复用未变更文件")
        else:
            typer.echo("♻️  增量模式：未发现旧索引，将执行全量重建")
    entries, skipped = build_index(path, exts=exts, max_size=max_size, prev=prev)
    out = save_index(entries, root)
    typer.echo(f"✅ 已索引 {len(entries)} 个文件，索引保存到 {out}")
    if skipped:
        typer.echo(f"⏭️  跳过 {len(skipped)} 个文件（不受支持类型/超大/不可读）")
        for s in skipped[:10]:
            extra = f" ({s['size']} 字节)" if s["reason"] == "too_large" else ""
            typer.echo(f"  - [{s['reason']}]{extra} {s['path']}")
        if len(skipped) > 10:
            typer.echo(f"  ... 其余 {len(skipped) - 10} 个省略")


@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="要问的问题，用引号包裹；省略则从 --file 或标准输入(管道)读取"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
    model: Optional[str] = typer.Option(None, "--model", help="指定模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="指定 API 地址"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="指定 API Key"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 格式输出"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="自定义系统提示（内联），覆盖默认助手提示"),
    system_prompt_file: Optional[str] = typer.Option(None, "--system-prompt-file", help="从文件读取系统提示（优先于 --system-prompt）"),
    no_context: bool = typer.Option(False, "--no-context", help="跳过仓库检索，直接把问题交给 LLM（适用于无需代码上下文的通用问题）"),
    question_file: Optional[str] = typer.Option(None, "--file", help="从文件读取问题（支持长/多行问题，优先于位置参数与管道）"),
    save_path: Optional[str] = typer.Option(None, "--save", help="把答案写入指定文件（便于脚本化消费/归档）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印检索概况（命中文件数 / 上下文字符数 / 估算 token）"),
        max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符），超出后停止追加更低相关文件"),
        no_stream: bool = typer.Option(False, "--no-stream", help="关闭流式输出，等生成完毕后一次性打印（兼容管道/脚本）"),
):
    """基于索引检索相关文件并调用 LLM 作答。"""
    # 问题来源优先级：--file > 位置参数 > 管道（stdin）
    if question_file:
        try:
            question = Path(question_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            typer.echo(f"⚠️ 无法读取问题文件：{exc}", err=True)
            raise typer.Exit(code=1)
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    if not question:
        typer.echo("⚠️ 问题不能为空（可传入参数，或 --file 提供，或用管道：echo '问题' | python agent.py ask）", err=True)
        raise typer.Exit(code=1)
    # 隐性问题：索引要求是「检索」的前置条件；--no-context 下无需索引也应允许提问
    if not no_context and not _require_index():
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key)
    sp = _resolve_system_prompt(system_prompt, system_prompt_file)
    _do_ask(
        question, top_k, config=cfg, as_json=as_json, min_score=min_score,
        system_prompt=sp, no_context=no_context, save_path=save_path, verbose=verbose,
        max_context_chars=max_context_chars, stream=not no_stream,
    )


@app.command()
def context(
    question: str = typer.Argument(..., help="要检索的问题，用引号包裹"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
    max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符）"),
):
    """仅展示检索到的上下文与参考文件（不调用 LLM）。

    便于排查检索质量、核对参考来源，或在不想消耗 LLM 额度时预览。
    """
    text, paths = build_context(question, top_k=top_k, min_score=min_score, max_context_chars=max_context_chars)
    if not paths:
        typer.echo("🔎 未检索到相关文件，请确认索引已建立且问题与仓库内容相关。")
        raise typer.Exit(code=1)
    typer.echo(f"🔎 召回 {len(paths)} 个文件，上下文长度 {len(text)} 字符：")
    for p in paths:
        typer.echo(f"  - {p}")
    if text:
        typer.echo("\n--- 上下文预览 ---")
        typer.echo(text[:2000])


def _excerpt(text: str, keyword: str, width: int = 60) -> str:
    """截取包含关键词的一小段上下文，便于用户确认命中位置。"""
    if not text:
        return ""
    low = text.lower()
    kw = keyword.lower()
    idx = low.find(kw)
    if idx == -1:
        return text[:width]
    start = max(0, idx - width // 2)
    end = min(len(text), start + width)
    return text[start:end].replace("\n", " ")


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
        excerpt = _excerpt(e.snippet or "", keyword)
        if excerpt:
            typer.echo(f"      …{excerpt}…")


@app.command()
def prune(
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
):
    """清理索引中已删除文件的陈旧条目（不调用 LLM）。"""
    from index_store import INDEX_FILE, prune_missing

    path = os.path.join(root, INDEX_FILE)
    removed = prune_missing(root, path)
    if removed:
        typer.echo(f"🧹 已移除 {removed} 个陈旧索引条目：{path}")
    else:
        typer.echo(f"✅ 索引干净，无需清理：{path}")


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
def config(
    model: Optional[str] = typer.Option(None, "--model", help="覆盖模型名（仅用于预览生效值，不持久化）"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API 地址（仅用于预览生效值）"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（仅用于预览生效值）"),
):
    """展示当前生效的 LLM 配置（环境变量与默认值合并后的结果，可观测性）。"""
    cfg = _build_config(model, base_url, api_key) or LLMConfig()
    typer.echo("⚙️  当前 LLM 配置（环境变量 + 默认值合并后）：")
    typer.echo(f"  base_url   : {cfg.base_url}")
    typer.echo(f"  model      : {cfg.model}")
    typer.echo(f"  mock       : {cfg.mock}")
    typer.echo(f"  timeout    : {cfg.timeout}s")
    typer.echo(f"  max_tokens : {cfg.max_tokens}")
    typer.echo(f"  temperature: {cfg.temperature}")
    typer.echo(f"  retries    : {cfg.retries}（退避基数 {cfg.backoff}s）")
    typer.echo(f"  api_key    : {'<已设置>' if cfg.api_key else '<未设置>'}")


@app.command()
def clear(
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
    yes: bool = typer.Option(False, "--yes", "-y", help="确认删除索引文件（避免误删，默认仅预览）"),
):
    """删除当前索引文件（重置索引）。

    默认只预览将被删除的文件，需加 --yes / -y 才真正删除，防止在非交互场景下误删。
    """
    from index_store import INDEX_FILE

    path = os.path.join(root, INDEX_FILE)
    if not os.path.exists(path):
        typer.echo(f"ℹ️  没有可删除的索引文件（{path}）")
        return
    if not yes:
        typer.echo(f"🔍 将删除索引文件：{path}\n（确认请加 --yes / -y）")
        return
    try:
        os.remove(path)
        typer.echo(f"🗑️  已删除索引文件：{path}")
    except OSError as exc:
        typer.echo(f"⚠️ 删除失败：{exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def files(
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
):
    """列出当前索引中的文件（路径与大小），用于审计索引内容（不调用 LLM）。"""
    from index_store import INDEX_FILE, load_index

    path = os.path.join(root, INDEX_FILE)
    entries = load_index(path)
    if not entries:
        typer.echo(f"⚠️  索引为空或不存在（{path}）。请先运行：python agent.py index <目录>")
        raise typer.Exit(code=1)
    typer.echo(f"📄 索引包含 {len(entries)} 个文件（{path}）：")
    for e in entries:
        typer.echo(f"  - [{e.size} B] {e.path}")


@app.command()
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="每轮召回的相关文件数量"),
    model: Optional[str] = typer.Option(None, "--model", help="指定模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="指定 API 地址"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="指定 API Key"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="自定义系统提示（内联），覆盖默认助手提示"),
    system_prompt_file: Optional[str] = typer.Option(None, "--system-prompt-file", help="从文件读取系统提示（优先于 --system-prompt）"),
    no_context: bool = typer.Option(False, "--no-context", help="跳过仓库检索，每轮直接把问题交给 LLM（纯通用对话）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印检索概况（命中文件数 / 上下文字符数 / 估算 token）"),
        max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符），超出后停止追加更低相关文件"),
        no_stream: bool = typer.Option(False, "--no-stream", help="关闭流式输出，等生成完毕后一次性打印"),
):
    """进入交互式多轮对话，每轮都带上检索到的上下文。输入 exit/quit 退出。"""
    if not no_context and not _require_index():
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key)
    sp = _resolve_system_prompt(system_prompt, system_prompt_file)
    typer.echo("💬 进入对话模式（输入 exit 或 quit 退出）：")
    history: list[dict] = []
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
        answer = _do_ask(question, top_k, config=cfg, history=history, system_prompt=sp, no_context=no_context, verbose=verbose, max_context_chars=max_context_chars, stream=not no_stream)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        typer.echo("")


def main():
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()

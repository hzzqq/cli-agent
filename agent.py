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

# 版本号：随每次功能性迭代递增，便于用户/脚本识别 CLI 能力级别。
VERSION = "1.0.0"

# 持久化配置文件：保存常用 LLM 接入项，避免每次运行重复敲 --model/--base-url/--api-key
CONFIG_FILE = ".cliagent_config.json"
# 允许写入/读取的配置文件键白名单（R2 类型安全：拒绝任意键，防配置注入）
_CONFIG_KEYS = ("model", "base_url", "api_key")


app = typer.Typer(
    help="垂直代码库问答 CLI 智能体：index 建索引，ask 单轮问答，chat 多轮对话。",
    no_args_is_help=True,
)


@app.command()
def version():
    """打印 cli-agent 版本号（便于脚本化识别与问题排查）。"""
    typer.echo(f"cli-agent {VERSION}")


def _cli_agent_commands():
    """返回当前注册的所有子命令名（用于补全脚本，避免硬编码随命令增减而漂移）。"""
    try:
        names = [c.name for c in app.registered_commands if c.name]
        if names:
            return names
    except Exception:
        pass
    return ["ask", "chat", "index", "search", "files", "related",
            "context", "explain", "prune", "stats", "config", "models",
            "clear", "version"]


def _bash_completion(commands):
    cmds = " ".join(commands)
    return (
        "# cli-agent bash 自动补全\n"
        f"_cli_agent_cmds='{cmds}'\n"
        "_cli_agent_complete() {\n"
        '  COMPREPLY=( $(compgen -W "$_cli_agent_cmds" -- "${COMP_WORDS[1]}") )\n'
        "}\n"
        "complete -F _cli_agent_complete cli-agent\n"
    )


def _zsh_completion(commands):
    cmds = " ".join(commands)
    return (
        "#compdef cli-agent\n"
        "_cli_agent() {\n"
        f'  _values "command" {cmds}\n'
        "}\n"
        "_cli_agent\n"
    )


@app.command()
def completion(
    shell: str = typer.Option("bash", "--shell", "-s", help="补全脚本目标 shell：bash 或 zsh"),
):
    """输出 shell 自动补全脚本（R1 新能力）。

    用法示例：source <(cli-agent completion)   # bash 启用 Tab 子命令补全，
    长命令行的可用性提升。脚本基于当前实际注册命令动态生成，命令增减自动同步。
    """
    names = _cli_agent_commands()
    script = _zsh_completion(names) if shell == "zsh" else _bash_completion(names)
    typer.echo(script)


def _index_health(root: str = ".") -> "tuple[str, str]":
    """返回 (状态, 说明)：索引健康诊断，供 _require_index 与 doctor 复用。

    状态取值：
      - "ok"      索引存在且可解析
      - "missing" 索引文件不存在（需要先 index）
      - "corrupt" 索引文件存在但无法解析（可能已损坏）

    R2 抽出：此前只有 ask/context/explain/related/chat 的 _require_index 内部
    能区分「未建」与「已损坏」；而 stats/files/search/prune 与新增的 doctor
    直接 load_index 命中损坏索引时会静默得到空结果、误判为「无文件」。现统一
    收敛到本函数，保证所有命令对索引健康的判定口径完全一致（DRY + 一致性）。
    """
    index_path = os.path.join(root, INDEX_FILE)
    # 默认 root="." 时回落到 0 参 load_index()（与历史/既有单测一致）；
    # 仅当显式给定 --root 时才按该目录索引校验。
    loaded = load_index(index_path) if root and root != "." else load_index()
    if not loaded:
        if os.path.exists(index_path):
            return "corrupt", (
                f"索引文件（{INDEX_FILE}）存在但无法解析（可能已损坏）。"
            )
        return "missing", f"尚未发现索引文件（{INDEX_FILE}）。"
    return "ok", f"索引正常（{len(loaded)} 个文件）。"


def _require_index(root: str = ".") -> bool:
    """检查是否已建索引，未建则打印友好提示并返回 False。

    root：索引文件所在目录（与 index 的 --root 对齐）。默认 "." 即 cwd 下
    INDEX_FILE，与历史行为一致；传入其它目录则校验该目录的索引。

    隐性问题：原先索引文件存在但已损坏（无法解析）时，load_index 静默返回
    []，导致提示误报「尚未发现索引文件」，误导用户以为是没建索引。这里
    通过 os.path.exists 区分「未建」与「已损坏」，给出准确诊断。
    """
    status, detail = _index_health(root)
    if status == "ok":
        return True
    if status == "corrupt":
        typer.echo(
            f"⚠️  {detail}\n请重新运行：python agent.py index <目录>"
        )
    else:
        typer.echo(
            f"⚠️  {detail}\n请先运行：python agent.py index <目录>\n"
            "例如：python agent.py index ."
        )
    return False


def _load_file_config(root: str = ".") -> dict:
    """读取持久化配置文件（.cliagent_config.json）中的 LLM 接入项。

    R2 类型安全/防御（隐性健壮性问题）：仅接受白名单键且值为字符串，
    拒绝任意键或畸形值被注入为 LLM 配置（避免配置文件被篡改后无声影响行为）；
    文件不存在 / 损坏 / 非 dict 一律返回空 dict，不抛错、不中断命令。
    """
    path = os.path.join(root, CONFIG_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for k in _CONFIG_KEYS:
        v = data.get(k)
        if isinstance(v, str) and v:  # 仅保留非空字符串，过滤 null/数字/列表等
            out[k] = v
    return out


def _save_file_config(cfg: LLMConfig, root: str = ".") -> None:
    """把生效的 LLM 配置写入持久化文件（仅白名单键）。"""
    data = {k: getattr(cfg, k) for k in _CONFIG_KEYS}
    path = os.path.join(root, CONFIG_FILE)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)


def _unset_file_config(keys: "list[str]", root: str = ".") -> "list[str]":
    """从持久化配置文件中删除指定白名单键（其余键保持不变），返回实际被删除的键。

    R1 新能力：与 --save 互补——此前只能写入配置文件、无法撤销某项错误持久化的
    接入项（如误存的 api_key）。现支持按需删除单个/多个键。
    仅白名单键可被删除，非白名单键被忽略（防御配置文件被滥用）；
    文件不存在 / 损坏时安全返回空列表（与 _load_file_config 一致）。
    """
    path = os.path.join(root, CONFIG_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    removed = []
    for k in keys:
        if k in _CONFIG_KEYS and k in data:
            del data[k]
            removed.append(k)
    if removed:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
    return removed


def _build_config(
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: Optional[float] = None,
    root: str = ".",
) -> Optional[LLMConfig]:
    """根据 CLI 覆盖项构造 LLMConfig；CLI 缺省时回落到持久化配置文件；
    两者皆无则沿用环境变量默认值（返回 None）。

    R1 新能力：配置文件让常用接入项「一次写入、处处复用」，无需每次敲长 flag。
    优先级：CLI flag > 配置文件 > 环境变量默认。
    CLI 生成参数（max_tokens/temperature/top_p/timeout）同样以 CLI 优先级最高，
    即便未设置 model/base_url/api_key 也单独生效（此前这些参数无法覆盖）。
    """
    file_cfg = _load_file_config(root)
    overrides = {
        "model": model or file_cfg.get("model"),
        "base_url": base_url or file_cfg.get("base_url"),
        "api_key": api_key or file_cfg.get("api_key"),
    }
    cfg = None
    if any(overrides.values()):
        cfg = LLMConfig()
        for k, v in overrides.items():
            if v:
                setattr(cfg, k, v)
    # 生成参数（成本/采样/超时控制）单独覆盖，优先级最高，且不依赖接入项是否设置
    gen = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "timeout": timeout,
    }
    if any(v is not None for v in gen.values()):
        cfg = cfg or LLMConfig()
        for k, v in gen.items():
            if v is not None:
                setattr(cfg, k, v)
    return cfg


def _validate_retrieval_opts(
    top_k: int, min_score: float = 0.0, max_context_chars: int = 6000
) -> None:
    """校验检索类命令的数值参数，越界即在 CLI 边界友好报错并退出（R1 输入护栏）。

    R3 类型安全/空值防护维度：此前 --top-k 0/负数、--min-score 负、--max-context-chars
    负会被静默吞掉——负数 top_k 经 retrieve_scored 钳制为 0 返回空召回，用户误以为
    「未检索到相关文件」；负阈值/负预算则完全无校验。现统一在命令入口拦截并给出明确诊断，
    避免「输入明显非法却表现正常」的隐性误导（R2 隐性可用性缺陷）。
    """
    if top_k is not None and top_k < 1:
        raise typer.BadParameter(f"--top-k 必须 >= 1（当前 {top_k}）")
    if min_score is not None and min_score < 0:
        raise typer.BadParameter(f"--min-score 必须 >= 0（当前 {min_score}）")
    if max_context_chars is not None and max_context_chars < 0:
        raise typer.BadParameter(f"--max-context-chars 必须 >= 0（当前 {max_context_chars}）")


def _resolve_system_prompt(
    inline: Optional[str], file_path: Optional[str]
) -> "Optional[str]":
    """解析系统提示来源：文件优先，其次内联文本，均无则返回 None（沿用默认）。"""
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
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
    explain: bool = False,
    index_path: "str | None" = None,
):
    # 隐性问题：--no-context 下不应再强制检索，否则会为「纯通用问题」无谓加载索引
    if no_context:
        context_text, paths = "", []
    else:
        context_text, paths = build_context(
            question, top_k=top_k, min_score=min_score,
            max_context_chars=max_context_chars, index_path=index_path,
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
    # R1 新能力：--explain 展示「为什么召回这些文件」——
    # 命中路径 + 相关度分数 + 命中关键词（频率降序），提升检索透明性。
    if explain and not no_context and paths:
        from retriever import explain_retrieval

        typer.echo("🔎 检索解释（按相关度）：")
        for hit in explain_retrieval(question, top_k=top_k, min_score=min_score,
                                     index_path=index_path):
            terms = "、".join(hit["terms"]) or "（无显式关键词匹配）"
            typer.echo(f"  {hit['score']:.2f}　{hit['path']}　命中词：{terms}")
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
    min_size: Optional[int] = typer.Option(
        None, "--min-size", help="跳过小于此字节数的文件（过滤空/极小占位文件等噪声）"
    ),
    incremental: bool = typer.Option(
        False, "--incremental", help="增量重建：未变更（mtime/size 不变）的文件直接复用旧索引，省去重复 I/O"
    ),
    exclude: Optional[str] = typer.Option(
        None, "--exclude", help="忽略模式（逗号分隔，fnmatch）：相对路径或文件名匹配则跳过，如 'tests/*,*.min.js'"
    ),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出索引摘要（file_count/skipped_count/skipped/index_path），便于流水线消费"),
):
    """递归遍历目录，建立文本文件索引。"""
    # R2 修复（隐性可观测性/失败快速）：原实现对不存在的目录静默执行，
    # os.walk 在缺失目录上不报错、只返回 0 个文件，用户误以为索引成功、
    # 却得到空索引并在后续 ask 时困惑「为何检索不到」。现显式区分并快速失败。
    if not os.path.isdir(path):
        typer.echo(f"⚠️ 索引目标路径不存在或不是目录：{path}", err=True)
        raise typer.Exit(code=1)
    exts = None
    if ext:
        exts = {e.strip().lower() for e in ext.split(",") if e.strip()}
        if not as_json:
            typer.echo(f"🔍 扩展名过滤：{', '.join(sorted(exts))}")
    if not as_json:
        typer.echo(f"🔍 正在索引目录：{path}")
    prev = load_index(os.path.join(root, INDEX_FILE)) if incremental else None
    if incremental:
        # 隐性可观测性：原实现在「无旧索引」时静默不提示，用户误以为增量生效；
        # 这里显式区分「复用」与「全量重建」，避免误导。
        if not as_json:
            if prev:
                typer.echo(f"♻️  增量模式：载入旧索引 {len(prev)} 条，复用未变更文件")
            else:
                typer.echo("♻️  增量模式：未发现旧索引，将执行全量重建")
    exclude_list = None
    if exclude:
        # 解析逗号分隔的忽略模式；逐个 strip 以容忍 "tests/* ,*.min.js" 这类空格
        exclude_list = [e.strip() for e in exclude.split(",") if e.strip()]
        if not as_json:
            typer.echo(f"🚫 忽略模式：{', '.join(exclude_list)}")
    entries, skipped = build_index(path, exts=exts, max_size=max_size, min_size=min_size, prev=prev, exclude=exclude_list)
    out = save_index(entries, root)
    if as_json:
        # R1 新能力：机读索引摘要，便于 CI / 流水线消费（与 search/files/related 一致）
        payload = {
            "file_count": len(entries),
            "skipped_count": len(skipped),
            "skipped": [{"path": s["path"], "reason": s["reason"]} for s in skipped],
            "index_path": out,
        }
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
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
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="最大生成 token 数（成本/长度控制，覆盖默认值）"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="采样温度（控制创造性，覆盖默认值）"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="请求超时秒数（覆盖默认值）"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="nucleus 采样阈值（覆盖默认值，1.0 即关闭）"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 格式输出"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="自定义系统提示（内联），覆盖默认助手提示"),
    system_prompt_file: Optional[str] = typer.Option(None, "--system-prompt-file", help="从文件读取系统提示（优先于 --system-prompt）"),
    no_context: bool = typer.Option(False, "--no-context", help="跳过仓库检索，直接把问题交给 LLM（适用于无需代码上下文的通用问题）"),
    question_file: Optional[str] = typer.Option(None, "--file", help="从文件读取问题（支持长/多行问题，优先于位置参数与管道）"),
    save_path: Optional[str] = typer.Option(None, "--save", help="把答案写入指定文件（便于脚本化消费/归档）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印检索概况（命中文件数 / 上下文字符数 / 估算 token）"),
        max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符），超出后停止追加更低相关文件"),
        no_stream: bool = typer.Option(False, "--no-stream", help="关闭流式输出，等生成完毕后一次性打印（兼容管道/脚本）"),
        explain: bool = typer.Option(False, "--explain", "-e", help="打印检索解释（命中文件+相关度+命中词），便于排查召回质量"),
        root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录（与 index 的 --root 对齐，便于查询非默认目录建立的索引）"),
):
    """基于索引检索相关文件并调用 LLM 作答。"""
    # R1 输入护栏：在入口校验检索数值参数，非法值（如 --top-k 0/负、--min-score 负）
    # 立即友好报错，避免被静默钳制成空召回误导用户。
    _validate_retrieval_opts(top_k, min_score, max_context_chars)
    # 问题来源优先级：--file > 位置参数 > 管道（stdin）
    if question_file:
        try:
            question = Path(question_file).read_text(encoding="utf-8", errors="ignore").strip()
        except OSError as exc:
            typer.echo(f"⚠️ 无法读取问题文件：{exc}", err=True)
            raise typer.Exit(code=1)
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    if not question:
        typer.echo("⚠️ 问题不能为空（可传入参数，或 --file 提供，或用管道：echo '问题' | python agent.py ask）", err=True)
        raise typer.Exit(code=1)
    # 隐性问题：索引要求是「检索」的前置条件；--no-context 下无需索引也应允许提问
    if not no_context and not _require_index(root):
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=timeout)
    sp = _resolve_system_prompt(system_prompt, system_prompt_file)
    index_path = os.path.join(root, INDEX_FILE)
    _do_ask(
        question, top_k, config=cfg, as_json=as_json, min_score=min_score,
        system_prompt=sp, no_context=no_context, save_path=save_path, verbose=verbose,
        max_context_chars=max_context_chars, stream=not no_stream,
        explain=explain, index_path=index_path,
    )


@app.command()
def context(
    question: str = typer.Argument(..., help="要检索的问题，用引号包裹"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
    max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符）"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出检索上下文与参考文件，便于脚本消费"),
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录（与 index 的 --root 对齐）"),
):
    """仅展示检索到的上下文与参考文件（不调用 LLM）。

    便于排查检索质量、核对参考来源，或在不想消耗 LLM 额度时预览。
    """
    _validate_retrieval_opts(top_k, min_score, max_context_chars)
    # R2 修复（隐性诊断缺陷）：原实现直接调用 build_context，在无索引时
    # 会误报「未检索到相关文件」，与「索引未建/已损坏」的真实原因混淆。
    # 现先经 _require_index 区分，与 ask/explain 等命令保持一致的诊断口径。
    if not _require_index(root):
        raise typer.Exit(code=1)
    index_path = os.path.join(root, INDEX_FILE)
    text, paths = build_context(question, top_k=top_k, min_score=min_score,
                                  max_context_chars=max_context_chars, index_path=index_path)
    if not paths:
        typer.echo("🔎 未检索到相关文件，请确认问题与仓库内容相关。")
        raise typer.Exit(code=1)
    if as_json:
        # R1 新能力：机读输出上下文与参考文件，与 search/files/related 对齐
        payload = {"context": text, "files": paths}
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"🔎 召回 {len(paths)} 个文件，上下文长度 {len(text)} 字符：")
    for p in paths:
        typer.echo(f"  - {p}")
    if text:
        typer.echo("\n--- 上下文预览 ---")
        typer.echo(text[:2000])


@app.command()
def explain(
    question: str = typer.Argument(..., help="要解释检索的问题，用引号包裹"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="召回的相关文件数量"),
    min_score: float = typer.Option(0.0, "--min-score", help="最低相关度阈值，过滤弱相关文件"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 数组输出检索解释，便于脚本消费"),
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录（与 index 的 --root 对齐）"),
):
    """展示「为什么召回了这些文件」：命中文件 + 相关度 + 命中关键词（不调用 LLM）。

    与 context/search 类似，explain 只做检索解释、不消耗 LLM 额度，
    适合排查检索质量、核对参考来源，或在提交问题前确认召回是否合理。
    """
    _validate_retrieval_opts(top_k, min_score, 6000)
    if not _require_index(root):
        raise typer.Exit(code=1)
    from retriever import explain_retrieval

    index_path = os.path.join(root, INDEX_FILE)
    hits = explain_retrieval(question, top_k=top_k, min_score=min_score,
                              index_path=index_path)
    if not hits:
        typer.echo("🔎 未检索到相关文件，请确认索引已建立且问题与仓库内容相关。")
        raise typer.Exit(code=1)
    if as_json:
        # R1 新能力：机读输出检索解释，与 search/files/related 对齐
        payload = [
            {"path": h["path"], "score": round(h["score"], 4), "terms": h["terms"]}
            for h in hits
        ]
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"🔎 针对问题「{question}」共召回 {len(hits)} 个文件：")
    for h in hits:
        terms = "、".join(h["terms"]) or "（无显式关键词匹配）"
        typer.echo(f"  {h['score']:.2f}　{h['path']}　命中词：{terms}")


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


def load_session(path: "Optional[str]") -> "list[dict]":
    """从 JSON 文件载入对话历史（多轮会话持久化，R1 新能力支撑）。

    让 chat 的多轮历史在 CLI 重启后仍能恢复。文件不存在 / 解析失败 /
    结构非法时返回空列表（不抛错）；对每条消息做基础校验，畸形项直接
    丢弃，避免把脏数据喂给 LLM。
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for m in data:
        if (isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                and isinstance(m.get("content"), str)):
            out.append({"role": m["role"], "content": m["content"]})
    return out


def save_session(path: "Optional[str]", history: "list[dict]") -> None:
    """把对话历史写入 JSON 文件（多轮会话持久化）。

    R2 修复（隐性健壮性问题）：原实现直接 open(path, "w")，当父目录不存在时
    会触发 FileNotFoundError 被静默吞掉——用户以为会话已落盘，重启后却发现
    历史丢失且无任何提示。现改为：自动创建父目录；若仍写入失败，向 stderr
    告警（不阻断对话），避免「悄无声息地丢历史」。
    """
    if not path:
        return
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        typer.echo(f"⚠️ 会话历史保存失败（{path}）：{exc}", err=True)


@app.command()
def search(
    keyword: str = typer.Argument(..., help="在索引内容中检索的关键词"),
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 数组输出命中结果（便于脚本消费）"),
):
    """在本仓库索引中按关键词检索（不调用 LLM），快速定位相关文件。"""
    from index_store import INDEX_FILE, search_index

    path = os.path.join(root, INDEX_FILE)
    hits = search_index(keyword, path)
    if not hits:
        if as_json:
            typer.echo(_json.dumps([], ensure_ascii=False))
        else:
            typer.echo("未找到匹配的文件。")
        raise typer.Exit(code=1)
    if as_json:
        payload = [
            {"path": e.path, "excerpt": _excerpt(e.snippet or "", keyword)}
            for e in hits
        ]
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"🔍 命中 {len(hits)} 个文件：")
    for e in hits:
        typer.echo(f"  - {e.path}")
        excerpt = _excerpt(e.snippet or "", keyword)
        if excerpt:
            typer.echo(f"      …{excerpt}…")


@app.command()
def related(
    file: str = typer.Argument(..., help="要查「相似文件」的目标文件路径"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="返回的最相似文件数量"),
    max_size: Optional[int] = typer.Option(
        None, "--max-size", help="仅读取文件前 N 字节作为内容样本（避免大文件读全量）"
    ),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 数组输出相似文件（便于脚本消费）"),
    root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录（与 index 的 --root 对齐）"),
):
    """找出与给定文件内容最相似的索引文件（不调用 LLM，基于 BM25 词重叠）。

    R1 新能力：快速定位「哪些文件与当前文件高度相关」，适用于重构时评估
    影响面、寻找可复用模块、或理解某文件在仓库中的关联结构。
    """
    _validate_retrieval_opts(top_k, 0.0, 6000)
    from retriever import find_related

    p = Path(file)
    if not p.exists():
        typer.echo(f"⚠️  文件不存在：{file}")
        raise typer.Exit(code=1)
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        typer.echo(f"⚠️  无法读取文件：{exc}")
        raise typer.Exit(code=1)
    if max_size:
        content = content[:max_size]
    # 复用与 ask 一致的索引前置检查（区分未建 / 已损坏）
    if not _require_index(root):
        raise typer.Exit(code=1)
    # R2 修复：find_related 对目标路径做归一化后排除自身，避免不同路径写法
    # 导致目标文件被当作「最相似」返回（详见 retriever.find_related 注释）
    index_path = os.path.join(root, INDEX_FILE)
    rel = find_related(content, file, top_k=top_k, index_path=index_path)
    if not rel:
        if as_json:
            typer.echo(_json.dumps([], ensure_ascii=False))
        else:
            typer.echo("未找到相似文件（索引可能未包含足够的可比内容，或目标文件内容过于独特）。")
        raise typer.Exit(code=1)
    if as_json:
        payload = [{"path": e.path, "score": round(s, 4)} for e, s in rel]
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"🔗 与 {file} 最相似的 {len(rel)} 个文件：")
    for e, s in rel:
        typer.echo(f"  {s:.2f}　{e.path}")


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
    model: Optional[str] = typer.Option(None, "--model", help="覆盖模型名（仅用于预览生效值/持久化）"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="覆盖 API 地址（仅用于预览生效值/持久化）"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="覆盖 API Key（仅用于预览生效值/持久化）"),
    save: bool = typer.Option(False, "--save", help="把当前生效配置写入 .cliagent_config.json，供后续运行复用"),
    unset: Optional[str] = typer.Option(None, "--unset", help="从配置文件删除指定键（逗号分隔，仅白名单内），如 --unset api_key,model"),
    check: bool = typer.Option(False, "--check", help="探测 LLM 端点可用性（调用 health 探针，离线 mock 模式直接返回可用）"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出配置 / 健康状态，便于脚本消费"),
):
    """展示当前生效的 LLM 配置（环境变量/配置文件/默认值合并后的结果，可观测性）。

    R1 新能力：加 --save 可把当前生效配置持久化到 .cliagent_config.json，
    后续 ask/chat/index 将自动读取，无需每次重复敲 --model/--base-url/--api-key。
    加 --unset KEY[,KEY...] 可撤销已持久化的某个/多个接入项（与 --save 互补，
    便于清理误存的 api_key 等）。
    加 --check 可探测 LLM 端点是否可用（health 探针）；--json 输出结构化结果。
    """
    if unset:
        # R1 新能力：撤销持久化配置项（独立于展示/保存流程，优先处理）
        keys = [k.strip() for k in unset.split(",") if k.strip()]
        invalid = [k for k in keys if k not in _CONFIG_KEYS]
        removed = _unset_file_config(keys, ".")
        if invalid:
            typer.echo(
                f"⚠️ 忽略非白名单键（不可删除）：{', '.join(invalid)}", err=True
            )
        if removed:
            typer.echo(f"🗑️  已从 {CONFIG_FILE} 删除配置项：{', '.join(removed)}")
        else:
            typer.echo(f"ℹ️  {CONFIG_FILE} 中无匹配键可删除（或文件不存在）")
        return
    cfg = _build_config(model, base_url, api_key) or LLMConfig()
    if check:
        # R1 新能力：端点健康探针，快速判断当前配置能否真正调用 LLM
        health = LLMClient(cfg).health()
        if as_json:
            out = {
                "config": {
                    "base_url": cfg.base_url,
                    "model": cfg.model,
                    "mock": cfg.mock,
                    "timeout": cfg.timeout,
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "api_key_set": bool(cfg.api_key),
                },
                "health": health,
            }
            typer.echo(_json.dumps(out, ensure_ascii=False, indent=2))
            return
        typer.echo("🔌 LLM 端点探测：")
        typer.echo(f"  状态   : {'✅ 可用' if health['ok'] else '❌ 不可用'}")
        typer.echo(f"  模式   : {'mock' if health['mock'] else '真实'}")
        typer.echo(f"  模型   : {health['model']}")
        if health["error"]:
            typer.echo(f"  错误   : {health['error']}")
    if save:
        _save_file_config(cfg)
        # R2 安全/可观测性：配置文件以明文保存 api_key，提示用户注意权限与泄露风险
        if cfg.api_key:
            typer.echo(
                "⚠️ 警告：配置文件将以明文保存 api_key，请注意文件权限与泄露风险"
                "（建议 chmod 600 或仅保存在可信环境）。",
                err=True,
            )
        typer.echo(f"💾 配置已保存到 {CONFIG_FILE}（后续运行将自动读取）")
    if as_json and not check:
        out = {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "mock": cfg.mock,
            "timeout": cfg.timeout,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "api_key_set": bool(cfg.api_key),
        }
        typer.echo(_json.dumps(out, ensure_ascii=False, indent=2))
        return
    typer.echo("⚙️  当前 LLM 配置（环境变量 + 配置文件 + 默认值合并后）：")
    typer.echo(f"  base_url   : {cfg.base_url}")
    typer.echo(f"  model      : {cfg.model}")
    typer.echo(f"  mock       : {cfg.mock}")
    typer.echo(f"  timeout    : {cfg.timeout}s")
    typer.echo(f"  max_tokens : {cfg.max_tokens}")
    typer.echo(f"  temperature: {cfg.temperature}")
    typer.echo(f"  retries    : {cfg.retries}（退避基数 {cfg.backoff}s）")
    typer.echo(f"  api_key    : {'<已设置>' if cfg.api_key else '<未设置>'}")


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出诊断结果，便于脚本消费"),
):
    """环境自检：一次性诊断索引健康、LLM 端点与配置，给出可执行建议。

    R1 新能力：此前用户遇到「ask 不返回相关内容」「config 明明配了却调不通」
    时没有任何统一入口快速定位问题，只能逐个手动试命令。doctor 聚合三类检查：
      1) 索引健康（missing / corrupt / ok）——复用 _index_health（与 ask 等命令口径一致）
      2) LLM 端点可用性（复用 LLMClient.health 探针）
      3) 当前生效配置解析（model / base_url / mock / api_key 是否已设置）
    输出分级状态（✅/⚠️/❌）与建议；存在「损坏索引」或「真实模式端点不可用」时
    以退出码 1 提示（便于 CI / 启动脚本判断是否阻断）。
    """
    index_status, index_detail = _index_health(".")
    health = LLMClient(LLMConfig()).health()
    cfg = _build_config(None, None, None) or LLMConfig()

    checks = {
        "index": {"status": index_status, "detail": index_detail},
        "llm": {
            "ok": health["ok"],
            "mock": health["mock"],
            "model": health["model"],
            "error": health.get("error"),
        },
        "config": {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "mock": cfg.mock,
            "api_key_set": bool(cfg.api_key),
        },
    }
    # 严重度判定：损坏索引 + 真实模式端点不可用 视为阻断项
    critical = (index_status == "corrupt") or (
        not health["ok"] and not health["mock"]
    )

    if as_json:
        typer.echo(_json.dumps({"critical": critical, **checks}, ensure_ascii=False, indent=2))
    else:
        icon = {"ok": "✅", "missing": "⚠️ ", "corrupt": "❌"}.get(index_status, "❓")
        typer.echo("🩺 cli-agent 环境自检")
        typer.echo(f"  索引   : {icon} {index_detail}")
        if health["ok"]:
            typer.echo(
                f"  LLM    : ✅ 可用（{'mock' if health['mock'] else '真实'} · {health['model']}）"
            )
        else:
            typer.echo(f"  LLM    : ❌ 不可用：{health.get('error')}")
        typer.echo(
            f"  配置   : base_url={cfg.base_url} · model={cfg.model} · "
            f"mock={cfg.mock} · api_key={'<已设置>' if cfg.api_key else '<未设置>'}"
        )
        if critical:
            typer.echo("❌ 存在阻断项，请先按上方提示修复。")
        else:
            typer.echo("✅ 环境基本就绪。")

    raise typer.Exit(code=1 if critical else 0)


@app.command()
def models(
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出模型列表，便于脚本消费"),
):
    """列出当前 LLM 端点可用的模型（探测 /models 端点，mock 模式返回当前模型）。

    R1 新能力：与 openwebui 的 /api/models 对称，让 CLI 用户也能快速查看可切换的模型，
    无需手动拼 curl。探测失败（端点不可用 / 鉴权错误）时打印错误并以退出码 1 结束。
    """
    cfg = _build_config(None, None, None) or LLMConfig()
    try:
        model_ids = LLMClient(cfg).list_models()
    except LLMError as exc:
        typer.echo(f"❌ 获取模型列表失败：{exc}", err=True)
        raise typer.Exit(code=1)
    if as_json:
        typer.echo(_json.dumps(
            {"models": model_ids, "mock": cfg.mock}, ensure_ascii=False, indent=2
        ))
        return
    if not model_ids:
        typer.echo("ℹ️  端点未返回任何模型。")
        return
    typer.echo(f"📋 可用模型（{'mock 模式' if cfg.mock else '端点 ' + cfg.base_url}）：")
    for mid in model_ids:
        typer.echo(f"  - {mid}")


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
    as_json: bool = typer.Option(False, "--json", help="以 JSON 数组输出索引文件（便于脚本消费）"),
):
    """列出当前索引中的文件（路径与大小），用于审计索引内容（不调用 LLM）。"""
    from index_store import INDEX_FILE, load_index

    path = os.path.join(root, INDEX_FILE)
    entries = load_index(path)
    if not entries:
        if as_json:
            typer.echo(_json.dumps([], ensure_ascii=False))
        else:
            typer.echo(f"⚠️  索引为空或不存在（{path}）。请先运行：python agent.py index <目录>")
        raise typer.Exit(code=1)
    if as_json:
        payload = [{"path": e.path, "size": e.size} for e in entries]
        typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"📄 索引包含 {len(entries)} 个文件（{path}）：")
    for e in entries:
        typer.echo(f"  - [{e.size} B] {e.path}")


def _save_transcript(path: "Optional[str]", history: "list[dict]") -> None:
    """把对话历史写出为 Markdown 转录文件（便于归档 / 分享 / 后续检索）。

    失败（路径不可写等）只告警不中断对话，避免「悄无声息丢转录」。
    """
    if not path or not history:
        return
    try:
        lines = ["# cli-agent 对话转录", ""]
        role_label = {"user": "## 你", "assistant": "## 助手", "system": "## 系统"}
        for m in history:
            role = m.get("role", "")
            lines.append(role_label.get(role, f"## {role}"))
            lines.append("")
            lines.append(m.get("content", ""))
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        typer.echo(f"⚠️ 对话转录保存失败（{path}）：{exc}", err=True)


@app.command()
def chat(
    top_k: int = typer.Option(5, "--top-k", "-k", help="每轮召回的相关文件数量"),
    model: Optional[str] = typer.Option(None, "--model", help="指定模型名称"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="指定 API 地址"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="指定 API Key"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="最大生成 token 数（成本/长度控制，覆盖默认值）"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="采样温度（控制创造性，覆盖默认值）"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="请求超时秒数（覆盖默认值）"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="nucleus 采样阈值（覆盖默认值，1.0 即关闭）"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="自定义系统提示（内联），覆盖默认助手提示"),
    system_prompt_file: Optional[str] = typer.Option(None, "--system-prompt-file", help="从文件读取系统提示（优先于 --system-prompt）"),
    no_context: bool = typer.Option(False, "--no-context", help="跳过仓库检索，每轮直接把问题交给 LLM（纯通用对话）"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="打印检索概况（命中文件数 / 上下文字符数 / 估算 token）"),
        max_context_chars: int = typer.Option(6000, "--max-context-chars", help="上下文预算上限（字符），超出后停止追加更低相关文件"),
        no_stream: bool = typer.Option(False, "--no-stream", help="关闭流式输出，等生成完毕后一次性打印"),
        explain: bool = typer.Option(False, "--explain", "-e", help="每轮打印检索解释（命中文件+相关度+命中词）"),
        session_file: Optional[str] = typer.Option(None, "--session", help="对话历史持久化文件（JSON）：进入时载入、每轮后保存，支持跨重启续聊"),
        save_file: Optional[str] = typer.Option(None, "--save", help="把整段对话转录为 Markdown 保存到该文件（退出时落盘）"),
        root: str = typer.Option(".", "--root", help="索引文件所在目录，默认当前目录（与 index 的 --root 对齐）"),
):
    """进入交互式多轮对话，每轮都带上检索到的上下文。输入 exit/quit 退出。"""
    # chat 暂不暴露 --min-score（检索质量由 top_k 控制），以默认值 0.0 校验越界
    _validate_retrieval_opts(top_k, 0.0, max_context_chars)
    if not no_context and not _require_index(root):
        raise typer.Exit(code=1)
    cfg = _build_config(model, base_url, api_key, max_tokens=max_tokens, temperature=temperature, top_p=top_p, timeout=timeout)
    sp = _resolve_system_prompt(system_prompt, system_prompt_file)
    index_path = os.path.join(root, INDEX_FILE)
    typer.echo("💬 进入对话模式（输入 exit 或 quit 退出）：")
    # R1 新能力：从持久化文件恢复多轮历史，使对话可跨 CLI 重启续聊
    history: list[dict] = load_session(session_file) if session_file else []
    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\n👋 再见。")
            if session_file:
                save_session(session_file, history)
            if save_file:
                _save_transcript(save_file, history)
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            typer.echo("👋 再见。")
            if session_file:
                save_session(session_file, history)
            if save_file:
                _save_transcript(save_file, history)
            break
        answer = _do_ask(question, top_k, config=cfg, history=history, system_prompt=sp, no_context=no_context, verbose=verbose, max_context_chars=max_context_chars, stream=not no_stream, explain=explain, index_path=index_path)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if session_file:
            save_session(session_file, history)
        typer.echo("")


def main():
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()

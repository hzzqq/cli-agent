"""cli-agent 结构化日志工具（R1 新能力：批处理/服务可观测性）。

设计要点：
- 诊断信息（进度/警告/错误）统一走 logging，**默认写入 stderr**，
  不污染面向机器/管道的 stdout 结果输出，保证既有脚本解析安全。
- 可通过 --log-file 落盘，便于长时间批处理任务事后排查。
- 提供 capture_logs 测试辅助，便于单测断言日志行为。
"""
from __future__ import annotations

import io
import logging
import sys
from contextlib import contextmanager

DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str = DEFAULT_FORMAT,
    force: bool = True,
) -> logging.Logger:
    """配置根日志（只配置一次，除非 force=True）。

    level: DEBUG/INFO/WARNING/ERROR（大小写不敏感）。
    log_file: 可选路径；为 None 时日志写入 stderr。
    返回根 logger，便于调用方 getLogger 派生。
    """
    global _CONFIGURED
    level = (level or "INFO").upper()
    numeric = getattr(logging, level, logging.INFO)
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    if force or not _CONFIGURED:
        root.handlers = [handler]
        root.setLevel(numeric)
        _CONFIGURED = True
    else:
        root.addHandler(handler)
        root.setLevel(min(root.level, numeric))
    return root


@contextmanager
def capture_logs(level: str = "DEBUG"):
    """测试辅助：捕获块内发出的日志记录，yield 一个可读 StringIO。

    用法::
        with capture_logs() as buf:
            logging.getLogger('x').info('hi')
        assert 'hi' in buf.getvalue()
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    old_level = root.level
    old_handlers = root.handlers[:]
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.DEBUG))
    try:
        yield buf
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)

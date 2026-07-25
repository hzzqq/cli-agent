"""log_utils 行为测试（R1 新能力：cli-agent 可观测性基础设施）。"""
from __future__ import annotations

import logging

from log_utils import setup_logging, capture_logs


def test_setup_logging_default_level():
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_case_insensitive():
    setup_logging("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("INFO", log_file=str(log_file))
    logging.getLogger("cli_agent").info("批处理开始")
    assert log_file.exists()
    assert "批处理开始" in log_file.read_text(encoding="utf-8")


def test_capture_logs_captures_records():
    with capture_logs() as buf:
        logging.getLogger("t").warning("something happened")
    assert "something happened" in buf.getvalue()
    assert "WARNING" in buf.getvalue()


def test_cli_callback_accepts_log_options(tmp_path):
    from typer.testing import CliRunner
    import agent

    runner = CliRunner()
    # R2 验收：旧调用方式不受影响，新 --log-level/--log-file 被接受且不报错
    log_file = tmp_path / "cli.log"
    r = runner.invoke(agent.app, ["--log-level", "DEBUG", "--log-file", str(log_file), "version"])
    assert r.exit_code == 0
    assert "cli-agent" in r.stdout
    assert log_file.exists()

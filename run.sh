#!/usr/bin/env bash
# cli-agent —— 垂直代码库问答 CLI
# 依赖：pip install typer openai
python agent.py index .
python agent.py ask "这个仓库是做什么的"

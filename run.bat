@echo off
REM cli-agent —— 垂直代码库问答 CLI
python agent.py index .
python agent.py ask "这个仓库是做什么的"
pause

@echo off
REM cli-agent —— 垂直代码库问答 CLI（离线 MOCK 模式，无需 API key）
where python >nul 2>nul || (echo [错误] 未检测到 python，请先安装 Python 并勾选 "Add to PATH"。 & pause & exit /b)
set MOCK_LLM=1
python agent.py index .
python agent.py chat
pause

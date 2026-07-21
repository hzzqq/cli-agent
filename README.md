# cli-agent · 垂直代码库问答 CLI 智能体（MVP）

给一个代码仓库，就能用中文提问：「这个仓库是做什么的？」「在哪里处理登录逻辑？」
基于本地索引 + 检索召回 + LLM 作答，模仿热门 CLI Agent 的交互范式。

## 特性

- `index <path>`：递归遍历目录，对文本类文件（`.py/.md/.txt/.js/.ts/.json` 等）建立索引（路径 + 大小 + 头部片段），保存为 `.cliagent_index.json`。
- `ask "<问题>"`：关键词检索 top-K 相关文件，把片段拼进 prompt 调用 LLM 作答，并打印「参考文件」列表。
- `chat`：进入交互式多轮对话，每轮都带上检索到的上下文。
- **离线降级**：设置 `MOCK_LLM=1` 时不调用真实接口，直接返回基于检索结果的 stub 答案，无 API key 也能完整演示流程。
- 友好错误处理：未建索引先 `ask` 会提示先运行 `index`；自动跳过 `.git/venv/二进制` 与大文件。

## 安装

```bash
cd mirrors/cli-agent
pip install -r requirements.txt
# 或
python -m pip install typer openai
```

## 配置（可选）

默认指向本地 Ollama。可用环境变量覆盖：

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"   # Ollama 默认地址
export OPENAI_API_KEY="ollama"                        # Ollama 一般不需要真实 key
export OPENAI_MODEL="qwen2.5:latest"                  # 使用的模型名
```

连云端（如 OpenAI / 兼容网关）：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-你的key"
```

## 用法

### 1. 建立索引

```bash
python agent.py index .
# ✅ 已索引 42 个文件，索引保存到 .cliagent_index.json
```

### 2. 单轮提问

```bash
python agent.py ask "这个仓库是做什么的"
python agent.py ask "登录逻辑在哪里实现" --top-k 3
```

### 3. 多轮对话

```bash
python agent.py chat
# 你> 这个项目的入口文件是哪个？
# 你> 它的参数怎么解析的？
# 你> exit
```

### 4. 离线 / MOCK 模式（无需 API key）

```bash
MOCK_LLM=1 python agent.py index .
MOCK_LLM=1 python agent.py ask "这个仓库是做什么的"
# （MOCK 模式）根据检索结果，关于「...」的相关内容可能分布在以下文件里：
#   - README.md
#   - agent.py
```

Windows PowerShell 设置环境变量：

```powershell
$env:MOCK_LLM="1"
python agent.py index .
python agent.py ask "这个仓库是做什么的"
```

## 目录说明

```
cli-agent/
├── agent.py          # CLI 主程序（index / ask / chat 命令）
├── retriever.py      # 检索引擎（简化 TF-IDF，召回 top-K）
├── llm_client.py     # OpenAI 兼容客户端 + MOCK 模式
├── index_store.py    # 索引构建 / 持久化（.cliagent_index.json）
├── requirements.txt
└── README.md
```

## 备注

- 检索使用内置的简化 TF-IDF，无外部向量库依赖；如需更强的语义召回，可后续替换为 embedding + 向量库。
- 索引文件 `.cliagent_index.json` 建议加入 `.gitignore`。

## 近期迭代（自驱动开发 10 轮）

- `ask` / `chat` 新增 `--model` / `--base-url` / `--api-key` 覆盖项，并支持 `--json` 结构化输出（便于脚本解析）
- `index` 新增 `--ext` 扩展名白名单过滤（如 `--ext .py,.md`）
- 简化 TF-IDF 检索，零向量库依赖；离线 `MOCK_LLM=1` 冒烟可用
- 附 `run.sh` / `run.bat` 一键示例

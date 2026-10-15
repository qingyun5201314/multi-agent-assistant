# 多模态 AI Agent 助手

一个基于 RAG + Agent 架构的多功能 AI 助手，支持知识库检索、长短期记忆、多会话管理、实时天气、数学计算、网页搜索和 AI 图像生成。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)

## 📸 功能演示

| 功能 | 说明 |
|------|------|
| 📚 **RAG知识库检索** | 上传文档，基于文档内容回答问题 |
| 🧠 **长短期记忆** | 短期用SQLite存会话，长期用FAISS存用户画像 |
| 💬 **多会话管理** | 像ChatGPT一样管理多个对话，自动生成标题 |
| 🔄 **智能摘要** | 长对话自动摘要，保留上下文不丢失 |
| 🎯 **语义去重** | 长期记忆自动过滤重复信息 |
| 🌤️ **实时天气** | 接入 Open-Meteo API，查询全球城市天气 |
| 🧮 **数学计算** | 支持复杂数学表达式 |
| ⏰ **时间查询** | 查询当前日期时间 |
| 🔍 **网页搜索** | 接入 Tavily API，专为AI设计的搜索 |
| 🎨 **AI图像生成** | 接入 ModelScope 通义万相，文生图 |
| 🤖 **Agent多步推理** | ReAct 范式，自动规划调用工具 |
| 💬 **Web界面** | 原生 HTML/CSS/JS，无需框架 |

## 🏗️ 技术架构

```
┌─────────────────────────────────────────┐
│           用户浏览器 (前端)              │
│      HTML + CSS + JavaScript            │
└─────────────────────────────────────────┘
                    ↓ HTTP
┌─────────────────────────────────────────┐
│         FastAPI 后端服务                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Agent (ReAct 范式)               │
│  思考 → 调用工具 → 观察 → 继续思考       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│              工具集                      │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │RAG检索   │ │天气查询  │ │计算器   │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │网页搜索  │ │时间查询  │ │图像生成 │ │
│  └──────────┘ └──────────┘ └─────────┘ │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│              记忆系统                    │
│  ┌────────────────┐ ┌────────────────┐ │
│  │短期记忆(SQLite)│ │长期记忆(FAISS) │ │
│  │·会话历史       │ │·用户画像       │ │
│  │·智能摘要       │ │·语义去重       │ │
│  └────────────────┘ └────────────────┘ │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- 建议使用 Conda 虚拟环境

```bash
conda create -n ai-assistant python=3.10
conda activate ai-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的密钥：

```bash
# LLM API（支持DeepSeek/Qwen/GLM等OpenAI兼容接口）
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 图像生成（ModelScope）
MODELSCOPE_TOKEN=你的魔搭token

# 网页搜索（Tavily）
TAVILY_API_KEY=你的tavily key
```

**获取密钥**：
- LLM API：[DeepSeek](https://platform.deepseek.com)
- 图像生成：[ModelScope](https://www.modelscope.cn)
- 网页搜索：[Tavily](https://tavily.com)（免费1000次/月）

### 4. 准备知识库（可选）

在项目根目录创建 `knowledge.txt`，每段之间用**空行**分隔：

```
Python是一种高级编程语言，由Guido van Rossum于1991年创建。

机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律。

RAG（检索增强生成）是一种结合信息检索和文本生成的技术。
```

### 5. 启动服务

```bash
python main.py
```

### 6. 访问

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```
multi-agent/
├── main.py              # FastAPI 入口 + 静态文件服务
├── agent.py             # Agent 核心（ReAct + 记忆系统）
├── tools.py             # 工具集（6个工具）
├── vector_db.py         # RAG向量数据库（FAISS + 语义分块）
├── database.py          # 短期记忆（SQLite）
├── long_term_memory.py  # 长期记忆（FAISS + 语义去重）
├── config.py            # 配置中心
├── knowledge.txt        # 示例知识库
├── static/              # 前端文件
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 核心实现

### Agent 工作流程（ReAct）

```python
# 1. 思考：LLM 决定下一步做什么
decision = self.think(user_input, session_id, history)

# 2. 如果判断需要工具
if "TOOL:" in decision:
    tool_name, args = parse(decision)
    result = await tools.execute(tool_name, args)
    history.append({"tool": tool_name, "result": result})

# 3. 如果判断可以回答
if "FINAL:" in decision:
    return decision.split("FINAL:")[-1]
```

### 记忆系统架构

**短期记忆（database.py）**：
- SQLite 存储会话历史
- 按 session_id 隔离不同对话
- 消息数超过20条时自动摘要

**长期记忆（long_term_memory.py）**：
- FAISS 存储用户画像
- LLM 判断"值不值得存"
- 语义去重（相似度 > 0.85 则跳过）
- 每次对话检索 top-3 相关记忆

### RAG 语义分块

```python
# 按标点分块（不是按字数硬切）
sentences = re.split(r'(?<=[。！？；.!?;])', text)
# 保留语义完整性，提高检索精度
```

### 工具路由

Agent 通过提示词规则决定用哪个工具：

| 问题类型 | 调用工具 |
|---------|---------|
| 概念解释 | `search_knowledge` |
| 实时信息 | `web_search` |
| 数学计算 | `calculate` |
| 时间查询 | `get_time` |
| 天气查询 | `get_weather` |
| 画图 | `generate_image` |

## 📊 项目亮点

1. **双层记忆系统**：短期 SQLite + 长期 FAISS，持久化存储
2. **智能摘要机制**：长对话自动压缩，避免上下文溢出
3. **语义去重**：用向量相似度过滤重复记忆，而非字面比较
4. **多会话管理**：支持多个独立对话，自动生成标题
5. **多模态能力**：文本对话 + 图像生成
6. **异步架构**：FastAPI + async/await，支持并发请求
7. **工具路由**：6个工具，Agent 自动选择
8. **RAG 语义分块**：按标点切分，保留语义完整性

## 🐛 踩坑记录

| 问题 | 解决 |
|------|------|
| `faiss-cpu` 在 Windows + Python 3.13 装不上 | 换 conda 环境 + Python 3.10 |
| `sentence-transformers` 和 `huggingface_hub` 版本冲突 | 降级 `huggingface_hub==0.25.2` |
| `openai` 和 `httpx` 版本不兼容 | 锁定 `openai==1.55.3` + `httpx==0.27.2` |
| HuggingFace 被墙，模型下载失败 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `webless` 同步函数不能在 async 中调用 | 改用 `httpx` 异步版本 |
| ModelScope 返回 `task_id` 而非直接返回图片 | 轮询 `/v1/tasks/{task_id}` 获取结果 |
| `requests` 阻塞事件循环 | 全部换成 `httpx.AsyncClient` |
| 中文搜索质量差（Bing/webless） | 换成 Tavily API（专为AI设计） |
| SQLite 加字段后旧数据库报错 | 删 `memory.db` 重建 |
| 记忆字面去重太绝对 | 改用向量相似度去重（阈值0.85） |

## 🔮 后续计划

- [ ] 部署到云服务器（Docker + Nginx）
- [ ] 加入重排（Rerank）优化检索质量
- [ ] 支持多文件上传（PDF、Word、Markdown）
- [ ] 加入语音输入/输出
- [ ] 本地部署 Stable Diffusion

## 📄 License

BSD-3-Clause

## 👤 作者

- GitHub: [@qingyun5201314](https://github.com/qingyun5201314)

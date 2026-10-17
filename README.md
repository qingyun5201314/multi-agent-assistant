# 多模态 AI Agent 助手

一个基于 RAG + Agent 架构的多功能 AI 助手。支持**多路召回**、**混合检索**、**LLM重排**、**上下文压缩**、长短期记忆、多会话管理、多模态图像生成及联网搜索。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)

## 📸 功能演示

| 功能 | 说明 |
|------|------|
| 📚 **生产级 RAG 链路** | 多路召回 → 混合检索 → RRF融合 → LLM重排 → 上下文压缩 |
| 🧠 **长短期记忆** | 短期用SQLite存会话，长期用FAISS存用户画像 |
| 💬 **多会话管理** | 像ChatGPT一样管理多个对话，自动生成标题 |
| 🔄 **智能摘要** | 长对话自动摘要，保留上下文不丢失 |
| 🎯 **语义去重** | 长期记忆自动过滤重复信息 |
| 🌤️ **实时天气** | 接入 Open-Meteo API |
| 🧮 **数学计算** | 支持复杂数学表达式 |
| ⏰ **时间查询** | 查询当前日期时间 |
| 🔍 **网页搜索** | 接入 Tavily API，专为AI设计的搜索 |
| 🎨 **AI图像生成** | 接入 ModelScope 通义万相 |
| 🤖 **Agent多步推理** | ReAct 范式，自动规划调用工具 |
| 💬 **Web界面** | 原生 HTML/CSS/JS，无需框架 |

## 🏗️ 技术架构

```text
┌─────────────────────────────────────────┐
│           用户浏览器 (前端)              │
└─────────────────────────────────────────┘
                    ↓ HTTP
┌─────────────────────────────────────────┐
│         FastAPI 后端服务                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         多会话管理（SQLite）             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Agent (ReAct 范式)               │
│  思考 → 调用工具 → 观察 → 继续思考       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           记忆系统 + 工具集              │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │短期记忆     │  │生产级 RAG 链路   │  │
│  │·SQLite      │  │·查询改写         │  │
│  │·智能摘要    │  │·多路召回         │  │
│  ├─────────────┤  │·向量+BM25混合检索│  │
│  │长期记忆     │  │·RRF融合          │  │
│  │·FAISS       │  │·LLM重排          │  │
│  │·语义去重    │  │·上下文压缩       │  │
│  └─────────────┘  ├──────────────────┤  │
│                    │搜索(Tavily)      │  │
│                    │图像(ModelScope)  │  │
│                    │天气 / 计算 / 时间 │  │
│                    └──────────────────┘  │
└─────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
conda create -n ai-assistant python=3.10
conda activate ai-assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入密钥：

```bash
# LLM API（支持DeepSeek/Qwen/GLM等OpenAI兼容接口）
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 图像生成（ModelScope）
MODELSCOPE_TOKEN=你的魔搭token

# 网页搜索（Tavily）
TAVILY_API_KEY=你的tavily key

# RAG 高级开关（默认全开）
USE_HYBRID_SEARCH=true
USE_RERANK=true
USE_COMPRESSION=true
```

### 4. 启动服务

```bash
python main.py
```

### 5. 访问

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```text
multi-agent/
├── main.py              # FastAPI 入口
├── agent.py             # Agent 核心（ReAct + 记忆 + 查询改写）
├── tools.py             # 工具集（6个工具）
├── vector_db.py         # 生产级 RAG（多路召回+混合检索+重排+压缩）
├── database.py          # 短期记忆（SQLite）
├── long_term_memory.py  # 长期记忆（FAISS + 语义去重）
├── config.py            # 配置中心
├── static/              # 前端
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 RAG 核心实现（杀手锏）

本项目实现了一套**五段式**的高阶 RAG 检索链路，彻底解决传统 RAG 召回不准、废话多的问题：

```text
用户查询
  ↓
1. 查询改写 (Query Rewrite)：用LLM提取关键词，解决口语化搜不到的问题
  ↓
2. 多路召回 (Multi-Query)：原词与改写词并行去检索
  ↓
3. 混合检索 (Hybrid Search)：向量(语义) + BM25(关键词)，中英文智能分词
  ↓
4. RRF 融合 + LLM 重排：先公平融合双路结果，再用LLM给候选打分精排
  ↓
5. 上下文压缩 (Context Compression)：将选出的候选提炼成紧凑的干货，去除废话
  ↓
交给大模型生成答案
```

**效果对比：**
- ❌ **传统 RAG**：搜出一堆完整段落，大模型容易被无关背景干扰（Lost in the Middle）。
- ✅ **本项目 RAG**：大模型只拿到经过提炼的核心事实，回答精准、干练、无幻觉。

## 📊 其他核心亮点

1. **双层记忆系统**：短期 SQLite + 长期 FAISS，持久化存储，自动语义去重。
2. **长对话遗忘机制**：消息数超过阈值自动触发 LLM 摘要，永不爆 Token。
3. **多会话隔离**：Session 独立，支持自动生成标题和侧边栏切换。
4. **多格式文档解析**：支持 PDF / TXT / MD 上传，同名文件自动覆盖，FAISS 本地持久化。
5. **多模态能力**：文本对话 + 图像生成闭环。
6. **全异步架构**：`httpx.AsyncClient` + 异步轮询，不阻塞事件循环。

## 🐛 踩坑记录

| 问题 | 解决 |
|------|------|
| `faiss-cpu` 在 Windows + Python 3.13 装不上 | 换 conda 环境 + Python 3.10 |
| `sentence-transformers` 和 `huggingface_hub` 版本冲突 | 降级 `huggingface_hub==0.25.2` |
| `openai` 和 `httpx` 版本不兼容 | 锁定 `openai==1.55.3` + `httpx==0.27.2` |
| HuggingFace 被墙 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `webless` 同步函数不能在 async 中调用 | 改用 `httpx` 异步版本 |
| ModelScope 返回 `task_id` 而非图片 | 轮询 `/v1/tasks/{task_id}` |
| `requests` 阻塞事件循环 | 全部换成 `httpx.AsyncClient` |
| 中文搜索质量差（Bing） | 换成 Tavily API |
| SQLite 加字段后旧数据库报错 | 删 `memory.db` 重建 |
| 记忆字面去重太绝对 | 改用向量相似度去重（阈值0.85） |
| BM25 对英文PDF命中为0 | 中英文智能分词 |
| 上传的文档重启后丢失 | FAISS + JSON 持久化 |
| 单纯 LLM 改写查询会偏离原意 | 实现多路召回（原词+改写词并行） |
| 上下文压缩时误杀短小精悍的内容 | 优化提示词并取消字数硬限制 |

## 📄 License

BSD-3-Clause

## 👤 作者

- GitHub: [@qingyun5201314](https://github.com/qingyun5201314)
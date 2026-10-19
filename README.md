# 多模态 AI Agent 助手

一个具备**生产级 RAG 链路**、**双层记忆系统**、**全量流式输出**和**多工具调度**的全栈 AI Agent 应用。支持知识库问答、长短期记忆、多会话管理、图像生成、联网搜索等能力。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)

## 🏗️ 系统架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                       用户浏览器 (前端)                     │
│  HTML + CSS + JavaScript | EventSource 流式接收 | 会话管理  │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP / SSE (Server-Sent Events)
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端服务                        │
│    异步流式响应 (StreamingResponse) | 多格式文件上传        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Agent Core (ReAct)                     │
│         思考 → 调用工具 → 观察 → 生成最终答案               │
│                 (基于全量流式 + 异步并发)                   │
└─────────────────────────────────────────────────────────────┘
        ↓                       ↓                       ↓
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  记忆系统     │       │  RAG 检索链路 │       │  工具生态     │
├───────────────┤       ├───────────────┤       ├───────────────┤
│短期: SQLite   │       │1. 查询改写    │       │天气查询       │
│长期: FAISS    │       │2. 多路召回    │       │数学计算       │
│自动摘要       │       │3. 混合检索    │       │时间查询       │
│语义去重       │       │4. RRF 融合    │       │Tavily 联网搜索│
│多会话管理     │       │5. LLM 重排    │       │文生图 (魔搭)  │
└───────────────┘       │6. 上下文压缩  │       └───────────────┘
                        └───────────────┘
```

## ✨ 核心功能特性

### 1. 生产级 RAG 链路（高精度检索）
传统 RAG 往往是“搜出来直接塞给大模型”，本项目实现了一套**六级深度检索架构**，彻底解决召回不准、废话多、大模型幻觉的问题：
- **查询改写**：用 LLM 提取检索关键词，解决口语化提问搜不到的问题。
- **多路召回**：保留原问题与改写后的问题，双路并行检索。
- **混合检索**：向量检索（语义匹配） + BM25（关键词匹配），并实现了中英文智能分词。
- **RRF 融合**：使用 Reciprocal Rank Fusion 算法，公平合并双路结果，解决不同检索器分数尺度不一致的问题。
- **LLM 重排**：用 LLM 对候选文档进行 0-10 分相关性打分，精排前 3 条。
- **上下文压缩**：将重排后的文档提炼成核心事实，剔除噪音，极大降低 Token 消耗并提升回答精度。

### 2. 双层记忆系统（越用越懂你）
- **短期记忆**：基于 SQLite 存储会话历史。长对话触发**智能摘要机制**，自动压缩早期对话，永不爆 Token。
- **长期记忆**：基于 FAISS 向量库存储用户画像。LLM 自动判断哪些信息值得记住（如用户职业、偏好），并在多轮对话中**语义去重**，跨会话持久生效。
- **多会话隔离**：支持类似 ChatGPT 的侧边栏多会话管理，按 `session_id` 物理隔离，并利用 LLM 自动生成 10 字以内的话题标题。

### 3. 极致流式体验（打字机效果）
- 后端采用 `StreamingResponse` + SSE 协议，前端使用 `EventSource` 接收。
- 攻克了 Python `asyncio` 与 TCP 网络缓冲的底层冲突（`asyncio.sleep(0)` 强制刷新），实现**真·逐字输出**。
- 在等待大模型思考期间，前端会实时显示“正在检索知识库”、“正在重排”等状态，极大缓解用户等待焦虑。

### 4. 多模态与工具生态
- **多格式解析**：支持上传 PDF / TXT / Markdown 文件，自动解析、语义分块并持久化存入本地 FAISS。
- **图像生成**：接入 ModelScope 通义万相模型，支持 Agent 自主判断并生成图像（异步轮询 + Base64 直接渲染）。
- **联网搜索**：接入专为 AI 设计的 Tavily API，解决模型知识时效性问题。
- **外部工具**：集成实时天气、数学计算、当前时间等基础工具。

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

### 4. 启动服务
```bash
python main.py
```
访问 `http://localhost:8000` 即可开始对话，或访问 `http://localhost:8000/docs` 查看 API 文档。

## 📁 项目结构

```text
multi-agent/
├── main.py              # FastAPI 入口，SSE 流式路由
├── agent.py             # Agent 核心（ReAct + 记忆调度 + 流式生成）
├── tools.py             # 工具集（异步/同步工具封装）
├── vector_db.py         # 生产级 RAG（多路召回 + 重排 + 压缩）
├── database.py          # 短期记忆（SQLite + 摘要机制）
├── long_term_memory.py  # 长期记忆（FAISS + 语义去重）
├── config.py            # 配置中心
├── static/              # 前端代码 (EventSource 渲染)
├── knowledge_db/        # (自动生成) 知识库持久化目录
├── memory.db            # (自动生成) 短期记忆数据库
└── requirements.txt
```

## 🐛 核心踩坑记录（工程经验）

| 问题现象 | 根因分析 | 解决方案 |
|---------|---------|---------|
| 后端 `print` 逐字出，前端瞬间全出 | TCP 协议与 Python asyncio 的底层缓冲积攒小包 | 在 SSE generator 中使用 `await asyncio.sleep(0)` 强制 flush |
| 前端 SSE 时不时断开或阻塞 | 代理软件/杀毒软件拦截 `localhost` 请求 | 前端改用原生 `EventSource`，后端添加 `X-Accel-Buffering: no` |
| BM25 对英文 PDF 命中率为 0 | 默认 `jieba` 分词无法处理英文长文本 | 实现中英文比例检测，英文使用正则 `\b\w+\b` 切词 |
| 纯向量检索搜不到专有名词 | 向量擅长语义，不擅长精准匹配 | 实现 BM25 + 向量双路混合检索，引入 RRF 融合 |
| 对话轮数过长导致 Context 爆炸 | 历史消息无限制堆积 | SQLite 记录 ID，超过 20 条自动触发 LLM 摘要压缩 |
| 上传的文件重启后丢失 | 仅存在内存中 | 引入 FAISS `write_index` 与 JSON 持久化落盘 |
| 异步函数中调用 `requests` 卡死 | 同步库阻塞了 FastAPI 事件循环 | 全量替换为 `httpx.AsyncClient` |

## 🗺️ 后续路线图 (Roadmap)

- [ ] **语音交互能力**：接入 Whisper 进行语音输入识别，接入 TTS 实现语音回答播报。
- [ ] **多智能体协作 (Multi-Agent)**：引入 LangGraph，实现“审查者”与“执行者”的对抗协作架构。
- [ ] **全链路监控**：接入 Langfuse，可视化追踪每次对话的耗时、Token 消耗与工具调用链。
- [ ] **多模态 RAG**：解析 PDF 中的图表，调用视觉大模型生成图片描述，实现图文混合检索。

## 📄 License

BSD-3-Clause

## 👤 作者

- GitHub: [@qingyun5201314](https://github.com/qingyun5201314)
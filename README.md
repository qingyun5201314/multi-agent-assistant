# 多模态 AI Agent 助手

一个具备**生产级 RAG 链路**、**双层记忆系统**、**全量流式输出**、**多模态交互（语音/图像）**和**全链路监控**的全栈 AI Agent 应用。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-BSD--3--Clause-blue)

## 🏗️ 系统架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                       用户浏览器 (前端)                     │
│  HTML/CSS/JS | EventSource 流式接收 | 录音(STT) | 语音播放  │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP / SSE (Server-Sent Events)
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端服务                        │
│  异步流式响应 | 多格式文件上传 | 本地 Whisper (GPU) | TTS   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Agent Core (ReAct)                     │
│         思考 → 调用工具 → 观察 → 生成最终答案               │
│           (带 Langfuse 全链路监控与 Token 成本追踪)         │
└─────────────────────────────────────────────────────────────┘
        ↓                       ↓                       ↓
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  记忆系统     │       │  RAG 检索链路 │       │  工具生态     │
├───────────────┤       ├───────────────┤       ├───────────────┤
│短期: SQLite   │       │1. 查询改写    │       │天气 / 计算    │
│长期: FAISS    │       │2. 多路召回    │       │Tavily 联网搜索│
│自动摘要       │       │3. 混合检索    │       │文生图 (魔搭)  │
│语义去重       │       │4. RRF 融合    │       │本地 Whisper   │
│多会话管理     │       │5. LLM 重排    │       │Edge TTS       │
└───────────────┘       │6. 上下文压缩  │       └───────────────┘
                        └───────────────┘
```

## ✨ 核心功能特性

### 1. 生产级 RAG 链路（高精度检索）
传统 RAG 往往是“搜出来直接塞给大模型”，本项目实现了一套**六级深度检索架构**，彻底解决召回不准、废话多、大模型幻觉的问题：
- **查询改写**：用 LLM 提取检索关键词，解决口语化提问搜不到的问题。
- **多路召回**：保留原问题与改写后的问题，双路并行检索。
- **混合检索**：向量检索（语义匹配） + BM25（关键词匹配），并实现了中英文智能分词。
- **RRF 融合**：使用 Reciprocal Rank Fusion 算法，公平合并双路结果。
- **LLM 重排**：用 LLM 对候选文档进行 0-10 分相关性打分，精排前 3 条。
- **上下文压缩**：将重排后的文档提炼成核心事实，剔除噪音，极大降低 Token 消耗并提升回答精度。

### 2. 双层记忆系统（越用越懂你）
- **短期记忆**：基于 SQLite 存储会话历史。长对话触发**智能摘要机制**，自动压缩早期对话，永不爆 Token。
- **长期记忆**：基于 FAISS 向量库存储用户画像。LLM 自动判断哪些信息值得记住，并在多轮对话中**语义去重**，跨会话持久生效。
- **多会话隔离**：支持类似 ChatGPT 的侧边栏多会话管理，利用 LLM 自动生成话题标题。

### 3. 多模态交互（能听会说）
- **语音输入 (STT)**：前端浏览器录音，后端基于本地部署的 `faster-whisper` 模型进行语音转文字，支持 GPU 加速，完全免费且保护隐私。
- **语音播报 (TTS)**：后端接入微软 Edge-TTS，支持将长文本转化为极度逼真的真人语音（如晓晓），前端支持一键静音/播放。
- **图像生成**：接入 ModelScope 通义万相模型，支持 Agent 自主判断并生成图像并流式渲染。

### 4. 极致流式与可观测性（工程化巅峰）
- **真·流式输出**：后端采用 `StreamingResponse` + SSE 协议，前端使用 `EventSource` 接收。攻克了 Python `asyncio` 与 TCP 底层冲突，实现零延迟打字机效果。
- **全链路监控 (Langfuse)**：接入 Langfuse 进行大模型应用监控。可实时查看每次对话的思考链路、工具调用耗时、Prompt 交互过程以及精确的 Token 成本消耗。

## 🚀 快速开始

### 1. 环境准备
```bash
conda create -n ai-assistant python=3.10
conda activate ai-assistant
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
# 特别说明：语音识别需要额外安装
pip install faster-whisper edge-tts
```

### 3. 配置环境变量
复制 `.env.example` 为 `.env`，填入你的密钥：
```bash
# LLM API
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# 图像生成与搜索
MODELSCOPE_TOKEN=你的魔搭token
TAVILY_API_KEY=你的tavily key

# Langfuse 监控
LANGFUSE_PUBLIC_KEY=pk-lf-你的公钥
LANGFUSE_SECRET_KEY=sk-lf-你的私钥
```

### 4. 启动服务
```bash
python main.py
```
访问 `http://localhost:8000` 即可开始语音对话。

## 📁 项目结构

```text
multi-agent/
├── main.py              # FastAPI 入口 (流式/STT/TTS 路由)
├── agent.py             # Agent 核心 (ReAct + 记忆调度 + Langfuse监控)
├── tools.py             # 工具集 (异步/同步工具封装)
├── vector_db.py         # 生产级 RAG (多路召回 + 重排 + 压缩)
├── database.py          # 短期记忆 (SQLite + 摘要机制)
├── long_term_memory.py  # 长期记忆 (FAISS + 语义去重)
├── config.py            # 配置中心
├── static/              # 前端代码 (EventSource + MediaRecorder)
├── knowledge_db/        # (自动生成) 知识库持久化目录
└── memory.db            # (自动生成) 短期记忆数据库
```

## 🐛 核心踩坑记录（工程经验）

| 问题现象 | 根因分析 | 解决方案 |
|---------|---------|---------|
| 后端 `print` 逐字出，前端瞬间全出 | TCP 协议与 Python asyncio 的底层缓冲积攒小包 | 在 SSE generator 中使用 `await asyncio.sleep(0)` 强制 flush |
| 前端 SSE 时不时断开或阻塞 | 代理软件/杀毒软件拦截 `localhost` 请求 | 前端改用原生 `EventSource`，后端添加 `X-Accel-Buffering: no` |
| BM25 对英文 PDF 命中率为 0 | 默认 `jieba` 分词无法处理英文长文本 | 实现中英文比例检测，英文使用正则 `\b\w+\b` 切词 |
| 异步函数中调用 `requests` 卡死 | 同步库阻塞了 FastAPI 事件循环 | 全量替换为 `httpx.AsyncClient` |
| Langfuse v2/v3/v4 API 频繁变动 | 官方底层从 OpenTelemetry 重写 | 升级至 v3/v4，改用 `start_as_current_observation` |
| 录音无权限或无法播放 | 浏览器安全策略限制 | 强制 HTTPS 或 localhost 环境，配置正确的 MIME 类型 |

## 🗺️ 后续可选路线图 (Roadmap)
*本项目已暂停更新，未来若继续开发，可考虑以下方向：*
- [ ] **多智能体协作 (Multi-Agent)**：引入 LangGraph，实现“审查者”与“执行者”的对抗协作架构。
- [ ] **多模态 RAG**：解析 PDF 中的图表，调用视觉大模型生成图片描述，实现图文混合检索。
- [ ] **长文本流式 TTS 优化**：将长文本切片发送，实现类似真人连读的语音交互。

## 📄 License
BSD-3-Clause

## 👤 作者
- GitHub: [@qingyun5201314](https://github.com/qingyun5201314)
# 多模态 AI Agent 助手

一个基于 RAG + Agent 架构的多功能 AI 助手，支持知识库检索、实时天气查询、数学计算、网页搜索和 AI 图像生成。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📸 功能演示

| 功能 | 说明 |
|------|------|
| 📚 **RAG知识库检索** | 上传文档，基于文档内容回答问题 |
| 🌤️ **实时天气查询** | 接入 Open-Meteo API，查询全球城市天气 |
| 🧮 **数学计算** | 支持复杂数学表达式 |
| ⏰ **时间查询** | 查询当前日期时间 |
| 🔍 **网页搜索** | 基于 webless 的实时信息检索 |
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
│         (main.py + 静态文件)             │
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
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
MODELSCOPE_TOKEN=你的魔搭token
```

### 4. 启动服务

```bash
python main.py
```

### 5. 访问

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```
multi-agent/
├── main.py              # FastAPI 入口 + 静态文件服务
├── agent.py             # Agent 核心（ReAct 范式）
├── vector_db.py         # 向量数据库（FAISS + 语义分块）
├── tools.py             # 工具集（6个工具）
├── config.py            # 配置中心
├── knowledge.txt        # 示例知识库
├── static/              # 前端文件
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🔧 核心实现

### Agent 工作流程（ReAct）

```python
# 1. 思考：LLM 决定下一步做什么
decision = self.think(user_input, history)

# 2. 如果判断需要工具
if "TOOL:" in decision:
    tool_name, args = parse(decision)
    result = await tools.execute(tool_name, args)
    history.append({"tool": tool_name, "result": result})

# 3. 如果判断可以回答
if "FINAL:" in decision:
    return decision.split("FINAL:")[-1]
```

### RAG 语义分块

```python
# 按标点分块（不是按字数硬切）
sentences = re.split(r'(?<=[。！？；.!?;])', text)
# 合并成不超过 max_size 的块
# 保留语义完整性，提高检索精度
```

### 工具路由

Agent 通过提示词规则决定用哪个工具：

- 概念问题 → `search_knowledge`
- 实时信息 → `web_search`
- 数学计算 → `calculate`
- 画图 → `generate_image`

## 📊 项目亮点

1. **多模态能力**：文本对话 + 图像生成，超越纯文本Agent
2. **RAG语义分块**：按标点切分，保留语义完整性，比按字数切分精度更高
3. **异步架构**：FastAPI + async/await，支持并发请求
4. **工具路由**：6个工具，Agent根据问题类型自动选择
5. **错误处理**：网络超时、API限流、工具失败都有兜底

## 🐛 踩坑记录

| 问题 | 解决 |
|------|------|
| `faiss-cpu` 在 Windows + Python 3.13 装不上 | 换 conda 环境 + Python 3.10 |
| `sentence-transformers` 和 `huggingface_hub` 版本冲突 | 降级 `huggingface_hub==0.25.2` |
| `openai` 和 `httpx` 版本不兼容 | 锁定 `openai==1.55.3` + `httpx==0.27.2` |
| HuggingFace 被墙，模型下载失败 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `webless` 同步函数不能在 async 中调用 | 改用异步 `search()` |
| ModelScope 返回 `task_id` 而非直接返回图片 | 轮询 `/v1/tasks/{task_id}` 获取结果 |
| 图片外链加载慢 | 下载转 base64 直接嵌入响应 |

## 🔮 后续计划

- [ ] 加入多轮对话记忆（向量数据库存储历史）
- [ ] 加入语音输入/输出
- [ ] 加入重排（Rerank）优化检索质量
- [ ] 部署到云服务器（Docker + Nginx）
- [ ] 本地部署 Stable Diffusion（替代 API）

## 📄 License

MIT License

## 👤 作者

[你的名字]

- GitHub: [@你的用户名](https://github.com/你的用户名)
- Email: 你的邮箱

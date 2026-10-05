from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from config import Config

app = FastAPI(
    title="多功能AI助手",
    description="RAG + Agent 综合助手 API",
    version="1.0.0"
)

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件夹
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局Agent实例
agent = Agent()

# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    query: str
    max_steps: Optional[int] = 5

class StepInfo(BaseModel):
    step: int
    action: str
    tool: Optional[str] = None
    args: Optional[str] = None
    result: Optional[Any] = None
    content: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: str
    steps: List[StepInfo]
    error: Optional[str] = None

class LoadRequest(BaseModel):
    filepath: str

# ==================== 接口 ====================

@app.get("/")
async def index():
    """首页：返回前端页面"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """核心聊天接口"""
    try:
        agent.max_steps = request.max_steps
        result = await agent.run(request.query)
        
        return ChatResponse(
            success=True,
            answer=result["answer"],
            steps=result["steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/load")
async def load_knowledge(request: LoadRequest):
    success = agent.load_knowledge(request.filepath)
    if success:
        return {"success": True, "message": f"已加载: {request.filepath}"}
    else:
        raise HTTPException(status_code=400, detail="加载失败")

@app.get("/knowledge/stats")
async def knowledge_stats():
    return agent.vector_db.get_stats()

@app.delete("/knowledge/clear")
async def clear_knowledge():
    agent.vector_db.clear()
    return {"success": True, "message": "知识库已清空"}

# ==================== 启动 ====================

if __name__ == "__main__":
    if os.path.exists("knowledge.txt"):
        agent.load_knowledge("knowledge.txt")
    
    print(f"\n🚀 服务启动中...")
    print(f"   网页: http://localhost:{Config.PORT}")
    print(f"   文档: http://localhost:{Config.PORT}/docs")
    print(f"   按 Ctrl+C 停止\n")
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
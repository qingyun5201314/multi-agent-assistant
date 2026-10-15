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
    description="RAG + Agent + 记忆 + 多会话",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

agent = Agent()

# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
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
    session_id: str
    error: Optional[str] = None

class LoadRequest(BaseModel):
    filepath: str

class SessionRequest(BaseModel):
    session_id: str

# ==================== 接口 ====================

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        agent.max_steps = request.max_steps
        result = await agent.run(request.query, session_id=request.session_id)
        
        return ChatResponse(
            success=True,
            answer=result["answer"],
            steps=result["steps"],
            session_id=result["session_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 多会话管理 ====================

@app.get("/sessions")
async def list_sessions():
    """获取所有会话列表"""
    return {"sessions": agent.get_all_sessions()}

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取某个会话的所有消息"""
    return {"messages": agent.get_session_messages(session_id)}

@app.post("/sessions/clear")
async def clear_session(request: SessionRequest):
    """清空某个会话的消息（保留会话）"""
    agent.clear_session(request.session_id)
    return {"success": True, "message": f"会话 {request.session_id} 已清空"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除某个会话"""
    agent.delete_session(session_id)
    return {"success": True, "message": f"会话 {session_id} 已删除"}

# ==================== 记忆管理 ====================

@app.get("/memory/stats")
async def memory_stats():
    return agent.get_memory_stats()

@app.get("/memory/all")
async def all_memories():
    return {"memories": agent.get_all_memories()}

@app.delete("/memory/clear")
async def clear_long_term_memory():
    agent.clear_long_term_memory()
    return {"success": True, "message": "长期记忆已清空"}

# ==================== 知识库管理 ====================

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
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from config import Config

app = FastAPI(title="多功能AI助手", version="1.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

agent = Agent()

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

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

# ==================== 流式聊天接口 ====================

@app.get("/chat_stream")
async def chat_stream(
    query: str = Query(...),
    session_id: str = Query("default"),
    max_steps: int = Query(5)
):
    agent.max_steps = max_steps
    async def event_generator():
        try:
            async for event in agent.run_stream(query, session_id=session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # 强制关闭 Nginx 等代理的缓冲
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        full_text = ""
        steps = []
        async for event in agent.run_stream(request.query, session_id=request.session_id):
            if event["type"] == "text":
                full_text += event["content"]
            elif event["type"] == "final":
                steps = event.get("steps", [])
        return ChatResponse(success=True, answer=full_text, steps=steps, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 其他管理接口 ====================

@app.get("/sessions")
async def list_sessions(): return {"sessions": agent.get_all_sessions()}

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str): return {"messages": agent.get_session_messages(session_id)}

@app.post("/sessions/clear")
async def clear_session(request: SessionRequest):
    agent.clear_session(request.session_id)
    return {"success": True, "message": "会话已清空"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    agent.delete_session(session_id)
    return {"success": True, "message": "会话已删除"}

@app.post("/knowledge/load")
async def load_knowledge(request: LoadRequest):
    success = agent.load_knowledge(request.filepath)
    if success: return {"success": True, "message": "已加载"}
    raise HTTPException(status_code=400, detail="加载失败")

@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    try:
        allowed_ext = ['.txt', '.md', '.markdown', '.pdf']
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")
        
        content = await file.read()
        success = agent.vector_db.load_from_bytes(content, file.filename)
        
        if success: return {"success": True, "message": "已上传", "stats": agent.vector_db.get_stats()}
        raise HTTPException(status_code=400, detail="加载失败")
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/sources")
async def knowledge_sources(): return {"sources": agent.vector_db.get_sources()}

@app.get("/knowledge/stats")
async def knowledge_stats(): return agent.vector_db.get_stats()

@app.delete("/knowledge/clear")
async def clear_knowledge():
    agent.vector_db.clear()
    return {"success": True, "message": "知识库已清空"}

@app.get("/memory/stats")
async def memory_stats(): return agent.get_memory_stats()

if __name__ == "__main__":
    print(f"\n🚀 服务启动中...")
    print(f"   请在浏览器访问: http://localhost:{Config.PORT}")
    print(f"   按 Ctrl+C 停止\n")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
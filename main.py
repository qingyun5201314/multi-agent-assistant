from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn
import os
import json
import asyncio
import tempfile
import edge_tts
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

from agent import Agent
from config import Config

app = FastAPI(title="多功能AI助手", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

agent = Agent()

# ==================== 加载本地 Whisper 模型 ====================
print("🎙️ 正在加载本地 Whisper 模型...")
try:
    # 自动检测是否有 GPU，如果没有则会降级到 CPU
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
except Exception:
    print("⚠️ GPU 加载失败，尝试使用 CPU 运行 Whisper...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper 模型加载完毕！")

# ==================== 数据模型 ====================

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

# ==================== 基础接口 ====================

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

# ==================== 语音识别 (STT) ====================

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """接收音频文件，返回识别出的文字"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name
        
        # 执行识别
        segments, info = whisper_model.transcribe(temp_path, beam_size=5, language="zh")
        text = "".join([segment.text for segment in segments]).strip()
        
        os.remove(temp_path)
        return {"success": True, "text": text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== 语音合成 (TTS) ====================

@app.get("/tts")
async def text_to_speech(text: str = Query(...)):
    """接收文字，返回合成的音频流"""
    try:
        # 使用微软 Edge 免费语音库，晓晓（女）
        voice = "zh-CN-XiaoxiaoNeural"
        communicate = edge_tts.Communicate(text, voice)
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                await asyncio.sleep(0) # 强制刷新缓冲
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式兼容接口"""
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

# ==================== 会话与知识库管理 ====================

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

# ==================== 启动 ====================

if __name__ == "__main__":
    print(f"\n🚀 服务启动中...")
    print(f"   请在浏览器访问: http://localhost:{Config.PORT}")
    print(f"   按 Ctrl+C 停止\n")
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, log_level="info")
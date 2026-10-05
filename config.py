import os
from dotenv import load_dotenv

# 加载 .env 文件（关键！）
load_dotenv()

class Config:
    """配置中心"""
    
    # ===== API配置 =====
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
    MODELSCOPE_TOKEN = os.getenv("MODELSCOPE_TOKEN", "")
    
    # 启动时校验
    if not API_KEY:
        raise ValueError(
            "❌ 请设置环境变量 OPENAI_API_KEY\n"
            "Windows设置方法：\n"
            "  1. 创建 .env 文件并填入\n"
            "  2. 或命令行: set OPENAI_API_KEY=your-key"
        )
    
    # ===== Agent参数 =====
    MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # ===== RAG参数 =====
    TOP_K = int(os.getenv("TOP_K", "3"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    
    # ===== 服务参数 =====
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
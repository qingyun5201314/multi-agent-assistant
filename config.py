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
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    
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

    # ===== 混合检索参数 =====
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"  # 是否启用混合检索
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))    # 向量检索取几条
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", "5"))        # BM25检索取几条
    RRF_K = int(os.getenv("RRF_K", "60"))                 # RRF公式的k值
    FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))      # 最终返回几条

    # ===== 重排参数 =====
    USE_RERANK = os.getenv("USE_RERANK", "true").lower() == "true"  # 是否启用重排
    RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10"))   # 重排候选数量
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))              # 重排后保留几条

    # ===== 上下文压缩参数 =====
    USE_COMPRESSION = os.getenv("USE_COMPRESSION", "true").lower() == "true"
import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from config import Config


class VectorDB:
    """向量数据库"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("📦 加载向量模型...")
        self.encoder = SentenceTransformer(model_name)
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings = None
        self.index = None
    
    def add(self, documents: List[str], metadatas: List[Dict] = None):
        if not documents:
            return
        
        new_embeddings = self.encoder.encode(documents)
        self.documents.extend(documents)
        self.metadata.extend(metadatas or [{"source": "unknown"}] * len(documents))
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.documents or self.index is None:
            return []
    
        query_embedding = self.encoder.encode([query])
        # 多取一些候选，然后去重
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k * 3, len(self.documents))
        )
    
        results = []
        seen_contents = set()
    
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                content = self.documents[idx]
                # 去重：内容相同就跳过
                if content in seen_contents:
                    continue
                seen_contents.add(content)
            
                results.append({
                    'content': content,
                    'source': self.metadata[idx].get('source', 'unknown'),
                    'score': float(1 / (1 + distances[0][i]))
                })
            
                # 够了就停
                if len(results) >= top_k:
                    break
    
        return results
    
    # ==================== 核心改动：段落独立分块 ====================
    
    def _split_by_semantic(self, text: str, max_chunk_size: int = None) -> List[str]:
        """
        按语义边界切分文本（改进版）
        
        规则：
        1. 优先按段落（\n\n）切分，每个段落作为独立块
        2. 如果段落超过 max_chunk_size，在句号、问号、感叹号处切分
        3. 如果单句还是超长，在逗号处切分
        4. 最后才按字数硬切
        """
        max_size = max_chunk_size or Config.CHUNK_SIZE
        
        # Step 1: 按双换行分段
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 如果双换行没分开，再按单换行分
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'\n', text)
        
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Step 2: 每段作为独立块处理（不跨段合并）
        chunks = []
        for para in paragraphs:
            if len(para) <= max_size:
                # 段落本身够短，直接作为一个块
                chunks.append(para)
            else:
                # 段落太长，按句子切分
                sentences = re.split(r'(?<=[。！？；.!?;])', para)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) <= max_size:
                        current += sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
                if current:
                    chunks.append(current)
        
        # Step 3: 处理仍然超长的块（在逗号处切分）
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_size * 1.5:
                final_chunks.append(chunk)
            else:
                parts = re.split(r'(?<=[，,、])', chunk)
                temp = ""
                for part in parts:
                    if len(temp) + len(part) <= max_size:
                        temp += part
                    else:
                        if temp:
                            final_chunks.append(temp)
                        temp = part
                if temp:
                    final_chunks.append(temp)
        
        # 过滤空块和太短的块
        final_chunks = [c.strip() for c in final_chunks if len(c.strip()) > 5]
        
        return final_chunks
    
    def load_from_file(self, filepath: str) -> bool:
        """从文件加载知识（使用语义分块）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用语义分块
            chunks = self._split_by_semantic(content, Config.CHUNK_SIZE)
            
            print(f"📊 分块结果：{len(chunks)} 块")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"  {i}. [{len(chunk)}字] {chunk[:40]}...")
            
            self.add(chunks, [{"source": filepath}] * len(chunks))
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        return {
            "文档数量": len(self.documents),
            "向量维度": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "平均块长度": sum(len(d) for d in self.documents) // len(self.documents) if self.documents else 0
        }
    
    def clear(self):
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
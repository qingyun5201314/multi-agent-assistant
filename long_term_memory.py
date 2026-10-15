import os
import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from config import Config


class LongTermMemory:
    """长期记忆（向量数据库 + 语义去重）"""
    
    def __init__(self, memory_dir: str = "memory_cache"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        print("🧠 加载长期记忆...")
        
        # 向量模型
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 存储文件
        self.index_path = os.path.join(memory_dir, "memory.faiss")
        self.metadata_path = os.path.join(memory_dir, "memory.json")
        
        # LLM（用于判断和提取）
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL
        
        # 加载或初始化
        self.memories: List[Dict] = []
        self.embeddings = None
        self.index = None
        self._load()
    
    def _load(self):
        """加载已有的记忆"""
        # 加载元数据
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
            print(f"   ✅ 加载了 {len(self.memories)} 条长期记忆")
        
        # 加载FAISS索引
        if os.path.exists(self.index_path) and self.memories:
            self.index = faiss.read_index(self.index_path)
            # 从索引重建 embeddings
            try:
                self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            except Exception as e:
                print(f"   ⚠️ 重建embeddings失败: {e}，将重建索引")
                self._rebuild_index()
        
        # 如果索引不存在但有记忆，重建
        if self.index is None and self.memories:
            self._rebuild_index()
    
    def _save(self):
        """保存记忆到磁盘"""
        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
    
    def _rebuild_index(self):
        """重建FAISS索引"""
        if not self.memories:
            self.index = None
            self.embeddings = None
            return
        
        # 重新编码所有记忆
        texts = [m["content"] for m in self.memories]
        self.embeddings = self.encoder.encode(texts)
        
        # 建索引
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))
    
    def should_store(self, user_input: str, assistant_reply: str) -> Dict:
        """用LLM判断：这条对话值不值得存到长期记忆"""
        prompt = f"""判断以下对话是否包含"值得长期记住的用户信息"。

值得存的信息类型：
- 用户的身份、职业、背景
- 用户的偏好、习惯
- 用户的长期目标、项目
- 用户明确要求记住的事实

不值得存的：
- 临时性问题（如"今天天气"）
- 客套话（如"谢谢"、"你好"）
- 通用知识问答

用户：{user_input}
助手：{assistant_reply}

请严格按JSON格式返回：
{{"should_store": true/false, "memory": "如果要存，提取出关键信息（一句话）；否则为空"}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # 去掉可能的markdown代码块标记
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"⚠️ 判断失败: {e}")
            return {"should_store": False, "memory": ""}
    
    def add_memory(self, content: str, session_id: str = "default", similarity_threshold: float = 0.85):
        """添加一条长期记忆（带语义去重）"""
        if not content.strip():
            return
        
        # ===== 语义去重：检索相似记忆 =====
        if self.memories and self.index is not None:
            # 用新内容去检索已有的记忆
            query_embedding = self.encoder.encode([content])
            distances, indices = self.index.search(
                query_embedding.astype('float32'),
                min(3, len(self.memories))  # 只查最相似的3条
            )
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.memories):
                    # 计算相似度
                    similarity = float(1 / (1 + distances[0][i]))
                    
                    if similarity >= similarity_threshold:
                        existing = self.memories[idx]["content"]
                        print(f"   ℹ️ 语义重复，跳过（相似度 {similarity:.3f}）")
                        print(f"      已有: {existing}")
                        print(f"      新加: {content}")
                        return
        
        # ===== 通过去重检查，存入 =====
        self.memories.append({
            "content": content,
            "session_id": session_id,
            "created_at": str(np.datetime64('now'))
        })
        
        # 重建索引
        self._rebuild_index()
        
        # 保存到磁盘
        self._save()
        
        print(f"   ✅ 新增长期记忆: {content}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关记忆"""
        if not self.memories or self.index is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.memories))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):
                score = float(1 / (1 + distances[0][i]))
                # 过滤太不相关的
                if score < 0.3:
                    continue
                results.append({
                    "content": self.memories[idx]["content"],
                    "score": score,
                    "created_at": self.memories[idx].get("created_at", "")
                })
        
        return results
    
    def get_all(self) -> List[Dict]:
        """获取所有记忆"""
        return self.memories
    
    def delete_memory(self, index: int):
        """删除指定索引的记忆"""
        if 0 <= index < len(self.memories):
            removed = self.memories.pop(index)
            self._rebuild_index()
            self._save()
            print(f"   🗑️ 已删除记忆: {removed['content']}")
            return True
        return False
    
    def clear(self):
        """清空所有长期记忆"""
        self.memories = []
        self.embeddings = None
        self.index = None
        
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        
        print("🗑️ 长期记忆已清空")
    
    def get_stats(self) -> Dict:
        """统计信息"""
        return {
            "记忆条数": len(self.memories)
        }
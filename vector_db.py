import re
import os
import json
import jieba
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from config import Config


class VectorDB:
    """向量数据库（多格式 + 持久化 + 混合检索 + 重排）"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', persist_dir: str = "knowledge_db"):
        print("📦 加载向量模型...")
        self.encoder = SentenceTransformer(model_name)
        
        # 持久化目录
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.data_path = os.path.join(persist_dir, "data.json")
        
        # 内存数据
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        
        # LLM 客户端（用于重排）
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL
        
        # 加载已有数据
        self._load()
    
    # ==================== 持久化 ====================
    
    def _load(self):
        if os.path.exists(self.data_path) and os.path.exists(self.index_path):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get("documents", [])
                self.metadata = data.get("metadata", [])
                
                if self.documents:
                    self.index = faiss.read_index(self.index_path)
                    self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
                    self._build_bm25()
                    print(f"✅ 已加载知识库：{len(self.documents)} 个文档块")
            except Exception as e:
                print(f"⚠️ 知识库加载失败: {e}")
                self.documents = []
                self.metadata = []
                self.index = None
                self.embeddings = None
    
    def _save(self):
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "metadata": self.metadata
                }, f, ensure_ascii=False)
            
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
        except Exception as e:
            print(f"⚠️ 知识库保存失败: {e}")
    
    def _rebuild_index(self):
        if not self.documents:
            self.index = None
            self.embeddings = None
            self.bm25 = None
            return
        
        self.embeddings = self.encoder.encode(self.documents)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float32'))
        self._build_bm25()
    
    # ==================== 分词 ====================
    
    def _tokenize(self, text: str) -> List[str]:
        """智能分词：中文用jieba，英文用正则"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'\w', text))
        
        if total_chars == 0:
            return []
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.3:
            return [w.strip() for w in jieba.cut(text) if w.strip()]
        else:
            return re.findall(r'\b\w+\b', text.lower())
    
    def _build_bm25(self):
        if not self.documents:
            self.bm25 = None
            return
        
        tokenized = [self._tokenize(doc) for doc in self.documents]
        tokenized = [t if t else ['empty'] for t in tokenized]
        self.bm25 = BM25Okapi(tokenized)
    
    # ==================== 核心操作 ====================
    
    def add(self, documents: List[str], metadatas: List[Dict] = None, source_name: str = None):
        if not documents:
            return
        
        if source_name:
            self._remove_by_source(source_name)
        
        self.documents.extend(documents)
        self.metadata.extend(metadatas or [{"source": "unknown"}] * len(documents))
        
        self._rebuild_index()
        self._save()
    
    def _remove_by_source(self, source_name: str):
        indices_to_keep = [
            i for i, m in enumerate(self.metadata)
            if m.get("source") != source_name
        ]
        
        if len(indices_to_keep) == len(self.documents):
            return
        
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadata = [self.metadata[i] for i in indices_to_keep]
        print(f"🗑️ 已移除同源旧文档: {source_name}")
    
    # ==================== 检索 ====================
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        if not self.documents or self.index is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.documents))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'idx': int(idx),
                    'content': self.documents[idx],
                    'source': self.metadata[idx].get('source', 'unknown'),
                    'vector_score': float(1 / (1 + distances[0][i]))
                })
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if not self.documents or self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'idx': int(idx),
                    'content': self.documents[idx],
                    'source': self.metadata[idx].get('source', 'unknown'),
                    'bm25_score': float(scores[idx])
                })
        return results
    
    def _rrf_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        rrf_scores = {}
        doc_map = {}
        
        for rank, r in enumerate(vector_results):
            idx = r['idx']
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
            doc_map[idx] = r
        
        for rank, r in enumerate(bm25_results):
            idx = r['idx']
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
            if idx not in doc_map:
                doc_map[idx] = r
        
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_items:
            doc = doc_map[idx].copy()
            doc['rrf_score'] = score
            results.append(doc)
        
        return results
    
    # ==================== 重排 ====================
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        用LLM对候选结果重排
        给每个候选打0-10分，按分数排序
        """
        if not candidates:
            return []
        
        # 如果候选本来就少，直接返回
        if len(candidates) <= top_k:
            return candidates
        
        # 构造候选列表
        candidates_text = ""
        for i, c in enumerate(candidates):
            # 每条截取前200字，避免prompt过长
            content_preview = c['content'][:200].replace('\n', ' ')
            candidates_text += f"\n[{i+1}] 来源: {c['source']}\n内容: {content_preview}\n"
        
        prompt = f"""你是一个相关性评估专家。请对以下候选文档进行相关性评分。

用户查询：{query}

候选文档：
{candidates_text}

评分标准：
- 10分：完全相关，直接回答用户问题
- 7-9分：高度相关，包含关键信息
- 4-6分：部分相关，包含部分有用信息
- 1-3分：弱相关，仅主题相近
- 0分：完全不相关

请为每个候选打分，只返回JSON格式，例如：
{{"1": 8, "2": 3, "3": 9, ...}}

只返回JSON，不要其他内容。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # 去掉markdown代码块
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            scores = json.loads(content)
            
            # 给候选加分数
            for i, c in enumerate(candidates):
                key = str(i + 1)
                c['rerank_score'] = scores.get(key, 0)
            
            # 按重排分数排序
            candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # 过滤掉0分的
            candidates = [c for c in candidates if c.get('rerank_score', 0) > 0]
            
            return candidates[:top_k]
        
        except Exception as e:
            print(f"⚠️ 重排失败，返回原始结果: {e}")
            return candidates[:top_k]
    
    # ==================== 主检索入口 ====================
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索（混合检索 + 重排）"""
        if not self.documents:
            return []
        
        # 是否启用混合检索
        if not Config.USE_HYBRID_SEARCH:
            results = self._vector_search(query, top_k)
            return [
                {'content': r['content'], 'source': r['source'], 'score': r['vector_score']}
                for r in results
            ]
        
        # Step 1: 混合检索
        print(f"🔀 [混合检索] 向量top-{Config.VECTOR_TOP_K} + BM25 top-{Config.BM25_TOP_K} → RRF融合")
        
        vector_results = self._vector_search(query, Config.VECTOR_TOP_K)
        bm25_results = self._bm25_search(query, Config.BM25_TOP_K)
        
        print(f"   向量命中: {len(vector_results)}, BM25命中: {len(bm25_results)}")
        
        # Step 2: RRF融合（多取一些作为重排候选）
        candidates = self._rrf_fusion(
            vector_results, 
            bm25_results, 
            k=Config.RRF_K
        )[:Config.RERANK_CANDIDATES]
        
        print(f"   RRF融合后候选: {len(candidates)} 条")
        
        # Step 3: 重排
        if Config.USE_RERANK and len(candidates) > 1:
            print(f"🎯 [重排] LLM打分中...")
            candidates = self._rerank(query, candidates, top_k)
            print(f"   重排后保留: {len(candidates)} 条")
        else:
            candidates = candidates[:top_k]
        
        # 返回
        return [
            {
                'content': r['content'],
                'source': r['source'],
                'score': r.get('rerank_score', r.get('rrf_score', 0)),
                'rrf_score': r.get('rrf_score', 0),
                'rerank_score': r.get('rerank_score', 0),
                'vector_score': r.get('vector_score', 0),
                'bm25_score': r.get('bm25_score', 0)
            }
            for r in candidates
        ]
    
    # ==================== 语义分块 ====================
    
    def _split_by_semantic(self, text: str, max_chunk_size: int = None) -> List[str]:
        max_size = max_chunk_size or Config.CHUNK_SIZE
        
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'\n', text)
        
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        for para in paragraphs:
            if len(para) <= max_size:
                chunks.append(para)
            else:
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
        
        return [c.strip() for c in final_chunks if len(c.strip()) > 5]
    
    # ==================== 文件解析 ====================
    
    def _parse_pdf(self, filepath: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            print(f"❌ PDF解析失败: {e}")
            return ""
    
    def _parse_txt(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                print(f"❌ 文件读取失败: {e}")
                return ""
        except Exception as e:
            print(f"❌ 文件读取失败: {e}")
            return ""
    
    def load_from_file(self, filepath: str) -> bool:
        try:
            if not os.path.exists(filepath):
                return False
            
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.pdf':
                content = self._parse_pdf(filepath)
            elif ext in ['.txt', '.md', '.markdown']:
                content = self._parse_txt(filepath)
            else:
                return False
            
            if not content.strip():
                return False
            
            chunks = self._split_by_semantic(content, Config.CHUNK_SIZE)
            source_name = os.path.basename(filepath)
            print(f"📄 解析 {source_name}：{len(chunks)} 块")
            
            self.add(chunks, [{"source": source_name}] * len(chunks), source_name=source_name)
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def load_from_bytes(self, content_bytes: bytes, filename: str) -> bool:
        try:
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.pdf':
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(content_bytes)
                    tmp_path = tmp.name
                try:
                    content = self._parse_pdf(tmp_path)
                finally:
                    os.unlink(tmp_path)
            elif ext in ['.txt', '.md', '.markdown']:
                try:
                    content = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    content = content_bytes.decode('gbk')
            else:
                return False
            
            if not content.strip():
                return False
            
            chunks = self._split_by_semantic(content, Config.CHUNK_SIZE)
            print(f"📄 解析 {filename}：{len(chunks)} 块")
            
            self.add(chunks, [{"source": filename}] * len(chunks), source_name=filename)
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    # ==================== 管理 ====================
    
    def get_sources(self) -> List[Dict]:
        sources = {}
        for meta in self.metadata:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        
        return [
            {"source": src, "chunk_count": count}
            for src, count in sources.items()
        ]
    
    def get_stats(self) -> Dict:
        return {
            "文档数量": len(self.documents),
            "向量维度": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "文档源数量": len(set(m.get("source", "unknown") for m in self.metadata)),
            "混合检索": "开启" if Config.USE_HYBRID_SEARCH else "关闭",
            "重排": "开启" if Config.USE_RERANK else "关闭"
        }
    
    def clear(self):
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        
        print("🗑️ 知识库已清空")
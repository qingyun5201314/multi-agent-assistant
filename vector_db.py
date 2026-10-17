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
    """向量数据库（多格式 + 持久化 + 混合检索 + 重排 + 多路召回 + 上下文压缩）"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', persist_dir: str = "knowledge_db"):
        print("📦 加载向量模型...")
        self.encoder = SentenceTransformer(model_name)
        
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.data_path = os.path.join(persist_dir, "data.json")
        
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL
        
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
    
    # ==================== 检索引擎 ====================
    
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
    
    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """LLM 重排"""
        if not candidates:
            return []
        
        if len(candidates) <= top_k:
            return candidates
        
        candidates_text = ""
        for i, c in enumerate(candidates):
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
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            scores = json.loads(content)
            
            for i, c in enumerate(candidates):
                key = str(i + 1)
                c['rerank_score'] = scores.get(key, 0)
            
            candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            candidates = [c for c in candidates if c.get('rerank_score', 0) > 0]
            
            return candidates[:top_k]
        
        except Exception as e:
            print(f"⚠️ 重排失败，返回原始结果: {e}")
            return candidates[:top_k]

    # ==================== 上下文压缩 ====================
    
    def _compress_context(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """用LLM提取核心上下文（保留来源，防止废话）"""
        if not candidates:
            return []
        
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"\n[片段 {i+1}]\n{c['content']}\n"
        
        prompt = f"""你是一个专业的信息提取专家。请从以下检索到的片段中，提取出能够直接回答用户查询的核心信息。

用户查询：{query}

检索到的片段：
{candidates_text}

提取要求：
1. 严格过滤掉无关的背景介绍、客套话、参考文献。
2. 只保留能回答问题的事实、数据或结论。
3. 必须保留信息的出处标注，如 [片段 1]。
4. 如果某个片段无法回答该查询，直接忽略该片段，不要提及它。
5. 即使提取出的内容很短，只要切中要害，也直接输出。
6. 【绝对禁止】不要输出"无相关内容"、"未找到"、"抱歉"等废话。如果没有提取到任何信息，直接输出空字符串。

提取结果："""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            compressed_content = response.choices[0].message.content.strip()
            
            # 如果模型真的没提取出任何东西，降级返回原结果
            if not compressed_content:
                print("   ⚠️ 压缩后无有效内容，保留原样")
                return candidates
                
            compressed_candidate = {
                'content': compressed_content,
                'source': "综合提炼（原自: " + ", ".join(set(c['source'] for c in candidates)) + "）",
                'score': candidates[0].get('rerank_score', 0),
                'rrf_score': candidates[0].get('rrf_score', 0),
                'rerank_score': candidates[0].get('rerank_score', 0)
            }
            
            return [compressed_candidate]
            
        except Exception as e:
            print(f"⚠️ 上下文压缩失败: {e}")
            return candidates

    # ==================== 多路召回检索入口 ====================
    
    def search_multi_query(self, queries: List[str], top_k: int = 3) -> List[Dict]:
        """
        多路查询检索 + 重排 + 压缩
        """
        if not self.documents:
            return []
            
        if not Config.USE_HYBRID_SEARCH:
            results = self._vector_search(queries[0], top_k)
            return [
                {'content': r['content'], 'source': r['source'], 'score': r['vector_score']}
                for r in results
            ]
        
        all_vector_results = []
        all_bm25_results = []
        
        for q in queries:
            if not q.strip():
                continue
                
            vec_res = self._vector_search(q, Config.VECTOR_TOP_K)
            all_vector_results.extend(vec_res)
            
            bm25_res = self._bm25_search(q, Config.BM25_TOP_K)
            all_bm25_results.extend(bm25_res)
        
        # 去重（根据 idx 保留最高分）
        unique_vector = {}
        for r in all_vector_results:
            idx = r['idx']
            if idx not in unique_vector or r['vector_score'] > unique_vector[idx]['vector_score']:
                unique_vector[idx] = r
                
        unique_bm25 = {}
        for r in all_bm25_results:
            idx = r['idx']
            if idx not in unique_bm25 or r['bm25_score'] > unique_bm25[idx]['bm25_score']:
                unique_bm25[idx] = r
                
        vector_results = list(unique_vector.values())
        bm25_results = list(unique_bm25.values())
        
        print(f"🔀 [多路检索] 合并后：向量候选 {len(vector_results)} 条，BM25候选 {len(bm25_results)} 条")
        
        # RRF融合
        candidates = self._rrf_fusion(
            vector_results, 
            bm25_results, 
            k=Config.RRF_K
        )[:Config.RERANK_CANDIDATES]
        
        print(f"   RRF融合后候选: {len(candidates)} 条")
        
        # 重排
        if Config.USE_RERANK and len(candidates) > 1:
            print(f"🎯 [重排] LLM打分中...")
            candidates = self._rerank(queries[0], candidates, top_k)
            print(f"   重排后保留: {len(candidates)} 条")
        else:
            candidates = candidates[:top_k]
            
        # 全局浓缩（上下文压缩）
        if Config.USE_COMPRESSION and len(candidates) > 0:
            print(f"🗜️ [压缩] 提取核心上下文...")
            candidates = self._compress_context(queries[0], candidates)
            print(f"   压缩后保留: {len(candidates)} 条")
            
        return [
            {
                'content': r['content'],
                'source': r['source'],
                'score': r.get('rerank_score', r.get('rrf_score', 0)),
                'rrf_score': r.get('rrf_score', 0),
                'rerank_score': r.get('rerank_score', 0)
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
            "重排": "开启" if Config.USE_RERANK else "关闭",
            "多路召回": "开启",
            "上下文压缩": "开启" if Config.USE_COMPRESSION else "关闭"
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
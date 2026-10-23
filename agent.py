import json
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
from langfuse import Langfuse
from vector_db import VectorDB
from tools import Tools
from config import Config
from database import Database
from long_term_memory import LongTermMemory

# 直接实例化，并显式传入密钥，避免因环境变量读取失败导致 Tracer 为 None
langfuse = Langfuse(
    public_key=Config.LANGFUSE_PUBLIC_KEY,
    secret_key=Config.LANGFUSE_SECRET_KEY,
    host="https://cloud.langfuse.com"
)

class Agent:
    """ReAct Agent（极致流式 + Langfuse 监控）"""
    
    MAX_MESSAGES_BEFORE_SUMMARY = 20
    KEEP_RECENT_MESSAGES = 10
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL
        self.vector_db = VectorDB()
        self.tools = Tools(self.vector_db)
        self.max_steps = Config.MAX_STEPS
        
        self.db = Database("memory.db")
        self.ltm = LongTermMemory("memory_cache")
        
        self.tool_descriptions = """
可用工具：
1. get_weather(city) - 查询天气
2. calculate(expression) - 数学计算
3. get_time() - 查询当前时间
4. search_knowledge(query) - 检索本地知识库
5. web_search(query) - 搜索互联网实时信息
6. generate_image(prompt) - 生成图像

工具选择规则：
- 概念解释、定义、专业术语 → 优先用 search_knowledge
- 知识库没有的答案，用 web_search
- 实时信息、新闻 → 用 web_search
- 数学计算 → 用 calculate
- 时间日期 → 用 get_time
- 画图、生成图片 → 用 generate_image

使用格式：TOOL: 工具名(参数)
直接回答：FINAL: 你的回答
"""
    
    # ==================== 基础接口 ====================
    
    def load_knowledge(self, filepath: str) -> bool:
        return self.vector_db.load_from_file(filepath)
    
    def get_all_sessions(self) -> List[Dict]:
        return self.db.get_all_sessions()
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        return self.db.get_all_messages(session_id)
    
    def delete_session(self, session_id: str):
        self.db.delete_session(session_id)
    
    def clear_session(self, session_id: str):
        self.db.clear_session(session_id)
        
    def get_memory_stats(self) -> Dict:
        return {
            "短期记忆": self.db.get_stats(),
            "长期记忆": self.ltm.get_stats()
        }
    
    def get_all_memories(self) -> List[Dict]:
        return self.ltm.get_all()
    
    def clear_long_term_memory(self):
        self.ltm.clear()

    # ==================== 核心逻辑 ====================
    
    def _rewrite_query(self, query: str) -> str:
        prompt = f"""你是一个搜索专家。请将用户的查询改写成2-4个精准的搜索关键词，用于在知识库中检索。
不要解释，直接输出关键词（用空格隔开）。如果原查询已经很简短精准，直接原样返回。
用户查询：{query}
改写结果："""
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=30
            )
            return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        except Exception:
            return query

    def _maybe_summarize(self, session_id: str):
        message_count = self.db.get_message_count(session_id)
        if message_count <= self.MAX_MESSAGES_BEFORE_SUMMARY: return
        info = self.db.get_session_summary(session_id)
        until_id = info["summary_until_id"]
        latest_id = self.db.get_latest_message_id(session_id)
        summary_end_id = latest_id - self.KEEP_RECENT_MESSAGES
        if summary_end_id <= until_id: return
        
        messages_to_summarize = self.db.get_old_messages(session_id, before_id=until_id, limit=50)
        messages_to_summarize = [m for m in messages_to_summarize if m["id"] <= summary_end_id]
        if not messages_to_summarize: return
        
        old_text = info["summary"] if info["summary"] else "（无）"
        new_text = "\n".join([f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in messages_to_summarize])
        prompt = f"请把以下对话压缩成200字以内的摘要。\n【已有摘要】{old_text}\n【新增对话】{new_text}\n输出摘要："
        try:
            res = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=300
            )
            self.db.update_session_summary(session_id, res.choices[0].message.content.strip(), summary_end_id)
        except Exception: pass

    def _generate_title(self, first_message: str) -> str:
        prompt = f"请用不超过10个字概括主题，只返回标题：\n{first_message}"
        try:
            res = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=30
            )
            return res.choices[0].message.content.strip().replace('"', '').replace('\n', '')[:20]
        except Exception:
            return first_message[:10]
    
    def _build_context(self, query: str, session_id: str) -> str:
        ltm_memories = self.ltm.search(query, top_k=3)
        ltm_text = "\n".join([f"- {m['content']}" for m in ltm_memories]) if ltm_memories else "无"
        
        summary = self.db.get_session_summary(session_id)["summary"]
        history = self.db.get_history(session_id, limit=20)
        history_text = "\n".join([f"{'用户' if m['role']=='user' else '助手'}: {m['content']}" for m in history])
        
        return f"【用户长期背景】\n{ltm_text}\n\n【早期摘要】\n{summary}\n\n【最近对话】\n{history_text}"
    
    def think(self, user_input: str, session_id: str, history: List[Dict]) -> str:
        context = self._build_context(user_input, session_id)
        prompt = f"""你是一个多功能AI助手。
{self.tool_descriptions}
{context}
【本轮执行记录】{json.dumps(history, ensure_ascii=False) if history else "无"}
【用户新问题】{user_input}
请决定下一步。"""
        
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=Config.TEMPERATURE, max_tokens=500
        )
        return response.choices[0].message.content
    
    # ==================== 流式输出与监控 ====================
    
    async def run_stream(self, user_input: str, session_id: str = "default"):
        """极致流式运行Agent（带 Langfuse 监控）"""
        
        # 使用 start_as_current_observation 兼容新版 SDK
        with langfuse.start_as_current_observation(
            name="chat_session", 
            as_type="span",
            input=user_input
        ) as root_span:
            
            # 更新 Trace 级别的 session_id（如果 SDK 支持的话，不支持就忽略）
            try:
                root_span.update_trace(session_id=session_id)
            except Exception:
                pass

            is_first_message = self.db.get_message_count(session_id) == 0
            self.db.add_message(session_id, "user", user_input)
            
            if is_first_message:
                title = self._generate_title(user_input)
                self.db.set_session_title(session_id, title)
                yield {"type": "title", "content": title}
            
            self._maybe_summarize(session_id)
            
            history = []
            steps_log = []
            
            for step in range(self.max_steps):
                # 记录“思考”这一步
                with root_span.start_as_current_observation(
                    name=f"think_step_{step+1}", 
                    as_type="span"
                ) as think_span:
                    decision = self.think(user_input, session_id, history)
                    think_span.update(output=decision[:100])
                
                if "FINAL:" in decision:
                    steps_log.append({"step": step + 1, "action": "final", "content": decision.split("FINAL:")[-1].strip()})
                    break
                
                elif "TOOL:" in decision:
                    tool_line = decision.split("TOOL:")[-1].strip()
                    if "(" in tool_line and ")" in tool_line:
                        tool_name = tool_line.split("(")[0].strip()
                        args = tool_line.split("(")[1].split(")")[0].strip()
                        
                        yield {"type": "status", "content": f"正在执行 {tool_name}..."}
                        
                        # 记录“工具执行”这一步
                        with root_span.start_as_current_observation(
                            name=f"tool_{tool_name}", 
                            as_type="span"
                        ) as tool_span:
                            if tool_name == "search_knowledge":
                                rewritten = self._rewrite_query(args)
                                yield {"type": "status", "content": f"改写查询: {rewritten}"}
                                result = await self.tools.execute(tool_name, [args, rewritten])
                            else:
                                result = await self.tools.execute(tool_name, args)
                            tool_span.update(output=json.dumps(result, ensure_ascii=False)[:200])
                        
                        history.append({"step": step + 1, "tool": tool_name, "args": args, "result": result})
                        steps_log.append({"step": step + 1, "action": "tool", "tool": tool_name, "args": args, "result": result})
                        
                        yield {"type": "tool_result", "tool": tool_name, "result": result}
                else:
                    steps_log.append({"step": step + 1, "action": "final", "content": decision})
                    break
            
            # ================= 最终强制流式生成阶段 =================
            context = self._build_context(user_input, session_id)
            tool_context = ""
            if history:
                tool_context = "【刚刚查询到的资料】\n" + "\n".join([f"工具 {h['tool']} 返回: {json.dumps(h['result'], ensure_ascii=False)}" for h in history])

            final_prompt = f"""你是一个多功能AI助手。
            
{context}

{tool_context}

用户问题：{user_input}

请结合上述所有信息，直接回答用户。不要输出"FINAL:"等多余字符。"""

            # 记录“大模型生成”这一步
            with root_span.start_as_current_observation(
                name="final_llm_generation",
                as_type="generation",
                model=self.model,
                input=final_prompt,
            ) as generation:
                
                response = self.client.chat.completions.create(
                    model=self.model, 
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=Config.TEMPERATURE, 
                    max_tokens=1000, 
                    stream=True
                )
                
                final_answer = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        final_answer += text
                        yield {"type": "text", "content": text}
                
                generation.update(output=final_answer)
            
            # 存储记忆
            self.db.add_message(session_id, "assistant", final_answer)
            judgment = self.ltm.should_store(user_input, final_answer)
            if judgment.get("should_store"):
                if judgment.get("memory"):
                    self.ltm.add_memory(judgment.get("memory"), session_id)
            
            # 通知前端结束
            yield {"type": "final", "steps": steps_log, "session_id": session_id}
            
            # 设置 Trace 级别的输出
            root_span.update(output=final_answer)
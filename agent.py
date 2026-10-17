import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from vector_db import VectorDB
from tools import Tools
from config import Config
from database import Database
from long_term_memory import LongTermMemory


class Agent:
    """ReAct Agent（支持短期记忆 + 长期记忆 + 多会话 + 遗忘机制 + 查询改写）"""
    
    # 遗忘机制配置
    MAX_MESSAGES_BEFORE_SUMMARY = 20   # 超过20条触发摘要
    KEEP_RECENT_MESSAGES = 10          # 摘要后保留最近10条
    
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

1. get_weather(city) - 查询天气，参数：城市名
2. calculate(expression) - 数学计算，参数：表达式
3. get_time() - 查询当前时间
4. search_knowledge(query) - 检索本地知识库
5. web_search(query) - 搜索互联网实时信息
6. generate_image(prompt) - 生成图像，参数：图像描述（英文效果更好）

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
    
    # ==================== 查询改写 ====================
    
    def _rewrite_query(self, query: str) -> str:
        """用LLM把口语化问题改写成检索关键词"""
        prompt = f"""你是一个搜索专家。请将用户的查询改写成2-4个精准的搜索关键词，用于在知识库中检索。
不要解释，直接输出关键词（用空格隔开）。
如果原查询已经很简短精准，直接原样返回。

用户查询：{query}
改写结果："""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=30
            )
            rewritten = response.choices[0].message.content.strip()
            rewritten = rewritten.replace('"', '').replace("'", "")
            return rewritten
        except Exception as e:
            print(f"⚠️ 查询改写失败: {e}")
            return query
    
    # ==================== 遗忘机制（摘要） ====================
    
    def _maybe_summarize(self, session_id: str):
        message_count = self.db.get_message_count(session_id)
        if message_count <= self.MAX_MESSAGES_BEFORE_SUMMARY:
            return
        
        info = self.db.get_session_summary(session_id)
        until_id = info["summary_until_id"]
        old_summary = info["summary"]
        
        latest_id = self.db.get_latest_message_id(session_id)
        summary_end_id = latest_id - self.KEEP_RECENT_MESSAGES
        
        if summary_end_id <= until_id:
            return
        
        messages_to_summarize = self.db.get_old_messages(session_id, before_id=until_id, limit=50)
        messages_to_summarize = [m for m in messages_to_summarize if m["id"] <= summary_end_id]
        
        if not messages_to_summarize:
            return
        
        old_text = old_summary if old_summary else "（无）"
        new_text_lines = []
        for msg in messages_to_summarize:
            role = "用户" if msg["role"] == "user" else "助手"
            new_text_lines.append(f"{role}: {msg['content']}")
        new_text = "\n".join(new_text_lines)
        
        prompt = f"""你是一个对话摘要助手。请把下面的对话压缩成一段简短的摘要。

【已有摘要】
{old_text}

【新增对话】
{new_text}

要求：
1. 摘要要保留重要信息（用户目标、关键事实、偏好等）
2. 去掉闲聊和无意义内容
3. 总长度不超过200字
4. 直接输出摘要文本，不要任何前缀

输出摘要："""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            new_summary = response.choices[0].message.content.strip()
            self.db.update_session_summary(session_id, new_summary, summary_end_id)
            print(f"📝 会话 {session_id} 已摘要（覆盖到消息ID {summary_end_id}）")
        except Exception as e:
            print(f"⚠️ 摘要失败: {e}")
    
    # ==================== 标题生成 ====================
    
    def _generate_title(self, first_message: str) -> str:
        prompt = f"""请用不超过10个字概括下面这句话的主题，只返回标题，不要任何其他内容：

{first_message}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30
            )
            title = response.choices[0].message.content.strip()
            title = title.replace('"', '').replace("'", '').replace('\n', '')
            return title[:20]
        except Exception as e:
            print(f"⚠️ 生成标题失败: {e}")
            return first_message[:10]
    
    # ==================== 上下文构建 ====================
    
    def _build_ltm_context(self, query: str) -> str:
        memories = self.ltm.search(query, top_k=3)
        if not memories:
            return ""
        
        lines = ["【关于用户的长期记忆】"]
        for mem in memories:
            lines.append(f"- {mem['content']}")
        
        return "\n".join(lines)
    
    def _build_history_context(self, session_id: str, limit: int = 20) -> str:
        info = self.db.get_session_summary(session_id)
        summary = info["summary"]
        history = self.db.get_history(session_id, limit=limit)
        
        parts = []
        if summary:
            parts.append(f"【早期对话摘要】\n{summary}")
        
        if history:
            history_lines = []
            for msg in history:
                role = "用户" if msg["role"] == "user" else "助手"
                history_lines.append(f"{role}: {msg['content']}")
            parts.append("【最近对话】\n" + "\n".join(history_lines))
        
        if not parts:
            return "无历史对话"
        
        return "\n\n".join(parts)
    
    # ==================== 核心逻辑 ====================
    
    def think(self, user_input: str, session_id: str, history: List[Dict]) -> str:
        ltm_context = self._build_ltm_context(user_input)
        history_context = self._build_history_context(session_id)
        
        prompt = f"""你是一个多功能AI助手。

{self.tool_descriptions}

{ltm_context}

【对话上下文】
{history_context}

【本轮执行记录】
{json.dumps(history, ensure_ascii=False) if history else "无"}

【用户新问题】
{user_input}

请决定下一步。注意：
- 结合历史对话理解用户的意图
- 结合长期记忆了解用户背景
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=Config.TEMPERATURE,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    async def run(self, user_input: str, session_id: str = "default") -> Dict:
        is_first_message = self.db.get_message_count(session_id) == 0
        self.db.add_message(session_id, "user", user_input)
        
        if is_first_message:
            title = self._generate_title(user_input)
            self.db.set_session_title(session_id, title)
            print(f"📝 新会话标题: {title}")
        
        self._maybe_summarize(session_id)
        
        history = []
        final_answer = None
        steps_log = []
        
        for step in range(self.max_steps):
            decision = self.think(user_input, session_id, history)
            
            if "FINAL:" in decision:
                final_answer = decision.split("FINAL:")[-1].strip()
                steps_log.append({
                    "step": step + 1,
                    "action": "final",
                    "content": final_answer
                })
                break
            
            elif "TOOL:" in decision:
                tool_line = decision.split("TOOL:")[-1].strip()
                
                if "(" in tool_line and ")" in tool_line:
                    tool_name = tool_line.split("(")[0].strip()
                    args = tool_line.split("(")[1].split(")")[0].strip()
                    
                    # 新增：拦截 search_knowledge 做多路查询
                    if tool_name == "search_knowledge":
                        print(f"🔍 [查询改写] 原问题: {args}")
                        rewritten = self._rewrite_query(args)
                        print(f"🔍 [查询改写] 改写词: {rewritten}")
                        result = await self.tools.execute(tool_name, [args, rewritten])
                    else:
                        result = await self.tools.execute(tool_name, args)
                    
                    history.append({
                        "step": step + 1,
                        "tool": tool_name,
                        "args": args,
                        "result": result
                    })
                    
                    steps_log.append({
                        "step": step + 1,
                        "action": "tool",
                        "tool": tool_name,
                        "args": args,
                        "result": result
                    })
            
            else:
                final_answer = decision
                steps_log.append({
                    "step": step + 1,
                    "action": "final",
                    "content": final_answer
                })
                break
        
        if final_answer is None:
            final_answer = self._summarize(user_input, history)
        
        self.db.add_message(session_id, "assistant", final_answer)
        
        judgment = self.ltm.should_store(user_input, final_answer)
        if judgment.get("should_store"):
            memory_content = judgment.get("memory", "")
            if memory_content:
                self.ltm.add_memory(memory_content, session_id)
        
        return {
            "answer": final_answer,
            "steps": steps_log,
            "history": history,
            "session_id": session_id
        }
    
    def _summarize(self, user_input: str, history: List[Dict]) -> str:
        prompt = f"""用户问题：{user_input}
执行记录：{json.dumps(history, ensure_ascii=False)}
请基于以上信息回答。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=Config.TEMPERATURE,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    # ==================== 多会话管理 ====================
    
    def get_all_sessions(self) -> List[Dict]:
        return self.db.get_all_sessions()
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        return self.db.get_all_messages(session_id)
    
    def delete_session(self, session_id: str):
        self.db.delete_session(session_id)
    
    def clear_session(self, session_id: str):
        self.db.clear_session(session_id)
    
    def load_knowledge(self, filepath: str) -> bool:
        return self.vector_db.load_from_file(filepath)
    
    def get_memory_stats(self) -> Dict:
        return {
            "短期记忆": self.db.get_stats(),
            "长期记忆": self.ltm.get_stats()
        }
    
    def get_all_memories(self) -> List[Dict]:
        return self.ltm.get_all()
    
    def clear_long_term_memory(self):
        self.ltm.clear()
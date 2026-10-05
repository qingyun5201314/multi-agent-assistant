import json
from typing import List, Dict, Any
from openai import OpenAI
from vector_db import VectorDB
from tools import Tools
from config import Config


class Agent:
    """ReAct Agent"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)
        self.model = Config.MODEL
        self.vector_db = VectorDB()
        self.tools = Tools(self.vector_db)
        self.max_steps = Config.MAX_STEPS
        
        self.tool_descriptions = """
可用工具：

1. get_weather(city) - 查询天气，参数：城市名
   例：get_weather("北京")

2. calculate(expression) - 数学计算，参数：表达式
   例：calculate("123 * 456")

3. get_time() - 查询当前时间，无参数
   例：get_time()

4. search_knowledge(query) - 检索本地知识库，参数：搜索关键词
   例：search_knowledge("RAG")

5. web_search(query) - 搜索互联网实时信息，参数：搜索关键词
   例：web_search("2026年AI最新进展")

6. generate_image(prompt) - 生成图像，参数：图像描述（英文效果更好）
   例：generate_image("a cyberpunk cat, neon lights, detailed")

工具选择规则（重要）：
- 概念解释、定义、专业术语 → 优先用 search_knowledge
- 知识库没有的答案，用 web_search
- 实时信息、新闻、最新动态 → 用 web_search
- 数学计算 → 用 calculate
- 时间日期 → 用 get_time
- 画图、生成图片、设计图像、视觉创作 → 用 generate_image
- 不要在有本地知识的情况下优先搜网

使用格式：TOOL: 工具名(参数)
直接回答：FINAL: 你的回答
"""
    
    def think(self, user_input: str, history: List[Dict]) -> str:
        """让LLM决定下一步"""
        prompt = f"""你是一个多功能AI助手。

{self.tool_descriptions}

用户问题：{user_input}
历史记录：{json.dumps(history, ensure_ascii=False) if history else "无"}

请决定下一步。"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=Config.TEMPERATURE,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    async def run(self, user_input: str) -> Dict:
        """运行Agent（异步版）"""
        history = []
        final_answer = None
        steps_log = []
        
        for step in range(self.max_steps):
            decision = self.think(user_input, history)
            
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
        
        return {
            "answer": final_answer,
            "steps": steps_log,
            "history": history
        }
    
    def _summarize(self, user_input: str, history: List[Dict]) -> str:
        """总结所有工具结果，生成最终回答"""
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
    
    def load_knowledge(self, filepath: str) -> bool:
        return self.vector_db.load_from_file(filepath)
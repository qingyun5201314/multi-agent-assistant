import datetime
import httpx
import base64
import json
from typing import Dict, Any, List
from vector_db import VectorDB
from config import Config


class Tools:
    """工具集"""
    
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
    
    # ==================== 天气 ====================
    
    async def get_weather(self, city: str) -> Dict:
        """查询真实天气（Open-Meteo API）"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "zh", "format": "json"}
                )
                geo_data = geo_resp.json()
                
                if not geo_data.get("results"):
                    return {"error": f"未找到城市: {city}"}
                
                location = geo_data["results"][0]
                
                weather_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": location["latitude"],
                        "longitude": location["longitude"],
                        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                        "timezone": "auto"
                    }
                )
                current = weather_resp.json().get("current", {})
                
                return {
                    "city": location.get("name", city),
                    "country": location.get("country", ""),
                    "temperature": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "condition": self._weather_code_to_text(current.get("weather_code", 0)),
                    "wind_speed": current.get("wind_speed_10m"),
                    "unit": "°C"
                }
        except httpx.TimeoutException:
            return {"error": "请求超时"}
        except Exception as e:
            return {"error": f"查询失败: {str(e)}"}
    
    def _weather_code_to_text(self, code: int) -> str:
        weather_map = {
            0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
            45: "雾", 48: "霜雾",
            51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            66: "冻雨", 67: "强冻雨",
            71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
            80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
            85: "小阵雪", 86: "大阵雪",
            95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹"
        }
        return weather_map.get(code, "未知")
    
    # ==================== 计算 ====================
    
    def calculate(self, expression: str) -> Dict:
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return {"error": "表达式包含非法字符"}
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}
    
    # ==================== 时间 ====================
    
    def get_time(self) -> Dict:
        now = datetime.datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A")
        }
    
    # ==================== 知识库 ====================
    
    def search_knowledge(self, query: str) -> Dict:
        results = self.vector_db.search(query, top_k=Config.TOP_K)
        if not results:
            return {"found": False, "message": "知识库为空或未找到相关文档"}
        return {"found": True, "count": len(results), "results": results}
    
    # ==================== 网页搜索（Tavily API） ====================
    
    async def web_search(self, query: str) -> Dict:
        """网页搜索（Tavily API，专为AI设计）"""
        try:
            print(f"🔍 [调试] 开始搜索: {query}")
            
            if not Config.TAVILY_API_KEY:
                return {"error": "未配置 TAVILY_API_KEY"}
            
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": Config.TAVILY_API_KEY,
                        "query": query,
                        "search_depth": "basic",  # basic 省额度，advanced 更慢但更准
                        "max_results": 5,
                        "include_answer": False,   # 不需要Tavily生成的答案
                        "include_raw_content": False
                    }
                )
                
                print(f"🔍 [调试] Tavily返回状态: {resp.status_code}")
                
                if resp.status_code != 200:
                    return {"error": f"Tavily返回 {resp.status_code}: {resp.text[:200]}"}
                
                data = resp.json()
                results = data.get("results", [])
                
                print(f"🔍 [调试] Tavily返回 {len(results)} 条结果")
                
                return {
                    "query": query,
                    "count": len(results),
                    "results": [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", "")[:300]
                        }
                        for r in results[:5]
                    ]
                }
        except httpx.TimeoutException:
            return {"error": "搜索请求超时"}
        except Exception as e:
            print(f"❌ [调试] web_search异常: {type(e).__name__}: {e}")
            return {"error": f"搜索失败: {str(e)}"}
    
    # ==================== 图像生成 ====================
    
    async def generate_image(self, prompt: str) -> Dict:
        """生成图像（ModelScope）"""
        try:
            print(f"🎨 [调试] 开始生成图像: {prompt}")
            
            if not Config.MODELSCOPE_TOKEN:
                return {"success": False, "error": "未配置 MODELSCOPE_TOKEN"}
            
            headers = {
                "Authorization": f"Bearer {Config.MODELSCOPE_TOKEN}",
                "Content-Type": "application/json",
                "X-ModelScope-Async-Mode": "true"
            }
            
            async with httpx.AsyncClient(timeout=90) as client:
                # Step 1: 创建任务
                create_resp = await client.post(
                    "https://api-inference.modelscope.cn/v1/images/generations",
                    headers=headers,
                    json={
                        "model": "Qwen/Qwen-Image",
                        "prompt": prompt,
                        "n": 1,
                        "size": "1024x1024"
                    }
                )
                
                if create_resp.status_code != 200:
                    return {"success": False, "error": f"创建任务失败: {create_resp.text[:200]}"}
                
                create_data = create_resp.json()
                task_id = create_data.get("task_id")
                
                if not task_id:
                    return {"success": False, "error": f"未获取到task_id: {create_data}"}
                
                print(f"🎨 [调试] 任务已创建: {task_id}")
                
                # Step 2: 轮询
                query_headers = {
                    "Authorization": f"Bearer {Config.MODELSCOPE_TOKEN}",
                    "X-ModelScope-Task-Type": "image_generation"
                }
                
                max_wait = 60
                poll_interval = 3
                elapsed = 0
                
                while elapsed < max_wait:
                    import asyncio
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    
                    query_resp = await client.get(
                        f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                        headers=query_headers
                    )
                    
                    if query_resp.status_code != 200:
                        continue
                    
                    query_data = query_resp.json()
                    status = query_data.get("task_status")
                    
                    print(f"🎨 [调试] 任务状态: {status} (已等待{elapsed}秒)")
                    
                    if status == "SUCCEED":
                        output_images = query_data.get("output_images", [])
                        if output_images:
                            image_url = output_images[0]
                            
                            # 下载图片转base64
                            img_resp = await client.get(image_url, timeout=30)
                            if img_resp.status_code == 200:
                                img_base64 = base64.b64encode(img_resp.content).decode('utf-8')
                                return {
                                    "success": True,
                                    "prompt": prompt,
                                    "image_url": f"data:image/png;base64,{img_base64}",
                                    "note": "图像已生成"
                                }
                            else:
                                return {
                                    "success": True,
                                    "prompt": prompt,
                                    "image_url": image_url,
                                    "note": "图像已生成"
                                }
                        else:
                            return {"success": False, "error": f"任务成功但无图片: {query_data}"}
                    
                    elif status == "FAILED":
                        return {"success": False, "error": f"任务失败: {query_data}"}
                
                return {"success": False, "error": "生成超时（60秒）"}
        
        except httpx.TimeoutException:
            return {"success": False, "error": "请求超时"}
        except Exception as e:
            print(f"❌ [调试] generate_image异常: {type(e).__name__}: {e}")
            return {"success": False, "error": f"生成失败: {str(e)}"}
    
    # ==================== 统一执行入口 ====================
    
    async def execute(self, tool_name: str, args: str) -> Any:
        """统一执行入口"""
        try:
            args = args.strip().strip('"').strip("'")
            
            if tool_name == "get_weather":
                return await self.get_weather(args)
            elif tool_name == "calculate":
                return self.calculate(args)
            elif tool_name == "get_time":
                return self.get_time()
            elif tool_name == "search_knowledge":
                return self.search_knowledge(args)
            elif tool_name == "web_search":
                return await self.web_search(args)
            elif tool_name == "generate_image":
                return await self.generate_image(args)
            else:
                return {"error": f"未知工具: {tool_name}"}
        except Exception as e:
            return {"error": f"工具执行失败: {str(e)}"}
import datetime
import requests
import base64
import time
from typing import Dict, Any
from vector_db import VectorDB
from config import Config

# 尝试导入 webless
try:
    from webless import search
    HAS_WEBLESS = True
    print("✅ webless 已加载，网页搜索可用")
except ImportError:
    HAS_WEBLESS = False
    print("⚠️ 未安装 webless，web_search 将使用模拟数据")


class Tools:
    """工具集"""
    
    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
    
    def get_weather(self, city: str) -> Dict:
        """查询真实天气（Open-Meteo API）"""
        try:
            geo_resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
                timeout=10
            )
            geo_data = geo_resp.json()
            
            if not geo_data.get("results"):
                return {"error": f"未找到城市: {city}"}
            
            location = geo_data["results"][0]
            
            weather_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto"
                },
                timeout=10
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
        except requests.exceptions.Timeout:
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
    
    def calculate(self, expression: str) -> Dict:
        try:
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return {"error": "表达式包含非法字符"}
            result = eval(expression)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}
    
    def get_time(self) -> Dict:
        now = datetime.datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A")
        }
    
    def search_knowledge(self, query: str) -> Dict:
        results = self.vector_db.search(query, top_k=Config.TOP_K)
        if not results:
            return {"found": False, "message": "知识库为空或未找到相关文档"}
        return {"found": True, "count": len(results), "results": results}
    
    async def web_search(self, query: str) -> Dict:
        """真实网页搜索（webless，异步版）"""
        if not HAS_WEBLESS:
            return {
                "query": query,
                "note": "webless未安装，返回模拟数据",
                "results": [
                    {"title": f"关于'{query}'的结果1", "snippet": "..."},
                    {"title": f"关于'{query}'的结果2", "snippet": "..."}
                ]
            }
        
        try:
            print(f"🔍 [调试] 开始搜索: {query}")
            result = await search(query, limit=5)
            print(f"🔍 [调试] 搜索返回: ok={result.ok}, hits={len(result.hits) if result.ok else 0}")
            
            if not result.ok:
                return {
                    "query": query,
                    "error": "所有搜索引擎请求失败",
                    "failed_engines": result.failed
                }
            
            return {
                "query": query,
                "count": len(result.hits),
                "results": [
                    {"title": hit.title, "url": hit.url, "snippet": hit.snippet}
                    for hit in result.hits
                ],
                "succeeded_engines": result.succeeded
            }
        except Exception as e:
            print(f"❌ [调试] web_search异常: {type(e).__name__}: {e}")
            return {"error": f"搜索失败: {str(e)}"}
    
    async def generate_image(self, prompt: str) -> Dict:
        """生成图像（ModelScope 魔搭社区，异步任务 + 轮询）"""
        try:
            print(f"🎨 [调试] 开始生成图像: {prompt}")
            
            if not Config.MODELSCOPE_TOKEN:
                return {"success": False, "error": "未配置 MODELSCOPE_TOKEN"}
            
            headers = {
                "Authorization": f"Bearer {Config.MODELSCOPE_TOKEN}",
                "Content-Type": "application/json",
                "X-ModelScope-Async-Mode": "true"  # ← 关键：开启异步模式
            }
            
            # Step 1: 创建生成任务
            create_resp = requests.post(
                "https://api-inference.modelscope.cn/v1/images/generations",
                headers=headers,
                json={
                    "model": "Qwen/Qwen-Image",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024"
                },
                timeout=30
            )
            
            print(f"🎨 [调试] 创建任务返回状态: {create_resp.status_code}")
            
            if create_resp.status_code != 200:
                return {"success": False, "error": f"创建任务失败: {create_resp.text[:200]}"}
            
            create_data = create_resp.json()
            task_id = create_data.get("task_id")
            
            if not task_id:
                return {"success": False, "error": f"未获取到task_id: {create_data}"}
            
            print(f"🎨 [调试] 任务已创建: {task_id}")
            
            # Step 2: 轮询查询任务结果（最多等60秒）
            query_headers = {
                "Authorization": f"Bearer {Config.MODELSCOPE_TOKEN}",
                "X-ModelScope-Task-Type": "image_generation"
            }
            
            max_wait = 60
            poll_interval = 3
            elapsed = 0
            
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval
                
                query_resp = requests.get(
                    f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                    headers=query_headers,
                    timeout=15
                )
                
                if query_resp.status_code != 200:
                    print(f"⚠️ [调试] 查询失败: {query_resp.status_code}")
                    continue
                
                query_data = query_resp.json()
                status = query_data.get("task_status")
                
                print(f"🎨 [调试] 任务状态: {status} (已等待{elapsed}秒)")
                
                if status == "SUCCEED":
                    # 提取图片URL
                    output_images = query_data.get("output_images", [])
                    if output_images:
                        image_url = output_images[0]
                        print(f"🎨 [调试] 图像URL: {image_url}")
                        
                        # 下载图片转base64（避免前端加载外链超时）
                        img_resp = requests.get(image_url, timeout=30)
                        if img_resp.status_code == 200:
                            img_base64 = base64.b64encode(img_resp.content).decode('utf-8')
                            return {
                                "success": True,
                                "prompt": prompt,
                                "image_url": f"data:image/png;base64,{img_base64}",
                                "note": "图像已生成"
                            }
                        else:
                            # 下载失败就直接返回外链
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
                
                # PROCESSING / PENDING → 继续轮询
            
            return {"success": False, "error": "生成超时（60秒），请重试"}
        
        except requests.exceptions.Timeout:
            return {"success": False, "error": "请求超时，请重试"}
        except Exception as e:
            print(f"❌ [调试] generate_image异常: {type(e).__name__}: {e}")
            return {"success": False, "error": f"生成失败: {str(e)}"}
    
    async def execute(self, tool_name: str, args: str) -> Any:
        """统一执行入口（异步）"""
        try:
            args = args.strip().strip('"').strip("'")
            
            if tool_name == "get_weather":
                return self.get_weather(args)
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
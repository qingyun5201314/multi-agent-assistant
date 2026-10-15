import sqlite3
import os
from typing import List, Dict, Optional
from datetime import datetime


class Database:
    """SQLite 数据库管理（短期记忆持久化 + 摘要）"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 会话表（含 summary 字段）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '新对话',
                summary TEXT DEFAULT '',
                summary_until_id INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 消息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # 索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== 会话管理 ====================
    
    def ensure_session(self, session_id: str):
        """确保会话存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO sessions (session_id) VALUES (?)
        """, (session_id,))
        
        cursor.execute("""
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def set_session_title(self, session_id: str, title: str):
        """设置会话标题"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (title, session_id))
        
        conn.commit()
        conn.close()
    
    def get_session_summary(self, session_id: str) -> Dict:
        """获取会话摘要状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT summary, summary_until_id FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {"summary": "", "summary_until_id": 0}
        
        return {
            "summary": row[0] or "",
            "summary_until_id": row[1] or 0
        }
    
    def update_session_summary(self, session_id: str, summary: str, until_id: int):
        """更新会话摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET summary = ?, summary_until_id = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (summary, until_id, session_id))
        
        conn.commit()
        conn.close()
    
    def get_all_sessions(self) -> List[Dict]:
        """获取所有会话（按更新时间倒序）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages WHERE session_id = s.session_id) as msg_count
            FROM sessions s
            ORDER BY s.updated_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4]
            }
            for row in rows
        ]
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取单个会话信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, title, created_at, updated_at 
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3]
        }
    
    def get_message_count(self, session_id: str) -> int:
        """获取会话的消息数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM messages WHERE session_id = ?
        """, (session_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    # ==================== 消息管理 ====================
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        self.ensure_session(session_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (session_id, role, content) 
            VALUES (?, ?, ?)
        """, (session_id, role, content))
        
        cursor.execute("""
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """获取会话历史（只返回摘要后的消息）"""
        info = self.get_session_summary(session_id)
        until_id = info["summary_until_id"]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, content FROM messages 
            WHERE session_id = ? AND id > ?
            ORDER BY id DESC 
            LIMIT ?
        """, (session_id, until_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]
    
    def get_old_messages(self, session_id: str, before_id: int, limit: int = 10) -> List[Dict]:
        """获取需要被摘要的旧消息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, role, content FROM messages 
            WHERE session_id = ? AND id > ? 
            ORDER BY id ASC 
            LIMIT ?
        """, (session_id, before_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {"id": row[0], "role": row[1], "content": row[2]}
            for row in rows
        ]
    
    def get_latest_message_id(self, session_id: str) -> int:
        """获取最新消息的ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MAX(id) FROM messages WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        return result or 0
    
    def get_all_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """
        获取会话的最近 N 条消息（前端用）
        只返回最近 limit 条（默认50），按时间顺序从旧到新
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 先按 id 倒序取最近 limit 条
        cursor.execute("""
            SELECT role, content, created_at FROM messages 
            WHERE session_id = ? 
            ORDER BY id DESC 
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 反转回时间顺序（旧→新）
        rows.reverse()
        
        return [
            {"role": row[0], "content": row[1], "created_at": row[2]}
            for row in rows
        ]
    
    def clear_session(self, session_id: str):
        """清空会话（消息+摘要+标题）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("""
            UPDATE sessions 
            SET title = '新对话', summary = '', summary_until_id = 0,
                updated_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def delete_session(self, session_id: str):
        """删除会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        message_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "会话数量": session_count,
            "消息总数": message_count
        }
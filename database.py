# Simple database setup using SQLite
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import threading

class DatabaseManager:
    def __init__(self, db_path: str = "chat_database.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    rate_limit INTEGER DEFAULT 100,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    api_key TEXT PRIMARY KEY,
                    requests_count INTEGER DEFAULT 0,
                    reset_time TIMESTAMP,
                    FOREIGN KEY (api_key) REFERENCES api_keys (api_key)
                )
            """)
            
            conn.commit()
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str], rate_limit: int = 100) -> str:
        """Create a new API key"""
        api_key_id = str(uuid.uuid4())
        api_key = f"sk-{uuid.uuid4().hex}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_keys (id, api_key, user_id, name, permissions, rate_limit)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (api_key_id, api_key, user_id, name, json.dumps(permissions), rate_limit))
            conn.commit()
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return user info"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM api_keys WHERE api_key = ? AND is_active = 1
            """, (api_key,))
            row = cursor.fetchone()
            
            if row:
                # Update last used
                conn.execute("""
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE api_key = ?
                """, (api_key,))
                conn.commit()
                
                return {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "permissions": json.loads(row["permissions"]),
                    "rate_limit": row["rate_limit"]
                }
        return None
    
    def create_conversation(self, user_id: str, title: str = "New Chat") -> str:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title)
                VALUES (?, ?, ?)
            """, (conversation_id, user_id, title))
            conn.commit()
        
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to a conversation"""
        message_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (id, conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (message_id, conversation_id, role, content, json.dumps(metadata or {})))
            
            # Update conversation last activity
            conn.execute("""
                UPDATE conversations SET last_activity = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
    
    def get_conversation_messages(self, conversation_id: str, user_id: str) -> List[Dict]:
        """Get all messages in a conversation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT m.* FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.id = ? AND c.user_id = ?
                ORDER BY m.timestamp
            """, (conversation_id, user_id))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": json.loads(row["metadata"] or "{}")
                })
            
            return messages
    
    def get_user_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT c.*, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = ? AND c.is_active = 1
                GROUP BY c.id
                ORDER BY c.last_activity DESC
            """, (user_id,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    "id": row["id"],
                    "title": row["title"],
                    "message_count": row["message_count"],
                    "created_at": row["created_at"],
                    "last_activity": row["last_activity"]
                })
            
            return conversations

# Global database instance
db = DatabaseManager()

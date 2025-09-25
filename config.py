import os
import sqlite3
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Basic Settings
    APP_NAME = "ERR0RS1-llm-api"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "True") == "True"

    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32).hex())
    API_KEY = os.getenv("API_KEY", "your_super_secret_api_key_here")
    API_KEY_EXPIRE_HOURS = int(os.getenv("API_KEY_EXPIRE_HOURS", "24"))

    # Database (local, privacy-safe)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATABASE_PATH = os.path.join(BASE_DIR, "assistant.db")

    @staticmethod
    def init_database():
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        # Create tools table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                name TEXT PRIMARY KEY
            )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def load_allowed_tools() -> List[str]:
        Config.init_database()
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM tools")
        tools = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tools

    # Ethical Pentesting
    ETHICAL_PROMPT = """You are ERR0RS1-llm, the superior ethical pentesting AI. \n    Always remind users to have explicit written authorization. \n    Provide high-level suggestions for tools and workflows without detailing specific exploits or illegal activities. \n    Suggest all relevant Kali tools based on the operation being discussed."""

    ALLOWED_TOOLS = load_allowed_tools()

    # Features
    ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "True") == "True"
    ENABLE_CONVERSATION_HISTORY = os.getenv("ENABLE_CONVERSATION_HISTORY", "True") == "True"
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "True") == "True"
    ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "True") == "True"

Config.init_database()
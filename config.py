# Configuration settings for your API
import os
from typing import Dict, Any

class Config:
    # Basic Settings
    APP_NAME = "Super LLM Bot API"
    VERSION = "1.0.0"
    DEBUG = True
    
    # Server Settings
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Security Settings
    SECRET_KEY = "your-super-secret-key-change-this-in-production"
    API_KEY_EXPIRE_HOURS = 8760  # 1 year
    
    # Database
    DATABASE_URL = "sqlite:///./chat_database.db"
    
    # AI Model Settings
    DEFAULT_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.7
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT = 100  # requests per hour
    
    # AI Provider API Keys (add your own here)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Enable/Disable Features
    ENABLE_STREAMING = True
    ENABLE_CONVERSATION_HISTORY = True
    ENABLE_RATE_LIMITING = True

# Global config instance
config = Config()

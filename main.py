# Main API application
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
import asyncio
import json
import time
import uuid
from datetime import datetime
import logging
import aiofiles
import openai
import anthropic
from config import config
from database import db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(default=config.DEFAULT_MODEL, description="AI model to use")
    temperature: Optional[float] = Field(default=config.DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=config.MAX_TOKENS, ge=1, le=8192)
    stream: Optional[bool] = Field(default=False, description="Stream response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class APIKeyRequest(BaseModel):
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(default=["chat"], description="Permissions")
    rate_limit: Optional[int] = Field(default=config.DEFAULT_RATE_LIMIT)

# ==================== AI PROVIDERS ====================

class AIProvider:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize OpenAI if API key is provided
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize Anthropic if API key is provided
        if config.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    
    async def generate_response(self, request: ChatRequest) -> Dict[str, Any]:
        """Generate AI response based on model type"""
        
        if request.model.startswith("gpt") and self.openai_client:
            return await self._generate_openai_response(request)
        elif request.model.startswith("claude") and self.anthropic_client:
            return await self._generate_anthropic_response(request)
        else:
            # Fallback to mock response if no API keys
            return await self._generate_mock_response(request)
    
    async def _generate_openai_response(self, request: ChatRequest) -> Dict[str, Any]:
        """Generate OpenAI response"""
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model": request.model,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return await self._generate_mock_response(request)
    
    async def _generate_anthropic_response(self, request: ChatRequest) -> Dict[str, Any]:
        """Generate Anthropic response"""
        try:
            # Convert messages format for Anthropic
            system_message = ""
            messages = []
            
            for msg in request.messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    messages.append({"role": msg.role, "content": msg.content})
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=request.model,
                system=system_message,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return {
                "content": response.content[0].text,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "model": request.model,
                "finish_reason": "stop"
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            return await self._generate_mock_response(request)
    
    async def _generate_mock_response(self, request: ChatRequest) -> Dict[str, Any]:
        """Generate mock response for testing"""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        user_message = request.messages[-1].content if request.messages else "Hello"
        
        responses = [
            f"I understand you're asking about: {user_message}. This is a mock response since no AI API keys are configured.",
            f"That's an interesting question about {user_message}. I'm a demo bot without real AI integration yet.",
            f"Thanks for your message: {user_message}. To get real AI responses, please add your OpenAI or Anthropic API key to the config.",
            f"I see you mentioned: {user_message}. I'm currently running in demo mode. Add your AI API keys for real responses!",
            f"Regarding {user_message} - I'm a test bot! Configure your AI provider API keys to chat with real AI models."
        ]
        
        import random
        content = random.choice(responses)
        
        return {
            "content": content,
            "tokens_used": len(content.split()) + len(user_message.split()),
            "model": request.model,
            "finish_reason": "stop"
        }

# ==================== AUTHENTICATION ====================

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key and return user info"""
    api_key = credentials.credentials
    
    user_info = db.validate_api_key(api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # TODO: Add rate limiting check here
    
    return user_info

# ==================== APP SETUP ====================

app = FastAPI(
    title=config.APP_NAME,
    description="A professional LLM bot API with authentication, conversations, and multiple AI providers",
    version=config.VERSION
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI provider
ai_provider = AIProvider()

# ==================== ROUTES ====================

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "name": config.APP_NAME,
        "version": config.VERSION,
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "streaming": config.ENABLE_STREAMING,
            "conversations": config.ENABLE_CONVERSATION_HISTORY,
            "rate_limiting": config.ENABLE_RATE_LIMITING
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Main chat endpoint"""
    
    # Check permissions
    if "chat" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Chat permission required")
    
    try:
        # Handle conversation
        conversation_id = request.conversation_id
        if not conversation_id and config.ENABLE_CONVERSATION_HISTORY:
            # Create new conversation
            conversation_id = db.create_conversation(
                current_user["user_id"], 
                f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        # Add user message to database
        if conversation_id and request.messages:
            user_message = request.messages[-1]
            db.add_message(conversation_id, user_message.role, user_message.content)
        
        # Generate AI response
        ai_response = await ai_provider.generate_response(request)
        
        # Add assistant message to database
        if conversation_id:
            db.add_message(conversation_id, "assistant", ai_response["content"])
        
        # Return response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ai_response["content"]
                },
                "finish_reason": ai_response["finish_reason"]
            }],
            "usage": {
                "prompt_tokens": ai_response["tokens_used"] - len(ai_response["content"].split()),
                "completion_tokens": len(ai_response["content"].split()),
                "total_tokens": ai_response["tokens_used"]
            },
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/conversations")
async def list_conversations(current_user: Dict = Depends(get_current_user)):
    """List user's conversations"""
    return db.get_user_conversations(current_user["user_id"])

@app.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get conversation messages"""
    messages = db.get_conversation_messages(conversation_id, current_user["user_id"])
    return {
        "conversation_id": conversation_id,
        "messages": messages
    }

@app.post("/v1/admin/keys")
async def create_api_key(request: APIKeyRequest):
    """Create new API key (no auth for demo - change in production!)"""
    
    # In production, add authentication here
    user_id = "demo_user"  # Replace with real user system
    
    api_key = db.create_api_key(
        user_id=user_id,
        name=request.name,
        permissions=request.permissions,
        rate_limit=request.rate_limit
    )
    
    return {
        "api_key": api_key,
        "name": request.name,
        "permissions": request.permissions,
        "rate_limit": request.rate_limit,
        "message": "API key created successfully! Save this key - you won't see it again."
    }

@app.get("/test")
async def test_page():
    """Serve test page"""
    try:
        async with aiofiles.open("test_client.html", mode="r") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <body>
        <h1>🤖 LLM API Test</h1>
        <p>Test page not found. Please create test_client.html file.</p>
        <p><a href="/docs">Go to API Documentation</a></p>
        </body>
        </html>
        """)

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = []
    
    if config.OPENAI_API_KEY:
        models.extend([
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ])
    
    if config.ANTHROPIC_API_KEY:
        models.extend([
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ])
    
    if not models:
        models = ["mock-model"]
    
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": "user"} for model in models]
    }

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 LLM Bot API is starting up!")
    
    # Create a demo API key
    demo_key = db.create_api_key(
        user_id="demo_user",
        name="Demo Key",
        permissions=["chat", "admin"],
        rate_limit=1000
    )
    
    logger.info("=" * 50)
    logger.info("🎉 API Started Successfully!")
    logger.info(f"📍 Server running on: http://localhost:{config.PORT}")
    logger.info(f"📖 API Docs: http://localhost:{config.PORT}/docs")
    logger.info(f"🧪 Test Page: http://localhost:{config.PORT}/test")
    logger.info(f"🔑 Demo API Key: {demo_key}")
    logger.info("=" * 50)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, reload=config.DEBUG)

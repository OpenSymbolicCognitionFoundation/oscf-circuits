from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sqlite3
import json
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import OpenAI
import os
from contextlib import asynccontextmanager
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_id: str
    session_id: Optional[str] = None
    extract_memories: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str
    memories_extracted: List[Dict[str, Any]] = []
    memories_used: List[Dict[str, Any]] = []

class Memory(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    memory_type: str = Field(..., description="Type: preference, fact, decision, commitment, etc.")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0)
    is_pinned: bool = Field(default=False)
    source_context: str = Field(default="", description="Original conversation context")
    tags: List[str] = Field(default_factory=list)
    eviction_score: float = Field(default=0.0, description="Score used for memory eviction (lower = more likely to be evicted)")
    is_protected: bool = Field(default=False, description="Protected memories cannot be auto-evicted")

class MemoryManager:
    def __init__(self, db_path: str = "memories.db", 
                 max_memories_per_user: int = 100,
                 eviction_threshold_days: int = 365,
                 min_importance_threshold: float = 0.3):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.max_memories_per_user = max_memories_per_user
        self.eviction_threshold_days = eviction_threshold_days
        self.min_importance_threshold = min_importance_threshold
        self._init_db()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.memory_vectors = {}
        self._load_existing_memories()
        
        # Start background eviction task
        self._start_eviction_scheduler()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    is_pinned BOOLEAN NOT NULL,
                    source_context TEXT,
                    tags TEXT,
                    eviction_score REAL DEFAULT 0.0,
                    is_protected BOOLEAN DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance_score)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eviction_score ON memories(eviction_score)
            """)
            
            # Add new columns to existing tables if they don't exist
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN eviction_score REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN is_protected BOOLEAN DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _load_existing_memories(self):
        """Load existing memories and compute vectors"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, content FROM memories")
            memories = cursor.fetchall()
            
        if memories:
            contents = [memory[1] for memory in memories]
            try:
                vectors = self.vectorizer.fit_transform(contents)
                for i, (memory_id, content) in enumerate(memories):
                    self.memory_vectors[memory_id] = vectors[i]
            except Exception as e:
                logger.warning(f"Error loading memory vectors: {e}")

    async def extract_memories_from_conversation(self, messages: List[ChatMessage], user_id: str) -> List[Memory]:
        """Extract memories from conversation using OpenAI"""
        conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[-5:]])  # Last 5 messages
        
        extraction_prompt = f"""
        Analyze this conversation and extract important information that should be remembered for future interactions.
        Focus on:
        1. User preferences and settings
        2. Important facts about the user
        3. Decisions made
        4. Commitments or promises
        5. Recurring patterns or needs

        Conversation:
        {conversation_text}

        Return a JSON list of memories, each with:
        - content: Brief, clear summary of the memory
        - memory_type: One of [preference, fact, decision, commitment, pattern, context]
        - importance_score: Float 0.0-1.0 based on likely future usefulness
        - tags: List of relevant keywords

        Only extract memories that would be useful for future conversations. Skip pleasantries and temporary information.
        """

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "system", "content": extraction_prompt}],
                response_format={"type": "json_object"}
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            memories = []
            
            for memory_data in extracted_data.get("memories", []):
                memory = Memory(
                    user_id=user_id,
                    content=memory_data.get("content", ""),
                    memory_type=memory_data.get("memory_type", "fact"),
                    importance_score=min(1.0, max(0.0, memory_data.get("importance_score", 0.5))),
                    source_context=conversation_text[-500:],  # Last 500 chars as context
                    tags=memory_data.get("tags", [])
                )
                memories.append(memory)
                
            return memories
            
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []

    async def store_memory(self, memory: Memory) -> bool:
        """Store a memory in the database"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO memories 
                        (id, user_id, content, memory_type, importance_score, created_at, 
                         last_accessed, access_count, is_pinned, source_context, tags,
                         eviction_score, is_protected)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.id, memory.user_id, memory.content, memory.memory_type,
                        memory.importance_score, memory.created_at.isoformat(),
                        memory.last_accessed.isoformat(), memory.access_count,
                        memory.is_pinned, memory.source_context, json.dumps(memory.tags),
                        memory.eviction_score, memory.is_protected
                    ))
                
                # Update vector index
                if hasattr(self.vectorizer, 'vocabulary_') and self.vectorizer.vocabulary_:
                    try:
                        vector = self.vectorizer.transform([memory.content])
                        self.memory_vectors[memory.id] = vector
                    except:
                        # Refit if needed
                        self._refit_vectorizer()
                
                # Check if user is over memory limit and trigger eviction if needed
                cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (memory.user_id,))
                memory_count = cursor.fetchone()[0]
                
                if memory_count > self.max_memories_per_user:
                    logger.info(f"User {memory.user_id} over memory limit ({memory_count}/{self.max_memories_per_user}), triggering eviction")
                    # Run eviction in background
                    asyncio.create_task(self._evict_memories_for_user(memory.user_id))
                
                return True
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                return False

    def _refit_vectorizer(self):
        """Refit the vectorizer with all current memories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id, content FROM memories")
                memories = cursor.fetchall()
            
            if memories:
                contents = [memory[1] for memory in memories]
                self.vectorizer.fit(contents)
                vectors = self.vectorizer.transform(contents)
                
                self.memory_vectors = {}
                for i, (memory_id, _) in enumerate(memories):
                    self.memory_vectors[memory_id] = vectors[i]
        except Exception as e:
            logger.error(f"Error refitting vectorizer: {e}")

    async def retrieve_relevant_memories(self, query: str, user_id: str, limit: int = 8) -> List[Memory]:
        """Retrieve the most relevant memories for a query"""
        if not query.strip():
            return []
            
        try:
            # Get user memories
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, content, memory_type, importance_score, created_at,
                           last_accessed, access_count, is_pinned, source_context, tags,
                           COALESCE(eviction_score, 0.0) as eviction_score,
                           COALESCE(is_protected, 0) as is_protected
                    FROM memories 
                    WHERE user_id = ? 
                    ORDER BY 
                        CASE WHEN is_pinned THEN 1 ELSE 0 END DESC,
                        importance_score DESC,
                        last_accessed DESC
                """, (user_id,))
                rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # Convert to Memory objects
            memories = []
            for row in rows:
                memory = Memory(
                    id=row[0], user_id=row[1], content=row[2], memory_type=row[3],
                    importance_score=row[4], created_at=datetime.fromisoformat(row[5]),
                    last_accessed=datetime.fromisoformat(row[6]), access_count=row[7],
                    is_pinned=bool(row[8]), source_context=row[9] or "",
                    tags=json.loads(row[10]) if row[10] else [],
                    eviction_score=row[11], is_protected=bool(row[12])
                )
                memories.append(memory)
            
            # If we have vectors, use semantic similarity
            if self.memory_vectors and hasattr(self.vectorizer, 'vocabulary_') and self.vectorizer.vocabulary_:
                try:
                    query_vector = self.vectorizer.transform([query])
                    similarities = []
                    
                    for memory in memories:
                        if memory.id in self.memory_vectors:
                            similarity = cosine_similarity(query_vector, self.memory_vectors[memory.id])[0][0]
                            similarities.append((memory, similarity))
                    
                    # Sort by similarity, importance, and recency
                    similarities.sort(key=lambda x: (
                        x[1] * 0.4 +  # Semantic similarity
                        x[0].importance_score * 0.4 +  # Importance
                        (1.0 if x[0].is_pinned else 0.0) * 0.2  # Pinned status
                    ), reverse=True)
                    
                    relevant_memories = [mem for mem, _ in similarities[:limit]]
                    
                    # Update access counts
                    for memory in relevant_memories:
                        await self._update_access_count(memory.id)
                    
                    return relevant_memories
                    
                except Exception as e:
                    logger.warning(f"Error with semantic search, falling back to importance: {e}")
            
            # Fallback to importance and recency
            sorted_memories = sorted(memories, key=lambda m: (
                m.is_pinned, m.importance_score, m.last_accessed
            ), reverse=True)
            
            return sorted_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    async def _update_access_count(self, memory_id: str):
        """Update memory access count and timestamp"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE memories 
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), memory_id))
            except Exception as e:
                logger.error(f"Error updating access count: {e}")

    async def get_memory_context(self, memories: List[Memory]) -> str:
        """Format memories into context for AI"""
        if not memories:
            return ""
        
        context_parts = ["Previous conversation context and user information:"]
        
        for memory in memories:
            context_parts.append(f"- {memory.memory_type.title()}: {memory.content}")
            if memory.tags:
                context_parts.append(f"  Tags: {', '.join(memory.tags)}")
        
        return "\n".join(context_parts)

# Initialize components
memory_manager = MemoryManager()
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI Chat Server with Memory System")
    # Create templates directory if it doesn't exist
    import os
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="AI Chat Server with Memory System",
    description="FastAPI server with OpenAI integration and intelligent memory management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # For demo purposes, we'll create a default user if no auth provided
    if not credentials:
        return "demo_user"
    return credentials.credentials  # In production, validate JWT and return user info

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    """Main chat endpoint with memory integration"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Retrieve relevant memories for context
        if request.messages:
            last_message = request.messages[-1].content
            relevant_memories = await memory_manager.retrieve_relevant_memories(
                last_message, request.user_id
            )
        else:
            relevant_memories = []
        
        # Build context with memories
        memory_context = await memory_manager.get_memory_context(relevant_memories)
        
        # Prepare messages for OpenAI
        openai_messages = []
        
        if memory_context:
            openai_messages.append({
                "role": "system", 
                "content": f"{memory_context}\n\nUse this context to provide more personalized and informed responses."
            })
        
        # Add conversation messages
        for msg in request.messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Get response from OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-nano-2025-08-07",
            messages=openai_messages
        )
        
        ai_response = response.choices[0].message.content
        
        # Extract memories from the conversation if requested
        extracted_memories = []
        if request.extract_memories:
            # Add the AI response to messages for memory extraction
            all_messages = request.messages + [ChatMessage(role="assistant", content=ai_response)]
            new_memories = await memory_manager.extract_memories_from_conversation(
                all_messages, request.user_id
            )
            
            # Store new memories
            for memory in new_memories:
                if await memory_manager.store_memory(memory):
                    extracted_memories.append(memory.dict())
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            memories_extracted=extracted_memories,
            memories_used=[memory.dict() for memory in relevant_memories]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}")
async def get_user_memories(user_id: str, current_user: str = Depends(get_current_user)):
    """Get all memories for a user"""
    try:
        memories = await memory_manager.retrieve_relevant_memories("", user_id, limit=100)
        return [memory.dict() for memory in memories]
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/{memory_id}/pin")
async def pin_memory(memory_id: str, current_user: str = Depends(get_current_user)):
    """Pin a memory to prevent eviction"""
    try:
        with sqlite3.connect(memory_manager.db_path) as conn:
            conn.execute("UPDATE memories SET is_pinned = 1 WHERE id = ?", (memory_id,))
        return {"status": "pinned"}
    except Exception as e:
        logger.error(f"Error pinning memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, current_user: str = Depends(get_current_user)):
    """Delete a specific memory"""
    try:
        with sqlite3.connect(memory_manager.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        
        # Remove from vector index
        if memory_id in memory_manager.memory_vectors:
            del memory_manager.memory_vectors[memory_id]
        
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/{memory_id}/protect")
async def protect_memory(memory_id: str, current_user: str = Depends(get_current_user)):
    """Protect a memory from automatic eviction"""
    try:
        success = await memory_manager.protect_memory(memory_id)
        if success:
            return {"status": "protected"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        logger.error(f"Error protecting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}/protect")
async def unprotect_memory(memory_id: str, current_user: str = Depends(get_current_user)):
    """Remove protection from a memory"""
    try:
        success = await memory_manager.unprotect_memory(memory_id)
        if success:
            return {"status": "unprotected"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        logger.error(f"Error unprotecting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}/stats")
async def get_memory_stats(user_id: str, current_user: str = Depends(get_current_user)):
    """Get memory statistics and eviction info for a user"""
    try:
        stats = await memory_manager.get_eviction_stats(user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memories/{user_id}/evict")
async def manual_eviction(user_id: str, current_user: str = Depends(get_current_user)):
    """Manually trigger memory eviction for a user"""
    try:
        await memory_manager._evict_memories_for_user(user_id)
        return {"status": "eviction_completed"}
    except Exception as e:
        logger.error(f"Error during manual eviction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)
    
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Write the chat template file if it doesn't exist
    template_path = "templates/chat.html"
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            # We'll copy the HTML content from our separate artifact
            f.write("""<!-- This will be populated by the HTML artifact content -->
{{ request }}
<!-- Template served by FastAPI -->""")
    
    logger.info("Starting server at http://localhost:8000")
    logger.info("Chat interface available at http://localhost:8000/")
    logger.info("API documentation at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
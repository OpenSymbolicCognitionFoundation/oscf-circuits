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
                    logger.info(f"User {memory.user_id} over memory limit ({memory_count}/{self.max_memories_per_user}), triggering immediate eviction")
                    # Run eviction immediately in background
                    def run_eviction():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self._evict_memories_for_user(memory.user_id))
                            loop.close()
                        except Exception as e:
                            logger.error(f"Error in immediate eviction: {e}")
                    
                    eviction_thread = threading.Thread(target=run_eviction, daemon=True, name=f"ImmediateEviction-{memory.user_id}")
                    eviction_thread.start()
                
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
        # allow empty query → return top by pin/importance/recency
        if not query.strip():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, content, memory_type, importance_score, created_at,
                           last_accessed, access_count, is_pinned, source_context, tags,
                           COALESCE(eviction_score, 0.0), COALESCE(is_protected, 0)
                    FROM memories WHERE user_id = ?
                """, (user_id,))
                rows = cursor.fetchall()

            memories = []
            for row in rows:
                memories.append(Memory(
                    id=row[0], user_id=row[1], content=row[2], memory_type=row[3],
                    importance_score=row[4], created_at=datetime.fromisoformat(row[5]),
                    last_accessed=datetime.fromisoformat(row[6]), access_count=row[7],
                    is_pinned=bool(row[8]), source_context=row[9] or "",
                    tags=json.loads(row[10]) if row[10] else [],
                    eviction_score=row[11], is_protected=bool(row[12])
                ))
            sorted_memories = sorted(memories, key=lambda m: (m.is_pinned, m.importance_score, m.last_accessed), reverse=True)
            return sorted_memories[:limit]
            
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

    # === Eviction / Scheduler / Protection / Stats ===========================
    def _start_eviction_scheduler(self):
        """
        Spawn a daemon background thread that periodically runs the eviction cycle.
        Uses its own asyncio loop to safely await async methods from a thread.
        """
        # avoid double-start
        if getattr(self, "_eviction_thread", None) and self._eviction_thread.is_alive():
            logger.info("Eviction scheduler already running")
            return

        # config with env overrides
        self._eviction_interval_seconds = int(os.getenv("EVICTION_INTERVAL_SECONDS", "900"))  # default 15 min
        self._stop_eviction = threading.Event()

        def _worker():
            thread_name = "MemoryEvictionWorker"
            try:
                threading.current_thread().name = thread_name
            except Exception:
                pass

            logger.info(
                "Starting %s (interval=%ss, max=%d, thresh_days=%d, min_importance=%.2f)",
                thread_name,
                self._eviction_interval_seconds,
                self.max_memories_per_user,
                self.eviction_threshold_days,
                self.min_importance_threshold,
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                while not self._stop_eviction.is_set():
                    try:
                        loop.run_until_complete(self._run_eviction_cycle())
                    except Exception as e:
                        logger.error("Eviction cycle failed: %s", e)

                    # sleep in 1s ticks so we can stop early if needed
                    slept = 0
                    while slept < self._eviction_interval_seconds and not self._stop_eviction.is_set():
                        time.sleep(1)
                        slept += 1
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                logger.info("%s stopped", thread_name)

        self._eviction_thread = threading.Thread(
            target=_worker, daemon=True, name="MemoryEvictionWorker"
        )
        self._eviction_thread.start()

    async def _run_eviction_cycle(self):
        """
        One full eviction pass across all users. Computes/update eviction scores,
        drops stale/low-importance memories, then trims to max per user.
        """
        started = time.time()
        with sqlite3.connect(self.db_path) as conn:
            users = [row[0] for row in conn.execute("SELECT DISTINCT user_id FROM memories")]
        if not users:
            return

        total_deleted = 0
        for uid in users:
            try:
                deleted = await self._evict_memories_for_user(uid)
                total_deleted += deleted
            except Exception as e:
                logger.error("Eviction error for user %s: %s", uid, e)

        logger.info("Eviction cycle done for %d users in %.2fs (deleted=%d)",
                    len(users), time.time() - started, total_deleted)

    def _compute_eviction_score(
        self,
        *,
        importance: float,
        last_accessed: datetime,
        access_count: int,
        is_pinned: bool,
        is_protected: bool,
        created_at: datetime,
    ) -> float:
        """
        Returns a score in [0,1], LOWER = more eligible for eviction.
        We compute a "value score" then invert to an eviction score.

        value ~= 0.5*importance + 0.25*recency + 0.15*popularity + 0.10*(pinned/protected)
        eviction = 1 - value
        """
        if is_protected:
            return 1.0  # effectively never evict

        now = datetime.now()
        idle_days = max(0.0, (now - last_accessed).total_seconds() / 86400.0)
        age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)

        # recency: 1.0 when accessed today; decays toward 0 as idle_days approaches threshold
        denom_days = max(1.0, float(self.eviction_threshold_days))
        recency = max(0.0, 1.0 - min(1.0, idle_days / denom_days))

        # popularity: log-normalized access count (cap to ~50)
        denom_pop = float(np.log(50.0)) if 50.0 > 1.0 else 1.0
        popularity = float(min(1.0, (np.log1p(access_count) / denom_pop)))

        pinned_bonus = 0.10 if is_pinned else 0.0

        value = (0.50 * float(importance)) + (0.25 * recency) + (0.15 * popularity) + pinned_bonus
        value = max(0.0, min(1.0, value))

        eviction = 1.0 - value
        return float(max(0.0, min(1.0, eviction)))

    async def _evict_memories_for_user(self, user_id: str) -> int:
        """
        Update eviction scores for a user's memories, hard-evict stale/low-importance items,
        then, if still above limit, evict by ascending eviction_score.
        Returns number of rows deleted.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, content, memory_type, importance_score, created_at, last_accessed,
                       access_count, is_pinned, COALESCE(eviction_score,0.0), COALESCE(is_protected,0)
                FROM memories
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()

        if not rows:
            return 0

        now = datetime.now()
        # compute new scores + stale flags
        computed = []
        for r in rows:
            mid = r[0]
            imp = float(r[3])
            created = datetime.fromisoformat(r[4])
            last_acc = datetime.fromisoformat(r[5])
            acc = int(r[6])
            pinned = bool(r[7])
            protected = bool(r[9])

            score = self._compute_eviction_score(
                importance=imp,
                last_accessed=last_acc,
                access_count=acc,
                is_pinned=pinned,
                is_protected=protected,
                created_at=created,
            )
            idle_days = (now - last_acc).days
            stale_and_low = (
                (idle_days >= self.eviction_threshold_days)
                and (imp < self.min_importance_threshold)
                and (not pinned)
                and (not protected)
            )
            computed.append((mid, score, pinned, protected, stale_and_low))

        # persist updated scores
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                for mid, score, *_ in computed:
                    conn.execute("UPDATE memories SET eviction_score = ? WHERE id = ?", (score, mid))

        # phase 1: evict stale/low-importance
        to_delete_ids = [mid for (mid, _s, pin, prot, stale) in computed if stale and not pin and not prot]
        deleted = 0
        if to_delete_ids:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    for mid in to_delete_ids:
                        cur = conn.execute(
                            "DELETE FROM memories WHERE id = ? AND is_pinned = 0 AND COALESCE(is_protected,0) = 0",
                            (mid,),
                        )
                        if getattr(cur, "rowcount", -1) != -1:
                            deleted += max(0, cur.rowcount)
                        self.memory_vectors.pop(mid, None)

        # phase 2: trim down to limit if still over
        # Check if user is over memory limit and trigger eviction if needed
        with sqlite3.connect(self.db_path) as c2:
            memory_count = c2.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (memory.user_id,)
            ).fetchone()[0]


        overflow = max(0, count - self.max_memories_per_user)
        if overflow > 0:
            # choose lowest scores among non-pinned, non-protected and not already removed
            remaining = [
                (mid, score)
                for (mid, score, pin, prot, _stale) in computed
                if (mid not in to_delete_ids) and (not pin) and (not prot)
            ]
            remaining.sort(key=lambda x: x[1])  # ascending = easiest to evict first
            evict_more = [mid for (mid, _s) in remaining[:overflow]]

            if evict_more:
                with self.lock:
                    with sqlite3.connect(self.db_path) as conn:
                        for mid in evict_more:
                            cur = conn.execute(
                                "DELETE FROM memories WHERE id = ? AND user_id = ? AND is_pinned = 0 AND COALESCE(is_protected,0) = 0",
                                (mid, user_id),
                            )
                            if getattr(cur, "rowcount", -1) != -1:
                                deleted += max(0, cur.rowcount)
                            self.memory_vectors.pop(mid, None)

        if deleted:
            logger.info("Evicted %d memories for user %s", deleted, user_id)
        return deleted

    async def protect_memory(self, memory_id: str) -> bool:
        """Mark a memory as protected (immune to auto-eviction)."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "UPDATE memories SET is_protected = 1 WHERE id = ?", (memory_id,)
                )
                return bool(getattr(cur, "rowcount", 0))

    async def unprotect_memory(self, memory_id: str) -> bool:
        """Remove protection flag from a memory."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "UPDATE memories SET is_protected = 0 WHERE id = ?", (memory_id,)
                )
                return bool(getattr(cur, "rowcount", 0))

    async def get_eviction_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Return counts and a preview of top eviction candidates for transparency/UX.
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            pinned = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ? AND is_pinned = 1", (user_id,)
            ).fetchone()[0]
            protected = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ? AND COALESCE(is_protected,0) = 1",
                (user_id,),
            ).fetchone()[0]

            # fetch candidates with current scores
            rows = conn.execute(
                """
                SELECT id, content, importance_score, created_at, last_accessed, access_count,
                       is_pinned, COALESCE(eviction_score,0.0), COALESCE(is_protected,0)
                FROM memories
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchall()

        # recompute scores in-memory for a live view
        candidates = []
        for r in rows:
            mid = r[0]
            content = r[1]
            imp = float(r[2])
            created = datetime.fromisoformat(r[3])
            last_acc = datetime.fromisoformat(r[4])
            acc = int(r[5])
            pin = bool(r[6])
            prot = bool(r[8])

            score = self._compute_eviction_score(
                importance=imp,
                last_accessed=last_acc,
                access_count=acc,
                is_pinned=pin,
                is_protected=prot,
                created_at=created,
            )
            candidates.append((mid, content, score, pin, prot, last_acc, imp, acc))

        # best eviction targets: lowest score and not pinned/protected
        ranked = [(mid, content, score, last_acc, imp, acc)
                  for (mid, content, score, pin, prot, last_acc, imp, acc) in candidates
                  if not pin and not prot]
        ranked.sort(key=lambda x: x[2])  # lowest score first

        over_by = max(0, total - self.max_memories_per_user)
        preview = [{
            "id": mid,
            "preview": (content[:120] + ("…" if len(content) > 120 else "")),
            "eviction_score": round(float(score), 4),
            "last_accessed": last_acc.isoformat(),
            "importance": round(float(imp), 3),
            "access_count": int(acc),
        } for (mid, content, score, last_acc, imp, acc) in ranked[: min(10, len(ranked))]]

        return {
            "user_id": user_id,
            "total": total,
            "pinned": pinned,
            "protected": protected,
            "limit": self.max_memories_per_user,
            "over_limit_by": over_by,
            "eviction_threshold_days": self.eviction_threshold_days,
            "min_importance_threshold": self.min_importance_threshold,
            "top_candidates": preview,
        }

# Initialize components
memory_manager = MemoryManager()
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI Chat Server with Memory System")
    # Create templates directory if it doesn't exist
    import os
    os.makedirs("templates", exist_ok=True)
    # os.makedirs("static", exist_ok=True)
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
# app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.post("/admin/eviction/test")
async def test_eviction_cycle(current_user: str = Depends(get_current_user)):
    """Test the full eviction cycle (admin endpoint)"""
    try:
        logger.info("Manual eviction cycle triggered")
        await memory_manager._run_eviction_cycle()
        return {"status": "test_eviction_completed", "message": "Check logs for details"}
    except Exception as e:
        logger.error(f"Error during test eviction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/scheduler/status")
async def scheduler_status(current_user: str = Depends(get_current_user)):
    """Check eviction scheduler status"""
    eviction_threads = []
    for thread in threading.enumerate():
        if "Eviction" in thread.name:
            eviction_threads.append({
                "name": thread.name,
                "alive": thread.is_alive(),
                "daemon": thread.daemon
            })
    
    return {
        "scheduler_running": any(t["alive"] for t in eviction_threads if "Worker" in t["name"]),
        "threads": eviction_threads,
        "config": {
            "max_memories_per_user": memory_manager.max_memories_per_user,
            "eviction_threshold_days": memory_manager.eviction_threshold_days,
            "min_importance_threshold": memory_manager.min_importance_threshold
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if eviction thread is running
    eviction_thread_running = False
    for thread in threading.enumerate():
        if thread.name == "MemoryEvictionWorker" and thread.is_alive():
            eviction_thread_running = True
            break
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "eviction_scheduler": "running" if eviction_thread_running else "stopped",
        "memory_config": {
            "max_memories_per_user": memory_manager.max_memories_per_user,
            "eviction_threshold_days": memory_manager.eviction_threshold_days,
            "min_importance_threshold": memory_manager.min_importance_threshold
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        exit(1)
    
    # Create necessary directories
    os.makedirs("templates", exist_ok=True)
    # os.makedirs("static", exist_ok=True)
    
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
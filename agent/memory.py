"""
Memory module with mem0 integration for persistent semantic memory.
Maintains both session history (local) and long-term memory (mem0).
"""

from datetime import datetime
from typing import Optional
import json


class SessionMemory:
    """Tracks all actions, transcriptions, and chat within a session."""

    def __init__(self):
        self.history = []
        self.chat_context = []
        self.session_start = datetime.now()

    def add_entry(
        self,
        transcription: str,
        intent_data: dict,
        results: list,
        audio_source: str = "microphone",
    ):
        """Add a completed action to history."""
        entry = {
            "id": len(self.history) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "audio_source": audio_source,
            "transcription": transcription,
            "primary_intent": intent_data.get("primary_intent", "unknown"),
            "all_intents": intent_data.get("intents", []),
            "confidence": intent_data.get("confidence", 0),
            "results": results,
            "status": "success" if all(r.get("status") == "success" for r in results) else "partial",
        }
        self.history.append(entry)

        self.chat_context.append({
            "role": "user",
            "content": transcription
        })
        if results:
            last_output = results[-1].get("output", "")
            if last_output:
                self.chat_context.append({
                    "role": "assistant",
                    "content": last_output[:500]
                })

        if len(self.chat_context) > 20:
            self.chat_context = self.chat_context[-20:]

        return entry

    def get_history_display(self) -> list:
        """Format history for Gradio display."""
        rows = []
        for entry in reversed(self.history):
            intents_str = ", ".join(entry.get("all_intents", []))
            status_icon = "✅" if entry.get("status") == "success" else "⚠️"
            files_created = [
                r.get("filename", "") for r in entry.get("results", [])
                if r.get("filename")
            ]
            files_str = ", ".join(files_created) if files_created else "—"

            rows.append([
                f"#{entry['id']}",
                entry["timestamp"],
                entry["transcription"][:60] + ("…" if len(entry["transcription"]) > 60 else ""),
                intents_str,
                f"{status_icon} {entry.get('status', 'unknown')}",
                files_str,
            ])
        return rows

    def get_stats(self) -> dict:
        """Get session statistics."""
        total = len(self.history)
        successful = sum(1 for e in self.history if e.get("status") == "success")
        files_created = sum(
            1 for e in self.history
            for r in e.get("results", [])
            if r.get("filepath")
        )
        intent_counts = {}
        for e in self.history:
            for intent in e.get("all_intents", []):
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return {
            "total_commands": total,
            "successful": successful,
            "files_created": files_created,
            "top_intent": max(intent_counts, key=intent_counts.get) if intent_counts else "—",
            "session_duration": str(datetime.now() - self.session_start).split('.')[0],
        }

    def clear(self):
        """Clear session history."""
        self.history.clear()
        self.chat_context.clear()

    def export_json(self) -> str:
        """Export session history as JSON string."""
        return json.dumps({
            "session_start": self.session_start.isoformat(),
            "history": self.history
        }, indent=2, default=str)


# ─────────────────────────────────────────────
# MEM0 PERSISTENT MEMORY LAYER
# ─────────────────────────────────────────────

class Mem0Memory:
    """
    Persistent semantic memory using the mem0 platform.
    
    Stores user preferences, facts, and interaction patterns
    so the agent "remembers" across sessions.
    
    Features:
    - Automatic fact extraction from conversations
    - Semantic search to retrieve relevant context
    - User-scoped memory isolation
    """

    def __init__(self, api_key: str = "", user_id: str = "voice_agent_user"):
        self.api_key = api_key.strip()
        self.user_id = user_id
        self._client = None
        self._available = False
        self._init_error = ""

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize the mem0 MemoryClient."""
        try:
            from mem0 import MemoryClient
            self._client = MemoryClient(api_key=self.api_key)
            self._available = True
            self._init_error = ""
        except ImportError:
            self._init_error = "mem0ai not installed. Run: pip install mem0ai"
            self._available = False
        except Exception as e:
            self._init_error = f"mem0 init failed: {str(e)}"
            self._available = False

    def update_key(self, api_key: str):
        """Update the mem0 API key and reinitialize."""
        self.api_key = api_key.strip()
        if self.api_key:
            self._init_client()
        else:
            self._client = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    def add(self, transcription: str, response: str, intent: str = ""):
        """
        Store a conversation exchange in mem0.
        mem0 automatically extracts facts and preferences.
        """
        if not self.is_available:
            return None

        try:
            messages = [
                {"role": "user", "content": transcription},
                {"role": "assistant", "content": response[:1000]},
            ]
            metadata = {"intent": intent} if intent else {}
            result = self._client.add(
                messages,
                user_id=self.user_id,
                metadata=metadata,
            )
            return result
        except Exception as e:
            print(f"[mem0] add error: {e}")
            return None

    def search(self, query: str, limit: int = 5) -> list:
        """
        Search mem0 for relevant memories based on a query.
        Returns a list of memory strings.
        """
        if not self.is_available:
            return []

        try:
            results = self._client.search(query, filters={"user_id": self.user_id})
            memories = []
            # Handle both list and dict response formats
            items = results if isinstance(results, list) else results.get("results", [])
            for item in items[:limit]:
                mem_text = item.get("memory", "") if isinstance(item, dict) else str(item)
                if mem_text:
                    memories.append(mem_text)
            return memories
        except Exception as e:
            print(f"[mem0] search error: {e}")
            return []

    def get_all(self) -> list:
        """
        Retrieve all stored memories for the current user.
        Returns a list of dicts with 'memory', 'created_at', etc.
        """
        if not self.is_available:
            return []

        try:
            results = self._client.get_all(filters={"user_id": self.user_id})
            items = results if isinstance(results, list) else results.get("results", [])
            return items
        except Exception as e:
            print(f"[mem0] get_all error: {e}")
            return []

    def get_context_prompt(self, query: str) -> str:
        """
        Build a context string from relevant memories for LLM injection.
        Returns a formatted string to prepend to system prompts.
        """
        memories = self.search(query, limit=5)
        if not memories:
            return ""

        context = "## Relevant User Context (from memory)\n"
        for i, mem in enumerate(memories, 1):
            context += f"  {i}. {mem}\n"
        context += "\nUse this context to personalize your response.\n"
        return context

    def get_display_memories(self) -> str:
        """Format all memories for UI display."""
        all_mems = self.get_all()
        if not all_mems:
            if not self.is_available:
                if self._init_error:
                    return f"⚠️ {self._init_error}"
                return "🔑 Enter your Mem0 API key to enable persistent memory."
            return "🧠 No memories stored yet. Start talking to build memory!\n\n💡 *Tip: Memories are processed asynchronously (5-10s delay). Click **Refresh Memories** after using voice commands.*"

        lines = [f"🧠 **{len(all_mems)} Memories Stored**\n"]
        for item in all_mems[:30]:
            mem = item.get("memory", str(item)) if isinstance(item, dict) else str(item)
            created = ""
            if isinstance(item, dict) and item.get("created_at"):
                created = f"  _{item['created_at'][:10]}_"
            lines.append(f"• {mem}{created}")

        return "\n".join(lines)

    def delete_all(self):
        """Delete all memories for the current user."""
        if not self.is_available:
            return False
        try:
            self._client.delete_all(filters={"user_id": self.user_id})
            return True
        except Exception as e:
            print(f"[mem0] delete_all error: {e}")
            return False

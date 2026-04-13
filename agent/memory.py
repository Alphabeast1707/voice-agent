"""
Session Memory module.
Maintains persistent history of actions and chat context within a session.
"""

from datetime import datetime
from typing import Optional
import json


class SessionMemory:
    """Tracks all actions, transcriptions, and chat within a session."""

    def __init__(self):
        self.history = []  # List of session entries
        self.chat_context = []  # For LLM chat history
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

        # Update chat context for LLM
        self.chat_context.append({
            "role": "user",
            "content": transcription
        })
        if results:
            last_output = results[-1].get("output", "")
            if last_output:
                self.chat_context.append({
                    "role": "assistant",
                    "content": last_output[:500]  # Truncate for context window
                })

        # Keep context window manageable (last 10 exchanges)
        if len(self.chat_context) > 20:
            self.chat_context = self.chat_context[-20:]

        return entry

    def get_history_display(self) -> list:
        """Format history for Gradio display."""
        rows = []
        for entry in reversed(self.history):  # Newest first
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

"""
Intent Understanding module using Groq API (Llama 3.3 70B).
Classifies user commands and extracts structured parameters.
"""

import json
import re
from typing import Optional


INTENT_SYSTEM_PROMPT = """You are an intent classification engine for a voice-controlled AI agent.

Analyze the user's transcribed speech and return a JSON object with this exact structure:
{
  "intents": ["intent1", "intent2"],
  "primary_intent": "main_intent",
  "parameters": {
    "filename": "optional filename if mentioned",
    "language": "programming language if code is requested",
    "content": "text content to summarize or write if provided",
    "description": "what the user wants to create/do",
    "folder": "folder name if mentioned"
  },
  "commands": [
    {
      "type": "intent_type",
      "params": {}
    }
  ],
  "confidence": 0.95,
  "raw_request": "cleaned version of what user asked"
}

Supported intents (use EXACTLY these strings):
- "create_file"      → User wants to create a new file or folder
- "write_code"       → User wants code generated and saved to a file
- "summarize_text"   → User wants text summarized
- "general_chat"     → General conversation, questions, no action needed
- "delete_file"      → User wants to delete a file (safety: always confirm)
- "list_files"       → User wants to see files in the output folder
- "read_file"        → User wants to read a file's contents

For compound commands (multiple intents), list ALL detected intents.
The "commands" array should have one entry per action to execute, in order.

For "write_code", always extract:
- "language": the programming language (python, javascript, etc.)
- "filename": suggested filename with correct extension
- "description": detailed description of what to code

For "create_file":
- "filename": the file/folder name
- "content": initial content if any

For "summarize_text":
- "content": the text to summarize (if provided inline)

IMPORTANT:
- If no filename is given, generate a sensible one.
- Always use .py for Python, .js for JavaScript, .ts for TypeScript, .txt for text, etc.
- Never suggest filenames outside the output/ folder.
- Return ONLY valid JSON, no markdown, no explanation."""

LLM_MODEL = "llama-3.3-70b-versatile"


def classify_intent(text: str, groq_api_key: str, chat_history: list = None) -> dict:
    """
    Classify the intent of transcribed speech using Groq LLM.
    
    Args:
        text: Transcribed text from audio
        groq_api_key: Groq API key
        chat_history: Optional list of previous messages for context
    
    Returns:
        dict with intent classification results
    """
    if not text or not text.strip():
        return {
            "intents": ["general_chat"],
            "primary_intent": "general_chat",
            "parameters": {},
            "commands": [{"type": "general_chat", "params": {"message": ""}}],
            "confidence": 0.0,
            "raw_request": "",
            "status": "error",
            "error": "No text provided for intent classification."
        }

    if not groq_api_key or not groq_api_key.strip():
        return _fallback_intent(text, error="Groq API key is missing.")

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key.strip())

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": f'Classify this voice command: "{text}"'},
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        result = json.loads(raw)
        result["status"] = "success"
        return result

    except json.JSONDecodeError as e:
        return _fallback_intent(text, error=f"Failed to parse intent JSON: {e}")
    except ImportError:
        return _fallback_intent(text, error="groq package not installed. Run: pip install groq")
    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "api_key" in error_msg.lower() or "401" in error_msg:
            error_msg = "Invalid Groq API key. Please check your key at console.groq.com."
        elif "rate" in error_msg.lower():
            error_msg = "Rate limit hit. Please wait a moment and try again."
        return _fallback_intent(text, error=f"Intent classification failed: {error_msg}")


def _fallback_intent(text: str, error: str = "") -> dict:
    """Simple keyword-based fallback intent detection."""
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["summarize", "summary", "summarise", "tldr"]):
        primary = "summarize_text"
    elif any(k in text_lower for k in ["write code", "create code", "generate code", "function", "class", "script", "program"]):
        primary = "write_code"
    elif any(k in text_lower for k in ["create file", "make file", "new file", "create folder", "make folder"]):
        primary = "create_file"
    elif any(k in text_lower for k in ["list files", "show files", "what files"]):
        primary = "list_files"
    else:
        primary = "general_chat"

    return {
        "intents": [primary],
        "primary_intent": primary,
        "parameters": {"description": text},
        "commands": [{"type": primary, "params": {"description": text}}],
        "confidence": 0.5,
        "raw_request": text,
        "status": "fallback",
        "error": error
    }


def get_intent_display_name(intent: str) -> str:
    """Human-readable intent names."""
    names = {
        "create_file": "📄 Create File",
        "write_code": "💻 Write Code",
        "summarize_text": "📝 Summarize Text",
        "general_chat": "💬 General Chat",
        "delete_file": "🗑️ Delete File",
        "list_files": "📂 List Files",
        "read_file": "👁️ Read File",
    }
    return names.get(intent, f"🔷 {intent.replace('_', ' ').title()}")

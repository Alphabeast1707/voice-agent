"""
Tool Execution module.
Handles: file creation, code generation, text summarization, file listing, reading.
All LLM tasks use Groq API (Llama 3.3 70B).

SAFETY: All file operations are restricted to the output/ directory.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


OUTPUT_DIR = Path(__file__).parent.parent / "output"
LLM_MODEL = "llama-3.3-70b-versatile"


def ensure_output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_path(filename: str) -> Path:
    """
    Resolve a safe path inside the output directory.
    Prevents path traversal attacks.
    """
    ensure_output_dir()
    clean = re.sub(r'[<>:"|?*]', '_', filename)
    clean = clean.lstrip('./')
    clean = Path(clean).name if '/' not in clean and '\\' not in clean else Path(clean).name
    return OUTPUT_DIR / clean


def _get_groq_client(api_key: str):
    """Get a configured Groq client instance."""
    from groq import Groq
    return Groq(api_key=api_key.strip())


def _llm_generate(client, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Helper to call Groq LLM and return text response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def execute_command(command: dict, intent_data: dict, groq_api_key: str, confirmed: bool = True) -> dict:
    """
    Execute a single command based on its type.
    
    Returns:
        dict with 'status', 'action', 'output', 'filepath' (optional)
    """
    cmd_type = command.get("type", "general_chat")
    params = command.get("params", {})

    # Merge top-level intent parameters
    all_params = {**intent_data.get("parameters", {}), **params}

    if cmd_type == "write_code":
        return execute_write_code(all_params, intent_data.get("raw_request", ""), groq_api_key)
    elif cmd_type == "create_file":
        return execute_create_file(all_params)
    elif cmd_type == "summarize_text":
        return execute_summarize(all_params, intent_data.get("raw_request", ""), groq_api_key)
    elif cmd_type == "list_files":
        return execute_list_files()
    elif cmd_type == "read_file":
        return execute_read_file(all_params)
    elif cmd_type == "general_chat":
        return execute_chat(all_params, intent_data.get("raw_request", ""), groq_api_key)
    elif cmd_type == "delete_file":
        return {
            "status": "skipped",
            "action": "Delete File",
            "output": "⚠️ File deletion requires explicit confirmation in settings."
        }
    else:
        return {
            "status": "error",
            "action": f"Unknown command: {cmd_type}",
            "output": f"No handler found for intent '{cmd_type}'."
        }


def execute_write_code(params: dict, raw_request: str, groq_api_key: str) -> dict:
    """Generate code using Groq LLM and save to file."""
    language = params.get("language", "python")
    description = params.get("description", raw_request)
    filename = params.get("filename", "")

    if not filename:
        ext_map = {
            "python": "py", "javascript": "js", "typescript": "ts",
            "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp",
            "rust": "rs", "go": "go", "ruby": "rb", "php": "php",
            "html": "html", "css": "css", "sql": "sql", "bash": "sh",
            "shell": "sh", "r": "r", "kotlin": "kt", "swift": "swift"
        }
        ext = ext_map.get(language.lower(), "py")
        words = re.sub(r'[^a-z0-9\s]', '', description.lower()).split()[:3]
        filename = "_".join(words) + f".{ext}" if words else f"generated_code.{ext}"

    filepath = safe_path(filename)

    if not groq_api_key:
        return {
            "status": "error",
            "action": "Write Code",
            "output": "Groq API key required for code generation."
        }

    try:
        client = _get_groq_client(groq_api_key)

        code_prompt = f"""Generate clean, well-commented {language} code for the following:

{description}

Requirements:
- Write production-quality code with proper error handling
- Include docstrings/comments explaining the code
- Make it complete and runnable
- Return ONLY the code, no markdown fences, no explanation

Language: {language}
"""
        code = _llm_generate(client, code_prompt, temperature=0.2)

        # Strip markdown if accidentally included
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)

        # Add header comment
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_comments = {
            "python": f'"""\nGenerated by Voice Agent\nRequest: {description}\nDate: {now}\n"""\n\n',
            "javascript": f'/**\n * Generated by Voice Agent\n * Request: {description}\n * Date: {now}\n */\n\n',
            "java": f'/**\n * Generated by Voice Agent\n * Request: {description}\n * Date: {now}\n */\n\n',
        }
        header = header_comments.get(language.lower(), f"// Generated by Voice Agent\n// {description}\n// {now}\n\n")
        final_code = header + code

        filepath.write_text(final_code, encoding="utf-8")

        return {
            "status": "success",
            "action": f"Write Code → {filename}",
            "output": final_code,
            "filepath": str(filepath),
            "filename": filename,
            "language": language
        }

    except Exception as e:
        return {
            "status": "error",
            "action": "Write Code",
            "output": f"Code generation failed: {str(e)}"
        }


def execute_create_file(params: dict) -> dict:
    """Create a file or folder in the output directory."""
    filename = params.get("filename", "")
    content = params.get("content", "")
    description = params.get("description", "")

    if not filename:
        words = re.sub(r'[^a-z0-9\s]', '', description.lower()).split()[:3]
        filename = "_".join(words) + ".txt" if words else f"file_{datetime.now().strftime('%H%M%S')}.txt"

    filepath = safe_path(filename)

    try:
        if filename.endswith('/') or (not Path(filename).suffix and '.' not in filename):
            dir_path = OUTPUT_DIR / filename.rstrip('/')
            dir_path.mkdir(parents=True, exist_ok=True)
            return {
                "status": "success",
                "action": f"Create Folder → {filename}",
                "output": f"✅ Folder '{filename}' created in output/",
                "filepath": str(dir_path)
            }
        else:
            initial_content = content or f"# {filename}\nCreated by Voice Agent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            filepath.write_text(initial_content, encoding="utf-8")
            return {
                "status": "success",
                "action": f"Create File → {filename}",
                "output": initial_content,
                "filepath": str(filepath),
                "filename": filename
            }
    except Exception as e:
        return {
            "status": "error",
            "action": "Create File",
            "output": f"File creation failed: {str(e)}"
        }


def execute_summarize(params: dict, raw_request: str, groq_api_key: str) -> dict:
    """Summarize text using Groq LLM."""
    content = params.get("content", "")
    filename = params.get("filename", "")

    if not content:
        content = raw_request

    if not content.strip():
        return {
            "status": "error",
            "action": "Summarize Text",
            "output": "No text provided to summarize. Please provide the text content."
        }

    if not groq_api_key:
        return {
            "status": "error",
            "action": "Summarize Text",
            "output": "Groq API key required for summarization."
        }

    try:
        client = _get_groq_client(groq_api_key)

        summary = _llm_generate(
            client,
            f"""Please provide a clear, concise summary of the following text.
Structure your summary with:
1. A one-sentence TL;DR
2. Key points (3-5 bullet points)
3. Main takeaway

Text to summarize:
{content}""",
            temperature=0.3,
            max_tokens=1000,
        )

        result = {
            "status": "success",
            "action": "Summarize Text",
            "output": summary
        }

        if filename:
            filepath = safe_path(filename)
            save_content = f"# Summary\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Original Text\n{content}\n\n## Summary\n{summary}\n"
            filepath.write_text(save_content, encoding="utf-8")
            result["filepath"] = str(filepath)
            result["action"] = f"Summarize Text → Save to {filename}"

        return result

    except Exception as e:
        return {
            "status": "error",
            "action": "Summarize Text",
            "output": f"Summarization failed: {str(e)}"
        }


def execute_list_files() -> dict:
    """List all files in the output directory."""
    ensure_output_dir()
    files = list(OUTPUT_DIR.iterdir())

    if not files:
        return {
            "status": "success",
            "action": "List Files",
            "output": "📂 The output/ folder is empty."
        }

    file_list = []
    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else 0
        modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        icon = "📄" if f.is_file() else "📁"
        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
        file_list.append(f"{icon} {f.name:40} {size_str:12} {modified}")

    output = f"📂 Files in output/ ({len(files)} items):\n\n"
    output += "\n".join(file_list)

    return {
        "status": "success",
        "action": "List Files",
        "output": output
    }


def execute_read_file(params: dict) -> dict:
    """Read a file from the output directory."""
    filename = params.get("filename", "")
    if not filename:
        return {
            "status": "error",
            "action": "Read File",
            "output": "Please specify a filename to read."
        }

    filepath = safe_path(filename)
    if not filepath.exists():
        return {
            "status": "error",
            "action": "Read File",
            "output": f"File '{filename}' not found in output/."
        }

    try:
        content = filepath.read_text(encoding="utf-8")
        return {
            "status": "success",
            "action": f"Read File → {filename}",
            "output": content,
            "filepath": str(filepath)
        }
    except Exception as e:
        return {
            "status": "error",
            "action": "Read File",
            "output": f"Failed to read file: {str(e)}"
        }


def execute_chat(params: dict, raw_request: str, groq_api_key: str) -> dict:
    """Handle general chat with Groq LLM."""
    message = params.get("message", raw_request)

    if not message.strip():
        return {
            "status": "success",
            "action": "General Chat",
            "output": "Hello! I'm your Voice Agent. I can create files, write code, summarize text, and chat. What would you like to do?"
        }

    if not groq_api_key:
        return {
            "status": "success",
            "action": "General Chat",
            "output": f"(API key needed for AI responses)\n\nYou said: {message}"
        }

    try:
        client = _get_groq_client(groq_api_key)
        reply = _llm_generate(
            client,
            message,
            system="You are a helpful voice-controlled AI agent. Be concise, friendly, and helpful. The user is interacting via voice, so keep responses conversational and clear.",
            temperature=0.7,
            max_tokens=1000,
        )
        return {
            "status": "success",
            "action": "General Chat",
            "output": reply
        }
    except Exception as e:
        return {
            "status": "error",
            "action": "General Chat",
            "output": f"Chat response failed: {str(e)}"
        }


def execute_all_commands(intent_data: dict, groq_api_key: str) -> list:
    """Execute all commands from intent classification (compound command support)."""
    commands = intent_data.get("commands", [])
    if not commands:
        commands = [{"type": intent_data.get("primary_intent", "general_chat"), "params": {}}]

    results = []
    for cmd in commands:
        result = execute_command(cmd, intent_data, groq_api_key)
        result["command_type"] = cmd.get("type", "unknown")
        results.append(result)

    return results

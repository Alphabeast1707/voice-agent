# Building a Voice-Controlled AI Agent: Architecture, Challenges, and Lessons Learned

*How I built an end-to-end voice command system using Groq, Llama 3.3, and mem0 — from speech to action in under 2 seconds.*

---

## Introduction

What if you could just *talk* to your computer and it would do things for you? Not just transcribe your words, but actually understand your intent and take action — create files, write code, summarize documents, and remember your preferences over time.

That's exactly what I built: **Voice Agent** — an AI-powered voice command system that converts spoken input into real actions through a five-stage pipeline. In this article, I'll walk through the architecture, the technical decisions I made, and the challenges I encountered along the way.

**GitHub**: [github.com/Alphabeast1707/voice-agent](https://github.com/Alphabeast1707/voice-agent)

---

## System Architecture

The system follows a clean, modular pipeline:

```
🎙️ Audio Input → 📝 Speech-to-Text → 🧠 Memory Retrieval → 🎯 Intent Classification → ⚙️ Tool Execution → 📂 Output
```

Each stage is handled by a separate module, making the system easy to extend, test, and debug. Here's how they fit together:

### 1. Audio Input (`app.py`)
The Gradio-based UI accepts audio from two sources:
- **Live microphone** recording
- **File upload** (.wav, .mp3)

This flexibility is critical for both real-time interaction and testing with pre-recorded samples.

### 2. Speech-to-Text (`agent/stt.py`)
Audio is transcribed using **Groq's Whisper v3 API** (`whisper-large-v3`). I chose Groq over local Whisper for a specific reason:

| Factor | Local Whisper | Groq Whisper |
|--------|-------------|--------------|
| Latency | 5-15s (CPU) | **0.5-2s** |
| Hardware | 6-10GB VRAM | None needed |
| Model | whisper-large-v3 | Same model |
| Free tier | N/A | 2,000 min/day |

The same model, 10x faster, with no GPU required. For a demo-focused project, this was a clear win. I kept a local fallback (`whisper-base`) for offline use.

### 3. Memory Retrieval (`agent/memory.py`)
Before classifying intent, the system queries **mem0** for relevant user context. If you've told the agent before that you prefer Python with type hints, that context gets injected into the LLM prompt automatically.

```python
# Before LLM call, retrieve relevant memories
mem0_context = mem0.get_context_prompt(transcription)
# Result: "User prefers Python with type hints and functional style"
```

This makes the agent *personalized* — it learns and adapts over time.

### 4. Intent Classification (`agent/intent.py`)
The transcription is sent to **Llama 3.3 70B** (via Groq) with a structured prompt that returns JSON:

```json
{
  "primary_intent": "write_code",
  "intents": ["write_code"],
  "confidence": 0.95,
  "parameters": {
    "language": "python",
    "description": "retry decorator with exponential backoff",
    "filename": "retry_decorator.py"
  },
  "commands": [{"type": "write_code", "params": {...}}]
}
```

The system supports **six intents**: `create_file`, `write_code`, `summarize_text`, `general_chat`, `list_files`, and `read_file`.

**Compound commands** are the interesting part — saying *"Summarize this and save it to notes.txt"* generates two commands that execute in sequence.

### 5. Tool Execution (`agent/tools.py`)
Each intent maps to a handler function. Code generation, summarization, and chat responses all go through the same `_llm_generate()` helper, enriched with mem0 context when available.

All file operations are **sandboxed** to the `output/` directory with path traversal prevention.

---

## Key Technical Challenges

### Challenge 1: Unified API Architecture

**Problem**: The original design used three separate APIs — Groq for STT, Anthropic for intent classification, and another for code generation. This meant three API keys, three failure points, and three rate limits to manage.

**Solution**: I consolidated everything onto **a single Groq API key**. Groq handles both Whisper STT *and* Llama 3.3 70B for all LLM tasks. One key, one billing dashboard, one rate limit. The `_llm_generate()` helper function abstracts all LLM calls:

```python
def _llm_generate(client, prompt, system="", temperature=0.2, max_tokens=2000):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
```

### Challenge 2: Reliable Intent Classification from Natural Speech

**Problem**: Voice input is messy. People say "um", speak in fragments, and phrase the same intent in dozens of different ways. Getting a 70B LLM to reliably return structured JSON from natural speech was harder than expected.

**Solution**: Three-layer approach:
1. **Structured system prompt** with explicit JSON schema and examples
2. **JSON extraction with regex fallback** — if the LLM wraps JSON in markdown fences, I strip them
3. **Keyword-based fallback** — if the LLM API fails entirely, a rule-based classifier catches common patterns like "create", "write", "summarize"

```python
# Fallback keyword detection
if "create" in text and "file" in text:
    return {"primary_intent": "create_file", ...}
```

This ensures the agent **never crashes** on unexpected input.

### Challenge 3: Compound Command Parsing

**Problem**: Users naturally chain commands: *"Write a sorting function and save it to sort.py"*. A single-intent classifier would miss the file creation step.

**Solution**: The intent classifier returns a `commands` array, not a single intent:

```json
{
  "commands": [
    {"type": "write_code", "params": {"description": "sorting function", "language": "python"}},
    {"type": "create_file", "params": {"filename": "sort.py"}}
  ]
}
```

The executor iterates through all commands sequentially, feeding the output of one into the next when relevant.

### Challenge 4: Persistent Memory Across Sessions

**Problem**: The original `SessionMemory` class was pure in-memory — restart the app and everything is gone. For a voice agent to be truly useful, it needs to remember things about the user.

**Solution**: Integrated **mem0** as a persistent semantic memory layer. mem0 doesn't just store raw strings — it automatically extracts facts and preferences from conversations:

```
User says: "I always write Python with type hints"
mem0 stores: "User prefers Python with type hints"
```

Next time the user says "write a function", the agent retrieves this memory and injects it into the LLM prompt, producing code with type hints automatically.

**Key learning**: mem0's platform processes memories asynchronously (5-10 second delay). I had to handle this gracefully in the UI rather than showing "no memories" immediately after a conversation.

### Challenge 5: The White Screen Problem (Gradio Theming)

**Problem**: I built a custom "Neon Dark" theme inspired by Google's Stitch design system. It looked great — except for a persistent white strip on the right side of the viewport that Gradio's default CSS was injecting.

**Solution**: I had to override *every* Gradio wrapper element's background:

```css
html, body, .gradio-container,
.main, .wrap, .contain, main, footer,
div[class*="footer"], .gr-form,
#component-0, .app {
    background: #121212 !important;
}
```

Lesson learned: when building custom themes in Gradio, you're fighting the framework's defaults. Be thorough with your CSS overrides.

---

## Safety & Human-in-the-Loop

An AI agent that creates and modifies files needs guardrails:

1. **Output Sandbox**: All file operations are restricted to `output/`. Path traversal attempts (e.g., `../../etc/passwd`) are sanitized.
2. **Confirmation Gate**: When enabled, file-writing operations pause and show a confirm/cancel prompt before executing.
3. **No Deletion**: File deletion is intentionally disabled to prevent accidental data loss.

```python
def safe_path(filename: str) -> Path:
    clean = re.sub(r'[<>:"|?*]', '_', filename)
    clean = clean.lstrip('./')
    return OUTPUT_DIR / Path(clean).name  # Prevents directory traversal
```

---

## Tech Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| **UI** | Gradio + Custom CSS | Rapid prototyping with full theme control |
| **STT** | Groq Whisper v3 | 10x real-time speed, free tier |
| **LLM** | Groq Llama 3.3 70B | Same API key as STT, sub-second latency |
| **Memory** | mem0 Platform | Automatic fact extraction, cross-session persistence |
| **Backend** | Pure Python | Zero framework overhead, easy to understand |

**Total setup**: `pip install -r requirements.txt` → `python app.py` → done.

---

## Results & Performance

| Metric | Value |
|--------|-------|
| End-to-end latency | **1.5-4 seconds** (voice → action) |
| STT accuracy | Excellent (whisper-large-v3) |
| Intent classification | ~95% on supported intents |
| Compound command support | Yes (2+ actions per voice command) |
| Memory personalization | Automatic after 1-2 interactions |

---

## What I'd Do Differently

1. **Streaming responses**: Currently, the entire pipeline blocks until all stages complete. Adding streaming would make it feel more responsive.
2. **Local LLM option**: For privacy-sensitive use cases, adding Ollama support would allow fully offline operation.
3. **Multi-turn conversations**: The current design is single-turn (one command → one action). Building true conversational state would enable more complex interactions.
4. **Voice output (TTS)**: Adding text-to-speech for agent responses would make the interaction feel more natural.

---

## Conclusion

Building Voice Agent taught me that the technical challenge isn't any single component — it's making them work together reliably. Speech recognition, intent classification, memory retrieval, tool execution, and UI rendering all need to handle failures gracefully and communicate state clearly to the user.

The key insight: **a voice agent is only as good as its weakest link**. Investing in fallback mechanisms (keyword detection, graceful degradation, sandbox safety) is what separates a demo from something you'd actually use.

The full source code is available on GitHub: [github.com/Alphabeast1707/voice-agent](https://github.com/Alphabeast1707/voice-agent)

---

*If you found this useful, feel free to star the repo or reach out with questions. I'm always happy to discuss AI agent architectures!*

**Tags**: `#AI` `#VoiceAgent` `#Python` `#Groq` `#LLM` `#mem0` `#MachineLearning` `#BuildInPublic`

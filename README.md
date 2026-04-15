# ⬡ Voice Agent

> A voice-controlled AI agent that transcribes speech, understands intent, remembers your preferences, and executes actions — all in a local web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![Whisper](https://img.shields.io/badge/STT-Whisper_v3_(Groq)-green?style=flat-square)
![Llama](https://img.shields.io/badge/LLM-Llama_3.3_70B_(Groq)-purple?style=flat-square)
![mem0](https://img.shields.io/badge/Memory-mem0-ff4081?style=flat-square)

---

## 🎯 What It Does

Voice Agent converts your spoken commands into real actions on your machine:

| Voice Command | What Happens |
|---|---|
| *"Create a Python file with a retry function"* | Generates & saves `retry_function.py` in `output/` |
| *"Summarize this text and save it to notes.txt"* | Summarizes via Llama, saves to `output/notes.txt` |
| *"Create a folder called experiments"* | Creates `output/experiments/` |
| *"List all files in the output folder"* | Shows file listing |
| *"What is recursion?"* | General chat response |

**It also remembers you.** Over time, mem0 learns your coding style, language preferences, and habits — making responses more personalized.

---

## 🏗️ Architecture

```
Audio Input (mic/upload)
        │
        ▼
┌───────────────────┐
│  Speech-to-Text   │  ← Groq Whisper API (whisper-large-v3)
│   agent/stt.py    │
└────────┬──────────┘
         │  Transcription
         ▼
┌───────────────────┐
│   Memory Layer    │  ← mem0 (retrieves relevant user context)
│  agent/memory.py  │
└────────┬──────────┘
         │  Context-enriched input
         ▼
┌───────────────────┐
│ Intent Classifier │  ← Groq Llama 3.3 70B
│  agent/intent.py  │
└────────┬──────────┘
         │  Structured JSON intent
         ▼
┌───────────────────┐
│  Tool Executor    │  ← Code gen, file ops, summarization
│  agent/tools.py   │
└────────┬──────────┘
         │  Results + store to mem0
         ▼
┌───────────────────┐
│   Gradio UI       │  ← http://localhost:7860
│     app.py        │
└───────────────────┘
         │
         ▼
     output/          ← All generated files land here (sandboxed)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Alphabeast1707/voice-agent.git
cd voice-agent
pip install -r requirements.txt
```

### 2. Get API Keys

| Key | Where to Get | Used For | Required? |
|-----|-------------|----------|-----------|
| **Groq API Key** | [console.groq.com](https://console.groq.com) | STT + LLM (single key) | ✅ Yes |
| **Mem0 API Key** | [app.mem0.ai](https://app.mem0.ai) | Persistent memory | ⬜ Optional |

### 3. Set Environment Variables

```bash
export GROQ_API_KEY="gsk_..."
export MEM0_API_KEY="m0-..."   # Optional
```

Or enter them directly in the UI.

### 4. Run

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## 📋 Supported Intents

| Intent | Trigger Words | Action |
|--------|--------------|--------|
| `create_file` | "create a file", "make a file", "new file" | Creates file in `output/` |
| `write_code` | "write code", "generate a function", "create a script" | Generates & saves code |
| `summarize_text` | "summarize", "give me a summary", "tldr" | Summarizes text via Llama |
| `general_chat` | Anything else | Conversational response |
| `list_files` | "list files", "show files" | Lists `output/` contents |
| `read_file` | "read file", "show me" | Reads file contents |

### Compound Commands
Multiple intents in one command are supported:
> *"Summarize this paragraph and save it to summary.txt"*
→ Executes both `summarize_text` AND `create_file` in sequence.

---

## 🧠 Persistent Memory (mem0)

Voice Agent integrates **mem0** for long-term semantic memory:

- **Auto-extracts facts** from conversations (e.g., "user prefers Python", "uses functional style")
- **Injects context** into LLM prompts for personalized responses
- **Persists across sessions** — restart the app, your preferences are still there
- **Memory tab** in the UI lets you view and manage stored memories

### How it works:
1. You say: *"Write a Python function using type hints"*
2. mem0 stores: *"User prefers Python with type hints"*
3. Next time you say *"Write a sorting function"* → it automatically uses Python with type hints

---

## 🛡️ Safety Features

- **Output Sandbox**: ALL file operations are restricted to the `output/` directory. Path traversal is prevented.
- **Human-in-the-Loop**: Enable the confirmation checkbox to require approval before any file operation executes.
- **Graceful Degradation**: Silent audio, API failures, and unknown intents are handled with clear error messages, never crashes.

---

## 🔄 Model Decisions

### Why Groq for STT (instead of local Whisper)?

| | Local Whisper | Groq Whisper API *(chosen)* |
|---|---|---|
| Model | whisper-large-v3 | whisper-large-v3 (same) |
| Speed | ~2–5× real-time (GPU) | **~10× real-time** |
| Hardware | 6–10 GB VRAM | None required |
| Free tier | N/A | 2,000 min/day |

A local fallback (`agent/stt.py::transcribe_local`) using `whisper-base` is also provided.

### Why Groq Llama for LLM?

- **Single API key** handles both STT and LLM — zero friction
- **Sub-second latency** on 70B parameter model via Groq's LPU
- **Free tier** with no credit card
- **Structured JSON output** for reliable intent parsing

---

## 📁 Project Structure

```
voice-agent/
├── app.py                    # Main Gradio application + Stitch theme
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── agent/
│   ├── __init__.py
│   ├── stt.py               # Speech-to-Text (Groq Whisper)
│   ├── intent.py            # Intent Classification (Groq Llama 3.3 70B)
│   ├── tools.py             # Tool Execution (file ops, code gen, chat)
│   └── memory.py            # Session memory + mem0 persistent memory
└── output/                  # ← ALL generated files go here
    └── .gitkeep
```

---

## 💡 Features

| Feature | Status | Details |
|---------|--------|---------|
| **Compound Commands** | ✅ | "Summarize X and save to file.txt" executes both actions |
| **Human-in-the-Loop** | ✅ | Confirmation required before file writes (toggleable) |
| **Graceful Degradation** | ✅ | Handles silent audio, bad API keys, unknown intents |
| **Session Memory** | ✅ | History tab shows all commands, statuses, files created |
| **Persistent Memory** | ✅ | mem0 remembers preferences across sessions |
| **Keyword Fallback** | ✅ | If LLM API fails, keyword-based intent detection kicks in |
| **Stitch-Inspired UI** | ✅ | Neon dark theme with feature card grid and glassmorphic cards |

---

## 🔒 Environment Variables

```bash
GROQ_API_KEY=gsk_...    # Groq API key (STT + LLM)
MEM0_API_KEY=m0-...     # Mem0 API key (persistent memory, optional)
```

Never commit API keys. They're in `.gitignore` and masked in the UI.

---

## 📊 Model Benchmark Notes

| Model | Task | Latency (typical) | Quality |
|-------|------|--------------------|---------|
| Groq `whisper-large-v3` | STT | ~0.5–2s | Excellent |
| Local `whisper-base` | STT | ~5–15s (CPU) | Good |
| Groq `llama-3.3-70b-versatile` | Intent + Code + Chat | ~0.5–2s | Excellent |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes
4. Open a Pull Request

# ⬡ Voice Agent

> A voice-controlled AI agent that transcribes speech, understands intent, and executes actions — all in a local web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)
![Whisper](https://img.shields.io/badge/STT-Whisper_v3_(Groq)-green?style=flat-square)
![Llama](https://img.shields.io/badge/LLM-Llama_3.3_70B_(Groq)-purple?style=flat-square)

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
│ Intent Classifier │  ← Groq Llama 3.3 70B
│  agent/intent.py  │
└────────┬──────────┘
         │  Structured JSON intent
         ▼
┌───────────────────┐
│  Tool Executor    │  ← Code gen, file ops, summarization
│  agent/tools.py   │
└────────┬──────────┘
         │  Results
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

### 2. Get API Key (free tier)

| Key | Where to Get | Used For |
|-----|-------------|----------|
| **Groq API Key** | [console.groq.com](https://console.groq.com) — free | STT + LLM (single key for everything) |

### 3. (Optional) Set via Environment Variable

```bash
export GROQ_API_KEY="gsk_..."
```

Or enter it directly in the UI.

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

## 🛡️ Safety Features

- **Output Sandbox**: ALL file operations are restricted to the `output/` directory. Path traversal is prevented.
- **Human-in-the-Loop**: Enable the confirmation checkbox to require approval before any file operation executes.
- **Graceful Degradation**: Silent audio, API failures, and unknown intents are handled with clear error messages, never crashes.

---

## 🔄 Model Decisions

### Why Groq for STT (instead of local Whisper)?

**Option A — Local Whisper** (e.g., `openai/whisper-large-v3` via HuggingFace Transformers):
- Requires **6–10 GB VRAM** for the large-v3 model
- ~2–5× real-time speed on a modern GPU; much slower on CPU
- No API dependency, fully offline

**Option B — Groq Whisper API** *(chosen)*:
- Uses the same `whisper-large-v3` model, hosted by Groq
- **~10× real-time** inference speed via Groq's LPU hardware
- Free tier: 2,000 audio minutes/day (more than enough for development)
- Works on any machine, no GPU required

**Decision**: Groq was chosen to maximize accessibility and development speed. A local fallback (`agent/stt.py::transcribe_local`) using `openai/whisper-base` is also provided for offline use.

### Why Groq Llama for Intent + Code Generation?

- **Single API key**: Same Groq key handles both STT and LLM — zero friction setup
- **Speed**: Groq's LPU delivers sub-second latency on 70B parameter model
- **Structured JSON output**: Llama 3.3 70B reliably returns the exact JSON schema required for intent parsing
- **Code quality**: Produces production-quality code with docstrings and error handling
- **Free tier**: Generous free tier with no credit card required

---

## 📁 Project Structure

```
voice-agent/
├── app.py                    # Main Gradio application
├── requirements.txt
├── README.md
├── .gitignore
├── agent/
│   ├── __init__.py
│   ├── stt.py               # Speech-to-Text (Groq Whisper)
│   ├── intent.py            # Intent Classification (Groq Llama 3.3 70B)
│   ├── tools.py             # Tool Execution (file ops, code gen, chat)
│   └── memory.py            # Session history & context
└── output/                  # ← ALL generated files go here
    └── .gitkeep
```

---

## 💡 Bonus Features Implemented

| Feature | Status | Details |
|---------|--------|---------|
| **Compound Commands** | ✅ | "Summarize X and save to file.txt" executes both actions |
| **Human-in-the-Loop** | ✅ | Confirmation required before file writes (toggleable) |
| **Graceful Degradation** | ✅ | Handles silent audio, bad API keys, unknown intents |
| **Session Memory** | ✅ | History tab shows all commands, statuses, files created |
| **Keyword Fallback** | ✅ | If LLM API fails, keyword-based intent detection kicks in |
| **Stitch-Inspired UI** | ✅ | Dark neon theme with feature card grid and glassmorphic cards |

---

## 🔒 Environment Variables

```bash
GROQ_API_KEY=gsk_...    # Groq API key (STT + LLM — single key for everything)
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

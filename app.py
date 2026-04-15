"""
Voice Agent — AI-powered voice command system.
Main application entry point using Gradio.
Integrates mem0 for persistent semantic memory.

Run: python app.py
"""

import os
import gradio as gr
from pathlib import Path
from agent import (
    transcribe_audio,
    classify_intent,
    get_intent_display_name,
    execute_all_commands,
    SessionMemory,
    Mem0Memory,
)
from agent.tools import OUTPUT_DIR, execute_list_files

# --- Session State ---
memory = SessionMemory()
mem0 = Mem0Memory(
    api_key=os.environ.get("MEM0_API_KEY", ""),
    user_id="voice_agent_user",
)

# ─────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    audio_input,
    groq_key: str,
    mem0_key: str,
    require_confirmation: bool,
    pending_state: dict,
):
    """
    Full pipeline: audio → STT → intent → mem0 context → tool execution → mem0 store.
    Yields progressive updates to the UI.
    """
    if not audio_input:
        yield (
            "⚠️ No audio provided. Please record or upload audio.",
            "", "", "", "", pending_state,
            gr.update(visible=False), gr.update(visible=False)
        )
        return

    if not groq_key.strip():
        yield (
            "⚠️ Groq API key is required.",
            "", "", "", "", pending_state,
            gr.update(visible=False), gr.update(visible=False)
        )
        return

    # Update mem0 key if changed
    if mem0_key.strip() and mem0_key.strip() != mem0.api_key:
        mem0.update_key(mem0_key)

    # STT
    yield (
        "🎙️ Transcribing audio...",
        "", "", "", "", pending_state,
        gr.update(visible=False), gr.update(visible=False)
    )

    stt_result = transcribe_audio(audio_input, groq_key)

    if stt_result["status"] == "error":
        yield (
            f"❌ Transcription failed: {stt_result.get('error', 'Unknown error')}",
            "", "", "", "", pending_state,
            gr.update(visible=False), gr.update(visible=False)
        )
        return

    transcription = stt_result["text"]
    model_info = f"Model: {stt_result['model']} | Language: {stt_result['language']} | Duration: {stt_result['duration']}s"

    yield (
        f"✅ Transcribed: \"{transcription}\"",
        transcription,
        f"🔍 Analyzing intent...",
        "", "", pending_state,
        gr.update(visible=False), gr.update(visible=False)
    )

    # Intent Classification
    intent_data = classify_intent(transcription, groq_key)

    if intent_data.get("status") == "error":
        yield (
            f"⚠️ Intent classification failed: {intent_data.get('error')}",
            transcription,
            "Fallback to keyword-based detection",
            "", "", pending_state,
            gr.update(visible=False), gr.update(visible=False)
        )

    all_intents = intent_data.get("intents", [intent_data.get("primary_intent", "general_chat")])
    intent_badges = "  ".join([get_intent_display_name(i) for i in all_intents])
    confidence = intent_data.get("confidence", 0)
    intent_display = f"{intent_badges}\n\nConfidence: {confidence:.0%}\n{model_info}"

    params = intent_data.get("parameters", {})
    if params:
        param_str = "\n".join([f"  • {k}: {v}" for k, v in params.items() if v])
        intent_display += f"\n\nParameters:\n{param_str}"

    # Retrieve mem0 context
    mem0_context = ""
    if mem0.is_available:
        mem0_context = mem0.get_context_prompt(transcription)
        if mem0_context:
            intent_display += "\n\n🧠 Memory context injected"

    # Confirmation Gate
    file_writing_intents = {"write_code", "create_file", "summarize_text"}
    needs_file_op = any(i in file_writing_intents for i in all_intents)

    if require_confirmation and needs_file_op:
        new_pending = {
            "transcription": transcription,
            "intent_data": intent_data,
            "groq_key": groq_key,
            "mem0_context": mem0_context,
        }
        yield (
            "✅ Analysis complete. Awaiting confirmation.",
            transcription,
            intent_display,
            "⏳ Human-in-the-Loop: File operation detected.\nClick **Confirm & Execute** to proceed, or **Cancel** to abort.",
            "",
            new_pending,
            gr.update(visible=True),
            gr.update(visible=True),
        )
        return

    # Execute with mem0 context
    yield (
        "⚙️ Executing commands...",
        transcription,
        intent_display,
        "Running...",
        "", pending_state,
        gr.update(visible=False), gr.update(visible=False)
    )

    results = execute_all_commands(intent_data, groq_key, mem0_context)
    memory.add_entry(transcription, intent_data, results)

    # Store in mem0
    _store_in_mem0(transcription, results, all_intents)

    action_log, final_output = format_results(results)

    yield (
        "✅ Done!",
        transcription,
        intent_display,
        action_log,
        final_output,
        {},
        gr.update(visible=False),
        gr.update(visible=False)
    )


def confirm_execution(pending_state: dict):
    if not pending_state:
        return (
            "⚠️ No pending command.",
            "", "", "", "",
            gr.update(visible=False), gr.update(visible=False)
        )

    transcription = pending_state.get("transcription", "")
    intent_data = pending_state.get("intent_data", {})
    groq_key = pending_state.get("groq_key", "")
    mem0_context = pending_state.get("mem0_context", "")

    results = execute_all_commands(intent_data, groq_key, mem0_context)
    memory.add_entry(transcription, intent_data, results)

    all_intents = intent_data.get("intents", [])
    _store_in_mem0(transcription, results, all_intents)

    intent_badges = "  ".join([get_intent_display_name(i) for i in all_intents])
    confidence = intent_data.get("confidence", 0)
    intent_display = f"{intent_badges}\n\nConfidence: {confidence:.0%}"

    action_log, final_output = format_results(results)

    return (
        "✅ Executed!",
        transcription,
        intent_display,
        action_log,
        final_output,
        gr.update(visible=False),
        gr.update(visible=False)
    )


def cancel_execution():
    return (
        "🚫 Cancelled.",
        gr.update(), gr.update(),
        "❌ Execution cancelled by user.",
        "",
        gr.update(visible=False), gr.update(visible=False)
    )


def _store_in_mem0(transcription: str, results: list, intents: list):
    """Store the interaction in mem0 for long-term recall."""
    if not mem0.is_available:
        return
    try:
        response = ""
        for r in results:
            if r.get("output"):
                response += r["output"][:500] + "\n"
        primary_intent = intents[0] if intents else "unknown"
        mem0.add(transcription, response.strip(), intent=primary_intent)
    except Exception as e:
        print(f"[mem0] store error: {e}")


def format_results(results: list) -> tuple:
    action_lines = []
    output_parts = []
    for i, r in enumerate(results, 1):
        status_icon = "✅" if r.get("status") == "success" else "❌"
        action = r.get("action", "Unknown action")
        action_lines.append(f"{status_icon} [{i}] {action}")
        if r.get("filepath"):
            action_lines.append(f"     📁 Saved: {r['filepath']}")
        if r.get("status") == "error":
            action_lines.append(f"     ⚠️ Error: {r.get('output', '')}")
        else:
            output = r.get("output", "")
            if output:
                output_parts.append(f"### {action}\n\n{output}")
    action_log = "\n".join(action_lines)
    final_output = "\n\n---\n\n".join(output_parts)
    return action_log, final_output


def refresh_file_list():
    result = execute_list_files()
    return result.get("output", "")


def get_session_stats():
    stats = memory.get_stats()
    mem0_status = "🟢 Connected" if mem0.is_available else "⚪ Not configured"
    return (
        f"**Commands:** {stats['total_commands']}  |  "
        f"**Successful:** {stats['successful']}  |  "
        f"**Files Created:** {stats['files_created']}  |  "
        f"**Session Time:** {stats['session_duration']}  |  "
        f"**Mem0:** {mem0_status}"
    )


def get_history_table():
    return memory.get_history_display()


def clear_session():
    memory.clear()
    return "🗑️ Session cleared.", []


def refresh_memories():
    """Refresh mem0 memories display."""
    return mem0.get_display_memories()


def clear_memories():
    """Clear all mem0 memories."""
    if mem0.delete_all():
        return "🗑️ All memories deleted.", mem0.get_display_memories()
    return "⚠️ Failed to delete memories (mem0 may not be configured).", mem0.get_display_memories()


def update_mem0_key(key: str):
    """Update the mem0 API key."""
    mem0.update_key(key)
    if mem0.is_available:
        return "🟢 Mem0 connected! Memories will persist across sessions."
    elif key.strip():
        return f"⚠️ {mem0._init_error}"
    else:
        return "⚪ Mem0 not configured. Enter your API key to enable persistent memory."


# ─────────────────────────────────────────────
# STITCH-INSPIRED NEON DARK THEME CSS
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans+Text:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #121212;
    --surface: #1e1e1e;
    --surface2: #252525;
    --surface3: #2c2c2c;
    --border: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.14);
    --cyan: #00E5FF;
    --cyan-dim: rgba(0, 229, 255, 0.10);
    --purple: #D500F9;
    --purple-dim: rgba(213, 0, 249, 0.10);
    --green: #69F0AE;
    --green-dim: rgba(105, 240, 174, 0.10);
    --orange: #FFAB40;
    --orange-dim: rgba(255, 171, 64, 0.10);
    --red: #FF5252;
    --pink: #FF4081;
    --pink-dim: rgba(255, 64, 129, 0.10);
    --text: #E8EAED;
    --text-secondary: #9AA0A6;
    --text-dim: #5F6368;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Google Sans Text', 'Google Sans', sans-serif;
    --radius: 20px;
    --radius-sm: 14px;
    --radius-xs: 10px;
}

* { box-sizing: border-box; }

html {
    background: var(--bg) !important;
}

body {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    min-height: 100vh;
    margin: 0;
}

.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

.main, .wrap, .contain, main, footer,
div[class*="footer"], .gr-form,
#component-0, .app {
    background: var(--bg) !important;
}
footer { display: none !important; }

/* ─── Header ─── */
.app-header {
    text-align: center;
    padding: 36px 20px 28px;
    margin-bottom: 28px;
    position: relative;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 180px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--purple), transparent);
}
.app-header h1 {
    font-family: var(--sans);
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
}
.app-header .subtitle {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-family: var(--mono);
    margin: 0 0 12px;
    letter-spacing: 0.02em;
}
.app-header .pipeline {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 16px;
    border-radius: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--text-secondary);
    letter-spacing: 0.04em;
}
.app-header .pipeline .s { color: var(--cyan); }
.app-header .pipeline .a { color: var(--text-dim); }

/* ─── Section Label ─── */
.sec-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
    padding-left: 2px;
}

/* ─── Feature Cards Grid ─── */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 4px;
}
.feature-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 18px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}
.feature-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.feature-card .card-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 12px;
}
.feature-card .card-title {
    font-family: var(--sans);
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text);
    margin-bottom: 4px;
}
.feature-card .card-desc {
    font-size: 0.76rem;
    color: var(--text-secondary);
    line-height: 1.4;
}
.icon-cyan { background: var(--cyan-dim); color: var(--cyan); }
.icon-purple { background: var(--purple-dim); color: var(--purple); }
.icon-green { background: var(--green-dim); color: var(--green); }
.icon-orange { background: var(--orange-dim); color: var(--orange); }
.icon-pink { background: var(--pink-dim); color: var(--pink); }

/* ─── Status Bar ─── */
.status-bar {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 18px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    min-height: 44px;
}

/* ─── Inputs & Textareas ─── */
textarea, .gr-text-input, input[type="text"], input[type="password"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.08) !important;
}

/* ─── Primary Button ─── */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, var(--cyan), #00B8D4) !important;
    border: none !important;
    color: #000 !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 24px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(0, 229, 255, 0.18) !important;
}
button.primary-btn:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0, 229, 255, 0.3) !important;
}

button:not(.primary-btn) {
    border-radius: var(--radius-xs) !important;
    font-family: var(--sans) !important;
    transition: all 0.2s ease !important;
}

/* ─── Accordion ─── */
.gr-accordion {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ─── Tabs ─── */
.gr-tab-item {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-xs) var(--radius-xs) 0 0 !important;
}

/* ─── Output Panel ─── */
.output-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: var(--radius-sm) !important;
    padding: 16px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

/* ─── Memory Panel ─── */
.memory-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--purple) !important;
    border-radius: var(--radius-sm) !important;
    padding: 16px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

/* ─── Stats ─── */
.stats-bar {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 16px !important;
    font-size: 0.78rem;
    color: var(--text-secondary);
}

.gr-audio { border-radius: var(--radius-sm) !important; }

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 24px 0 !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

label {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

/* ─── Architecture Diagram ─── */
.arch-diagram {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    line-height: 1.8;
    text-align: center;
}
.arch-diagram .node {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 10px;
    font-weight: 500;
    margin: 2px 0;
}
.arch-diagram .node-cyan {
    background: var(--cyan-dim);
    color: var(--cyan);
    border: 1px solid rgba(0,229,255,0.15);
}
.arch-diagram .node-purple {
    background: var(--purple-dim);
    color: var(--purple);
    border: 1px solid rgba(213,0,249,0.15);
}
.arch-diagram .node-green {
    background: var(--green-dim);
    color: var(--green);
    border: 1px solid rgba(105,240,174,0.15);
}
.arch-diagram .arrow {
    color: var(--text-dim);
    font-size: 0.9rem;
    margin: 0 6px;
}

/* ─── Mem0 status indicator ─── */
.mem0-status {
    font-family: var(--mono);
    font-size: 0.75rem;
    padding: 6px 12px;
    border-radius: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
}
"""

def create_app():
    with gr.Blocks(
        title="Voice Agent",
    ) as app:

        pending_state = gr.State({})

        # ── Header ───────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>⬡ Voice Agent</h1>
            <p class="subtitle">AI-Powered Voice Command System</p>
            <div class="pipeline">
                <span class="s">AUDIO</span>
                <span class="a">→</span>
                <span class="s">STT</span>
                <span class="a">→</span>
                <span class="s">MEMORY</span>
                <span class="a">→</span>
                <span class="s">INTENT</span>
                <span class="a">→</span>
                <span class="s">ACTION</span>
                <span class="a">→</span>
                <span class="s">OUTPUT</span>
            </div>
        </div>
        """)

        # ── Feature Cards (like AI Studio) ───────────────────
        gr.HTML("""
        <div class="sec-label">🧠 Capabilities</div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="card-icon icon-cyan">📄</div>
                <div class="card-title">Create File</div>
                <div class="card-desc">Create files and folders in the sandboxed output/ directory via voice.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon icon-purple">💻</div>
                <div class="card-title">Write Code</div>
                <div class="card-desc">Generate production-quality code in any language and save it automatically.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon icon-green">📝</div>
                <div class="card-title">Summarize Text</div>
                <div class="card-desc">Get concise summaries with TL;DR, key points, and main takeaways.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon icon-orange">💬</div>
                <div class="card-title">General Chat</div>
                <div class="card-desc">Ask questions and get conversational AI responses via voice.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon icon-cyan">📂</div>
                <div class="card-title">List Files</div>
                <div class="card-desc">Browse all generated files in the output folder with sizes and dates.</div>
            </div>
            <div class="feature-card">
                <div class="card-icon icon-pink">🧠</div>
                <div class="card-title">Persistent Memory</div>
                <div class="card-desc">Remembers your preferences across sessions via mem0 — learns as you use it.</div>
            </div>
        </div>
        """)

        gr.HTML('<hr>')

        # ── Main Layout ──────────────────────────────────────
        with gr.Row():
            # LEFT COLUMN — Input & Config
            with gr.Column(scale=1, min_width=340):

                gr.HTML('<div class="sec-label">⚡ Configuration</div>')
                with gr.Accordion("🔑 API Keys", open=True):
                    groq_key = gr.Textbox(
                        label="Groq API Key (STT + LLM)",
                        placeholder="gsk_...",
                        type="password",
                        value=os.environ.get("GROQ_API_KEY", ""),
                        info="Single key for Whisper STT + Llama LLM — free at console.groq.com"
                    )
                    mem0_key = gr.Textbox(
                        label="Mem0 API Key (Persistent Memory)",
                        placeholder="m0-...",
                        type="password",
                        value=os.environ.get("MEM0_API_KEY", ""),
                        info="Optional — enables memory across sessions. Free at app.mem0.ai"
                    )
                    mem0_status = gr.Textbox(
                        value=update_mem0_key(os.environ.get("MEM0_API_KEY", "")),
                        interactive=False,
                        show_label=False,
                        lines=1,
                        elem_classes=["mem0-status"],
                    )

                gr.HTML('<div class="sec-label" style="margin-top:18px">🎤 Audio Input</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or Upload Audio",
                    format="wav",
                    interactive=True,
                )

                require_confirm = gr.Checkbox(
                    label="🛡️ Human-in-the-Loop (confirm before file ops)",
                    value=True,
                    info="Require confirmation before creating/writing files."
                )

                run_btn = gr.Button(
                    "▶  Process Audio",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

                with gr.Row(visible=False) as confirm_row:
                    confirm_btn = gr.Button("✅ Confirm & Execute", variant="primary", size="sm")
                    cancel_btn = gr.Button("❌ Cancel", variant="stop", size="sm")

                # Architecture diagram
                gr.HTML("""
                <div style="margin-top:18px;">
                    <div class="sec-label">🏗️ Architecture</div>
                    <div class="arch-diagram">
                        <span class="node node-cyan">🎙️ Mic / Upload</span>
                        <br>
                        <span class="arrow">↓</span>
                        <br>
                        <span class="node node-cyan">Groq Whisper v3</span>
                        <span style="font-size:0.65rem;color:var(--text-dim);"> STT</span>
                        <br>
                        <span class="arrow">↓</span>
                        <br>
                        <span class="node node-purple">🧠 mem0</span>
                        <span style="font-size:0.65rem;color:var(--text-dim);"> Context</span>
                        <br>
                        <span class="arrow">↓</span>
                        <br>
                        <span class="node node-purple">Llama 3.3 70B</span>
                        <span style="font-size:0.65rem;color:var(--text-dim);"> Intent</span>
                        <br>
                        <span class="arrow">↓</span>
                        <br>
                        <span class="node node-green">Tool Executor</span>
                        <span style="font-size:0.65rem;color:var(--text-dim);"> Action</span>
                        <br>
                        <span class="arrow">↓</span>
                        <br>
                        <span class="node node-cyan">📂 output/</span>
                    </div>
                </div>
                """)

            # RIGHT COLUMN — Output
            with gr.Column(scale=2):

                status_box = gr.Textbox(
                    label="",
                    value="⏳ Ready — Record or upload audio to begin.",
                    interactive=False,
                    elem_classes=["status-bar"],
                    show_label=False,
                    lines=1,
                )

                with gr.Tabs():
                    with gr.Tab("📋 Results"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                transcription_out = gr.Textbox(
                                    label="🎙️ Transcription",
                                    lines=2,
                                    interactive=False,
                                    placeholder="Transcribed speech will appear here..."
                                )
                            with gr.Column(scale=1):
                                intent_out = gr.Textbox(
                                    label="🧠 Detected Intent",
                                    lines=2,
                                    interactive=False,
                                    placeholder="Intent analysis will appear here..."
                                )
                        action_out = gr.Textbox(
                            label="⚙️ Actions Taken",
                            lines=3,
                            interactive=False,
                            placeholder="Execution log will appear here..."
                        )
                        output_out = gr.Textbox(
                            label="📤 Output / Result",
                            lines=12,
                            interactive=False,
                            placeholder="Generated content, summaries, or responses will appear here...",
                        )

                    with gr.Tab("🧠 Memory"):
                        gr.HTML("""
                        <div style="margin-bottom:12px;">
                            <div class="sec-label">🧠 Persistent Memory — powered by mem0</div>
                            <p style="font-size:0.78rem;color:var(--text-secondary);margin:0;line-height:1.5;">
                                Mem0 automatically extracts facts and preferences from your conversations.
                                These memories are used to personalize future responses — the agent learns
                                your coding style, language preferences, and more over time.
                            </p>
                        </div>
                        """)
                        memory_display = gr.Markdown(
                            value=mem0.get_display_memories(),
                            elem_classes=["memory-panel"]
                        )
                        with gr.Row():
                            refresh_mem_btn = gr.Button("🔄 Refresh Memories", size="sm")
                            clear_mem_btn = gr.Button("🗑️ Clear All Memories", size="sm", variant="stop")
                        mem_action_status = gr.Textbox(
                            value="",
                            interactive=False,
                            show_label=False,
                            lines=1,
                            visible=False,
                        )

                    with gr.Tab("📂 Files"):
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            gr.HTML('<span style="font-size:0.75rem;color:var(--text-dim);font-family:var(--mono);">Sandboxed output/ folder</span>')
                        file_list_out = gr.Textbox(
                            label="",
                            lines=14,
                            interactive=False,
                            show_label=False,
                            value=refresh_file_list(),
                            elem_classes=["output-panel"]
                        )

                    with gr.Tab("📜 History"):
                        stats_out = gr.Markdown(
                            get_session_stats(),
                            elem_classes=["stats-bar"]
                        )
                        history_table = gr.Dataframe(
                            headers=["#", "Time", "Transcription", "Intents", "Status", "Files"],
                            datatype=["str"] * 6,
                            interactive=False,
                            value=get_history_table(),
                            wrap=True,
                        )
                        with gr.Row():
                            refresh_history_btn = gr.Button("🔄 Refresh History", size="sm")
                            clear_history_btn = gr.Button("🗑️ Clear Session", size="sm", variant="stop")

        # ── Example Commands ───────────────────────────────────
        gr.HTML('<hr>')
        gr.HTML('<div class="sec-label">💡 Example Voice Commands</div>')
        with gr.Row():
            gr.Examples(
                examples=[
                    ["Create a Python file with a retry decorator function"],
                    ["Write a JavaScript function to debounce events and save it"],
                    ["Summarize this text: AI is transforming every industry with new capabilities."],
                    ["Create a folder called experiments"],
                    ["List all files in the output folder"],
                    ["What is the difference between REST and GraphQL?"],
                ],
                inputs=[],
                label="",
            )

        # ── Wire up events ─────────────────────────────────────
        pipeline_outputs = [
            status_box, transcription_out, intent_out, action_out, output_out,
            pending_state, confirm_btn, cancel_btn
        ]

        run_btn.click(
            fn=run_pipeline,
            inputs=[audio_input, groq_key, mem0_key, require_confirm, pending_state],
            outputs=pipeline_outputs,
        )

        confirm_btn.click(
            fn=confirm_execution,
            inputs=[pending_state],
            outputs=[
                status_box, transcription_out, intent_out, action_out, output_out,
                confirm_btn, cancel_btn
            ]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[confirm_row]
        )

        cancel_btn.click(
            fn=cancel_execution,
            outputs=[
                status_box, transcription_out, intent_out, action_out, output_out,
                confirm_btn, cancel_btn
            ]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[confirm_row]
        )

        run_btn.click(
            fn=lambda pb: gr.update(visible=bool(pb)),
            inputs=[pending_state],
            outputs=[confirm_row],
            queue=False
        )

        # Mem0 key update
        mem0_key.change(
            fn=update_mem0_key,
            inputs=[mem0_key],
            outputs=[mem0_status],
        )

        # Memory tab
        refresh_mem_btn.click(fn=refresh_memories, outputs=[memory_display])
        clear_mem_btn.click(
            fn=clear_memories,
            outputs=[mem_action_status, memory_display]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[mem_action_status]
        )

        refresh_btn.click(fn=refresh_file_list, outputs=[file_list_out])
        refresh_history_btn.click(
            fn=lambda: (get_session_stats(), get_history_table()),
            outputs=[stats_out, history_table]
        )
        clear_history_btn.click(
            fn=clear_session,
            outputs=[status_box, history_table]
        )

    return app


if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  ⬡  Voice Agent — Starting up")
    print("═" * 52)
    print(f"  Output folder: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("  STT: Groq Whisper (whisper-large-v3)")
    print("  LLM: Groq Llama (llama-3.3-70b-versatile)")
    print(f"  Mem0: {'🟢 Connected' if mem0.is_available else '⚪ Not configured'}")
    print("  UI: http://localhost:7860")
    print("═" * 52 + "\n")

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="fuchsia",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Google Sans Text"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    )

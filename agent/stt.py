"""
Speech-to-Text module using Groq's Whisper API.

WHY GROQ INSTEAD OF LOCAL MODEL:
- Local Whisper models (whisper-large-v3) require 6-10GB VRAM and significant CPU/RAM.
- Groq offers Whisper-large-v3 via API with ~10x real-time speed at low latency.
- For production reliability and accessibility without GPU requirements, Groq is preferred.
- If running locally is required, swap `transcribe_with_groq` with `transcribe_local`.
"""

import os
import tempfile
from pathlib import Path


def transcribe_audio(audio_path: str, groq_api_key: str) -> dict:
    """
    Transcribe audio file to text using Groq Whisper API.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, .m4a, etc.)
        groq_api_key: Groq API key
    
    Returns:
        dict with 'text', 'language', 'duration', 'model', 'status'
    """
    if not audio_path or not os.path.exists(audio_path):
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model": "whisper-large-v3",
            "status": "error",
            "error": "Audio file not found."
        }

    if not groq_api_key or not groq_api_key.strip():
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model": "whisper-large-v3",
            "status": "error",
            "error": "Groq API key is missing."
        }

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key.strip())

        file_size = os.path.getsize(audio_path)
        if file_size > 25 * 1024 * 1024:  # 25MB limit
            return {
                "text": "",
                "language": "unknown",
                "duration": 0,
                "model": "whisper-large-v3",
                "status": "error",
                "error": "Audio file exceeds 25MB limit."
            }

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(Path(audio_path).name, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                language=None,  # auto-detect
                temperature=0.0,
            )

        text = transcription.text.strip() if hasattr(transcription, 'text') else ""
        language = getattr(transcription, 'language', 'unknown')
        duration = getattr(transcription, 'duration', 0)

        if not text:
            return {
                "text": "",
                "language": language,
                "duration": duration,
                "model": "whisper-large-v3",
                "status": "error",
                "error": "Could not transcribe audio. The audio may be silent or unintelligible."
            }

        return {
            "text": text,
            "language": language,
            "duration": round(duration, 2) if duration else 0,
            "model": "whisper-large-v3",
            "status": "success"
        }

    except ImportError:
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model": "whisper-large-v3",
            "status": "error",
            "error": "groq package not installed. Run: pip install groq"
        }
    except Exception as e:
        error_msg = str(e)
        if "auth" in error_msg.lower() or "api_key" in error_msg.lower() or "401" in error_msg:
            error_msg = "Invalid Groq API key. Please check your key at console.groq.com."
        elif "rate" in error_msg.lower():
            error_msg = "Rate limit hit. Please wait a moment and try again."
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model": "whisper-large-v3",
            "status": "error",
            "error": f"Transcription failed: {error_msg}"
        }


def transcribe_local(audio_path: str) -> dict:
    """
    Fallback: Transcribe using local Whisper model (requires transformers + torch).
    Only use if Groq is unavailable.
    """
    try:
        from transformers import pipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",  # smaller model for local use
            device=device,
        )
        result = pipe(audio_path, return_timestamps=False)
        return {
            "text": result["text"].strip(),
            "language": "unknown",
            "duration": 0,
            "model": "whisper-base (local)",
            "status": "success"
        }
    except Exception as e:
        return {
            "text": "",
            "language": "unknown",
            "duration": 0,
            "model": "whisper-base (local)",
            "status": "error",
            "error": str(e)
        }

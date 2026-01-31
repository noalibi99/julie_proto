# Julie - French Insurance Voice Assistant

AI-powered voice assistant for CNP Assurances (French life insurance).

## Overview

Julie handles customer calls with:
- **Voice Pipeline**: WebRTC VAD → Groq Whisper STT → Groq LLM → ElevenLabs/gTTS
- **RAG**: Qdrant vector store for insurance knowledge retrieval
- **Claims**: Lookup status, file new claims with guided conversation
- **Intent Classification**: Smart routing (greeting, claim status, file claim, transfer, etc.)
- **Web Interface**: Admin panel + push-to-talk voice call simulation

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd julieee
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API (STT + LLM) |
| `ELEVENLABS_API_KEY` | No | ElevenLabs TTS (falls back to gTTS) |

## Run

```bash
source venv/bin/activate

# CLI voice assistant
python main.py

# Web interface (admin + voice)
python run_web.py
# Open http://localhost:8000
```

## Project Structure

```
julie/
├── core/           # Agent orchestration, intents, logging
├── audio/          # WebRTC VAD
├── stt/            # Groq Whisper
├── llm/            # Groq LLM + prompts
├── tts/            # ElevenLabs / gTTS
├── rag/            # Qdrant + LangChain retrieval
├── claims/         # Claims database + filing flow
└── web/            # FastAPI backend + static frontend
```

## Tests

```bash
pytest -v
```

## Web Interface

- **Admin** (`/admin`): Upload docs, view stats, manage claims
- **Voice** (`/voice`): Push-to-talk call simulation with Julie

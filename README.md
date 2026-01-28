# Julie - French Insurance Voice Assistant

AI-powered voice assistant for a French insurance company.

## Architecture

```
julie/
├── config.py           # Configuration management
├── core/
│   └── agent.py        # Main orchestration layer
├── audio/
│   └── vad.py          # Voice Activity Detection (WebRTC)
├── stt/
│   └── providers.py    # Speech-to-Text (Groq Whisper)
├── llm/
│   ├── providers.py    # Language Models (Groq)
│   └── prompts.py      # System prompts
├── tts/
│   └── providers.py    # Text-to-Speech (ElevenLabs/gTTS)
└── interfaces/
    ├── cli.py          # Command line interface
    └── telephony.py    # Asterisk integration (coming soon)
```

## Features

- **STT**: Groq Whisper (whisper-large-v3-turbo)
- **VAD**: WebRTC VAD for reliable speech detection
- **LLM**: Groq (llama-3.3-70b-versatile) with insurance knowledge
- **TTS**: ElevenLabs (natural) or gTTS (fallback)

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Run

```bash
source venv/bin/activate
python main.py              # Run CLI interface
python main.py --help       # Show options
python julie.py             # Legacy single-file version
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for STT and LLM |
| `ELEVENLABS_API_KEY` | No | ElevenLabs API key (falls back to gTTS) |

## Usage

Just speak! Julie will:
1. Listen for your voice (WebRTC VAD detects speech)
2. Transcribe with Whisper
3. Respond using the LLM
4. Speak the response back

Say "au revoir" to exit.

## Roadmap

- [x] Core pipeline (STT → LLM → TTS)
- [x] WebRTC VAD for speech detection
- [x] ElevenLabs professional voice
- [x] Modular architecture
- [ ] RAG knowledge base (Qdrant)
- [ ] Telephony integration (Asterisk)
- [ ] Backend API integration

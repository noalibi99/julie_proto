# Julie - French Insurance Voice Assistant

AI-powered voice assistant for a French insurance company.

## Features

- **STT**: Groq Whisper (whisper-large-v3-turbo)
- **VAD**: WebRTC VAD for reliable speech detection
- **LLM**: Groq (llama-3.3-70b-versatile) with insurance knowledge
- **TTS**: gTTS (French)

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
python julie.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (get it at https://console.groq.com) |

## Usage

Just speak! Julie will:
1. Listen for your voice (WebRTC VAD detects speech)
2. Transcribe with Whisper
3. Respond using the LLM
4. Speak the response back

Say "au revoir" to exit.

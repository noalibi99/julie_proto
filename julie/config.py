"""
Configuration management for Julie.

All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AudioConfig:
    """Audio settings."""
    sample_rate: int = 16000
    channels: int = 1
    silence_duration: float = 1.5
    min_speech_duration: float = 0.3
    max_record_duration: float = 30.0
    vad_aggressiveness: int = 2  # 0-3


@dataclass
class STTConfig:
    """Speech-to-Text settings."""
    provider: Literal["groq"] = "groq"
    model: str = "whisper-large-v3-turbo"
    language: str = "fr"
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))


@dataclass
class LLMConfig:
    """LLM settings."""
    provider: Literal["groq"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 150
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))


@dataclass
class TTSConfig:
    """Text-to-Speech settings."""
    provider: Literal["elevenlabs", "gtts"] = field(
        default_factory=lambda: "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "gtts"
    )
    language: str = "fr"
    voice: str = "charlotte"  # ElevenLabs voice
    model: str = "eleven_flash_v2_5"  # Cheapest with French
    api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    
    # Voice settings for professional tone
    stability: float = 0.7
    similarity_boost: float = 0.8
    style: float = 0.0


@dataclass
class RAGConfig:
    """RAG/Knowledge Base settings (for future use)."""
    enabled: bool = False
    provider: Literal["qdrant"] = "qdrant"
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection: str = "insurance_knowledge"


@dataclass
class TelephonyConfig:
    """Telephony settings (for future use)."""
    enabled: bool = False
    provider: Literal["asterisk"] = "asterisk"
    host: str = field(default_factory=lambda: os.getenv("ASTERISK_HOST", "localhost"))
    port: int = 5038


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    telephony: TelephonyConfig = field(default_factory=TelephonyConfig)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.stt.api_key:
            errors.append("GROQ_API_KEY not set (required for STT)")
        if not self.llm.api_key:
            errors.append("GROQ_API_KEY not set (required for LLM)")
        if self.tts.provider == "elevenlabs" and not self.tts.api_key:
            errors.append("ELEVENLABS_API_KEY not set but elevenlabs selected")
        
        return errors

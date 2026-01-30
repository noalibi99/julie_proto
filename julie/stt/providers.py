"""
Speech-to-Text providers.
"""

import io
from abc import ABC, abstractmethod
from pathlib import Path

from groq import Groq

from julie.config import STTConfig


class BaseSTT(ABC):
    """Abstract base class for STT providers."""
    
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text."""
        pass
    
    def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio from a file path."""
        with open(file_path, "rb") as f:
            return self.transcribe(f.read())


class GroqSTT(BaseSTT):
    """Speech-to-Text using Groq Whisper API."""
    
    def __init__(self, config: STTConfig | None = None):
        self.config = config or STTConfig()
        
        if not self.config.api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        
        self.client = Groq(api_key=self.config.api_key)
        self.model = self.config.model
        self.language = self.config.language
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            result = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=self.language,
                response_format="text",
                temperature=0.0,
            )
            
            text = result.strip() if isinstance(result, str) else str(result).strip()
            return text
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""
    
    def transcribe_file(self, file_path: str) -> str:
        """Transcribe audio from a file path (supports various formats)."""
        try:
            path = Path(file_path)
            
            with open(file_path, "rb") as f:
                audio_file = io.BytesIO(f.read())
                # Use actual filename extension for format detection
                audio_file.name = f"audio{path.suffix}"
            
            result = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=self.language,
                response_format="text",
                temperature=0.0,
            )
            
            text = result.strip() if isinstance(result, str) else str(result).strip()
            return text
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""


def get_stt(config: STTConfig | None = None) -> BaseSTT:
    """Factory function to get STT instance."""
    config = config or STTConfig()
    
    if config.provider == "groq":
        return GroqSTT(config)
    else:
        raise ValueError(f"Unknown STT provider: {config.provider}")

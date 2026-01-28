"""
Text-to-Speech providers.
"""

import os
import tempfile
from abc import ABC, abstractmethod

from gtts import gTTS

from julie.config import TTSConfig


class BaseTTS(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (for telephony)."""
        pass


class GTTSProvider(BaseTTS):
    """Text-to-Speech using gTTS (free, robotic)."""
    
    def __init__(self, config: TTSConfig | None = None):
        self.config = config or TTSConfig()
        self.language = self.config.language
    
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)
                temp_path = f.name
            
            os.system(f"afplay {temp_path} 2>/dev/null")
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"[gTTS Error] {e}")
    
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        try:
            import io
            tts = gTTS(text=text, lang=self.language, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            print(f"[gTTS Error] {e}")
            return b""


class ElevenLabsProvider(BaseTTS):
    """Text-to-Speech using ElevenLabs (natural, paid)."""
    
    VOICES = {
        "charlotte": "XB0fDUnXU5powFXDhCwa",  # Charlotte - French female
        "thomas": "GBv7mTt0atIp3Br8iCZE",     # Thomas - French male
        "default": "XB0fDUnXU5powFXDhCwa",
    }
    
    def __init__(self, config: TTSConfig | None = None):
        self.config = config or TTSConfig()
        
        if not self.config.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        
        self.api_key = self.config.api_key
        self.voice_id = self.VOICES.get(self.config.voice, self.VOICES["default"])
        self.model = self.config.model
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        
        # Voice settings
        self.stability = self.config.stability
        self.similarity_boost = self.config.similarity_boost
        self.style = self.config.style
    
    def _call_api(self, text: str) -> bytes:
        """Call ElevenLabs API and return audio bytes."""
        import urllib.request
        import json
        import ssl
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        
        data = json.dumps({
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": True,
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(self.api_url, data=data, headers=headers)
        
        # SSL context for macOS
        ssl_context = ssl.create_default_context()
        try:
            import certifi
            ssl_context.load_verify_locations(certifi.where())
        except ImportError:
            ssl_context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(req, context=ssl_context) as response:
            return response.read()
    
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        try:
            audio_data = self._call_api(text)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            os.system(f"afplay {temp_path} 2>/dev/null")
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"[ElevenLabs Error] {e}")
    
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        try:
            return self._call_api(text)
        except Exception as e:
            print(f"[ElevenLabs Error] {e}")
            return b""


def get_tts(config: TTSConfig | None = None) -> BaseTTS:
    """Factory function to get TTS instance."""
    config = config or TTSConfig()
    
    if config.provider == "elevenlabs":
        print("Using ElevenLabs TTS (natural voice)")
        return ElevenLabsProvider(config)
    else:
        print("Using gTTS (set ELEVENLABS_API_KEY for natural voice)")
        return GTTSProvider(config)

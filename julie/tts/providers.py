"""
Text-to-Speech providers with streaming support.
"""

import os
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Iterator, Callable

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
    
    def speak_streaming(self, text: str) -> None:
        """Stream audio playback for lower latency (override if supported)."""
        # Default: fall back to non-streaming
        self.speak(text)


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
    """Text-to-Speech using ElevenLabs (natural, paid) with streaming support."""
    
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
        self.stream_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        
        # Voice settings
        self.stability = self.config.stability
        self.similarity_boost = self.config.similarity_boost
        self.style = self.config.style
        
        # Streaming buffer size
        self.chunk_size = 4096
    
    def _get_headers(self) -> dict:
        """Get API headers."""
        return {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
    
    def _get_payload(self, text: str) -> bytes:
        """Get API payload."""
        import json
        return json.dumps({
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": True,
            }
        }).encode("utf-8")
    
    def _get_ssl_context(self):
        """Get SSL context for API calls."""
        import ssl
        ssl_context = ssl.create_default_context()
        try:
            import certifi
            ssl_context.load_verify_locations(certifi.where())
        except ImportError:
            ssl_context = ssl._create_unverified_context()
        return ssl_context
    
    def _call_api(self, text: str) -> bytes:
        """Call ElevenLabs API and return audio bytes."""
        import urllib.request
        
        req = urllib.request.Request(
            self.api_url, 
            data=self._get_payload(text), 
            headers=self._get_headers()
        )
        
        with urllib.request.urlopen(req, context=self._get_ssl_context()) as response:
            return response.read()
    
    def _stream_api(self, text: str) -> Iterator[bytes]:
        """Stream audio chunks from ElevenLabs API."""
        import urllib.request
        
        req = urllib.request.Request(
            self.stream_url,
            data=self._get_payload(text),
            headers=self._get_headers()
        )
        
        with urllib.request.urlopen(req, context=self._get_ssl_context()) as response:
            while True:
                chunk = response.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
    
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
    
    def speak_streaming(self, text: str) -> None:
        """
        Stream audio with lower latency.
        
        Collects chunks and starts playback as soon as possible.
        Uses threading to play audio while still receiving data.
        """
        try:
            chunks = []
            first_chunk_received = threading.Event()
            all_chunks_received = threading.Event()
            temp_path = None
            
            def receive_chunks():
                """Background thread to receive audio chunks."""
                nonlocal temp_path
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, mode='wb') as f:
                        temp_path = f.name
                        chunk_count = 0
                        for chunk in self._stream_api(text):
                            f.write(chunk)
                            chunk_count += 1
                            # Signal after receiving enough data to start playback
                            if chunk_count >= 2 and not first_chunk_received.is_set():
                                f.flush()
                                first_chunk_received.set()
                except Exception as e:
                    print(f"[Streaming Error] {e}")
                finally:
                    all_chunks_received.set()
            
            # Start receiving in background
            receiver = threading.Thread(target=receive_chunks, daemon=True)
            receiver.start()
            
            # Wait for first chunks (with timeout)
            first_chunk_received.wait(timeout=5.0)
            
            # Wait for all chunks before playing (simpler, more reliable)
            all_chunks_received.wait(timeout=30.0)
            
            if temp_path and os.path.exists(temp_path):
                os.system(f"afplay {temp_path} 2>/dev/null")
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"[ElevenLabs Streaming Error] {e}")
            # Fallback to non-streaming
            self.speak(text)
    
    def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        try:
            return self._call_api(text)
        except Exception as e:
            print(f"[ElevenLabs Error] {e}")
            return b""
    
    def synthesize_streaming(self, text: str, on_chunk: Callable[[bytes], None]) -> None:
        """
        Stream audio bytes for real-time telephony.
        
        Args:
            text: Text to synthesize
            on_chunk: Callback called for each audio chunk
        """
        try:
            for chunk in self._stream_api(text):
                on_chunk(chunk)
        except Exception as e:
            print(f"[ElevenLabs Streaming Error] {e}")


def get_tts(config: TTSConfig | None = None) -> BaseTTS:
    """Factory function to get TTS instance."""
    config = config or TTSConfig()
    
    if config.provider == "elevenlabs":
        print("Using ElevenLabs TTS (natural voice)")
        return ElevenLabsProvider(config)
    else:
        print("Using gTTS (set ELEVENLABS_API_KEY for natural voice)")
        return GTTSProvider(config)

#!/usr/bin/env python3
"""
Julie Voice Assistant - Minimal Implementation

STT: Groq Whisper (whisper-large-v3-turbo - cheapest)
VAD: WebRTC VAD for reliable speech detection
LLM: Groq with generic insurance knowledge
TTS: ElevenLabs (natural) or gTTS (fallback)
"""

import os
import io
import wave
import tempfile
import collections
from dotenv import load_dotenv

import numpy as np
import sounddevice as sd
import webrtcvad
from groq import Groq
from gtts import gTTS

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# VAD Configuration
SILENCE_DURATION = 1.5       # Seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.3    # Minimum speech before stopping allowed
MAX_RECORD_DURATION = 30.0   # Maximum recording time

# ============================================================
# STT - Groq Whisper
# ============================================================

class STT:
    """Speech-to-Text using Groq Whisper API."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = "whisper-large-v3-turbo"  # Cheapest Whisper model
    
    def transcribe(self, audio_bytes: bytes, language: str = "fr") -> str:
        """Transcribe WAV audio bytes to text."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            result = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=language,
                response_format="text",
                temperature=0.0,
            )
            
            text = result.strip() if isinstance(result, str) else str(result).strip()
            return text
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

# ============================================================
# VAD - WebRTC Voice Activity Detection
# ============================================================

class VAD:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        aggressiveness: int = 2,  # 0-3, higher = more aggressive filtering
        silence_duration: float = SILENCE_DURATION,
        min_speech_duration: float = MIN_SPEECH_DURATION,
        max_duration: float = MAX_RECORD_DURATION,
    ):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_duration = max_duration
        
        # WebRTC VAD requires 16kHz, 32kHz, or 48kHz
        # and frame sizes of 10, 20, or 30 ms
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = 30  # 30ms frames
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
    
    def record(self) -> bytes | None:
        """Record audio with WebRTC VAD."""
        print("🎤 Listening... (speak now)")
        
        # Ring buffer to keep audio before speech starts
        num_padding_frames = int(0.3 * 1000 / self.frame_duration_ms)  # 300ms padding
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        
        # Recording state
        triggered = False
        voiced_frames = []
        silence_frame_count = 0
        speech_frame_count = 0
        
        frames_for_silence = int(self.silence_duration * 1000 / self.frame_duration_ms)
        min_speech_frames = int(self.min_speech_duration * 1000 / self.frame_duration_ms)
        max_frames = int(self.max_duration * 1000 / self.frame_duration_ms)
        
        frame_count = 0
        is_done = False
        
        def callback(indata, frames, time_info, status):
            nonlocal triggered, silence_frame_count, speech_frame_count, frame_count, is_done
            
            if is_done:
                raise sd.CallbackStop()
            
            # Get frame as bytes (int16)
            frame = indata[:, 0].tobytes()
            frame_count += 1
            
            # Check if speech using webrtcvad
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                is_speech = False
            
            if not triggered:
                ring_buffer.append(frame)
                if is_speech:
                    print("▓", end="", flush=True)
                    speech_frame_count += 1
                    # Need a few consecutive speech frames to trigger
                    if speech_frame_count >= 3:
                        triggered = True
                        # Add ring buffer (audio before speech)
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
                    print("░", end="", flush=True)
                    speech_frame_count = 0
            else:
                voiced_frames.append(frame)
                if is_speech:
                    print("▓", end="", flush=True)
                    silence_frame_count = 0
                else:
                    print("░", end="", flush=True)
                    silence_frame_count += 1
                    
                    if silence_frame_count >= frames_for_silence:
                        print("\n✓ Done")
                        is_done = True
                        raise sd.CallbackStop()
            
            if frame_count >= max_frames:
                print("\n[Max duration]")
                is_done = True
                raise sd.CallbackStop()
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype='int16',
                callback=callback,
                blocksize=self.frame_size
            ):
                while not is_done:
                    sd.sleep(50)
        except sd.CallbackStop:
            pass
        except Exception as e:
            print(f"\nError: {e}")
            return None
        
        print()
        
        if not voiced_frames:
            print("No speech detected")
            return None
        
        # Combine frames
        audio = b''.join(voiced_frames)
        audio_array = np.frombuffer(audio, dtype=np.int16)
        duration = len(audio_array) / self.sample_rate
        print(f"Recorded {duration:.1f}s")
        
        return self._to_wav(audio_array)
    
    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        
        buf.seek(0)
        return buf.read()

# ============================================================
# LLM - Groq with generic insurance knowledge
# ============================================================

SYSTEM_PROMPT = """Tu es Julie, une assistante vocale pour une compagnie d'assurance française.

RÈGLES:
- Réponds TOUJOURS en français
- Sois concise (2-3 phrases max, c'est pour la voix)
- Sois professionnelle et empathique
- Si tu ne sais pas, dis-le honnêtement

INFORMATIONS GÉNÉRALES SUR L'ASSURANCE:

TYPES DE CONTRATS:
- Assurance vie: épargne et transmission de patrimoine
- Assurance habitation: protection du logement et des biens
- Assurance auto: couverture véhicule et responsabilité civile
- Assurance santé: complémentaire aux remboursements Sécurité Sociale

SINISTRES (Déclaration):
- Délai de déclaration: 5 jours ouvrés (2 jours pour vol)
- Documents nécessaires: constat amiable, photos, factures
- Numéro sinistre: fourni après déclaration

CONTACTS:
- Service client: 01 23 45 67 89
- Urgences sinistres: 0 800 123 456 (gratuit 24h/24)
- Email: contact@assurance.fr

HORAIRES:
- Lundi-Vendredi: 8h-20h
- Samedi: 9h-17h
- Dimanche: fermé

FAQ:
Q: Comment déclarer un sinistre?
R: Appelez le 0 800 123 456 ou connectez-vous à votre espace client.

Q: Quand serai-je remboursé?
R: Généralement sous 30 jours après réception du dossier complet.

Q: Comment modifier mon contrat?
R: Contactez votre conseiller ou faites-le depuis l'espace client.

Q: Comment résilier mon contrat?
R: Envoyez une lettre recommandée 2 mois avant l'échéance annuelle.
"""

class LLM:
    """LLM using Groq with generic insurance knowledge."""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.conversation_history = []
    
    def respond(self, user_text: str) -> str:
        """Generate a response to user input."""
        try:
            # Build messages
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history (last 6 messages)
            messages.extend(self.conversation_history[-6:])
            
            # Add current message
            messages.append({"role": "user", "content": user_text})
            
            # Call Groq
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Update history
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": reply})
            
            return reply
            
        except Exception as e:
            print(f"[LLM Error] {e}")
            return "Désolée, je n'ai pas pu traiter votre demande."
    
    def welcome(self) -> str:
        """Return welcome message."""
        return "Bonjour, je suis Julie, votre assistante AssuranceVie. Comment puis-je vous aider?"

# ============================================================
# TTS - Text-to-Speech (ElevenLabs or gTTS)
# ============================================================

class TTSgTTS:
    """Text-to-Speech using gTTS (free, robotic)."""
    
    def __init__(self, language: str = "fr"):
        self.language = language
    
    def speak(self, text: str):
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


class TTSElevenLabs:
    """Text-to-Speech using ElevenLabs (natural, paid)."""
    
    # French voice IDs from ElevenLabs
    VOICES = {
        "charlotte": "XB0fDUnXU5powFXDhCwa",  # Charlotte - French female
        "thomas": "GBv7mTt0atIp3Br8iCZE",     # Thomas - French male  
        "default": "XB0fDUnXU5powFXDhCwa",    # Default to Charlotte
    }
    
    def __init__(self, voice: str = "default"):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        
        self.voice_id = self.VOICES.get(voice, self.VOICES["default"])
        self.model = "eleven_flash_v2_5"  # Cheapest model with French support
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        import urllib.request
        import json
        import ssl
        
        try:
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            }
            
            data = json.dumps({
                "text": text,
                "model_id": self.model,
                "voice_settings": {
                    "stability": 0.7,           # Higher = more consistent, professional
                    "similarity_boost": 0.8,    # Voice clarity
                    "style": 0.0,               # Lower = less expressive, more professional
                    "use_speaker_boost": True,  # Clearer audio
                }
            }).encode("utf-8")
            
            req = urllib.request.Request(self.api_url, data=data, headers=headers)
            
            # Create SSL context (fixes macOS certificate issue)
            ssl_context = ssl.create_default_context()
            try:
                import certifi
                ssl_context.load_verify_locations(certifi.where())
            except ImportError:
                # Fallback: use unverified context (less secure but works)
                ssl_context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(req, context=ssl_context) as response:
                audio_data = response.read()
            
            # Save and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            os.system(f"afplay {temp_path} 2>/dev/null")
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"[ElevenLabs Error] {e}")


def get_tts() -> TTSElevenLabs | TTSgTTS:
    """Get TTS engine: ElevenLabs if API key set, else gTTS."""
    if os.getenv("ELEVENLABS_API_KEY"):
        print("Using ElevenLabs TTS (natural voice)")
        return TTSElevenLabs()
    else:
        print("Using gTTS (set ELEVENLABS_API_KEY for natural voice)")
        return TTSgTTS()

# ============================================================
# MAIN - Julie Voice Assistant
# ============================================================

def main():
    print("=" * 50)
    print("  JULIE - Voice Assistant")
    print("=" * 50)
    print()
    
    # Initialize components
    print("Initializing...")
    stt = STT()
    vad = VAD()
    llm = LLM()
    tts = get_tts()
    print("Ready!\n")
    
    # Welcome
    welcome = llm.welcome()
    print(f"Julie: {welcome}")
    tts.speak(welcome)
    
    # Main loop
    while True:
        try:
            print("\n" + "-" * 40)
            
            # Record with VAD
            audio = vad.record()
            if audio is None:
                continue
            
            # Transcribe
            print("Transcribing...")
            text = stt.transcribe(audio)
            
            if not text or text in [".", "...", " "]:
                print("Could not understand, try again")
                continue
            
            print(f"You: {text}")
            
            # Check for exit
            exit_words = ["au revoir", "bye", "quit", "exit", "stop"]
            if any(word in text.lower() for word in exit_words):
                goodbye = "Au revoir!"
                print(f"Julie: {goodbye}")
                tts.speak(goodbye)
                break
            
            # Generate response
            response = llm.respond(text)
            print(f"Julie: {response}")
            tts.speak(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()

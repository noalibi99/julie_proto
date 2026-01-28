"""
Voice Activity Detection using WebRTC VAD.
"""

import io
import wave
import collections
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
import webrtcvad

from julie.config import AudioConfig


class BaseVAD(ABC):
    """Abstract base class for VAD implementations."""
    
    @abstractmethod
    def record(self) -> bytes | None:
        """Record audio until silence detected. Returns WAV bytes or None."""
        pass


class WebRTCVAD(BaseVAD):
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        
        self.sample_rate = self.config.sample_rate
        self.channels = self.config.channels
        self.silence_duration = self.config.silence_duration
        self.min_speech_duration = self.config.min_speech_duration
        self.max_duration = self.config.max_record_duration
        
        # WebRTC VAD setup
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        self.frame_duration_ms = 30  # 30ms frames
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
    
    def record(self) -> bytes | None:
        """Record audio with WebRTC VAD."""
        print("🎤 Listening... (speak now)")
        
        # Ring buffer to keep audio before speech starts
        num_padding_frames = int(0.3 * 1000 / self.frame_duration_ms)
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
            
            frame = indata[:, 0].tobytes()
            frame_count += 1
            
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                is_speech = False
            
            if not triggered:
                ring_buffer.append(frame)
                if is_speech:
                    print("▓", end="", flush=True)
                    speech_frame_count += 1
                    if speech_frame_count >= 3:
                        triggered = True
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
                channels=self.channels,
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
        
        audio = b''.join(voiced_frames)
        audio_array = np.frombuffer(audio, dtype=np.int16)
        duration = len(audio_array) / self.sample_rate
        print(f"Recorded {duration:.1f}s")
        
        return self._to_wav(audio_array)
    
    def _to_wav(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        
        buf.seek(0)
        return buf.read()


def get_vad(config: AudioConfig | None = None) -> BaseVAD:
    """Factory function to get VAD instance."""
    return WebRTCVAD(config)

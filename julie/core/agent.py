"""
Julie Agent - Main orchestration layer.

The Agent coordinates all components (VAD, STT, LLM, TTS) and manages
the conversation flow. It can be used by different interfaces:
- CLI (current)
- Telephony/Asterisk (future)
- WebSocket API (future)
"""

from typing import Callable

from julie.config import Config
from julie.audio.vad import get_vad, BaseVAD
from julie.stt.providers import get_stt, BaseSTT
from julie.llm.providers import get_llm, BaseLLM
from julie.tts.providers import get_tts, BaseTTS


class Agent:
    """
    Julie Voice Agent - orchestrates the conversation flow.
    
    The agent is interface-agnostic and can be used by:
    - CLI interface (with microphone)
    - Telephony interface (with Asterisk audio streams)
    - WebSocket interface (with browser audio)
    """
    
    def __init__(
        self,
        config: Config | None = None,
        custom_stt: BaseSTT | None = None,
        custom_llm: BaseLLM | None = None,
        custom_tts: BaseTTS | None = None,
        custom_vad: BaseVAD | None = None,
    ):
        """
        Initialize the agent with optional custom components.
        
        Args:
            config: Configuration object (uses defaults if None)
            custom_stt: Override STT provider
            custom_llm: Override LLM provider
            custom_tts: Override TTS provider
            custom_vad: Override VAD provider
        """
        self.config = config or Config()
        
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Initialize components (use custom or create from config)
        self.stt = custom_stt or get_stt(self.config.stt)
        self.llm = custom_llm or get_llm(self.config.llm)
        self.tts = custom_tts or get_tts(self.config.tts)
        self.vad = custom_vad or get_vad(self.config.audio)
        
        # Exit words
        self.exit_words = ["au revoir", "bye", "quit", "exit", "stop"]
    
    def welcome(self) -> str:
        """Get welcome message."""
        return self.llm.welcome()
    
    def process_text(self, text: str) -> str:
        """
        Process text input and return response.
        
        This is the core brain logic, usable by any interface.
        """
        return self.llm.respond(text)
    
    def process_audio(self, audio_bytes: bytes) -> tuple[str, str]:
        """
        Process audio input and return (transcription, response).
        
        Args:
            audio_bytes: WAV audio bytes
            
        Returns:
            Tuple of (user_text, agent_response)
        """
        text = self.stt.transcribe(audio_bytes)
        
        if not text or text.strip() in [".", "...", " ", ""]:
            return "", ""
        
        response = self.llm.respond(text)
        return text, response
    
    def should_exit(self, text: str) -> bool:
        """Check if user wants to exit."""
        return any(word in text.lower() for word in self.exit_words)
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.llm.reset()
    
    def speak(self, text: str) -> None:
        """Speak text using TTS."""
        self.tts.speak(text)
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (for telephony)."""
        return self.tts.synthesize(text)
    
    def record(self) -> bytes | None:
        """Record audio using VAD."""
        return self.vad.record()


class ConversationHandler:
    """
    Handles a single conversation session.
    
    Provides hooks for different interfaces to customize behavior.
    """
    
    def __init__(
        self,
        agent: Agent,
        on_listening: Callable[[], None] | None = None,
        on_user_text: Callable[[str], None] | None = None,
        on_agent_text: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        self.agent = agent
        self.on_listening = on_listening or (lambda: None)
        self.on_user_text = on_user_text or (lambda t: None)
        self.on_agent_text = on_agent_text or (lambda t: None)
        self.on_error = on_error or (lambda e: None)
        
        self.running = False
    
    def start(self) -> None:
        """Start the conversation loop."""
        self.running = True
        
        # Welcome
        welcome = self.agent.welcome()
        self.on_agent_text(welcome)
        self.agent.speak(welcome)
        
        while self.running:
            try:
                self.on_listening()
                
                # Record
                audio = self.agent.record()
                if audio is None:
                    continue
                
                # Process
                text, response = self.agent.process_audio(audio)
                
                if not text:
                    self.on_error("Could not understand")
                    continue
                
                self.on_user_text(text)
                
                # Check exit
                if self.agent.should_exit(text):
                    goodbye = "Au revoir!"
                    self.on_agent_text(goodbye)
                    self.agent.speak(goodbye)
                    break
                
                # Respond
                self.on_agent_text(response)
                self.agent.speak(response)
                
            except KeyboardInterrupt:
                self.running = False
    
    def stop(self) -> None:
        """Stop the conversation loop."""
        self.running = False

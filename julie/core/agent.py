"""
Julie Agent - Main orchestration layer with RAG and Claims integration.

The Agent coordinates all components (VAD, STT, LLM, TTS, RAG, Claims) and manages
the conversation flow. It can be used by different interfaces:
- CLI (current)
- Telephony/Asterisk (future)
- WebSocket API (future)
"""

from typing import Callable, Optional
import re

from julie.config import Config
from julie.audio.vad import get_vad, BaseVAD
from julie.stt.providers import get_stt, BaseSTT
from julie.llm.providers import get_llm, BaseLLM
from julie.tts.providers import get_tts, BaseTTS
from julie.rag.retriever import RAGRetriever
from julie.claims.service import ClaimsService
from julie.claims.filing import ClaimFilingManager
from julie.core.intents import IntentClassifier, Intent
from julie.core.logging import CallLogger


class Agent:
    """
    Julie Voice Agent - orchestrates the conversation flow.
    
    The agent is interface-agnostic and can be used by:
    - CLI interface (with microphone)
    - Telephony interface (with Asterisk audio streams)
    - WebSocket interface (with browser audio)
    """
    
    # Pattern to detect claim ID in user message
    # Matches CLM followed by optional space/hyphen and alphanumeric ID
    CLAIM_ID_PATTERN = re.compile(r'CLM[\s-]*([A-Z0-9\s-]{3,15})', re.IGNORECASE)
    
    # Keywords that trigger claim filing flow
    FILING_KEYWORDS = [
        "déclarer", "declarer", "nouvelle demande", "nouveau sinistre",
        "enregistrer", "signaler", "faire une demande", "ouvrir un dossier",
        "déposer", "deposer", "créer une demande"
    ]
    
    def __init__(
        self,
        config: Config | None = None,
        custom_stt: BaseSTT | None = None,
        custom_llm: BaseLLM | None = None,
        custom_tts: BaseTTS | None = None,
        custom_vad: BaseVAD | None = None,
        enable_rag: bool = True,
        knowledge_dir: str = "knowledge/documents",
        enable_logging: bool = True,
        use_streaming_tts: bool = True,
    ):
        """
        Initialize the agent with optional custom components.
        
        Args:
            config: Configuration object (uses defaults if None)
            custom_stt: Override STT provider
            custom_llm: Override LLM provider
            custom_tts: Override TTS provider
            custom_vad: Override VAD provider
            enable_rag: Enable RAG knowledge retrieval
            knowledge_dir: Directory containing knowledge documents
            enable_logging: Enable call logging
            use_streaming_tts: Use streaming TTS for lower latency
        """
        self.config = config or Config()
        self.use_streaming_tts = use_streaming_tts
        
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Initialize core components (use custom or create from config)
        self.stt = custom_stt or get_stt(self.config.stt)
        self.llm = custom_llm or get_llm(self.config.llm)
        self.tts = custom_tts or get_tts(self.config.tts)
        self.vad = custom_vad or get_vad(self.config.audio)
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Initialize call logger
        self.call_logger: Optional[CallLogger] = None
        if enable_logging:
            self.call_logger = CallLogger()
        
        # Initialize RAG
        self.rag: Optional[RAGRetriever] = None
        if enable_rag:
            self._init_rag(knowledge_dir)
        
        # Initialize Claims service and filing manager
        self.claims = ClaimsService()
        self.claims.seed_mock_data()  # Seed mock claims for testing
        self.filing_manager = ClaimFilingManager()  # State machine for claim filing
        
        # Exit words
        self.exit_words = ["au revoir", "bye", "quit", "exit", "stop"]
        
        # Transfer keywords
        self.transfer_words = ["transférer", "transferer", "conseiller", "humain", "agent", "parler à quelqu'un"]
        
        # RAG relevance threshold
        self.rag_relevance_threshold = 0.35
    
    def _init_rag(self, knowledge_dir: str) -> None:
        """Initialize RAG with knowledge documents."""
        try:
            import os
            self.rag = RAGRetriever(in_memory=True)
            
            if os.path.exists(knowledge_dir):
                print(f"Loading knowledge from {knowledge_dir}...")
                count = self.rag.ingest_directory(knowledge_dir)
                print(f"Loaded {count} document chunks into RAG")
            else:
                print(f"Knowledge directory not found: {knowledge_dir}")
        except Exception as e:
            print(f"[RAG Init Error] {e}")
            self.rag = None
    
    def welcome(self) -> str:
        """Get welcome message."""
        return self.llm.welcome()
    
    def _detect_claim_id(self, text: str) -> Optional[str]:
        """Extract claim ID from user text if present."""
        match = self.CLAIM_ID_PATTERN.search(text.upper())
        if match:
            # Clean up: remove spaces/hyphens from the captured ID
            raw_id = match.group(1).replace(" ", "").replace("-", "")
            return f"CLM-{raw_id}"
        return None
    
    def _wants_transfer(self, text: str) -> bool:
        """Check if user wants to be transferred to a human."""
        text_lower = text.lower()
        return any(word in text_lower for word in self.transfer_words)
    
    def _wants_to_file_claim(self, text: str) -> bool:
        """Check if user wants to file a new claim."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.FILING_KEYWORDS)
    
    def process_text(self, text: str) -> str:
        """
        Process text input and return response.
        
        This is the core brain logic with RAG, claims, and filing integration.
        Uses intent classification for smarter routing.
        
        Priority order:
        1. Check if in active claim filing flow
        2. Route based on classified intent
        3. Fallback to knowledge base (RAG)
        """
        # Classify intent
        intent, confidence = self.intent_classifier.classify(text)
        
        # Priority 1: If in active claim filing flow, continue that flow
        if self.filing_manager.is_active():
            # Check for correction intent within filing
            if intent == Intent.CORRECTION:
                # The filing manager handles corrections internally
                pass
            
            is_complete, response = self.filing_manager.process_input(text)
            
            if is_complete:
                # All info collected, save the claim
                data = self.filing_manager.get_collected_data()
                success, message = self.claims.file_claim(
                    full_name=data["full_name"],
                    contract_id=data["contract_id"],
                    description=data["description"],
                    incident_date=data["incident_date"],
                    claim_type=data["claim_type"],
                )
                self.filing_manager.state.reset()
                return message
            
            return response
        
        # Priority 2: Route based on classified intent
        if intent == Intent.GOODBYE:
            return "Au revoir et à bientôt!"
        
        if intent == Intent.TRANSFER:
            return "Je vous transfère à un conseiller. Veuillez patienter un instant."
        
        if intent == Intent.CLAIM_STATUS:
            claim_id = self._detect_claim_id(text)
            if claim_id:
                found, message = self.claims.lookup_claim(claim_id)
                return message
            return "Pouvez-vous me donner votre numéro de dossier ? Il commence par CLM."
        
        if intent == Intent.FILE_CLAIM:
            return self.filing_manager.start_filing()
        
        # Priority 3: Check for explicit transfer request (backup)
        if self._wants_transfer(text):
            return "Je vous transfère à un conseiller. Veuillez patienter un instant."
        
        # Priority 4: Check for explicit claim ID mention
        claim_id = self._detect_claim_id(text)
        if claim_id:
            found, message = self.claims.lookup_claim(claim_id)
            return message
        
        # Priority 5: Check if user wants to file a new claim (backup)
        if self._wants_to_file_claim(text):
            return self.filing_manager.start_filing()
        
        # Priority 6: Answer from knowledge base (RAG) with relevance check
        context = ""
        has_relevant_context = False
        
        if self.rag:
            try:
                # Use new relevance-aware retrieval
                context = self.rag.get_context(text, k=3)
                has_relevant_context = self.rag.has_relevant_context(text, k=3)
            except Exception as e:
                print(f"[RAG Error] {e}")
        
        # Handle off-topic or no relevant context
        if intent == Intent.OFF_TOPIC or (not has_relevant_context and not context):
            return (
                "Je suis désolée, cette question ne fait pas partie de mon domaine. "
                "Je peux vous aider avec vos contrats d'assurance vie CNP, "
                "vos demandes en cours, ou vous transférer à un conseiller."
            )
        
        # If no context found, the LLM will respond with "I don't have this information"
        return self.llm.respond(text, context=context)
    
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
        
        response = self.process_text(text)
        return text, response
    
    def should_exit(self, text: str) -> bool:
        """Check if user wants to exit."""
        return any(word in text.lower() for word in self.exit_words)
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.llm.reset()
    
    def speak(self, text: str) -> None:
        """Speak text using TTS (uses streaming if enabled)."""
        if self.use_streaming_tts and hasattr(self.tts, 'speak_streaming'):
            self.tts.speak_streaming(text)
        else:
            self.tts.speak(text)
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes (for telephony)."""
        return self.tts.synthesize(text)
    
    def record(self) -> bytes | None:
        """Record audio using VAD."""
        return self.vad.record()
    
    def start_call_log(self) -> str | None:
        """Start a new call log and return the call ID."""
        if self.call_logger:
            return self.call_logger.start_call()
        return None
    
    def log_turn(self, call_id: str, user_text: str, agent_response: str, intent: Intent | None = None) -> None:
        """Log a conversation turn."""
        if self.call_logger and call_id:
            self.call_logger.log_turn(
                call_id=call_id,
                user_text=user_text,
                agent_response=agent_response,
                intent=intent.name if intent else None
            )
    
    def end_call_log(self, call_id: str, outcome: str = "completed") -> None:
        """End a call log with outcome."""
        if self.call_logger and call_id:
            self.call_logger.end_call(call_id, outcome)
    
    def get_call_stats(self) -> dict:
        """Get call statistics."""
        if self.call_logger:
            return self.call_logger.get_stats()
        return {}


class ConversationHandler:
    """
    Handles a single conversation session with call logging.
    
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
        self.call_id: str | None = None
    
    def start(self) -> None:
        """Start the conversation loop with call logging."""
        self.running = True
        
        # Start call logging
        self.call_id = self.agent.start_call_log()
        
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
                
                # Log the turn
                if self.call_id:
                    intent, _ = self.agent.intent_classifier.classify(text)
                    self.agent.log_turn(self.call_id, text, response, intent)
                
                # Check exit
                if self.agent.should_exit(text):
                    goodbye = "Au revoir!"
                    self.on_agent_text(goodbye)
                    self.agent.speak(goodbye)
                    
                    # End call log
                    if self.call_id:
                        self.agent.end_call_log(self.call_id, outcome="completed")
                    break
                
                # Respond
                self.on_agent_text(response)
                self.agent.speak(response)
                
            except KeyboardInterrupt:
                # End call log on interrupt
                if self.call_id:
                    self.agent.end_call_log(self.call_id, outcome="interrupted")
                self.running = False
    
    def stop(self) -> None:
        """Stop the conversation loop."""
        if self.call_id:
            self.agent.end_call_log(self.call_id, outcome="stopped")
        self.running = False

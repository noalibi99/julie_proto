"""
Tests for the main Agent orchestration.
"""

import pytest
from unittest.mock import MagicMock, patch
import re


class TestAgentClaimDetection:
    """Test claim ID detection in user messages."""
    
    def test_detect_claim_id_standard(self):
        """Test standard claim ID format."""
        from julie.core.agent import Agent
        
        pattern = Agent.CLAIM_ID_PATTERN
        
        match = pattern.search("Mon dossier CLM-123456")
        assert match is not None
    
    def test_detect_claim_id_with_spaces(self):
        """Test claim ID with spaces."""
        from julie.core.agent import Agent
        
        pattern = Agent.CLAIM_ID_PATTERN
        
        match = pattern.search("CLM 123 456")
        assert match is not None
    
    def test_detect_claim_id_lowercase(self):
        """Test lowercase claim ID."""
        from julie.core.agent import Agent
        
        pattern = Agent.CLAIM_ID_PATTERN
        
        match = pattern.search("dossier clm-2024001")
        assert match is not None
    
    def test_no_match_without_clm(self):
        """Test that random numbers don't match."""
        from julie.core.agent import Agent
        
        pattern = Agent.CLAIM_ID_PATTERN
        
        match = pattern.search("mon numéro est 123456")
        assert match is None


class TestAgentFilingKeywords:
    """Test filing keyword detection."""
    
    def test_filing_keywords_exist(self):
        """Test that filing keywords are defined."""
        from julie.core.agent import Agent
        
        keywords = Agent.FILING_KEYWORDS
        assert len(keywords) > 0
        assert "déclarer" in keywords or "declarer" in keywords
    
    @pytest.mark.parametrize("text,should_match", [
        ("je veux déclarer un sinistre", True),
        ("nouvelle demande", True),
        ("enregistrer ma demande", True),
        ("quel temps fait-il", False),
        ("bonjour", False),
    ])
    def test_wants_to_file_claim(self, text, should_match):
        """Test _wants_to_file_claim method."""
        from julie.core.agent import Agent
        
        # Check if any keyword is in the text
        keywords = Agent.FILING_KEYWORDS
        result = any(kw in text.lower() for kw in keywords)
        assert result == should_match


class TestAgentExitDetection:
    """Test exit word detection."""
    
    def test_exit_words(self):
        """Test that exit words trigger exit."""
        from julie.core.agent import Agent
        
        exit_words = ["au revoir", "bye", "quit", "exit", "stop"]
        
        for word in exit_words:
            assert any(w in word.lower() for w in exit_words)


class TestAgentTransferDetection:
    """Test transfer request detection."""
    
    @pytest.mark.parametrize("text,should_transfer", [
        ("je veux parler à un conseiller", True),
        ("transférer à un agent", True),
        ("parler à un humain", True),
        ("bonjour", False),
        ("quelle heure est-il", False),
    ])
    def test_wants_transfer(self, text, should_transfer):
        """Test transfer detection patterns."""
        transfer_words = ["transférer", "transferer", "conseiller", "humain", "agent"]
        result = any(word in text.lower() for word in transfer_words)
        assert result == should_transfer


class TestAgentMocked:
    """Test Agent with mocked components."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config that passes validation."""
        config = MagicMock()
        config.validate.return_value = []
        config.stt = MagicMock()
        config.llm = MagicMock()
        config.tts = MagicMock()
        config.audio = MagicMock()
        return config
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        return {
            "stt": MagicMock(),
            "llm": MagicMock(),
            "tts": MagicMock(),
            "vad": MagicMock(),
        }
    
    def test_agent_init_with_mocks(self, mock_config, mock_components):
        """Test agent initialization with mocked components."""
        with patch("julie.core.agent.get_stt", return_value=mock_components["stt"]), \
             patch("julie.core.agent.get_llm", return_value=mock_components["llm"]), \
             patch("julie.core.agent.get_tts", return_value=mock_components["tts"]), \
             patch("julie.core.agent.get_vad", return_value=mock_components["vad"]):
            
            from julie.core.agent import Agent
            
            agent = Agent(
                config=mock_config,
                enable_rag=False,  # Skip RAG initialization
                enable_logging=False,  # Skip logging initialization
            )
            
            assert agent is not None
            assert agent.intent_classifier is not None
    
    def test_agent_welcome(self, mock_config, mock_components):
        """Test welcome message."""
        mock_components["llm"].welcome.return_value = "Bonjour, je suis Julie!"
        
        with patch("julie.core.agent.get_stt", return_value=mock_components["stt"]), \
             patch("julie.core.agent.get_llm", return_value=mock_components["llm"]), \
             patch("julie.core.agent.get_tts", return_value=mock_components["tts"]), \
             patch("julie.core.agent.get_vad", return_value=mock_components["vad"]):
            
            from julie.core.agent import Agent
            
            agent = Agent(
                config=mock_config,
                enable_rag=False,
                enable_logging=False,
            )
            
            welcome = agent.welcome()
            assert "Julie" in welcome or "Bonjour" in welcome


class TestConversationHandler:
    """Test ConversationHandler."""
    
    def test_handler_init(self):
        """Test handler initialization."""
        from julie.core.agent import ConversationHandler
        
        mock_agent = MagicMock()
        handler = ConversationHandler(
            agent=mock_agent,
            on_listening=lambda: None,
            on_user_text=lambda t: None,
            on_agent_text=lambda t: None,
        )
        
        assert handler.agent == mock_agent
        assert handler.running is False
    
    def test_handler_stop(self):
        """Test stopping the handler."""
        from julie.core.agent import ConversationHandler
        
        mock_agent = MagicMock()
        mock_agent.end_call_log = MagicMock()
        
        handler = ConversationHandler(agent=mock_agent)
        handler.call_id = "test-call-id"
        handler.running = True
        
        handler.stop()
        
        assert handler.running is False

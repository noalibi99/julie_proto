"""
Tests for call logging.
"""

import pytest
import os
import tempfile
from julie.core.logging import CallLogger, CallLog, ConversationTurn, TurnSpeaker, CallOutcome


@pytest.fixture
def logger():
    """Create a CallLogger with temporary database."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    log = CallLogger(db_path=db_path)
    yield log
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestCallLogDataclass:
    """Test CallLog dataclass."""
    
    def test_create_call_log(self):
        log = CallLog()
        assert log.call_id is not None
        assert log.start_time is not None
        assert log.total_turns == 0
    
    def test_add_turn(self):
        log = CallLog()
        log.add_turn(TurnSpeaker.USER, "Hello", intent="GREETING")
        assert log.total_turns == 1
        assert log.user_turns == 1
        assert log.agent_turns == 0
    
    def test_add_agent_turn(self):
        log = CallLog()
        log.add_turn(TurnSpeaker.AGENT, "Welcome!")
        assert log.agent_turns == 1
    
    def test_end_call(self):
        log = CallLog()
        log.end_call(CallOutcome.RESOLVED)
        assert log.end_time is not None
        assert log.outcome == CallOutcome.RESOLVED
    
    def test_end_call_with_transfer(self):
        log = CallLog()
        log.end_call(CallOutcome.TRANSFERRED, reason="User requested")
        assert log.transfer_reason == "User requested"
    
    def test_add_claim_created(self):
        log = CallLog()
        log.add_claim_created("CLM-123")
        assert "CLM-123" in log.claims_created
    
    def test_add_claim_queried(self):
        log = CallLog()
        log.add_claim_queried("CLM-456")
        log.add_claim_queried("CLM-456")  # Duplicate
        assert log.claims_queried == ["CLM-456"]  # Should not duplicate
    
    def test_to_dict(self):
        log = CallLog()
        log.add_turn(TurnSpeaker.USER, "Hello")
        data = log.to_dict()
        assert "call_id" in data
        assert "turns" in data
        assert len(data["turns"]) == 1


class TestConversationTurn:
    """Test ConversationTurn dataclass."""
    
    def test_create_turn(self):
        turn = ConversationTurn(speaker=TurnSpeaker.USER, text="Hello")
        assert turn.speaker == TurnSpeaker.USER
        assert turn.text == "Hello"
    
    def test_to_dict(self):
        turn = ConversationTurn(
            speaker=TurnSpeaker.USER,
            text="Hello",
            intent="GREETING",
            confidence=0.95
        )
        data = turn.to_dict()
        assert data["speaker"] == "user"
        assert data["intent"] == "GREETING"


class TestCallLoggerBasics:
    """Test basic CallLogger operations."""
    
    def test_start_call(self, logger):
        call_id = logger.start_call()
        assert call_id is not None
        assert len(call_id) > 10  # UUID format
    
    def test_log_turn(self, logger):
        call_id = logger.start_call()
        logger.log_turn(call_id, "Hello", "Bonjour!", intent="GREETING")
        # Should not raise
    
    def test_end_call(self, logger):
        call_id = logger.start_call()
        logger.log_turn(call_id, "Hello", "Welcome")
        logger.end_call(call_id, "completed")
        # Should be saved to DB
    
    def test_end_call_saves_to_db(self, logger):
        call_id = logger.start_call()
        logger.log_turn(call_id, "Test", "Response")
        logger.end_call(call_id, "completed")
        
        # Retrieve from DB
        result = logger.get_by_id(call_id)
        assert result is not None
        assert result["call_id"] == call_id


class TestCallLoggerPersistence:
    """Test database persistence."""
    
    def test_save_and_retrieve(self, logger):
        log = CallLog()
        log.add_turn(TurnSpeaker.USER, "Hello")
        log.add_turn(TurnSpeaker.AGENT, "Welcome!")
        log.end_call(CallOutcome.RESOLVED)
        
        logger.save(log)
        
        retrieved = logger.get_by_id(log.call_id)
        assert retrieved is not None
        assert retrieved["total_turns"] == 2
        assert retrieved["outcome"] == "resolved"
    
    def test_get_recent(self, logger):
        # Create multiple calls
        for i in range(5):
            call_id = logger.start_call()
            logger.log_turn(call_id, f"Hello {i}", f"Response {i}")
            logger.end_call(call_id, "completed")
        
        recent = logger.get_recent(limit=3)
        assert len(recent) == 3
    
    def test_get_stats(self, logger):
        # Create some calls
        call_id = logger.start_call()
        logger.end_call(call_id, "completed")
        
        call_id2 = logger.start_call()
        logger.end_call(call_id2, "transferred")
        
        stats = logger.get_stats(days=7)
        assert stats["total_calls"] == 2
        assert "resolved" in stats["outcomes"] or "transferred" in stats["outcomes"]


class TestCallLoggerOutcomes:
    """Test different call outcomes."""
    
    @pytest.mark.parametrize("outcome,expected", [
        ("completed", "resolved"),
        ("resolved", "resolved"),
        ("transferred", "transferred"),
        ("abandoned", "abandoned"),
        ("interrupted", "abandoned"),
        ("error", "error"),
    ])
    def test_outcome_mapping(self, logger, outcome, expected):
        call_id = logger.start_call()
        logger.end_call(call_id, outcome)
        
        result = logger.get_by_id(call_id)
        assert result["outcome"] == expected


class TestCallLoggerClaims:
    """Test claim tracking in calls."""
    
    def test_record_claim_created(self, logger):
        call_id = logger.start_call()
        logger.record_claim_created(call_id, "CLM-12345")
        logger.end_call(call_id, "completed")
        
        result = logger.get_by_id(call_id)
        assert "CLM-12345" in result["claims_created"]
    
    def test_record_claim_queried(self, logger):
        call_id = logger.start_call()
        logger.record_claim_queried(call_id, "CLM-67890")
        logger.end_call(call_id, "completed")
        
        result = logger.get_by_id(call_id)
        assert "CLM-67890" in result["claims_queried"]

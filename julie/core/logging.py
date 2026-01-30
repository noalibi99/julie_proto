"""
Call logging and analytics.

Tracks all conversations for quality monitoring and analysis.
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid


class CallOutcome(Enum):
    """Possible call outcomes."""
    RESOLVED = "resolved"           # User's issue was handled
    TRANSFERRED = "transferred"     # Transferred to human agent
    ABANDONED = "abandoned"         # User left without resolution
    CLAIM_FILED = "claim_filed"     # New claim was created
    ERROR = "error"                 # Technical error occurred


class TurnSpeaker(Enum):
    """Who is speaking in a conversation turn."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    speaker: TurnSpeaker
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: Optional[int] = None  # For audio turns
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker.value,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
        }


@dataclass
class CallLog:
    """Complete log of a single call/conversation."""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    outcome: CallOutcome = CallOutcome.ABANDONED
    turns: List[ConversationTurn] = field(default_factory=list)
    claims_created: List[str] = field(default_factory=list)
    claims_queried: List[str] = field(default_factory=list)
    transfer_reason: Optional[str] = None
    error_message: Optional[str] = None
    caller_info: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    total_turns: int = 0
    user_turns: int = 0
    agent_turns: int = 0
    avg_response_time_ms: Optional[float] = None
    
    def add_turn(
        self,
        speaker: TurnSpeaker,
        text: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            intent=intent,
            confidence=confidence,
            duration_ms=duration_ms,
        )
        self.turns.append(turn)
        self.total_turns += 1
        
        if speaker == TurnSpeaker.USER:
            self.user_turns += 1
        elif speaker == TurnSpeaker.AGENT:
            self.agent_turns += 1
    
    def end_call(self, outcome: CallOutcome, reason: Optional[str] = None) -> None:
        """Mark the call as ended."""
        self.end_time = datetime.now()
        self.outcome = outcome
        if outcome == CallOutcome.TRANSFERRED:
            self.transfer_reason = reason
        elif outcome == CallOutcome.ERROR:
            self.error_message = reason
    
    def add_claim_created(self, claim_id: str) -> None:
        """Record a claim that was created during this call."""
        self.claims_created.append(claim_id)
    
    def add_claim_queried(self, claim_id: str) -> None:
        """Record a claim that was queried during this call."""
        if claim_id not in self.claims_queried:
            self.claims_queried.append(claim_id)
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get call duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "call_id": self.call_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "outcome": self.outcome.value,
            "turns": [t.to_dict() for t in self.turns],
            "claims_created": self.claims_created,
            "claims_queried": self.claims_queried,
            "transfer_reason": self.transfer_reason,
            "error_message": self.error_message,
            "caller_info": self.caller_info,
            "total_turns": self.total_turns,
            "user_turns": self.user_turns,
            "agent_turns": self.agent_turns,
            "duration_seconds": self.get_duration_seconds(),
        }


class CallLogger:
    """
    Persists call logs to SQLite database.
    
    Provides methods for logging and querying call data.
    """
    
    def __init__(self, db_path: str = "data/call_logs.db"):
        """Initialize the call logger."""
        self.db_path = db_path
        self._active_calls: Dict[str, CallLog] = {}  # Track in-progress calls
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    outcome TEXT NOT NULL,
                    total_turns INTEGER DEFAULT 0,
                    user_turns INTEGER DEFAULT 0,
                    agent_turns INTEGER DEFAULT 0,
                    duration_seconds REAL,
                    claims_created TEXT,
                    claims_queried TEXT,
                    transfer_reason TEXT,
                    error_message TEXT,
                    caller_info TEXT,
                    turns_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calls_start_time 
                ON calls(start_time)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calls_outcome 
                ON calls(outcome)
            """)
            
            conn.commit()
    
    def save(self, call_log: CallLog) -> None:
        """Save a call log to the database."""
        data = call_log.to_dict()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO calls (
                    call_id, start_time, end_time, outcome,
                    total_turns, user_turns, agent_turns, duration_seconds,
                    claims_created, claims_queried, transfer_reason,
                    error_message, caller_info, turns_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["call_id"],
                data["start_time"],
                data["end_time"],
                data["outcome"],
                data["total_turns"],
                data["user_turns"],
                data["agent_turns"],
                data["duration_seconds"],
                json.dumps(data["claims_created"]),
                json.dumps(data["claims_queried"]),
                data["transfer_reason"],
                data["error_message"],
                json.dumps(data["caller_info"]),
                json.dumps(data["turns"]),
            ))
            conn.commit()
    
    def get_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get a call log by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM calls WHERE call_id = ?",
                (call_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent call logs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM calls ORDER BY start_time DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get call statistics for the last N days."""
        from datetime import timedelta
        
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total calls
            total = conn.execute(
                "SELECT COUNT(*) as count FROM calls WHERE start_time >= ?",
                (since,)
            ).fetchone()["count"]
            
            # By outcome
            outcomes = {}
            for row in conn.execute(
                "SELECT outcome, COUNT(*) as count FROM calls WHERE start_time >= ? GROUP BY outcome",
                (since,)
            ):
                outcomes[row["outcome"]] = row["count"]
            
            # Average duration
            avg_duration = conn.execute(
                "SELECT AVG(duration_seconds) as avg FROM calls WHERE start_time >= ? AND duration_seconds IS NOT NULL",
                (since,)
            ).fetchone()["avg"]
            
            # Claims created
            claims_created = conn.execute(
                "SELECT COUNT(*) as count FROM calls WHERE start_time >= ? AND claims_created != '[]'",
                (since,)
            ).fetchone()["count"]
            
            return {
                "period_days": days,
                "total_calls": total,
                "outcomes": outcomes,
                "avg_duration_seconds": round(avg_duration, 1) if avg_duration else None,
                "calls_with_claims_created": claims_created,
                "resolution_rate": round(outcomes.get("resolved", 0) / max(total, 1) * 100, 1),
            }
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "call_id": row["call_id"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "outcome": row["outcome"],
            "total_turns": row["total_turns"],
            "user_turns": row["user_turns"],
            "agent_turns": row["agent_turns"],
            "duration_seconds": row["duration_seconds"],
            "claims_created": json.loads(row["claims_created"] or "[]"),
            "claims_queried": json.loads(row["claims_queried"] or "[]"),
            "transfer_reason": row["transfer_reason"],
            "error_message": row["error_message"],
            "caller_info": json.loads(row["caller_info"] or "{}"),
            "turns": json.loads(row["turns_json"] or "[]"),
        }
    
    # ============================================================
    # Convenience methods for simpler Agent integration
    # ============================================================
    
    def start_call(self) -> str:
        """
        Start tracking a new call.
        
        Returns:
            The call_id for this conversation
        """
        call_log = CallLog()
        self._active_calls[call_log.call_id] = call_log
        return call_log.call_id
    
    def log_turn(
        self,
        call_id: str,
        user_text: str,
        agent_response: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Log a conversation turn (user + agent).
        
        Args:
            call_id: The call ID from start_call()
            user_text: What the user said
            agent_response: What the agent responded
            intent: Detected intent (optional)
            confidence: Intent confidence (optional)
        """
        call_log = self._active_calls.get(call_id)
        if call_log:
            # Log user turn
            call_log.add_turn(
                speaker=TurnSpeaker.USER,
                text=user_text,
                intent=intent,
                confidence=confidence,
            )
            # Log agent turn
            call_log.add_turn(
                speaker=TurnSpeaker.AGENT,
                text=agent_response,
            )
    
    def end_call(self, call_id: str, outcome: str = "completed") -> None:
        """
        End a call and save to database.
        
        Args:
            call_id: The call ID from start_call()
            outcome: One of 'completed', 'transferred', 'abandoned', 'error'
        """
        call_log = self._active_calls.pop(call_id, None)
        if call_log:
            # Map string outcome to enum
            outcome_map = {
                "completed": CallOutcome.RESOLVED,
                "resolved": CallOutcome.RESOLVED,
                "transferred": CallOutcome.TRANSFERRED,
                "abandoned": CallOutcome.ABANDONED,
                "error": CallOutcome.ERROR,
                "interrupted": CallOutcome.ABANDONED,
                "stopped": CallOutcome.ABANDONED,
                "claim_filed": CallOutcome.CLAIM_FILED,
            }
            call_log.end_call(outcome_map.get(outcome, CallOutcome.RESOLVED))
            self.save(call_log)
    
    def record_claim_created(self, call_id: str, claim_id: str) -> None:
        """Record that a claim was created during this call."""
        call_log = self._active_calls.get(call_id)
        if call_log:
            call_log.add_claim_created(claim_id)
    
    def record_claim_queried(self, call_id: str, claim_id: str) -> None:
        """Record that a claim was queried during this call."""
        call_log = self._active_calls.get(call_id)
        if call_log:
            call_log.add_claim_queried(claim_id)


# Singleton instance
_logger: Optional[CallLogger] = None


def get_call_logger() -> CallLogger:
    """Get or create the call logger singleton."""
    global _logger
    if _logger is None:
        _logger = CallLogger()
    return _logger

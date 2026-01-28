"""
Julie Voice Assistant - French Insurance AI Agent

Modular architecture ready for:
- CLI interface (current)
- Telephony integration (Asterisk)
- RAG knowledge base (Qdrant)
- Backend API integration
"""

from julie.config import Config
from julie.core.agent import Agent

__version__ = "0.2.0"
__all__ = ["Config", "Agent"]

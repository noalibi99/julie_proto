"""
LLM providers and brain logic with RAG integration.
"""

from abc import ABC, abstractmethod
from typing import Optional

from groq import Groq

from julie.config import LLMConfig
from julie.llm.prompts import SYSTEM_PROMPT, RAG_SYSTEM_PROMPT


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def respond(self, user_text: str, context: str = "") -> str:
        """Generate a response to user input."""
        pass
    
    @abstractmethod
    def welcome(self) -> str:
        """Return welcome message."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset conversation history."""
        pass


class GroqLLM(BaseLLM):
    """LLM using Groq API with optional RAG context."""
    
    def __init__(self, config: LLMConfig | None = None, system_prompt: str | None = None):
        self.config = config or LLMConfig()
        
        if not self.config.api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        self.client = Groq(api_key=self.config.api_key)
        self.model = self.config.model
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        
        self.base_system_prompt = system_prompt or SYSTEM_PROMPT
        self.conversation_history: list[dict] = []
    
    def respond(self, user_text: str, context: str = "") -> str:
        """
        Generate a response to user input.
        
        Args:
            user_text: User's message
            context: Optional RAG context to inject into prompt
            
        Returns:
            Assistant response
        """
        try:
            # Build system prompt with context if provided
            if context:
                system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
            else:
                system_prompt = self.base_system_prompt
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history[-6:])
            messages.append({"role": "user", "content": user_text})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            reply = response.choices[0].message.content.strip()
            
            self.conversation_history.append({"role": "user", "content": user_text})
            self.conversation_history.append({"role": "assistant", "content": reply})
            
            return reply
            
        except Exception as e:
            print(f"[LLM Error] {e}")
            return "Désolée, je n'ai pas pu traiter votre demande."
    
    def welcome(self) -> str:
        """Return welcome message."""
        return "Bonjour, je suis Julie, votre assistante CNP Assurances. Comment puis-je vous aider avec votre assurance vie?"
    
    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []


def get_llm(config: LLMConfig | None = None, system_prompt: str | None = None) -> BaseLLM:
    """Factory function to get LLM instance."""
    config = config or LLMConfig()
    
    if config.provider == "groq":
        return GroqLLM(config, system_prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")

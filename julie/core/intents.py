"""
Intent classification for conversation routing.

Uses a combination of keyword matching and LLM classification
for accurate intent detection.
"""

from enum import Enum
from typing import Optional, Tuple
import re


class Intent(Enum):
    """Possible user intents."""
    GREETING = "greeting"           # Hello, bonjour
    GOODBYE = "goodbye"             # Au revoir, bye
    FAQ = "faq"                     # Question about insurance
    CLAIM_STATUS = "claim_status"   # Looking up existing claim
    FILE_CLAIM = "file_claim"       # Wants to create new claim
    TRANSFER = "transfer"           # Wants human agent
    CONFIRM = "confirm"             # Yes, confirming something
    DENY = "deny"                   # No, denying something
    CORRECTION = "correction"       # Wants to correct previous info
    OFF_TOPIC = "off_topic"         # Not insurance related
    UNKNOWN = "unknown"             # Cannot determine


class IntentClassifier:
    """
    Classifies user intent from text input.
    
    Uses a hierarchical approach:
    1. Pattern matching for high-confidence intents
    2. Keyword scoring for medium-confidence
    3. Falls back to UNKNOWN for LLM to handle
    """
    
    # Patterns for exact matching (high confidence)
    PATTERNS = {
        Intent.GREETING: [
            r"^bonjour\b", r"^salut\b", r"^hello\b", r"^bonsoir\b",
            r"^coucou\b", r"^hey\b"
        ],
        Intent.GOODBYE: [
            r"\bau revoir\b", r"\bbye\b", r"\bà bientôt\b", r"\bbonne journée\b",
            r"\bmerci au revoir\b", r"\bc'est tout\b", r"^merci$"
        ],
        Intent.CLAIM_STATUS: [
            r"\bclm[-\s]?\d", r"\bsuivre.*(demande|sinistre|dossier)\b",
            r"\bétat.*(demande|sinistre|dossier)\b", r"\bstatut\b",
            r"\boù en est\b", r"\bavancement\b"
        ],
        Intent.TRANSFER: [
            r"\btransférer\b", r"\bconseiller\b", r"\bhumain\b",
            r"\bparler à quelqu'un\b", r"\bagent\b", r"\bvrai personne\b"
        ],
        Intent.FILE_CLAIM: [
            r"\bd[ée]clarer\b", r"\bnouvelle demande\b", r"\bnouveau sinistre\b",
            r"\benregistrer.*(demande|sinistre)\b", r"\bsignaler\b",
            r"\bfaire une demande\b", r"\bouvrir.*dossier\b", r"\bd[ée]poser\b",
            r"\bcr[ée]er.*(demande|dossier)\b", r"\bsinistre\b"
        ],
        Intent.CONFIRM: [
            r"^oui\b", r"^yes\b", r"^correct\b", r"^exactement\b",
            r"^c'est ça\b", r"^confirme\b", r"^d'accord\b", r"^ok\b",
            r"^tout à fait\b", r"^absolument\b"
        ],
        Intent.DENY: [
            r"^non\b", r"^no\b", r"^pas du tout\b", r"^incorrect\b",
            r"^faux\b", r"^ce n'est pas\b"
        ],
        Intent.CORRECTION: [
            r"\bcorriger\b", r"\bmodifier\b", r"\bchanger\b", r"\berreur\b",
            r"\bpas correct\b", r"\bc'est pas\b", r"\ben fait\b",
            r"\bfinalement\b", r"\bje voulais dire\b", r"\bpardon\b",
            r"\bnon c'est\b", r"\bplutôt\b"
        ],
    }
    
    # Keywords with weights for scoring
    KEYWORDS = {
        Intent.FAQ: {
            "qu'est-ce": 0.8, "comment": 0.6, "pourquoi": 0.5,
            "quels": 0.5, "quelle": 0.5, "combien": 0.6,
            "assurance vie": 0.7, "contrat": 0.4, "bénéficiaire": 0.6,
            "rachat": 0.5, "versement": 0.5, "fiscalité": 0.7,
            "avantages": 0.5, "frais": 0.6, "rendement": 0.6,
            "euros": 0.4, "unités de compte": 0.7, "arbitrage": 0.5,
            "clause": 0.5, "décès": 0.4, "succession": 0.6,
        },
        Intent.OFF_TOPIC: {
            "météo": 0.9, "temps qu'il fait": 0.9, "température": 0.8,
            "football": 0.9, "sport": 0.8, "politique": 0.8,
            "restaurant": 0.9, "film": 0.9, "musique": 0.8,
            "voyage": 0.7, "vacances": 0.7, "actualités": 0.8,
            "recette": 0.9, "blague": 0.9, "histoire drôle": 0.9,
        },
    }
    
    # Claim ID pattern
    CLAIM_ID_PATTERN = re.compile(r'CLM[\s-]*[A-Z0-9\s-]{3,15}', re.IGNORECASE)
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize classifier.
        
        Args:
            confidence_threshold: Minimum score to return an intent
        """
        self.confidence_threshold = confidence_threshold
        
        # Compile patterns
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.PATTERNS.items()
        }
    
    def classify(self, text: str) -> Tuple[Intent, float]:
        """
        Classify user intent from text.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        text_lower = text.lower().strip()
        
        # Check for claim ID first (highest priority)
        if self.CLAIM_ID_PATTERN.search(text):
            return Intent.CLAIM_STATUS, 1.0
        
        # Try pattern matching
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return intent, 0.95
        
        # Try keyword scoring
        scores = {}
        for intent, keywords in self.KEYWORDS.items():
            score = 0.0
            matches = 0
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    score += weight
                    matches += 1
            if matches > 0:
                # Normalize by number of matches
                scores[intent] = min(score / max(matches, 1), 1.0)
        
        if scores:
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]
            if best_score >= self.confidence_threshold:
                return best_intent, best_score
        
        # Default to FAQ for question-like inputs
        if any(text_lower.startswith(q) for q in ["qu", "comment", "pourquoi", "est-ce", "puis-je", "peut-on"]):
            return Intent.FAQ, 0.5
        
        return Intent.UNKNOWN, 0.0
    
    def is_correction_request(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user wants to correct a specific field.
        
        Args:
            text: User input
            
        Returns:
            Tuple of (is_correction, field_name)
        """
        text_lower = text.lower()
        
        # Check for correction intent
        intent, confidence = self.classify(text)
        if intent != Intent.CORRECTION and confidence < 0.5:
            return False, None
        
        # Try to identify which field to correct
        field_patterns = {
            "type": [r"type", r"demande", r"sinistre"],
            "name": [r"nom", r"prénom", r"appelle"],
            "contract": [r"contrat", r"numéro", r"cnp"],
            "date": [r"date", r"jour", r"quand"],
            "description": [r"description", r"détails", r"situation"],
        }
        
        for field_name, patterns in field_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return True, field_name
        
        # Correction without specific field
        return True, None


# Singleton instance
_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the intent classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier

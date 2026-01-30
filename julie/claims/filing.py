"""
Claim filing state machine.

Manages the structured conversation flow for filing new claims.
Each field is collected one by one with validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import re

from julie.claims.models import ClaimType


class FilingStep(Enum):
    """Steps in the claim filing flow."""
    IDLE = "idle"                    # Not in filing mode
    ASK_TYPE = "ask_type"            # Asking for claim type
    ASK_NAME = "ask_name"            # Asking for full name
    ASK_CONTRACT = "ask_contract"    # Asking for contract number
    ASK_DATE = "ask_date"            # Asking for incident date
    ASK_DESCRIPTION = "ask_description"  # Asking for description
    CONFIRM = "confirm"              # Confirming all info
    COMPLETED = "completed"          # Filing complete


@dataclass
class FilingState:
    """
    Tracks the state of a claim filing conversation.
    
    Stores collected information and current step.
    """
    step: FilingStep = FilingStep.IDLE
    claim_type: Optional[ClaimType] = None
    full_name: Optional[str] = None
    contract_id: Optional[str] = None
    incident_date: Optional[str] = None
    description: Optional[str] = None
    
    # Error tracking
    last_error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.step = FilingStep.IDLE
        self.claim_type = None
        self.full_name = None
        self.contract_id = None
        self.incident_date = None
        self.description = None
        self.last_error = None
        self.retry_count = 0
    
    def is_active(self) -> bool:
        """Check if currently in filing mode."""
        return self.step not in (FilingStep.IDLE, FilingStep.COMPLETED)
    
    def get_collected_summary(self) -> str:
        """Get summary of collected information."""
        parts = []
        if self.claim_type:
            parts.append(f"Type: {self.claim_type.value}")
        if self.full_name:
            parts.append(f"Nom: {self.full_name}")
        if self.contract_id:
            parts.append(f"Contrat: {self.contract_id}")
        if self.incident_date:
            parts.append(f"Date: {self.incident_date}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return " | ".join(parts) if parts else "Aucune information collectée"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for claim creation."""
        return {
            "claim_type": self.claim_type.value if self.claim_type else None,
            "full_name": self.full_name,
            "contract_id": self.contract_id,
            "incident_date": self.incident_date,
            "description": self.description,
        }


class ClaimFilingManager:
    """
    Manages the claim filing conversation flow.
    
    This is a state machine that guides the user through
    collecting all required information step by step.
    """
    
    # Claim type keywords for detection
    TYPE_KEYWORDS = {
        ClaimType.DECES: ["décès", "deces", "mort", "décédé", "decede"],
        ClaimType.RACHAT_PARTIEL: ["rachat partiel", "retrait partiel", "retirer une partie"],
        ClaimType.RACHAT_TOTAL: ["rachat total", "retrait total", "clôture", "cloture", "tout retirer"],
        ClaimType.INVALIDITE: ["invalidité", "invalidite", "handicap", "incapacité"],
        ClaimType.AVANCE: ["avance", "prêt", "pret", "emprunt"],
        ClaimType.ARBITRAGE: ["arbitrage", "transfert", "changer de support"],
    }
    
    # Contract ID pattern
    CONTRACT_PATTERN = re.compile(r'[A-Z]{2,4}[\s-]*\d{4}[\s-]*\d{3,6}', re.IGNORECASE)
    
    # Date patterns (flexible French formats)
    DATE_PATTERNS = [
        r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})',  # DD/MM/YYYY
        r"aujourd'hui",
        r"hier",
        r"(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*(\d{2,4})?",
    ]
    
    def __init__(self):
        """Initialize the filing manager."""
        self.state = FilingState()
    
    def start_filing(self) -> str:
        """
        Start the claim filing process.
        
        Returns:
            First question to ask the user
        """
        self.state.reset()
        self.state.step = FilingStep.ASK_TYPE
        
        return (
            "Je vais vous aider à enregistrer votre demande. "
            "Quel type de demande souhaitez-vous effectuer ? "
            "Par exemple: décès, rachat partiel, rachat total, invalidité, avance, ou arbitrage."
        )
    
    def process_input(self, user_text: str) -> tuple[bool, str]:
        """
        Process user input based on current step.
        
        Args:
            user_text: What the user said
            
        Returns:
            Tuple of (is_complete, response)
            - is_complete: True if all info collected and ready to save
            - response: What Julie should say next
        """
        text_lower = user_text.lower().strip()
        
        # Check for cancel/abort
        if any(word in text_lower for word in ["annuler", "stop", "arrêter", "non merci"]):
            self.state.reset()
            return False, "D'accord, j'annule l'enregistrement de la demande. Comment puis-je vous aider autrement ?"
        
        # Process based on current step
        if self.state.step == FilingStep.ASK_TYPE:
            return self._process_type(user_text)
        elif self.state.step == FilingStep.ASK_NAME:
            return self._process_name(user_text)
        elif self.state.step == FilingStep.ASK_CONTRACT:
            return self._process_contract(user_text)
        elif self.state.step == FilingStep.ASK_DATE:
            return self._process_date(user_text)
        elif self.state.step == FilingStep.ASK_DESCRIPTION:
            return self._process_description(user_text)
        elif self.state.step == FilingStep.CONFIRM:
            return self._process_confirmation(user_text)
        
        return False, "Une erreur s'est produite. Recommençons."
    
    def _process_type(self, text: str) -> tuple[bool, str]:
        """Process claim type input."""
        text_lower = text.lower()
        
        # Try to match claim type
        for claim_type, keywords in self.TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                self.state.claim_type = claim_type
                self.state.step = FilingStep.ASK_NAME
                
                # Empathetic response for death claims
                if claim_type == ClaimType.DECES:
                    return False, "Je suis désolée pour cette situation difficile. Quel est le nom complet du titulaire du contrat ?"
                else:
                    return False, f"Très bien, une demande de {claim_type.value}. Quel est votre nom complet ?"
        
        # Not recognized
        self.state.retry_count += 1
        if self.state.retry_count >= self.state.max_retries:
            self.state.reset()
            return False, "Je n'ai pas réussi à comprendre le type de demande. Souhaitez-vous être transféré à un conseiller ?"
        
        return False, (
            "Je n'ai pas compris le type de demande. "
            "Les types possibles sont: décès, rachat partiel, rachat total, invalidité, avance, ou arbitrage. "
            "Lequel souhaitez-vous ?"
        )
    
    def _process_name(self, text: str) -> tuple[bool, str]:
        """Process full name input."""
        # Clean up common speech artifacts
        name = text.strip()
        
        # Remove common filler words from STT
        fillers = ["c'est", "c est", "il s'appelle", "elle s'appelle", "je m'appelle", 
                   "mon nom est", "le nom est", "oui", "donc", "alors", "euh"]
        name_lower = name.lower()
        for filler in fillers:
            if name_lower.startswith(filler):
                name = name[len(filler):].strip()
                name_lower = name.lower()
        
        # Remove punctuation
        name = name.replace(",", " ").replace(".", " ").strip()
        
        words = name.split()
        
        # Accept if at least 2 words
        if len(words) >= 2:
            # Capitalize properly
            self.state.full_name = " ".join(w.capitalize() for w in words)
            self.state.step = FilingStep.ASK_CONTRACT
            self.state.retry_count = 0
            return False, f"Merci. Quel est votre numéro de contrat ?"
        
        self.state.retry_count += 1
        if self.state.retry_count >= self.state.max_retries:
            self.state.reset()
            return False, "Je n'ai pas réussi à enregistrer le nom. Souhaitez-vous être transféré à un conseiller ?"
        
        return False, "Pouvez-vous me donner le nom complet, prénom et nom de famille ?"
    
    def _process_contract(self, text: str) -> tuple[bool, str]:
        """Process contract ID input."""
        # Try to extract contract number
        text_clean = text.upper().replace(" ", "").replace("-", "")
        
        # Look for pattern like CNP2020001234
        match = re.search(r'[A-Z]{2,4}\d{7,10}', text_clean)
        if match:
            contract_id = match.group(0)
            # Format nicely: CNP-2020-001234
            if len(contract_id) >= 11:
                self.state.contract_id = f"{contract_id[:3]}-{contract_id[3:7]}-{contract_id[7:]}"
            else:
                self.state.contract_id = contract_id
            
            self.state.step = FilingStep.ASK_DATE
            self.state.retry_count = 0
            return False, "Parfait. Quelle est la date de l'événement ?"
        
        self.state.retry_count += 1
        if self.state.retry_count >= self.state.max_retries:
            self.state.reset()
            return False, "Je n'ai pas réussi à enregistrer le numéro de contrat. Souhaitez-vous être transféré à un conseiller ?"
        
        return False, "Je n'ai pas compris le numéro de contrat. Il ressemble généralement à CNP-2020-001234. Pouvez-vous le répéter ?"
    
    def _process_date(self, text: str) -> tuple[bool, str]:
        """Process incident date input."""
        text_lower = text.lower().strip()
        
        # Handle relative dates
        if "aujourd'hui" in text_lower:
            self.state.incident_date = datetime.now().strftime("%d/%m/%Y")
        elif "hier" in text_lower:
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            self.state.incident_date = yesterday.strftime("%d/%m/%Y")
        else:
            # Try to extract date
            date_match = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})', text)
            if date_match:
                day, month, year = date_match.groups()
                if len(year) == 2:
                    year = "20" + year
                self.state.incident_date = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
            else:
                # Try month names
                months = {
                    "janvier": "01", "février": "02", "mars": "03", "avril": "04",
                    "mai": "05", "juin": "06", "juillet": "07", "août": "08",
                    "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
                }
                for month_name, month_num in months.items():
                    if month_name in text_lower:
                        day_match = re.search(r'(\d{1,2})', text)
                        if day_match:
                            day = day_match.group(1).zfill(2)
                            year = datetime.now().year
                            year_match = re.search(r'20\d{2}', text)
                            if year_match:
                                year = year_match.group(0)
                            self.state.incident_date = f"{day}/{month_num}/{year}"
                            break
        
        if self.state.incident_date:
            self.state.step = FilingStep.ASK_DESCRIPTION
            self.state.retry_count = 0
            return False, "Merci. Pouvez-vous décrire brièvement la situation ?"
        
        self.state.retry_count += 1
        if self.state.retry_count >= self.state.max_retries:
            self.state.reset()
            return False, "Je n'ai pas réussi à comprendre la date. Souhaitez-vous être transféré à un conseiller ?"
        
        return False, "Je n'ai pas compris la date. Pouvez-vous la donner au format jour/mois/année, par exemple 15/01/2026 ?"
    
    def _process_description(self, text: str) -> tuple[bool, str]:
        """Process description input."""
        description = text.strip()
        
        if len(description) >= 10:  # Minimum meaningful description
            self.state.description = description
            self.state.step = FilingStep.CONFIRM
            self.state.retry_count = 0
            
            # Build confirmation summary
            summary = (
                f"Récapitulatif de votre demande: "
                f"Type: {self.state.claim_type.value}, "
                f"Nom: {self.state.full_name}, "
                f"Contrat: {self.state.contract_id}, "
                f"Date: {self.state.incident_date}. "
                f"Confirmez-vous ces informations ? Vous pouvez aussi me dire quel champ modifier."
            )
            return False, summary
        
        self.state.retry_count += 1
        if self.state.retry_count >= self.state.max_retries:
            self.state.reset()
            return False, "Je n'ai pas pu enregistrer la description. Souhaitez-vous être transféré à un conseiller ?"
        
        return False, "Pouvez-vous me donner plus de détails sur votre situation ?"
    
    def _process_confirmation(self, text: str) -> tuple[bool, str]:
        """Process confirmation input - supports corrections."""
        text_lower = text.lower()
        
        positive = ["oui", "yes", "correct", "exactement", "c'est ça", "confirme", "d'accord", "ok", "parfait"]
        negative = ["non", "no", "pas correct", "erreur", "recommencer", "tout refaire"]
        
        # Check for specific field correction requests
        correction_result = self._check_correction_request(text_lower)
        if correction_result:
            return correction_result
        
        if any(word in text_lower for word in positive):
            self.state.step = FilingStep.COMPLETED
            return True, ""  # Signal to save the claim
        
        if any(word in text_lower for word in negative):
            # Ask which field to correct instead of restarting everything
            return False, (
                "Que souhaitez-vous modifier ? "
                "Le type de demande, le nom, le numéro de contrat, la date, ou la description ?"
            )
        
        return False, "Confirmez-vous ces informations ? Dites oui ou indiquez ce que vous souhaitez modifier."
    
    def _check_correction_request(self, text_lower: str) -> tuple[bool, str] | None:
        """
        Check if user wants to correct a specific field.
        
        Returns:
            Tuple (is_complete, response) if correction detected, None otherwise
        """
        # Field detection patterns
        field_patterns = {
            FilingStep.ASK_TYPE: ["type", "demande", "sinistre", "nature"],
            FilingStep.ASK_NAME: ["nom", "prénom", "appelle", "titulaire"],
            FilingStep.ASK_CONTRACT: ["contrat", "numéro", "cnp", "police"],
            FilingStep.ASK_DATE: ["date", "jour", "quand", "événement"],
            FilingStep.ASK_DESCRIPTION: ["description", "détail", "situation", "explication"],
        }
        
        # Correction trigger words
        correction_triggers = [
            "modifier", "changer", "corriger", "erreur", "pas bon", "faux",
            "c'est pas", "plutôt", "en fait", "non c'est", "le nom c'est",
            "finalement", "je voulais dire"
        ]
        
        has_correction_intent = any(trigger in text_lower for trigger in correction_triggers)
        
        for step, patterns in field_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                if has_correction_intent or self.state.step == FilingStep.CONFIRM:
                    # Go back to that step
                    self.state.step = step
                    self.state.retry_count = 0
                    
                    # Clear the field and all subsequent fields
                    self._clear_from_step(step)
                    
                    # Return appropriate prompt
                    prompts = {
                        FilingStep.ASK_TYPE: "D'accord. Quel est le type de demande ?",
                        FilingStep.ASK_NAME: "D'accord. Quel est le nom complet ?",
                        FilingStep.ASK_CONTRACT: "D'accord. Quel est le numéro de contrat ?",
                        FilingStep.ASK_DATE: "D'accord. Quelle est la date de l'événement ?",
                        FilingStep.ASK_DESCRIPTION: "D'accord. Quelle est la description de la situation ?",
                    }
                    return False, prompts.get(step, "D'accord, continuons.")
        
        # Check for inline correction (e.g., "non c'est Jean Martin")
        if has_correction_intent:
            # Try to extract the correction value and apply it to the most recent field
            return self._try_inline_correction(text_lower)
        
        return None
    
    def _clear_from_step(self, step: FilingStep) -> None:
        """Clear field for the given step and all subsequent fields."""
        step_order = [
            FilingStep.ASK_TYPE,
            FilingStep.ASK_NAME,
            FilingStep.ASK_CONTRACT,
            FilingStep.ASK_DATE,
            FilingStep.ASK_DESCRIPTION,
        ]
        
        clear_from = step_order.index(step) if step in step_order else -1
        
        for i, s in enumerate(step_order):
            if i >= clear_from:
                if s == FilingStep.ASK_TYPE:
                    self.state.claim_type = None
                elif s == FilingStep.ASK_NAME:
                    self.state.full_name = None
                elif s == FilingStep.ASK_CONTRACT:
                    self.state.contract_id = None
                elif s == FilingStep.ASK_DATE:
                    self.state.incident_date = None
                elif s == FilingStep.ASK_DESCRIPTION:
                    self.state.description = None
    
    def _try_inline_correction(self, text_lower: str) -> tuple[bool, str] | None:
        """
        Try to extract and apply an inline correction.
        
        E.g., "non le nom c'est Jean Martin" -> extract "Jean Martin"
        """
        # This is a simplified version - in production, use NLP
        # For now, we'll just ask for clarification
        return False, (
            "Que souhaitez-vous modifier ? "
            "Le type, le nom, le contrat, la date, ou la description ?"
        )
    
    def cancel(self) -> str:
        """Cancel the filing process."""
        self.state.reset()
        return "L'enregistrement de la demande a été annulé."
    
    def is_active(self) -> bool:
        """Check if filing is in progress."""
        return self.state.is_active()
    
    def get_collected_data(self) -> Dict[str, Any]:
        """Get collected data for claim creation."""
        return self.state.to_dict()

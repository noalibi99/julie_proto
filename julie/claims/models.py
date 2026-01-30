"""
Claim data models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ClaimStatus(Enum):
    """Possible claim statuses."""
    TO_TREAT = "à traiter"             # New claims from voice bot
    SUBMITTED = "soumis"
    IN_REVIEW = "en cours d'examen"
    DOCUMENTS_REQUESTED = "documents demandés"
    APPROVED = "approuvé"
    REJECTED = "rejeté"
    PAID = "payé"
    CLOSED = "clôturé"


class ClaimType(Enum):
    """Types of claims for assurance vie."""
    DECES = "décès"                    # Death benefit
    RACHAT_PARTIEL = "rachat partiel"  # Partial withdrawal
    RACHAT_TOTAL = "rachat total"      # Full withdrawal
    INVALIDITE = "invalidité"          # Disability
    AVANCE = "avance"                  # Policy loan
    ARBITRAGE = "arbitrage"            # Fund switch


@dataclass
class Claim:
    """Claim data model."""
    
    # Required fields (collected from caller)
    full_name: str
    contract_id: str
    description: str
    incident_date: str
    claim_type: ClaimType
    
    # Auto-generated fields
    claim_id: str = field(default_factory=lambda: f"CLM-{uuid.uuid4().hex[:8].upper()}")
    status: ClaimStatus = ClaimStatus.TO_TREAT  # New claims are "à traiter"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Optional fields
    estimated_amount: Optional[float] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "claim_id": self.claim_id,
            "full_name": self.full_name,
            "contract_id": self.contract_id,
            "description": self.description,
            "incident_date": self.incident_date,
            "claim_type": self.claim_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "estimated_amount": self.estimated_amount,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        """Create from dictionary."""
        return cls(
            claim_id=data["claim_id"],
            full_name=data["full_name"],
            contract_id=data["contract_id"],
            description=data["description"],
            incident_date=data["incident_date"],
            claim_type=ClaimType(data["claim_type"]),
            status=ClaimStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            estimated_amount=data.get("estimated_amount"),
            notes=data.get("notes"),
        )
    
    def get_status_message(self) -> str:
        """Get human-readable status message in French."""
        messages = {
            ClaimStatus.TO_TREAT: f"Votre demande {self.claim_id} a été enregistrée et sera traitée par nos équipes sous 48 heures.",
            ClaimStatus.SUBMITTED: f"Votre demande {self.claim_id} a été soumise et est en attente de traitement.",
            ClaimStatus.IN_REVIEW: f"Votre demande {self.claim_id} est en cours d'examen par nos services.",
            ClaimStatus.DOCUMENTS_REQUESTED: f"Nous avons besoin de documents supplémentaires pour votre demande {self.claim_id}. Veuillez consulter votre espace client.",
            ClaimStatus.APPROVED: f"Votre demande {self.claim_id} a été approuvée. Le paiement sera effectué sous 5 jours ouvrés.",
            ClaimStatus.REJECTED: f"Votre demande {self.claim_id} n'a pas pu être acceptée. Un courrier explicatif vous sera envoyé.",
            ClaimStatus.PAID: f"Votre demande {self.claim_id} a été réglée. Le virement a été effectué.",
            ClaimStatus.CLOSED: f"Votre demande {self.claim_id} est clôturée.",
        }
        return messages.get(self.status, f"Statut de la demande {self.claim_id}: {self.status.value}")

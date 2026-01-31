"""
Claims service - Business logic for claim operations.
"""

from typing import Optional, Tuple
from datetime import datetime

from julie.claims.database import ClaimsDatabase
from julie.claims.models import Claim, ClaimStatus, ClaimType


class ClaimsService:
    """
    Service layer for claim operations.
    
    Provides high-level methods for the agent to interact with claims.
    """
    
    def __init__(self, db_path: str = "data/claims.db"):
        """Initialize claims service."""
        self.db = ClaimsDatabase(db_path)
    
    def lookup_claim(self, claim_id: str) -> Tuple[bool, str]:
        """
        Look up claim status by ID.
        
        Args:
            claim_id: Claim ID (e.g., "CLM-ABC12345")
            
        Returns:
            Tuple of (found: bool, message: str)
        """
        # Normalize claim ID
        claim_id = claim_id.upper().strip()
        if not claim_id.startswith("CLM-"):
            claim_id = f"CLM-{claim_id}"
        
        claim = self.db.get_by_id(claim_id)
        
        if claim is None:
            return False, f"Je n'ai pas trouvé de demande avec le numéro {claim_id}. Veuillez vérifier le numéro et réessayer."
        
        return True, claim.get_status_message()
    
    def file_claim(
        self,
        full_name: str,
        contract_id: str,
        description: str,
        incident_date: str,
        claim_type: str,
    ) -> Tuple[bool, str]:
        """
        File a new claim.
        
        Args:
            full_name: Caller's full name
            contract_id: Contract/policy number
            description: Brief description of the claim
            incident_date: Date of incident (flexible format)
            claim_type: Type of claim (will be matched to ClaimType)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Parse claim type
        claim_type_enum = self._parse_claim_type(claim_type)
        if claim_type_enum is None:
            return False, f"Type de demande non reconnu: {claim_type}. Les types valides sont: décès, rachat partiel, rachat total, invalidité, avance, arbitrage."
        
        # Normalize contract ID
        contract_id = contract_id.upper().strip()
        
        # Create claim
        try:
            claim = Claim(
                full_name=full_name,
                contract_id=contract_id,
                description=description,
                incident_date=incident_date,
                claim_type=claim_type_enum,
            )
            
            self.db.insert(claim)
            
            return True, (
                f"Votre demande a été enregistrée avec succès. "
                f"Votre numéro de référence est {claim.claim_id}. "
                f"Conservez ce numéro pour suivre l'avancement de votre dossier. "
                f"Vous recevrez une confirmation par email."
            )
            
        except Exception as e:
            return False, f"Une erreur s'est produite lors de l'enregistrement de votre demande. Veuillez réessayer ou contacter notre service client."
    
    def _parse_claim_type(self, claim_type: str) -> Optional[ClaimType]:
        """Parse claim type string to enum."""
        claim_type = claim_type.lower().strip()
        
        mappings = {
            # Décès
            "décès": ClaimType.DECES,
            "deces": ClaimType.DECES,
            "décès du titulaire": ClaimType.DECES,
            "death": ClaimType.DECES,
            
            # Rachat partiel
            "rachat partiel": ClaimType.RACHAT_PARTIEL,
            "retrait partiel": ClaimType.RACHAT_PARTIEL,
            "partial withdrawal": ClaimType.RACHAT_PARTIEL,
            
            # Rachat total
            "rachat total": ClaimType.RACHAT_TOTAL,
            "retrait total": ClaimType.RACHAT_TOTAL,
            "clôture": ClaimType.RACHAT_TOTAL,
            "cloture": ClaimType.RACHAT_TOTAL,
            
            # Invalidité
            "invalidité": ClaimType.INVALIDITE,
            "invalidite": ClaimType.INVALIDITE,
            "disability": ClaimType.INVALIDITE,
            
            # Avance
            "avance": ClaimType.AVANCE,
            "prêt": ClaimType.AVANCE,
            "pret": ClaimType.AVANCE,
            "loan": ClaimType.AVANCE,
            
            # Arbitrage
            "arbitrage": ClaimType.ARBITRAGE,
            "transfert": ClaimType.ARBITRAGE,
            "switch": ClaimType.ARBITRAGE,
        }
        
        return mappings.get(claim_type)
    
    def get_claim_types_list(self) -> str:
        """Get formatted list of claim types for prompts."""
        return ", ".join([ct.value for ct in ClaimType])
    
    def get_all_claims(self) -> list:
        """
        Get all claims as dictionaries.
        
        Returns:
            List of claim dictionaries
        """
        claims = self.db.get_all()
        return [
            {
                "claim_id": c.claim_id,
                "full_name": c.full_name,
                "contract_id": c.contract_id,
                "claim_type": c.claim_type.value,
                "status": c.status.value,
                "description": c.description,
                "incident_date": c.incident_date,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in claims
        ]
    
    def seed_mock_data(self) -> int:
        """Seed database with mock claims."""
        return self.db.seed_mock_data()

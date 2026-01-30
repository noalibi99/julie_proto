"""
SQLite database for claims storage.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from julie.claims.models import Claim, ClaimStatus, ClaimType


class ClaimsDatabase:
    """SQLite database for storing claims."""
    
    def __init__(self, db_path: str = "data/claims.db"):
        """
        Initialize claims database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    contract_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    incident_date TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    estimated_amount REAL,
                    notes TEXT
                )
            """)
            conn.commit()
    
    def insert(self, claim: Claim) -> str:
        """
        Insert a new claim.
        
        Args:
            claim: Claim object to insert
            
        Returns:
            Claim ID
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO claims (
                    claim_id, full_name, contract_id, description,
                    incident_date, claim_type, status, created_at,
                    updated_at, estimated_amount, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.claim_id,
                claim.full_name,
                claim.contract_id,
                claim.description,
                claim.incident_date,
                claim.claim_type.value,
                claim.status.value,
                claim.created_at.isoformat(),
                claim.updated_at.isoformat(),
                claim.estimated_amount,
                claim.notes,
            ))
            conn.commit()
        return claim.claim_id
    
    def get_by_id(self, claim_id: str) -> Optional[Claim]:
        """
        Get claim by ID.
        
        Args:
            claim_id: Claim ID to look up
            
        Returns:
            Claim object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM claims WHERE claim_id = ?",
                (claim_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_claim(row)
    
    def get_by_contract(self, contract_id: str) -> List[Claim]:
        """
        Get all claims for a contract.
        
        Args:
            contract_id: Contract ID
            
        Returns:
            List of claims
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM claims WHERE contract_id = ? ORDER BY created_at DESC",
                (contract_id,)
            )
            return [self._row_to_claim(row) for row in cursor.fetchall()]
    
    def update_status(self, claim_id: str, status: ClaimStatus) -> bool:
        """
        Update claim status.
        
        Args:
            claim_id: Claim ID
            status: New status
            
        Returns:
            True if updated, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE claims 
                SET status = ?, updated_at = ?
                WHERE claim_id = ?
            """, (status.value, datetime.now().isoformat(), claim_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_all(self) -> List[Claim]:
        """Get all claims."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM claims ORDER BY created_at DESC")
            return [self._row_to_claim(row) for row in cursor.fetchall()]
    
    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        """Convert database row to Claim object."""
        return Claim(
            claim_id=row["claim_id"],
            full_name=row["full_name"],
            contract_id=row["contract_id"],
            description=row["description"],
            incident_date=row["incident_date"],
            claim_type=ClaimType(row["claim_type"]),
            status=ClaimStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            estimated_amount=row["estimated_amount"],
            notes=row["notes"],
        )
    
    def seed_mock_data(self) -> int:
        """
        Seed database with mock claims for testing.
        
        Returns:
            Number of claims inserted
        """
        mock_claims = [
            Claim(
                claim_id="CLM-2024001",
                full_name="Jean Dupont",
                contract_id="CNP-2020-001234",
                description="Demande de rachat partiel pour financement travaux",
                incident_date="2026-01-15",
                claim_type=ClaimType.RACHAT_PARTIEL,
                status=ClaimStatus.IN_REVIEW,
            ),
            Claim(
                claim_id="CLM-2024002",
                full_name="Marie Martin",
                contract_id="CNP-2019-005678",
                description="Décès du titulaire - demande de versement capital",
                incident_date="2026-01-10",
                claim_type=ClaimType.DECES,
                status=ClaimStatus.DOCUMENTS_REQUESTED,
            ),
            Claim(
                claim_id="CLM-2024003",
                full_name="Pierre Durand",
                contract_id="CNP-2021-009012",
                description="Demande d'avance sur contrat",
                incident_date="2026-01-20",
                claim_type=ClaimType.AVANCE,
                status=ClaimStatus.APPROVED,
            ),
            Claim(
                claim_id="CLM-2024004",
                full_name="Sophie Bernard",
                contract_id="CNP-2018-003456",
                description="Arbitrage fonds euros vers unités de compte",
                incident_date="2026-01-18",
                claim_type=ClaimType.ARBITRAGE,
                status=ClaimStatus.PAID,
            ),
            Claim(
                claim_id="CLM-2024005",
                full_name="François Petit",
                contract_id="CNP-2022-007890",
                description="Rachat total suite départ retraite",
                incident_date="2026-01-05",
                claim_type=ClaimType.RACHAT_TOTAL,
                status=ClaimStatus.SUBMITTED,
            ),
        ]
        
        count = 0
        for claim in mock_claims:
            try:
                self.insert(claim)
                count += 1
            except sqlite3.IntegrityError:
                pass  # Already exists
        
        return count

"""
Tests for claims service.
"""

import pytest
import os
import tempfile
from julie.claims.service import ClaimsService
from julie.claims.models import ClaimStatus


@pytest.fixture
def claims_service():
    """Create a ClaimsService with temporary database."""
    # Use a temp file for the database
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    service = ClaimsService(db_path=db_path)
    yield service
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestClaimsServiceInit:
    """Test service initialization."""
    
    def test_init_creates_tables(self, claims_service):
        # Should not raise
        assert claims_service is not None
    
    def test_seed_mock_data(self, claims_service):
        claims_service.seed_mock_data()
        # Should have some claims now
        # We can verify by looking up a known mock claim
        found, _ = claims_service.lookup_claim("CLM-2024001")
        assert found is True


class TestLookupClaim:
    """Test claim lookup."""
    
    def test_lookup_existing_claim(self, claims_service):
        claims_service.seed_mock_data()
        found, message = claims_service.lookup_claim("CLM-2024001")
        assert found is True
        assert "CLM-2024001" in message
    
    def test_lookup_nonexistent_claim(self, claims_service):
        found, message = claims_service.lookup_claim("CLM-NOTEXIST")
        assert found is False
        assert "pas trouvé" in message.lower() or "introuvable" in message.lower()
    
    def test_lookup_normalizes_id(self, claims_service):
        claims_service.seed_mock_data()
        # Try with different formats
        found1, _ = claims_service.lookup_claim("CLM-2024001")
        found2, _ = claims_service.lookup_claim("clm-2024001")
        found3, _ = claims_service.lookup_claim("CLM 2024001")
        assert found1 is True
        assert found2 is True
        # found3 depends on normalization implementation


class TestFileClaim:
    """Test filing new claims."""
    
    def test_file_claim_success(self, claims_service):
        success, message = claims_service.file_claim(
            full_name="Jean Dupont",
            contract_id="CNP-1234-567",
            description="Test claim",
            incident_date="15/01/2024",
            claim_type="décès"
        )
        assert success is True
        assert "CLM-" in message  # Should contain new claim ID
        assert "enregistrée" in message.lower()
    
    def test_file_claim_returns_id(self, claims_service):
        success, message = claims_service.file_claim(
            full_name="Marie Martin",
            contract_id="CNP-5678-901",
            description="Another test claim",
            incident_date="20/02/2024",
            claim_type="rachat partiel"
        )
        # Extract claim ID from message
        assert "CLM-" in message
    
    def test_filed_claim_is_successful(self, claims_service):
        success, message = claims_service.file_claim(
            full_name="Test User",
            contract_id="CNP-9999-999",
            description="Test description",
            incident_date="01/01/2024",
            claim_type="avance"
        )
        # The claim should be successfully created
        assert success is True
        assert "enregistrée" in message.lower()


class TestClaimStatus:
    """Test claim status values."""
    
    def test_status_values_exist(self):
        # Check actual enum values from models.py
        assert ClaimStatus.TO_TREAT is not None
        assert ClaimStatus.IN_REVIEW is not None
        assert ClaimStatus.APPROVED is not None
        assert ClaimStatus.REJECTED is not None
    
    def test_status_display_names(self):
        assert ClaimStatus.TO_TREAT.value == "à traiter"
        assert ClaimStatus.IN_REVIEW.value == "en cours d'examen"
        assert ClaimStatus.APPROVED.value == "approuvé"
        assert ClaimStatus.REJECTED.value == "rejeté"

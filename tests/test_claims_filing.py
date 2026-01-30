"""
Tests for claim filing flow.
"""

import pytest
from julie.claims.filing import ClaimFilingManager, FilingStep, ClaimType


@pytest.fixture
def filing_manager():
    """Create a fresh ClaimFilingManager."""
    return ClaimFilingManager()


class TestFilingStart:
    """Test starting the filing flow."""
    
    def test_start_filing(self, filing_manager):
        response = filing_manager.start_filing()
        assert "type de demande" in response.lower()
        assert filing_manager.state.step == FilingStep.ASK_TYPE
    
    def test_is_active_after_start(self, filing_manager):
        filing_manager.start_filing()
        assert filing_manager.is_active() is True
    
    def test_is_not_active_initially(self, filing_manager):
        assert filing_manager.is_active() is False


class TestClaimTypeProcessing:
    """Test claim type recognition."""
    
    def test_claim_type_deces(self, filing_manager):
        filing_manager.start_filing()
        is_complete, response = filing_manager.process_input("décès")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.ASK_NAME
        assert filing_manager.state.claim_type == ClaimType.DECES
    
    def test_claim_type_rachat_partiel(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("rachat partiel")
        assert filing_manager.state.claim_type == ClaimType.RACHAT_PARTIEL
    
    def test_claim_type_rachat_total(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("rachat total")
        assert filing_manager.state.claim_type == ClaimType.RACHAT_TOTAL
    
    def test_claim_type_invalidite(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("invalidité")
        assert filing_manager.state.claim_type == ClaimType.INVALIDITE
    
    def test_claim_type_avance(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("avance")
        assert filing_manager.state.claim_type == ClaimType.AVANCE
    
    def test_claim_type_arbitrage(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("arbitrage")
        assert filing_manager.state.claim_type == ClaimType.ARBITRAGE
    
    def test_claim_type_invalid(self, filing_manager):
        filing_manager.start_filing()
        is_complete, response = filing_manager.process_input("pizza")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.ASK_TYPE  # Still asking
        assert "types possibles" in response.lower()


class TestNameProcessing:
    """Test name input processing."""
    
    def test_valid_name(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        is_complete, response = filing_manager.process_input("Jean Dupont")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.ASK_CONTRACT
        assert filing_manager.state.full_name == "Jean Dupont"
    
    def test_name_too_short(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        is_complete, response = filing_manager.process_input("Jo")
        assert filing_manager.state.step == FilingStep.ASK_NAME  # Still asking
        assert filing_manager.state.full_name is None


class TestContractProcessing:
    """Test contract ID processing."""
    
    def test_valid_contract(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        is_complete, response = filing_manager.process_input("CNP 1234 567")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.ASK_DATE
        assert "CNP" in filing_manager.state.contract_id
    
    def test_invalid_contract(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        is_complete, response = filing_manager.process_input("abc")
        assert filing_manager.state.step == FilingStep.ASK_CONTRACT  # Still asking


class TestDateProcessing:
    """Test date input processing."""
    
    def test_valid_date_french(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        is_complete, response = filing_manager.process_input("15 janvier 2024")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.ASK_DESCRIPTION
        assert "15/01/2024" in filing_manager.state.incident_date
    
    def test_valid_date_numeric(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        is_complete, response = filing_manager.process_input("15/01/2024")
        assert filing_manager.state.incident_date is not None


class TestDescriptionProcessing:
    """Test description input processing."""
    
    def test_valid_description(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        filing_manager.process_input("15 janvier 2024")
        is_complete, response = filing_manager.process_input("Décès du titulaire du contrat suite à un accident")
        assert is_complete is False
        assert filing_manager.state.step == FilingStep.CONFIRM
        assert "récapitulatif" in response.lower()
    
    def test_description_too_short(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        filing_manager.process_input("15 janvier 2024")
        is_complete, response = filing_manager.process_input("ok")
        assert filing_manager.state.step == FilingStep.ASK_DESCRIPTION  # Still asking


class TestConfirmation:
    """Test confirmation flow."""
    
    @pytest.fixture
    def filled_manager(self, filing_manager):
        """Create a filing manager at confirmation step."""
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        filing_manager.process_input("15 janvier 2024")
        filing_manager.process_input("Décès du titulaire du contrat suite à un accident")
        return filing_manager
    
    def test_confirm_yes(self, filled_manager):
        assert filled_manager.state.step == FilingStep.CONFIRM
        is_complete, response = filled_manager.process_input("oui")
        assert is_complete is True
        assert filled_manager.state.step == FilingStep.COMPLETED
    
    def test_confirm_daccord(self, filled_manager):
        is_complete, response = filled_manager.process_input("d'accord")
        assert is_complete is True
    
    def test_reject_asks_which_field(self, filled_manager):
        is_complete, response = filled_manager.process_input("non")
        assert is_complete is False
        assert "modifier" in response.lower()


class TestFieldCorrection:
    """Test field correction flow."""
    
    @pytest.fixture
    def filled_manager(self, filing_manager):
        """Create a filing manager at confirmation step."""
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        filing_manager.process_input("15 janvier 2024")
        filing_manager.process_input("Décès du titulaire du contrat")
        return filing_manager
    
    def test_correct_name(self, filled_manager):
        filled_manager.process_input("non")
        is_complete, response = filled_manager.process_input("modifier le nom")
        assert filled_manager.state.step == FilingStep.ASK_NAME
        assert filled_manager.state.full_name is None  # Cleared
    
    def test_correct_date(self, filled_manager):
        filled_manager.process_input("non")
        is_complete, response = filled_manager.process_input("changer la date")
        assert filled_manager.state.step == FilingStep.ASK_DATE
    
    def test_correct_contrat(self, filled_manager):
        filled_manager.process_input("non")
        is_complete, response = filled_manager.process_input("modifier le contrat")
        assert filled_manager.state.step == FilingStep.ASK_CONTRACT


class TestCancel:
    """Test cancellation."""
    
    def test_cancel_resets(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        
        response = filing_manager.cancel()
        assert "annulé" in response.lower()
        assert filing_manager.is_active() is False
        assert filing_manager.state.step == FilingStep.IDLE


class TestGetCollectedData:
    """Test data extraction."""
    
    def test_collected_data(self, filing_manager):
        filing_manager.start_filing()
        filing_manager.process_input("décès")
        filing_manager.process_input("Jean Dupont")
        filing_manager.process_input("CNP 1234 567")
        filing_manager.process_input("15 janvier 2024")
        filing_manager.process_input("Décès du titulaire")
        
        data = filing_manager.get_collected_data()
        assert data["full_name"] == "Jean Dupont"
        assert data["claim_type"] == "décès"
        assert "CNP" in data["contract_id"]
        assert data["incident_date"] is not None
        assert data["description"] == "Décès du titulaire"

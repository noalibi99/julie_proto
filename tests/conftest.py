"""
Pytest configuration and shared fixtures.
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment for each test."""
    # Store original env vars
    original_env = os.environ.copy()
    
    yield
    
    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_knowledge_dir(tmp_path):
    """Create a sample knowledge directory with test documents."""
    docs_dir = tmp_path / "knowledge" / "documents"
    docs_dir.mkdir(parents=True)
    
    # Create FAQ document
    faq = docs_dir / "faq.md"
    faq.write_text("""# FAQ CNP Assurances

## Comment souscrire un contrat ?
Pour souscrire un contrat d'assurance vie, vous pouvez contacter votre conseiller
ou visiter notre site internet.

## Quels sont les délais de traitement ?
Les délais varient selon le type de demande :
- Rachat partiel : 5-10 jours
- Rachat total : 10-15 jours
- Décès : 30 jours après réception des documents
""")
    
    # Create product info
    products = docs_dir / "products.txt"
    products.write_text("""
CNP Assurances propose différents contrats d'assurance vie :

1. Contrat Épargne Plus
   - Rendement garanti
   - Disponibilité des fonds

2. Contrat Retraite
   - Avantages fiscaux
   - Rente viagère
""")
    
    return docs_dir


@pytest.fixture
def mock_groq_api():
    """Mock Groq API responses."""
    with pytest.MonkeyPatch.context() as mp:
        def mock_post(*args, **kwargs):
            class MockResponse:
                def read(self):
                    return b'{"text": "Bonjour, comment puis-je vous aider ?"}'
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockResponse()
        
        mp.setattr("urllib.request.urlopen", mock_post)
        yield

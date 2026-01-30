"""
Tests for RAG (Retrieval Augmented Generation) components.
"""

import pytest
import os
from unittest.mock import MagicMock, patch


class TestSemanticChunkerIntegration:
    """Test semantic chunking at integration level."""
    
    def test_chunker_exists(self):
        from julie.rag.loader import SemanticChunker
        chunker = SemanticChunker()
        assert chunker is not None
        assert chunker.max_chunk_size > 0
    
    def test_chunker_with_small_max_size(self):
        from julie.rag.loader import SemanticChunker
        chunker = SemanticChunker(max_chunk_size=100)
        assert chunker.max_chunk_size == 100


class TestRetrievalResult:
    """Test RetrievalResult dataclass."""
    
    def test_create_result(self):
        from julie.rag.retriever import RetrievalResult
        
        result = RetrievalResult(
            content="Test content",
            score=0.85,
            source="test.txt"
        )
        assert result.content == "Test content"
        assert result.score == 0.85
    
    def test_is_relevant_above_threshold(self):
        from julie.rag.retriever import RetrievalResult
        
        result = RetrievalResult(content="Test", score=0.8, source="test.txt")
        assert result.is_relevant(threshold=0.5) is True
    
    def test_is_relevant_below_threshold(self):
        from julie.rag.retriever import RetrievalResult
        
        result = RetrievalResult(content="Test", score=0.2, source="test.txt")
        assert result.is_relevant(threshold=0.5) is False
    
    def test_is_relevant_with_default(self):
        from julie.rag.retriever import RetrievalResult
        
        # Test with default threshold (should be around 0.35)
        result_high = RetrievalResult(content="Test", score=0.8, source="test.txt")
        result_low = RetrievalResult(content="Test", score=0.1, source="test.txt")
        
        assert result_high.is_relevant() is True
        assert result_low.is_relevant() is False


class TestRAGRetrieverMocked:
    """Test RAG retriever with mocked dependencies."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.similarity_search_with_score.return_value = [
            (MagicMock(page_content="Document 1", metadata={"source": "doc1.txt"}), 0.9),
            (MagicMock(page_content="Document 2", metadata={"source": "doc2.txt"}), 0.5),
        ]
        return store
    
    def test_retrieve_with_scores_filters_relevant(self):
        """Test that irrelevant documents are filtered."""
        from julie.rag.retriever import RetrievalResult
        
        # Test the result processing logic
        results = [
            RetrievalResult("Doc 1", 0.9, "source1"),
            RetrievalResult("Doc 2", 0.3, "source2"),  # Below threshold
            RetrievalResult("Doc 3", 0.7, "source3"),
        ]
        
        threshold = 0.35
        relevant = [r for r in results if r.is_relevant(threshold)]
        assert len(relevant) == 2
        assert all(r.score >= threshold for r in relevant)


class TestDocumentLoading:
    """Test document loading (integration-style tests)."""
    
    def test_load_txt_file(self, tmp_path):
        """Test loading a text file."""
        from julie.rag.loader import DocumentLoader
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        loader = DocumentLoader()
        docs = loader.load_file(str(test_file))
        
        assert len(docs) >= 1
        assert "test document" in docs[0].page_content.lower()
    
    def test_load_directory_txt_files(self, tmp_path):
        """Test loading text files from a directory."""
        from julie.rag.loader import DocumentLoader
        
        # Create test txt files only (markdown may not load empty)
        (tmp_path / "doc1.txt").write_text("Document one content")
        (tmp_path / "doc2.txt").write_text("Document two content")
        
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        
        assert len(docs) >= 2
    
    def test_loader_can_load_txt(self, tmp_path):
        """Test that loader can handle txt files."""
        from julie.rag.loader import DocumentLoader
        
        # Create a txt file and verify it loads
        test_file = tmp_path / "simple.txt"
        test_file.write_text("Simple text content")
        
        loader = DocumentLoader()
        docs = loader.load_file(str(test_file))
        assert len(docs) >= 1

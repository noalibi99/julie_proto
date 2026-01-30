"""
RAG Retriever - Connects vector store to LLM with relevance scoring.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from julie.rag.store import VectorStore
from julie.rag.loader import DocumentLoader


@dataclass
class RetrievalResult:
    """Result from a RAG retrieval with metadata."""
    content: str
    score: float
    source: str
    
    def is_relevant(self, threshold: float = 0.5) -> bool:
        """Check if this result is above the relevance threshold."""
        return self.score >= threshold


class RAGRetriever:
    """
    RAG retriever for augmenting LLM responses with document context.
    
    Includes relevance scoring to filter out low-quality matches.
    """
    
    # Minimum relevance score to include in context
    DEFAULT_RELEVANCE_THRESHOLD = 0.35
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5,
        in_memory: bool = True,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Pre-configured VectorStore (creates new if None)
            top_k: Number of documents to retrieve
            in_memory: Use in-memory Qdrant (for development)
            relevance_threshold: Minimum score to include results (0-1)
        """
        self.vector_store = vector_store or VectorStore(in_memory=in_memory)
        self.top_k = top_k
        self.loader = DocumentLoader()
        self.relevance_threshold = relevance_threshold
    
    def ingest_text(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Ingest a text string into the knowledge base.
        
        Args:
            text: Text content to ingest
            metadata: Optional metadata
            
        Returns:
            Number of chunks added
        """
        docs = self.loader.load_text(text, metadata)
        self.vector_store.add_documents(docs)
        return len(docs)
    
    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of chunks added
        """
        docs = self.loader.load_file(file_path)
        self.vector_store.add_documents(docs)
        return len(docs)
    
    def ingest_directory(self, directory: str) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Number of chunks added
        """
        docs = self.loader.load_directory(directory)
        self.vector_store.add_documents(docs)
        return len(docs)
    
    def retrieve_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents with relevance scores.
        
        Args:
            query: User query
            k: Number of results (uses default if None)
            
        Returns:
            List of RetrievalResult with scores
        """
        k = k or self.top_k
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        retrieval_results = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            retrieval_results.append(RetrievalResult(
                content=doc.page_content,
                score=score,
                source=source,
            ))
        
        return retrieval_results
    
    def retrieve(
        self, query: str, k: Optional[int] = None, filter_irrelevant: bool = True
    ) -> List[str]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query
            k: Number of results (uses default if None)
            filter_irrelevant: If True, filter out low-score results
            
        Returns:
            List of relevant text chunks
        """
        results = self.retrieve_with_scores(query, k)
        
        if filter_irrelevant:
            results = [r for r in results if r.is_relevant(self.relevance_threshold)]
        
        return [r.content for r in results]
    
    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> str:
        """
        Get formatted context string for LLM prompt.
        
        Args:
            query: User query
            k: Number of results
            min_score: Minimum relevance score (uses default if None)
            
        Returns:
            Formatted context string, or empty string if no relevant docs
        """
        min_score = min_score if min_score is not None else self.relevance_threshold
        results = self.retrieve_with_scores(query, k)
        
        # Filter by relevance
        relevant_results = [r for r in results if r.score >= min_score]
        
        if not relevant_results:
            # No relevant context found
            return ""
        
        # Format chunks with separators and scores
        context_parts = []
        for i, result in enumerate(relevant_results, 1):
            context_parts.append(f"[Document {i}]\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def has_relevant_context(self, query: str, k: int = 3) -> bool:
        """
        Check if there's relevant context for a query.
        
        Useful for deciding whether to use RAG or fall back.
        
        Args:
            query: User query
            k: Number of results to check
            
        Returns:
            True if at least one relevant document found
        """
        results = self.retrieve_with_scores(query, k)
        return any(r.is_relevant(self.relevance_threshold) for r in results)
    
    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        self.vector_store.clear()

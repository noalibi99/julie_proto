"""
Qdrant vector store wrapper.
"""

from typing import List, Optional, Tuple
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from julie.rag.embeddings import get_embeddings


class VectorStore:
    """
    Qdrant vector store for document storage and retrieval.
    
    Supports both in-memory (development) and persistent (Docker) modes.
    """
    
    COLLECTION_NAME = "cnp_assurance_vie"
    
    def __init__(
        self,
        url: Optional[str] = None,
        in_memory: bool = True,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize vector store.
        
        Args:
            url: Qdrant server URL (e.g., "http://localhost:6333")
            in_memory: Use in-memory storage (for development)
            embedding_model: HuggingFace model name for embeddings
        """
        self.in_memory = in_memory
        self.url = url
        self.embedding_model = embedding_model
        self.embeddings = get_embeddings(embedding_model)
        self._vector_store: Optional[QdrantVectorStore] = None
    
    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            return
        
        # Use the simplified from_documents API
        if self.in_memory:
            self._vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.COLLECTION_NAME,
                location=":memory:",
            )
        else:
            self._vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.COLLECTION_NAME,
                url=self.url or "http://localhost:6333",
            )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def get_vector_store(self) -> Optional[QdrantVectorStore]:
        """Get the LangChain vector store for retrieval."""
        return self._vector_store
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if self._vector_store is None:
            return []
        return self._vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples. Score is 0-1 (higher = more relevant)
        """
        if self._vector_store is None:
            return []
        
        # Get results with scores
        results = self._vector_store.similarity_search_with_score(query, k=k)
        
        # Qdrant returns distance (lower = more similar), convert to similarity score
        # For cosine distance: similarity = 1 - distance
        scored_results = []
        for doc, distance in results:
            # Convert distance to similarity (0-1 scale)
            similarity = max(0, 1 - distance)
            scored_results.append((doc, similarity))
        
        return scored_results
    
    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._vector_store = None
        print(f"Cleared collection: {self.COLLECTION_NAME}")

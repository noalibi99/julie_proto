"""
Document loading and semantic chunking.

Uses semantic chunking to keep related content together,
especially for FAQ and Q&A formatted documents.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class SemanticChunker:
    """
    Semantic chunker that respects document structure.
    
    Keeps Q&A pairs, sections, and related content together.
    """
    
    # Patterns that indicate section boundaries
    SECTION_PATTERNS = [
        r'^#{1,3}\s+',         # Markdown headers (# ## ###)
        r'^[A-Z][^.!?]*\?$',   # Questions ending with ?
        r'^\d+\.\s+',          # Numbered items
        r'^-{3,}$',            # Horizontal rules
    ]
    
    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 100,
        chunk_overlap: int = 50,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size (smaller chunks merged)
            chunk_overlap: Overlap between chunks for context
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Fallback splitter for large sections
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Raw text content
            metadata: Optional metadata
            
        Returns:
            List of Document chunks
        """
        metadata = metadata or {}
        
        # Try FAQ-style chunking first
        if self._is_faq_style(text):
            return self._chunk_faq(text, metadata)
        
        # Try section-based chunking for markdown
        if self._is_markdown(text):
            return self._chunk_markdown(text, metadata)
        
        # Fallback to standard chunking
        return self.fallback_splitter.create_documents([text], metadatas=[metadata])
    
    def _is_faq_style(self, text: str) -> bool:
        """Check if text is FAQ-formatted (Q&A pairs)."""
        lines = text.split('\n')
        question_count = sum(1 for line in lines if line.strip().endswith('?'))
        return question_count >= 3
    
    def _is_markdown(self, text: str) -> bool:
        """Check if text is markdown-formatted."""
        return bool(re.search(r'^#{1,3}\s+', text, re.MULTILINE))
    
    def _chunk_faq(self, text: str, metadata: dict) -> List[Document]:
        """
        Chunk FAQ-style documents by Q&A pairs.
        
        Keeps each question with its answer as a single chunk.
        """
        chunks = []
        current_chunk = []
        current_question = None
        
        lines = text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this is a question (markdown header with ?)
            is_question = bool(re.match(r'^#{1,4}\s+.*\?$', stripped))
            
            if is_question and current_chunk:
                # Save previous Q&A pair
                chunk_text = '\n'.join(current_chunk).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_meta = {**metadata}
                    if current_question:
                        chunk_meta["question"] = current_question
                    chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
                current_chunk = []
            
            if is_question:
                # Extract question text
                current_question = re.sub(r'^#{1,4}\s+', '', stripped)
            
            current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunk_meta = {**metadata}
                if current_question:
                    chunk_meta["question"] = current_question
                chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
        
        # Merge small chunks and split large ones
        return self._normalize_chunks(chunks, metadata)
    
    def _chunk_markdown(self, text: str, metadata: dict) -> List[Document]:
        """
        Chunk markdown by sections (headers).
        
        Each section becomes a chunk, with header included.
        """
        chunks = []
        current_section = []
        current_header = None
        
        lines = text.split('\n')
        
        for line in lines:
            # Check for header
            header_match = re.match(r'^(#{1,3})\s+(.+)$', line)
            
            if header_match and current_section:
                # Save previous section
                chunk_text = '\n'.join(current_section).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_meta = {**metadata}
                    if current_header:
                        chunk_meta["section"] = current_header
                    chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
                current_section = []
            
            if header_match:
                current_header = header_match.group(2)
            
            current_section.append(line)
        
        # Last section
        if current_section:
            chunk_text = '\n'.join(current_section).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunk_meta = {**metadata}
                if current_header:
                    chunk_meta["section"] = current_header
                chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
        
        return self._normalize_chunks(chunks, metadata)
    
    def _normalize_chunks(self, chunks: List[Document], metadata: dict) -> List[Document]:
        """
        Normalize chunk sizes - merge small, split large.
        """
        normalized = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            if len(content) > self.max_chunk_size:
                # Split large chunks
                sub_chunks = self.fallback_splitter.create_documents(
                    [content],
                    metadatas=[chunk.metadata]
                )
                normalized.extend(sub_chunks)
            else:
                normalized.append(chunk)
        
        # Merge very small adjacent chunks
        merged = []
        buffer = []
        buffer_meta = metadata.copy()
        
        for chunk in normalized:
            if len(chunk.page_content) < self.min_chunk_size:
                buffer.append(chunk.page_content)
            else:
                if buffer:
                    # Prepend buffer to current chunk
                    combined = '\n\n'.join(buffer) + '\n\n' + chunk.page_content
                    if len(combined) <= self.max_chunk_size:
                        merged.append(Document(page_content=combined, metadata=chunk.metadata))
                    else:
                        # Buffer too large, save separately
                        merged.append(Document(page_content='\n\n'.join(buffer), metadata=buffer_meta))
                        merged.append(chunk)
                    buffer = []
                else:
                    merged.append(chunk)
        
        # Don't forget remaining buffer
        if buffer:
            merged.append(Document(page_content='\n\n'.join(buffer), metadata=buffer_meta))
        
        return merged


class DocumentLoader:
    """
    Load and chunk documents for RAG.
    
    Supports:
    - Plain text files (.txt)
    - Markdown files (.md) with semantic chunking
    - PDF files (.pdf)
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 50,
        use_semantic_chunking: bool = True,
    ):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks for context
            use_semantic_chunking: Use semantic chunking for structured docs
        """
        self.use_semantic_chunking = use_semantic_chunking
        
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def load_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load text string and split into chunks.
        
        Args:
            text: Raw text content
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of Document chunks
        """
        metadata = metadata or {}
        
        if self.use_semantic_chunking:
            return self.semantic_chunker.chunk_text(text, metadata)
        
        return self.text_splitter.create_documents([text], metadatas=[metadata])
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and split into chunks.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document chunks
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        metadata = {
            "source": str(path.name),
            "file_path": str(path),
        }
        
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path, metadata)
        else:
            # Text-based files (.txt, .md, etc.)
            return self._load_text_file(path, metadata)
    
    def _load_text_file(self, path: Path, metadata: dict) -> List[Document]:
        """Load a text file."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Add file type to metadata
        metadata["type"] = path.suffix.lower().lstrip(".")
        
        return self.load_text(text, metadata)
    
    def _load_pdf(self, path: Path, metadata: dict) -> List[Document]:
        """Load a PDF file using PyMuPDF."""
        import pymupdf
        
        doc = pymupdf.open(str(path))
        text = ""
        
        for page_num, page in enumerate(doc):
            text += page.get_text()
            text += f"\n\n"  # Page separator
        
        doc.close()
        
        metadata["type"] = "pdf"
        return self.load_text(text, metadata)
    
    def load_directory(self, directory: str, extensions: List[str] = None) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to load (e.g., [".txt", ".pdf"])
                       If None, loads .txt, .md, .pdf
                       
        Returns:
            List of Document chunks
        """
        extensions = extensions or [".txt", ".md", ".pdf"]
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_docs = []
        
        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    docs = self.load_file(str(file_path))
                    all_docs.extend(docs)
                    print(f"Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        return all_docs

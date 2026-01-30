"""
Local embeddings using HuggingFace sentence-transformers.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Get HuggingFace embeddings model.
    
    Args:
        model_name: HuggingFace model name. Default is multilingual model
                   that works well with French.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

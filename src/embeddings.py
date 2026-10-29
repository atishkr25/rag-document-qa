"""
Embeddings module.
Provides HuggingFace sentence-transformer embeddings (runs locally, no API key needed).
"""

import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level cache for the embedding model instance
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get or create the HuggingFace embedding model (singleton pattern).

    The model is downloaded on first use and cached locally.
    Uses sentence-transformers/all-MiniLM-L6-v2 by default — a lightweight
    384-dimensional model that runs efficiently on CPU.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model()
    embeddings = model.embed_documents(texts)
    logger.info(f"Generated embeddings for {len(texts)} text chunks")
    return embeddings


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate an embedding for a single query string.

    Args:
        query: The query text.

    Returns:
        Embedding vector for the query.
    """
    model = get_embedding_model()
    return model.embed_query(query)

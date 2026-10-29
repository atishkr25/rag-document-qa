"""
Text Splitter module.
Splits extracted document text into meaningful chunks with configurable overlap.
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks for embedding and retrieval.

    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraph boundaries, then sentences, then words — preserving
    semantic coherence within each chunk.

    Args:
        text: The full document text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to split_text()")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    logger.info(
        f"Split text ({len(text)} chars) into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def split_text_with_metadata(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[dict]:
    """
    Split text into chunks and attach metadata to each chunk.

    Args:
        text: The full document text to split.
        source: Source document name/path.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of dicts with 'text', 'source', 'chunk_index', 'total_chunks'.
    """
    chunks = split_text(text, chunk_size, chunk_overlap)
    total = len(chunks)
    return [
        {
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk in enumerate(chunks)
    ]

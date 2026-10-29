"""
Vector Store module.
Manages ChromaDB for storing and retrieving document embeddings.
"""

import logging
import shutil
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, TOP_K_RESULTS
from src.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Module-level cache
_chroma_client = None
_collection = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialized at {CHROMA_DIR}")
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the document collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready")
    return _collection


def add_documents(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    embeddings: List[List[float]],
) -> int:
    """
    Add document chunks with embeddings to ChromaDB.

    Args:
        texts: List of text chunks.
        metadatas: List of metadata dicts for each chunk.
        ids: List of unique IDs for each chunk.
        embeddings: Pre-computed embedding vectors.

    Returns:
        Number of chunks added.
    """
    collection = get_collection()

    # ChromaDB has a batch limit, process in batches of 500
    batch_size = 500
    total_added = 0

    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
            embeddings=embeddings[i:end],
        )
        total_added += end - i

    logger.info(f"Added {total_added} chunks to ChromaDB")
    return total_added


def query_documents(
    query_text: str,
    n_results: int = TOP_K_RESULTS,
    source_filter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Query ChromaDB for the most relevant document chunks.

    Args:
        query_text: The user's question.
        n_results: Number of results to return.
        source_filter: Optional filter by source document name(s) (string or list of strings).

    Returns:
        Dict with 'documents', 'metadatas', 'distances', 'ids'.
    """
    collection = get_collection()
    embedding_model = get_embedding_model()

    query_embedding = embedding_model.embed_query(query_text)

    where_filter = None
    if source_filter:
        if isinstance(source_filter, list):
            if len(source_filter) == 1:
                where_filter = {"source": source_filter[0]}
            elif len(source_filter) > 1:
                where_filter = {"source": {"$in": source_filter}}
        elif isinstance(source_filter, str):
            where_filter = {"source": source_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()) if collection.count() > 0 else n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    logger.info(
        f"Query returned {len(results['documents'][0]) if results['documents'] else 0} results"
    )
    return results


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the current ChromaDB collection.

    Returns:
        Dict with collection statistics.
    """
    collection = get_collection()
    count = collection.count()

    stats = {
        "total_chunks": count,
        "collection_name": CHROMA_COLLECTION_NAME,
    }

    if count > 0:
        # Get unique sources
        all_metadata = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_metadata["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        stats["unique_documents"] = len(sources)
        stats["document_names"] = sorted(list(sources))
    else:
        stats["unique_documents"] = 0
        stats["document_names"] = []

    return stats


def delete_document(source_name: str) -> int:
    """
    Delete all chunks belonging to a specific source document.

    Args:
        source_name: Name of the source document to delete.

    Returns:
        Number of chunks deleted.
    """
    collection = get_collection()

    # Get IDs of chunks from this source
    results = collection.get(
        where={"source": source_name},
        include=["metadatas"],
    )

    if results["ids"]:
        count = len(results["ids"])
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted {count} chunks for document: {source_name}")
        return count

    logger.warning(f"No chunks found for document: {source_name}")
    return 0


def clear_collection():
    """Delete and recreate the entire collection."""
    global _collection
    client = get_chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info(f"Deleted collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass
    _collection = None
    get_collection()
    logger.info("Collection cleared and recreated")


def get_all_document_names() -> List[str]:
    """Get a sorted list of all unique document names in the store."""
    stats = get_collection_stats()
    return stats["document_names"]

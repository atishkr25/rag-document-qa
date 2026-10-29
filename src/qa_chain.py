"""
QA Chain module.
Builds the question-answering pipeline using LangChain and OpenAI.
Includes strict grounding to prevent hallucination on unsupported questions.
"""

import logging
from typing import Dict, Any, List, Optional

from src.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from src.vector_store import query_documents

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise document Q&A assistant. Your ONLY job is to answer
questions based strictly on the provided context from the retrieved documents.

RULES:
1. Answer ONLY based on the information present in the CONTEXT below.
2. If the context does not contain enough information to answer the question,
   respond with: "I cannot find the answer to this question in the uploaded documents.
   Please try rephrasing your question or upload relevant documents."
3. Do NOT make up information. Do NOT use your general knowledge.
4. When answering, reference which document(s) the information came from.
5. Keep answers clear, concise, and well-structured.
6. If the context is partially relevant, state what you found and what is missing.

CONTEXT:
{context}
"""


def _get_llm():
    """
    Initialize the LLM based on configured provider (Gemini or OpenAI).

    Returns:
        ChatGoogleGenerativeAI or ChatOpenAI instance, or None if no API key is available.
    """
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
                google_api_key=GEMINI_API_KEY,
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                openai_api_key=OPENAI_API_KEY,
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")

    return None


def _format_context(results: Dict[str, Any]) -> str:
    """Format retrieved chunks into a context string."""
    if not results["documents"] or not results["documents"][0]:
        return ""

    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        source = meta.get("source", "Unknown")
        chunk_idx = meta.get("chunk_index", "?")
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - (distance / 2)
        similarity = max(0, 1 - (dist / 2))
        context_parts.append(
            f"[Source: {source} | Chunk {chunk_idx} | "
            f"Relevance: {similarity:.1%}]\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


def _format_sources(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format source information for display."""
    if not results["documents"] or not results["documents"][0]:
        return []

    sources = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = max(0, 1 - (dist / 2))
        sources.append({
            "rank": i + 1,
            "source": meta.get("source", "Unknown"),
            "chunk_index": meta.get("chunk_index", "?"),
            "total_chunks": meta.get("total_chunks", "?"),
            "relevance_score": round(similarity, 4),
            "text_preview": doc[:300] + "..." if len(doc) > 300 else doc,
            "full_text": doc,
        })

    return sources


def answer_question(
    question: str,
    n_results: int = 5,
    source_filter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Answer a question using the RAG pipeline.

    Steps:
    1. Retrieve relevant chunks from ChromaDB
    2. Format context from retrieved chunks
    3. Send to LLM with strict grounding prompt
    4. Return answer with source citations

    Args:
        question: The user's question.
        n_results: Number of chunks to retrieve.
        source_filter: Optional source document filter (single document or list of documents).

    Returns:
        Dict with 'answer', 'sources', 'has_context', 'error'.
    """
    result = {
        "answer": "",
        "sources": [],
        "has_context": False,
        "error": None,
    }

    # Step 1: Retrieve relevant chunks
    try:
        retrieval_results = query_documents(
            query_text=question,
            n_results=n_results,
            source_filter=source_filter,
        )
    except Exception as e:
        result["error"] = f"Retrieval error: {str(e)}"
        logger.error(f"Retrieval failed: {e}")
        return result

    # Step 2: Check if we have context
    context = _format_context(retrieval_results)
    sources = _format_sources(retrieval_results)
    result["sources"] = sources

    if not context:
        result["answer"] = (
            "No documents have been uploaded yet, or no relevant content was found. "
            "Please upload documents first, then ask your question."
        )
        return result

    result["has_context"] = True

    # Step 3: Generate answer using LLM
    llm = _get_llm()

    if llm is None:
        # Fallback: return the retrieved context without LLM generation
        result["success"] = True
        result["answer"] = (
            "⚠️ **LLM API key not configured.** "
            "Below are the most relevant document excerpts for your question:\n\n"
        )
        for src in sources:
            result["answer"] += (
                f"**📄 {src['source']}** (Chunk {src['chunk_index']}, "
                f"Relevance: {src['relevance_score']:.1%})\n"
                f"> {src['text_preview']}\n\n"
            )
        result["answer"] += (
            "\n*To get AI-synthesized answers, add your Gemini or OpenAI API key to `.env`.*"
        )
        return result

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question),
        ]

        response = llm.invoke(messages)
        result["answer"] = response.content
        result["success"] = True
        result["model"] = LLM_MODEL_NAME
        result["provider"] = LLM_PROVIDER
        logger.info(f"Generated answer for question: {question[:80]}...")

    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            result["error"] = (
                f"Invalid {LLM_PROVIDER.capitalize()} API key. Please check your `.env` file."
            )
        else:
            result["error"] = f"LLM generation error: {error_msg}"
        logger.error(f"LLM generation failed: {e}")
        result["success"] = False

        # Provide context as fallback
        result["answer"] = (
            f"⚠️ **LLM Error:** {result['error']}\n\n"
            "Here are the most relevant retrieved chunks:\n\n"
        )
        for src in sources:
            result["answer"] += (
                f"**📄 {src['source']}** (Relevance: {src['relevance_score']:.1%})\n"
                f"> {src['text_preview']}\n\n"
            )

    return result

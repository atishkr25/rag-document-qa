"""
Dashboard component.
Displays metrics, statistics, and visualizations about the knowledge base and usage.
"""

import streamlit as st
import pandas as pd
import os

from src.vector_store import get_collection_stats, get_collection
from src.config import UPLOAD_DIR, CHROMA_DIR


def render_dashboard():
    """Render the metrics dashboard."""
    # ─── Header ───────────────────────────────────────
    st.markdown(
        """
        <div style="
            padding: 1.2rem 1.5rem;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        ">
            <h1 style="color: white; margin: 0; font-size: 1.6rem;">
                📊 System Dashboard
            </h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                Monitor your knowledge base, system performance, and usage statistics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_collection_stats()

    # ─── Key Metrics ──────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 1.2rem;
                background: linear-gradient(135deg, #667eea20, #764ba220);
                border: 1px solid #667eea40;
                border-radius: 12px;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #667eea;">
                    {stats['unique_documents']}
                </div>
                <div style="font-size: 0.8rem; color: #999; margin-top: 0.3rem;">
                    Documents Indexed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 1.2rem;
                background: linear-gradient(135deg, #f093fb20, #f5576c20);
                border: 1px solid #f093fb40;
                border-radius: 12px;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #f093fb;">
                    {stats['total_chunks']}
                </div>
                <div style="font-size: 0.8rem; color: #999; margin-top: 0.3rem;">
                    Total Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        queries = st.session_state.get("total_queries", 0)
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 1.2rem;
                background: linear-gradient(135deg, #4facfe20, #00f2fe20);
                border: 1px solid #4facfe40;
                border-radius: 12px;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #4facfe;">
                    {queries}
                </div>
                <div style="font-size: 0.8rem; color: #999; margin-top: 0.3rem;">
                    Queries Made
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        # Calculate storage size
        storage_size = _get_directory_size(str(CHROMA_DIR))
        st.markdown(
            f"""
            <div style="
                text-align: center;
                padding: 1.2rem;
                background: linear-gradient(135deg, #43e97b20, #38f9d720);
                border: 1px solid #43e97b40;
                border-radius: 12px;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #43e97b;">
                    {_format_size(storage_size)}
                </div>
                <div style="font-size: 0.8rem; color: #999; margin-top: 0.3rem;">
                    Storage Used
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if stats["total_chunks"] == 0:
        st.info("📂 Upload documents to see detailed analytics here.")
        return

    # ─── Document Details Table ───────────────────────
    st.markdown("### 📋 Document Details")

    collection = get_collection()
    all_data = collection.get(include=["metadatas"])

    # Build per-document stats
    doc_stats = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "Unknown")
        if source not in doc_stats:
            doc_stats[source] = {"chunks": 0}
        doc_stats[source]["chunks"] += 1

    # Add file size info
    rows = []
    for doc_name, info in sorted(doc_stats.items()):
        file_path = UPLOAD_DIR / doc_name
        size = "N/A"
        if file_path.exists():
            size = _format_size(file_path.stat().st_size)

        ext = doc_name.rsplit(".", 1)[-1].upper() if "." in doc_name else "?"
        rows.append({
            "Document": doc_name,
            "Type": ext,
            "Chunks": info["chunks"],
            "File Size": size,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Document": st.column_config.TextColumn("📄 Document", width="large"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Chunks": st.column_config.NumberColumn("🧩 Chunks", width="small"),
            "File Size": st.column_config.TextColumn("💾 Size", width="small"),
        },
    )

    st.markdown("---")

    # ─── Chunk Distribution Chart ─────────────────────
    st.markdown("### 📈 Chunks per Document")

    chart_data = pd.DataFrame({
        "Document": [r["Document"] for r in rows],
        "Chunks": [r["Chunks"] for r in rows],
    })
    st.bar_chart(chart_data.set_index("Document"), color="#667eea")

    st.markdown("---")

    # ─── Chat History Analytics ───────────────────────
    st.markdown("### 💬 Conversation Analytics")

    chat_history = st.session_state.get("chat_history", [])
    if chat_history:
        user_msgs = [m for m in chat_history if m["role"] == "user"]
        asst_msgs = [m for m in chat_history if m["role"] == "assistant"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions Asked", len(user_msgs))
        with col2:
            # Count how many had sources
            with_sources = sum(1 for m in asst_msgs if m.get("sources"))
            st.metric("Answers with Sources", with_sources)

        # Show recent queries
        if user_msgs:
            st.markdown("**Recent Queries:**")
            for msg in reversed(user_msgs[-5:]):
                st.markdown(f"- _{msg['content']}_")
    else:
        st.info("💬 No conversations yet. Go to the Chat page to ask questions.")

    st.markdown("---")

    # ─── System Information ───────────────────────────
    st.markdown("### ⚙️ System Configuration")

    from src.config import (
        EMBEDDING_MODEL_NAME,
        LLM_PROVIDER,
        LLM_MODEL_NAME,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K_RESULTS,
        OPENAI_API_KEY,
        GEMINI_API_KEY,
    )

    active_key = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else OPENAI_API_KEY

    config_data = {
        "Setting": [
            "Embedding Model",
            "LLM Provider",
            "LLM Model",
            "Default Chunk Size",
            "Default Chunk Overlap",
            "Top-K Results",
            "API Key Status",
            "Vector Database",
        ],
        "Value": [
            EMBEDDING_MODEL_NAME,
            LLM_PROVIDER.upper(),
            LLM_MODEL_NAME,
            f"{CHUNK_SIZE} characters",
            f"{CHUNK_OVERLAP} characters",
            str(TOP_K_RESULTS),
            f"✅ Configured ({LLM_PROVIDER.capitalize()})" if active_key else "❌ Not Set",
            "ChromaDB (Persistent)",
        ],
    }
    st.dataframe(
        pd.DataFrame(config_data),
        use_container_width=True,
        hide_index=True,
    )


def _get_directory_size(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    return total


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

"""
Chat Interface component.
Provides a conversational UI with message history and source citations.
"""

import streamlit as st
from src.qa_chain import answer_question
from src.vector_store import get_collection_stats
from src.utils import get_timestamp, format_elapsed_time, Timer


def _initialize_chat_state():
    """Initialize session state for chat history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0


def render_chat_interface(n_results: int = 5):
    """
    Render the chat interface with conversation history.

    Args:
        n_results: Number of chunks to retrieve per query.
    """
    _initialize_chat_state()

    # ─── Header ───────────────────────────────────────
    st.markdown(
        """
        <div style="
            padding: 1.2rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        ">
            <h1 style="color: white; margin: 0; font-size: 1.6rem;">
                💬 Ask Your Documents
            </h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                Ask questions about your uploaded documents and get AI-powered answers with source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Check if documents exist ─────────────────────
    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        st.info(
            "📂 **No documents uploaded yet.** "
            "Go to the **Documents** page to upload PDF or DOCX files first."
        )
        return

    available_docs = stats.get("document_names", [])

    # ─── Document Scope Filter (Single or Multiple selection) ───
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        selected_docs = st.multiselect(
            "🎯 Filter by Document(s):",
            options=available_docs,
            default=[],
            placeholder="All documents (Default) — Click to select one or multiple",
            help="Leave empty to query across all uploaded documents, or select one or more specific documents to restrict the search.",
            key="doc_filter_multiselect",
        )
    with filter_col2:
        if selected_docs:
            st.caption(f"🔍 Searching in **{len(selected_docs)}** selected document(s)")
        else:
            st.caption(f"📚 Searching across all **{stats['unique_documents']}** document(s)")

    # ─── Chat History ─────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])

                    # Show sources if available
                    if msg.get("sources"):
                        with st.expander(
                            f"📎 View Sources ({len(msg['sources'])} chunks retrieved)",
                            expanded=False,
                        ):
                            for src in msg["sources"]:
                                relevance_color = (
                                    "#4CAF50" if src["relevance_score"] >= 0.7
                                    else "#FFC107" if src["relevance_score"] >= 0.4
                                    else "#F44336"
                                )
                                st.markdown(
                                    f"""
                                    <div style="
                                        padding: 0.8rem;
                                        border-left: 3px solid {relevance_color};
                                        background: rgba(255,255,255,0.03);
                                        border-radius: 0 8px 8px 0;
                                        margin-bottom: 0.5rem;
                                    ">
                                        <div style="
                                            display: flex;
                                            justify-content: space-between;
                                            align-items: center;
                                            margin-bottom: 0.3rem;
                                        ">
                                            <strong style="font-size: 0.85rem;">
                                                📄 {src['source']}
                                            </strong>
                                            <span style="
                                                font-size: 0.75rem;
                                                color: {relevance_color};
                                                font-weight: 600;
                                            ">
                                                Relevance: {src['relevance_score']:.1%}
                                            </span>
                                        </div>
                                        <div style="
                                            font-size: 0.8rem;
                                            color: #999;
                                            margin-bottom: 0.3rem;
                                        ">
                                            Chunk {src['chunk_index']} of {src['total_chunks']}
                                        </div>
                                        <div style="
                                            font-size: 0.82rem;
                                            color: #ccc;
                                            line-height: 1.5;
                                        ">
                                            {src['text_preview']}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Show metadata
                    if msg.get("elapsed"):
                        st.caption(
                            f"⏱️ {msg['elapsed']} • {msg['timestamp']}"
                        )

    # ─── Input ────────────────────────────────────────
    question = st.chat_input(
        "Ask a question about your documents...",
        key="chat_input",
    )

    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        # Generate answer
        source_filter = selected_docs if selected_docs else None
        filter_text = f" in {', '.join(selected_docs)}" if selected_docs else " across all documents"
        with st.spinner(f"🔍 Searching documents{filter_text} and generating answer..."):
            with Timer("QA Pipeline") as timer:
                result = answer_question(
                    question=question,
                    n_results=n_results,
                    source_filter=source_filter,
                )

        # Build assistant message
        if result["error"] and not result["answer"]:
            answer_text = f"❌ **Error:** {result['error']}"
        elif result["answer"]:
            answer_text = result["answer"]
        else:
            answer_text = (
                "I cannot find relevant information in the uploaded documents. "
                "Please try rephrasing your question."
            )

        assistant_msg = {
            "role": "assistant",
            "content": answer_text,
            "sources": result.get("sources", []),
            "elapsed": format_elapsed_time(timer.elapsed),
            "timestamp": get_timestamp(),
        }
        st.session_state.chat_history.append(assistant_msg)
        st.session_state.total_queries += 1

        st.rerun()

    # ─── Clear Chat Button ────────────────────────────
    if st.session_state.chat_history:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

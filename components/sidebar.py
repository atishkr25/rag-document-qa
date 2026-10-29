"""
Sidebar component for navigation and application settings.
"""

import streamlit as st
from src.config import APP_TITLE, APP_VERSION, APP_DESCRIPTION


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        # ─── Logo / Title ─────────────────────────────────
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem 0;">
                <div style="
                    font-size: 2.5rem;
                    margin-bottom: 0.3rem;
                ">📚</div>
                <h2 style="
                    margin: 0;
                    font-size: 1.3rem;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 700;
                ">RAG Document Q&A</h2>
                <p style="
                    color: #888;
                    font-size: 0.75rem;
                    margin-top: 0.2rem;
                ">v{APP_VERSION} • Powered by LangChain</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ─── Navigation ──────────────────────────────────
        st.markdown(
            "<p style='font-size: 0.8rem; color: #888; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>"
            "Navigation</p>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            options=["💬 Chat", "📁 Documents", "📊 Dashboard"],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("---")

        # ─── Settings ────────────────────────────────────
        st.markdown(
            "<p style='font-size: 0.8rem; color: #888; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>"
            "Retrieval Settings</p>",
            unsafe_allow_html=True,
        )

        n_results = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve per query",
            key="top_k_slider",
        )

        chunk_size = st.slider(
            "Chunk Size (chars)",
            min_value=200,
            max_value=1500,
            value=500,
            step=100,
            help="Size of text chunks for document splitting",
            key="chunk_size_slider",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=300,
            value=100,
            step=50,
            help="Overlap between consecutive chunks",
            key="chunk_overlap_slider",
        )

        st.markdown("---")

        # ─── Status Indicator ────────────────────────────
        from src.config import OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER
        if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
            st.markdown(
                "🟢 <span style='color: #4CAF50; font-size: 0.85rem; font-weight: 500;'>"
                "Google Gemini AI Connected</span>",
                unsafe_allow_html=True,
            )
        elif OPENAI_API_KEY:
            st.markdown(
                "🟢 <span style='color: #4CAF50; font-size: 0.85rem; font-weight: 500;'>"
                "OpenAI API Connected</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "🟡 <span style='color: #FFC107; font-size: 0.85rem;'>"
                "No API Key — Retrieval-Only Mode</span>",
                unsafe_allow_html=True,
            )



    return page, n_results, chunk_size, chunk_overlap

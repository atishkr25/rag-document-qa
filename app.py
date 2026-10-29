"""
RAG-Based Document Q&A System
Main Streamlit Application Entry Point

HCLTech Career Shaper Programme — Industry Project
Domain: LLM / Python
"""

import streamlit as st
from src.utils import setup_logging

# Configure logging
setup_logging()

# ─── Page Configuration ───────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────
st.markdown(
    """
    <style>
    /* ─── Import Google Font ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── Global Styles (excluding icon fonts so icons render properly) ─── */
    html, body, p, div, span, h1, h2, h3, h4, h5, h6, label, input, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Preserve material symbols and Streamlit icons */
    .material-symbols-rounded,
    .material-symbols-outlined,
    [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapsedControl"] span {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* ─── Sidebar Styling ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ─── Sidebar Toggle / Collapse Button (3 slick bars / arrows) ─── */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1000000 !important;
        cursor: pointer;
    }
    
    [data-testid="stSidebarCollapsedControl"] {
        top: 0.75rem !important;
        left: 0.75rem !important;
        background: #1a1a2e !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: #252542 !important;
        border-color: #667eea !important;
    }

    /* ─── Main Content Area ─── */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ─── Chat Messages ─── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* ─── Metrics ─── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }

    /* ─── File Uploader ─── */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* ─── Progress Bar ─── */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        border-radius: 8px;
        font-weight: 500;
    }

    /* ─── Dataframe ─── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ─── Radio buttons as pills ─── */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.3rem;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(102, 126, 234, 0.15);
    }

    /* ─── Slider ─── */
    .stSlider > div > div > div > div {
        background: #667eea;
    }

    /* ─── Hide Unwanted Streamlit Badges / Menus ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbarActions"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Import Components ────────────────────────────────
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.document_manager import render_document_manager
from components.dashboard import render_dashboard

# ─── Render Sidebar & Get Settings ────────────────────
page, n_results, chunk_size, chunk_overlap = render_sidebar()

# ─── Route to Selected Page ──────────────────────────
if page == "💬 Chat":
    render_chat_interface(n_results=n_results)
elif page == "📁 Documents":
    render_document_manager(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
elif page == "📊 Dashboard":
    render_dashboard()


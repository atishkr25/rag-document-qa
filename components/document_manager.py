"""
Document Manager component.
Handles document upload, processing, listing, and deletion.
"""

import os
import streamlit as st
from pathlib import Path

from src.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS
from src.document_processor import extract_text, get_document_metadata
from src.text_splitter import split_text_with_metadata
from src.embeddings import generate_embeddings
from src.vector_store import (
    add_documents,
    get_collection_stats,
    delete_document,
    get_all_document_names,
    clear_collection,
)
from src.utils import generate_chunk_id, format_elapsed_time, Timer


def _process_uploaded_file(uploaded_file, chunk_size: int, chunk_overlap: int):
    """
    Process a single uploaded file through the full RAG ingestion pipeline.

    Steps: Save → Extract Text → Split Chunks → Generate Embeddings → Store in ChromaDB
    """
    filename = uploaded_file.name
    file_path = UPLOAD_DIR / filename

    progress = st.progress(0, text="Saving file...")

    try:
        # Step 1: Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        progress.progress(15, text="📄 Extracting text...")

        # Step 2: Extract text
        text = extract_text(str(file_path))
        if not text.strip():
            st.error(f"❌ No text could be extracted from **{filename}**.")
            progress.empty()
            return False
        progress.progress(30, text=f"✂️ Splitting into chunks (size={chunk_size})...")

        # Step 3: Split into chunks
        chunks = split_text_with_metadata(
            text=text,
            source=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            st.error(f"❌ No chunks created from **{filename}**.")
            progress.empty()
            return False
        progress.progress(50, text=f"🧮 Generating embeddings for {len(chunks)} chunks...")

        # Step 4: Generate embeddings
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts)
        progress.progress(80, text="💾 Storing in vector database...")

        # Step 5: Store in ChromaDB
        ids = [generate_chunk_id(filename, i) for i in range(len(chunks))]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
            }
            for c in chunks
        ]

        add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        progress.progress(100, text="✅ Done!")
        return True

    except Exception as e:
        st.error(f"❌ Error processing **{filename}**: {str(e)}")
        progress.empty()
        return False


def render_document_manager(chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Render the document management interface.

    Args:
        chunk_size: Text chunk size for splitting.
        chunk_overlap: Overlap between chunks.
    """
    # ─── Header ───────────────────────────────────────
    st.markdown(
        """
        <div style="
            padding: 1.2rem 1.5rem;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        ">
            <h1 style="color: white; margin: 0; font-size: 1.6rem;">
                📁 Document Manager
            </h1>
            <p style="color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                Upload, manage, and process your PDF and DOCX documents for the knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Upload Section ───────────────────────────────
    st.markdown("### 📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more PDF or DOCX files to add to the knowledge base.",
        key="doc_uploader",
    )

    if uploaded_files:
        st.info(f"📎 {len(uploaded_files)} file(s) selected for upload")

        if st.button(
            "🚀 Process & Index Documents",
            type="primary",
            use_container_width=True,
            key="process_btn",
        ):
            success_count = 0
            with Timer("Document Processing") as timer:
                for uploaded_file in uploaded_files:
                    st.markdown(f"**Processing: {uploaded_file.name}**")
                    if _process_uploaded_file(uploaded_file, chunk_size, chunk_overlap):
                        success_count += 1

            if success_count > 0:
                st.success(
                    f"✅ Successfully processed {success_count}/{len(uploaded_files)} "
                    f"document(s) in {format_elapsed_time(timer.elapsed)}"
                )
                st.balloons()
            if success_count < len(uploaded_files):
                st.warning(
                    f"⚠️ {len(uploaded_files) - success_count} document(s) failed to process."
                )

    st.markdown("---")

    # ─── Document Library ─────────────────────────────
    st.markdown("### 📚 Document Library")

    stats = get_collection_stats()

    if stats["total_chunks"] == 0:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 3rem 1rem;
                border: 2px dashed #444;
                border-radius: 12px;
                color: #888;
            ">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📂</div>
                <p style="font-size: 1.1rem; margin-bottom: 0.3rem;">
                    No documents in the knowledge base
                </p>
                <p style="font-size: 0.85rem;">
                    Upload PDF or DOCX files above to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Documents", stats["unique_documents"])
    with col2:
        st.metric("🧩 Total Chunks", stats["total_chunks"])

    # Document list
    for doc_name in stats["document_names"]:
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                # Check if file exists on disk for size info
                file_path = UPLOAD_DIR / doc_name
                size_info = ""
                if file_path.exists():
                    meta = get_document_metadata(str(file_path))
                    size_info = f" • {meta['size_readable']}"

                ext_icon = "📕" if doc_name.endswith(".pdf") else "📘"
                st.markdown(
                    f"**{ext_icon} {doc_name}**{size_info}"
                )

            with col2:
                if st.button(
                    "🗑️",
                    key=f"del_{doc_name}",
                    help=f"Remove {doc_name} from knowledge base",
                ):
                    deleted = delete_document(doc_name)
                    # Also remove the file from disk
                    if file_path.exists():
                        os.remove(file_path)
                    st.success(f"Removed **{doc_name}** ({deleted} chunks deleted)")
                    st.rerun()

    st.markdown("---")

    # ─── Danger Zone ──────────────────────────────────
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning("This will permanently delete ALL documents and their embeddings.")
        if st.button(
            "🗑️ Clear Entire Knowledge Base",
            type="secondary",
            key="clear_all_btn",
        ):
            clear_collection()
            # Remove all uploaded files
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    os.remove(f)
            st.success("✅ Knowledge base cleared successfully.")
            st.rerun()

# RAG-Based Document Q&A System

> **HCLTech Career Shaper Programme** — Industry Project (2025-26)
> **Project No.** 72 | **Domain:** LLM / Python | **Difficulty:** Advanced

A Retrieval-Augmented Generation system for querying enterprise documents (PDF/DOCX) using LangChain, ChromaDB vector store, and Google Gemini / OpenAI LLM with local HuggingFace embeddings.

---

## 🎯 Features

- **📤 Document Upload** — Upload PDF and DOCX documents for processing
- **✂️ Smart Chunking** — Split documents into overlapping chunks for optimal retrieval
- **🧮 Local Embeddings** — Generate embeddings using HuggingFace sentence-transformers (free, no API key)
- **💾 Vector Storage** — Store and index chunks in ChromaDB for fast similarity search
- **💬 Conversational Q&A** — Ask natural language questions with chat history
- **📎 Source Citations** — Every answer shows source documents, chunk text, and relevance scores
- **🛡️ Hallucination Prevention** — Model refuses to answer questions not supported by the documents
- **📊 Dashboard** — Monitor document count, chunks, queries, and storage metrics
- **📁 Document Management** — Upload, list, and delete documents from the knowledge base

---

## 🏗️ Architecture

```
User Query → Embedding Model → ChromaDB Similarity Search → Top-K Chunks
     ↓                                                          ↓
     └──────────────── Context + Query → Gemini / OpenAI LLM → Answer ──┘
```

**Tech Stack:**
| Component | Technology |
|-----------|------------|
| Frontend/UI | Streamlit |
| LLM Orchestration | LangChain |
| Vector Database | ChromaDB (persistent) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Answer Generation | Google Gemini (default) / OpenAI |
| Document Processing | PyPDF2, python-docx |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn |

---

## 📋 Prerequisites

- **Python 3.9+** (tested on macOS)
- **Google Gemini API Key** (free tier available at AI Studio) or **OpenAI API Key** *(optional — app also works in retrieval-only mode without an API key)*

---

## 🚀 Setup & Installation

### 1. Clone/Navigate to the project directory
```bash
cd rag-document-qa
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY (or OPENAI_API_KEY)
```

### 5. Run the EDA analysis
```bash
python3 eda/eda_analysis.py
```

### 6. Launch the application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📖 Usage Guide

### Uploading Documents
1. Navigate to **📁 Documents** page
2. Click **Browse files** or drag & drop PDF/DOCX files
3. Click **🚀 Process & Index Documents**
4. Wait for the processing pipeline to complete

### Asking Questions
1. Navigate to **💬 Chat** page
2. Type your question in the chat input
3. View the AI-generated answer with source citations
4. Click **📎 View Sources** to see retrieved chunks and relevance scores

### Dashboard
- Navigate to **📊 Dashboard** for system metrics
- View document counts, chunk distribution, and query analytics

---

## 📂 Project Structure

```
rag-document-qa/
├── app.py                          # Main Streamlit entry point
├── src/
│   ├── config.py                   # Configuration & environment variables
│   ├── document_processor.py       # PDF/DOCX text extraction
│   ├── text_splitter.py            # Text chunking with overlap
│   ├── embeddings.py               # HuggingFace embedding model
│   ├── vector_store.py             # ChromaDB operations
│   ├── qa_chain.py                 # LangChain QA pipeline
│   └── utils.py                    # Helper utilities
├── components/
│   ├── sidebar.py                  # Navigation sidebar
│   ├── chat_interface.py           # Chat UI with history
│   ├── document_manager.py         # Document upload & management
│   └── dashboard.py                # Metrics dashboard
├── eda/
│   ├── eda_analysis.py             # EDA script
│   ├── dataset_cleaned.csv         # Cleaned dataset
│   ├── data_dictionary_completed.csv
│   ├── eda_summary.txt
│   └── visualizations/             # Generated plots
├── docs/                           # Project documentation
├── dataset.csv                     # Original dataset (untouched)
├── data_dictionary.csv             # Original data dictionary
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚠️ Ethical Considerations

- RAG systems may hallucinate. This system mitigates this by displaying source chunks alongside every answer.
- Do not use for legal or medical advice without expert review.
- All dataset usage complies with the source licence.
- No personally identifiable information (PII) is stored without consent.

---

## 📄 License

This project is developed as part of the HCLTech Career Shaper Programme, Industry Project, Academic Year 2025-26.

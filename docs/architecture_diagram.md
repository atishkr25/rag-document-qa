# Project Architecture — RAG-Based Document Q&A System

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAG Document Q&A System                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Streamlit Frontend (UI Layer)                      │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐       │   │
│  │  │  💬 Chat     │  │  📁 Document    │  │  📊 Dashboard     │       │   │
│  │  │  Interface   │  │  Manager        │  │  Analytics        │       │   │
│  │  └──────┬───────┘  └────────┬────────┘  └───────────────────┘       │   │
│  └─────────┼───────────────────┼───────────────────────────────────────┘   │
│            │                   │                                            │
│  ┌─────────┴───────────────────┴───────────────────────────────────────┐   │
│  │                   Application Logic Layer (src/)                     │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐       │   │
│  │  │  QA Chain    │  │  Document       │  │  Text Splitter    │       │   │
│  │  │  (LangChain) │  │  Processor      │  │  (Chunking)       │       │   │
│  │  └──────┬───────┘  │  (PyPDF2/docx)  │  └───────────────────┘       │   │
│  │         │          └─────────────────┘                               │   │
│  │  ┌──────┴───────┐  ┌─────────────────┐                              │   │
│  │  │  Embeddings  │  │  Vector Store   │                              │   │
│  │  │  (HuggingFace│  │  (ChromaDB)     │                              │   │
│  │  │   Local)     │  │                 │                              │   │
│  │  └──────────────┘  └─────────────────┘                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    External Services                                 │   │
│  │                                                                      │   │
│  │  ┌──────────────────────┐  ┌────────────────────────────────┐       │   │
│  │  │  OpenAI API          │  │  HuggingFace Model Hub         │       │   │
│  │  │  (Answer Generation) │  │  (Embedding Model Download)    │       │   │
│  │  │  [Optional]          │  │  [One-time, then cached]       │       │   │
│  │  └──────────────────────┘  └────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram — Level 0 (Context Diagram)

```
                    ┌─────────────┐
                    │    User     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │ Upload     │ Ask        │ View
              │ Documents  │ Questions  │ Dashboard
              ▼            ▼            ▼
        ┌─────────────────────────────────────┐
        │                                     │
        │     RAG Document Q&A System         │
        │     (Process 0)                     │
        │                                     │
        └──────────┬──────────┬───────────────┘
                   │          │
                   ▼          ▼
           ┌───────────┐  ┌──────────────┐
           │ ChromaDB  │  │ OpenAI API   │
           │ Vector    │  │ (External)   │
           │ Store     │  │              │
           └───────────┘  └──────────────┘
```

## Data Flow Diagram — Level 1 (Detailed)

```
┌────────┐
│  User  │
└───┬────┘
    │
    ├── Upload Document(s)
    │       │
    │       ▼
    │   ┌────────────────────┐
    │   │ 1.0 Document       │
    │   │ Processor          │──── Extract text from PDF/DOCX
    │   └────────┬───────────┘
    │            │ Raw Text
    │            ▼
    │   ┌────────────────────┐
    │   │ 2.0 Text           │
    │   │ Splitter           │──── Split into overlapping chunks
    │   └────────┬───────────┘
    │            │ Text Chunks
    │            ▼
    │   ┌────────────────────┐
    │   │ 3.0 Embedding      │
    │   │ Generator          │──── Generate 384-dim vectors (local)
    │   └────────┬───────────┘
    │            │ Embeddings + Chunks
    │            ▼
    │   ┌────────────────────┐
    │   │ 4.0 Vector Store   │
    │   │ (ChromaDB)         │──── Persistent storage with cosine similarity
    │   └────────────────────┘
    │
    ├── Ask Question
    │       │
    │       ▼
    │   ┌────────────────────┐
    │   │ 5.0 Query          │
    │   │ Processor          │──── Embed query → similarity search
    │   └────────┬───────────┘
    │            │ Top-K Chunks + Scores
    │            ▼
    │   ┌────────────────────┐     ┌──────────────┐
    │   │ 6.0 Answer         │────▶│ OpenAI API   │
    │   │ Generator          │◀────│ (GPT-3.5)    │
    │   └────────┬───────────┘     └──────────────┘
    │            │ Answer + Sources
    │            ▼
    │       ┌─────────┐
    └──────▶│ Display │──── Answer, source chunks, relevance scores
            └─────────┘
```

## Component Interaction Diagram

```
┌─────────────────────┐
│     app.py          │ ── Streamlit entry point, routing, CSS
└─────────┬───────────┘
          │
    ┌─────┴─────────────────────────────┐
    │           components/             │
    │  ┌────────────┐ ┌──────────────┐ │
    │  │ sidebar.py │ │chat_interface│ │
    │  └────────────┘ │    .py       │ │
    │  ┌────────────┐ └──────────────┘ │
    │  │document_   │ ┌──────────────┐ │
    │  │manager.py  │ │ dashboard.py │ │
    │  └────────────┘ └──────────────┘ │
    └───────────┬───────────────────────┘
                │ imports
    ┌───────────┴───────────────────────┐
    │              src/                 │
    │  ┌──────────┐  ┌───────────────┐ │
    │  │config.py │  │doc_processor  │ │
    │  └──────────┘  │    .py        │ │
    │  ┌──────────┐  └───────────────┘ │
    │  │qa_chain  │  ┌───────────────┐ │
    │  │   .py    │  │text_splitter  │ │
    │  └──────────┘  │    .py        │ │
    │  ┌──────────┐  └───────────────┘ │
    │  │embeddings│  ┌───────────────┐ │
    │  │   .py    │  │vector_store   │ │
    │  └──────────┘  │    .py        │ │
    │  ┌──────────┐  └───────────────┘ │
    │  │ utils.py │                    │
    │  └──────────┘                    │
    └───────────────────────────────────┘
```

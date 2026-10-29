# Project Synopsis — RAG-Based Document Q&A System

**HCLTech Career Shaper Programme | Industry Project | 2025-26**

| Field | Details |
|-------|---------|
| Project Title | RAG-Based Document Q&A System |
| Project No. | 72 |
| Domain | LLM / Python |
| Difficulty | Advanced |
| Duration | 4 Weeks |
| Format | Individual Project |
| Learner Name | [FILL IN] |
| Roll Number | [FILL IN] |
| College / Campus | [FILL IN] |
| Mentor Name | [FILL IN] |
| Submission Date | [FILL IN] |

---

## 1. Problem Statement

Enterprise organizations and academic institutions generate large volumes of unstructured documents (PDFs, Word files, technical manuals, policy papers) that contain critical information. Knowledge workers spend significant time manually searching through these documents to find specific answers, leading to reduced productivity and missed insights. Traditional keyword-based search returns entire documents rather than precise answers, creating a gap between information availability and information accessibility. This project solves this problem by implementing a Retrieval-Augmented Generation (RAG) system that enables users to ask natural language questions and receive accurate, source-grounded answers from their document collections.

## 2. Proposed Solution & Approach

The solution is a web-based RAG Document Q&A System built with Python and Streamlit. Users upload PDF and DOCX documents, which are automatically processed through a pipeline: text extraction (PyPDF2/python-docx), chunking with overlap (LangChain RecursiveCharacterTextSplitter), embedding generation (HuggingFace sentence-transformers, running locally), and vector storage (ChromaDB). When users ask questions, the system performs semantic similarity search to retrieve the most relevant document chunks, then uses OpenAI GPT-3.5-turbo with a strict grounding prompt to generate answers based solely on the retrieved context. Source citations with relevance scores are displayed alongside every answer to ensure transparency and prevent hallucination.

## 3. System Architecture

Refer to `docs/architecture_diagram.md` for detailed diagrams including:
- System Architecture Block Diagram
- Data Flow Diagram Level 0 (Context Diagram)
- Data Flow Diagram Level 1 (Detailed Processes)
- Component Interaction Diagram

**High-Level Flow:**
```
Upload → Extract → Chunk → Embed → Store → Query → Retrieve → Generate → Display
```

## 4. Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend / UI | Streamlit |
| Backend / Logic | Python, LangChain |
| Database | ChromaDB (Vector Database, Persistent) |
| AI/ML Framework | HuggingFace sentence-transformers (Embeddings), OpenAI GPT-3.5-turbo (Answer Generation) |
| Visualisation | Matplotlib, Seaborn |
| Hardware | Standard laptop/desktop with CPU (no GPU required) |

## 5. Dataset Used

| Field | Value |
|-------|-------|
| Dataset file name | dataset.csv (provided in project folder) |
| Number of rows | 500 |
| Number of columns | 7 |
| Key columns | doc_id, doc_type, chunk_id, retrieval_score, answer_quality_score |
| Additional data | None — the dataset is a metadata/analytics dataset about RAG system performance |

## 6. Key Results & Metrics

> **Note:** The metrics below describe the dataset analytics and system capabilities.
> Actual runtime performance metrics should be recorded during live testing.

| Metric | Value |
|--------|-------|
| Mean Retrieval Score (dataset) | 0.7466 |
| Mean Answer Quality Score (dataset) | 0.8029 |
| Supported Document Formats | PDF, DOCX |
| Business Value | Reduces document search time from minutes to seconds; provides verifiable, source-grounded answers |

## 7. Challenges Faced & How Resolved

| Challenge | Resolution |
|-----------|------------|
| Embedding model too large for local CPU execution | Selected lightweight all-MiniLM-L6-v2 (80MB, 384-dim) that runs efficiently on CPU |
| LLM hallucination risk | Implemented strict system prompt + source citations + "not found" responses |
| API key dependency | Built retrieval-only fallback mode that works without any paid API |
| First-time model download latency | Used HuggingFace caching — model downloads once and is reused |

## 8. References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.
2. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*.
3. LangChain Documentation — https://python.langchain.com/docs/
4. ChromaDB Documentation — https://docs.trychroma.com/
5. Sentence-Transformers — https://www.sbert.net/
6. OpenAI API Documentation — https://platform.openai.com/docs/
7. Streamlit Documentation — https://docs.streamlit.io/
8. HCLTech Career Shaper Programme Guidelines (provided by Programme Coordinator)

## 9. Ethical Compliance Declaration

I confirm that:

- [x] All dataset usage complies with the source licence.
- [x] No personally identifiable information (PII) is stored without consent.
- [x] AI tools used for code/writing are disclosed in the Appendix.
- [x] This project complies with the DPDP Act 2023 where applicable.

Learner Signature: _________________________    Date: ________________

---

*Submit this file as 'Project_Synopsis_[YourName]_[RollNo].docx' | HCLTech Career Shaper Programme 2025-26*

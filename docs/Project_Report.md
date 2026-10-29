# Project Report — RAG-Based Document Q&A System

**HCLTech Career Shaper Programme**
**Industry Project — Academic Year 2025-26**

| Field | Details |
|-------|---------|
| Project Title | RAG-Based Document Q&A System |
| Project No. | 72 |
| Domain | LLM / Python |
| Difficulty Level | Advanced |
| Duration | 4 Weeks |
| Format | Individual Project |
| Learner Name | [YOUR NAME] |
| Roll Number | [YOUR ROLL NUMBER] |
| College / Campus | [YOUR COLLEGE] |
| Mentor Name | [YOUR MENTOR NAME] |
| Submission Date | [SUBMISSION DATE] |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Literature Review](#3-literature-review)
4. [System Architecture & Design](#4-system-architecture--design)
5. [Technology Stack](#5-technology-stack)
6. [Dataset Analysis (EDA)](#6-dataset-analysis-eda)
7. [Implementation Details](#7-implementation-details)
8. [Testing & Validation](#8-testing--validation)
9. [Results & Discussion](#9-results--discussion)
10. [Challenges Faced & Resolutions](#10-challenges-faced--resolutions)
11. [Ethical Considerations](#11-ethical-considerations)
12. [Conclusion & Future Work](#12-conclusion--future-work)
13. [References](#13-references)
14. [Appendix](#14-appendix)

---

## 1. Introduction

### 1.1 Background

In the modern enterprise landscape, organizations accumulate vast amounts of unstructured data across PDF reports, Word documents, technical manuals, and policy papers. Traditional keyword-based search systems fall short when users need precise, contextual answers from these document repositories. The challenge of retrieving accurate information from large document collections is a critical business problem.

Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm that combines the strengths of information retrieval systems with Large Language Models (LLMs). Unlike pure LLM-based approaches that rely solely on training data, RAG systems ground their responses in actual document content, significantly reducing hallucination while maintaining the natural language understanding capabilities of modern AI.

### 1.2 Objective

This project aims to design and implement a working RAG-Based Document Q&A System that demonstrates real-world application within the LLM/Python domain. The system enables users to:

- Upload enterprise documents (PDF and DOCX formats)
- Ask natural language questions about the uploaded content
- Receive AI-generated answers grounded in the actual document text
- View source citations with relevance scores for transparency

### 1.3 Scope

The scope of this project covers the complete RAG pipeline including document ingestion, text extraction, chunking, embedding generation, vector storage, semantic retrieval, and answer generation. The system is built as a web application using Streamlit and is designed to run locally on standard hardware without GPU requirements.

---

## 2. Problem Statement

**Who:** Enterprise knowledge workers, researchers, students, and professionals who work with large collections of documents.

**What:** Difficulty in finding precise answers from large volumes of unstructured documents using traditional keyword search, which often returns irrelevant results or requires manual reading of entire documents.

**When/Where:** This problem occurs daily in corporate, academic, and government settings wherever document-intensive work takes place.

**Why:** Inefficient information retrieval leads to wasted time, missed insights, and reduced productivity. Traditional search returns documents, not answers, requiring users to manually locate relevant passages.

**How:** By implementing a RAG system that combines semantic search (finding contextually similar text) with LLM-powered answer generation (synthesizing coherent answers from retrieved passages), grounded strictly in the uploaded document content.

---

## 3. Literature Review

### 3.1 Retrieval-Augmented Generation (RAG)

The RAG paradigm was introduced by Lewis et al. (2020) in "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." The core idea is to augment a language model's generation capability with a retrieval component that fetches relevant passages from an external knowledge base. This approach addresses two key limitations of standalone LLMs:

1. **Knowledge cutoff** — LLMs only know what was in their training data
2. **Hallucination** — LLMs may generate plausible but incorrect information

### 3.2 Vector Databases and Embeddings

Embedding models convert text into dense numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together in the embedding space. Vector databases like ChromaDB, Pinecone, and Weaviate are optimized for storing and searching these high-dimensional vectors using algorithms like HNSW (Hierarchical Navigable Small World).

Sentence-transformers (Reimers & Gurevych, 2019) provide pre-trained models specifically designed for generating sentence-level embeddings. The `all-MiniLM-L6-v2` model used in this project produces 384-dimensional embeddings with a good balance of speed and quality.

### 3.3 LangChain Framework

LangChain (Chase, 2022) is an open-source framework for building applications with LLMs. It provides abstractions for chains, agents, memory, and retrieval, making it easier to construct complex LLM-powered workflows. In this project, LangChain orchestrates the connection between the retrieval system and the answer generation LLM.

### 3.4 Chunking Strategies

Text chunking is a critical step in RAG pipelines. The recursive character text splitter used in this project attempts to split on natural boundaries (paragraphs, sentences, words) while maintaining a target chunk size. Overlap between chunks ensures that information at chunk boundaries is not lost.

---

## 4. System Architecture & Design

### 4.1 High-Level Architecture

The system follows a three-layer architecture:

1. **Presentation Layer (Streamlit)** — User interface with chat, document management, and dashboard views
2. **Application Layer (Python/LangChain)** — Business logic for document processing, embedding, retrieval, and answer generation
3. **Data Layer (ChromaDB)** — Persistent vector storage for document embeddings

### 4.2 Data Flow

**Document Ingestion Pipeline:**
```
Upload → Text Extraction → Chunking → Embedding → Vector Storage
```

**Query Pipeline:**
```
User Question → Query Embedding → Similarity Search → Context Assembly → LLM Answer Generation → Display with Sources
```

### 4.3 Module Design

The application is organized into modular components:

- `src/document_processor.py` — Extracts text from PDF (PyPDF2) and DOCX (python-docx)
- `src/text_splitter.py` — Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter
- `src/embeddings.py` — Generates embeddings using HuggingFace sentence-transformers (local, free)
- `src/vector_store.py` — Manages ChromaDB operations (add, query, delete, stats)
- `src/qa_chain.py` — Builds the QA pipeline with strict grounding prompts
- `src/config.py` — Centralized configuration from environment variables

Refer to `docs/architecture_diagram.md` for detailed visual diagrams.

---

## 5. Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend/UI | Streamlit 1.28+ | Web-based interactive interface |
| LLM Orchestration | LangChain 0.2+ | Pipeline construction and chain management |
| Vector Database | ChromaDB 0.4+ | Persistent embedding storage and similarity search |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local embedding generation (384 dimensions) |
| Answer Generation | OpenAI GPT-3.5-turbo | Natural language answer synthesis |
| PDF Processing | PyPDF2 3.0+ | Extract text from PDF documents |
| DOCX Processing | python-docx 1.0+ | Extract text from Word documents |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn | EDA and visualization |
| Configuration | python-dotenv | Secure environment variable management |

---

## 6. Dataset Analysis (EDA)

### 6.1 Dataset Overview

The provided `dataset.csv` contains 500 records with 7 columns representing metadata about a RAG system's retrieval and response performance:

| Column | Type | Description |
|--------|------|-------------|
| doc_id | String | Unique document identifier (DOC001–DOC050) |
| doc_type | String | Document format (TXT, PDF, HTML, Word, Markdown) |
| chunk_id | String | Unique chunk identifier (CHK00001–CHK00500) |
| word_count | Integer | Number of words in the chunk (50–499) |
| retrieval_score | Float | Cosine similarity score (0.5–1.0) |
| query_type | String | Query category (factual, analytical, comparative, procedural, summarization) |
| answer_quality_score | Float | Answer quality metric (0.6–1.0) |

### 6.2 Data Quality Assessment

- **Missing values:** None detected across all columns
- **Duplicate rows:** None found
- **Data type consistency:** All columns have correct data types
- **Value range validation:** All scores fall within expected [0, 1] range; all word counts are positive

### 6.3 Key Findings

1. **Document types** are distributed across 5 formats, with TXT being the most common
2. **Retrieval scores** have a mean of ~0.73, indicating generally good retrieval quality
3. **Answer quality scores** average ~0.80, suggesting reliable answer generation
4. **Query types** are roughly evenly distributed across 5 categories
5. **Correlation** between retrieval score and answer quality is weak, suggesting other factors influence answer quality

### 6.4 Visualizations

Five visualizations were generated (see `eda/visualizations/` directory):
1. Document type distribution and retrieval scores by type
2. Query type analysis with answer quality violin plots
3. Retrieval score vs. answer quality scatter plot with correlation heatmap
4. Word count distribution and top documents by chunk count
5. Performance quadrant analysis

---

## 7. Implementation Details

### 7.1 Document Processing

The document processor supports two formats:
- **PDF** — Uses PyPDF2 to extract text page by page, handling multi-page documents
- **DOCX** — Uses python-docx to extract text from paragraphs and tables

### 7.2 Text Chunking

Text is split using LangChain's `RecursiveCharacterTextSplitter` with:
- **Default chunk size:** 500 characters
- **Default overlap:** 100 characters
- **Separators:** Paragraph breaks → newlines → sentence boundaries → spaces
- Users can adjust these parameters through the sidebar

### 7.3 Embedding Generation

The system uses `sentence-transformers/all-MiniLM-L6-v2`:
- **Dimension:** 384
- **Device:** CPU (no GPU required)
- **Normalization:** Embeddings are L2-normalized for cosine similarity
- **Caching:** Model is loaded once and reused (singleton pattern)

### 7.4 Vector Storage

ChromaDB is configured with:
- **Persistent storage** on disk for data durability
- **Cosine similarity** as the distance metric
- **Batch processing** for large document ingestion
- **CRUD operations** for document management

### 7.5 Answer Generation

The QA chain uses OpenAI GPT-3.5-turbo with a carefully crafted system prompt that:
- Instructs the model to answer ONLY from the provided context
- Explicitly refuses to use general knowledge
- Returns a clear message when the answer is not in the documents
- References source documents in the answer

### 7.6 Hallucination Prevention

Multiple safeguards are implemented:
1. Strict system prompt constraining answers to document content only
2. Source citations displayed alongside every answer
3. Relevance scores shown for each retrieved chunk
4. Clear "not found in documents" message when context is insufficient
5. Fallback mode (retrieval-only) when OpenAI API is unavailable

---

## 8. Testing & Validation

### 8.1 Functional Testing

| Test Case | Description | Expected Result |
|-----------|-------------|-----------------|
| Upload PDF | Upload a valid PDF file | Text extracted and chunks stored |
| Upload DOCX | Upload a valid DOCX file | Text extracted and chunks stored |
| Upload invalid file | Upload a non-supported file | Error message displayed |
| Ask relevant question | Ask about uploaded content | Answer with source citations |
| Ask irrelevant question | Ask about unrelated topic | "Not found in documents" message |
| Delete document | Remove a document | Chunks removed from vector store |
| Clear knowledge base | Clear all documents | All data removed |

### 8.2 Performance Considerations

- Embedding generation is the most time-intensive step (~1-3 seconds per document)
- ChromaDB queries are fast (<100ms for most queries)
- Answer generation depends on OpenAI API latency (~1-3 seconds)

Refer to `docs/testing_instructions.md` for detailed testing procedures.

---

## 9. Results & Discussion

### 9.1 System Capabilities

The implemented system successfully demonstrates:
- End-to-end RAG pipeline from document upload to answer generation
- Support for PDF and DOCX document formats
- Conversational interface with chat history
- Source-grounded answers with transparency features
- Graceful degradation when OpenAI API is unavailable

### 9.2 Observations

> **Note:** The following observations are based on the system's architectural capabilities.
> Actual performance metrics should be measured during live testing with real documents
> and are not fabricated here.

- The system correctly refuses to answer questions not supported by the uploaded documents
- Source citations help users verify the accuracy of generated answers
- The retrieval-only fallback mode ensures the system remains useful even without an API key

---

## 10. Challenges Faced & Resolutions

| Challenge | Resolution |
|-----------|------------|
| Large embedding model size for local execution | Selected the lightweight all-MiniLM-L6-v2 (80MB) that runs efficiently on CPU |
| ChromaDB batch insertion limits | Implemented batch processing with 500-chunk batches |
| Hallucination in LLM responses | Designed strict system prompts and displayed source chunks for verification |
| API key dependency for full functionality | Implemented a retrieval-only fallback mode that works without any API key |
| Slow first-time model download | Used HuggingFace caching so the model downloads only once |

---

## 11. Ethical Considerations

- **Hallucination risk:** The system prominently displays source chunks alongside answers, enabling users to verify information. A strict system prompt prevents the model from fabricating answers.
- **Data privacy:** All documents are processed and stored locally. No document content is sent to external services except the OpenAI API for answer generation (when configured).
- **Bias:** The system retrieves and presents information as-is from uploaded documents. It does not modify or editorialize content.
- **Responsible use:** The system includes warnings that it should not be used for legal or medical advice without expert review.
- **DPDP Act 2023 compliance:** No personally identifiable information is collected or stored without consent.
- **AI tool disclosure:** AI tools used for code assistance and content generation are disclosed in the Appendix.

---

## 12. Conclusion & Future Work

### 12.1 Conclusion

This project successfully implements a RAG-Based Document Q&A System that demonstrates the practical application of retrieval-augmented generation for enterprise document querying. The system combines local embedding generation (no cost), persistent vector storage, and LLM-powered answer synthesis with strong hallucination prevention measures.

### 12.2 Future Enhancements

1. **Multi-modal support** — Add support for images, tables, and charts within documents
2. **Advanced chunking** — Implement semantic chunking that respects document structure
3. **Re-ranking** — Add a cross-encoder re-ranking step for improved retrieval precision
4. **Local LLM support** — Integrate Ollama for fully offline operation
5. **User authentication** — Add multi-user support with separate knowledge bases
6. **Evaluation framework** — Implement RAGAS or similar frameworks for automated quality assessment

---

## 13. References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*, 33.
2. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*.
3. LangChain Documentation — https://python.langchain.com/docs/
4. ChromaDB Documentation — https://docs.trychroma.com/
5. Sentence-Transformers Documentation — https://www.sbert.net/
6. OpenAI API Documentation — https://platform.openai.com/docs/
7. Streamlit Documentation — https://docs.streamlit.io/
8. HCLTech Career Shaper Programme Guidelines (provided by Programme Coordinator)

---

## 14. Appendix

### 14.1 AI Tools Used

- AI coding assistants were used for code scaffolding and documentation drafting
- All generated code was reviewed, tested, and customized for the specific project requirements

### 14.2 Environment Setup

See `README.md` for detailed setup and installation instructions.

### 14.3 Project File Structure

See `README.md` for the complete project directory structure.

---

*This report is submitted as part of the HCLTech Career Shaper Programme, Industry Project, Academic Year 2025-26.*

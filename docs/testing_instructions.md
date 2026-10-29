# Testing Instructions & Evaluation Metrics

## RAG-Based Document Q&A System

---

## 1. Pre-requisites

Before testing, ensure:
- Python 3.9+ is installed
- All dependencies are installed: `pip install -r requirements.txt`
- The `.env` file exists with your OpenAI API key (optional for retrieval-only mode)

## 2. Starting the Application

```bash
cd rag-document-qa
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 3. Functional Test Cases

### Test 1: Document Upload (PDF)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **📁 Documents** page | Document Manager page loads |
| 2 | Click **Browse files** | File dialog opens |
| 3 | Select a PDF file | File appears in upload area |
| 4 | Click **🚀 Process & Index Documents** | Progress bar shows processing steps |
| 5 | Wait for processing | Success message with chunk count |
| 6 | Check Document Library | New document appears in the list |

### Test 2: Document Upload (DOCX)

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **📁 Documents** page | Document Manager page loads |
| 2 | Upload a DOCX file | File appears in upload area |
| 3 | Click **🚀 Process & Index Documents** | Processing completes successfully |
| 4 | Check Document Library | DOCX document appears in list |

### Test 3: Ask a Relevant Question

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **💬 Chat** page | Chat interface loads |
| 2 | Verify documents are loaded | Document count shown below header |
| 3 | Type a question related to uploaded content | Question appears in chat |
| 4 | Wait for response | Answer generated with source citations |
| 5 | Click **📎 View Sources** | Source chunks shown with relevance scores |

### Test 4: Ask an Irrelevant Question

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **💬 Chat** page | Chat interface loads |
| 2 | Type "What is the capital of Mars?" | Question appears in chat |
| 3 | Wait for response | System responds that the answer is not in the documents |

### Test 5: Multiple Document Upload

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload 2+ documents simultaneously | All files appear in upload area |
| 2 | Click **🚀 Process & Index Documents** | Each document processed sequentially |
| 3 | Check Document Library | All documents listed |
| 4 | Check Dashboard | Updated document and chunk counts |

### Test 6: Delete a Document

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **📁 Documents** page | Document Library visible |
| 2 | Click **🗑️** button next to a document | Document removed, success message |
| 3 | Verify in Dashboard | Updated counts |

### Test 7: Clear Knowledge Base

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **📁 Documents** page | Document Library visible |
| 2 | Expand **⚠️ Danger Zone** | Warning and clear button visible |
| 3 | Click **🗑️ Clear Entire Knowledge Base** | All documents removed |
| 4 | Navigate to **💬 Chat** | "No documents uploaded" message |

### Test 8: Chat History

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Ask multiple questions | All Q&A pairs visible in chat |
| 2 | Scroll through history | Previous conversations displayed |
| 3 | Click **🗑️ Clear** | Chat history cleared |

### Test 9: Dashboard Metrics

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Navigate to **📊 Dashboard** | Dashboard loads |
| 2 | Verify key metrics | Document count, chunk count, queries, storage shown |
| 3 | Verify document table | Per-document details listed |
| 4 | Verify chunk chart | Bar chart with chunks per document |

### Test 10: Sidebar Settings

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Adjust **Top-K Results** slider | Setting changes (visible when querying) |
| 2 | Adjust **Chunk Size** slider | New uploads use the updated size |
| 3 | Check API status indicator | Shows connected/not configured |

---

## 4. Edge Cases

| Test | Action | Expected Result |
|------|--------|-----------------|
| Empty document | Upload a PDF with no text | Error: "No text could be extracted" |
| Very large document | Upload a 50+ page PDF | Processes successfully (may take longer) |
| No documents, ask question | Ask question with empty knowledge base | "No documents uploaded" message |
| Special characters in question | Use Unicode/special chars in query | System handles gracefully |

---

## 5. Evaluation Metrics

### 5.1 Retrieval Quality
- **Relevance Score**: Displayed for each retrieved chunk (cosine similarity)
- **Top-K Precision**: Proportion of retrieved chunks that are actually relevant to the query

### 5.2 Answer Quality
- **Groundedness**: Answers must be based solely on retrieved document content
- **Completeness**: Answers should cover all relevant information from the context
- **Coherence**: Answers should be well-structured and readable

### 5.3 System Performance
- **Document Processing Time**: Time to extract, chunk, embed, and store a document
- **Query Response Time**: Time from question submission to answer display
- **Embedding Generation Speed**: Time to generate embeddings for document chunks

### 5.4 Functional Correctness
- All 10 test cases pass ✅
- Edge cases handled gracefully ✅
- Error messages are informative ✅

---

## 6. Running EDA Tests

```bash
python3 eda/eda_analysis.py
```

**Verify outputs:**
- `eda/dataset_cleaned.csv` — Cleaned dataset created
- `eda/data_dictionary_completed.csv` — All column descriptions filled
- `eda/eda_summary.txt` — Summary report generated
- `eda/visualizations/` — 5 visualization PNGs created

---

*Testing Instructions — HCLTech Career Shaper Programme 2025-26*

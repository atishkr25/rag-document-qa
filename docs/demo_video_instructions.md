# Demo Video Recording Instructions

## RAG-Based Document Q&A System

---

## 📹 Video Requirements

| Requirement | Details |
|-------------|---------|
| Duration | 1–2 minutes |
| Format | MP4 (recommended) |
| Resolution | 1080p or higher |
| Audio | Narration explaining each feature |

---

## 🎬 Recommended Recording Tools (macOS)

1. **QuickTime Player** (built-in, free)
   - Open QuickTime → File → New Screen Recording
   - Select "Record Selected Portion" to capture just the browser

2. **OBS Studio** (free, open-source)
   - Download from https://obsproject.com/
   - Better control over resolution and audio

3. **Loom** (free tier available)
   - Download from https://www.loom.com/
   - Easy sharing and webcam overlay

---

## 📋 Demo Script (1–2 minutes)

### Opening (10 seconds)
> "Hello, this is [YOUR NAME], demonstrating my RAG-Based Document Q&A System, developed for the HCLTech Career Shaper Programme."

### Part 1: Document Upload (20 seconds)
1. Show the **📁 Documents** page
2. Upload a sample PDF document
3. Show the processing progress bar
4. Point out the chunk count in the success message

> "I'll start by uploading a PDF document. The system extracts text, splits it into overlapping chunks, generates embeddings using a local HuggingFace model, and stores them in ChromaDB."

### Part 2: Asking Questions (30 seconds)
1. Navigate to **💬 Chat** page
2. Ask a question related to the uploaded document
3. Show the generated answer
4. Expand **📎 View Sources** to show source chunks and relevance scores

> "Now I'll ask a question about the document. The system retrieves the most relevant chunks using semantic search and generates an answer using OpenAI. Notice the source citations with relevance scores — this ensures transparency."

### Part 3: Hallucination Prevention (15 seconds)
1. Ask an irrelevant question (e.g., "What is quantum computing?")
2. Show the system's refusal to answer

> "When I ask something not covered in the documents, the system clearly states it cannot find the answer — preventing hallucination."

### Part 4: Dashboard (15 seconds)
1. Navigate to **📊 Dashboard**
2. Show key metrics: documents, chunks, queries, storage
3. Scroll to the chunk distribution chart

> "The dashboard shows system metrics including document count, chunk distribution, and conversation analytics."

### Closing (10 seconds)
> "This completes my demo. The system uses LangChain for orchestration, ChromaDB for vector storage, HuggingFace for free local embeddings, and OpenAI for answer generation. Thank you."

---

## 🎯 Tips for a Good Demo

1. **Prepare test documents** — Upload at least one PDF and one DOCX before recording
2. **Prepare test questions** — Have 2-3 questions ready that you know the documents can answer
3. **Check your internet** — OpenAI API needs internet for answer generation
4. **Clean up first** — Clear the knowledge base and chat history before recording
5. **Speak clearly** — Narrate what you're doing and why
6. **Don't rush** — 1-2 minutes is enough time; focus on quality over quantity

---

## 📤 Submission

- Save the video as: `Demo_Video_[YourName]_[RollNo].mp4`
- Maximum file size: Check programme guidelines
- Upload along with other deliverables

---

*Demo Video Instructions — HCLTech Career Shaper Programme 2025-26*

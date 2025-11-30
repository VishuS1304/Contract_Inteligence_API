Below is a **clean, concise, submission-ready `design.md`** file (≤2 pages worth of content).
It includes architecture, data model, chunking rationale, fallback behavior, and security notes — exactly what the assignment requires.

---

# **design.md**

# Contract Intelligence API — System Design

## 1. Overview

The Contract Intelligence API ingests PDF contracts, extracts structured information, indexes the text for semantic search, and exposes endpoints for extraction, audit, and Q&A.
The system is optimized for **deterministic testability**, **fallback-safe behavior**, and **explainability** through citations.

---

## 2. High-Level Architecture

```
                 ┌─────────────────────────────┐
                 │          Client UI           │
                 │  Swagger / curl / browser    │
                 └───────────────┬─────────────┘
                                 │ HTTP
                                 ▼
                    ┌─────────────────────────┐
                    │       FastAPI App       │
                    │  (app/main.py routes)   │
                    └─────────────────────────┘
         ┌──────────────────────┬────────────────────────┬──────────────────────┐
         ▼                      ▼                        ▼
┌────────────────┐   ┌─────────────────────┐   ┌──────────────────────────┐
│  Storage Layer │   │   Extractor Module  │   │        QA Layer          │
│ store/pdfs     │   │ PDF → text, fields  │   │ chunking + embeddings    │
│ store/texts    │   │ optional page maps  │   │ vector search + LLM ask  │
└────────────────┘   └─────────────────────┘   └──────────────────────────┘
         ▼                      ▼                        ▼
         └───────────────┬──────┴───────────────┬────────┘
                         ▼                       ▼
                ┌────────────────┐      ┌──────────────────┐
                │  index.json    │      │ vectorstore/FAISS │
                │ metadata store │      │ embeddings/chunks │
                └────────────────┘      └──────────────────┘
```

---

## 3. Data Model

### **Document Metadata (index.json)**

```json
{
  "documents": {
    "<doc_id>": {
      "id": "<doc_id>",
      "meta": {}
    }
  }
}
```

### **Filesystem Layout (STORE_DIR)**

```
store/
 ├─ pdfs/     # raw uploaded PDFs
 ├─ texts/    # extracted plain text files
 └─ index.json
```

### **Chunk Representation**

```
{
  "id": "<chunk_id>",
  "doc_id": "<document_id>",
  "text": "<chunk text>",
  "embedding": [ .. 384 dims .. ]
}
```

Chunks are stored through the vectorstore implementation (FAISS / in-memory dict fallback).

---

## 4. Chunking Strategy

### **Why chunk the document?**

* LLMs answer better when provided small, relevant contexts.
* Embeddings on large texts lose semantic locality.
* Reduces compute cost and improves recall.

### **Chunking rules**

* Default chunk size: **500 characters**
* Overlap: **50 characters**
* Tuned to contract language (dense clauses + cross-references)

### **Page boundaries**

* Extractor optionally returns `page_map`, but for simplicity the current indexer relies on flattened text; page_map may be used for improved citations in later iterations.

---

## 5. QA Pipeline Logic

### Step 1: Retrieve top-K chunks

* Query → embedding → FAISS/k-NN → top matches
* If embeddings unavailable → fallback to keyword scoring

### Step 2: Build LLM prompt

Uses `prompts/qa_prompt.txt`:

* Provide context chunks
* Require short precise answer
* Require explicit citations

### Step 3: Post-process

* Score = cosine similarity average
* Citations = chunk_ids contributing most to LLM answer
* Empty answer returned when:

  * No chunks retrieved
  * Model uncertain by instruction

### **Fallback behavior**

| Condition                           | Behavior                           |
| ----------------------------------- | ---------------------------------- |
| Embedding model fails               | Keyword search fallback            |
| extractor.save_extracted_text fails | Writes file directly into `/texts` |
| QA ask_stream missing               | Falls back to synchronous `ask()`  |
| No text extracted                   | Empty default answer (`score=0`)   |

All fallbacks are **non-fatal** to keep API responsive.

---

## 6. Background Processing Model

For ingestion:

1. `save_upload()` stores PDF
2. Background task runs:

   * extract full text
   * save text
   * index chunks
   * update `index.json`

### Deterministic Mode for Tests

`FORCE_SYNC_EXTRACT=1`
→ entire extraction + indexing is run synchronously in the ingest request.
This guarantees test reproducibility and avoids race conditions.

---

## 7. Security Considerations

### **File Upload Safety**

* Only PDFs accepted (FastAPI validation)
* Filenames are ignored → random UUID doc_ids
* Writes restricted to STORE_DIR

### **LLM Output Filtering**

* Prompts restrict hallucination
* All answers cite chunks (traceability)
* If no relevant content → answer is empty

### **Server Security**

* CORS disabled by default
* No external network calls in the extractor or QA modules
* Logs redact file names and personal data where relevant

### **Data Isolation**

* Users cannot guess paths: document_ids are UUID-hex
* No direct PDF or text exposure except via controlled endpoints

---

## 8. Trade-offs & Simplifications

### Pros

* Very test-friendly due to deterministic extraction mode
* Simple file-based storage (easy to mount in Docker)
* Low-resource: MiniLM embedding model + FAISS
* Clear fallback paths for every subsystem

### Cons

* No relational DB (migrations unused)
* Chunking is not page-aware yet
* Keyword fallback is simplistic
* Extraction accuracy depends on PDF quality (PyPDF)

---

## 9. Future Improvements

* Add contract-specific clause classifiers
* Support multiple languages
* Use page-aware contextual chunking
* Add `/documents/{id}/download` endpoint with controlled access
* Improve citation accuracy using page_map offsets

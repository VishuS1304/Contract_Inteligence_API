Here is a **clean, production-ready README.md** matching your project and assignment requirements.

---

# 📄 Contract Intelligence API

**AI-powered PDF ingestion, text extraction, contract auditing, and question answering.**
Built with **FastAPI**, **FAISS**, **Sentence Transformers**, and a modular vectorstore + extraction pipeline.

---

## 🚀 Features

* Upload multiple PDFs with `/ingest`
* Async & sync text extraction from PDFs
* Field extraction & rule-based auditing
* Embedding-based Q&A across documents
* Streaming `/ask/stream` endpoint
* Fully tested (pytest)
* Dockerized & production-ready
* Metrics via `/metrics` (Prometheus)

---

## What this project does
- Ingest PDFs, store metadata and extract text.
- Extract structured fields (parties, effective_date, term, governing_law, payment_terms, termination, auto_renewal, confidentiality, indemnity, liability_cap, signatories).
- RAG QA: answer questions grounded only in uploaded docs, returning answer + citations (doc id + page ranges).
- Audit: detect risky clauses (auto-renewal with <30d notice, unlimited liability, broad indemnity).
- Streaming / SSE for long answers.
- Metrics (Prometheus) and OpenAPI docs.

## Public contracts used (add links here)
- NDA sample 1 — [https://www.startupindia.gov.in/content/dam/invest-india/Templates/public/Tools_templates/internal_templates/Lets_Venture/NON_DISCLOSURE_AGREEMENT.pdf]
- MSA sample 2 — [https://www.globalsign.com/en/repository/GlobalSign_Master_Services_Agreement.pdf]
- Terms-of-Service sample — [https://www.termsfeed.com/public/uploads/2021/12/sample-terms-of-service-template.pdf]

## Quickstart
1. Place PDF files in `data/`.
2. `docker-compose up --build`
3. POST `/ingest` with files (or use curl to upload).
4. Use `/extract`, `/ask`, `/audit`, `/ask/stream`.


# 🛠️ Setup

## 1️⃣ Clone Repo

```bash
git clone https://github.com/<your-username>/Contract-Intelligence-API.git
cd Contract-Intelligence-API
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3️⃣ Set Environment Variables

| Variable             | Description                   | Default  |
| -------------------- | ----------------------------- | -------- |
| `STORE_DIR`          | Folder for pdfs, texts, index | `store/` |
| `FORCE_SYNC_EXTRACT` | Force extraction during tests | `0`      |
| `LOGLEVEL`           | Logging level                 | `INFO`   |

Example:

```bash
set STORE_DIR=store
set LOGLEVEL=DEBUG
set FORCE_SYNC_EXTRACT=1
```

---

# 🐳 Docker Setup

## Build

```bash
docker build -t contract-intel .
```

## Run

```bash
docker run -p 8000:8000 \
  -e STORE_DIR=/app/store \
  -v $(pwd)/store:/app/store \
  contract-intel
```

## Docker Compose

```bash
docker compose up --build
```

---

# 🧪 Running Tests

```bash
pytest -q
```

---

# 📘 API Endpoints

## 🔹 1. Health

```bash
GET /healthz
```

## 🔹 2. Ingest PDFs

```bash
POST /ingest
Form: files[] = <pdfs>
```

Example:

```bash
curl -X POST "http://localhost:8000/ing
```

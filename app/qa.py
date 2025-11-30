# app/qa.py

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import os
import re
import math
import heapq
import time

logger = logging.getLogger("contract-intel.qa")
logging.basicConfig(level=logging.INFO)

# Try to import SentenceTransformer and faiss if available
EMBEDDING_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDING_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    np = None
    EMBEDDING_AVAILABLE = False
    logger.info("sentence-transformers not available; falling back to token-overlap scoring.")

# Import helper to extract text if needed
try:
    from .extractor import extract_structured_fields, extract_text_from_pdf, save_extracted_text
except Exception:
    # fallback: if module path differs, try relative import
    from extractor import extract_structured_fields, extract_text_from_pdf, save_extracted_text

STORE_TEXT_DIR = Path(os.getenv("STORE_TEXT_DIR", "./store/texts"))
STORE_DIR = Path(os.getenv("STORE_DIR", "./store"))

# in-memory index structures
# docs: dict document_id -> {"text": str, "page_map": [(page, start, end)], "embedding": np.array or None}
_docs: Dict[str, Dict[str, Any]] = {}
_embedding_model = None


def _load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _embedding_model
    if not EMBEDDING_AVAILABLE:
        return None
    if _embedding_model is None:
        try:
            logger.info("Loading embedding model %s ...", model_name)
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            _embedding_model = None
    return _embedding_model


def _embed_text(texts: List[str]) -> Optional[Any]:
    """
    Return embeddings for list of texts (np.array) or None if model not available.
    """
    model = _load_embedding_model()
    if model is None:
        return None
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        return None


def _read_text_for_doc(document_id: str) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Read plain text for document_id from store/texts/{document_id}.txt;
    if not present, attempt to extract from ./store/{document_id}.pdf and save text.
    Returns (text, page_map) where page_map may be [] if text file used (no mapping).
    """
    txt_path = STORE_TEXT_DIR / f"{document_id}.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        # We don't have page_map saved with simple .txt — page_map empty
        return text, []
    # try PDF fallback
    pdf_path = STORE_DIR / f"{document_id}.pdf"
    if pdf_path.exists():
        text, page_map = extract_text_from_pdf(str(pdf_path))
        # save text for next time
        try:
            SAVE_DIR = STORE_TEXT_DIR
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            dest = SAVE_DIR / f"{document_id}.txt"
            dest.write_text(text, encoding="utf-8", errors="ignore")
        except Exception:
            logger.exception("Failed to persist extracted text for %s", document_id)
        return text, page_map
    raise FileNotFoundError(f"Neither text nor pdf found for document {document_id}")


def index_document(document_id: str, use_embedding: bool = True) -> bool:
    """
    Index a single document (read its text, compute embedding if available).
    Returns True on success.
    """
    try:
        text, page_map = _read_text_for_doc(document_id)
    except FileNotFoundError:
        logger.warning("Document %s not found for indexing", document_id)
        return False
    except Exception as e:
        logger.exception("Failed to read document %s: %s", document_id, e)
        return False

    doc_entry = {"text": text, "page_map": page_map, "embedding": None}
    _docs[document_id] = doc_entry

    # compute embedding for document in chunks: split into paragraphs/sentences for better retrieval granularity
    if EMBEDDING_AVAILABLE and use_embedding:
        try:
            # chunk text into passages ~200 tokens (approx 200-400 chars)
            passages = []
            idx_map = []
            CHUNK_SIZE = 800  # characters
            t = text or ""
            for i in range(0, len(t), CHUNK_SIZE):
                chunk = t[i: i + CHUNK_SIZE]
                if chunk.strip():
                    passages.append(chunk.strip())
                    idx_map.append((i, i + len(chunk)))
            embeddings = _embed_text(passages)
            # store passages and embeddings for this doc
            doc_entry["passages"] = passages
            doc_entry["passage_spans"] = idx_map
            doc_entry["passage_embeddings"] = embeddings  # numpy array (n_passages, dim)
        except Exception as e:
            logger.exception("Failed to compute embeddings for %s: %s", document_id, e)
    else:
        # for fallback we do not store embeddings; rely on token-overlap scoring
        doc_entry["passages"] = [text] if text else []
        doc_entry["passage_spans"] = [(0, len(text))] if text else []
        doc_entry["passage_embeddings"] = None

    return True


def index_all(use_embedding: bool = True) -> int:
    """
    Index all documents found under ./store/texts/*.txt or ./store/*.pdf
    Returns number of documents indexed.
    """
    indexed = 0
    # prefer text files to enumerate document IDs
    if STORE_TEXT_DIR.exists():
        for p in STORE_TEXT_DIR.glob("*.txt"):
            doc_id = p.stem
            ok = index_document(doc_id, use_embedding=use_embedding)
            if ok:
                indexed += 1
    # also scan store/*.pdf for any pdfs without text counterpart
    for pdf in STORE_DIR.glob("*.pdf"):
        doc_id = pdf.stem
        if doc_id not in _docs:
            ok = index_document(doc_id, use_embedding=use_embedding)
            if ok:
                indexed += 1
    logger.info("Indexed %d documents (use_embedding=%s)", indexed, use_embedding)
    return indexed


# ---------- scoring helpers ----------

def _cosine_similarity(a, b):
    # numeric safe cosine
    if a is None or b is None:
        return 0.0
    try:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))
    except Exception:
        return 0.0


def _score_text_overlap(question: str, passage: str) -> float:
    """
    Fallback scoring: token overlap ratio (simple and fast).
    """
    q_tokens = set(re.findall(r"\w+", question.lower()))
    p_tokens = set(re.findall(r"\w+", passage.lower()))
    if not q_tokens or not p_tokens:
        return 0.0
    inter = q_tokens.intersection(p_tokens)
    score = len(inter) / max(1.0, math.log(1 + len(p_tokens)))
    return float(score)


# ---------- search & answer ----------

def search(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search across indexed documents and return list of top_k results:
    Each result: {"document_id", "score", "span_start", "span_end", "excerpt"}
    If embeddings available, does semantic similarity across passages; otherwise uses token-overlap scoring.
    """
    results = []
    if not _docs:
        logger.info("Index empty; running index_all()")
        index_all(use_embedding=EMBEDDING_AVAILABLE)

    # compute question embedding if possible
    q_emb = None
    if EMBEDDING_AVAILABLE:
        try:
            q_emb = _embed_text([question])[0]
        except Exception:
            q_emb = None

    # For each doc, check passages
    for doc_id, doc in _docs.items():
        passages = doc.get("passages", [])
        spans = doc.get("passage_spans", [])
        emb_matrix = doc.get("passage_embeddings", None)
        for i, passage in enumerate(passages):
            score = 0.0
            if q_emb is not None and emb_matrix is not None:
                try:
                    p_emb = emb_matrix[i]
                    score = _cosine_similarity(q_emb, p_emb)
                except Exception:
                    score = _score_text_overlap(question, passage)
            else:
                score = _score_text_overlap(question, passage)

            if score and score > 0.0:
                start, end = spans[i] if i < len(spans) else (0, len(passage))
                results.append({
                    "document_id": doc_id,
                    "score": float(score),
                    "span_start": int(start),
                    "span_end": int(end),
                    "excerpt": passage.strip()[:1000],
                })

    # rank by score and return top_k unique document-level results (keep best passage per doc)
    best_per_doc: Dict[str, Dict[str, Any]] = {}
    for r in results:
        did = r["document_id"]
        if did not in best_per_doc or r["score"] > best_per_doc[did]["score"]:
            best_per_doc[did] = r

    ranked = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def synthesize_answer(question: str, hits: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Build a simple answer string by concatenating short excerpts around evidence.
    Returns (answer_text, avg_score, citations)
    citations: list of {document_id, start, end, excerpt}
    """
    if not hits:
        return "", 0.0, []

    parts = []
    scores = []
    citations = []

    for h in hits:
        did = h["document_id"]
        score = float(h.get("score", 0.0))
        scores.append(score)
        excerpt = h.get("excerpt", "")
        # add a short prefix describing doc id and score
        parts.append(f"[doc:{did} score:{score:.3f}] {excerpt}")
        citations.append({
            "document_id": did,
            "start": h.get("span_start", 0),
            "end": h.get("span_end", 0),
            "excerpt": excerpt
        })

    answer_text = "\n\n".join(parts)
    avg_score = float(sum(scores) / max(1, len(scores)))
    return answer_text, avg_score, citations


def answer(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Perform search + synthesize answer. Returns:
    {"answer": str, "score": float, "citations": [...]}
    """
    try:
        hits = search(question, top_k=top_k)
        answer_text, avg_score, citations = synthesize_answer(question, hits)
        return {"answer": answer_text, "score": float(avg_score), "citations": citations}
    except Exception as e:
        logger.exception("QA failed for question=%s: %s", question, e)
        return {"answer": "", "score": 0.0, "citations": []}


def stream_answer(question: str, top_k: int = 3, chunk_size: int = 40):
    """
    Generator streaming tokens/chunks of the answer. Yields small JSON-ish strings or plain text chunks.
    Example usage in a FastAPI SSE endpoint:
        return EventSourceResponse(stream_answer("what is nda", top_k=2))
    """
    res = answer(question, top_k=top_k)
    text = res.get("answer", "") or ""
    # simple chunking by characters to simulate streaming tokens
    idx = 0
    n = len(text)
    if n == 0:
        yield "data: {}\n\n"
        return
    while idx < n:
        chunk = text[idx: idx + chunk_size]
        idx += chunk_size
        # yield a json-like data (clients will parse as text)
        yield f"data: {chunk}\n\n"
        time.sleep(0.02)  # small throttle for liveliness


# ---------------- CLI helpers for manual testing ----------------
if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="Index all documents first")
    ap.add_argument("--q", help="question to ask")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    if args.index:
        n = index_all(use_embedding=EMBEDDING_AVAILABLE)
        print(f"Indexed {n} docs.")

    if args.q:
        out = answer(args.q, top_k=args.topk)
        print(json.dumps(out, indent=2))



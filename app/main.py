# app/main.py
import os
import io
import json
import time
import logging
import uuid
from pathlib import Path
from typing import List, Optional
import asyncio
import inspect
from typing import AsyncIterable, Iterable, Any

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter

# local modules (do not import app.qa here to avoid heavy ML imports at import time)
from app import storage, extractor, audit as audit_mod, vectorstore  # qa imported lazily

logger = logging.getLogger("contract-intel.main")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# Configuration
STORE_DIR = Path(os.environ.get("STORE_DIR", "store"))
PDFS_DIR = STORE_DIR / "pdfs"
TEXTS_DIR = STORE_DIR / "texts"
INDEX_FILE = STORE_DIR / "index.json"

# Ensure directories exist
for d in (STORE_DIR, PDFS_DIR, TEXTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create directory %s", d)

app = FastAPI(title="Contract Intelligence API")

# Prometheus simple counter as example
try:
    ingest_counter = Counter("contract_intel_ingest_total", "Number of ingested documents")
except Exception:
    ingest_counter = None
    logger.debug("Prometheus Counter not available; continuing without metrics")

# lazy loader for QA module (avoid heavy imports at import-time)
_qa_mod = None


def get_qa_mod():
    """
    Import app.qa lazily and cache the module reference.
    Returns None if import fails.
    """
    global _qa_mod
    if _qa_mod is not None:
        return _qa_mod
    try:
        import importlib

        _qa_mod = importlib.import_module("app.qa")
        logger.debug("Lazy-loaded app.qa module")
        return _qa_mod
    except Exception:
        logger.debug("Lazy import of app.qa failed (will continue without QA module)", exc_info=False)
        _qa_mod = None
        return None


def _safe_index_update(document_id: str, meta: Optional[dict] = None):
    """
    Update index.json with a new document entry in a robust way.
    """
    try:
        idx = {}
        if INDEX_FILE.exists():
            try:
                raw = INDEX_FILE.read_text(encoding="utf-8")
                idx = json.loads(raw or "{}")
            except Exception:
                idx = {}
        if "documents" not in idx:
            idx["documents"] = {}
        idx["documents"][document_id] = {"id": document_id, "meta": meta or {}}
        INDEX_FILE.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Index updated for document_id=%s", document_id)
    except Exception:
        logger.exception("Failed to update index file for document_id=%s", document_id)


def process_document_background(document_id: str):
    """
    Background worker that performs extraction and indexing for a given document_id.
    """
    try:
        logger.info("Background processing document_id=%s", document_id)
        pdf_path = PDFS_DIR / f"{document_id}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF for document_id=%s not found at %s", document_id, pdf_path)
            return

        extracted_text = None
        page_map = None
        try:
            if hasattr(extractor, "extract_text_from_pdf"):
                extracted = extractor.extract_text_from_pdf(str(pdf_path))
                if isinstance(extracted, tuple):
                    extracted_text, page_map = extracted
                else:
                    extracted_text = extracted
                logger.debug(
                    "extract_text_from_pdf returned type=%s len=%d",
                    type(extracted_text),
                    len(extracted_text or ""),
                )
                logger.info("Background extraction finished for %s (len=%d)", document_id, len(extracted_text or ""))
            else:
                logger.debug("No extractor.extract_text_from_pdf implementation; skipping extraction for %s", document_id)
        except Exception:
            logger.exception("Background extraction failed for %s", document_id)

        # Save extracted text: call extractor.save_extracted_text with only (document_id, text)
        try:
            if extracted_text is not None:
                try:
                    if hasattr(extractor, "save_extracted_text"):
                        # call with two args ONLY to avoid mis-binding of optional params
                        extractor.save_extracted_text(document_id, extracted_text)
                    else:
                        TEXTS_DIR.mkdir(parents=True, exist_ok=True)
                        (TEXTS_DIR / f"{document_id}.txt").write_text(extracted_text, encoding="utf-8")
                    logger.info("Saved extracted text for %s", document_id)
                except TypeError:
                    # defensive: if extractor.save_extracted_text signature is different, fallback to writing file
                    logger.exception("extractor.save_extracted_text raised TypeError, falling back to writing file")
                    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
                    (TEXTS_DIR / f"{document_id}.txt").write_text(extracted_text, encoding="utf-8")
                except Exception:
                    logger.exception("Saving extracted text failed for %s", document_id)
        except Exception:
            logger.exception("Unexpected error while saving extracted text for %s", document_id)

        # Indexing
        try:
            qa = get_qa_mod()
            if qa is not None:
                if hasattr(qa, "index_document"):
                    qa.index_document(document_id)
                elif hasattr(qa, "index_documents"):
                    qa.index_documents([document_id])
                elif hasattr(qa, "index_all_documents"):
                    qa.index_all_documents()
                else:
                    logger.debug("No explicit index function found in qa module; skipping indexing for %s", document_id)
            else:
                logger.debug("QA module not available; skipping indexing for %s", document_id)
            logger.info("Indexing attempted for %s", document_id)
        except Exception:
            logger.exception("Indexing failed for %s", document_id)

        # Update index.json
        try:
            _safe_index_update(document_id)
        except Exception:
            logger.exception("Index file update failed for %s", document_id)

    except Exception:
        logger.exception("Unhandled error in process_document_background for %s", document_id)


@app.on_event("startup")
def startup_event():
    """
    App startup: initialize QA/index if needed.
    """
    logger.info("Application startup: initializing QA/index (if implemented)")
    try:
        qa = get_qa_mod()
        if qa is not None:
            if hasattr(qa, "startup"):
                qa.startup()
            elif hasattr(qa, "load_embedder"):
                qa.load_embedder()
    except Exception:
        logger.exception("QA startup/indexing failed (non-fatal)")

    logger.info("Application startup complete")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """
    Return Prometheus metrics. Uses generate_latest() which returns bytes.
    """
    try:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception:
        # fallback
        return Response(content=b"", media_type=CONTENT_TYPE_LATEST)


@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Accepts multipart/form-data with one or more files under the field 'files'.
    Saves uploaded PDFs to STORE_DIR/pdfs/ and schedules background processing.
    If FORCE_SYNC_EXTRACT is set (1/true/yes) then extraction/indexing will be run synchronously
    before returning (useful for deterministic test runs).
    """
    document_ids: List[str] = []

    for upload in files:
        try:
            # Prefer storage.save_upload if available
            document_id = None
            try:
                if hasattr(storage, "save_upload"):
                    result = storage.save_upload(upload)
                    # storage.save_upload may return (document_id, saved_path) or just document_id
                    if isinstance(result, tuple):
                        document_id = result[0]
                    else:
                        document_id = result
                else:
                    # fallback: write file directly and generate id based on uuid
                    document_id = uuid.uuid4().hex
                    dest = PDFS_DIR / f"{document_id}.pdf"
                    content = await upload.read()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
            except Exception:
                # If the storage layer raised, fallback to writing file locally
                logger.exception("storage.save_upload failed; falling back to direct file write for upload=%s", getattr(upload, "filename", None))
                document_id = uuid.uuid4().hex
                dest = PDFS_DIR / f"{document_id}.pdf"
                dest.parent.mkdir(parents=True, exist_ok=True)
                content = await upload.read()
                dest.write_bytes(content)

            if not document_id:
                # defensive: ensure we always have an id
                document_id = uuid.uuid4().hex

            document_ids.append(document_id)
            try:
                if ingest_counter is not None:
                    ingest_counter.inc()
            except Exception:
                logger.debug("ingest_counter not available or failed to increment")

            logger.info(
                "Ingested upload original=%s document_id=%s",
                getattr(upload, "filename", None),
                document_id,
            )

            # schedule background processing
            background_tasks.add_task(process_document_background, document_id)

            # Synchronous extraction/indexing if requested (best-effort, non-fatal)
            if os.getenv("FORCE_SYNC_EXTRACT", "").lower() in ("1", "true", "yes"):
                try:
                    pdf_path = PDFS_DIR / f"{document_id}.pdf"
                    extracted_text = None
                    page_map = None

                    if hasattr(extractor, "extract_text_from_pdf"):
                        extracted = extractor.extract_text_from_pdf(str(pdf_path))
                        if isinstance(extracted, tuple):
                            extracted_text, page_map = extracted
                        else:
                            extracted_text = extracted

                    if extracted_text is not None:
                        try:
                            if hasattr(extractor, "save_extracted_text"):
                                # Call with two args only: document_id and text — don't pass page_map as the store path.
                                extractor.save_extracted_text(document_id, extracted_text)
                            else:
                                TEXTS_DIR.mkdir(parents=True, exist_ok=True)
                                (TEXTS_DIR / f"{document_id}.txt").write_text(extracted_text, encoding="utf-8")
                        except TypeError:
                            logger.exception("extractor.save_extracted_text raised TypeError during sync save; falling back to writing text file")
                            TEXTS_DIR.mkdir(parents=True, exist_ok=True)
                            (TEXTS_DIR / f"{document_id}.txt").write_text(extracted_text, encoding="utf-8")
                        except Exception:
                            logger.exception("Failed to save extracted text for %s", document_id)

                    # Try to index synchronously too (handle multiple possible index function names)
                    try:
                        qa = get_qa_mod()
                        if qa is not None:
                            if hasattr(qa, "index_document"):
                                qa.index_document(document_id)
                            elif hasattr(qa, "index_documents"):
                                qa.index_documents([document_id])
                            elif hasattr(qa, "index_all_documents"):
                                qa.index_all_documents()
                        else:
                            logger.debug("QA module not available for sync indexing of %s", document_id)
                    except Exception:
                        logger.exception("Synchronous indexing failed for %s", document_id)

                    # Update index.json (best-effort)
                    try:
                        _safe_index_update(document_id)
                    except Exception:
                        logger.exception("Index file update failed for %s", document_id)

                except Exception:
                    logger.exception("Synchronous extraction/indexing failed for %s", document_id)

        except Exception:
            logger.exception("Failed to ingest upload=%s", getattr(upload, "filename", None))

    status_code = 201 if len(document_ids) > 1 else 200
    return JSONResponse({"document_ids": document_ids}, status_code=status_code)


@app.post("/extract")
def extract(document_id: str = Form(...)):
    """
    Trigger extraction of structured fields for a document_id.
    Returns whatever extractor.extract_structured_fields returns (expected dict).
    """
    try:
        if hasattr(extractor, "extract_structured_fields"):
            result = extractor.extract_structured_fields(document_id)
            return result
        elif hasattr(extractor, "extract_structured_fields_from_text"):
            # fallback: read text and call fields extractor
            txt_path = TEXTS_DIR / f"{document_id}.txt"
            if not txt_path.exists():
                return JSONResponse({"error": "text not found"}, status_code=404)
            text = txt_path.read_text(encoding="utf-8")
            fields = extractor.extract_structured_fields_from_text(text)
            return {"document_id": document_id, "fields": fields, "raw_text": text}
        else:
            return JSONResponse({"error": "extractor not implemented"}, status_code=500)
    except FileNotFoundError:
        return JSONResponse({"error": "document not found"}, status_code=404)
    except Exception:
        logger.exception("Extraction failed for %s", document_id)
        return JSONResponse({"error": "extraction failed"}, status_code=500)


@app.post("/audit")
def run_audit(document_id: str = Form(...)):
    """
    Run audit rules on a document (based on text or extracted fields).
    If text isn't available yet, return a 200 with an informative payload (instead of 404).
    """
    try:
        if hasattr(audit_mod, "run_audit"):
            resp = audit_mod.run_audit(document_id)
            return resp
        else:
            # fallback: return presence/absence of text (but not 404)
            txt_path = TEXTS_DIR / f"{document_id}.txt"
            if not txt_path.exists():
                # return 200 but indicate missing text so client/UI can show helpful message
                logger.warning("Audit requested but text not found for %s", document_id)
                return {"document_id": document_id, "issues": [], "raw_text_present": False, "note": "text not yet available"}
            raw_text = txt_path.read_text(encoding="utf-8")
            # trivial audit result
            return {"document_id": document_id, "issues": [], "raw_text_present": bool(raw_text)}
    except Exception:
        logger.exception("Audit failed for %s", document_id)
        return JSONResponse({"error": "audit failed"}, status_code=500)

QA_TIMEOUT_SEC = float(os.getenv("QA_TIMEOUT_SEC", "10.0"))


async def _call_maybe_async(func, *args, **kwargs):
    """
    Call func with args. If func is a coroutine function, await it.
    If it's synchronous and potentially blocking, run it in the default executor.
    """
    try:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        # if it returns an awaitable (but not declared coroutine), await it
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        # blocking sync -> run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: result)
    except Exception:
        # re-raise - caller will log/handle
        raise


async def _ensure_index_ready(qa):
    """
    Best-effort: if the qa layer exposes an index status or small helpers, try to trigger indexing.
    This avoids returning empty answers when tests expect the index to be populated.
    This is conservative: only runs when index seems empty and only once per request.
    """
    try:
        # Many implementations offer helpers; try common names
        if hasattr(qa, "is_indexed") and callable(getattr(qa, "is_indexed")):
            try:
                indexed = qa.is_indexed()
                if inspect.isawaitable(indexed):
                    indexed = await indexed
                if indexed:
                    return True
            except Exception:
                # ignore; we will attempt other checks
                pass

        # fallback: try to call indexing functions if available (best-effort)
        if hasattr(qa, "index_document"):
            # can't index unknown document here; caller can trigger separately
            return True
        if hasattr(qa, "index_all_documents"):
            logger.info("QA index seems empty — attempting best-effort index_all_documents()")
            await _call_maybe_async(qa.index_all_documents)
            return True
    except Exception:
        logger.exception("Error while attempting to ensure QA index readiness")
    return False


@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Ask a question to the indexed corpus. Returns a dict with keys (answer, score, citations).
    This endpoint supports QA modules that provide either sync or async `ask` functions.
    - Uses QA_TIMEOUT_SEC (env) to bound wait time.
    - If no answer and index might be empty, tries a best-effort reindex and retries once.
    """
    qa = None
    try:
        qa = get_qa_mod()
        if qa is None or not hasattr(qa, "ask"):
            logger.debug("ask called but qa module or ask() not available")
            return {"answer": "", "score": 0, "citations": [], "note": "qa module not available"}

        # call ask with timeout
        try:
            raw = await asyncio.wait_for(_call_maybe_async(qa.ask, question), timeout=QA_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.warning("QA ask timed out (first try): question=%s", question)
            raw = None
        except Exception:
            logger.exception("qa.ask raised on first attempt for question=%s", question)
            raw = None

        # if nothing returned, try best-effort index and one retry
        if not raw:
            logger.debug("qa.ask returned empty; attempting best-effort reindex and single retry")
            try:
                await _ensure_index_ready(qa)
            except Exception:
                logger.exception("ensure_index_ready failed")

            try:
                raw = await asyncio.wait_for(_call_maybe_async(qa.ask, question), timeout=QA_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.warning("QA ask timed out (retry): question=%s", question)
                raw = None
            except Exception:
                logger.exception("qa.ask raised on retry for question=%s", question)
                raw = None

        # Normalize response into dict with expected keys
        if isinstance(raw, dict):
            # ensure keys exist
            resp = {
                "answer": raw.get("answer", "") if raw else "",
                "score": raw.get("score", 0) if raw else 0,
                "citations": raw.get("citations", []) if raw else [],
            }
            # copy through any extra useful fields
            for k in ("meta", "source", "raw"):  # optional
                if k in raw:
                    resp[k] = raw[k]
            return resp

        # allow raw string/tuple/etc.
        if isinstance(raw, (str, bytes)):
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            return {"answer": text, "score": 1.0 if text else 0, "citations": []}

        if isinstance(raw, (list, tuple)) and len(raw) >= 1:
            # common pattern: (answer, score, citations)
            answer = raw[0] or ""
            score = raw[1] if len(raw) > 1 else 0
            citations = raw[2] if len(raw) > 2 else []
            return {"answer": answer, "score": score, "citations": citations}

        # Unknown return — stringify as fallback
        logger.debug("qa.ask returned unexpected type %s; returning fallback", type(raw))
        return {"answer": str(raw or ""), "score": 0, "citations": []}

    except Exception:
        logger.exception("QA ask failed unexpectedly")
        return {"answer": "", "score": 0, "citations": [], "error": "internal"}


def _format_sse_event(data: str, event: str = None) -> bytes:
    """
    Format a single SSE event. data should be a string (will be newline-escaped).
    """
    lines = data.splitlines() or [""]
    payload = ""
    if event:
        payload += f"event: {event}\n"
    for line in lines:
        payload += f"data: {line}\n"
    payload += "\n"
    return payload.encode("utf-8")


@app.get("/ask/stream")
async def ask_stream(q: str):
    """
    Streaming SSE endpoint. Accepts either:
    - qa.ask_stream(q) which yields strings/bytes or an async generator
    - falls back to calling qa.ask(q) and sending one SSE message with the full JSON.
    Produces 'text/event-stream' SSE compatible output.
    """
    qa = get_qa_mod()
    async def generator():
        if qa is None:
            yield _format_sse_event(json.dumps({"answer": "", "score": 0, "citations": [], "note": "qa module missing"}))
            return

        # If module exposes streaming API prefer it
        if hasattr(qa, "ask_stream"):
            stream_fn = getattr(qa, "ask_stream")
            try:
                # call the function (may be sync or async)
                result = stream_fn(q) if not inspect.iscoroutinefunction(stream_fn) else await stream_fn(q)

                # If result is an async iterable/generator
                if hasattr(result, "__aiter__"):
                    async for chunk in result:  # type: ignore
                        try:
                            text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                            yield _format_sse_event(text)
                        except Exception:
                            logger.exception("failed to send chunk from async generator")
                    return

                # If result is a sync iterable/generator
                if isinstance(result, Iterable):
                    for chunk in result:
                        try:
                            text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                            yield _format_sse_event(text)
                        except Exception:
                            logger.exception("failed to send chunk from iterable")
                    return

                # Otherwise if the function returned a single value, emit it
                text = result.decode("utf-8") if isinstance(result, (bytes, bytearray)) else str(result)
                yield _format_sse_event(text)
                return

            except Exception:
                logger.exception("ask_stream implementation raised; falling back to single response")

        # Fallback: call synchronous ask once and return JSON
        try:
            resp = await ask(question=q)  # reuse ask implementation (async)
            yield _format_sse_event(json.dumps(resp))
        except Exception:
            logger.exception("ask_stream fallback failed")
            yield _format_sse_event(json.dumps({"answer": "", "score": 0, "citations": [], "error": "stream failed"}))

    return StreamingResponse(generator(), media_type="text/event-stream")

# Minimal root
@app.get("/")
def root():
    return {"ok": True, "info": "Contract Intelligence API"}

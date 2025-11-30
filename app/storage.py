# app/storage.py

from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from uuid import uuid4
from datetime import datetime
import json
import shutil
import tempfile
import threading
import logging
import os

logger = logging.getLogger("contract-intel.storage")
logging.basicConfig(level=logging.INFO)

# Default directories (can be overridden by env vars if desired)
BASE_STORE_DIR = Path(os.getenv("STORE_DIR", "./store")).resolve()
PDF_DIR = BASE_STORE_DIR / "pdfs"
TEXT_DIR = BASE_STORE_DIR / "texts"
INDEX_FILE = BASE_STORE_DIR / "index.json"

# In-memory lock for index file writes (simple cross-thread safety)
_index_lock = threading.Lock()


def init_store():
    """
    Create store directories and index file (if missing).
    Safe to call multiple times.
    """
    for d in (BASE_STORE_DIR, PDF_DIR, TEXT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # create index file if missing
    if not INDEX_FILE.exists():
        atomic_write_json(INDEX_FILE, {"documents": {}})
        logger.info("Created new index file at %s", INDEX_FILE)


def atomic_write_json(path: Path, data: Dict[str, Any]):
    """
    Atomically write JSON to `path` by using a temporary file then moving.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="idx-", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        # atomic move
        os.replace(tmp_path, path)
    except Exception:
        # ensure tmp removed on failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def read_index() -> Dict[str, Any]:
    """
    Read and return the JSON index. If missing or invalid, returns an empty structure.
    """
    init_store()
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("index file corrupted")
            if "documents" not in data:
                data["documents"] = {}
            return data
    except FileNotFoundError:
        init_store()
        return {"documents": {}}
    except Exception as e:
        logger.exception("Failed to read index: %s", e)
        # attempt to recreate a safe index
        return {"documents": {}}


def write_index(index: Dict[str, Any]):
    """
    Safely write index JSON to disk with a lock to avoid concurrent writes in same process.
    Note: this does not protect across multiple OS processes; for that you'd need a file lock.
    """
    init_store()
    with _index_lock:
        atomic_write_json(INDEX_FILE, index)


def _make_document_record(filename: str, original_name: str, size: int) -> Dict[str, Any]:
    """
    Build metadata record for new document.
    """
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "id": filename,  # filename is used as document_id
        "original_filename": original_name,
        "stored_filename": filename + ".pdf",
        "size": size,
        "created_at": now,
        "updated_at": now,
        "status": "stored",  # possible: stored, processed, error
        "notes": None,
    }


async def save_uploaded_files(files: List[UploadFile]) -> List[str]:
    """
    Save a list of FastAPI UploadFile objects to the PDF store.
    Returns list of generated document_ids (strings).

    This function:
    - generates a uuid4() doc id for each file
    - saves as {docid}.pdf under PDF_DIR
    - updates index.json with metadata
    """
    init_store()
    if not files:
        return []

    idx = read_index()
    docs_index = idx.setdefault("documents", {})

    saved_ids: List[str] = []
    for up in files:
        try:
            # generate document id
            doc_id = uuid4().hex
            target_name = f"{doc_id}.pdf"
            target_path = PDF_DIR / target_name

            # write stream to temporary file then move for atomicity
            # use UploadFile.file which is a SpooledTemporaryFile, so we need to rewind
            try:
                await up.seek(0)
            except Exception:
                pass

            # Save file to disk
            with tempfile.NamedTemporaryFile(delete=False, dir=str(PDF_DIR), suffix=".pdf") as tmpf:
                # read in chunks
                try:
                    while True:
                        chunk = await up.read(65536)
                        if not chunk:
                            break
                        tmpf.write(chunk)
                finally:
                    tmpf.flush()
                    os.fsync(tmpf.fileno())
                tmp_name = tmpf.name

            # move to final location
            os.replace(tmp_name, target_path)

            size = target_path.stat().st_size

            # create metadata record
            rec = _make_document_record(doc_id, up.filename or "uploaded.pdf", size)
            docs_index[doc_id] = rec
            saved_ids.append(doc_id)
            logger.info("Saved upload as %s (original=%s size=%d)", target_path, up.filename, size)
        except Exception as e:
            logger.exception("Failed to save uploaded file %s: %s", getattr(up, "filename", None), e)
            # continue processing other files

    # persist index
    write_index(idx)
    return saved_ids


def get_pdf_path(document_id: str) -> Optional[Path]:
    """
    Return path to stored PDF for the given document_id, or None if not found.
    """
    p = PDF_DIR / f"{document_id}.pdf"
    return p if p.exists() else None


def get_text_path(document_id: str) -> Optional[Path]:
    """
    Return path to stored extracted text file (if exists).
    """
    p = TEXT_DIR / f"{document_id}.txt"
    return p if p.exists() else None


def list_documents() -> List[Dict[str, Any]]:
    """
    Return a list of document metadata records (from index.json), sorted by created_at desc.
    """
    idx = read_index()
    docs = list(idx.get("documents", {}).values())
    try:
        docs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    except Exception:
        pass
    return docs


def get_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Return metadata record for a document_id, or None if not found.
    """
    idx = read_index()
    return idx.get("documents", {}).get(document_id)


def update_metadata(document_id: str, patch: Dict[str, Any]) -> bool:
    """
    Update metadata record fields (shallow update) and persist.
    Returns True if updated, False if not found.
    """
    idx = read_index()
    docs = idx.setdefault("documents", {})
    rec = docs.get(document_id)
    if not rec:
        return False
    rec.update(patch)
    rec["updated_at"] = datetime.utcnow().isoformat() + "Z"
    docs[document_id] = rec
    write_index(idx)
    return True


def delete_document(document_id: str) -> bool:
    """
    Delete both the PDF and text (if exists) and remove entry from index.
    Returns True if existed and deleted, False if not found.
    """
    idx = read_index()
    docs = idx.get("documents", {})
    rec = docs.get(document_id)
    if not rec:
        return False

    # remove files
    removed_any = False
    pdf_path = PDF_DIR / f"{document_id}.pdf"
    try:
        if pdf_path.exists():
            pdf_path.unlink()
            removed_any = True
    except Exception:
        logger.exception("Failed to remove pdf %s", pdf_path)

    text_path = TEXT_DIR / f"{document_id}.txt"
    try:
        if text_path.exists():
            text_path.unlink()
            removed_any = True
    except Exception:
        logger.exception("Failed to remove text %s", text_path)

    # remove from index
    try:
        docs.pop(document_id, None)
        write_index(idx)
    except Exception:
        logger.exception("Failed to update index when deleting %s", document_id)

    return True


# initialize when module imported
init_store()


# ---- quick CLI for manual operations ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", help="document_id to show metadata")
    ap.add_argument("--delete", help="document_id to delete")
    args = ap.parse_args()
    if args.list:
        for d in list_documents():
            print(json.dumps(d, indent=2))
    if args.show:
        print(json.dumps(get_metadata(args.show) or {}, indent=2))
    if args.delete:
        ok = delete_document(args.delete)
        print("deleted" if ok else "not found")

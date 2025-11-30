# tests/test_ingest.py

import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import os

DATA_DIR = Path("data")


@pytest.mark.skipif(not (DATA_DIR / "nda.pdf").exists(), reason="sample NDA PDF missing in data/")
def test_ingest_single_file(client: TestClient):
    nda_path = DATA_DIR / "nda.pdf"
    with open(nda_path, "rb") as fh:
        files = {"files": ("nda.pdf", fh, "application/pdf")}
        r = client.post("/ingest", files=files)

    assert r.status_code in (200, 201)
    body = r.json()
    assert "document_ids" in body
    doc_ids = body["document_ids"]
    assert isinstance(doc_ids, list) and len(doc_ids) == 1

    document_id = doc_ids[0]

    # check that file exists under STORE_DIR/pdfs/
    store_dir = Path(os.environ.get("STORE_DIR", "store"))
    pdfs_dir = store_dir / "pdfs"
    assert (pdfs_dir / f"{document_id}.pdf").exists(), f"Expected PDF file at {pdfs_dir / f'{document_id}.pdf'}"

    # index.json should list the document_id
    index_file = store_dir / "index.json"
    deadline = time.time() + 6.0
    found_in_index = False
    while time.time() < deadline:
        if index_file.exists():
            try:
                idx = json.loads(index_file.read_text(encoding="utf-8"))
                docs = idx.get("documents", {})
                if document_id in docs:
                    found_in_index = True
                    break
            except Exception:
                pass
        time.sleep(0.2)
    assert found_in_index, "Document id not found in index.json within timeout"


@pytest.mark.skipif(not (DATA_DIR / "nda.pdf").exists() or not (DATA_DIR / "msa.pdf").exists(),
                    reason="sample NDA/MSA PDFs missing in data/")
def test_ingest_multiple_files_and_background_processing(client: TestClient):
    nda_path = DATA_DIR / "nda.pdf"
    msa_path = DATA_DIR / "msa.pdf"
    with open(nda_path, "rb") as f1, open(msa_path, "rb") as f2:
        files = [
            ("files", ("nda.pdf", f1, "application/pdf")),
            ("files", ("msa.pdf", f2, "application/pdf")),
        ]
        r = client.post("/ingest", files=files)

    assert r.status_code in (200, 201)
    body = r.json()
    assert "document_ids" in body
    doc_ids = body["document_ids"]
    assert isinstance(doc_ids, list) and len(doc_ids) >= 2

    store_dir = Path(os.environ.get("STORE_DIR", "store"))
    pdfs_dir = store_dir / "pdfs"
    texts_dir = store_dir / "texts"
    index_file = store_dir / "index.json"

    # Verify PDFs saved
    for did in doc_ids:
        assert (pdfs_dir / f"{did}.pdf").exists(), f"Missing uploaded pdf for document_id {did}"

    # Verify index entries
    deadline = time.time() + 6.0
    indexed_ok = False
    while time.time() < deadline:
        if index_file.exists():
            try:
                idx = json.loads(index_file.read_text(encoding="utf-8"))
                docs = idx.get("documents", {})
                if all(did in docs for did in doc_ids):
                    indexed_ok = True
                    break
            except Exception:
                pass
        time.sleep(0.2)
    assert indexed_ok, "Not all uploaded document_ids found in index.json within timeout"

    # Wait for background extraction to produce at least one text file (best-effort)
    deadline = time.time() + 12.0
    text_found = False
    while time.time() < deadline:
        if any((texts_dir / f"{did}.txt").exists() for did in doc_ids):
            text_found = True
            break
        time.sleep(0.5)

    # It's acceptable for extraction to fail in constrained environments; assert at least PDF/index ok.
    # But warn/fail if no text was produced (optional strictness)
    assert text_found, f"No extracted text found in {texts_dir} within timeout; background processing may not have completed"

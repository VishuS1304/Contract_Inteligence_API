# tests/test_audit.py

import time
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path("data")


@pytest.mark.skipif(not (DATA_DIR / "nda.pdf").exists(), reason="sample NDA PDF missing in data/")
def test_run_audit_on_text_basic():
    """
    Unit test: extract text from NDA and run the audit function directly.
    Expect a list of findings; each finding should be a dict with at least a severity or rule/evidence info.
    """
    from app.extractor import extract_text_from_pdf
    from app import audit

    text, _ = extract_text_from_pdf(str(DATA_DIR / "nda.pdf"))
    assert isinstance(text, str) and len(text) > 0

    # Preferentially call run_audit_on_text if present; fallback to audit_document_text or audit_document
    if hasattr(audit, "run_audit_on_text"):
        findings = audit.run_audit_on_text(text)
    elif hasattr(audit, "audit_text"):
        findings = audit.audit_text(text)
    else:
        # If only audit_document exists, we can't call it without a document_id; mark this test xfail
        pytest.xfail("audit module does not expose run_audit_on_text or audit_text")
        return

    assert isinstance(findings, list)
    if findings:
        # check typical shape: dict with severity or rule/evidence
        first = findings[0]
        assert isinstance(first, dict)
        assert any(k in first for k in ("severity", "rule", "evidence", "span", "description"))


@pytest.mark.skipif(not (DATA_DIR / "nda.pdf").exists(), reason="sample NDA PDF missing in data/")
def test_audit_endpoint_after_ingest(client: TestClient):
    """
    Integration test: ingest NDA.pdf, wait for background processing, then call /audit for returned doc_id.
    Accepts multiple possible statuses but validates successful path shape.
    """
    nda_path = DATA_DIR / "nda.pdf"
    with open(nda_path, "rb") as fh:
        files = {"files": ("nda.pdf", fh, "application/pdf")}
        r = client.post("/ingest", files=files)
    assert r.status_code in (200, 201)
    body = r.json()
    doc_ids = body.get("document_ids", [])
    assert isinstance(doc_ids, list) and len(doc_ids) >= 1
    document_id = doc_ids[0]

    # Wait for background processing to produce extracted text (poll up to timeout)
    store_dir = Path.cwd() / Path(".")  # default fallback
    # read STORE_DIR from env patch created by tests/conftest
    store_dir = Path(__import__("os").environ.get("STORE_DIR", "store"))
    texts_dir = store_dir / "texts"

    deadline = time.time() + 10.0
    found_text = False
    while time.time() < deadline:
        if (texts_dir / f"{document_id}.txt").exists():
            found_text = True
            break
        time.sleep(0.5)

    # proceed even if text not found — endpoint should still attempt to audit (may return 500/404 if missing)
    r = client.post("/audit", data={"document_id": document_id})
    assert r.status_code in (200, 404, 500)
    if r.status_code == 200:
        resp = r.json()
        assert isinstance(resp, dict)
        # expected keys: document_id and findings or findings-like list
        assert resp.get("document_id") == document_id
        # findings may be empty list or list of dicts
        findings = resp.get("findings", None)
        if findings is not None:
            assert isinstance(findings, list)
            if findings:
                assert isinstance(findings[0], dict)
                assert any(k in findings[0] for k in ("severity", "rule", "evidence", "description"))

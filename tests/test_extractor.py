# tests/test_extractor.py

import pytest
from pathlib import Path
from app.extractor import extract_text_from_pdf, extract_structured_fields_from_text, extract_structured_fields, save_extracted_text

DATA_DIR = Path("data")


@pytest.mark.skipif(not (DATA_DIR / "nda.pdf").exists(), reason="sample NDA PDF missing in data/")
def test_extract_text_from_pdf_reads_pages_and_returns_text_and_pagemap():
    pdf_path = DATA_DIR / "nda.pdf"
    text, page_map = extract_text_from_pdf(str(pdf_path))
    assert isinstance(text, str)
    assert len(text) > 0, "Extracted text should not be empty for a text PDF"
    assert isinstance(page_map, list)
    # each entry in page_map should be a tuple of (page_no, start, end)
    if page_map:
        assert isinstance(page_map[0], tuple) and len(page_map[0]) == 3


@pytest.mark.skipif(not (DATA_DIR / "msa.pdf").exists(), reason="sample MSA PDF missing in data/")
def test_extract_structured_fields_from_text_finds_some_keys():
    pdf_path = DATA_DIR / "msa.pdf"
    text, _ = extract_text_from_pdf(str(pdf_path))
    fields = extract_structured_fields_from_text(text)
    assert isinstance(fields, dict)
    # Expect some keys to exist in the heuristic output
    expected_keys = {"parties", "effective_date", "governing_law", "auto_renewal", "confidentiality", "indemnity", "liability_cap"}
    assert any(k in fields for k in expected_keys)


@pytest.mark.skipif(not (DATA_DIR / "tos.pdf").exists(), reason="sample ToS PDF missing in data/")
def test_extract_structured_fields_top_level_api_works_with_pdf_path(tmp_path):
    # Use the top-level convenience function that accepts pdf_path (no document_id)
    pdf_path = DATA_DIR / "tos.pdf"
    res = extract_structured_fields(pdf_path=str(pdf_path))
    assert isinstance(res, dict)
    assert "raw_text" in res and isinstance(res["raw_text"], str)
    assert "fields" in res and isinstance(res["fields"], dict)
    # ensure _meta is present in fields
    assert "_meta" in res["fields"]


def test_save_extracted_text_writes_file_and_is_readable(tmp_path):
    sample_id = "testdoc"
    text = "This is a sample extracted text.\n\n--PAGE 1--\nContent."
    # save to default store/texts (module will create dir) but use tmp path by swapping environment if needed.
    # Here we just save to tmp_path to avoid touching real store
    store_text_dir = tmp_path / "texts"
    store_text_dir.mkdir(parents=True, exist_ok=True)
    dest = save_extracted_text(sample_id, text, store_text_dir=store_text_dir)
    assert dest.exists()
    read_back = dest.read_text(encoding="utf-8")
    assert "sample extracted text" in read_back.lower()

# tests/test_vectorestore.py

import os
import json
import tempfile
import time
import pytest
from pathlib import Path

from app.vectorstore import VectorBase

def test_vectorbase_add_search_remove_clear(tmp_path):
    # create a temporary store dir for this VectorBase instance
    vb_store = tmp_path / "vectors"
    vb = VectorBase(store_dir=str(vb_store))

    # ensure starting clean
    vb.clear()
    assert vb.info()["count"] == 0

    # add two small docs
    vb.add("doc_a", "This document mentions governing law and jurisdiction: California.")
    vb.add("doc_b", "Payment terms: 30 days net. Termination clause referenced here.")
    info = vb.info()
    assert info["count"] == 2

    # basic search for 'governing law' should return doc_a on top (or at least include it)
    res = vb.search("governing law", top_k=3)
    assert isinstance(res, list)
    assert any(r["doc_id"] == "doc_a" for r in res), f"Expected doc_a in search results, got {res}"

    # search for 'payment' should prefer doc_b
    res2 = vb.search("payment terms", top_k=3)
    assert isinstance(res2, list)
    assert any(r["doc_id"] == "doc_b" for r in res2)

    # remove doc_a and ensure count and search reflect removal
    removed = vb.remove("doc_a")
    assert removed is True
    assert vb.info()["count"] == 1
    res3 = vb.search("governing law", top_k=3)
    # doc_a should no longer appear
    assert all(r["doc_id"] != "doc_a" for r in res3)

    # clear everything
    vb.clear()
    assert vb.info()["count"] == 0

def test_vectorbase_save_and_load(tmp_path):
    vb_store = tmp_path / "vectors2"
    vb = VectorBase(store_dir=str(vb_store))
    vb.clear()

    vb.add("d1", "alpha beta gamma")
    vb.add("d2", "delta epsilon zeta")
    assert vb.info()["count"] == 2

    # save to disk
    vb.save()
    # capture meta files exist
    assert (vb_store / "meta.json").exists()

    # create a new instance pointing to same store and load
    vb2 = VectorBase(store_dir=str(vb_store))
    vb2.load()
    info2 = vb2.info()
    # count may reflect meta ids length
    assert info2["count"] == 2

    # searches should work on loaded instance
    r = vb2.search("epsilon", top_k=2)
    assert isinstance(r, list)
    assert any(res["doc_id"] == "d2" for res in r)

    # cleanup
    vb2.clear()
    vb2.save()
    assert vb2.info()["count"] == 0

def test_vectorbase_info_and_edge_cases(tmp_path):
    vb_store = tmp_path / "vectors3"
    vb = VectorBase(store_dir=str(vb_store))
    vb.clear()

    # info returns expected keys
    info = vb.info()
    assert "count" in info and "dim" in info and "faiss" in info and "embedding_available" in info

    # searching empty index returns empty list
    empty_search = vb.search("anything", top_k=3)
    assert isinstance(empty_search, list)
    assert empty_search == []

    # removing non-existent id returns False
    assert vb.remove("nonexistent") is False

    # add then remove again
    vb.add("only", "some sample text")
    assert vb.info()["count"] == 1
    assert vb.remove("only") is True
    assert vb.info()["count"] == 0

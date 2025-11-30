# tests/test_utils.py

import os
import re
import pytest
from pathlib import Path
from app.utils import chunk_text, parse_notice_days, parse_currency_number, safe_join, tail_text_file

def test_chunk_text_basic():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100  # 2600 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert isinstance(chunks, list)
    # chunk tuples: (start, end, chunk_text)
    assert all(isinstance(c[0], int) and isinstance(c[1], int) and isinstance(c[2], str) for c in chunks)
    # ensure coverage approximates original length
    covered = chunks[-1][1] - chunks[0][0]
    assert covered >= len(text) - 10  # allow small variance
    # ensure overlap exists between consecutive chunks
    if len(chunks) >= 2:
        assert chunks[0][1] > chunks[1][0]

@pytest.mark.parametrize("s,expected", [
    ("30 days notice prior to termination", 30),
    ("must provide 60 calendar days' notice", 60),
    ("no less than 15 days before renewal", 15),
    ("prior to renewal within 7 days", 7),
    ("no mention here", None),
])
def test_parse_notice_days_various(s, expected):
    assert parse_notice_days(s) == expected

@pytest.mark.parametrize("s,cur,number", [
    ("USD 1,234.50 payable", "USD", 1234.50),
    ("$ 1000 is stated", "$", 1000.0),
    ("Rs. 1,00,000 penalty", "Rs.", 100000.0),
    ("EUR: 99.99 fee", "EUR", 99.99),
    ("no currency here", None, None),
])
def test_parse_currency_number_various(s, cur, number):
    res = parse_currency_number(s)
    if number is None:
        assert res is None
    else:
        assert res is not None
        # currency match (case-insensitive, allow variant)
        assert cur.lower().replace(".", "") in res["currency"].lower().replace(".", "")
        # numeric close
        assert abs(res["number"] - number) < 1e-6

def test_safe_join_prevents_traversal(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "sub").mkdir()
    good = safe_join(str(base), "sub", "file.txt")
    assert good.startswith(str(base))
    # attempt traversal
    with pytest.raises(ValueError):
        safe_join(str(base), "..", "etc", "passwd")

def test_tail_text_file_returns_last_lines(tmp_path):
    p = tmp_path / "big.txt"
    lines = [f"line {i}" for i in range(100)]
    p.write_text("\n".join(lines), encoding="utf-8")
    tail = tail_text_file(p, n_lines=5)
    assert isinstance(tail, str)
    last_lines = tail.splitlines()
    assert len(last_lines) == 5
    assert last_lines[0] == "line 95"
    assert last_lines[-1] == "line 99"

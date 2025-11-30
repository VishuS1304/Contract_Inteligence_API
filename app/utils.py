# app/utils.py

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

logger = logging.getLogger("contract-intel.utils")
logging.basicConfig(level=logging.INFO)

# Regexes
_NOTICE_DAYS_RE = re.compile(
    r"(?P<days>\d{1,3})\s*(?:calendar|business)?\s*(?:days|day)\s*(?:notice|prior to|before)",
    re.I,
)
_CURRENCY_NUMBER_RE = re.compile(
    r"(?P<currency>\$|USD|EUR|INR|Rs\.?)\s*[:\-\s]?\s*(?P<number>[\d,]+(?:\.\d+)?)", re.I
)
_WHITESPACE_RE = re.compile(r"\s+")


# ----------------------
# File / JSON utilities
# ----------------------
def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write JSON data to `path`.
    Uses an OS-level rename of a temp file to avoid partial writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="tmpjson-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def read_json_safe(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read JSON from path safely; if missing or invalid, returns default (or {}).
    """
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to read json from %s: %s", path, e)
        return default if default is not None else {}


# ----------------------
# Text helpers
# ----------------------
def json_line(obj: Any) -> str:
    """
    Return an object as a single JSON line (newline-terminated) suitable for SSE/plain-stream.
    """
    return json.dumps(obj, ensure_ascii=False) + "\n"


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and normalize newlines to single spaces for compact matching."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Tuple[int, int, str]]:
    """
    Split text into chunks of approximately `chunk_size` characters with `overlap` overlap.
    Returns a list of tuples (start_index, end_index, chunk_text).
    """
    if not text:
        return []

    n = len(text)
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        start = end - overlap if (end - overlap) > start else end
    return chunks


def excerpt(text: str, start: int, end: int, max_len: int = 300) -> str:
    """
    Return a trimmed excerpt for the span [start:end] with length capped by max_len.
    """
    start = max(0, start)
    end = min(len(text), end)
    s = text[start:end]
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ----------------------
# Lightweight parsers
# ----------------------
def parse_notice_days(text: str):
    """
    Extract simple notice periods like:
    - "30 days notice"
    - "60 calendar days' notice"
    - "within 7 days"
    - "no less than 15 days"
    Returns integer or None.
    """
    import re

    # Any number followed by "day" or "days"
    m = re.search(r"(\d+)\s*(?:calendar\s*)?days?", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def parse_currency_number(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse currency + number from text. Returns {"currency": str, "number": float, "raw": str} or None.
    """
    if not text:
        return None
    m = _CURRENCY_NUMBER_RE.search(text)
    if not m:
        return None
    currency = m.group("currency").strip()
    num_raw = m.group("number").replace(",", "")
    try:
        number = float(num_raw)
    except Exception:
        number = None
    return {"currency": currency, "number": number, "raw": m.group(0)}


def find_all_spans(pattern: re.Pattern, text: str, max_hits: int = 50) -> List[Dict[str, Any]]:
    """
    Return a list of matches found by `pattern` (compiled regex).
    Each item is {"start": int, "end": int, "match": str}.
    """
    hits: List[Dict[str, Any]] = []
    for m in pattern.finditer(text):
        hits.append({"start": m.start(), "end": m.end(), "match": m.group(0)})
        if len(hits) >= max_hits:
            break
    return hits


# ----------------------
# Helpers & small utilities
# ----------------------
def ensure_dirs(*paths: str) -> None:
    """
    Ensure directories exist for provided paths.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def safe_join(base: str, *parts: str) -> str:
    """
    Safely join path parts relative to `base` and return as string.
    Prevents traversal escapes.
    """
    basep = Path(base).resolve()
    candidate = basep.joinpath(*parts).resolve()
    if not str(candidate).startswith(str(basep)):
        raise ValueError("Attempt to escape base path")
    return str(candidate)


def timed(name: Optional[str] = None):
    """
    Decorator to time a function and log duration.
    Usage:
        @timed("extract_text")
        def extract(...):
            ...
    """
    def deco(func):
        _name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.info("timed(%s): %.3fs", _name, elapsed)
        return wrapper
    return deco


# ----------------------
# Small CLI-ish helpers used in debugging
# ----------------------
def tail_text_file(path: Path, n_lines: int = 20) -> str:
    """
    Return last n_lines of a text file, safe for large files.
    """
    path = Path(path)
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = 4096
            data = b""
            while len(data.splitlines()) <= n_lines and end > 0:
                read_size = min(size, end)
                f.seek(end - read_size)
                chunk = f.read(read_size)
                data = chunk + data
                end -= read_size
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            return "\n".join(lines[-n_lines:])
    except Exception:
        logger.exception("tail_text_file failed for %s", path)
        return ""


# ----------------------
# Exports (for convenience)
# ----------------------
__all__ = [
    "atomic_write_json",
    "read_json_safe",
    "json_line",
    "normalize_whitespace",
    "chunk_text",
    "excerpt",
    "parse_notice_days",
    "parse_currency_number",
    "find_all_spans",
    "ensure_dirs",
    "safe_join",
    "timed",
    "tail_text_file",
]

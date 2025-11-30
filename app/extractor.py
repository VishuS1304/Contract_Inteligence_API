# app/extractor.py
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import re
import logging
from PyPDF2 import PdfReader
from dateutil import parser as dateparser

logger = logging.getLogger("contract-intel.extractor")
logging.basicConfig(level=logging.INFO)

DEFAULT_STORE_TEXT_DIR = Path("./store/texts")
DEFAULT_STORE_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(path: str) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Extract text from a PDF file.

    Returns:
        text (str): full extracted text (pages joined by '\n\n---PAGE {i}---\n\n')
        page_map (list of tuples): [(page_index, start_char_index, end_char_index), ...]
           This lets callers map a character span back to a page (for citations).
    """
    text_chunks: List[str] = []
    page_map: List[Tuple[int, int, int]] = []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        with p.open("rb") as fh:
            reader = PdfReader(fh)
            total_pages = len(reader.pages)
            cur_pos = 0
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    logger.exception("Failed to extract text from page %d of %s", i, path)
                    page_text = ""
                # Surround with page separator - helpful for later mapping
                delim = f"\n\n---PAGE {i+1}/{total_pages}---\n\n"
                page_content = delim + page_text + "\n"
                text_chunks.append(page_content)
                start = cur_pos + len(delim)  # start of actual page_text (after delim)
                end = start + len(page_text)
                page_map.append((i + 1, start, end))
                cur_pos += len(page_content)
    except Exception as e:
        logger.exception("PDF read failed for %s: %s", path, e)
        raise

    full_text = "".join(text_chunks)
    return full_text, page_map


def save_extracted_text(document_id: str, text: str, store_text_dir: Path = DEFAULT_STORE_TEXT_DIR) -> Path:
    """
    Save plain text to store/texts/{document_id}.txt. Returns Path to saved file.
    """
    store_text_dir = Path(store_text_dir)
    store_text_dir.mkdir(parents=True, exist_ok=True)
    dest = store_text_dir / f"{document_id}.txt"
    dest.write_text(text, encoding="utf-8", errors="ignore")
    return dest


# -------------------
# Heuristics for structured field extraction
# -------------------

# Useful regexes
RE_PARTIES = re.compile(
    r"(?:This\s+Agreement\s+(?:is\s+made\s+)?(?:on|as|effective).*?between\s+(.+?)\s+and\s+(.+?)[\.,\n])",
    re.I | re.S,
)
RE_PARTIES_ALT = re.compile(
    r"(?:Between|BETWEEN)\s*:\s*(.+?)\s*(?:and|AND)\s*(.+?)[\n\.,]",
    re.I | re.S,
)

RE_EFFECTIVE_DATE = re.compile(
    r"\beffective\s+date[:\s]*(?P<date>[^;\n,]+)|effective\s+as\s+of\s+(?P<date2>[^;\n,]+)",
    re.I,
)

RE_GOVERNING_LAW = re.compile(r"\b(governing law|governed by the laws of)\b[:\s]*(?P<law>[^.\n]+)", re.I)
RE_TERM = re.compile(r"\b(term|duration)\b[:\s]*(?P<term>[^.\n]+)", re.I)

RE_AUTO_RENEW = re.compile(r"\b(auto[- ]?renew|automatic renewal|renews automatically)\b", re.I)
RE_NOTICE_DAYS = re.compile(r"(?P<days>\d{1,3})\s*(?:calendar|business)?\s*days\s*(?:notice|prior|before)", re.I)

RE_TERMINATION = re.compile(r"\b(terminate|termination|terminated|terminates)\b", re.I)

RE_CONFIDENTIAL = re.compile(r"\b(confidential information|confidentiality|non-?disclosur(?:e)?)\b", re.I)

RE_INDEMNITY = re.compile(r"\b(indemnif(?:y|ies|ication)|indemnity|hold harmless|defend and hold harmless)\b", re.I)

RE_LIABILITY_CAP = re.compile(
    r"\b(?:cap on liability|liability cap|cap(?:s)? on liability|limit(?:ation)? of liability)\b[^.\n]{0,120}",
    re.I,
)

RE_CURRENCY_AMOUNT = re.compile(r"(?P<currency>\$|USD|EUR|INR|Rs\.?)\s*[\-:\s]?\s*(?P<number>[\d,]+(?:\.\d+)?)", re.I)


def _first_match_text(pattern: re.Pattern, text: str) -> Optional[Tuple[int, int, str]]:
    m = pattern.search(text)
    if not m:
        return None
    return (m.start(), m.end(), m.group(0).strip())


def _find_all_excerpts(pattern: re.Pattern, text: str, max_hits=5) -> List[Dict[str, Any]]:
    hits = []
    for m in pattern.finditer(text):
        if len(hits) >= max_hits:
            break
        hits.append({"start": m.start(), "end": m.end(), "excerpt": text[m.start(): m.end()].strip()})
    return hits


def _parse_date(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        d = dateparser.parse(text, fuzzy=True, dayfirst=False)
        if d:
            return d.date().isoformat()
    except Exception:
        return None
    return None


def parse_liability_cap_context(text: str) -> Optional[Dict[str, Any]]:
    """
    Look for liability cap phrases and attempt to parse currency/number from the nearby text.
    """
    cap_match = RE_LIABILITY_CAP.search(text)
    if not cap_match:
        return None
    start, end = cap_match.start(), cap_match.end()
    context_start = max(0, start - 200)
    context_end = min(len(text), end + 200)
    context = text[context_start:context_end]

    amount_match = RE_CURRENCY_AMOUNT.search(context)
    meta = {"raw": cap_match.group(0).strip(), "context": context.strip()}
    if amount_match:
        meta["currency"] = amount_match.group("currency")
        num = amount_match.group("number").replace(",", "")
        try:
            meta["amount"] = float(num)
        except Exception:
            meta["amount"] = amount_match.group("number")
        meta["amount_raw"] = amount_match.group(0)
    return meta


def extract_structured_fields_from_text(text: str) -> Dict[str, Any]:
    """
    Run a set of heuristics over raw text and return structured fields.
    The returned dict contains:
        parties (list), effective_date (ISO) or raw, term (raw), governing_law (raw),
        payment_terms (list of excerpts), termination (list of excerpts), auto_renewal (dict),
        confidentiality (dict), indemnity (dict), liability_cap (dict), signatories (list)
    """
    out: Dict[str, Any] = {
        "parties": [],
        "effective_date": None,
        "term": None,
        "governing_law": None,
        "payment_terms": [],
        "termination": [],
        "auto_renewal": {"detected": False, "notice_days": None, "evidence": []},
        "confidentiality": {"detected": False, "evidence": []},
        "indemnity": {"detected": False, "evidence": []},
        "liability_cap": None,
        "signatories": [],
    }

    # Parties
    m = RE_PARTIES.search(text)
    if not m:
        m = RE_PARTIES_ALT.search(text)
    if m:
        try:
            # group extraction: often group(1), group(2)
            parties_raw = [g.strip() for g in m.groups() if g and g.strip()]
            out["parties"] = parties_raw
        except Exception:
            out["parties"] = [m.group(0).strip()]

    # effective date
    m = RE_EFFECTIVE_DATE.search(text)
    if m:
        date_text = m.group("date") or m.group("date2") or m.group(0)
        parsed = _parse_date(date_text)
        out["effective_date"] = parsed or date_text.strip()

    # governing law
    m = RE_GOVERNING_LAW.search(text)
    if m:
        law = m.group("law").strip()
        out["governing_law"] = law

    # term
    m = RE_TERM.search(text)
    if m:
        out["term"] = m.group("term").strip()

    # payment_terms: search for words like "payment", "fees", "remit", capture nearby sentences
    payment_hits = _find_all_excerpts(re.compile(r"\b(payment|payment terms|fees|payment shall|compensation|remit)\b", re.I), text, max_hits=6)
    out["payment_terms"] = payment_hits

    # termination excerpts
    termination_hits = _find_all_excerpts(RE_TERMINATION, text, max_hits=10)
    out["termination"] = termination_hits

    # auto-renewal detection & notice days
    ar_hits = _find_all_excerpts(RE_AUTO_RENEW, text, max_hits=6)
    if ar_hits:
        out["auto_renewal"]["detected"] = True
        out["auto_renewal"]["evidence"] = ar_hits
        # try to find a notice days number near any auto-renew match
        for h in ar_hits:
            ctx_start = max(0, h["start"] - 200)
            ctx_end = min(len(text), h["end"] + 200)
            ctx = text[ctx_start:ctx_end]
            m_notice = RE_NOTICE_DAYS.search(ctx)
            if m_notice:
                try:
                    out["auto_renewal"]["notice_days"] = int(m_notice.group("days"))
                    out["auto_renewal"]["evidence"].append({"start": ctx_start + m_notice.start(), "end": ctx_start + m_notice.end(), "excerpt": m_notice.group(0)})
                    break
                except Exception:
                    pass

    # confidentiality
    conf_hits = _find_all_excerpts(RE_CONFIDENTIAL, text, max_hits=6)
    if conf_hits:
        out["confidentiality"]["detected"] = True
        out["confidentiality"]["evidence"] = conf_hits

    # indemnity
    indemn_hits = _find_all_excerpts(RE_INDEMNITY, text, max_hits=6)
    if indemn_hits:
        out["indemnity"]["detected"] = True
        out["indemnity"]["evidence"] = indemn_hits

    # liability cap
    liability_meta = parse_liability_cap_context(text)
    if liability_meta:
        out["liability_cap"] = liability_meta

    # signatories heuristic: look for "IN WITNESS" or "Signed by" sections
    sign_block = re.search(r"(?:IN WITNESS WHEREOF|IN WITNESS)\b(.{0,400})", text, re.I | re.S)
    sign_hits = []
    if sign_block:
        blk = sign_block.group(0)
        # attempt to find lines with names and titles
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        for ln in lines:
            # simple heuristic: lines containing comma or "By:" or "Signature" or title words
            if re.search(r"(Signature|By:|Printed Name|Name:|Title:|Title|Director|Manager|CEO|CTO|CFO)", ln, re.I):
                sign_hits.append(ln)
    # fallback: find "Signed" pages at end
    if not sign_hits:
        tail = text[-2000:] if len(text) > 2000 else text
        for m in re.finditer(r"(Signed:|Signature:|Signed by|By:)\s*(.+)", tail, re.I):
            sign_hits.append(m.group(0).strip())
    out["signatories"] = sign_hits

    # Attach raw counts and quick stats
    out["_meta"] = {
        "word_count": len(text.split()),
        "char_count": len(text),
    }

    return out


def extract_structured_fields(document_id: str = None, pdf_path: str = None, store_text_dir: Path = DEFAULT_STORE_TEXT_DIR) -> Dict[str, Any]:
    """
    Top-level convenience function.

    Either provide document_id (will read ./store/{document_id}.pdf or ./store/texts/{document_id}.txt),
    or provide pdf_path to a PDF file.

    Returns:
        {
          "document_id": str,
          "raw_text": str,
          "page_map": [(page_no, start, end), ...],
          "fields": {...}  # as returned by extract_structured_fields_from_text
        }
    """
    if document_id is None and pdf_path is None:
        raise ValueError("Either document_id or pdf_path must be provided")

    # If document_id provided, prefer reading pre-extracted text first
    text = ""
    page_map: List[Tuple[int, int, int]] = []
    if document_id:
        # try text file first
        txt_file = Path(store_text_dir) / f"{document_id}.txt"
        if txt_file.exists():
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            # No page_map available in this case
        else:
            # try PDF in store root (common layout: store/{document_id}.pdf)
            pdf_candidate = Path("./store") / f"{document_id}.pdf"
            if pdf_candidate.exists():
                text, page_map = extract_text_from_pdf(str(pdf_candidate))
                # save extracted text for faster future audits
                try:
                    save_extracted_text(document_id, text, store_text_dir=store_text_dir)
                except Exception:
                    logger.exception("Failed to save extracted text for %s", document_id)
            else:
                raise FileNotFoundError(f"Neither extracted text nor PDF found for document_id={document_id}")

    else:
        # pdf_path provided
        text, page_map = extract_text_from_pdf(pdf_path)
        # optionally save if document_id provided later

    # run field extraction heuristics
    fields = extract_structured_fields_from_text(text)

    return {
        "document_id": document_id,
        "raw_text": text,
        "page_map": page_map,
        "fields": fields,
    }


# quick CLI usage for local validation
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="path to pdf")
    ap.add_argument("--doc", help="document_id (will look in ./store/{doc}.pdf or ./store/texts/{doc}.txt)")
    args = ap.parse_args()

    try:
        res = extract_structured_fields(document_id=args.doc, pdf_path=args.pdf)
        print(json.dumps(res["fields"], indent=2))
    except Exception as e:
        logger.exception("Extraction failed: %s", e)
        raise

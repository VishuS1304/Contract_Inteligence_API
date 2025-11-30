# app/audit.py
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("contract-intel.audit")
logging.basicConfig(level=logging.INFO)

# Default location where extracted text files are stored (one .txt per document_id)
DEFAULT_TEXT_DIR = Path("./store/texts")


def read_document_text(document_id: str, text_dir: Path = DEFAULT_TEXT_DIR) -> str:
    """
    Read the extracted plain text for a document_id.
    Raises FileNotFoundError if text not present.
    """
    p = text_dir / f"{document_id}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Text for document_id={document_id} not found at {p}")
    text = p.read_text(encoding="utf-8", errors="ignore")
    return text


def excerpt_span(text: str, start: int, end: int, max_len: int = 300) -> str:
    """
    Return a safe excerpt for the evidence span with some context.
    """
    s = max(0, start)
    e = min(len(text), end)
    excerpt = text[s:e]
    if len(excerpt) > max_len:
        excerpt = excerpt[: max_len - 3] + "..."
    return excerpt.strip()


def find_all_spans(pattern: re.Pattern, text: str, group: int = 0, flags: int = 0) -> List[Tuple[int, int, str]]:
    """
    Helper: return list of (start, end, matched_text) for all pattern occurrences.
    """
    spans = []
    for m in pattern.finditer(text):
        try:
            matched = m.group(group)
        except Exception:
            matched = m.group(0)
        spans.append((m.start(), m.end(), matched))
    return spans


# --- Patterns / heuristics used by rules ----

AUTO_RENEW_PATTERNS = [
    re.compile(r"\b(auto[- ]?renew|automatic renewal|renews automatically)\b", re.I),
    re.compile(r"\b(renewal will occur|renew(?:al)?(?: will occur)?)\b", re.I),
]

NOTICE_DAYS_PATTERN = re.compile(
    r"(?P<notice>\b\d{1,3}\b)\s*(?:calendar|business)?\s*(?:days|day)\s*(?:notice|prior to|before)",
    re.I,
)

UNLIMITED_LIABILITY_PATTERNS = [
    re.compile(r"\b(unlimited liability|liability is unlimited|no limit on liability)\b", re.I),
    re.compile(r"\b(no(?:\s+limit(?:s)?)?\s+on\s+liability)\b", re.I),
]

INDEMNITY_PATTERNS = [
    re.compile(r"\b(indemnif(?:y|ies|ication)|indemnity)\b", re.I),
    # detect phrases suggesting broad indemnity
    re.compile(r"\b(indemnif(?:y|ies)\s+and\s+hold\s+harmless)\b", re.I),
]

CONFIDENTIALITY_PATTERNS = [
    re.compile(r"\b(confidentiality|confidential information|non-?disclosur(?:e|e))\b", re.I)
]

TERMINATION_PATTERN = re.compile(
    r"\b(termination|terminate|terminated|terminates)\b", re.I
)

LIABILITY_CAP_PATTERN = re.compile(
    r"\b(?:liability\s+cap|cap on liability|cap(?:s)?\s+of)\s*(?:[:\-\s,])?\s*(?P<amount>\$?\s?\d{1,3}(?:[,\d{3}])*(?:\.\d{1,2})?\s*(?:USD|INR|EUR|Rs\.?)?)",
    re.I,
)

CURRENCY_NUMBER_PATTERN = re.compile(
    r"(?P<currency>\$|USD|EUR|INR|Rs\.?)\s*[:\-]?\s*(?P<number>[\d,]+(?:\.\d+)?)", re.I
)


# --- Rule logic functions ---


def detect_auto_renewal(text: str) -> Optional[Dict[str, Any]]:
    """
    Detect auto-renewal clauses and attempt to parse notice days.
    """
    hits = []
    for pat in AUTO_RENEW_PATTERNS:
        spans = find_all_spans(pat, text)
        for start, end, m in spans:
            hits.append((start, end, m))

    if not hits:
        return None

    # attempt to find notice days near any hit (search within 250 chars around the match)
    notice_days: Optional[int] = None
    evidence_spans = []
    for start, end, matched in hits:
        context_start = max(0, start - 250)
        context_end = min(len(text), end + 250)
        context = text[context_start:context_end]

        # record matched auto-renew phrase
        evidence_spans.append({"start": start, "end": end, "excerpt": excerpt_span(text, start, end)})

        m_notice = NOTICE_DAYS_PATTERN.search(context)
        if m_notice:
            try:
                d = int(m_notice.group("notice"))
                notice_days = d
                # compute absolute offset for the match
                notice_start = context_start + m_notice.start()
                notice_end = context_start + m_notice.end()
                evidence_spans.append({"start": notice_start, "end": notice_end, "excerpt": excerpt_span(text, notice_start, notice_end)})
            except Exception:
                pass

    # severity logic:
    # - High if auto-renew exists and notice_days is present and < 30
    # - Medium if auto-renew exists and notice >=30 or notice not found
    severity = "medium"
    meta = {}
    if notice_days is not None:
        meta["notice_days"] = notice_days
        if notice_days < 30:
            severity = "high"
        else:
            severity = "medium"
    else:
        # no explicit notice found -> treat as medium risk (could be high depending on clause wording)
        severity = "medium"

    return {
        "rule": "auto_renewal",
        "severity": severity,
        "evidence_spans": evidence_spans,
        "metadata": meta,
        "explanation": "Auto-renewal clause detected; check required notice period and renewal cancellation process.",
    }


def detect_unlimited_liability(text: str) -> Optional[Dict[str, Any]]:
    hits = []
    for pat in UNLIMITED_LIABILITY_PATTERNS:
        spans = find_all_spans(pat, text)
        for s, e, m in spans:
            hits.append((s, e, m))

    if not hits:
        return None

    evidence_spans = [{"start": s, "end": e, "excerpt": excerpt_span(text, s, e)} for s, e, _ in hits]
    return {
        "rule": "unlimited_liability",
        "severity": "high",
        "evidence_spans": evidence_spans,
        "explanation": "Language suggests unlimited liability or lack of liability cap — high risk to provider/party.",
    }


def detect_broad_indemnity(text: str) -> Optional[Dict[str, Any]]:
    # look for indemnity occurrences
    hits = []
    for pat in INDEMNITY_PATTERNS:
        spans = find_all_spans(pat, text)
        for s, e, m in spans:
            hits.append((s, e, m))

    if not hits:
        return None

    # further heuristics: if indemnity occurs and words like "hold harmless" or "defend" are used broadly, raise severity
    evidence_spans = []
    severity = "medium"
    for s, e, m in hits:
        evidence_spans.append({"start": s, "end": e, "excerpt": excerpt_span(text, s, e)})
        # check a small window for "no limit", "full", "all losses" which may indicate broad indemnity
        context_start = max(0, s - 200)
        context_end = min(len(text), e + 200)
        context = text[context_start:context_end].lower()
        if any(term in context for term in ["all losses", "full indemnity", "without limitation", "defend and hold harmless"]):
            severity = "high"

    return {
        "rule": "broad_indemnity",
        "severity": severity,
        "evidence_spans": evidence_spans,
        "explanation": "Indemnity language found; check scope, reciprocity, caps, and limitations.",
    }


def detect_confidentiality_weakness(text: str) -> Optional[Dict[str, Any]]:
    """
    If confidentiality clause is missing entirely -> high risk.
    If narrow/confined confidentiality (e.g., limited to 'written' only) -> medium risk.
    """
    found = False
    for pat in CONFIDENTIALITY_PATTERNS:
        if pat.search(text):
            found = True
            break

    if not found:
        return {
            "rule": "missing_confidentiality",
            "severity": "high",
            "evidence_spans": [],
            "explanation": "No confidentiality clause found in the document.",
        }

    # If confidentiality exists, check for narrow limitation words nearby
    # e.g., only 'in writing', 'only oral communications excluded', 'for a limited time'
    narrow_terms = re.compile(r"\b(in writing|written notice|limited to|for a period of)\b", re.I)
    matches = []
    for m in CONFIDENTIALITY_PATTERNS:
        for s, e, matched in find_all_spans(m, text):
            context_start = max(0, s - 200)
            context_end = min(len(text), e + 200)
            context = text[context_start:context_end]
            if narrow_terms.search(context):
                matches.append({"start": s, "end": e, "excerpt": excerpt_span(text, s, e)})

    if matches:
        return {
            "rule": "narrow_confidentiality",
            "severity": "medium",
            "evidence_spans": matches,
            "explanation": "Confidentiality clause present but contains limiting language; review scope/duration/definitions.",
        }

    return None


def detect_termination_short_notice(text: str) -> Optional[Dict[str, Any]]:
    """
    Detect termination language and try to find termination notice days (<=30 days is flagged).
    """
    findings = []
    for m in TERMINATION_PATTERN.finditer(text):
        s, e = m.start(), m.end()
        context_start = max(0, s - 300)
        context_end = min(len(text), e + 300)
        context = text[context_start:context_end]
        m_notice = NOTICE_DAYS_PATTERN.search(context)
        if m_notice:
            try:
                days = int(m_notice.group("notice"))
                start_abs = context_start + m_notice.start()
                end_abs = context_start + m_notice.end()
                findings.append((s, e, days, start_abs, end_abs))
            except Exception:
                findings.append((s, e, None, None, None))
        else:
            findings.append((s, e, None, None, None))

    if not findings:
        return None

    evidence_spans = []
    severity = "low"
    metadata = {"termination_matches": []}
    for s, e, days, ns, ne in findings:
        evidence_spans.append({"start": s, "end": e, "excerpt": excerpt_span(text, s, e)})
        if days is not None:
            metadata["termination_matches"].append({"notice_days": days})
            if days <= 30:
                severity = "high"
            elif days <= 60 and severity != "high":
                severity = "medium"

    return {
        "rule": "termination_notice",
        "severity": severity,
        "evidence_spans": evidence_spans,
        "metadata": metadata,
        "explanation": "Review termination notice periods; short notice may be high risk.",
    }


def detect_liability_cap(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract any liability cap amount and currency.
    If no cap phrase found and liability language exists, may indicate high risk (no cap)
    """
    # search for explicit cap phrase
    m = LIABILITY_CAP_PATTERN.search(text)
    if m:
        matched = m.group(0)
        groups = m.groupdict()
        amount = groups.get("amount")
        # try to pull currency and numeric portion
        cap_data = {"raw": matched, "parsed": amount}
        start, end = m.start(), m.end()
        return {
            "rule": "liability_cap",
            "severity": "info",
            "evidence_spans": [{"start": start, "end": end, "excerpt": excerpt_span(text, start, end)}],
            "metadata": cap_data,
            "explanation": "Liability cap detected; check amounts and currency.",
        }

    # fallback: if the doc mentions liability frequently but no cap phrase, flag medium-high
    liability_mentions = re.findall(r"\bliabilit(?:y|ies)\b", text, re.I)
    if len(liability_mentions) >= 3:
        return {
            "rule": "no_liability_cap_found",
            "severity": "medium",
            "evidence_spans": [],
            "explanation": "Document mentions liability multiple times but no explicit liability cap phrase detected.",
        }

    return None


# --- Main audit orchestration ---

def run_audit_on_text(text: str) -> List[Dict[str, Any]]:
    """
    Run all heuristics on a piece of document text and return findings.
    """
    findings = []

    try:
        for detector in [
            detect_auto_renewal,
            detect_unlimited_liability,
            detect_broad_indemnity,
            detect_confidentiality_weakness,
            detect_termination_short_notice,
            detect_liability_cap,
        ]:
            try:
                res = detector(text)
                if res:
                    findings.append(res)
            except Exception as e:
                logger.exception("Detector %s failed: %s", detector.__name__, e)
    except Exception as e:
        logger.exception("run_audit_on_text failed: %s", e)
        raise

    return findings


def audit_document(document_id: str, text_dir: Path = DEFAULT_TEXT_DIR) -> Dict[str, Any]:
    """
    Top-level helper: read extracted text for document_id and run the audits.
    Returns a dict with findings, document_id, and some basic metadata.
    """
    try:
        text = read_document_text(document_id, text_dir=text_dir)
    except FileNotFoundError as fe:
        logger.warning("audit_document: text not found for %s: %s", document_id, fe)
        return {"document_id": document_id, "error": "text not found", "findings": []}
    except Exception as e:
        logger.exception("audit_document: failed to read text for %s: %s", document_id, e)
        return {"document_id": document_id, "error": "failed to read text", "findings": []}

    findings = run_audit_on_text(text)

    # Compact results a bit: ensure evidence excerpts are present and trimmed
    for f in findings:
        if "evidence_spans" in f:
            for ev in f["evidence_spans"]:
                if "excerpt" not in ev:
                    s = ev.get("start", 0) or 0
                    e = ev.get("end", min(len(text), s + 80))
                    ev["excerpt"] = excerpt_span(text, s, e)

    return {"document_id": document_id, "findings": findings}


# quick CLI usage if run as script
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("document_id", help="document id (basename of .txt in store/texts)")
    ap.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    args = ap.parse_args()

    result = audit_document(args.document_id, text_dir=Path(args.text_dir))
    import json
    print(json.dumps(result, indent=2))

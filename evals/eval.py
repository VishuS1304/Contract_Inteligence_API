#!/usr/bin/env python3
"""
Simple eval script: posts each question to local API /ask and checks if the
reference substring appears in the returned answer. Prints accuracy.
"""
import json
import requests
from pathlib import Path

API = "http://127.0.0.1:8000/ask"

def read_qs(path="questions.jsonl"):
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def ask(question):
    # use multipart form to match curl/test usage
    r = requests.post(API, files={"question": (None, question)}, timeout=30)
    try:
        return r.json().get("answer", "")
    except Exception:
        return ""

def main():
    data = read_qs(Path(__file__).parent / "questions.jsonl")
    total = len(data)
    correct = 0
    results = []
    for rec in data:
        q = rec["question"]
        ref = rec["reference"]
        ans = ask(q).lower().strip()
        ok = False
        if ref and ref.lower().strip() in ans:
            ok = True
            correct += 1
        results.append({"q": q, "answer": ans, "reference": ref, "match": ok})
        print(f"Q: {q}\nA: {ans}\nREF: {ref}\nMATCH: {ok}\n---")
    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.3f} ({correct}/{total})")
    # write one-line score summary
    out = Path(__file__).parent / "score.txt"
    out.write_text(f"Accuracy: {acc:.3f} ({correct}/{total})\n", encoding="utf-8")

if __name__ == "__main__":
    main()

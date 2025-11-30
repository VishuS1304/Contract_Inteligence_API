# app/vectorbase.py

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("contract-intel.vectorbase")
logging.basicConfig(level=logging.INFO)

# Try optional dependencies
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAVE_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np  # type: ignore
    HAVE_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    HAVE_ST = False

# Defaults
DEFAULT_STORE_DIR = Path("./store/vectors")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384  # default embedding dim for all-MiniLM-L6-v2

# Thread lock for simple concurrency protection
_lock = threading.Lock()


class VectorBase:
    def __init__(self, store_dir: Path | str = DEFAULT_STORE_DIR, embed_model: str = DEFAULT_EMBED_MODEL):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.store_dir / "meta.json"
        self.vectors_path = self.store_dir / "vectors.npy"  # fallback persistence
        self.faiss_path = self.store_dir / "index.faiss"    # faiss persistence
        self._ids: List[str] = []       # ordered list of doc_ids corresponding to vectors rows
        self._texts: Dict[str, str] = {}  # optional store of raw text (small projects)
        self._emb_dim: int = DEFAULT_DIM
        self._index = None              # faiss index object or None
        self._vectors = None            # numpy array used when faiss not available
        self._model_name = embed_model
        self._model = None              # lazy-loaded SentenceTransformer model
        self._dirty = False

        # Try to load existing persistence if present
        self._load_meta()
        self._load_index_or_vectors()

    # ----------------- Embedding model helpers -----------------
    def _ensure_model(self):
        if self._model is None:
            if not HAVE_ST:
                raise RuntimeError("sentence-transformers is not installed; cannot compute embeddings")
            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            # attempt to set dim from model; default stays if we can't find it
            try:
                emb = self._model.encode(["hello"], convert_to_numpy=True, show_progress_bar=False)
                self._emb_dim = int(emb.shape[1])
            except Exception:
                logger.debug("Could not determine embedding dim from model; using default %s", self._emb_dim)

    def _embed_texts(self, texts: List[str]):
        self._ensure_model()
        try:
            emb = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return emb
        except Exception as e:
            logger.exception("Embedding failure: %s", e)
            raise

    # ----------------- Persistence helpers -----------------
    def _load_meta(self):
        if self.meta_path.exists():
            try:
                data = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self._ids = data.get("ids", [])
                self._texts = data.get("texts", {})
                self._emb_dim = int(data.get("dim", self._emb_dim))
                logger.info("Loaded meta: %d ids (dim=%d)", len(self._ids), self._emb_dim)
            except Exception:
                logger.exception("Failed to load meta.json - starting fresh")
                self._ids = []
                self._texts = {}
        else:
            self._ids = []
            self._texts = {}

    def _write_meta(self):
        data = {"ids": self._ids, "texts": self._texts, "dim": self._emb_dim}
        self.meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._dirty = False

    def _load_index_or_vectors(self):
        """
        Load FAISS index if available and file exists; otherwise load numpy vectors if present.
        """
        if HAVE_FAISS and self.faiss_path.exists():
            try:
                self._index = faiss.read_index(str(self.faiss_path))
                logger.info("Loaded faiss index from %s", self.faiss_path)
                return
            except Exception:
                logger.exception("Failed to load faiss index; falling back to numpy vectors")

        # fallback to numpy vectors
        if np is not None and self.vectors_path.exists():
            try:
                self._vectors = np.load(str(self.vectors_path))
                logger.info("Loaded numpy vectors from %s (shape=%s)", self.vectors_path, getattr(self._vectors, "shape", None))
            except Exception:
                logger.exception("Failed to load vectors.npy; starting fresh")
                self._vectors = None

    def _save_index_or_vectors(self):
        """
        Persist index state. If using faiss, write index file; else save numpy vectors.
        Always persist meta.json.
        """
        # ensure store_dir exists
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if HAVE_FAISS and self._index is not None:
            try:
                faiss.write_index(self._index, str(self.faiss_path))
                logger.info("Persisted faiss index to %s", self.faiss_path)
            except Exception:
                logger.exception("Failed to write faiss index")
        else:
            if np is not None and self._vectors is not None:
                try:
                    np.save(str(self.vectors_path), self._vectors)
                    logger.info("Persisted numpy vectors to %s", self.vectors_path)
                except Exception:
                    logger.exception("Failed to save vectors.npy")

        # always write meta
        self._write_meta()

    # ----------------- Core API -----------------
    def add(self, doc_id: str, text: str):
        """
        Add a document by id and text. Computes embedding and appends to index.
        If doc_id already exists, it will be replaced.
        """
        if not doc_id or text is None:
            raise ValueError("doc_id and text are required")

        with _lock:
            # if exists, remove first
            if doc_id in self._ids:
                logger.info("doc_id %s already exists; removing before re-adding", doc_id)
                self.remove(doc_id)

            # compute embedding (for entire document)
            emb = None
            try:
                emb = self._embed_texts([text])
            except Exception:
                emb = None

            if HAVE_FAISS and emb is not None:
                # ensure index with correct dim
                d = emb.shape[1]
                if self._index is None:
                    logger.info("Creating new faiss IndexFlatIP dim=%d", d)
                    self._index = faiss.IndexFlatIP(d)
                    self._emb_dim = d
                # add vector
                try:
                    self._index.add(emb.astype("float32"))
                except Exception:
                    # sometimes need to convert
                    self._index.add(emb)
                # append metadata
                self._ids.append(doc_id)
            else:
                # numpy fallback
                if np is None:
                    raise RuntimeError("Numpy is required for numpy fallback but not available")
                vec = emb.astype("float32") if emb is not None else self._random_vector_fallback()
                if self._vectors is None:
                    self._vectors = vec
                else:
                    self._vectors = np.vstack([self._vectors, vec])
                self._ids.append(doc_id)

            # store text snippet (keeps small projects handy)
            self._texts[doc_id] = text[: 64 * 1024]  # cap saved text to 64KB
            self._dirty = True
            logger.info("Added doc_id=%s to vectorbase (total=%d)", doc_id, len(self._ids))

    def _random_vector_fallback(self):
        """
        If embedding model failed and numpy available, create a deterministic pseudo-vector fallback
        based on hash; not ideal but avoids crashes.
        """
        import hashlib
        if np is None:
            raise RuntimeError("Numpy required for fallback vector")
        seed = 1234
        vec = (np.arange(self._emb_dim, dtype="float32") + seed) % 1000
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.reshape(1, -1)

    def remove(self, doc_id: str) -> bool:
        """
        Remove a document by id. Returns True if removed, False if not found.
        Note: removal from faiss.IndexFlatIP is not trivial; this implementation rebuilds index when needed.
        """
        with _lock:
            if doc_id not in self._ids:
                return False
            idx = self._ids.index(doc_id)
            # remove from ids and texts
            self._ids.pop(idx)
            self._texts.pop(doc_id, None)

            # rebuild index / vectors without the removed row
            if HAVE_FAISS and self._index is not None:
                try:
                    # extract all embeddings from saved numpy (if present) or rebuild from stored texts
                    # easiest reliable approach: rebuild index from scratch using stored texts
                    logger.info("Rebuilding faiss index after removal (this may be slow for many docs)")
                    self._index = None
                    existing_texts = [self._texts.get(iid, "") for iid in self._ids]
                    # clear vectors store and re-add
                    self._vectors = None
                    for iid, txt in zip(self._ids, existing_texts):
                        try:
                            emb = self._embed_texts([txt])
                        except Exception:
                            emb = None
                        if self._index is None and emb is not None:
                            self._index = faiss.IndexFlatIP(emb.shape[1])
                        if emb is not None:
                            self._index.add(emb.astype("float32"))
                    logger.info("Rebuild complete (count=%d)", len(self._ids))
                except Exception:
                    logger.exception("Failed to rebuild faiss index after removal")
            else:
                # numpy fallback: remove row from _vectors
                if self._vectors is not None:
                    try:
                        import numpy as _np  # local import to avoid errors if numpy missing
                        self._vectors = _np.delete(self._vectors, idx, axis=0)
                        logger.info("Removed vector row at index %d (new shape=%s)", idx, getattr(self._vectors, "shape", None))
                    except Exception:
                        logger.exception("Failed to remove vector row; clearing vectors and rebuilding from texts")
                        # rebuild from available texts
                        try:
                            arrs = []
                            for iid in self._ids:
                                txt = self._texts.get(iid, "")
                                emb = self._embed_texts([txt]) if HAVE_ST else None
                                if emb is not None:
                                    arrs.append(emb)
                            if arrs:
                                self._vectors = _np.vstack(arrs)
                            else:
                                self._vectors = None
                        except Exception:
                            self._vectors = None

            self._dirty = True
            return True

    def clear(self):
        """Remove all stored vectors & metadata (in-memory and persisted files)."""
        with _lock:
            self._ids = []
            self._texts = {}
            self._index = None
            self._vectors = None
            self._dirty = True
            # remove persisted files
            try:
                if self.faiss_path.exists():
                    self.faiss_path.unlink()
                if self.vectors_path.exists():
                    self.vectors_path.unlink()
                if self.meta_path.exists():
                    self.meta_path.unlink()
                logger.info("Cleared persisted vectorbase files")
            except Exception:
                logger.exception("Failed to remove persisted vector files")

    # ----------------- Search -----------------
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the query and return top_k results: list of {"doc_id","score"} ordered by score desc.
        """
        with _lock:
            if not self._ids:
                logger.debug("VectorBase is empty; returning []")
                return []

            q_emb = None
            if HAVE_ST:
                try:
                    q_emb = self._embed_texts([query])
                except Exception:
                    q_emb = None

            results: List[Tuple[float, int]] = []  # (score, idx)
            if HAVE_FAISS and self._index is not None and q_emb is not None:
                try:
                    D, I = self._index.search(q_emb.astype("float32"), k=top_k)
                    for score, idx in zip(D[0].tolist(), I[0].tolist()):
                        if idx < 0:
                            continue
                        results.append((float(score), int(idx)))
                except Exception:
                    logger.exception("Faiss search failed; will fallback to numpy scoring")

            if (not results) and (self._vectors is not None and np is not None and q_emb is not None):
                # numpy dot-product / cosine
                try:
                    q = q_emb[0]
                    # normalize
                    def norm(a): return a / (np.linalg.norm(a) + 1e-9)
                    qn = norm(q)
                    V = np.array([norm(v) for v in self._vectors])
                    scores = V.dot(qn)
                    # argsort desc
                    idxs = list(reversed(scores.argsort().tolist()))
                    for idx in idxs[:top_k]:
                        results.append((float(scores[idx]), int(idx)))
                except Exception:
                    logger.exception("Numpy scoring failed")

            # final fallback: if embeddings not available, do simple token overlap between query and stored texts
            if not results:
                try:
                    import re as _re
                    q_tokens = set(_re.findall(r"\w+", query.lower()))
                    scored = []
                    for i, did in enumerate(self._ids):
                        txt = self._texts.get(did, "") or ""
                        p_tokens = set(_re.findall(r"\w+", txt.lower()))
                        inter = q_tokens.intersection(p_tokens)
                        sc = float(len(inter)) / max(1.0, len(p_tokens))
                        scored.append((sc, i))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    for sc, idx in scored[:top_k]:
                        results.append((float(sc), int(idx)))
                except Exception:
                    logger.exception("Token-overlap fallback failed")

            # format results
            out = []
            seen = set()
            for score, idx in results:
                if idx < 0 or idx >= len(self._ids):
                    continue
                did = self._ids[idx]
                if did in seen:
                    continue
                seen.add(did)
                out.append({"doc_id": did, "score": float(score)})
            return out

    # ----------------- Persistence control -----------------
    def save(self):
        """Persist index & meta to disk."""
        with _lock:
            if not self._dirty:
                # still write meta to capture any change in texts/ids
                self._write_meta()
                logger.debug("No changes flagged, meta still persisted")
                return
            self._save_index_or_vectors()
            logger.info("VectorBase saved to disk")

    def load(self):
        """Reload metadata & persisted index from disk (replaces current in-memory state)."""
        with _lock:
            self._load_meta()
            self._load_index_or_vectors()
            logger.info("VectorBase loaded from disk")

    # ----------------- Utilities -----------------
    def info(self) -> Dict[str, Any]:
        """Return simple status info."""
        return {
            "count": len(self._ids),
            "dim": self._emb_dim,
            "faiss": HAVE_FAISS,
            "embedding_available": HAVE_ST,
            "store_dir": str(self.store_dir),
        }


# -------------------- CLI for quick tests --------------------
if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true", help="clear persisted index")
    ap.add_argument("--add", nargs=2, metavar=("DOCID", "TEXT"), help="add a small doc (docid text)")
    ap.add_argument("--search", metavar="QUERY", help="search query")
    ap.add_argument("--save", action="store_true", help="save to disk")
    ap.add_argument("--info", action="store_true", help="show info")
    args = ap.parse_args()

    vb = VectorBase()
    if args.clear:
        vb.clear()
        print("cleared")
    if args.add:
        docid, txt = args.add
        vb.add(docid, txt)
        print("added", docid)
    if args.search:
        res = vb.search(args.search, top_k=5)
        print(json.dumps(res, indent=2))
    if args.save:
        vb.save()
        print("saved")
    if args.info:
        print(json.dumps(vb.info(), indent=2))

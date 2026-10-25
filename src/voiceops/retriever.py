"""Hybrid retrieval: multilingual dense embeddings + BM25, fused with reciprocal rank fusion.

Dense search handles paraphrase ("send it back" vs "reverse pickup") and Hindi queries;
BM25 catches exact tokens such as SKUs, amounts and section words. Superseded documents
and marketing copy are down-weighted so the current policy wins (KB-00.1).
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .kb import Chunk, load_chunks

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*|[ऀ-ॿ]+")
SKU_RE = re.compile(r"\b([A-Z]{3})[-\s]?(\d{4})\b", re.I)
RRF_K = 60


DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
               "six": "6", "seven": "7", "eight": "8", "nine": "9"}
# ASR homophones, only trusted when they sit inside a run of digits.
DIGIT_HOMOPHONES = {"to": "2", "too": "2", "for": "4", "won": "1", "ate": "8"}
SPOKEN_RUN_RE = re.compile(
    r"\b(?:(?:%s|\d)(?:[\s-]+|$))+" % "|".join(list(DIGIT_WORDS) + list(DIGIT_HOMOPHONES)), re.I)


def normalize_spoken_numbers(text: str) -> str:
    """'ele-zero zero one to' -> 'ele-0012'; 'eight nine seven eight' -> '8978'."""
    def fix(m: re.Match) -> str:
        words = re.split(r"[\s-]+", m.group(0).strip())
        digits = [DIGIT_WORDS.get(w.lower()) or DIGIT_HOMOPHONES.get(w.lower()) or w for w in words]
        n_real = sum(w.lower() in DIGIT_WORDS or w.isdigit() for w in words)
        if n_real < 2:  # a lone "one" or "to" is just a word
            return m.group(0)
        return "".join(digits) + " "
    return SPOKEN_RUN_RE.sub(fix, text)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        return {"id": self.chunk.id, "doc": self.chunk.doc, "title": self.chunk.title,
                "score": round(self.score, 4), "text": self.chunk.text.strip()}


class Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.search_text()) for c in self.chunks])
        self._embedder = None
        self._lock = threading.Lock()
        self.vectors = self._load_vectors()
        # Product titles without the colour/variant suffix, for name matching.
        self._product_names = [
            (c, c.title.split(",")[0].lower()) for c in self.chunks if c.doc_type == "catalog"]

    # ---- dense index -------------------------------------------------------------
    def _embed(self, texts: list[str], query: bool) -> np.ndarray:
        with self._lock:
            if self._embedder is None:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.CACHE_DIR / "models"))
            fn = self._embedder.query_embed if query else self._embedder.passage_embed
            vecs = np.array(list(fn(texts)), dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def _load_vectors(self) -> np.ndarray | None:
        texts = [c.search_text() for c in self.chunks]
        key = hashlib.sha256(("\x00".join(texts) + config.EMBED_MODEL).encode()).hexdigest()[:16]
        path = config.CACHE_DIR / f"kb_vectors_{key}.npy"
        if path.exists():
            return np.load(path)
        try:
            vecs = self._embed(texts, query=False)
        except Exception as exc:  # no network for the model download, etc.
            log.warning("Dense embeddings unavailable (%s); falling back to BM25 only.", exc)
            return None
        np.save(path, vecs)
        return vecs

    # ---- search ------------------------------------------------------------------
    def search(self, query: str, k: int = config.TOP_K, include_catalog: bool = True) -> list[Hit]:
        n = len(self.chunks)
        fused = np.zeros(n)
        query = normalize_spoken_numbers(query)
        # Name a product -> also search with its catalog attributes, so the policies that
        # govern it surface ("innerwear" finds RET-5, "assured" finds WAR-2).
        mentioned = self.mentioned_products(query)
        rows = [_row(self.by_id[m]) for m in sorted(mentioned)]
        cats = {r.get("return_category", "") for r in rows} - {""}
        for r in rows:
            query += (f"\n{r.get('return_category', '')} category return window"
                      + (" ZentroMart Assured warranty" if r.get("assured") == "True" else "")
                      + (" brand warranty" if r.get("warranty_months", "0") != "0" else "")
                      + (" marketplace seller" if r.get("fulfilled_by") not in (None, config.COMPANY) else ""))

        bm = self.bm25.get_scores(tokenize(query))
        for rank, i in enumerate(np.argsort(-bm)[:30]):
            if bm[i] > 0:
                fused[i] += 1.0 / (RRF_K + rank)

        if self.vectors is not None:
            sims = self.vectors @ self._embed([query], query=True)[0]
            for rank, i in enumerate(np.argsort(-sims)[:30]):
                fused[i] += 1.0 / (RRF_K + rank)

        # Exact product mentions (SKU or name) pin the catalog row, and current policy
        # sections naming that product's category get a nudge.
        for i, c in enumerate(self.chunks):
            if c.id in mentioned:
                fused[i] += 0.05
            elif cats and c.doc_type == "policy" and any(cat in c.text.lower() for cat in cats):
                fused[i] += 0.01
            if c.superseded:
                fused[i] *= 0.6
            elif c.doc_type == "marketing":
                fused[i] *= 0.8

        order = [i for i in np.argsort(-fused) if fused[i] > 0]
        if not include_catalog:
            order = [i for i in order if self.chunks[i].doc_type != "catalog"]
        # Cap catalog rows so near-duplicate product variants don't crowd out policy text.
        hits, n_catalog = [], 0
        for i in order:
            c = self.chunks[i]
            if c.doc_type == "catalog":
                if n_catalog >= 2 and c.id not in mentioned:
                    continue
                n_catalog += 1
            hits.append(Hit(c, float(fused[i])))
            if len(hits) >= k:
                break
        return hits

    def mentioned_products(self, query: str) -> set[str]:
        """Products named in the query. An explicit SKU wins over a name shared by variants."""
        by_sku = {f"products.csv:{a.upper()}-{b}" for a, b in SKU_RE.findall(query)} & self.by_id.keys()
        if by_sku:
            return by_sku
        q = query.lower()
        return {c.id for c, name in self._product_names if name and name in q}

    def find_products(self, text: str, limit: int = 5) -> list[Chunk]:
        """Catalog lookup by SKU or fuzzy name, for the agent's product tool."""
        text = normalize_spoken_numbers(text)
        ids = self.mentioned_products(text)
        hits = [self.by_id[i] for i in sorted(ids)]
        if len(hits) < limit:
            words = set(tokenize(text))
            scored = []
            for c, name in self._product_names:
                overlap = len(words & set(tokenize(c.title)))
                if overlap and c not in hits:
                    scored.append((overlap, c))
            scored.sort(key=lambda t: -t[0])
            hits += [c for _, c in scored[: limit - len(hits)]]
        return hits[:limit]


def _row(chunk: Chunk) -> dict:
    """Catalog chunk text ('k: v; k: v') back to a dict."""
    return dict(p.split(": ", 1) for p in chunk.text.split("; ") if ": " in p)


_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER

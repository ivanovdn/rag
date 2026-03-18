"""
Pure-Python BM25 keyword index for hybrid search.

No external dependencies — just math and re.
The index is stored in memory and persisted to a JSON file so it
survives restarts without re-ingestion.

BM25 parameters (Okapi BM25):
  k1 = 1.2  (term frequency saturation)
  b  = 0.75 (document length normalization)
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from config import settings
from ingest.chunk_models import PolicyChunk

logger = logging.getLogger(__name__)

_INDEX_PATH = Path(".bm25_index.json")

# BM25 parameters
_K1 = 1.2
_B = 0.75

# Stop words to skip during tokenization
_STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or the "
    "to was were with that this these those not but they them their its "
    "will can may shall should would could also been being into such than "
    "each other which who whom what where when how all any both few more "
    "most no nor some do does did doing done".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class BM25Index:
    """In-memory BM25 index with JSON persistence."""

    def __init__(self):
        # chunk_id -> {doc_id, doc_title, section_display, clause_number, doc_link, text, tokens}
        self.documents: dict[str, dict] = {}
        # term -> set of chunk_ids containing that term
        self.inverted_index: dict[str, set[str]] = {}
        # Pre-computed: average document length
        self.avg_dl: float = 0.0

    def _recompute_avg_dl(self):
        if not self.documents:
            self.avg_dl = 0.0
            return
        total = sum(len(d["tokens"]) for d in self.documents.values())
        self.avg_dl = total / len(self.documents)

    def add_chunks(self, chunks: list[PolicyChunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self.documents[chunk.chunk_id] = {
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "section": chunk.section,
                "section_number": chunk.section_number,
                "clause": chunk.clause,
                "clause_number": chunk.clause_number,
                "section_display": chunk.section_display,
                "doc_link": chunk.doc_link,
                "text": chunk.text,
                "tokens": tokens,
            }
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(chunk.chunk_id)

        self._recompute_avg_dl()

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks for a given doc_id. Returns number removed."""
        to_remove = [
            cid for cid, d in self.documents.items() if d["doc_id"] == doc_id
        ]
        for cid in to_remove:
            tokens = set(self.documents[cid]["tokens"])
            for token in tokens:
                if token in self.inverted_index:
                    self.inverted_index[token].discard(cid)
                    if not self.inverted_index[token]:
                        del self.inverted_index[token]
            del self.documents[cid]

        if to_remove:
            self._recompute_avg_dl()
        return len(to_remove)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search the index. Returns list of (chunk_id, bm25_score) sorted by score desc.
        """
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self.documents)
        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            doc_ids_with_term = self.inverted_index[token]
            df = len(doc_ids_with_term)
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for cid in doc_ids_with_term:
                doc = self.documents[cid]
                tf = Counter(doc["tokens"])[token]
                dl = len(doc["tokens"])
                # BM25 term score
                numerator = tf * (_K1 + 1)
                denominator = tf + _K1 * (1 - _B + _B * dl / self.avg_dl)
                score = idf * numerator / denominator
                scores[cid] = scores.get(cid, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_document(self, chunk_id: str) -> dict | None:
        """Get stored document metadata by chunk_id."""
        return self.documents.get(chunk_id)

    def save(self, path: Path | None = None) -> None:
        """Persist index to JSON."""
        path = path or _INDEX_PATH
        data = {
            "documents": {
                cid: {k: v for k, v in doc.items() if k != "tokens"}
                for cid, doc in self.documents.items()
            }
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        logger.info(f"BM25 index saved: {len(self.documents)} chunks → {path}")

    def load(self, path: Path | None = None) -> bool:
        """Load index from JSON. Returns True if loaded successfully."""
        path = path or _INDEX_PATH
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.documents = {}
            self.inverted_index = {}

            for cid, doc in data["documents"].items():
                tokens = _tokenize(doc["text"])
                doc["tokens"] = tokens
                self.documents[cid] = doc
                for token in set(tokens):
                    if token not in self.inverted_index:
                        self.inverted_index[token] = set()
                    self.inverted_index[token].add(cid)

            self._recompute_avg_dl()
            logger.info(f"BM25 index loaded: {len(self.documents)} chunks from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False


# --- Module-level singleton ---

_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Get or create the singleton BM25 index, loading from disk if available."""
    global _index
    if _index is None:
        _index = BM25Index()
        _index.load()
    return _index


def add_chunks_to_bm25(chunks: list[PolicyChunk]) -> None:
    """Add chunks to the BM25 index and persist."""
    idx = get_bm25_index()
    idx.add_chunks(chunks)
    idx.save()


def remove_document_from_bm25(doc_id: str) -> int:
    """Remove document from BM25 index and persist. Returns chunks removed."""
    idx = get_bm25_index()
    removed = idx.remove_document(doc_id)
    if removed:
        idx.save()
    return removed


def search_bm25(query: str, top_k: int | None = None) -> list[tuple[str, float, dict]]:
    """
    Search BM25 index. Returns list of (chunk_id, score, metadata_dict).
    """
    idx = get_bm25_index()
    k = top_k or settings.hybrid_bm25_candidates
    results = idx.search(query, top_k=k)
    return [
        (cid, score, idx.get_document(cid))
        for cid, score in results
        if idx.get_document(cid) is not None
    ]

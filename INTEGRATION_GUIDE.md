# Hybrid Search Integration Guide
# ================================
# 
# This file shows the EXACT changes needed in your existing codebase.
# Two new files are added, and three existing files need small edits.
#
# New files:
#   rag/bm25_index.py      ← BM25 keyword index (new)
#   rag/hybrid_search.py   ← RRF fusion logic (new)
#
# Modified files:
#   config.py              ← add 2 new settings
#   ingest/pipeline.py     ← add BM25 sync after Qdrant upsert
#   rag/tools/search_policies.py  ← swap vector search → hybrid search


# ============================================================
# 1. config.py — ADD these fields to your Settings class
# ============================================================

# Add to the Settings class, in the "Retrieval" group:

"""
    # Hybrid search
    bm25_enabled: bool = True              # set False to disable BM25 (vector-only fallback)
    hybrid_vector_candidates: int = 20     # how many to retrieve from Qdrant before RRF
    hybrid_bm25_candidates: int = 20       # how many to retrieve from BM25 before RRF
"""

# Add to .env:
"""
BM25_ENABLED=true
HYBRID_VECTOR_CANDIDATES=20
HYBRID_BM25_CANDIDATES=20
"""


# ============================================================
# 2. ingest/pipeline.py — ADD BM25 sync
# ============================================================

# At the top, add import:
"""
from rag.bm25_index import add_chunks_to_bm25, remove_document_from_bm25
"""

# In ingest_document(), AFTER the upsert_chunks() call, add:
"""
    # Sync BM25 index
    if settings.bm25_enabled:
        add_chunks_to_bm25(chunks)
        logger.info(f"Added {len(chunks)} chunks to BM25 index")
"""

# In ingest_document(), BEFORE delete_document() (the Qdrant delete), add:
"""
    # Remove from BM25 index first (before Qdrant delete)
    if settings.bm25_enabled:
        remove_document_from_bm25(doc_id)
"""

# The full ingest_document function should look like:
"""
def ingest_document(filepath: Path, doc_link: str) -> int:
    chunks = parse_docx(filepath, doc_link)
    if not chunks:
        return 0
    
    doc_id = chunks[0].doc_id
    
    # Remove old version (both stores)
    if settings.bm25_enabled:
        remove_document_from_bm25(doc_id)
    delete_document(doc_id)
    
    # Embed and store
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(chunks, embeddings)
    
    # Sync BM25
    if settings.bm25_enabled:
        add_chunks_to_bm25(chunks)
    
    return len(chunks)
"""


# ============================================================
# 3. rag/tools/search_policies.py — SWAP to hybrid search
# ============================================================

# This is the biggest change. Replace the vector-only search
# with a call to hybrid_search_formatted().

# BEFORE (current code, approximately):
"""
from rag.embeddings import embed_query
from rag.vector_store import search_vectors
from config import settings

def search_policies(query: str, top_k: int = 6) -> str:
    vector = embed_query(query)
    results = search_vectors(vector, top_k=top_k)
    
    if not results or results[0].score < settings.min_confidence_score:
        return "NO_RELEVANT_POLICY_FOUND"
    
    # ... format results ...
"""

# AFTER (with hybrid search):
"""
from rag.hybrid_search import hybrid_search_formatted
from config import settings

def search_policies(query: str, top_k: int = 6) -> str:
    \"\"\"
    Search approved compliance policy documents for information relevant to the query.
    Uses hybrid search (semantic + keyword matching) for best accuracy.
    
    Args:
        query: Natural language search query describing what policy info you need.
               Be specific — include relevant terms, clause numbers, or policy names.
        top_k: Number of most relevant policy sections to return (default 6).
    
    Returns:
        Formatted policy excerpts with document name, section, clause number,
        and link. Returns "NO_RELEVANT_POLICY_FOUND" if no policies match.
    \"\"\"
    if settings.bm25_enabled:
        return hybrid_search_formatted(
            query=query,
            top_k=top_k,
            min_confidence=settings.min_confidence_score,
        )
    else:
        # Fallback: vector-only search (original behavior)
        from rag.embeddings import embed_query
        from rag.vector_store import search_vectors
        
        vector = embed_query(query)
        results = search_vectors(vector, top_k=top_k)
        
        if not results or results[0].score < settings.min_confidence_score:
            return "NO_RELEVANT_POLICY_FOUND"
        
        lines = []
        for r in results:
            p = r.payload
            lines.append(
                f"[SCORE: {r.score:.2f}] "
                f"Document: {p.get('doc_title', '')} | "
                f"Section: {p.get('section_display', '')} | "
                f"Clause: {p.get('clause_number', '')} | "
                f"Link: {p.get('doc_link', '')}\\n"
                f"Text: {p.get('text', '')}"
            )
        return "\\n\\n---\\n\\n".join(lines)
"""


# ============================================================
# 4. requirements.txt — NO new dependencies needed!
# ============================================================

# The BM25 implementation is pure Python (just math and re).
# No need for rank_bm25, nltk, or any external library.
# This was intentional — fewer dependencies = fewer problems.


# ============================================================
# 5. Updated project structure
# ============================================================

"""
compliance-bot/
├── config.py                          # + bm25_enabled, hybrid_*_candidates
│
├── ingest/
│   ├── chunk_models.py
│   ├── docx_parser.py
│   └── pipeline.py                    # + BM25 sync after Qdrant upsert
│
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_index.py                  # ← NEW: BM25 keyword index
│   ├── hybrid_search.py               # ← NEW: RRF fusion logic
│   ├── agent.py
│   └── tools/
│       ├── search_policies.py         # ← MODIFIED: uses hybrid_search
│       ├── get_section.py
│       ├── clarify.py
│       └── escalate.py
"""


# ============================================================
# 6. How to test
# ============================================================

"""
# After making the changes, re-ingest your documents:
python scripts/ingest_all.py --folder ./policies

# This will:
# 1. Parse DOCX files (unchanged)
# 2. Embed and upsert to Qdrant (unchanged)
# 3. Build BM25 index alongside (new)

# Then test with queries that exercise both search types:

# Semantic match (vector wins):
python scripts/test_query.py "What's the process for reporting misconduct?"

# Keyword match (BM25 wins):
python scripts/test_query.py "What does clause 4.2.1 say about PEP screening?"

# Both (hybrid wins):
python scripts/test_query.py "EDD requirements for high-risk customers"

# Check the logs — you'll see:
# INFO: Vector search: 20 results, top score: 0.83
# INFO: BM25 search: 20 results, top score: 12.34
# INFO: Top result after RRF: AML Policy | 4.2.1 (from vector #2, BM25 #1)
#                                          ↑ BM25 keyword match boosted this
"""

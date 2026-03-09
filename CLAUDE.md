# Compliance Q&A Bot — Claude Code Instructions

## Project Overview

Build an **internal Compliance Q&A Bot** using an **Agentic RAG** architecture.
The bot answers employee questions **strictly from approved internal policy documents (DOCX, ~50 files)**.
If an answer cannot be grounded in policy, it escalates to the Compliance team with full context.

**LLM: Local only via Ollama. No external API calls for inference.**

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Ollama (`qwen2.5:14b` default) | Local, no cloud |
| Embedding | `nomic-embed-text` via Ollama | Local embedding |
| Vector Store | Qdrant (Docker, local) | Lightweight, Python-native |
| RAG Framework | LlamaIndex | Native Ollama + Qdrant support |
| Agent | LlamaIndex `ReActAgent` | Tool-calling loop |
| Document Parsing | `python-docx` | Section/clause-aware chunking |
| Backend API | FastAPI | REST + WebSocket |
| Frontend | React + Tailwind CSS | Single-page chat UI |
| Metadata DB | SQLite (via SQLAlchemy) | Escalations, sessions, users |
| Escalation | Email (SMTP) + internal queue | Configurable |

---

## Project Structure

```
compliance-bot/
├── CLAUDE.md                    # This file
├── .env.example                 # Environment variables template
├── docker-compose.yml           # Qdrant + optional services
├── requirements.txt
│
├── ingest/
│   ├── __init__.py
│   ├── docx_parser.py           # Section/clause-aware DOCX chunker
│   ├── chunk_models.py          # Pydantic models for chunks
│   └── pipeline.py              # Full ingestion pipeline
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py            # Ollama nomic-embed-text setup
│   ├── vector_store.py          # Qdrant collection setup
│   ├── retriever.py             # Hybrid retrieval + reranking
│   ├── agent.py                 # ReActAgent + 4 tools
│   └── tools/
│       ├── __init__.py
│       ├── search_policies.py   # Tool 1: semantic search
│       ├── get_section.py       # Tool 2: fetch full clause
│       ├── clarify.py           # Tool 3: ask user clarification
│       └── escalate.py          # Tool 4: escalate to Compliance
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   ├── routes/
│   │   ├── chat.py              # POST /chat, WS /chat/ws
│   │   ├── escalations.py       # GET/PATCH /escalations (admin)
│   │   └── documents.py         # POST /ingest, GET /documents
│   ├── models.py                # Pydantic request/response schemas
│   └── auth.py                  # Simple role-based (user/admin)
│
├── db/
│   ├── __init__.py
│   ├── database.py              # SQLAlchemy engine + session
│   └── models.py                # Escalation, Session, User tables
│
├── notification/
│   ├── __init__.py
│   └── email.py                 # SMTP escalation email
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── CitationCard.jsx
│   │   │   └── EscalationPanel.jsx  # Admin view
│   │   └── api/
│   │       └── client.js
│   └── public/
│
├── tests/
│   ├── test_parser.py
│   ├── test_retrieval.py
│   └── test_agent.py
│
└── scripts/
    ├── ingest_all.py            # CLI: ingest all docs from folder
    └── test_query.py            # CLI: test agent with a query
```

---

## Phase 1: Document Ingestion Pipeline

### 1.1 DOCX Parser — `ingest/docx_parser.py`

This is the most critical component. Naive token-splitting destroys compliance documents.
You MUST implement **hierarchical, structure-aware chunking**.

**Rules:**
- Parse heading hierarchy: `Heading 1` → `Heading 2` → `Heading 3` = Document → Section → Subsection
- Detect numbered clauses via regex: `^\d+(\.\d+)*\s` (e.g. `4.2.1 Data Retention`)
- Each chunk = one atomic clause or paragraph under a heading
- Never split a single numbered clause across two chunks
- Minimum chunk size: 50 tokens. Maximum: 400 tokens
- If a clause exceeds 400 tokens, split at sentence boundary and mark as `part_1_of_N`

**Chunk metadata schema (store in Qdrant payload):**

```python
class PolicyChunk(BaseModel):
    chunk_id: str           # uuid
    doc_id: str             # slugified filename
    doc_title: str          # e.g. "Remote Work Policy"
    doc_filename: str       # e.g. "remote_work_policy_v3.docx"
    doc_link: str           # URL or file path to source document
    section_path: list[str] # ["4. Data Privacy", "4.2 Retention", "4.2.1 Personal Data"]
    section_display: str    # "4. Data Privacy > 4.2 Retention > 4.2.1 Personal Data"
    clause_number: str      # "4.2.1" or "" if no clause number
    text: str               # actual chunk text
    char_count: int
    chunk_index: int        # position within document
    last_updated: str       # ISO date
```

**Implementation approach:**

```python
import re
from docx import Document
from pathlib import Path

def extract_heading_level(paragraph) -> int | None:
    """Returns 1, 2, 3 etc. for Heading styles, None otherwise"""

def extract_clause_number(text: str) -> str | None:
    """Regex: matches 1.2, 4.2.1, 3. etc at start of paragraph"""
    pattern = r'^(\d+(?:\.\d+)*\.?)\s'
    match = re.match(pattern, text.strip())
    return match.group(1) if match else None

def parse_docx(filepath: Path, doc_link: str) -> list[PolicyChunk]:
    """Main parser. Returns flat list of PolicyChunk objects."""
    doc = Document(filepath)
    chunks = []
    current_headings = ["", "", ""]  # [h1, h2, h3]
    current_text_buffer = []
    
    for para in doc.paragraphs:
        level = extract_heading_level(para)
        if level:
            # flush buffer as chunk
            # update heading path
            ...
        else:
            clause_num = extract_clause_number(para.text)
            if clause_num and current_text_buffer:
                # flush previous clause as chunk
                ...
            current_text_buffer.append(para.text)
    
    # flush final buffer
    return chunks
```

Also handle **tables** in DOCX: convert each table row to a text line `"Column1: Value | Column2: Value"` and attach to the current section chunk.

---

### 1.2 Ingestion Pipeline — `ingest/pipeline.py`

```python
async def ingest_document(filepath: Path, doc_link: str):
    """Parse → embed → upsert to Qdrant"""
    chunks = parse_docx(filepath, doc_link)
    embeddings = await embed_chunks([c.text for c in chunks])
    await upsert_to_qdrant(chunks, embeddings)
    return len(chunks)

async def ingest_folder(folder: Path, base_url: str):
    """Batch ingest all .docx files in a folder"""
    for docx_file in folder.glob("*.docx"):
        doc_link = f"{base_url}/{docx_file.name}"
        count = await ingest_document(docx_file, doc_link)
        print(f"Ingested {docx_file.name}: {count} chunks")
```

**CLI script `scripts/ingest_all.py`:**
```bash
python scripts/ingest_all.py --folder ./policies --base-url http://internal.company.com/policies
```

---

## Phase 2: Vector Store Setup

### 2.1 Qdrant Collection — `rag/vector_store.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "compliance_policies"
VECTOR_DIM = 768  # nomic-embed-text output dimension

def init_qdrant():
    client = QdrantClient(url="http://localhost:6333")
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
    return client
```

**Qdrant runs via Docker:**
```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
```

---

### 2.2 Embeddings — `rag/embeddings.py`

Use Ollama's `nomic-embed-text` model. Do NOT use OpenAI or any remote embedding API.

```python
from llama_index.embeddings.ollama import OllamaEmbedding

def get_embedding_model():
    return OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0}
    )
```

Ensure `nomic-embed-text` is pulled: `ollama pull nomic-embed-text`

---

## Phase 3: The 4 Agent Tools

These are the heart of the agentic approach. Each tool must have a precise docstring because the LLM reads it to decide when to call it.

### Tool 1: `search_policies` — `rag/tools/search_policies.py`

```python
from llama_index.core.tools import FunctionTool

def search_policies(query: str, top_k: int = 6) -> str:
    """
    Search the approved internal policy and procedure documents 
    using semantic similarity. Use this tool FIRST for any compliance question.
    Returns relevant policy chunks with their exact section references.
    Input: a natural language query describing what policy information is needed.
    Output: list of matching policy chunks with doc title, section path, clause number, and text.
    Always call this tool before attempting to answer any compliance question.
    """
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed(query),
        limit=top_k,
        with_payload=True
    )
    
    if not results or results[0].score < 0.45:
        return "NO_RELEVANT_POLICY_FOUND"
    
    formatted = []
    for r in results:
        p = r.payload
        formatted.append(
            f"[SCORE: {r.score:.2f}] "
            f"Document: {p['doc_title']} | "
            f"Section: {p['section_display']} | "
            f"Clause: {p['clause_number']} | "
            f"Link: {p['doc_link']}\n"
            f"Text: {p['text']}\n"
        )
    
    return "\n---\n".join(formatted)
```

**Confidence threshold:** If top result score < 0.45 → return `"NO_RELEVANT_POLICY_FOUND"`. The agent must then call `escalate`.

---

### Tool 2: `get_section` — `rag/tools/get_section.py`

```python
def get_section(doc_id: str, clause_number: str) -> str:
    """
    Retrieve the FULL text of a specific policy section or clause by its 
    exact identifier. Use this tool when you need the complete wording of 
    a specific clause for precise citation, or when search_policies returned 
    a partial chunk and you need the full clause text.
    Input: doc_id (document slug) and clause_number (e.g. "4.2.1").
    Output: complete clause text with full section path.
    """
    results = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="clause_number", match=MatchValue(value=clause_number))
            ]
        ),
        limit=10
    )
    # return assembled full section text
```

---

### Tool 3: `ask_clarification` — `rag/tools/clarify.py`

```python
def ask_clarification(question_to_user: str) -> str:
    """
    Ask the user a clarifying question BEFORE searching policies, when the 
    original question is ambiguous and could refer to multiple different 
    policy areas. For example: "reporting something" could mean reporting 
    an incident, reporting to regulators, or reporting a colleague.
    Use sparingly — only when ambiguity would lead to wrong policy retrieval.
    Input: the clarifying question to ask the user.
    Output: a formatted clarification request (the agent should pause and 
    present this to the user before proceeding).
    """
    return f"CLARIFICATION_NEEDED: {question_to_user}"
```

---

### Tool 4: `escalate_to_compliance` — `rag/tools/escalate.py`

```python
def escalate_to_compliance(
    reason: str,
    unanswered_question: str,
    search_attempted: bool = True
) -> str:
    """
    Escalate to the Compliance team when: (1) no relevant policy was found 
    after searching, (2) policies are ambiguous or contradictory, 
    (3) the question requires legal interpretation beyond policy text, or
    (4) confidence in the retrieved answer is low.
    NEVER guess or answer from general knowledge. If unsure → escalate.
    Input: reason for escalation, the original user question, whether search was attempted.
    Output: escalation confirmation with ticket ID.
    This tool saves the full conversation context automatically.
    """
    ticket = create_escalation_ticket(
        question=unanswered_question,
        reason=reason,
        conversation_context=get_current_session_context(),
        search_attempted=search_attempted
    )
    send_escalation_email(ticket)
    return (
        f"ESCALATED: Your question has been forwarded to the Compliance team "
        f"(Ticket #{ticket.id}). They will respond within 2 business days. "
        f"Reason: {reason}"
    )
```

---

## Phase 4: The Agent — `rag/agent.py`

### System Prompt (critical — do not simplify)

```python
SYSTEM_PROMPT = """You are a Compliance Assistant for [Company Name].

RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. You ONLY answer questions using the approved internal policy documents.
2. You NEVER answer from general knowledge, legal training, or the internet.
3. EVERY answer must cite: Document Name, Section Path, Clause Number, and Link.
4. If search_policies returns NO_RELEVANT_POLICY_FOUND → you MUST call escalate_to_compliance immediately.
5. If retrieved chunks are ambiguous, incomplete, or contradictory → call escalate_to_compliance.
6. If a question touches multiple policy areas → call search_policies multiple times with different queries.
7. If the question is vague → call ask_clarification first.
8. You are NOT a lawyer. Do not provide legal interpretation. Cite policy text verbatim.

ANSWER FORMAT (always use this structure):
---
**Answer:** [direct answer based on policy text]

**Policy Sources:**
- 📄 [Document Title] | [Section Path] | Clause [X.X.X]
  > "[exact quote from policy]"
  🔗 [link to document]

**Note:** [any important caveats, e.g. "this policy was last updated [date]"]
---

ESCALATION FORMAT:
If you must escalate, say:
"I was unable to find a confirmed answer in the current approved policies. 
Your question has been forwarded to the Compliance team. [ticket info]"
"""
```

### Agent Initialization

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

def build_agent(session_id: str) -> ReActAgent:
    llm = Ollama(
        model="qwen2.5:14b",  # or llama3.3:70b for higher quality
        base_url="http://localhost:11434",
        request_timeout=120.0,
        context_window=32768,
        temperature=0.0,       # deterministic — critical for compliance
        system_prompt=SYSTEM_PROMPT
    )
    
    tools = [
        FunctionTool.from_defaults(fn=search_policies),
        FunctionTool.from_defaults(fn=get_section),
        FunctionTool.from_defaults(fn=ask_clarification),
        FunctionTool.from_defaults(fn=escalate_to_compliance),
    ]
    
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=8,       # prevent infinite loops
        context=session_id      # for conversation memory
    )
    
    return agent
```

**temperature=0.0 is mandatory.** Compliance answers must be deterministic and reproducible.

---

## Phase 5: FastAPI Backend — `api/`

### Endpoints

```
POST   /api/chat              # Send message, get agent response
GET    /api/chat/{session_id}/history   # Conversation history
POST   /api/ingest            # Upload + ingest new document (admin)
GET    /api/documents         # List all indexed documents (admin)
DELETE /api/documents/{doc_id}  # Remove document from index (admin)
GET    /api/escalations       # List escalations (admin only)
PATCH  /api/escalations/{id}  # Mark escalation resolved (admin)
GET    /api/health            # Health check
```

### Chat Request/Response Schema

```python
class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str

class Citation(BaseModel):
    doc_title: str
    section_path: str
    clause_number: str
    doc_link: str
    excerpt: str            # ≤150 chars of relevant text

class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    citations: list[Citation]
    is_escalated: bool
    escalation_ticket_id: str | None
    confidence: float       # 0.0–1.0 based on top retrieval score
    processing_time_ms: int
```

### Chat Route Implementation

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    agent = get_or_create_agent(request.session_id)
    
    start = time.time()
    response = await agent.achat(request.message)
    elapsed = int((time.time() - start) * 1000)
    
    citations = extract_citations_from_response(response.response)
    is_escalated = "ESCALATED:" in response.response
    
    save_message(request.session_id, request.message, response.response)
    
    return ChatResponse(
        session_id=request.session_id,
        answer=response.response,
        citations=citations,
        is_escalated=is_escalated,
        processing_time_ms=elapsed,
        ...
    )
```

---

## Phase 6: Database — `db/models.py`

```python
class EscalationTicket(Base):
    __tablename__ = "escalations"
    id = Column(String, primary_key=True)  # e.g. "ESC-2024-0042"
    session_id = Column(String)
    user_id = Column(String)
    original_question = Column(Text)
    reason = Column(Text)
    conversation_context = Column(JSON)    # full message history
    status = Column(String, default="open")  # open | in_review | resolved
    assigned_to = Column(String, nullable=True)
    created_at = Column(DateTime)
    resolved_at = Column(DateTime, nullable=True)
    resolution_note = Column(Text, nullable=True)

class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String)
    created_at = Column(DateTime)
    last_active = Column(DateTime)
    message_count = Column(Integer, default=0)

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)       # "user" | "assistant"
    content = Column(Text)
    citations = Column(JSON, nullable=True)
    is_escalated = Column(Boolean, default=False)
    created_at = Column(DateTime)
```

---

## Phase 7: Frontend — `frontend/`

### React Chat UI Requirements

**Component: `ChatWindow.jsx`**
- Message list with auto-scroll
- Input box + send button
- Loading indicator while agent is thinking ("Searching policies...")
- Session ID management (localStorage)

**Component: `MessageBubble.jsx`**
- User messages: right-aligned, blue
- Bot messages: left-aligned, white
- Escalated messages: orange border + escalation badge
- Always show `CitationCard` below bot messages if citations exist

**Component: `CitationCard.jsx`**
- Show: Document title, section path, clause number
- Clickable link to source document
- Expandable excerpt of quoted text
- Design: compact card, max 2 cards visible, "+N more" if >2

**Component: `EscalationPanel.jsx`** (admin only)
- Table of open escalations
- Columns: Ticket ID, User, Question, Date, Status, Assigned To
- Click row → expand full conversation context
- "Resolve" button → PATCH `/api/escalations/{id}`

---

## Phase 8: Configuration — `.env.example`

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:14b            # or llama3.3:70b
EMBEDDING_MODEL=nomic-embed-text
LLM_TEMPERATURE=0.0
LLM_REQUEST_TIMEOUT=120

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=compliance_policies

# Documents
POLICY_DOCS_FOLDER=./policies
POLICY_BASE_URL=http://intranet.company.com/policies

# Retrieval
RETRIEVAL_TOP_K=6
MIN_CONFIDENCE_SCORE=0.45         # below this → escalate

# Escalation Email
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USER=bot@company.com
SMTP_PASSWORD=
COMPLIANCE_TEAM_EMAIL=compliance@company.com
ESCALATION_TICKET_PREFIX=ESC

# API
API_SECRET_KEY=changeme
ADMIN_API_KEY=changeme

# SQLite
DATABASE_URL=sqlite:///./compliance_bot.db
```

---

## Implementation Order

Build in this exact sequence. Each step must be working before the next:

```
Step 1  → docker-compose up (Qdrant running, health check passes)
Step 2  → ollama pull qwen2.5:14b && ollama pull nomic-embed-text
Step 3  → ingest/docx_parser.py  (test with 1 sample DOCX)
Step 4  → rag/embeddings.py + rag/vector_store.py
Step 5  → scripts/ingest_all.py  (verify chunks appear in Qdrant)
Step 6  → rag/tools/search_policies.py  (test raw search, verify citations)
Step 7  → rag/tools/get_section.py + escalate.py + clarify.py
Step 8  → rag/agent.py  (test via scripts/test_query.py CLI first)
Step 9  → db/ models + migrations
Step 10 → api/ FastAPI routes
Step 11 → notification/email.py
Step 12 → frontend/ React UI
Step 13 → tests/
```

---

## Critical Requirements & Constraints

### Must Never Violate
- `temperature=0.0` on LLM at all times
- Agent must never answer without citing a retrieved chunk
- If `search_policies` returns `NO_RELEVANT_POLICY_FOUND` → next call must be `escalate_to_compliance`
- No external HTTP calls during inference (no web search, no remote APIs)
- Escalation must save the **full conversation context**, not just the last message

### Performance Targets
- Single-query response: < 30s on M4 Pro with `qwen2.5:14b`
- Ingestion of 50 documents: < 5 minutes total
- Qdrant search latency: < 200ms

### Security
- Admin endpoints require `X-Admin-Key` header
- User endpoints require `session_id` (UUID, client-generated)
- No PII stored beyond what's in the conversation
- Qdrant not exposed outside localhost

---

## Testing

### `tests/test_parser.py`
- Test that a DOCX with 3 sections produces correct `section_path` metadata
- Test that numbered clauses (4.2.1) are correctly detected
- Test that tables are converted to text and attached to parent section

### `tests/test_retrieval.py`
- Test that `search_policies("data retention personal data")` returns chunk with clause_number
- Test that score < 0.45 returns `NO_RELEVANT_POLICY_FOUND`
- Test `get_section("remote_work_policy", "3.2")` returns correct full text

### `tests/test_agent.py`
Three mandatory test cases:
1. **Happy path**: "How many days of annual leave do employees get?" → answer with citation
2. **Escalation path**: "Can we share client data with our parent company in Germany?" → escalated
3. **Multi-policy path**: "Can I work remotely from Spain for 2 months?" → cites Remote Work Policy AND Data Privacy Policy

---

## Common Pitfalls — Avoid These

| Pitfall | Fix |
|---|---|
| Chunking by token count | Always chunk by document structure (headings/clauses) |
| Single retrieval call per question | For complex questions, agent must call `search_policies` multiple times |
| Hallucinated citations | Extracted citations must only come from retrieved chunk metadata, never from LLM output |
| Session state lost between requests | Store conversation history in `ChatMessage` DB table, reload on each request |
| Large DOCX tables broken | Convert all table cells to key:value text before chunking |
| Ollama timeout on 70B model | Set `request_timeout=180`, use streaming response to frontend |
| Qdrant empty results on first run | Always call `ingest_all.py` before starting the API server |

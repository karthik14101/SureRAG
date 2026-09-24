# SURE-GraphRAG — Architecture

How the system is put together, what each part does, and why it was built that
way. Read top to bottom: the three diagrams go from the whole system, to how a
document gets in, to what happens when a question is asked.

- [1. System overview](#1-system-overview)
- [2. Ingestion — how a document becomes searchable](#2-ingestion--how-a-document-becomes-searchable)
- [3. Answering — the SURE loop](#3-answering--the-sure-loop)
- [4. Where data lives](#4-where-data-lives)
- [5. Design decisions worth knowing](#5-design-decisions-worth-knowing)

---

## 1. System overview

Everything runs on one machine. Embeddings are computed locally and cost
nothing; the only outbound network call is answer generation, and even that is
optional if you point it at a local model.

```
   ┌───────────────────────────────────────────────────────────────┐
   │  BROWSER                                                      │
   │  React 19 · Vite · Tailwind v4                                │
   │                                                               │
   │   Chat  ·  Reasoning Trail  ·  Chunk Explorer  ·  Graph tab   │
   └───────────────────────────┬───────────────────────────────────┘
                               │  REST + Server-Sent Events
                               │  (bearer token in the header)
   ┌───────────────────────────▼───────────────────────────────────┐
   │  BACKEND — FastAPI (Python 3.12, async)                       │
   │                                                               │
   │   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   │
   │   │ API routes  │   │ SURE agent   │   │ Ingestion worker │   │
   │   │ auth, kb,   │──▶│ orchestrator │   │ async queue,     │   │
   │   │ chat, graph │   │ (the loop)   │   │ N background     │   │
   │   │ chunks, job │   │              │   │ tasks            │   │
   │   └─────────────┘   └──────┬───────┘   └────────┬─────────┘   │
   │                            │                    │             │
   │   ┌────────────────────────▼────────────────────▼─────────┐   │
   │   │ Local models (CPU, no API key, no network)            │   │
   │   │   MiniLM-L6-v2  384-d dense embeddings                │   │
   │   │   BM25 (fastembed)  sparse keyword vectors            │   │
   │   │   ms-marco-MiniLM cross-encoder  reranking            │   │
   │   └───────────────────────────────────────────────────────┘   │
   └───┬───────────────┬───────────────┬───────────────┬───────────┘
       │               │               │               │
   ┌───▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────────┐
   │ SQLite  │   │  Qdrant   │   │   Neo4j   │   │ LLM provider  │
   │ (WAL)   │   │  :6333    │   │   :7687   │   │ ONE of:       │
   │         │   │           │   │           │   │  Azure OpenAI │
   │ users   │   │ dense +   │   │ Entity    │   │  Gemini       │
   │ KBs     │   │ sparse    │   │ Chunk     │   │  Groq         │
   │ docs    │   │ vectors,  │   │ Document  │   │  Ollama ⌂     │
   │ chats   │   │ payload   │   │ RELATED   │   │  HuggingFace ⌂│
   │ jobs    │   │ filters   │   │ MENTIONED │   │               │
   └─────────┘   └───────────┘   └───────────┘   └───────────────┘
        local         docker          docker       ⌂ = fully local
```

**Points**

1. **The browser talks to exactly one backend.** No direct database access from
   the client, so every read is tenant-filtered server-side by `user_id` and
   `kb_id` before it leaves the process.
2. **Answers stream over Server-Sent Events**, not a polled endpoint. The same
   channel carries reasoning-trail steps as each stage completes, which is what
   lets the UI show *what* the engine is doing rather than only *that* it is
   working.
3. **Three local models run on CPU** and are warmed at startup so the first
   question does not pay for a cold load. None of them calls out.
4. **Four stores, each with one job.** SQLite holds relational truth, Qdrant
   holds vectors, Neo4j holds relationships, and the filesystem holds extracted
   figures. Nothing is duplicated between them except the chunk id, which is
   the join key everywhere.
5. **The LLM provider is swappable by one environment variable.** Two of the
   five options run entirely on your machine, so the system can work with no
   internet at all.
6. **Ingestion never blocks a request.** Uploads return `202 Accepted` and a
   job id; a background worker does the parsing, embedding and graph building.

---

## 2. Ingestion — how a document becomes searchable

```
  UPLOAD                                              (HTTP 202 + job id)
    │
    ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ STAGING — refuse bad input before it costs anything              │
 │                                                                  │
 │   read in 1 MB chunks, abort past the size limit                 │
 │   extension allowlist          .pdf .docx .txt .md .json .zip …  │
 │   magic-byte check             content must match the extension  │
 │   path sanitising              Path(name).name — no traversal    │
 │   SHA-256                      skip a file the KB already has    │
 │   ZIP limits                   depth, entry count, uncompressed  │
 └────────────────────────────┬─────────────────────────────────────┘
                              │ staged to disk, queued
                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ WORKER  (async queue, runs in the background)                    │
 │                                                                  │
 │  ① PARSE ─── per-format parser                                   │
 │        PDF → text + figures + captions (PyMuPDF)                 │
 │        DOCX, TXT/MD, JSON, images (OCR)                          │
 │                                                                  │
 │  ② CHUNK ── split on the largest natural boundary that fits:     │
 │        paragraph → sentence → word, with overlap.                │
 │        Never merges across a page or section boundary.           │
 │                                                                  │
 │  ③ EMBED ── every chunk gets TWO vectors                         │
 │        dense  384-d MiniLM   (meaning)                           │
 │        sparse BM25           (exact words: codes, part numbers)  │
 │                    │                                             │
 │                    └──────────────────▶ Qdrant (one point each)  │
 │                                                                  │
 │  ④ GRAPH ── entity + relationship extraction                     │
 │        selective by default: only entity-dense chunks get an     │
 │        LLM call, which is ~60% cheaper than doing every chunk    │
 │                    │                                             │
 │                    └──────────────────▶ Neo4j                    │
 │                                                                  │
 │  ⑤ RECORD ─ rows, counts and figures                             │
 │                    │                                             │
 │                    ├──────────────────▶ SQLite                   │
 │                    └──────────────────▶ data/media/              │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 progress streamed to the browser
                 (stage, current file, per-file errors)
```

**Points**

1. **Validation happens before work does.** Extension, magic bytes, size and
   hash are all checked while the file is still being read, so a 2 GB upload or
   a `.exe` renamed to `.pdf` is rejected without ever being parsed.
2. **A file already in the knowledge base is skipped by SHA-256**, not by name.
   Re-uploading the same document twice costs nothing.
3. **Chunk boundaries decide retrieval quality more than model choice.** A fact
   split across two chunks is usually unfindable, so the chunker splits on the
   largest natural boundary that fits and overlaps neighbours.
4. **Two vectors per chunk, not one.** Dense embeddings miss exact strings —
   part numbers, article numbers, error codes — because they encode meaning,
   not spelling. BM25 catches precisely those. Both live on the same Qdrant
   point so a single query can use both.
5. **Graph extraction is selective by default.** Running an LLM over every
   chunk is the expensive way to build a graph; only entity-dense chunks earn
   the call. Set `GRAPH_EXTRACTION=on` for a richer graph, or `off` to skip it.
6. **Failure is per-file, not per-upload.** One corrupt PDF in a ZIP of fifty
   marks that file failed and reports why; the rest still ingest.

---

## 3. Answering — the SURE loop

This is the part that differs most from a typical RAG pipeline. An ordinary
system retrieves, then generates. This one interposes a judgement between the
two, and is willing to go back.

```
  QUESTION
     │
     ▼
 ┌──────────┐   Is it standalone? "what about the second one?" is not.
 │ CONDENSE │   Rewrites follow-ups against the conversation.
 │    +     │   Skipped entirely — no model call — when already standalone,
 │  ROUTE   │   and merged into ONE call with the route when both are needed.
 └────┬─────┘   When no rewrite is needed the vector search starts here.
      ▼
 ┌──────────┐   DIRECT ── chit-chat / opinion → answer, no retrieval ──┐
 │  ROUTE   │   VECTOR ── one passage probably holds the answer        │
 │ heuristic│   GRAPH  ── "how does X relate to Y"                     │
 │ first,   │   HYBRID ── needs both text and relationships            │
 │ model if │   MULTIHOP─ decompose into sub-questions first           │
 │ unsure   │                                                          │
 └────┬─────┘                                                          │
      ▼                                                                │
 ┌───────────────────────────────────────────────────┐                 │
 │ RETRIEVE                                          │                 │
 │   dense + BM25, fused server-side with Reciprocal │                 │
 │   Rank Fusion — ONE Qdrant round trip             │                 │
 │   GRAPH route ALSO runs the vector search and     │                 │
 │   merges: the graph enhances, it cannot veto      │                 │
 └────┬──────────────────────────────────────────────┘                 │
      ▼                                                                │
 ┌───────────────────────────────────────────────────┐                 │
 │ RERANK — local cross-encoder reads question and   │                 │
 │ passage together. BLENDED 60/40 with retrieval    │                 │
 │ order, never allowed to decide alone.             │                 │
 │ Reorders the pool; it never deletes from it.      │                 │
 └────┬──────────────────────────────────────────────┘                 │
      ▼                                                                │
 ┌═══════════════════════════════════════════════════┐                 │
 ║ VERIFY  ← the part most RAG systems do not have   ║                 │
 ║                                                   ║                 │
 ║ Reads 12 passages AND a description of what this  ║                 │
 ║ corpus actually contains, then answers:           ║                 │
 ║   sufficient?   score 0–1                         ║                 │
 ║   in_scope?     is the gap even findable here?    ║                 │
 ║   missing[]     what specifically is absent       ║                 │
 ╚════┬══════════════════════════════════╤═══════════╝                 │
      │ sufficient                       │ not sufficient              │
      │                                  ▼                             │
      │                     ┌─────────────────────────┐                │
      │                     │ EXPAND                  │                │
      │                     │  · targeted queries in  │                │
      │                     │    document vocabulary  │                │
      │                     │  · HyDE probe — embed a │                │
      │                     │    hypothetical ANSWER  │                │
      │                     │  · widen the graph one  │                │
      │                     │    hop                  │                │
      │                     └───────────┬─────────────┘                │
      │                                 │                              │
      │       ┌─────────────────────────┘                              │
      │       │  STOP if: the score did not rise ≥5 points,            │
      │       │           or the corpus plainly lacks this,            │
      │       │           or a budget is spent                         │
      │       └──▶ back to RERANK ──▶ VERIFY   (max 2 passes)          │
      ▼                                                                │
 ┌───────────────────────────────────────────────────┐                 │
 │ SYNTHESIZE ── main model, streamed token by token │◀────────────────┘
 │   reads the top 10 passages + graph facts         │
 │   thin evidence → told to say so, not to pad      │
 │   out of scope  → capped at 150 honest words      │
 └────┬──────────────────────────────────────────────┘
      ▼
 ┌───────────────────────────────────────────────────┐
 │ CITE ── every [n] validated against the source    │
 │ list, invented ones stripped, survivors renumbered│
 │ densely so the reader sees 1,2,3                  │
 └────┬──────────────────────────────────────────────┘
      ▼
   ANSWER  +  clickable citations  +  figures  +  reasoning trail
```

**Points**

1. **Routing decides the strategy, not the user.** Five routes, and the cheapest
   one — `DIRECT` — skips retrieval entirely for conversational turns, so
   "thanks, that helps" does not trigger a database search. A heuristic gate
   settles the obvious cases for free; only an ambiguous question costs a model
   call, and when the question also needs rewriting both judgements are made in
   the same call rather than two round trips.
2. **Retrieval is one round trip, not two.** Dense and sparse run as prefetch
   branches inside a single Qdrant query and are fused server-side with
   Reciprocal Rank Fusion, so there is no client-side merging of two
   incompatible score scales. The two query encoders run concurrently, and
   where a step issues several searches at once — one per sub-question on the
   multi-hop route, one per probe on an expansion pass — they go out together
   rather than in series, capped so a wide turn cannot thrash the thread pool.
3. **The graph enhances retrieval; it cannot override it.** The `GRAPH` route
   also runs the ordinary vector search and merges both. This matters: an
   entity the extractor never created cannot be traversed to, and a graph-only
   pool would silently answer from whatever it did find.
4. **Reranking blends rather than replaces.** The cross-encoder is trained on
   web passages and is measurably weaker on structured records, so retrieval
   order is kept as a prior at 40%. It reorders the pool and never truncates
   it — a cut here would discard evidence between expansion passes.
5. **Verification is a separate model call that never sees the answer.**
   Separating judging from answering is what stops a model rationalising the
   context it already has.
6. **The verifier is told what the corpus contains**, measured from a sample of
   the actual passages: the fields, if they are structured records; otherwise
   how many files, what page range, and whether the passages are running text,
   flattened tables or figure captions. Without that it cannot tell "we missed
   it" from "this was never ingested" — and will loop forever chasing the
   second one. It is also what licenses the out-of-scope verdict: the engine
   will not declare material absent from a corpus it never managed to describe.
7. **The loop stops on measured progress, not on what the model says.** If a
   full expansion pass did not raise the score, the next one will not either.
   Softer rules were tried and failed; the history is in `backend/tests/`.
8. **Four independent budgets** bound every turn: 2 expansion passes, 6
   reasoning-model calls, 12 fast-tier calls, 45 seconds. Whichever trips
   first, the engine answers with what it has rather than hanging.
9. **Citations are validated, not trusted.** Models cite `[7]` when five
   sources exist. Every marker is checked against the real source list and
   invented ones are removed from the text.

---

## 4. Where data lives

| Store | Holds | Why there |
|---|---|---|
| **SQLite** (`data/sure.db`) | users, sessions, knowledge bases, documents, chat messages, citations, jobs | Relational truth with foreign keys and cascades. WAL mode so the background worker can write while requests read |
| **Qdrant** (`:6333`) | one point per chunk: 384-d dense vector, BM25 sparse vector, and the payload used for filtering | Purpose-built for hybrid search and server-side fusion |
| **Neo4j** (`:7687`) | `Entity`, `Chunk`, `Document` nodes; `RELATED` and `MENTIONED_IN` edges | Traversal is a graph problem. Answering "which suppliers are affected by the Q3 recall" means following edges, not matching passages |
| **`data/media/`** | extracted figures and thumbnails, UUID-named | Binary blobs do not belong in a row or a vector |

The **chunk id is the join key**. A Qdrant point, a Neo4j `Chunk` node and a
SQLite citation row all carry the same id, which is how a `[1]` in an answer
opens the original passage in its surrounding context.

---

## 5. Design decisions worth knowing

**Why an explicit loop instead of a graph framework.** The shape is fixed and
small: one cycle, a node set that never changes at runtime, and every
transition a plain `if`. Written out, the whole control flow fits on one screen
and the step trace is trivial to emit for the UI.

**Why two confidence numbers.** *Evidence coverage* is the verifier's verdict on
the retrieved pool, formed before a word is written. *Answer grounding* is the
share of the written answer's claims that carry a citation. They are allowed to
disagree, and the disagreement is informative: on a question whose answer must
be assembled from several provisions, coverage reports partial because no
single passage states the conclusion, while the assembled answer is fully
cited.

**Why a fast tier.** Routing, condensing, decomposition and verification are
short structured calls that do not need a reasoning model. On Azure they run on
the *same* deployment at minimal reasoning effort, so no extra quota is needed —
which matters because reasoning tokens are billed as output, and a router call
that deliberates costs as much as a paragraph of answer. Leave the fast
settings empty and every provider behaves exactly as before.

**Why the reasoning trail is visible.** About two thirds of a turn happens
before the first token exists. Showing the retrieval, the score, the gaps and
the stop reason turns that wait into something legible, and makes the engine
auditable rather than magic.

**What this architecture is not.** Single-node and single-process: the worker
queue is in memory, so scaling out means moving it to a real queue. It is built
for one person's documents on one machine, and the design leans on that
throughout.

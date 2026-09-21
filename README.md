# SURE-GraphRAG

**Adaptive Hybrid Graph-Agentic Retrieval-Augmented Generation Engine**

Ask questions about your own documents and get answers that cite the exact passage
they came from — with the figures from those pages shown alongside, and a visible
record of how the evidence was gathered.

Everything runs on your machine. Embeddings are local and free; only the final
answer generation calls an LLM API, and you choose which one.

---

## What makes it different

| | |
|---|---|
| **Hybrid retrieval** | Dense semantic vectors **and** BM25 keyword matching, fused with Reciprocal Rank Fusion inside Qdrant — one round trip. Dense search alone misses part numbers and error codes; BM25 catches them. |
| **Knowledge graph** | Entities and relationships extracted into Neo4j, so questions like *"which suppliers are affected by the Q3 recall?"* are answered by following edges, not by hoping one passage says it. |
| **SURE verifier** | Before writing an answer, the engine judges whether the retrieved evidence is actually **sufficient**. If not, it names the gaps and runs a targeted expansion pass. This is what stops confident answers from thin context. |
| **Verifiable citations** | Every claim carries a `[n]` marker you can click to read the original passage in its surrounding context, with a relevance score. Hallucinated citations are stripped automatically. |
| **Multimodal** | PDF and Word figures are extracted, paired with their captions, OCR'd, and surfaced next to the answer that references them. |

---

## Table of contents

- [Step 0 — Install the prerequisites](#step-0--install-the-prerequisites)
- [Step 1 — Get an LLM API key](#step-1--get-an-llm-api-key-free)
- [Step 2 — Configure](#step-2--configure)
- [Step 3 — Start the databases](#step-3--start-the-databases)
- [Step 4 — Start the backend](#step-4--start-the-backend)
- [Step 5 — Start the frontend](#step-5--start-the-frontend)
- [Step 6 — Validate everything](#step-6--validate-everything)
- [Daily use](#daily-use-after-the-first-setup)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)

---

# Step 0 — Install the prerequisites

You need four things. If you already have one, skip that section.

Open **PowerShell** (press `Win`, type `powershell`, press Enter) and run the
checks below. Anything that prints a version number is already installed.

```powershell
docker --version
python --version
node --version
```

---

### 0.1 Docker Desktop (runs the two databases)

Docker lets you run Qdrant and Neo4j without installing either one directly.

1. Download **Docker Desktop for Windows** from
   <https://www.docker.com/products/docker-desktop/> and run the installer.
2. When it asks, leave **"Use WSL 2 instead of Hyper-V"** ticked. This is the
   recommended backend and the installer sets up WSL 2 for you.
3. **Restart your computer** when prompted. This is not optional — Docker will not
   work until you do.
4. After restarting, launch **Docker Desktop** from the Start menu and wait until
   the whale icon in your system tray stops animating. The dashboard should say
   **"Engine running"**.
5. Verify in PowerShell:

   ```powershell
   docker --version
   docker run --rm hello-world
   ```

   The second command downloads a tiny test image and prints
   *"Hello from Docker!"*. If you see that, Docker works.

> **Docker Desktop must be running** whenever you use SURE-GraphRAG. It does not
> start automatically unless you enable that in its settings
> (*Settings → General → Start Docker Desktop when you sign in*).

---

### 0.2 Python 3.11 or 3.12

1. Download from <https://www.python.org/downloads/>. **Python 3.12 is
   recommended.** Avoid 3.13 for now — some of the ML libraries do not yet ship
   prebuilt wheels for it, and installation will try to compile from source.
2. Run the installer. **Tick "Add python.exe to PATH"** on the very first screen —
   this is the single most commonly missed step.
3. Verify:

   ```powershell
   python --version
   ```

   You should see `Python 3.12.x`. If PowerShell opens the Microsoft Store
   instead, PATH was not set: re-run the installer and choose *Modify → Add
   Python to environment variables*.

---

### 0.3 Node.js 18+ (runs the web interface)

1. Download the **LTS** version from <https://nodejs.org/>.
2. Run the installer and accept the defaults.
3. Verify:

   ```powershell
   node --version
   npm --version
   ```

---

### 0.4 Tesseract OCR — *optional*

Only needed to read text out of **image files and scanned PDFs**. Everything else
works without it, and the app detects its absence and carries on.

1. Download the Windows installer from
   <https://github.com/UB-Mannheim/tesseract/wiki>.
2. Run it and **note the install path** (usually
   `C:\Program Files\Tesseract-OCR`).
3. Either tick *"Add to PATH"* during installation, or set this line in your
   `.env` later:

   ```ini
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

4. Verify:

   ```powershell
   tesseract --version
   ```

> Without Tesseract: with `LLM_PROVIDER=gemini` (or `azure` with a vision-capable
> deployment such as GPT-4o) images are still described by a vision model. With `LLM_PROVIDER=groq` (no vision model available),
> images are indexed by filename and metadata only.

---

# Step 1 — Get an LLM API key (free)

Pick **one**. Gemini and Groq have free tiers that are plenty for this; Azure uses
your organisation's own deployment.

### Option A — Google AI Studio (recommended: it can also read images)

1. Go to <https://aistudio.google.com/app/apikey>.
2. Sign in with a Google account and click **Create API key**.
3. Copy the key — it starts with `AIza...`.

### Option B — GroqCloud (extremely fast)

1. Go to <https://console.groq.com/keys>.
2. Sign up and click **Create API Key**.
3. Copy the key — it starts with `gsk_...`.

### Option C — Azure OpenAI / Azure AI Foundry (your own deployment)

You need **three values** from the [Foundry portal](https://ai.azure.com), all on
your deployment's page:

1. **Endpoint** — paste it exactly as shown. A bare endpoint
   (`https://your-resource.openai.azure.com`) and the full **Target URI**
   (`…/openai/deployments/<name>/chat/completions?api-version=…`) both work: the
   app keeps the host and reads the deployment name out of the URL for you.
2. **Key** — *Key 1* from the resource's **Keys and Endpoint** page.
3. **Deployment name** — the name *you chose* when deploying, shown in the
   **Deployments** tab. This is **not always the model name**, and it is the
   single most common Azure setup mistake.

Any chat deployment works — GPT-4o, GPT-4.1, GPT-5, the o-series, or a Foundry
model such as DeepSeek-R1. Reasoning models (GPT-5, o1/o3/o4) reject some request
parameters that chat models accept; the app detects this from Azure's first
response and adapts automatically, so you do not need to know which kind you
deployed.

### Option D — Ollama (fully local, no key, no internet after the download)

1. Install Ollama from <https://ollama.com>. It runs as a background service.
2. Pull a model in a terminal: `ollama pull qwen2.5:7b` (about 5 GB). On a
   CPU-only laptop, `llama3.2:3b` or `qwen2.5:3b` are much faster.
3. Optional, to caption images: `ollama pull gemma3:4b`, then set
   `OLLAMA_VISION_MODEL=gemma3:4b`.

### Option E — HuggingFace transformers (fully local, in-process)

Set `HF_MODEL` to any chat model repo id from <https://huggingface.co>, or to a
local folder. The weights download once into `.hf_cache` when the backend starts.
There is no server to run and nothing extra to install. For gated models such as
Llama or Gemma, accept the licence on the model page and put a read token in
`HF_TOKEN`. Keep to 0.5B–3B models on CPU; 7B and larger need a GPU.

> **Local models are slower.** For Ollama and HuggingFace the app automatically
> uses the longer `LOCAL_LLM_TIMEOUT_SECONDS` and `LOCAL_AGENT_TIMEOUT_SECONDS`.
> Setting `GRAPH_EXTRACTION=off` makes ingestion much faster, because it skips
> the LLM call for each chunk.

> **Model names change.** If you later get a *"model not found"* error, open the
> provider's console, copy the current model id, and update `GEMINI_MODEL` or
> `GROQ_MODEL` in your `.env`. It is a one-line change.

---

# Step 2 — Configure

In PowerShell, go to the project folder and create your `.env` from the template:

```powershell
cd C:\Users\KARTHIK\SURERag
Copy-Item .env.example .env
notepad .env
```

Change **one or two lines**, save, and close Notepad:

**If you chose Google AI Studio:**
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...your-key-here...
```

**If you chose Groq:**
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...your-key-here...
```

**If you chose Azure:**
```ini
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...your-key-here...
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

**If you chose Ollama:**
```ini
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

**If you chose HuggingFace:**
```ini
LLM_PROVIDER=huggingface
HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

> Azure uses the `openai` Python package. If you installed the backend before
> Azure support was added, run `pip install -r requirements.txt` once more inside
> your activated virtual environment.

Everything else already has a working default.

---

# Step 3 — Start the databases

Make sure **Docker Desktop is running** first, then:

```powershell
cd C:\Users\KARTHIK\SURERag
docker compose up -d
```

The first run downloads about 1 GB and takes a few minutes. Afterwards it starts
in seconds.

Check both containers are healthy:

```powershell
docker compose ps
```

You want to see `sure_qdrant` and `sure_neo4j` both `running`. Neo4j takes
around 30–45 seconds to report healthy on a cold start — that is normal.

You can also open their dashboards in a browser:
- Qdrant → <http://localhost:6333/dashboard>
- Neo4j → <http://localhost:7474> (user `neo4j`, password `sureGraph2024`)

---

# Step 4 — Start the backend

**Use a new PowerShell window** and leave it open — this runs the server.

```powershell
cd C:\Users\KARTHIK\SURERag\backend

# Create an isolated Python environment (once)
python -m venv .venv

# Activate it (every time you open a new terminal)
.\.venv\Scripts\Activate.ps1

# Install dependencies (once, takes 3-10 minutes)
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run the server
python run.py
```

### If activation is blocked

PowerShell may refuse the activation script with *"running scripts is disabled on
this system"*. Allow it for your own account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y`, then run `.\.venv\Scripts\Activate.ps1` again. Your prompt will show
`(.venv)` when it worked.

### What you should see

```
  ___ _   _ ___ ___     ___               _    ___   _   ___
 / __| | | | _ \ __|__ / __|_ _ __ _ _ __| |_ | _ \ /_\ / __|
 ...
INFO     Starting SURE-GraphRAG
INFO       provider           gemini
INFO       model              gemini-3-flash
INFO     SQLite schema ready at ...\data\sure.db
INFO     Qdrant collection 'sure_chunks' ready
INFO     Neo4j ready at bolt://localhost:7687
INFO     Ingestion worker started (2 parallel slot(s))
INFO     Ready on http://127.0.0.1:8000  (docs at /docs)
```

> **Note on addresses:** the server binds to `127.0.0.1` (this machine only).
> If you ever set `BACKEND_HOST=0.0.0.0` to reach it from your phone or another
> computer, remember that `0.0.0.0` is a *listen on everything* wildcard, not a
> destination — you still browse to `http://127.0.0.1:8000` locally.

> **First run downloads the embedding model** (about 90 MB from HuggingFace).
> The log may pause for a minute. This is a download, not a hang, and it happens
> only once — the model is cached in `.hf_cache/` inside the project.

---

# Step 5 — Start the frontend

**Open a third PowerShell window** and leave it open too.

```powershell
cd C:\Users\KARTHIK\SURERag\frontend

# Install dependencies (once, takes 1-2 minutes)
npm install

# Run the dev server
npm run dev
```

You should see:

```
  VITE v6.x.x  ready in 512 ms
  ➜  Local:   http://localhost:5173/
```

**Open <http://localhost:5173> in your browser.**

---

# Step 6 — Validate everything

### 6.1 Automated check

With all three windows running, open a **fourth** PowerShell window:

```powershell
# Databases up?
docker compose ps

# Backend alive?
curl http://127.0.0.1:8000/health

# Full stack check - this is the important one
curl http://127.0.0.1:8000/api/v1/health/deep
```

The deep check returns a JSON report on every component. You want
`"ok":true` and each of `sqlite`, `qdrant`, `embeddings` and `llm` showing
`"ok":true`.

For readable output:

```powershell
(curl http://127.0.0.1:8000/api/v1/health/deep).Content | ConvertFrom-Json | ConvertTo-Json -Depth 4
```

> `neo4j` and `ocr` are allowed to be `false` — the app works without them, with
> graph queries falling back to vector search.

### 6.2 Check it from the interface

Inside the app, click the **activity icon** (top right) at any time to see the
same status report without leaving the browser.

### 6.3 End-to-end test

1. Go to <http://localhost:5173> and **create an account** (any username, password
   at least 8 characters). It is stored locally in SQLite.
2. Click **New knowledge base**, name it `Test`, and create it.
3. **Drag a PDF onto the dropzone.** Watch the progress bar: parsing → embedding →
   graphing → ready.
4. Click **Chat** and ask something the document actually answers.
5. Confirm you see:
   - the answer streaming in, token by token
   - superscript `[1]` markers inside the text — **click one** to open the source
     passage in its original context
   - source chips beneath the answer
   - any relevant figures from the document
   - **"How this answer was built"** — expand it to see the route taken, the
     evidence-sufficiency score, and whether an expansion pass was needed

If all five happen, the whole stack is working.

---

# Daily use (after the first setup)

You need three things running. Start Docker Desktop first, then:

```powershell
# Terminal 1 - databases
cd C:\Users\KARTHIK\SURERag
docker compose up -d

# Terminal 2 - backend
cd C:\Users\KARTHIK\SURERag\backend
.\.venv\Scripts\Activate.ps1
python run.py

# Terminal 3 - frontend
cd C:\Users\KARTHIK\SURERag\frontend
npm run dev
```

To stop: press `Ctrl+C` in terminals 2 and 3, then `docker compose down`.
Your data is preserved. (`docker compose down -v` would erase the vector and
graph data — only use that when you deliberately want a clean slate.)

---

# Troubleshooting

### "Cannot reach the backend"
The browser cannot see FastAPI. Check terminal 2 is still running and shows
*"Ready on http://127.0.0.1:8000"*. If it exited, scroll up for the error.

### "Cannot reach Qdrant" / search does not work
Docker Desktop is not running, or the containers are stopped:
```powershell
docker compose ps
docker compose up -d
docker compose logs qdrant
```

### Neo4j shows unhealthy or "graph features disabled"
Neo4j needs ~45 seconds to start. If it persists:
```powershell
docker compose logs neo4j
docker compose restart neo4j
```
The app works without it — you simply lose relationship-style answers.

### "The Gemini/Groq API key was rejected"
The key in `.env` is wrong or has a stray space. Re-copy it from the provider
console. **Restart the backend after any `.env` change** — it is read once at
startup.

### "Model 'X' is not available"
The provider retired that model id. Open the provider's console, copy the current
id, update `GEMINI_MODEL` or `GROQ_MODEL` in `.env`, and restart the backend.

### "Rate limiting this key"
Free tiers allow roughly 10–30 requests per minute. Wait a moment, or switch
`LLM_PROVIDER` to the other provider. Set `GRAPH_EXTRACTION=off` to cut LLM usage
during ingestion dramatically.

### Azure: "deployment was not found at this endpoint"
`AZURE_OPENAI_DEPLOYMENT` must be the **deployment** name from Foundry's
Deployments tab, not the model name. A deployment created in the last few minutes
can also still be provisioning — wait and retry.

### Azure: "rejected the credentials"
On Azure a 401 usually means the key and endpoint belong to **different
resources**. Copy both from the same resource's *Keys and Endpoint* page.

### Azure: "content filter blocked this request"
Azure's Responsible AI filter rejected the prompt or the answer. Rephrase, or
review the content filter policy attached to the deployment in Foundry.

### Azure: an older resource returns 404 on every call
The app defaults to Azure's v1 API. For a resource that predates v1 support, set
a dated version instead: `AZURE_OPENAI_API_VERSION=2025-04-01-preview`.

### `pip install` fails building a package
Almost always Python 3.13. Install Python 3.12, delete the `.venv` folder, and
repeat Step 4.

### "running scripts is disabled on this system"
See [the activation note in Step 4](#if-activation-is-blocked).

### Uploads stay at "Queued" forever
The backend restarted mid-ingestion. Those documents are marked failed on the
next start — delete them in the knowledge base view and re-upload.

### "EMBEDDING MODEL MISMATCH — refusing to start"
You changed `EMBEDDING_MODEL` after indexing documents. Old vectors are not
comparable to new ones, so search would silently return wrong results. Either
restore the old value in `.env`, or wipe and re-index:
```powershell
docker compose down -v
Remove-Item .\data\sure.db*
docker compose up -d
```

### Port already in use
Something else holds 8000, 5173, 6333 or 7687. Find and stop it:
```powershell
netstat -ano | findstr :8000
```
Or change `BACKEND_PORT` in `.env`.

---

# How it works

### The retrieval loop

```
condense ─► route ─┬─► vector ────┐
                   ├─► graph  ────┼─► rerank ─► ★ VERIFY ─┬─► synthesize ─► answer
                   ├─► hybrid ────┤                       │
                   └─► multihop ──┘                       │
                            ▲                             │ insufficient
                            └────────── expand ◄──────────┘   (max 2 passes)
```

1. **Condense** — rewrites follow-ups into standalone questions, so *"what about
   the second one?"* becomes something a vector index can actually match.
2. **Route** — a free heuristic settles obvious cases; only genuinely ambiguous
   questions cost a classifier call.
   - `VECTOR` — a fact stated in one passage
   - `GRAPH` — how named things relate
   - `HYBRID` — both (the safe default)
   - `MULTIHOP` — decomposed into sub-questions first
   - `DIRECT` — conversational, no retrieval at all
3. **Retrieve** — dense + BM25 fused by RRF, and/or entity traversal in Neo4j.
4. **Rerank** *(optional)* — a local cross-encoder reorders the shortlist.
5. **★ Verify** — the SURE gate. Scores whether the evidence can support an
   answer and lists exactly what is missing.
6. **Expand** — if insufficient: targeted queries from the gap list, a HyDE probe
   (embedding a hypothetical *answer*, since documents are written as statements
   rather than questions), and a wider graph hop. Then it re-verifies.
7. **Synthesize** — writes the answer with `[n]` markers, validates every
   citation against the real source list, and attaches the figures from cited
   passages.

Three hard budgets bound every turn — iterations, LLM calls, and wall-clock time —
so no question can loop or drain your quota.

### Ingestion

| Format | Text | Images |
|---|---|---|
| **PDF** | Reading-order blocks, repeated headers/footers stripped | Embedded figures extracted, paired with the nearest caption, blank panels discarded |
| **DOCX** | Headings, paragraphs, tables as markdown | Inline images in document order |
| **TXT / MD** | Heading-aware sections | — |
| **JSON** | Flattened to `path = value`, one block per record | — |
| **Images** | OCR + vision caption + EXIF | The file itself, plus a thumbnail |
| **ZIP** | Recursively unpacked | Recursively unpacked |

Archives are hardened against zip-slip, zip bombs, symlink entries and entry
floods. Files are deduplicated per knowledge base by SHA-256.

### Data isolation

Every row carries `user_id`, and every query filters on it at the dependency
layer. Images are served through an ownership-checked endpoint rather than a
static mount, so one user's extracted figures are never reachable by URL.

### Storage layout

| Store | Holds |
|---|---|
| **SQLite** (`data/sure.db`) | Users, sessions, knowledge bases, documents, chunk text, citations, chat history |
| **Qdrant** (`:6333`) | Dense (384-dim) + sparse BM25 vectors, one collection, tenant-isolated by payload |
| **Neo4j** (`:7687`) | `Entity`, `Chunk`, `Document` nodes and `RELATED` / `MENTIONED_IN` edges |
| **`data/media/`** | Extracted figures and thumbnails, UUID-named |

---

# Configuration reference

Every setting lives in `.env` and is documented inline there. The ones worth
knowing:

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `azure`, `ollama` or `huggingface` — switches the whole app |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any model you have pulled (`ollama list`) |
| `OLLAMA_NUM_CTX` | `8192` | Context window; Ollama's own default would truncate RAG prompts |
| `HF_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Any HuggingFace chat model repo id or local folder |
| `HF_DEVICE` | `auto` | `auto` (cuda → mps → cpu), `cpu`, `cuda` |
| `AZURE_OPENAI_REASONING` | `auto` | `auto` detects reasoning deployments and adapts; `true`/`false` forces it |
| `AZURE_OPENAI_REASONING_EFFORT` | `low` | Thinking budget for reasoning deployments: `minimal`/`low`/`medium`/`high` |
| `SURE_THRESHOLD` | `0.65` | Sufficiency score below which expansion triggers. Raise for stricter answers, lower for faster ones |
| `AGENT_MAX_ITERATIONS` | `2` | Expansion passes allowed per question |
| `AGENT_TIMEOUT_SECONDS` | `45` | Hard wall-clock cap per question |
| `GRAPH_EXTRACTION` | `selective` | `on` (richest graph) / `selective` (~60% cheaper) / `off` (no LLM calls during ingestion) |
| `RERANK_ENABLED` | `false` | Local cross-encoder reranking. Biggest single precision win; adds ~0.3s per query and an 80 MB download |
| `RETRIEVAL_TOP_K` | `12` | Passages retrieved per query |
| `CHUNK_SIZE_TOKENS` | `800` | Chunk size. Smaller = more precise citations, larger = more context per chunk |
| `INGEST_CONCURRENCY` | `2` | Files processed in parallel |
| `OCR_ENABLED` | `true` | Auto-disables if Tesseract is not installed |

**Restart the backend after changing `.env`.**

---

## Project layout

```
SURERag/
├── docker-compose.yml          Qdrant + Neo4j
├── .env                        your configuration (created in Step 2)
├── backend/
│   ├── run.py                  start here
│   └── app/
│       ├── main.py             FastAPI app + startup sequence
│       ├── agent/              the SURE loop (orchestrator + 9 nodes)
│       ├── ingestion/          parsers, chunker, pipeline, worker
│       ├── vectorstore/        Qdrant hybrid search
│       ├── graph/              Neo4j extraction + traversal
│       ├── llm/                Gemini / Groq providers
│       ├── embeddings/         local MiniLM + BM25
│       ├── api/routes/         HTTP endpoints
│       ├── services/           business logic
│       └── db/                 SQLAlchemy models
├── frontend/
│   └── src/
│       ├── pages/              dashboard, knowledge base, chat
│       ├── components/         UI, chat, citations, media, agent trail
│       ├── store/              zustand state
│       └── api/                fetch client + SSE parsing
└── data/                       SQLite database and extracted media
```

API documentation is generated automatically at <http://127.0.0.1:8000/docs>.

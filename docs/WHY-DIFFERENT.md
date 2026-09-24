# Why SURE-GraphRAG is different

Most RAG systems are the same four steps: embed the question, fetch the nearest
passages, paste them into a prompt, generate. That pipeline has one structural
flaw — **it cannot tell a good retrieval from a bad one.** Whatever comes back
is what the model answers from, so weak context does not produce a weak answer.
It produces a confident wrong one.

This document explains what is done differently here, with the evidence for
each claim, and ends with an honest list of what the system is *not* better at.

- [The core difference: it checks before it answers](#the-core-difference-it-checks-before-it-answers)
- [Side by side](#side-by-side)
- [The eight things that are actually different](#the-eight-things-that-are-actually-different)
- [What it is not better at](#what-it-is-not-better-at)
- [How these claims were tested](#how-these-claims-were-tested)

---

## The core difference: it checks before it answers

Between retrieval and generation there is a separate model call — the **SURE
verifier** — that never sees the answer and never writes one. Its only job is to
judge the evidence:

```
   ordinary RAG      retrieve ──────────────────▶ generate
                                  (hope)

   SURE-GraphRAG     retrieve ──▶ VERIFY ──▶ generate
                          ▲          │
                          └─ expand ─┘   only while it is measurably working
```

The verifier returns four things: whether the evidence is sufficient, a score,
**what specifically is missing**, and — the part that matters most — whether the
gap is even findable in this corpus. That last flag is what separates *"search
harder"* from *"this document does not contain that."*

A system that cannot make that distinction will do one of two bad things: keep
searching for something that was never ingested, or quietly answer anyway.

---

## Side by side

| | Typical RAG | SURE-GraphRAG |
|---|---|---|
| **Evidence quality** | Unknown. Whatever the retriever returned | Scored before generating; gaps named specifically |
| **Weak retrieval** | Answers anyway, confidently | Runs a targeted expansion pass, or says what is absent |
| **Question not in the corpus** | Produces a plausible-sounding answer from adjacent text | Says so plainly, capped at 150 words, badge reads *"Not in this KB"* |
| **Retrieval** | Dense vectors only | Dense **+** BM25, fused server-side in one round trip |
| **Exact strings** (part numbers, article numbers, codes) | Frequently missed — embeddings encode meaning, not spelling | Caught by the BM25 half |
| **Relationships across documents** | Only if one passage happens to state it | Graph traversal follows the edges |
| **Ranking** | Vector similarity | Cross-encoder reranking, blended with retrieval order |
| **Citations** | Whatever the model wrote | Every marker validated against the real source list; invented ones stripped |
| **Figures and diagrams** | Dropped at ingestion | Extracted, OCR'd, paired with captions, shown beside the answer |
| **Transparency** | A black box | Every step, timing, score and stop reason visible |
| **Confidence shown** | One number, or none | Two — evidence coverage *and* answer grounding — which may disagree |
| **Cost** | API calls for embeddings *and* generation | Embeddings, keyword search and reranking are local and free |
| **Runs offline** | No | Yes, with Ollama or HuggingFace |

---

## The eight things that are actually different

### 1. Evidence is judged before an answer exists

The verifier reads twelve passages — more than the answer will use — and scores
coverage. Judging a wider window than the synthesiser reads is deliberate: a gap
reported while the evidence sits just outside the window sends the whole loop
chasing something it already has.

It is also told **what the corpus is made of**, measured from a sample of real
passages: *"192 of the 200 passages sampled are structured records with exactly
these fields: article, title, description."* Without that, a verifier cannot
distinguish a retrieval failure from material that was never ingested, and will
loop forever trying to find the latter.

### 2. The loop stops on measurement, not on what the model says

Expansion is expensive — new queries, a hypothetical-answer probe, a graph
widening, another verification. It is worth that only if it works. So the rule
is a single question: **did the sufficiency score actually rise?**

Everything softer was tried first and failed on live traffic:

- *Compare the verifier's gap list as text* → failed. The model rephrases the
  same complaint every round, so identical gaps never looked identical.
- *Continue while new passages arrive* → failed. New passages always arrive; the
  corpus is large and the queries keep changing. The loop ran until a hard
  budget killed it.
- *Tell the verifier in its prompt to be less demanding* → failed. It
  paraphrased its way around the instruction.

Each of those failures is preserved as a test in `backend/tests/test_stop_rules.py`.
That history is the point: the rule is deliberately the strictest one that
survived contact with reality.

The strictness is affordable because expansion, when it does fire, earns its
seconds. Across five questions deliberately phrased in everyday language rather
than in the document's vocabulary, the hypothetical-answer probe changed the
outcome once — and that once was decisive. For *“can the Centre dismiss a state
government”*, Article 356 (President's Rule) was **absent from base retrieval
entirely**; the probe brought it to rank 2. The other four questions kept their
target passage inside the verifier's window either way. A cheaper loop that
skipped the probe would have answered that one question from whatever text
happened to be nearest — or declared President's Rule absent from the
Constitution.

### 3. The graph enhances retrieval; it cannot veto it

This is a correction of the obvious design, and it was learned the hard way.
Originally the graph route *replaced* the vector search. Then a question about a
constitutional article returned nothing useful, because the entity extractor had
never created a node for it — and a graph route that finds *something*
irrelevant never falls back.

Now the graph route runs the vector search too and merges both. The graph
contributes the relational layer that vector search cannot see; it does not get
to decide what evidence the answer is allowed.

### 4. Reranking blends rather than decides

A cross-encoder reads the question and each passage *together*, which is far
more accurate than comparing two independently-computed embeddings. But it is
trained on web prose and is measurably weaker on structured records. The corpus
here is the Constitution of India, stored one article per record, and the
question under test named two of them outright. Measured before and after the
cross-encoder pass:

| Passage the question named by number | Before rerank | Cross-encoder alone | Blended 60/40 |
|---|---|---|---|
| Article 45 — free and compulsory education | rank 2 | **rank 7** | rank 3 |
| Article 51A — fundamental duties of a citizen | rank 7 | **rank 11** | rank 7 |

The verifier reads twelve passages, so rank 11 is one place from falling out of
the window entirely. Dense retrieval and BM25 had already put both where they
belonged; the reranker, reading records that look nothing like the web prose it
was trained on, pushed the two passages the question had *asked for* toward the
edge of what the verifier would ever see.

So it is blended 60/40 with retrieval order — enough to reorder, not enough to
overrule. It also only ever reorders the pool, never truncates it, because
reranking runs again after each expansion pass and anything dropped is dropped
for the rest of the turn.

The same rule applies to the verifier's window, which is only the leading twelve
passages and slides as expansion merges new ones in. A pass that lowers the
score has made the evidence worse, so the answer is written from the pool that
scored best rather than the one that happened to be last — without discarding
anything the later pass found.

### 5. Two confidence numbers, allowed to disagree

- **Evidence coverage** — the verifier's verdict on the pool, before writing.
- **Answer grounding** — the share of the written answer's claims that cite a
  source.

They measure different things and their disagreement is informative. A question
whose answer must be assembled from several provisions reports partial coverage
— no single passage states the conclusion — while the assembled answer is
100% cited. Reporting one number would hide that; reporting both explains it.

Sentences *about* the evidence ("the sources do not explain X") are excluded
from grounding, because the engine asks for that sentence when evidence is thin
and it would be perverse to then penalise the answer for writing it.

### 6. It will tell you the answer is not there

When the verifier concludes the corpus does not hold this kind of material, the
badge reads **"Not in this KB"**, the answer is capped at a short honest
statement of what is absent, and it names the nearest thing that *is* present.

The alternative — which most systems do — is to produce a fluent paragraph
assembled from whatever was nearest. That is the single most dangerous failure
mode in retrieval, because it is indistinguishable from a good answer.

Saying it wrongly is its own failure, though, and the conditions for saying it
are deliberately narrow. A low score alone is not enough: a corpus can hold the
answer under a name the questioner did not use, and a verifier reading for the
questioner's words will score that at the floor. So the verdict is never drawn
over the verifier's own “yes, this is findable here”, and never drawn at all
about a corpus whose shape was not actually measured. Asserting what a corpus
lacks, without having established what it holds, is a guess wearing a verdict's
clothes.

### 7. Citations are verified, not trusted

Models cite `[7]` when five sources exist, or cite a source they did not use.
Every marker is checked against the real source list, invalid ones are stripped
from the text rather than rendered as dead links, and the survivors are
renumbered densely so the reader sees 1, 2, 3.

Each chip opens the original passage **in its surrounding context**, so a claim
can be checked against the document rather than against a snippet chosen to
support it.

### 8. It runs on your machine, and adapts to whatever model you have

Embeddings, keyword indexing and reranking are all local and free. Only answer
generation calls out, and five providers are supported behind one environment
variable — two of them (Ollama, HuggingFace) fully local, so the system can work
with no internet at all.

The Azure provider **negotiates with the API at runtime**: it discovers whether
a deployment is a reasoning model, whether it accepts `temperature`, JSON mode, a
system role, or `max_completion_tokens`, and adapts from the error responses. You
do not have to know which kind of model you deployed. Every one of those paths is
reproduced in `backend/tests/test_azure_provider.py` with a fake client, so it is
tested without an Azure account.

---

## What it is not better at

Any honest comparison needs this section.

- **Latency.** Verification and expansion cost real seconds. A plain
  retrieve-and-generate pipeline will beat this on a question it happens to get
  right the first time. The trade is deliberate: seconds for the ability to
  know when the evidence is thin.
- **Scale.** Single node, single process, in-memory work queue. This is built
  for one person's documents on one machine. Scaling out means moving the queue
  and the session store somewhere real.
- **Quality of the graph.** Entity extraction is only as good as the LLM doing
  it, and it produces noise — near-duplicate entities, and generic hubs that
  connect to everything. There is filtering for both, but a hand-curated
  knowledge graph will beat an extracted one.
- **It cannot fix an incomplete corpus.** If the source document is missing a
  section, the system will correctly and repeatedly tell you it is missing. That
  is the right behaviour, but it is not an answer.
- **The verifier is a model, not an oracle.** It is pessimistic on questions
  whose answer must be composed from several passages. The loop is bounded in
  code precisely because the prompt could not be relied on to fix that.

---

## How these claims were tested

Nothing above is a design intention that was never checked.

- **111 backend tests and 26 frontend checks**, all offline — no API key, no
  running database. Run them with `python tests/run.py` (no test framework
  needed) or with `pytest`.
- **Most tests encode a real regression.** Reranking that deleted evidence the
  verifier had already confirmed present; a scope flag that was dead code
  because a missing JSON key reads as `False`; a call budget spent entirely on
  cheap calls; a 2 GB upload held in memory before being rejected; an expansion
  pass that pushed a passage the verifier had just credited out of the window
  and answered from the poorer pool. Each is locked down by the test that found
  it.
- **Measured on a real corpus** — the Constitution of India, 480 chunks, 1,346
  entities, 3,056 relationships — across repeated runs of the same five
  questions, comparing latency, sufficiency, citation count and which passages
  reached the answer.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for how the pieces fit together, and
the main [`README`](../README.md) for setup.

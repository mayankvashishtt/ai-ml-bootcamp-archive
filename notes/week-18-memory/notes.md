# Week 18 — Memory

**Date:** 16/05/2026
**Source:** `downloads/week-18-memory.pdf` (58 slides, 65-slide deck · 2h45m runtime) — *no Colab notebook this week*
**Notion page:** https://100xschool.notion.site/36affffa33e58074907df2639f411331

> A survey lecture in 13 parts, tracing memory from "just use a markdown file" to graph and temporal architectures — and ending with **"memory is not solved."**

---

## Part 1 — The problem we can't yet name

**The scenario:**

```
MAR   started PyTorch
APR   switched to JAX
MAY   back to PyTorch

"What framework should I use?"
```

**What should it say?** Not "you use JAX" (stale), not "you've used PyTorch and JAX" (true but useless). The right answer requires knowing *the current state* and probably *the trajectory*.

### The distinction

> ### **RAG retrieves by similarity. Memory tracks state over time.**

This is the sentence the whole lecture hangs on, and it's the cleanest statement of the difference in the course. Retrieval finds *what matches*; memory knows *what is currently true*. A similarity search over the transcript returns all three framework mentions with roughly equal confidence — that's exactly the failure.

**Two problems, one word:**
- *"What do I know."*
- *"What do I remember about you."*
- **The boundary between them is fuzzy.**

---

## Part 2 — Three types of memory

| Type | Content |
|---|---|
| **Episodic** | Events |
| **Semantic** | Facts |
| **Procedural** | Skills |

**The human pipeline:** sensory → working → long-term (episodic, semantic, procedural).

The taxonomy is borrowed from cognitive psychology and is genuinely useful for engineering, because each type wants different storage: episodic needs timestamps and ordering, semantic needs deduplication and conflict resolution, procedural is closer to a prompt or a script than to data.

---

## Part 3 — Ship memory Monday

**The solution: a file.**

- **Plain markdown**
- **Load at session start**
- **Save at session end**

> **Files on disk — same primitive, three companies:**

| Product | Implementation |
|---|---|
| **Claude Code** | `CLAUDE.md` + auto-memory |
| **Codex** | `AGENTS.md` layered |
| **Hermes** | `MEMORY.md` + `USER.md` |

**What they don't do:**
- **No embeddings**
- **No vector store**
- **Relevance via grep, or an LLM side-call**

Note this is the third time the course has landed here: ChatGPT's memory (Week 10), Claude Code's search (Week 11), and now memory generally — **production systems keep choosing plain text over vector databases.** The reason is consistent: determinism, debuggability, and no retrieval failure mode.

---

## Part 4 — Where the file breaks

| # | Failure | Detail |
|---|---|---|
| **1** | **The cap** | A hard ceiling — **anything after line 200 stays on disk, unread** |
| **2** | **Semantic miss** | *"how do I start training?"* vs `bash scripts/cpt-train.sh` — **no lexical overlap**, so grep finds nothing |
| **3** | **Temporal & portability** | *"Delhi"* and *"Bangalore"* stored with **equal weight — both still true** · file is **local to one machine** · **no team sharing** |

Failure 2 is precisely Week 9's argument for hybrid search, now inverted: grep fails where embeddings succeed. Failure 3 is the Part 1 problem in miniature — **append-only storage cannot represent change.**

---

## Part 5 — Borrow from the OS

> **What if the LLM managed its own memory?**
>
> - **RAM is the context window**
> - **Disk is the external store**
> - **The LLM pages memory in and out**

**Virtual memory, ported.** Same pattern, different substrate.

**Three tiers:**

| Tier | Role |
|---|---|
| **Core** | Always loaded |
| **Recall** | Searchable history |
| **Archival** | Cold storage |

**The lineage: MemGPT (2023) → Letta**
- The LLM **calls memory functions directly**
- **Agent runtime**, not a service
- **Opinionated by design**

The OS analogy is apt: virtual memory solved exactly this problem — a small fast store, a large slow one, and automatic paging. The twist is that here **the LLM is the pager**, deciding what to load, which is the Week 11 "give the model control over its own context" idea again.

---

## Part 6 — Memory as a layer

> **What if a separate system extracted facts?** A memory layer alongside the conversation that **watches every turn, extracts facts, stores them.**

### The Delhi → Bangalore problem

- **Append fails**
- **Two contradictory memories**
- **Need write-time logic**

### Four decisions at write time

| Decision | When |
|---|---|
| **ADD** | Genuinely new |
| **UPDATE** | Modifies existing |
| **DELETE** | Invalidates old |
| **NOOP** | Already known |

> ### **Write-time intelligence, not just write-time storage.**

This is the key architectural move of the section. Naive memory treats writing as an append — cheap, and wrong. Doing the reasoning **at write time** means each write costs an LLM call, but reads become trivial and the store stays consistent. Doing it at read time keeps writes cheap but makes every read pay for conflict resolution, repeatedly, forever.

**One published approach:** the **mem0** paper (ECAI 2025), reporting **LoCoMo 91.6% — vendor-reported.** Other shapes exist: **Letta, Zep.**

---

## Part 7 — Memory as a graph

**Three relationship types:**

| Relationship | Meaning |
|---|---|
| **updates** | New replaces old |
| **extends** | New adds to existing |
| **derives** | New inferred from existing |

> **The graph stores connections, not just facts.**

**Supermemory's graph in practice:**
- **Custom engine**, not a generic graph DB
- **Traversal-based retrieval**
- **Earns its complexity over months — not days**

That last line is the honest engineering note. A graph is more expressive than a file, but the complexity only pays off over a long relationship with a user. **Don't start here.**

---

## Part 8 — Memories with time

```
JAN   joined Anthropic
OCT   left, started own lab

"Where did the user work last spring?"
```

The correct answer is **Anthropic** — which requires knowing not just what's true *now* but what was true *then*.

**Time signatures:**
- `event_start`
- `event_end`
- **Not metadata — structural**

> **The old memory isn't deleted — its window closes.**

**"Not metadata — structural"** is the important claim: validity intervals must be first-class, because queries are *about* them. If time is an annotation you can filter on, you can't answer "last spring" — you need the store to represent that a fact was true over an interval.

**Two implementations:** mem0 temporal layer (May 2026) · **Zep / Graphiti — bi-temporal edges** · LongMemEval 94.8% (vendor-reported).

*Bi-temporal* means tracking both when something was true in the world **and** when the system learned it — so you can also answer "what did we believe last month?"

---

## Part 9 — Forgetting

**The expiring exam:**

> *"Exam tomorrow on transformer attention."*
> Six months later — *"What do I usually study around finals?"*
>
> **First memory is now stale. Second still useful.**

Elegant example: the *specific* fact expires, but the *pattern* it contributes to does not. Delete the memory and you lose the pattern too.

**Delete or down-weight?**

| | |
|---|---|
| **Delete** | **Lossy** |
| **Down-weight** | **Lossless** |
| Scaling band | **0.3× to 1.5×** |

> **Lower weight, not lost — recoverable when needed.**

**Old idea, new home:**

| Year | |
|---|---|
| **1885** | Ebbinghaus forgetting curve |
| **2023** | MemoryBank |
| **2026** | mem0 memory decay |

A 140-year-old psychology result reappearing as a production feature.

---

## Part 10 — Evaluation

**Three benchmarks:** **LoCoMo** (Snap) · **LongMemEval** (vendor-published) · **ConvoMem** (Salesforce)

**Vendor numbers:** mem0 LoCoMo 91.6% · Supermemory #1 across three · Zep 94.8% DMR

> **All numbers vendor-reported. Interpret accordingly.**

### But what's being measured?

> - **Transcript still available at query time**
> - **The system can re-search**
> - **That's retrieval — not memory consolidation**

**This is the sharpest critique in the lecture.** If the full conversation is still accessible when the question is asked, a system can simply search it — which measures retrieval quality, not whether anything was *remembered*. The benchmark scores may be largely measuring RAG.

### A cleaner test — NoReplay (Agarwal et al.)

- **One-pass ingest**
- **Frozen scratchpad**
- **Transcript discarded**

That's a real memory test: you see the conversation once, write down what you think matters, and the original is destroyed. Everything then depends on **what you chose to write** — which is exactly the capability being claimed.

---

## Part 11 — Production deep-dives

### Claude Code
- **Two layers:** `CLAUDE.md` + auto-memory
- **Four memory types**
- **Sonnet side-call, not embeddings**
- **No embeddings. No vector store. LLM as relevance filter.**

### Codex
- **`AGENTS.md`** — layered discovery
- **Memories** — async between sessions
- **Grep over `MEMORY.md`, not vectors**
- **Two pipelines:** static `AGENTS.md` + async Memories

**"LLM as relevance filter"** is the pattern worth naming: instead of embedding everything and ranking by cosine distance, hand a cheap model the candidate memories and ask which are relevant. More expensive per query, far more accurate, and it inherits the model's judgment rather than a distance metric's.

### Trade-offs

| | |
|---|---|
| **Letta** | Most rigorous OS pattern |
| **mem0** | Most published research |
| **Supermemory** | Breadth across interfaces |

---

## Part 12 — The hardest case: voice

```
Total round trip:   500–800 ms
Memory budget:      50–100 ms
A network call alone takes about that.
```

**The inversion:**
- **Nothing expensive between turns**
- **Pre-load before the call**
- **Async writes after**

**Three tiers:**

| Tier | Latency | Mode |
|---|---|---|
| **Hot cache** | 1–5 ms | **blocking** |
| **Background retrieval** | 50–150 ms | **async** |
| **Async writes** | — | **latency irrelevant** |

**Failure modes:** **cold start** (anonymous caller) · **race condition** (pre-load vs write-back) · **per-turn extraction cost**

The voice case forces the general principle into the open: at a 50 ms budget you cannot *fetch* anything meaningful, so everything must already be there. Which is exactly the closing takeaway.

---

## Part 13 — Horizon

**What's still open:** cross-tool portability · privacy & consent · team memory · **a core-identity layer**

**A five-layer future architecture (one sketch):**

| Layer | Role |
|---|---|
| **Sensory** | Filter incoming across modalities |
| **Short-term** | Active reasoning, scratchpad |
| **Long-term** | Semantic + episodic with relationships |
| **Memory managers** | Background consolidation, pruning |
| **Core** | Stable identity |

Note this mirrors the human pipeline from Part 2, with **memory managers** added — background processes that consolidate and prune, which biology does during sleep and no current system does well.

---

## The takeaway

> - **Memory is not solved.**
> - **Infrastructure is mature.**
> - **Semantics are still craft.**
> - **Speed is set by what you prepared, not what you fetch.**

An unusually honest ending. The storage layer is a solved engineering problem; deciding *what* to remember, *when* it stops being true, and *what it means* remains judgment work with no standard answer.

---

## Key takeaways

1. **RAG retrieves by similarity; memory tracks state over time.** The Delhi/Bangalore and PyTorch/JAX cases both break similarity search.
2. **Start with a markdown file** — Claude Code, Codex, and Hermes all do, with **no embeddings and no vector store**.
3. **The file breaks three ways:** a size cap, semantic misses under grep, and no representation of time or team.
4. **The OS analogy** — context is RAM, disk is the store, the LLM pages memory in and out (MemGPT → Letta), across core / recall / archival tiers.
5. **Write-time intelligence beats write-time storage:** ADD / UPDATE / DELETE / NOOP.
6. **Graphs store relationships** (updates / extends / derives) but **earn their complexity over months, not days.**
7. **Time must be structural, not metadata** — `event_start`/`event_end`; the old memory's window closes rather than being deleted.
8. **Down-weight rather than delete** (0.3×–1.5×) — the specific fact expires while the pattern survives. Ebbinghaus 1885 → mem0 2026.
9. **Benchmark scores are suspect:** if the transcript is available at query time, you're measuring retrieval, not memory. **NoReplay** is the cleaner test.
10. **LLM-as-relevance-filter** is the production pattern, not embeddings.
11. **Voice imposes a 50–100 ms budget**, forcing a hot-cache / background / async-write split.
12. **Speed is set by what you prepared, not what you fetch.**

---

## Glossary

| Term | Meaning |
|---|---|
| **Episodic memory** | Memory of events. |
| **Semantic memory** | Memory of facts. |
| **Procedural memory** | Memory of skills and how-to. |
| **Working memory** | The active scratchpad — the context window. |
| **`CLAUDE.md` / `AGENTS.md` / `MEMORY.md`** | Plain-markdown memory files used by Claude Code, Codex, and Hermes. |
| **Auto-memory** | Memories written automatically during a session. |
| **LLM as relevance filter** | Using a model side-call, rather than embeddings, to select relevant memories. |
| **MemGPT / Letta** | OS-inspired memory architecture where the LLM pages its own memory. |
| **Core / Recall / Archival** | Always-loaded / searchable / cold-storage memory tiers. |
| **Write-time intelligence** | Deciding ADD/UPDATE/DELETE/NOOP when storing, not when reading. |
| **mem0 / Zep / Supermemory** | Memory-layer products: most-published research / bi-temporal graph / breadth. |
| **updates / extends / derives** | Graph relationship types between memories. |
| **`event_start` / `event_end`** | Structural validity interval for a memory. |
| **Bi-temporal** | Tracking both when a fact was true and when it was learned. |
| **Memory decay** | Down-weighting rather than deleting aging memories (0.3×–1.5×). |
| **Ebbinghaus forgetting curve** | 1885 result on memory decay over time. |
| **LoCoMo / LongMemEval / ConvoMem** | Memory benchmarks from Snap, vendors, and Salesforce. |
| **NoReplay** | Evaluation with one-pass ingest, a frozen scratchpad, and the transcript discarded. |
| **Hot cache** | Pre-loaded memory served in 1–5 ms, blocking. |
| **Cold start** | No prior memory available — e.g. an anonymous caller. |

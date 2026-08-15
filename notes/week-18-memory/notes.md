# Week 18 — Memory

**Date:** 16/05/2026
**Source:** `downloads/week-18-memory.pdf` (58 slides, 65-slide deck · 2h45m runtime) — *no Colab notebook this week*
**Notion page:** https://100xschool.notion.site/36affffa33e58074907df2639f411331

> A survey lecture in 13 parts, tracing memory from "just use a markdown file" to graph and temporal architectures — and ending with **"memory is not solved."**

---

## 0. The idea in plain language

Week 8 established the fact that makes this week necessary: **the model has no memory.** It is stateless. Every apparent act of remembering is your application resending text.

So if you want an assistant that knows you across weeks, you have to build that. This week is how — and the honest conclusion is that **nobody has solved it well yet.**

**The example that frames everything:**

```
MAR   you started using PyTorch
APR   you switched to JAX
MAY   you switched back to PyTorch

"What framework should I use?"
```

What should the assistant say? Not *"you use JAX"* — that's stale. Not *"you've used PyTorch and JAX"* — true, useless, and evasive. The right answer needs **the current state**, and probably the trajectory that produced it.

**And this is exactly what RAG cannot do**, which is the key distinction of the week:

> **RAG retrieves by similarity. Memory tracks state over time.**

Ask RAG about frameworks and it happily returns all three conversations — they're all about frameworks, so they're all similar. It has no notion that April **superseded** March, or that May **superseded** April. Similarity has no arrow of time. **Memory needs one.**

**The practical advice, which is refreshingly unglamorous: start with a markdown file.** Genuinely. Write facts about the user to a text file and paste it into the system prompt. It's deterministic, inspectable, editable, debuggable, and it will carry you further than you expect. Week 10's finding that ChatGPT appears to use deterministically-assembled plain text rather than vector search is the same lesson from production.

**Then it breaks**, and the lecture walks through where — the file grows past what you want to send every turn, facts contradict each other, and nothing knows which version is current. Each failure motivates the next architecture: OS-inspired tiering (hot/warm/cold), memory as a separate service layer, **graphs** (entities and relationships, so "Alice manages Bob" is queryable), and **temporal** models (facts with valid-from/valid-until, so supersession is representable).

**The part most systems skip entirely is forgetting.** A memory that only accumulates becomes a liability — stale facts get retrieved confidently, and the context bloats until Week 10's context rot bites. Deciding what to *discard*, and how to mark facts as superseded rather than merely adding new ones, is as important as deciding what to store, and it is much harder.

**Two threads worth holding as you read.** Memory is the same **finite-context** problem from Weeks 4 and 10, now stretched across *time* rather than within one request. And the closing note is unusually candid for a course lecture: **memory is not solved.** Treat the architectures here as a menu of trade-offs, not a settled answer.

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

## Common confusions

**"Isn't memory just RAG over past conversations?"** No, and this is the central distinction. **RAG retrieves by similarity; memory tracks state over time.** Similarity has no arrow of time — it returns all three framework conversations equally, with no notion that April superseded March. Memory needs supersession, recency, and a concept of "current."

**"Why start with a markdown file? That seems too simple."** Because it's deterministic, inspectable, editable by hand, and trivially debuggable — you always know exactly what the model sees. Embedding-based memory fails in ways you can't see. Week 10's ChatGPT reverse-engineering found essentially this approach in production. Start here; add machinery only when you hit a specific wall.

**"When does the file actually break?"** Three points: it grows past what you want to resend every turn; facts start contradicting each other with nothing marking which is current; and you need to query relationships ("who does Alice work with?") that flat text can't answer. Each is a different fix.

**"What does 'memory as a graph' buy?"** Entities and relationships become **queryable structure** rather than prose. "Alice manages Bob," "Bob owns the billing service" lets you answer "who should I ask about billing?" by traversal. Flat text requires the model to re-derive that from scratch every time, and it often won't.

**"Why do facts need timestamps?"** Because most real facts have a **validity window**, not a permanent truth value. "Uses JAX" was true in April and false in May. A temporal model stores valid-from/valid-until, so supersession is representable and you can ask "what was true then?" as well as "what's true now."

**"Why is forgetting hard?"** Because deciding what *doesn't* matter requires judgement, and mistakes are asymmetric and invisible. Forget something important and the assistant looks broken; keep everything and you get stale facts retrieved confidently plus context rot (Week 10). Most systems skip forgetting entirely and slowly degrade.

**"Should I store everything just in case?"** No. An append-only memory is a liability that grows. Stale facts don't announce themselves — they get retrieved with the same confidence as current ones, and the model has no way to tell.

**"How do I evaluate a memory system?"** Not by "did it store the fact" but by outcomes: given a real conversation history, does it produce the *right current answer*? Build cases with deliberate supersession (the PyTorch→JAX→PyTorch shape), contradictions, and facts that should have been forgotten. S3's discipline applies — this is exactly the kind of system that looks fine on happy paths.

**"Is memory a solved problem?"** No — the lecture says so explicitly, which is worth respecting. Treat the architectures as a menu of trade-offs rather than a recommended stack, and be suspicious of products claiming otherwise.

**"Why is voice the hardest case?"** No scrollback for the user to check, latency budgets too tight for elaborate retrieval, and transcription errors that get stored as facts. Everything that's merely annoying in text becomes load-bearing.

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

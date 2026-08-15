# Week 11 — Recursive Language Models (RLM)

**Subtitle:** The cutting edge of context processing, and the real-world debate that proves it matters
**Date:** 28/03/2026
**Sources:** `downloads/week-11-recursive-language-model.pdf` (13 slides) · `downloads/week-11-recursive-language-model.ipynb` (22 cells)
**Notion page:** https://100xschool.notion.site/334ffffa33e5809baca5eacdf27cb4ad
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=saM65Y8hD4SzY3XE2e486,mtX3p7DQKAQ1Aie-qP4VIg)

> This is **Phase 4** of the spectrum Week 10 ended on: *giving the model more control over its own context.* The lecture pairs the research idea (RLM) with the live industry argument that makes it concrete — **Claude Code vs Cursor**.

---

## 1. The RLM paradigm

**Recursive Language Models** — Alex Zhang, Tim Kraska & Omar Khattab (MIT CSAIL). Paper: `arxiv.org/abs/2512.24601`

| Traditional LLM | RLM |
|---|---|
| The prompt **IS** the input to the network | The prompt is stored as a **VARIABLE** |
| Every token goes through attention | The model interacts with it via **CODE** |
| **Result: context rot happens** | **Result: no context rot. Infinite scale.** |

**The reframe in one sentence:** stop treating context as *something the model reads* and start treating it as *an environment the model explores*.

This is a genuinely different move from everything in Weeks 9–10. RAG improved *what gets selected* before the model sees it. RLM changes *who does the selecting* — the model itself, at inference time, with a general-purpose programming language.

> ⚠️ Sanity-check the citation: `arxiv.org/abs/2512.24601` is not a well-formed arXiv ID for a paper predating this lecture (2512 = December 2025), and `24601` is Jean Valjean's prisoner number. Verify the reference before citing it anywhere; the *ideas* below stand on their own and match published work in this area.

---

## 2. The architecture

| Component | Role |
|---|---|
| **1. Root LLM** | Receives **only the query**. Does **not** see the massive input directly. Acts as the **controller / researcher**. |
| **2. REPL environment** | Stores the input as a **Python variable**. Executes code written by the root LLM. Returns results (slices, searches) to the model. |
| **3. Available tools** | `context[:1000]` (peek) · `re.findall(...)` (search) · `context[a:b]` (slice) · `rlm_agent(q, chunk)` (**recurse**) |

> **The agentic loop: Write Code → Execute → See Output → Iterate.**
> **Same pattern as agents, but the "tool" is its own context.**

That last line is the key insight. Week 8 gave the agent tools pointing *outward* (Wikipedia, files, shell). RLM points a tool *inward* at the prompt itself. The context becomes a queryable data structure rather than a wall of text.

**`rlm_agent(q, chunk)` is the recursive step** — it spawns a fresh LLM call on a *subset* of the context, which can itself recurse.

---

## 3. Step-by-step example

**Query:** "What was ACME Corp's Q3 APAC revenue?"
**Context:** a 500-page annual report, stored as a Python variable.

| Step | Action |
|---|---|
| **1. Initial peek** | Root LLM reads `context[:2000]` to find the table of contents and document structure |
| **2. Targeted search** | Runs `re.findall(r'APAC\|Asia Pacific', context)` to locate relevant offsets |
| **3. Context slicing** | Slices `context[45000:52000]` and calls a **sub-LLM** to extract the specific Q3 figure |
| **4. Final synthesis** | Root LLM receives the sub-agent's answer and compiles the final response |

> **The root model never processes the full 500 pages. Zero context rot.**

Note step 1: it reads the table of contents first — the same structural move PageIndex makes (Week 10), but discovered *at runtime by the model* rather than pre-built by an ingestion pipeline.

---

## 4. The power of recursion

```
Root LLM (Depth 0)
├── rlm_agent(query, chunk_1)        [D1]
│   ├── rlm_agent(q, sub_1a)         [D2]
│   └── rlm_agent(q, sub_1b)         [D2]
└── rlm_agent(query, chunk_2)        [D1]
    └── rlm_agent(q, sub_2a)         [D2]
```

**Human-like research:**
1. **Scan** — check the table of contents / index
2. **Narrow** — identify relevant chapters
3. **Deep-dive** — read specific sections carefully
4. **Synthesize** — combine findings into an answer

> **Scalability: 10M tokens? 100M tokens? No problem. Each level sees only a SMALL, focused piece. Results flow back up.**

The structure is divide-and-conquer: **no single LLM call ever sees a large context**, so context rot is avoided *by construction* rather than mitigated. Total work still scales with document size — but it's distributed across many small, reliable calls instead of concentrated in one unreliable large one.

---

## 5. Why RLM wins — four claims

1. **No context rot** — each call sees only a small, focused piece; the "lost in the middle" problem **completely disappears**.
2. **Infinite scale** — 10K or 100M tokens, same approach; just recurse deeper.
3. **Active exploration** — the model **isn't passive.** It chooses its path, filters noise via Python, and reads only what matters.
4. **Cost-effective** — irrelevant text is filtered by **string operations (free)**, not by the transformer (expensive). **Only "gold" tokens hit the GPU.**

Claim 4 is the economically important one and easy to miss. A regex over 10M characters costs effectively nothing; passing 10M characters through attention costs a fortune. **Moving filtering from the GPU to the CPU is where the savings come from.**

---

## 6. Results

| Configuration | Performance gain | Context capacity |
|---|---|---|
| **RLM(GPT-5-mini)** vs vanilla GPT-5 | **+34.2 points** | 100× beyond window |
| **RLM(Qwen3-8B)** vs base Qwen3-8B | **+28.3% average** | Infinite recursion |
| **RLM(Llama-3-70B)** vs vanilla Llama-3 | **+19.5 points** | Scales to 10M+ tokens |

> **"A mini model using RLM beats a full model with raw context."**
> Handles inputs up to **100× beyond the model context window** at comparable or lower cost.

The GPT-5-mini row is the headline: **architecture beat model size.** A weaker model with a better relationship to its context outperformed a stronger model drowning in tokens. That's the thesis of the entire lecture.

> ⚠️ These figures come from the (unverified) paper reference above. Treat the *direction* as well-supported by Week 10's context-rot evidence, and the exact magnitudes as needing a primary source.

---

## 7. The real-world debate: Claude Code vs Cursor

> Both are billion-dollar products solving the **same problem** differently.

| | **Claude Code** — Agentic Search | **Cursor** — Indexed RAG |
|---|---|---|
| **Method** | Shell tools (`grep`, `glob`) to explore actively | Embeddings + vector DB mapping code semantically |
| **Index** | **None required** | Requires background indexing |

### Claude Code — agentic search

**The toolkit:** `ls`, `glob`, `grep`, `cat`. **No embeddings. No index.**

**The process:**
1. **Discovery** — "What files exist?" → `ls`, `glob`
2. **Navigation** — "Where is this function?" → `grep`
3. **Inspection** — "What does it do?" → `cat`
4. **Context** — "What calls it?" → `grep` references

> **Philosophy: active exploration over passive consumption.** The model **decides** the search strategy, iterating until it understands.

This is RLM's pattern applied to a filesystem: the codebase is the environment, shell commands are the code, and the model navigates rather than ingests.

### Why agentic search works — three reasons

1. **Precision** — code has **exact symbols**. `grep` gives exact matches with zero ambiguity. Embeddings often return **"fuzzy positives"** — similar but technically wrong.
2. **Freshness** — code changes constantly. Agentic search reads the **actual current file**. No waiting for re-indexing, re-chunking, or re-embedding after every save.
3. **Security & simplicity** — no index to build or maintain; **no sensitive code stored in third-party vector databases**; everything stays local.

> *"Even for our own codebase [at Anthropic], it's very sensitive. We don't want to upload it to a third-party thing."* — **Boris Cherny, Anthropic**

Point 1 is Week 9's finding restated: embeddings are bad at exact identifiers, and **code is almost entirely exact identifiers.** A function name is not a concept to be approximated — it either matches or it doesn't. Code is close to the worst possible domain for semantic search and the best possible one for keyword search.

### Cursor — indexed RAG, done well

| Technique | What it does |
|---|---|
| **Smart chunking** | **Tree-sitter** parses code into ASTs; chunks at **syntactic boundaries** (functions/classes) so logic is never split in half |
| **Incremental sync** | **Merkle trees** detect exactly which files changed; re-embeds only those, instantly |
| **Team reuse** | Clones share **92% similarity** — new team members reuse teammates' indexes, dropping setup to near zero |
| **Semantic search** | Conceptual queries like *"Where do we handle rate limiting?"* without knowing function names |

> **Result: semantic search improved accuracy by 12.5%.**

These are serious engineering answers to Week 9's problems: AST chunking is "structure-aware chunking" done properly, and Merkle trees solve staleness elegantly by hashing the file tree so only changed subtrees need re-embedding.

**And Cursor's semantic search solves something grep genuinely cannot:** finding code whose name you don't know. `grep "rate limit"` misses a function called `throttleRequests`.

### Comparing the struggles

| **Claude Code (agentic)** | **Cursor (indexed RAG)** |
|---|---|
| **Token burn** — grep results load directly into context, increasing costs | **Index staleness** — 5-minute sync delay can cause file drift |
| **Latency** — sequential "20 questions" with the filesystem can be slow | **Setup time** — large repos can take hours to index initially |
| **Conceptual search** — hard to find what you can't explicitly name | **Semantic noise** — "similar but wrong" results distract the model |

> **Neither is perfect; they offer different trade-offs for different workflows.**

Note the symmetry: agentic search trades **latency and token burn** for **precision and freshness**; indexed RAG trades **staleness and noise** for **speed and conceptual reach**.

---

## 8. The 2026 consensus

> **THE BEST TEAMS USE BOTH.**
> - **Claude Code** — deep, autonomous, agentic work. Best for complex multi-file refactors.
> - **Cursor** — quick, visual, interactive work. Best for semantic discovery and rapid UI iteration.

### The core lesson

> **How the model relates to context is the fundamental design decision of modern AI systems.**
> **RLM, Claude Code, and Cursor are just different answers to the same problem.**
>
> *"Understanding why each approach fits its domain is what separates someone who uses AI tools from someone who builds them."*
>
> **Don't blindly stuff the context window.**

---

## Key takeaways

1. **RLM stores the prompt as a variable and lets the model query it with code** — context becomes an environment, not a wall of text.
2. **The root LLM never sees the full input.** It's a controller issuing code against a REPL.
3. **`rlm_agent(q, chunk)` is the recursive primitive** — spawn a fresh call on a subset, which may recurse further.
4. **Context rot is avoided by construction**, not mitigated: no single call ever holds a large context.
5. **Filtering moves from the GPU to the CPU.** Regex is free; attention isn't. Only gold tokens reach the model.
6. **Architecture can beat model size** — RLM(GPT-5-mini) reportedly outperformed vanilla GPT-5.
7. **Claude Code is RLM applied to a filesystem** — `ls`/`glob`/`grep`/`cat` as the exploration language.
8. **Code is the worst case for embeddings** (exact symbols) and the best case for grep.
9. **Cursor's engineering is genuinely good** — AST chunking via Tree-sitter, Merkle-tree incremental sync, index reuse across 92%-similar clones.
10. **Grep can't find what you can't name** — that's the real gap semantic search fills.
11. **Both approaches are valid; the trade-offs differ.** Latency/tokens vs staleness/noise.
12. **How the model relates to context is *the* design decision.** Don't blindly stuff the window.

---

## Glossary

| Term | Meaning |
|---|---|
| **RLM** | Recursive Language Model — the prompt is a variable the model manipulates via code. |
| **Root LLM** | The controller that sees only the query and writes code to explore the context. |
| **REPL environment** | Sandbox holding the context as a variable and executing model-written code. |
| **`rlm_agent(q, chunk)`** | Recursive call spawning a sub-LLM on a slice of context. |
| **Depth (D0, D1, D2)** | Level in the recursion tree. |
| **Active exploration** | The model choosing what to read, versus passively receiving retrieved text. |
| **Gold tokens** | The small relevant subset actually worth sending through the transformer. |
| **Agentic search** | Exploring a codebase with shell tools rather than a prebuilt index. |
| **Fuzzy positive** | A semantically similar but technically wrong retrieval result. |
| **Tree-sitter** | Parser generator producing ASTs, used for syntax-aware code chunking. |
| **AST** | Abstract Syntax Tree — a program's structural representation. |
| **Merkle tree** | Hash tree enabling detection of exactly which files changed. |
| **Index staleness** | The gap between the current code and what the index reflects. |
| **Token burn** | Context consumed (and paid for) by tool output such as grep results. |

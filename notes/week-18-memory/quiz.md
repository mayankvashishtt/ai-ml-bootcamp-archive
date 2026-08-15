# Week 18 — Quiz (20 questions)

**Topic:** Memory for LLM systems
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The lecture's core distinction is:
- A) RAG is faster; memory is more accurate
- B) RAG retrieves by similarity; memory tracks state over time
- C) RAG is for documents; memory is for code
- D) RAG uses embeddings; memory uses graphs

**2.** The three types of memory borrowed from cognitive psychology are:
- A) Short, medium, long
- B) Episodic, semantic, procedural
- C) Hot, warm, cold
- D) Core, recall, archival

**3.** Claude Code, Codex, and Hermes all implement memory using:
- A) Vector databases
- B) Plain markdown files
- C) Knowledge graphs
- D) Fine-tuned adapters

**4.** Which is NOT one of the three ways the markdown file breaks?
- A) The cap — a hard size ceiling
- B) Semantic miss — exact-match retrieval
- C) Temporal and portability limits
- D) Excessive embedding cost

**5.** In the OS analogy, the context window corresponds to:
- A) Disk
- B) RAM
- C) The CPU cache
- D) Swap space

**6.** MemGPT's three memory tiers are:
- A) Episodic, semantic, procedural
- B) Core, recall, archival
- C) Hot, background, async
- D) Sensory, short-term, long-term

**7.** The four write-time decisions are:
- A) READ, WRITE, MODIFY, EXECUTE
- B) ADD, UPDATE, DELETE, NOOP
- C) STORE, INDEX, RANK, SERVE
- D) CREATE, MERGE, SPLIT, ARCHIVE

**8.** The three graph relationship types are:
- A) parent, child, sibling
- B) updates, extends, derives
- C) before, during, after
- D) causes, correlates, contradicts

**9.** Time signatures (`event_start`, `event_end`) are described as:
- A) Optional metadata
- B) Structural, not metadata
- C) Only needed for voice applications
- D) Derived at query time

**10.** For an expiring memory, the lecture recommends:
- A) Immediate deletion
- B) Down-weighting (0.3×–1.5×), which is lossless
- C) Moving it to a separate database
- D) Re-embedding it

**11.** The critique of memory benchmarks is that:
- A) The datasets are too small
- B) The transcript is still available at query time, so they measure retrieval rather than consolidation
- C) They only test English
- D) They require proprietary models

**12.** In a voice application with a 500–800 ms round trip, the memory budget is:
- A) 5–10 ms
- B) 50–100 ms
- C) 200–300 ms
- D) 400–500 ms

---

## Short answer

**13.** Explain the PyTorch → JAX → PyTorch scenario and why similarity-based retrieval gives the wrong answer.

**14.** Describe the three failure modes of the markdown-file approach, and say which technique from earlier weeks addresses each.

**15.** Explain the OS/virtual-memory analogy and what is novel about who does the paging.

**16.** Explain the ADD/UPDATE/DELETE/NOOP decision and the trade-off between write-time and read-time intelligence.

**17.** Explain why time signatures must be structural rather than metadata, using the Anthropic example.

**18.** Explain the expiring-exam example and why down-weighting beats deletion.

**19.** Explain the NoReplay critique. Why does transcript availability undermine a memory benchmark, and what does NoReplay change?

**20.** Design a memory system for a voice assistant with a 60 ms budget. Address the three failure modes named in the lecture.

---
---

## Answer key

**1. B** — RAG retrieves by similarity; memory tracks state over time.

**2. B** — Episodic (events), semantic (facts), procedural (skills).

**3. B** — Plain markdown: `CLAUDE.md` + auto-memory, layered `AGENTS.md`, and `MEMORY.md` + `USER.md`. No embeddings, no vector store.

**4. D** — Embedding cost is not a failure mode, since these systems use no embeddings at all.

**5. B** — RAM. Disk is the external store, and the LLM pages memory between them.

**6. B** — Core (always loaded), recall (searchable history), archival (cold storage).

**7. B** — ADD, UPDATE, DELETE, NOOP.

**8. B** — updates (new replaces old), extends (new adds to existing), derives (new inferred from existing).

**9. B** — Structural, not metadata, because queries are about the validity interval itself.

**10. B** — Down-weight within a 0.3×–1.5× scaling band; deletion is lossy, down-weighting is lossless and recoverable.

**11. B** — If the transcript is available at query time, the system can simply re-search it, which measures retrieval rather than whether anything was consolidated into memory.

**12. B** — 50–100 ms, which is roughly the cost of a single network call.

**13.** The user started with PyTorch in March, switched to JAX in April, and returned to PyTorch in May; they then ask "What framework should I use?" **Similarity retrieval fails** because all three mentions are near-identical in topic — same user, same subject, same vocabulary — so a vector search surfaces them with roughly equal scores and no basis for preferring one. Nothing in a similarity score encodes recency or supersession, so the system may confidently answer "JAX" from a stale memory, or return all three, which is true but useless. The correct answer requires knowing **which statement is currently in force**, and ideally that the user has oscillated. That is state tracking, not retrieval — hence "RAG retrieves by similarity; memory tracks state over time."

**14.** **(i) The cap** — a hard size ceiling, where anything past roughly line 200 stays on disk unread. This is the context-budget problem from Weeks 10 and 17, addressed by **compaction, summarisation, or tiered storage** so only high-signal content is loaded. **(ii) Semantic miss** — "how do I start training?" does not lexically overlap `bash scripts/cpt-train.sh`, so grep returns nothing. This is exactly the case **dense/semantic search** handles well, and Week 9's conclusion applies: **hybrid search**, since grep is excellent for exact identifiers and terrible for paraphrase, while embeddings are the reverse. **(iii) Temporal and portability** — "Delhi" and "Bangalore" are stored with equal weight because append-only files cannot represent supersession, and the file lives on one machine with no team sharing. This needs **write-time intelligence** (UPDATE rather than ADD) and **structural time signatures**, plus a shared store for portability.

**15.** The analogy maps the **context window to RAM** — small, fast, and where computation happens — and an **external store to disk**, large and slow, with memory **paged in and out** between them as needed. It is the same pattern virtual memory has used for decades, ported to a different substrate. **What is novel is who does the paging.** In an operating system the kernel pages transparently, and the running program neither knows nor decides. Here **the LLM calls memory functions directly**, choosing what to load into its own context and what to write back — MemGPT's design, carried into Letta as an agent runtime rather than a service. This is the same movement seen in Weeks 10 and 11: shifting control of context from the surrounding pipeline to the model itself, so the model actively manages its working set instead of passively receiving whatever was retrieved.

**16.** When a new fact arrives, a memory layer must decide whether it is **ADD** (genuinely new), **UPDATE** (modifies something existing), **DELETE** (invalidates something existing), or **NOOP** (already known). Naive systems treat every write as an append, which is why "Delhi" and "Bangalore" end up coexisting as contradictory memories. **The trade-off:** doing this reasoning **at write time** costs an LLM call per write — expensive, and it makes writes slow — but it keeps the store consistent, so reads are trivial lookups over facts that are already reconciled. Doing it **at read time** keeps writes cheap but forces every single read to detect and resolve the same contradictions, repeatedly and forever, with the cost growing as contradictions accumulate. Since reads typically vastly outnumber writes, and since latency usually matters more at read time (acutely so for voice), paying once at write time is generally correct — hence "write-time intelligence, not just write-time storage."

**17.** The user joined Anthropic in January and left in October to start their own lab; the question is "Where did the user work **last spring**?" and the correct answer is **Anthropic**, even though it is no longer true today. **Why metadata is insufficient:** if timestamps are merely annotations attached to records, the store represents "this fact was written on date X," not "this fact was true from X to Y." Answering a question *about* an interval then requires reconstructing validity from write timestamps, which fails as soon as facts arrive out of order, are backfilled, or describe a period other than when they were recorded. **Structural time** means `event_start` and `event_end` are first-class properties of the memory itself, so the store can directly answer "what was true during this window." It also changes what supersession means: when the user leaves, **the old memory is not deleted — its window closes**, preserving history while making the current state unambiguous. Zep/Graphiti extends this to **bi-temporal** edges, tracking both when something was true and when the system learned it, which additionally answers "what did we believe last month?"

**18.** A user says "Exam tomorrow on transformer attention." Six months later they ask "What do I usually study around finals?" **The specific memory is stale** — that exam is long past and the fact is no longer actionable — **but it still contributes to a durable pattern** about study behaviour around exam periods. **Deletion is lossy** in exactly this way: removing the expired fact also destroys its contribution to any aggregate or pattern derived from it, and the loss is irreversible, so a memory wrongly judged stale can never be recovered. **Down-weighting is lossless**: applying a scaling factor within a 0.3×–1.5× band lowers the memory's influence on retrieval and reasoning without removing it, so it stops crowding out current facts while remaining available when a query — like the one about finals — actually needs it. The idea is old: Ebbinghaus described the forgetting curve in 1885, MemoryBank applied it in 2023, and mem0 shipped memory decay in 2026.

**19.** Existing memory benchmarks such as LoCoMo and LongMemEval typically leave the **full transcript available at query time**, so a system can simply **re-search the conversation** when asked a question. That measures how well it retrieves from a document it still possesses — which is RAG — rather than whether it consolidated anything into memory at all. A system with no memory layer whatsoever could score well by searching the transcript, so high vendor-reported numbers (91.6%, 94.8%) may be substantially measuring retrieval quality. **NoReplay (Agarwal et al.) changes three things:** ingest is **one-pass**, the scratchpad is **frozen** after ingestion, and the **transcript is discarded**. The system therefore sees the conversation exactly once, must decide at that moment what is worth recording, and is then evaluated with no ability to go back. Performance depends entirely on **what it chose to write and how it structured it** — which is precisely the capability "memory" is supposed to name. The broader caution stands regardless: all the headline numbers are vendor-reported, so interpret accordingly.

**20.** **The budget dictates the architecture: at 60 ms you cannot fetch anything meaningful, so everything must already be present.** A single network call consumes roughly the entire budget, which forces the inversion — nothing expensive between turns, pre-load before the call, and async writes after. **Three tiers:** a **hot cache** of the caller's core memories held in process memory, served in 1–5 ms and blocking, populated at call setup from the caller ID; **background retrieval** running asynchronously at 50–150 ms, fetching deeper context that lands mid-conversation rather than blocking the first turn; and **async writes**, where extraction and the ADD/UPDATE/DELETE/NOOP decision happen entirely after the call, where latency is irrelevant. **Cold start (anonymous caller):** there is no identity to pre-load, so degrade gracefully — serve from a generic profile, build session-local memory in the scratchpad during the call, and if identity is established mid-call, warm the hot cache in the background rather than blocking. **Race condition (pre-load versus write-back):** if a previous call's async write is still in flight when a new call pre-loads, the caller gets stale state. Version the memory store and have pre-load read a consistent snapshot, or make write-back idempotent and reconcile by `event_start`; simplest is a short write barrier keyed by user, so pre-load either waits a bounded few milliseconds or knowingly reads the prior version. **Per-turn extraction cost:** do not run extraction on every turn. Buffer the transcript and extract in batch after the call, or extract on a low-frequency trigger — turn count or detected topic change — so the per-turn path stays free of LLM calls entirely. The governing principle is the lecture's own: **speed is set by what you prepared, not what you fetch.**

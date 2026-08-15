# Week 10 — Quiz (20 questions)

**Topic:** Why RAG Breaks — Context Rot, Production Reality, PageIndex
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** Context rot, per Chroma Research (2025), was demonstrated across:
- A) 3 models
- B) 18 frontier LLMs
- C) A single GPT-4 variant
- D) Only open-source models

**2.** The mechanistic cause of context rot is:
- A) Running out of GPU memory
- B) The finite attention budget being stretched across more tokens
- C) Tokenizer failures at long lengths
- D) Positional encoding overflow

**3.** In the Dresden needle experiment, performance TANKED when:
- A) The context exceeded 1M tokens
- B) Four similar-but-wrong distractors were added
- C) The needle was moved to the start
- D) The question was rephrased

**4.** The counter-intuitive finding about document quality is:
- A) Random text is harder than coherent documents
- B) Coherent documents are harder than random text, because models follow narrative arcs
- C) Document formatting has no measurable effect
- D) Only PDFs cause problems

**5.** Which is NOT listed as a production challenge for naive RAG?
- A) Distractors
- B) Context rot
- C) Passive model
- D) Embedding model cost

**6.** ChatGPT's memory architecture reportedly uses:
- A) A vector database with embedding search
- B) No vector DB, no RAG, no embeddings — four layers of plain text
- C) Fine-tuning on user conversations
- D) A knowledge graph

**7.** In ChatGPT's conversation summaries layer, what is included?
- A) Both user and assistant messages
- B) Only user messages, not assistant replies
- C) Only assistant replies
- D) Only system prompts

**8.** The stated goal of advanced RAG techniques is summarised as:
- A) "Retrieve as much as possible"
- B) "Be surgical. Every token added is a bet against accuracy."
- C) "Bigger context windows solve everything"
- D) "Fine-tune instead of retrieve"

**9.** PageIndex's core claim is:
- A) Embeddings are faster than trees
- B) Similarity ≠ relevance
- C) Chunking should be smaller
- D) BM25 outperforms vectors

**10.** On FinanceBench, the reported accuracies were:
- A) Naive RAG 54% vs PageIndex 98.7%
- B) Naive RAG 80% vs PageIndex 85%
- C) Naive RAG 98.7% vs PageIndex 54%
- D) Both approximately 70%

**11.** In PageIndex, navigation is:
- A) Passive Top-K selection
- B) Active — the LLM walks the tree and reasons about where to look
- C) Random sampling of sections
- D) Based on document length

**12.** The four phases of the RAG evolution spectrum are:
- A) Naive → Advanced → PageIndex → RLM
- B) BM25 → Vector → Hybrid → Rerank
- C) Small → Medium → Large → Frontier
- D) Chunk → Embed → Store → Search

---

## Short answer

**13.** Explain context rot and derive its cause from the attention mechanism you learned in Week 4.

**14.** Explain why coherent documents are harder than random filler, and what this implies for how you build a RAG corpus.

**15.** Explain why multi-hop questions degrade faster than single-hop lookups under long context.

**16.** "More retrieved chunks is always safer." Refute this using this week's material.

**17.** Describe ChatGPT's four memory layers and give two engineering advantages of this design over vector-based retrieval.

**18.** Explain "similarity ≠ relevance" with an example, and why it caps what embedding-based retrieval can achieve.

**19.** Explain how PageIndex works mechanically, referencing the notebook's "peek behind the curtain" cell. Why does preserving structure help multi-hop?

**20.** The FinanceBench numbers are 54% vs 98.7%. Critique this comparison — what would you need to know before concluding that vector RAG should be abandoned?

---
---

## Answer key

**1. B** — 18 frontier LLMs.

**2. B** — Self-attention is finite; every token competes, so more tokens means attention is stretched thinner per token.

**3. B** — Adding four similar-but-wrong distractors.

**4. B** — Coherent documents are harder, because models follow narrative arcs instead of locating the needle.

**5. D** — Embedding model cost is not listed. The three named are distractors, context rot, and the passive model.

**6. B** — No vector databases, no RAG, no embeddings search; instead four layers of deterministically assembled text.

**7. B** — Only user messages, not assistant replies, across roughly 15 lightweight digests.

**8. B** — "Be surgical. Every token added is a bet against accuracy."

**9. B** — Similarity ≠ relevance.

**10. A** — 54% naive RAG versus 98.7% PageIndex.

**11. B** — Active: the LLM walks the tree and reasons about which sections to open.

**12. A** — Naive RAG (passive) → Advanced RAG (smarter retrieval) → PageIndex (reasoning retrieval) → RLM (active exploration).

**13.** **Context rot** is the finding that model reliability degrades as input length grows — "the 10,000th token is not treated as reliably as the 100th" — and that this degradation is unpredictable and occurs even on simple tasks. **The mechanism follows from attention:** in self-attention every token computes a score against every other token, and softmax normalises each row so the weights **sum to exactly 1**. Attention is therefore a **fixed budget** distributed among competitors. If a critical token received 0.4 of that budget in a 1,000-token context, it cannot also receive 0.4 when competing with 100,000 tokens — the budget is divided ever more thinly, and the signal from the important token is diluted below the noise. Anthropic's framing follows directly: context is "a resource with diminishing marginal returns."

**14.** Random filler is easy to ignore because none of it competes for relevance — the needle is the only thing resembling an answer, so attention naturally concentrates on it. A **coherent document** creates two problems: it has a narrative arc that the model is trained to follow, pulling attention along the argument rather than toward an isolated fact; and its passages are topically related to the query, so many of them are *plausible* answers competing directly with the correct one. This is why adding four similar-but-wrong distractors made performance tank while generic filler did not. **Implication for corpus design:** near-duplicate and highly similar documents are actively harmful, so deduplicate aggressively and prune superseded versions; retrieve fewer, more precisely targeted chunks rather than many topically-adjacent ones; and use metadata filtering to exclude whole categories of plausible-but-wrong material — for example filtering by date so last year's policy cannot compete with this year's.

**15.** A **single-hop** lookup needs one fact to win the attention competition — one success. A **multi-hop** question requires that *every* required fact survives dilution **and** that the model connects them. In the notebook's experiment the model must locate Fact A ("Vasquez leads AURORA-7") near the start, Fact B ("Vasquez launch = March 15") near the end, and then link Vasquez → AURORA-7 → March 15. Each retrieval is independently degraded by context growth, so the joint probability of success falls roughly multiplicatively; and the linking step itself needs both facts held with enough attention weight to be related, which is a stricter condition than either alone. The facts are also deliberately placed far apart, maximising the intervening distractors. Hence "find X" survives long context much better than "connect X and Y."

**16.** It is false because retrieved chunks carry two costs that scale with K. **First, distractors:** Top-K necessarily returns the *most similar* chunks, and beyond the genuinely relevant ones, the next-most-similar are precisely the plausible-but-wrong passages shown to make performance tank. **Second, context rot:** each additional chunk lengthens the prompt, thinning the attention budget so the correct chunk gets less weight even when it is present. Increasing K therefore raises recall while simultaneously lowering the model's ability to use what was recalled — the reason for the maxim *"every token added is a bet against accuracy."* The right response is precision, not volume: retrieve broadly (100–500 candidates) but **rerank down to a small final set**, using an expensive cross-encoder to move the truly relevant chunks to top-1 rather than handing everything to the model.

**17.** **Layer 1 — session metadata:** device, location, subscription tier, usage patterns, injected once at session start. **Layer 2 — explicit facts:** durable user preferences stored as plain text, e.g. "User prefers Python over JavaScript." **Layer 3 — conversation summaries:** roughly 15 lightweight digests of recent chats, containing **only user messages**. **Layer 4 — current session:** the full sliding window of the active conversation. **Advantage 1 — determinism and debuggability:** context is assembled by fixed rules, so you always know exactly what the model sees; there is no similarity search that can silently return the wrong thing, no embedding drift when models are updated, and failures are reproducible. **Advantage 2 — no retrieval failure mode and lower cost:** since nothing is searched, there is no top-K to tune, no distractor problem from near-miss chunks, no embedding infrastructure to run, and no re-indexing when the corpus changes. (Also creditable: summarising only user turns halves storage while keeping the durable signal, since preferences and projects live in what the user says.)

**18.** Embedding similarity measures whether two texts are *about the same thing*; relevance measures whether a passage *helps answer this question*. These come apart in both directions. **Example:** for "What are the exceptions to the remote work policy?", a chunk stating the general policy is highly similar — sharing nearly all its vocabulary — but does not contain the exceptions and is therefore irrelevant, while a short clause under a different heading listing the exceptions may share few words and score poorly. Conversely, a chunk about a *different* company's remote policy may be extremely similar and entirely irrelevant. **Why this caps embedding retrieval:** similarity is a **proxy** for relevance, and every improvement — better chunking, hybrid search, contextual retrieval — optimises the proxy rather than the target. No amount of proxy-tuning closes a gap that exists because the two quantities are genuinely different, which is the argument for retrieval that *reasons* about relevance instead of measuring distance.

**19.** PageIndex parses a document into a **tree** mirroring its natural structure — sections and subsections as nodes carrying `title`, `node_id`, `page_index`, and a `summary` — with **no chunking and no embeddings**. At query time the tree is serialised to text and given to an LLM, which **reasons about which nodes to open**; only those specific pages are then retrieved and used to generate the answer. The notebook's "peek behind the curtain" cell reproduces this with a plain prompt: it renders the tree as indented text and asks *"Which sections (node IDs) would you look at? Explain your reasoning step by step"* — demonstrating that the mechanism is a **table of contents plus an LLM that reads it**, exactly how a person uses a document, with no special model involved. **Why structure helps multi-hop:** the heading hierarchy is the author's own semantic organisation, which chunking destroys and embeddings then try to reconstruct statistically. With the map available, the model can consult it, open one section, see what it learned, and *then* decide where to look next — an iterative, self-directed process — whereas similarity search must guess all needed chunks in a single shot, before knowing what the first chunk contains.

**20.** The comparison is **naive RAG versus PageIndex**, not *advanced* RAG versus PageIndex — and Week 9 showed that naive semantic search is a deliberately weak baseline, beaten substantially by recursive chunking, contextual retrieval, hybrid search with RRF, and cross-encoder reranking. A fair evaluation would pit PageIndex against that full pipeline. The benchmark also favours PageIndex structurally: **FinanceBench** consists of long, heavily-structured financial filings with clear section hierarchies — precisely the document type where structure preservation pays most — so the result may not transfer to unstructured corpora such as chat logs, support tickets, or scraped web pages, where there is no tree to build. **Before concluding vector RAG should be abandoned, you would want:** results against a strong advanced-RAG baseline; results across several benchmarks with different document types; latency and cost per query (tree navigation implies multiple LLM calls, likely slower and dearer than a single vector lookup); ingestion cost and time (1–3 minutes per PDF in the notebook); behaviour at corpus scale, since a tree per document does not obviously answer "which of 100,000 documents?"; and whether the evaluation was run by a neutral party or by PageIndex itself. The honest reading is that reasoning-based retrieval is a genuinely promising direction on structured documents, not that embeddings are obsolete.

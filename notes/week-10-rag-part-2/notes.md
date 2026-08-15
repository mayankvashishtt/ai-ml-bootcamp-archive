# Week 10 — RAG Part 2: Why RAG Breaks & What Comes Next

**Subtitle:** Context Rot, Production Reality, and the Evolution of Retrieval
**Date:** 23/03/2026
**Sources:** `downloads/week-10-rag-part-2.pdf` (14 slides) · `downloads/week-10-rag-part-2.ipynb` (34 cells)
**Notion page:** https://100xschool.notion.site/32fffffa33e5800f8864dd917b36376c
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=xaD5a1TWnmTDldugcVIGr,lUHkHMdGHTVUFcQ9CUP-QQ)

> Week 9 built the RAG pipeline. **This week attacks it.** The core claim: *RAG isn't dead — but naive RAG is*, and the "just use a 1M-token context window" answer is worse than it looks.

---

## 1. Context rot — the hidden killer

> **The 10,000th token is NOT treated as reliably as the 100th.**
> — Chroma Research (2025), tested across **18 frontier LLMs**

| The promise | The reality |
|---|---|
| *"Just give the model more context! 1M+ token windows let RAG retrieve everything and solve hallucination."* | **Models do not read context uniformly.** Performance degrades unpredictably as input length grows — **even on simple tasks.** |

> **More tokens often = worse answers.**

### Why it happens — the attention budget

Self-attention is a **finite resource**. Every token computes scores against every other token, and softmax forces those weights to **sum to 1** (Week 4). So attention is a fixed budget being divided among ever more competitors.

> **More tokens = attention stretched thinner.**

As sequence length grows, the model "forgets" to focus on critical information, producing erratic and unreliable outputs.

> **"Context must be treated as a resource with diminishing marginal returns."** — Anthropic
>
> *"It's like asking someone to read War and Peace in one sitting, then answer a specific question."*

This follows directly from the mechanism you already know: if a needle token held 0.4 of the attention budget at 1K tokens, it cannot also hold 0.4 when competing against 100× more tokens.

---

## 2. The distractors problem

**Experiment — needle in a haystack, with inference required:**

- **Question:** "Which character has been to Dresden?"
- **Needle:** "Yuki lives next to the Semper Opera House"
- **Requires inference:** the Semper Opera House is in Dresden

| Condition | Result |
|---|---|
| Short context | **>90% accuracy** |
| 32K context | **Accuracy drops dramatically** |
| Add 4 similar-but-wrong distractors | **Performance TANKS** |

### The counter-intuitive insight

> **Coherent documents are harder than random text. Models follow narrative arcs instead of finding the needle.**
>
> **YOUR NICELY FORMATTED DOCUMENTS MIGHT BE HURTING RETRIEVAL.**

This inverts normal intuition. Random filler is easy to ignore because nothing competes for relevance. A well-written document *pulls* the model along its argument, and plausible-but-wrong passages actively compete with the correct one. **Distractors are more damaging than noise.**

### The notebook's multi-hop version

The notebook makes the test harder by requiring **two** facts to be connected:

- **Fact A:** "Dr. Elena Vasquez leads the AURORA-7 initiative." *(placed ~25% in)*
- **Fact B:** "Dr. Vasquez confirmed the launch window opens on March 15, 2024." *(placed ~75% in)*
- **Question:** "What is the launch date for AURORA-7?"

The model must chain **Vasquez → AURORA-7 → March 15**. Filler is deliberately *similar* — other projects, dates, and people — so the distractors are plausible.

Tested at 5, 20, 60, 120, 200 filler paragraphs:

```
>> Tiny  (2 facts + 5 fillers)   ~296 tokens  [PASS]
>> Small (2 facts + 20 fillers)  ~948 tokens  [PASS]
...
```

> **Multi-hop reasoning degrades much faster than simple lookup.**

Single-hop lookup needs one token to win the attention competition; multi-hop needs *two separated facts* to both win **and** be linked. The failure probability compounds — which is why "find X" survives long context far better than "connect X and Y."

---

## 3. Naive RAG vs production reality

| The tutorial version | Production challenges |
|---|---|
| 1. Query → Embed<br>2. Search Vector DB<br>3. Get Top-K Chunks<br>4. Stuff into Prompt<br>5. Generate Answer | **Distractors** — Top-K brings back similar-but-wrong chunks that confuse the model<br>**Context rot** — stuffing all K chunks stretches attention thin<br>**Passive model** — the LLM takes whatever noise you give it with no way to verify relevance |

> *"Works great on demos with short, clean documents."*
> **YOUR RAG WORKS ON DEMOS. IT BREAKS ON PRODUCTION DATA.**

**Note the tension with Week 9:** more retrieved chunks improves *recall* but worsens *context rot* and adds *distractors*. Top-K is not "bigger is safer" — it's a genuine trade-off with an optimum well below "as much as fits."

**"Passive model"** is the conceptual hinge of the whole lecture. In naive RAG the model has no agency over its own context — it cannot ask for more, reject a chunk, or navigate. Everything in §6–7 is about giving it that agency.

---

## 4. How ChatGPT actually manages memory

| The common assumption | The production reality |
|---|---|
| *"ChatGPT remembers past conversations using RAG over chat history and vector databases."* | **NO vector databases. NO RAG. NO embeddings search.** |

Reverse-engineering reveals **surgical context engineering over brute-force retrieval**:

| Layer | Content |
|---|---|
| **1. Session metadata** | Device, location, subscription tier, usage patterns. Injected **once** at session start. |
| **2. Explicit facts** | Long-term preferences stored as **text**, e.g. *"User prefers Python over JavaScript."* |
| **3. Conversation summaries** | **~15 lightweight digests** of recent chats — **only user messages, not assistant replies.** |
| **4. Current session** | The full sliding window of the active conversation. |

Two details worth pausing on:

- **Only user messages are summarised.** User turns carry the durable signal (preferences, projects, style); assistant replies are derivative and would double the storage for little gain.
- **It's all plain text, deterministically assembled.** No similarity search means no retrieval failures, no embedding drift, and fully predictable context — you always know exactly what the model sees.

> ⚠️ This is described as *reverse-engineered*, not documented. Treat it as a credible account of the architectural *approach* rather than a verified spec, and note it may already have changed.

---

## 5. Making RAG work — advanced techniques

> **RAG ISN'T DEAD — BUT NAIVE RAG IS.**

| 1. Better retrieval | 2. Better chunking | 3. Better generation |
|---|---|---|
| **Hybrid search** — BM25 + semantic | **Semantic chunking** — split by meaning, not character count | **Citation & attribution** — link every claim to a source chunk |
| **Reranking** — cross-encoders score candidates | **Parent-child chunks** — retrieve small, provide large | **Self-verification** — check the answer is supported by context |
| **Query transformation** — HyDE and rewriting | **Metadata filtering** — narrow by date, source, type | **Context engineering** — be surgical about what the model sees |

> ### **GOAL: BE SURGICAL. EVERY TOKEN ADDED IS A BET AGAINST ACCURACY.**

That last line is the thesis of the week, and it directly reverses the naive instinct to retrieve more "just in case."

### Recap of the techniques (detail in Week 9)

**Hybrid search** — query *"Error 404 in user auth module"*: BM25 finds the exact `Error 404` string; vector search finds documents about *authentication*. Merged via **RRF** → high precision **and** high recall.

**Reranking** — Stage 1 retrieval is **fast and scalable** (top 100–500 candidates from millions; goal: **high recall**, don't miss the answer). Stage 2 reranking is **surgical precision** via an expensive cross-encoder. Better than cosine similarity, handles complex reasoning, moves relevant chunks to **top-1**.

**HyDE** — an LLM generates a fake answer, that answer is embedded and used for search, and the real documents retrieved are fed to the LLM. **Searching "answer-to-answer" is more accurate than "query-to-answer."**

---

## 6. The vectorless frontier — PageIndex

> ### **SIMILARITY ≠ RELEVANCE**

The premise: a chunk can be *semantically similar* to a query while being *irrelevant* to answering it — and the reverse. Embedding distance is a **proxy** for relevance, and optimising the proxy has limits.

| Feature | Vector RAG | PageIndex |
|---|---|---|
| **Retrieval** | Similarity (embeddings) | **Reasoning (tree index)** |
| **Chunking** | Arbitrary splits | **Natural sections** |
| **Structure** | **Destroyed by chunking** | **Preserved (pages/sections)** |
| **Navigation** | Passive (Top-K) | **Active (LLM walks the tree)** |
| **Multi-hop** | Hard / unreliable | **Natural / native** |

### FinanceBench accuracy

```
Naive RAG:   54%
PageIndex:   98.7%
```

> ⚠️ Read this comparison carefully. It's **naive** RAG versus PageIndex, not *advanced* RAG (hybrid + contextual + reranking) versus PageIndex — and FinanceBench is built on long, heavily structured financial filings, exactly the document type where structure preservation helps most. The result is real and striking; it is not a claim that embeddings are obsolete for all corpora.

### How it works, in the notebook

```python
from pageindex import PageIndexClient
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

result = pi_client.submit_document(PDF_PATH)      # builds a tree — takes 1–3 min
tree = pi_client.get_tree(doc_id)
```

Applied to the **DeepSeek-R1 paper** (arXiv 2501.12948 — the same paper used in Week 15). The tree has nodes with `title`, `node_id`, `page_index`, `summary`, and nested `nodes`.

> **"No chunks! No embeddings! Just the document's natural structure."**

Querying is a chat call scoped to the document:

```python
response = pi_client.chat_completions(
    messages=[{"role": "user", "content": q}], doc_id=doc_id
)
```

### Peeking behind the curtain

The notebook's best cell reproduces the mechanism manually — serialise the tree to text, hand it to any LLM, and ask which sections to open:

```python
prompt = f"""You are navigating a research paper to answer a question.

Here is the document's tree structure:

{tree_text}

Question: {query}

Which sections (node IDs) would you look at? Explain your reasoning step by step."""
```

> **"This is EXACTLY what PageIndex does under the hood. The LLM reads the tree and REASONS about where to look. Only those specific pages get sent for answer generation."**

**This demystifies it completely.** PageIndex is a **table of contents plus an LLM that reads it** — exactly how a human uses a document. There's no special model, and you could build a workable version yourself with any structured document.

**Why structure beats chunking:** a heading hierarchy encodes the author's own semantic organisation, which chunking discards and embeddings then try to reconstruct statistically. Multi-hop becomes natural because the model can consult the map, look at one section, then decide where to go next — an *iterative* process, versus similarity search's one-shot guess.

---

## 7. The RAG evolution spectrum

| Phase | Name | Character |
|---|---|---|
| **1** | **Naive RAG** | **Passive retrieval.** Embed, search, stuff. The model receives whatever the retriever finds. |
| **2** | **Advanced RAG** | **Smarter retrieval.** Hybrid, reranking, HyDE. Improving the quality of what the model sees. |
| **3** | **PageIndex** | **Reasoning retrieval.** Vectorless tree navigation; the model reasons about structure. |
| **4** | **RLM** | **Active exploration.** Recursive Language Models — the model **writes code to explore context as an environment.** |

> ### **TREND: GIVING THE MODEL MORE CONTROL OVER ITS OWN CONTEXT.**

That single sentence is the through-line. Phases 1→4 progressively move the decision of *what the model sees* from the pipeline to the model. Phase 4 is Week 11.

---

## Key takeaways

1. **Context rot is real and measured** — 18 frontier models, degradation with length even on simple tasks. The 10,000th token isn't handled like the 100th.
2. **The cause is the attention budget** — softmax weights sum to 1, so more tokens means thinner attention per token.
3. **Distractors hurt more than noise**, and **coherent documents are harder than random text** because the model follows the narrative.
4. **Multi-hop degrades far faster than single-hop** — two facts must both survive *and* be connected.
5. **More retrieved chunks is a trade-off, not a free win** — recall up, distractors and rot up too.
6. **Naive RAG's deepest flaw is that the model is passive** — it can't verify, reject, or ask for more.
7. **ChatGPT's memory uses no vector DB and no RAG** — four layers of deterministically assembled plain text.
8. **Only user messages are summarised** in conversation digests; that's where the durable signal lives.
9. **Every token added is a bet against accuracy.** Be surgical.
10. **Similarity ≠ relevance** — embedding distance is a proxy, and proxies have ceilings.
11. **PageIndex = a table of contents + an LLM that reads it**, preserving structure that chunking destroys.
12. **The trajectory is toward model control of context**: passive → smarter → reasoning → active exploration.

---

## Glossary

| Term | Meaning |
|---|---|
| **Context rot** | Degradation in reliability as input length grows; later tokens handled less reliably. |
| **Attention budget** | The finite, softmax-normalised attention each token can distribute. |
| **Needle in a haystack** | Benchmark hiding a target fact in long filler. |
| **Distractor** | Plausible but incorrect content that competes with the correct answer. |
| **Multi-hop** | Requiring two or more facts to be found and connected. |
| **Top-K retrieval** | Returning the K most similar chunks. |
| **Passive model** | A model with no control over what enters its context. |
| **Context engineering** | Deliberately deciding what the model sees, rather than stuffing. |
| **Conversation summary** | A compact digest of an earlier chat used as durable memory. |
| **Sliding window** | Retaining only the most recent portion of a conversation. |
| **Semantic chunking** | Splitting by meaning rather than fixed size. |
| **Parent-child chunks** | Embedding small chunks, supplying their larger parents to the LLM. |
| **Self-verification** | Having the model check its answer against the provided context. |
| **PageIndex** | Vectorless retrieval navigating a document tree by LLM reasoning. |
| **Tree index** | Hierarchical representation of a document's sections and pages. |
| **FinanceBench** | Benchmark over financial filings; 54% naive RAG vs 98.7% PageIndex. |
| **RLM** | Recursive Language Model — the model writes code to explore context as an environment (Week 11). |

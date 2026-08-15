# Week 9 — Quiz (20 questions)

**Topic:** RAG from the Ground Up, Part 1
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** According to the decision table, RAG is the right choice when:
- A) The model needs a new writing style
- B) The model doesn't KNOW something — external or private data
- C) You want faster inference
- D) The knowledge base is under 500 pages

**2.** Faithfulness improved from ~0.47 to ~0.80 by switching from:
- A) GPT-3.5 to GPT-4
- B) Fixed-size chunking to recursive chunking
- C) Chroma to Pinecone
- D) Semantic search to hybrid search

**3.** Semantic (dense) search performs *worst* on:
- A) Conceptual questions in natural language
- B) Exact identifiers like dates, IDs, and acronyms
- C) Synonyms such as car/automobile
- D) Long descriptive queries

**4.** Reciprocal Rank Fusion combines result lists using:
- A) The average of the raw similarity scores
- B) Rank position via `1 / (k + rank)`
- C) A trained neural ranker
- D) The maximum score from either list

**5.** RRF uses ranks rather than scores primarily because:
- A) Ranks are faster to compute
- B) Cosine distances and BM25 scores are on incompatible scales
- C) Scores are unavailable from BM25
- D) Ranks require less memory

**6.** Anthropic's contextual retrieval reports how many fewer failures when combined with BM25 and reranking?
- A) 20%
- B) 45%
- C) 67%
- D) 90%

**7.** A bi-encoder differs from a cross-encoder in that it:
- A) Encodes query and document together
- B) Encodes query and document separately, enabling precomputation
- C) Requires an LLM call per document
- D) Only works on short documents

**8.** In the full pipeline, the order is:
- A) Rerank → hybrid search → contextualize
- B) Contextual chunks → hybrid search (top 10) → rerank (top 3) → generate
- C) Generate → rerank → retrieve
- D) Hybrid search → generate → rerank

**9.** HyDE works by:
- A) Rewriting the question into three variants
- B) Generating a hypothetical answer and embedding that instead of the question
- C) Asking a broader background question first
- D) Compressing retrieved chunks

**10.** Which is NOT one of the four RAG prompt design rules?
- A) Grounding — answer only from context
- B) Escape hatch — say so if you don't know
- C) Citations — cite sources with metadata
- D) Brevity — never exceed 100 words

**11.** ANN in vector databases stands for, and means:
- A) Artificial Neural Network — a learned index
- B) Approximate Nearest Neighbor — fast, slightly lossy similarity search
- C) Adaptive Node Navigation — a graph traversal method
- D) Augmented Nearest Node — exact search with caching

**12.** In the "What was ACME's total revenue in Q3 2024?" test, the top retrieved chunk was from:
- A) The Q3 2024 report, as expected
- B) The Q2 2024 report — ranked above the correct Q3 chunk
- C) The employee handbook
- D) The board meeting notes

---

## Short answer

**13.** Explain the three-way distinction between context window, fine-tuning, and RAG. Give an example use case for each.

**14.** Why is chunking called the place where "80% of RAG systems fail"? Contrast the two implementations, and explain the `len(c.split()) > 10` filter.

**15.** Explain why the "working" revenue query is actually evidence of a retrieval problem, and state the debugging principle it teaches.

**16.** List four query types that break naive semantic search, and explain the common cause.

**17.** Explain contextual retrieval: the problem, the mechanism, the cost, and where it helps most.

**18.** Explain the two-pass retrieval architecture. Why can't a cross-encoder do the first pass?

**19.** Hybrid search ranked the correct RS256 chunk at #2, whereas naive semantic had it at #1. Explain what happened and what it teaches about evaluating retrieval.

**20.** Explain why HyDE works despite the hypothetical answer often being factually wrong.

---
---

## Answer key

**1. B** — RAG addresses missing *knowledge*. Fine-tuning is for new skills, style, or behaviour; the context window suffices for small corpora.

**2. B** — Fixed-size (~0.47) to recursive (~0.80) chunking.

**3. B** — Exact identifiers. Embeddings encode topic and meaning, so `2024-09-15` and `2024-03-22` look nearly identical.

**4. B** — `1 / (k + rank + 1)`, summed across lists, with k = 60.

**5. B** — Cosine distances and BM25 scores occupy different, unnormalised scales and cannot be meaningfully averaged; rank is comparable across both.

**6. C** — 67% fewer retrieval failures.

**7. B** — Separately, which allows document embeddings to be precomputed once and compared cheaply, making it scalable but less precise.

**8. B** — Contextual chunks → hybrid search (top 10) → rerank (top 3) → generate.

**9. B** — It generates a hypothetical answer and embeds that, since answers resemble documents more than questions do.

**10. D** — Brevity is not one of them. The four are grounding, escape hatch, citations, and structure.

**11. B** — Approximate Nearest Neighbor: trading a small amount of recall for orders-of-magnitude speed on millions of vectors.

**12. B** — The Q2 report at dist=0.200, ranked above the correct Q3 chunk at dist=0.208.

**13.** **Context window** — put the entire corpus into the prompt. Appropriate for small, stable knowledge bases (under ~500 pages), e.g. answering questions about a single 40-page product spec. **Fine-tuning** — change *how* the model behaves: a new skill, style, output format, or domain-specific tone, e.g. making a model reliably produce your company's structured incident-report format. **RAG** — change *what* the model knows by retrieving external or private data at query time, e.g. answering questions about an internal wiki that changes weekly. The common mistake is fine-tuning to inject facts: it is expensive, must be redone whenever the data changes, and produces uncitable answers.

**14.** Chunking happens before any retrieval and determines the units the system can ever return, so a bad cut is unrecoverable downstream — no reranker can restore context that was destroyed at ingestion. **Fixed-size** splits every N words with overlap, ignoring structure, so chunks begin mid-sentence with their subject stranded in the previous chunk (*"headcount to 34,500. Engineering teams grew by…"*). **Recursive** splits on paragraphs first, keeping any paragraph that fits intact, and only falls back to sentence splitting for oversized paragraphs — so chunks align with the author's own semantic boundaries. The measured gap is 0.47 versus 0.80 faithfulness. **The `len(c.split()) > 10` filter** discards fragments left over from splitting: very short chunks embed poorly because they carry too little signal to place meaningfully in vector space, and they pollute results by matching weakly against many queries.

**15.** The query returned the **Q2** chunk ahead of the **Q3** chunk (0.200 vs 0.208) despite explicitly asking about Q3. Both chunks discuss ACME's quarterly revenue in billions using near-identical phrasing, and **embeddings capture topic rather than identity**, so the quarter label barely registers in the vector. The system still probably produced the right answer, because the correct chunk was in the top 5 and the LLM could read the "Q3" label — but the retrieval ranking was wrong, and on a harder question, or with a smaller k, that would surface as a real failure. **The principle:** *inspect the retrieved chunks, not just the final answer.* A competent LLM papers over mediocre retrieval, which hides the defect until it silently breaks.

**16.** (i) **Specific dates** — "What changed on 2024-09-15?" (ii) **Exact commands or literal strings** — "What is the Slack command to trigger a rollback?" (iii) **Acronyms and technical identifiers** — "What was the RS256 migration?" (iv) **Cross-document synthesis** — "What is ACME's AI strategy and how does it connect to current products?" **Common cause for the first three:** embeddings represent *semantic topic*, so tokens whose value lies in their exact form — dates, IDs, error codes, ticket numbers, commands — collapse toward each other in vector space; every date looks like every other date. The fourth has a different cause: each chunk is embedded in isolation, so no single chunk contains the bridge between two documents, and similarity search naturally returns a cluster from whichever document matches best.

**17.** **Problem:** chunks lose the context that gives them meaning. "The company's revenue grew by 3%" does not say which company, which period, or which document — so its embedding cannot encode any of that, and it will not be retrieved for a query naming the company or quarter. **Mechanism:** at ingestion, pass the chunk *together with its full source document* to an LLM and ask for a 2–3 sentence situating description naming the document, section, entities, and time period; prepend that to the chunk before embedding and BM25 indexing. **Cost:** one LLM call per chunk with the whole document in the prompt — 90 calls for 90 chunks here — but strictly **one-time at ingestion**, with no query-time cost, and heavily reducible via prompt caching since the document prefix repeats across chunks. **Where it helps most:** cross-document questions, since each chunk now advertises what it is *about* rather than only what it says, letting the retriever connect an AI-strategy chunk in the board minutes to a product chunk in the quarterly report. Anthropic reports **67% fewer failures** combined with BM25 and reranking.

**18.** **First pass — bi-encoder:** the query and each document are encoded **separately** into vectors, so all document embeddings can be computed once at ingestion and stored; retrieval is then a cheap similarity comparison, letting the system scan millions of records to produce the top 50–100 candidates. **Second pass — cross-encoder:** the query and a candidate document are fed in **together** through full attention, so the model can weigh their interaction directly and judge relevance far more accurately, reordering the shortlist to a final top 3–5. **Why a cross-encoder cannot do the first pass:** because it processes query and document jointly, nothing can be precomputed — every candidate requires its own forward pass at query time. Scoring an entire corpus would mean millions of transformer passes per query, which is computationally impossible. The architecture is therefore a deliberate trade: cheap recall over everything, then expensive precision over a shortlist.

**19.** A schema-management chunk containing the word "migration" scored well on BM25 and, once fused, displaced the authentication chunk that semantic search had correctly ranked first. BM25 matches term frequency without understanding which sense of "migration" is meant, and RRF rewards documents ranked well by *either* retriever — so a strong keyword match on a shared but irrelevant term can outrank the correct result. **What it teaches:** hybrid search improves performance **over a distribution of queries**, not on every individual query; any fusion method can degrade a case that one retriever already handled well. Consequently retrieval must be evaluated on a **representative query set with known correct chunks**, measuring aggregate metrics such as recall@k or MRR — never by eyeballing one or two examples, which is exactly how misleading claims like "BM25 distinguishes RS256" get made.

**20.** Because HyDE exploits a **stylistic** rather than factual property of the embedding space. Questions and documents are written in different registers: "What is the home office stipend?" is short, interrogative, and sparse, whereas the passage that answers it — "The company provides a one-time home office stipend of $1,000…" — is declarative, detailed, and full of the domain vocabulary the corpus uses. Embedding the question therefore compares an interrogative vector against declarative ones, which is a mismatch. A **generated hypothetical answer** is declarative and uses plausible domain vocabulary, so it lands much nearer the real documents in vector space even when its specific facts are invented. The generated figure may be wrong, but the surrounding phrasing, terminology, and structure are right — and those are what drive similarity. The hypothetical is a **retrieval probe, never the answer**: the real chunks are retrieved and passed to the model, which grounds its response in them.

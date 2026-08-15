# S6 — Embeddings, Properly

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 3 introduces embeddings conceptually and Weeks 9–10 use them constantly, but the course never explains how an embedding *model* is trained, why cosine similarity is the right metric, or how to choose or improve one.

**Fills the gap after:** Week 3 (embeddings intro), Week 9 (RAG)
**Prerequisites:** Weeks 3, 4, 9

---

## 1. An embedding model is not an LLM

Week 3 taught the *concept* — token → vector, similar meanings nearby. Weeks 9–10 called `text-embedding-3-small` without explaining what it is.

| | **LLM** | **Embedding model** |
|---|---|---|
| Trained to | Predict the next token | Place similar texts near each other |
| Output | A sequence of tokens | **One fixed-length vector** |
| Size | Billions of parameters | Typically 100M–1B |
| Objective | Cross-entropy | **Contrastive** |
| Direction | Autoregressive (causal mask) | Usually **bidirectional** |

**The key structural difference:** an LLM's job is *generation*, an embedding model's job is *compression into a geometry*. Different objective, different architecture choices, different failure modes.

**Bidirectional matters.** Week 7's causal mask exists so a language model can't cheat by reading ahead. An embedding model has no such constraint — it isn't predicting anything, so every token may attend to every other. This is why encoder architectures (BERT-lineage) dominate embedding work.

**Pooling.** Attention produces one vector *per token*; you need one per document. The pooling step collapses them:
- **Mean pooling** — average all token vectors. Simple, strong, most common.
- **CLS pooling** — use a special prepended token's vector, trained to summarise the sequence.
- **Last-token pooling** — used when adapting a decoder-only LLM into an embedding model.

---

## 2. How embedding models are trained: contrastive learning

**The objective in one sentence: pull related texts together in vector space, push unrelated ones apart.**

You need **triples**:

```
anchor    "How do I reset my password?"
positive  "Password reset instructions: click Forgot Password..."   ← should be CLOSE
negative  "Our refund policy allows returns within 30 days"          ← should be FAR
```

The loss (InfoNCE, or a softmax over similarities) maximises the anchor–positive similarity relative to anchor–negative similarities.

### In-batch negatives — the trick that makes it practical

Explicitly labelling negatives is expensive. Instead: take a batch of N (anchor, positive) pairs and treat **every other pair's positive as a negative** for this anchor. One batch of 64 pairs yields 63 negatives per anchor for free.

**Consequence: batch size is a quality parameter for embedding training.** Larger batches mean more negatives per anchor and a sharper signal. This is why embedding models are trained with unusually large batches, sometimes in the thousands.

### Hard negatives — where the quality is

In-batch negatives are mostly *easy* — a random other document is obviously unrelated, so the model learns little from it.

**Hard negatives** are texts that look similar but aren't relevant:

```
anchor           "How do I reset my password?"
positive         "To reset your password, click Forgot Password..."
hard negative    "To change your username, go to Account Settings..."   ← same domain, wrong answer
```

Mining hard negatives — usually by retrieving top-k with a current model and taking the wrong ones — is the single biggest lever on embedding quality.

**This is Week 14's insight in another form.** Contrastive learning teaches *boundaries*; hard negatives are where the boundary actually lies. Easy negatives just confirm the obvious, exactly as Week 14 said of easy preference pairs.

### The typical pipeline

```
1. Pretrain           → masked language modelling or similar
2. Weak contrastive   → huge noisy pairs (title↔body, question↔answer, citation pairs)
3. Fine-tune          → curated pairs with mined hard negatives
```

---

## 3. Bi-encoder vs cross-encoder — deepened

Week 9 introduced both. The important detail is *why* the split is forced.

| | **Bi-encoder** | **Cross-encoder** |
|---|---|---|
| Encodes | Query and document **separately** | Query and document **together** |
| Precompute? | **Yes** — documents embedded once, at ingestion | **No** — every pair needs a forward pass |
| Query cost | One embedding + a vector search | One forward pass **per candidate** |
| Interaction | None until the final dot product | Full attention across the pair |
| Scales to | Millions of documents | Tens of candidates |

**Why a cross-encoder cannot do first-pass retrieval:** because query and document are processed jointly, nothing can be precomputed. Scoring a million documents means a million transformer passes per query. Not slow — infeasible.

**Why a bi-encoder is less accurate:** the document is compressed to a fixed vector **before the query is known**. The encoder must guess what future queries might care about. A cross-encoder sees the query and the document simultaneously, so attention can align specific query terms against specific document spans.

**This is the real explanation for the two-pass architecture** — cheap recall over everything, expensive precision over a shortlist. It isn't a heuristic; it's forced by what can and cannot be precomputed.

**Late interaction (ColBERT)** sits between them: store a vector *per token* rather than per document, and score with a cheap max-similarity operation at query time. More accurate than a bi-encoder, far cheaper than a cross-encoder — at a large storage cost.

---

## 4. Similarity metrics

| Metric | Formula | Behaviour |
|---|---|---|
| **Cosine** | `(a·b)/(‖a‖‖b‖)` | Angle only — magnitude-independent |
| **Dot product** | `a·b` | Angle **and** magnitude |
| **Euclidean (L2)** | `‖a−b‖` | Straight-line distance |

**On normalised vectors, all three give identical rankings.** Cosine is a dot product on unit vectors; L2 distance is a monotonic function of that dot product. If your vectors are unit-length, the choice is about convention and computational convenience, not results.

**When it does matter:** if vectors are *not* normalised, dot product rewards long vectors — which can be right (magnitude sometimes encodes confidence or document importance) or badly wrong (long documents dominating purely because of length).

**Practical rule: normalise, then use cosine or dot product.** Most modern embedding APIs return normalised vectors already; check rather than assume.

**Why angle rather than distance?** Direction encodes *what the text is about* while magnitude tends to encode incidental properties like length. Week 3's word-math (King − Man + Woman ≈ Queen) works because relationships are **directions**, not distances.

---

## 5. Symmetric vs asymmetric search — the mistake people make

| | Symmetric | Asymmetric |
|---|---|---|
| Compares | Two things of the same kind | A **short query** to a **long document** |
| Example | Duplicate-question detection | Search, RAG |
| Needs | One embedding space | **Query and document treated differently** |

**RAG is asymmetric**, and this is why many embedding models take an **instruction prefix**:

```
Query:    "Represent this question for retrieving relevant passages: How do I reset my password?"
Document: "Represent this passage for retrieval: To reset your password, click..."
```

**Getting this wrong silently degrades retrieval.** Embedding a query with the document prefix (or omitting prefixes entirely on a model that expects them) can cost several points of recall with no error message — you simply get worse results.

**This is a partial explanation for Week 9's HyDE.** HyDE works by generating a hypothetical answer and embedding *that*, converting an asymmetric query→document comparison into a symmetric document→document one. Instruction prefixes attack the same mismatch at the model level; HyDE attacks it at the pipeline level.

---

## 6. Dimensions and Matryoshka

| Dimensions | Trade-off |
|---|---|
| 384 | Small, fast, cheap to store; less nuance |
| 768 | Common middle ground |
| 1536 | Standard for larger API models |
| 3072+ | Diminishing returns; 4× the storage of 768 |

**Storage arithmetic:** 1M documents × 1536 dims × 4 bytes = **~6 GB** just for vectors, before the index. Dimension is a real infrastructure decision.

**Matryoshka Representation Learning (MRL)** trains a model so that **truncating the vector still works** — the first 256 dimensions form a usable embedding, the first 512 a better one, and the full vector the best.

Why that's useful: a **two-stage retrieval on one embedding**. Search cheaply over truncated 256-dim vectors, then re-rank the top candidates using full-length vectors. Big speed and memory savings with a small accuracy cost, and no second model.

**Check whether your embedding model supports MRL before truncating.** Truncating a non-MRL embedding degrades it badly, because information is spread across all dimensions rather than front-loaded.

---

## 7. Choosing a model

**MTEB** (Massive Text Embedding Benchmark) is the standard leaderboard.

**Read it with S3's and Week 20's scepticism:**

- **Contamination** — MTEB datasets are public and may be in training data. A model tuned against the benchmark isn't necessarily better on your data.
- **Domain mismatch** — MTEB is broad and general; your corpus is specific. Legal, medical, and code retrieval behave very differently.
- **Averages hide variance** — an aggregate score conceals that a model may be excellent at classification and mediocre at retrieval, which is the task you care about.

**What actually decides it:** build a small eval set from *your* corpus with known correct answers, and measure recall@k (S3). Also weigh the practical constraints — dimension, max sequence length, licence, self-host vs API, and multilingual support if you need it.

**Max sequence length is the underrated criterion.** A model with a 512-token limit silently truncates longer chunks, so content past the limit becomes invisible to retrieval. Check that your chunk size fits.

---

## 8. Fine-tuning embeddings on your domain

Underused, and often more effective than upgrading to a bigger general model.

**Why it works:** general embedding models place vectors according to general semantics. In a specialised domain, the distinctions that matter are ones the general model never learned — in finance, "liquidity" and "solvency" are near-synonyms generally and critically different technically.

**What you need:** (query, relevant-document) pairs. Sources that already exist in most products:
- **Search logs with clicks** — the click is your positive
- **Support tickets** — question and resolving article
- **Q&A pairs** — from documentation
- **Synthetic pairs** — generate questions per chunk with an LLM (Week 13's synthetic data pipeline, pointed at retrieval)

**Then mine hard negatives** — retrieve top-k with the current model, and the incorrect ones become your hard negatives. This is the step that produces most of the gain.

**Expect meaningful improvement** from a few thousand good pairs. It's usually cheaper than the alternatives and it compounds with everything else in the pipeline.

**The cost to remember:** fine-tuning changes the vector space, so **you must re-embed the entire corpus.** Budget for that, and version your index against the model that produced it.

---

## 9. Vector index algorithms

The database side of Week 9's "ANN."

**Exact search** compares against every vector — perfectly accurate, O(n), fine up to ~100K vectors.

| Index | How | Trade-off |
|---|---|---|
| **HNSW** | Multi-layer navigable small-world graph; greedy descent | Fast, high recall, **memory-hungry** |
| **IVF** | Cluster vectors; search only the nearest clusters | Lower memory; recall depends on how many clusters you probe |
| **PQ** | Compress vectors into quantized sub-codes | Massive memory saving, accuracy cost |
| **IVF-PQ** | Both together | Billion-scale on modest hardware |

**HNSW is the default** for most workloads under ~10M vectors. Reach for IVF-PQ when memory becomes the binding constraint.

**The universal knob:** every ANN index trades **recall against speed**, exposed as a search-effort parameter (`ef_search` in HNSW, `nprobe` in IVF). **Tune it explicitly and measure recall** — the default is a guess about your workload, and a silent recall drop here looks exactly like a bad embedding model.

---

## 10. Failure modes, revisited with mechanism

Week 9 observed these empirically. Here's why they happen.

| Failure | Mechanism |
|---|---|
| **Exact identifiers** (`2024-09-15`, `RS256`) | Trained on semantic similarity; every date is semantically "a date." The distinguishing information is the exact string, which is precisely what the model compresses away. **Hence hybrid search.** |
| **Negation** | "Contains gluten" and "gluten-free" share nearly all their tokens and context; contrastive training rarely includes negation pairs |
| **Antonyms** (Week 3's `warmer`/`colder`) | Opposites appear in near-identical contexts, so distributional training places them close together |
| **Length bias** | Long documents average over more content, diluting any specific topic — mean pooling's weakness |
| **Domain drift** | The corpus contains vocabulary the model never saw |

**These aren't bugs to be patched — they're consequences of the training objective.** That's why the fixes are architectural (hybrid search, reranking, contextual retrieval) rather than "use a better embedding model."

---

## 11. Beyond text

**Multimodal (CLIP-style)** — train image and text encoders jointly with a contrastive loss over (image, caption) pairs, producing **one shared space**. You can then search images with text. Same objective, two modalities.

**Code embeddings** — trained on (docstring, code) and (issue, fix) pairs, so natural-language queries find code. Relevant to Week 11's Claude-Code-vs-Cursor debate: this is what Cursor's semantic search runs on, and why it can find `throttleRequests` from "rate limiting" while grep cannot.

---

## 12. How this connects to the course

| Course moment | What this adds |
|---|---|
| **W3** — embeddings intro | How the model is actually trained |
| **W3** — antonym failure (`warmer`/`colder`) | Distributional training explains it |
| **W9** — bi vs cross-encoder | Why the split is *forced*, not chosen |
| **W9** — embeddings fail on identifiers | The mechanism, and why hybrid search is the fix |
| **W9** — HyDE | Asymmetric→symmetric conversion |
| **W9** — ANN | HNSW, IVF, PQ, and the recall/speed knob |
| **W10** — "similarity ≠ relevance" | Contrastive training optimises similarity; relevance is the target |
| **W11** — Cursor's semantic code search | Code embeddings are what makes it work |
| **W13** — synthetic data | Repurposable for embedding fine-tuning pairs |
| **W14** — hard preference pairs teach most | Hard negatives, same principle |

---

## Key takeaways

1. **An embedding model is not a small LLM** — different objective, usually bidirectional, one fixed vector out.
2. **Contrastive learning pulls positives together and pushes negatives apart.**
3. **In-batch negatives make it practical**, which is why batch size is a quality parameter.
4. **Hard negatives are the biggest quality lever** — the same "boundaries, not centres" insight as Week 14.
5. **The bi/cross split is forced by precomputation**, not chosen for convenience.
6. **On normalised vectors, cosine, dot, and L2 rank identically.** Normalise, then stop worrying.
7. **RAG is asymmetric search** — use the right instruction prefixes, or lose recall silently.
8. **HyDE converts asymmetric to symmetric.** That's why it works.
9. **Matryoshka embeddings can be truncated**; ordinary ones cannot.
10. **Read MTEB sceptically** — contamination, domain mismatch, hidden variance. Build your own eval.
11. **Check max sequence length** — silent truncation makes content invisible.
12. **Fine-tuning on domain pairs beats upgrading models**, but requires re-embedding everything.
13. **Every ANN index has a recall/speed knob.** Tune it and measure, or you'll blame the model.
14. **The failure modes follow from the objective**, which is why the fixes are architectural.

---

## Glossary

| Term | Meaning |
|---|---|
| **Embedding model** | A model producing one fixed-length vector per text. |
| **Contrastive learning** | Training to pull related texts together and push unrelated apart. |
| **InfoNCE** | The standard contrastive loss — a softmax over similarities. |
| **Anchor / positive / negative** | The query / a relevant text / an irrelevant text. |
| **In-batch negatives** | Using other pairs in the batch as negatives, for free. |
| **Hard negative** | A superficially similar but irrelevant text — the highest-value training signal. |
| **Pooling** | Collapsing per-token vectors into one — mean, CLS, or last-token. |
| **Bi-encoder** | Encodes query and document separately; precomputable and scalable. |
| **Cross-encoder** | Encodes them jointly; accurate but not precomputable. |
| **Late interaction (ColBERT)** | Per-token vectors with cheap query-time matching. |
| **Cosine / dot / L2** | Angle / angle-and-magnitude / straight-line distance. |
| **Normalisation** | Scaling to unit length, making the three metrics rank-equivalent. |
| **Symmetric / asymmetric search** | Same-kind comparison / short query against long document. |
| **Instruction prefix** | A task-describing prefix some models require per input type. |
| **Matryoshka (MRL)** | Embeddings that remain usable when truncated. |
| **MTEB** | The standard embedding benchmark. |
| **recall@k** | Fraction of relevant documents appearing in the top k. |
| **HNSW / IVF / PQ** | Graph index / cluster index / vector compression. |
| **`ef_search` / `nprobe`** | Search-effort knobs trading recall against speed. |
| **CLIP** | Contrastively trained joint image-text embedding space. |

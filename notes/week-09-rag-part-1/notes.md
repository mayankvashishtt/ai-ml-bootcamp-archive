# Week 9 — RAG from the Ground Up (Part 1)

**Subtitle:** Retrieval-Augmented Generation — Build it. Break it. Fix it.
**Date:** 13/03/2026
**Sources:** `downloads/week-09-rag-part-1.pdf` (12 slides) · `downloads/week-09-rag-part-1.ipynb` (51 cells)
**Notion page:** https://100xschool.notion.site/322ffffa33e580798824de76633122de
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=y4F46fapKAXAyzeJZSgYD,WlwtkiN5brjUIKcX4U50Yg)

> The notebook's structure is the pedagogy: **build the naive version, deliberately break it with hard queries, then fix each failure with a specific upgrade.** Every technique below is introduced as the answer to a failure you've already watched happen.

---

## 1. Why LLMs need RAG

1. **Frozen in time** — LLMs only know their training data. No current events, no recent updates.
2. **No private knowledge** — they haven't seen your company handbook, Slack, or internal docs.
3. **Confident hallucination** — when they don't know, they **confidently invent**. *"It only knows training data. Your docs don't exist in its world."*

### The workflow

```
User Question → RETRIEVE → AUGMENT prompt → GENERATE answer
```

### When to use what — the decision table

| Strategy | When to use |
|---|---|
| **Context window** | Small knowledge base (< 500 pages)? Just stuff it all in. |
| **Fine-tuning** | The model needs a new **skill, style, or behaviour**. |
| **RAG** | The model doesn't **KNOW** something (external/private data). |

**Memorise this distinction — it's the most commonly confused decision in applied AI.** Fine-tuning changes *how* a model behaves; RAG changes *what* it knows. Fine-tuning to inject facts is the classic expensive mistake (revisited in Week 12).

Note also that Level 0 is "don't build RAG at all." With modern context windows, small corpora genuinely don't need a retrieval pipeline.

---

## 2. Ingestion Part 1 — Loading and parsing

> **GARBAGE IN, GARBAGE OUT.**

Raw documents are rarely clean text:
- **PDFs** → tables often destroyed or misaligned
- **Word** → structural headers and metadata lost
- **HTML** → navigation bars and footers included

| Tool | Best for |
|---|---|
| **Unstructured.io** | General purpose, multi-format |
| **LlamaParse** | Complex PDFs and table extraction |
| **Docling** | High-speed, structure-aware parsing |

> **"If your parser can't handle structure, your RAG is broken before it starts."**

---

## 3. Ingestion Part 2 — Chunking

> **"This is where 80% of RAG systems fail. Every cut destroys context."**

| Strategy | Description | Faithfulness |
|---|---|---|
| **Fixed-size** | Split every N tokens with overlap. **Often cuts mid-sentence.** | **~0.47** |
| **Recursive** | Split on paragraphs → sentences → characters. Respects natural boundaries. | **~0.80** |
| **Advanced** | Structure-aware (headers), AST-based (code), Parent-Child (retrieve large, embed small) | — |

**0.47 → 0.80 from a chunking change alone.** This is the highest-leverage decision in the pipeline and it happens before you've written a single retrieval query.

### The two implementations

```python
def chunk_fixed(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks
```

```python
def chunk_recursive(text, max_size=200):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_size:
            chunks.append(para)                      # paragraph fits — keep it whole
        else:
            sentences = para.replace(". ", ".\n").split("\n")
            current, current_len = [], 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > max_size and current:
                    chunks.append(" ".join(current))
                    current, current_len = [sent], sent_len
                else:
                    current.append(sent); current_len += sent_len
            if current:
                chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) > 10]   # drop fragments
```

**Observed difference:** fixed-size chunk 2 begins mid-sentence — *"headcount to 34,500. Engineering teams grew by…"* — with the subject lost to the previous chunk. Recursive chunks start at real section headings.

Note the final filter `len(c.split()) > 10` — dropping tiny fragments. Short chunks retrieve badly because they carry too little signal to embed meaningfully.

---

## 4. Building naive RAG — then breaking it

```python
def get_embeddings(texts, model="text-embedding-3-small"):
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    resp = oai.embeddings.create(input=cleaned, model=model)
    return [d.embedding for d in resp.data]
```

Stored in **ChromaDB** with `metadata={"hnsw:space": "cosine"}` — 90 chunks over 8 documents (financial reports, employee handbook, engineering wiki, board minutes).

### ✅ What works

**"What was ACME's total revenue in Q3 2024?"**
```
[1] dist=0.200 | Q2 2024 Financial Report ...total revenue of $3.9 billion for Q2...
[2] dist=0.208 | Company Q3 2024 Report   ...total revenue of $4.2 billion for Q3...
```

> 🔍 **Look closely — this "working" example is already subtly broken.** The **Q2** chunk ranks *above* the Q3 chunk (0.200 vs 0.208). The embedding sees "revenue," "ACME," "billion," "quarter" and can barely distinguish Q2 from Q3, because **embeddings encode topic, not identity**. The right chunk is in the top-5 so the LLM probably answers correctly — but it got there despite retrieval, not because of it.

> **"Pay attention to the retrieved chunks, not just the final answer. The LLM might paper over bad retrieval, but the chunks reveal the problem."**

This is the single most useful debugging habit in RAG.

### 💥 What breaks

| Query | Failure mode |
|---|---|
| **"What changed on 2024-09-15?"** | Correct chunk retrieved but at **dist=0.551** — barely above noise (0.719, 0.745…). Embeddings are **terrible with dates**: `2024-09-15` and `2024-03-22` are semantically near-identical. |
| **"What is the Slack command to trigger a rollback?"** | Retrieved the *automatic* rollback section, not the slash command. Semantic search matched "rollback" as a concept, missing the literal string. |
| **"What was the RS256 migration and when?"** | Correct chunk at #1 but with weak separation; `RS256` is a rare token that embeddings represent poorly. |
| **Cross-document synthesis** (AI strategy ↔ products) | Retrieved *only* board-meeting chunks. Couldn't bridge to the product document. |

**The pattern: semantic search fails on exact identifiers** — dates, IDs, error codes, acronyms, commands, ticket numbers. Precisely the things people search for at work.

---

## 5. Fix 1 — Hybrid search (semantic + BM25)

| Semantic (dense) | Keyword (BM25 / sparse) |
|---|---|
| Finds **meaning and concepts** | Finds **exact terms, names, IDs** |
| Matches "car" ↔ "automobile" ↔ "vehicle" | Essential for `JIRA-2847`, product codes |
| Great for natural-language queries | Great for literal strings |

> **Industry consensus: hybrid beats semantic alone every time.**

### Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(semantic, keyword, k=60):
    scores = {}
    for rank, (idx, _) in enumerate(semantic):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(keyword):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

> *"Simple, effective, no hyperparameters to tune."*

**Why RRF works:** it fuses on **rank, not score** — which matters because cosine distances and BM25 scores are on incompatible scales and can't be meaningfully averaged. The `k=60` constant damps the difference between top ranks, so a document ranked #1 by one retriever and #3 by the other beats one ranked #1 by only a single retriever. **Agreement across retrievers is rewarded.**

Both retrievers fetch **top 20** before fusing, then the fused list is cut to 5 — retrieve broad, then narrow.

### Metadata filtering — the "WHERE" clause

```python
results = vector_db.search(query, filter={"year": 2024})
```

> **"If you know the user is asking about Q3, why search Q1 and Q2? Filter first, then search."**

Always store **source, page, and date** alongside vectors — for filtering *and* for citations.

### Honest assessment of the results

The date query improved (correct chunk clearly #1). The Slack command query improved. But:

> ⚠️ **On "RS256 migration," hybrid ranked the correct chunk at #2, whereas naive semantic had it at #1.** The notebook's caption says BM25 "distinguishes RS256," which overstates it. Fusion is a net win across a query *distribution*, not a guaranteed improvement on every individual query — a schema-management chunk containing "migration" scored well on BM25 and displaced the right answer. This is normal and worth internalising: **evaluate retrieval over a query set, never on one example.**

---

## 6. Fix 2 — Contextual retrieval (Anthropic's technique)

**The problem:** a chunk like *"revenue grew 3%"* loses context — which company? which year?

```
BEFORE: "The company's revenue grew by 3%..."
AFTER:  "[ACME Corp SEC filing, Q2 2023. Previous quarter: $314M.]
         The company's revenue grew by 3%..."
```

> **67% fewer failures** when combined with BM25 and reranking. **One-time ingestion cost, massive retrieval gain.**

### Implementation — an LLM call per chunk

```python
def contextualize_chunk(chunk, full_doc, title):
    resp = ant.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=150,
        messages=[{"role": "user", "content": f"""<document title="{title}">
{full_doc}
</document>

Here is a chunk from that document:
<chunk>
{chunk}
</chunk>

Write a SHORT (2-3 sentence) context that situates this chunk within the document.
Include: which document, what section/topic, key entities or time periods.
This will be prepended to the chunk for search.

Context:"""}]
    )
    return f"{resp.content[0].text.strip()}\n\n{chunk}"
```

Note the **whole document** is passed as context for each chunk — that's what lets the model say "this is the Revenue section of the Q3 report."

**Result on chunk 0:**

```
BEFORE: "Revenue and Growth
         ACME Corporation reported total revenue of $4.2 billion for Q3 2024..."

AFTER:  "This excerpt is from ACME Corporation's Q3 2024 financial report, specifically
         from the 'Revenue and Growth' section. It provides a breakdown of the company's
         $4.2 billion quarterly revenue across three main business divisions...

         Revenue and Growth
         ACME Corporation reported total revenue of $4.2 billion for Q3 2024..."
```

**Cost:** 90 chunks = 90 LLM calls, each including the full document. Real but **one-time**, and prompt caching makes it far cheaper in practice since the document prefix repeats.

**Where it shines:** the cross-document AI-strategy question. The prepended context tells the embedding *"this chunk is about AI strategy from the board meeting"* versus *"this chunk is about product launches from the Q3 report,"* so the retriever can connect them.

---

## 7. Fix 3 — Reranking (two-pass retrieval)

| First pass: **Bi-Encoder** | Second pass: **Cross-Encoder** |
|---|---|
| **Fast & rough** | **Slow & precise** |
| Compresses query and docs **separately** | Sees query and doc **together** |
| Scans millions of records → top 50–100 | Re-orders candidates → final top 5 |

**Why the split exists:** a bi-encoder can pre-compute document embeddings once and compare with a cheap dot product, so it scales to millions — but query and document never interact, so nuance is lost. A cross-encoder runs full attention over the pair, capturing nuance, but must run *per pair* — impossible over a whole corpus, ideal over 10 candidates.

```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, results, top_k=3):
    pairs = [(question, r["chunk"]) for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
```

**Full pipeline:** contextual chunks → hybrid search (**top 10**) → rerank (**top 3**) → generate.

### Contextual compression

Retrieved chunks are often **90% noise, 10% gold**. Compression uses an LLM to extract only the relevant parts before the final prompt.

> **"The highest-ROI upgrade after chunking. More important as your data grows."**

---

## 8. Query transformation

| Technique | What it does |
|---|---|
| **1. Rewriting** | Turn a messy question into a clean, standalone query optimised for vector search |
| **2. Multi-Query** | Generate 3–5 phrasings, search all, deduplicate → higher recall |
| **3. HyDE** | Generate a *fake answer* first, embed **that** instead of the question |
| **4. Step-Back** | Ask a broader background question first to retrieve high-level context |

> **"The hypothetical answer is often closer to matching documents than the original question."**

HyDE's logic: questions and answers are written in *different registers*. "What's the stipend?" looks nothing like "The company provides a one-time home office stipend of $1,000." A hypothetical answer lives in the same space as the documents — so even a factually wrong hypothesis retrieves the right chunk, because you only need it to be *stylistically* right.

---

## 9. RAG prompt design

**Four rules:**
1. **Grounding** — *"Answer ONLY from provided context."*
2. **Escape hatch** — *"If you don't know, say so."*
3. **Citations** — *"Cite your sources with metadata."*
4. **Structure** — label each chunk clearly.

```python
context = "\n".join(f"[Source: {r['meta']['title']}]\n{r['chunk']}" for r in results)

prompt = f"""Answer based ONLY on the context below.
If the context doesn't have the answer, say "I don't have enough information."
Cite your sources.

Context:
{context}

Question: {question}

Answer:"""
```

> **"RAG without citations is just a fancier hallucination machine."**

The escape hatch matters most. Without it the model fills gaps from its weights, and you get an answer that *looks* grounded but isn't — worse than no answer, because it's unfalsifiable to the user.

---

## 10. The end-to-end pipeline

```
INGESTION:  Docs → Parse → Chunk → Contextualize → Embed → Store
                 ↓
RETRIEVAL:  Question → Transform → Hybrid Search → Rerank → Compress
                 ↓
GENERATION: Prompt → LLM → Answer with Citations
```

### Vector storage

- **Prototyping:** Chroma, FAISS
- **Production:** Pinecone, Weaviate, Qdrant
- **SQL-native:** pgvector (Postgres)

Vector DBs find top-k matches in milliseconds using **ANN (Approximate Nearest Neighbor)** — approximate because exact nearest-neighbour search over millions of vectors is too slow; a small recall sacrifice buys orders of magnitude in speed.

---

## 11. The upgrade path

```
Level 0: Stuff context window       (small docs only)
Level 1: Naive semantic search       ← we started here
Level 2: Better chunking             (recursive, structure-aware)
Level 3: Contextual retrieval        (Anthropic's technique)
Level 4: Hybrid search               (semantic + BM25 + RRF)
Level 5: Reranking                   (cross-encoder)
Level 6: Query transformation        (next class)
Level 7: Agentic RAG                 (next class)
```

> **Each level fixes a specific failure. Start simple. Upgrade where it hurts.**

---

## Homework

1. **Replace the sample docs** with your own (lecture notes, textbook, project docs). Run the same tests.
2. **Try two chunking strategies** on your data. Which works better? Why?
3. **Find YOUR failure cases** — exact terms? dates? cross-doc?
4. **Read:** [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

**Next class:** Self-RAG, GraphRAG, RAPTOR, Agentic RAG — and *why Claude Code abandoned RAG entirely.*

---

## Key takeaways

1. **RAG for knowledge, fine-tuning for behaviour, context window for small corpora.** Don't confuse these.
2. **Chunking is the highest-leverage decision** — 0.47 → 0.80 faithfulness from recursive vs fixed-size alone.
3. **Debug by inspecting retrieved chunks, not answers.** The LLM papers over bad retrieval.
4. **Embeddings fail on exact identifiers** — dates, IDs, acronyms, commands. This is the core motivation for hybrid search.
5. **RRF fuses on rank, not score**, sidestepping incomparable scales and rewarding cross-retriever agreement.
6. **Hybrid wins on average, not on every query** — evaluate over a query set.
7. **Contextual retrieval prepends LLM-generated situating text** before embedding: 67% fewer failures, one-time cost.
8. **Two-pass retrieval:** bi-encoder for recall at scale, cross-encoder for precision on the shortlist.
9. **HyDE works because answers resemble documents more than questions do.**
10. **Always ground, always give an escape hatch, always cite.**

---

## Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — retrieve relevant text, add to prompt, generate. |
| **Chunking** | Splitting documents into retrievable pieces. |
| **Fixed-size chunking** | Splitting every N tokens with overlap; cuts mid-thought. |
| **Recursive chunking** | Splitting on paragraphs → sentences → characters. |
| **Faithfulness** | Metric for whether an answer is supported by retrieved context. |
| **Parent-Child chunking** | Embedding small chunks but retrieving their larger parents. |
| **Contextual retrieval** | Prepending LLM-generated situating context to each chunk before embedding. |
| **Vector database** | Store for embeddings supporting fast similarity search (Chroma, Pinecone, pgvector). |
| **ANN** | Approximate Nearest Neighbor — fast, slightly lossy similarity search. |
| **Dense / semantic search** | Retrieval by embedding similarity; matches meaning. |
| **Sparse search / BM25** | Keyword retrieval by term statistics; matches exact strings. |
| **Hybrid search** | Combining dense and sparse retrieval. |
| **RRF** | Reciprocal Rank Fusion — merges ranked lists via `1/(k + rank)`. |
| **Bi-encoder** | Encodes query and document separately; fast, scalable, less precise. |
| **Cross-encoder** | Encodes query and document together; precise, slow, used for reranking. |
| **Reranking** | Reordering a candidate shortlist with a more accurate model. |
| **Contextual compression** | Using an LLM to strip irrelevant text from retrieved chunks. |
| **Query rewriting** | Rewriting a user question into a retrieval-optimised query. |
| **Multi-query** | Generating several phrasings and merging their results. |
| **HyDE** | Hypothetical Document Embeddings — embed a generated answer instead of the question. |
| **Step-back prompting** | Asking a broader question first to retrieve background context. |
| **Grounding** | Instructing the model to answer only from supplied context. |
| **Escape hatch** | Explicit permission to say "I don't know." |

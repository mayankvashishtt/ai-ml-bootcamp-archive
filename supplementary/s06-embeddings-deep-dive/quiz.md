# S6 — Quiz (20 questions)

**Topic:** Embeddings, Properly
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** An embedding model differs from an LLM primarily in that it:
- A) Is always larger
- B) Is trained with a contrastive objective to produce one fixed-length vector
- C) Uses a causal mask
- D) Generates text token by token

**2.** Embedding models are usually bidirectional because:
- A) It halves training cost
- B) They aren't predicting the next token, so there's nothing to cheat at
- C) Causal masks break pooling
- D) It's required for cosine similarity

**3.** In-batch negatives work by:
- A) Labelling negatives by hand
- B) Treating every other pair's positive in the batch as a negative
- C) Sampling from a negative corpus
- D) Randomly perturbing the anchor

**4.** Batch size is a *quality* parameter in embedding training because:
- A) Larger batches converge faster
- B) More in-batch negatives per anchor gives a sharper contrastive signal
- C) It reduces memory fragmentation
- D) It enables Matryoshka truncation

**5.** A hard negative is:
- A) A text that is obviously unrelated
- B) A superficially similar but irrelevant text
- C) A duplicate of the positive
- D) A negatively-signed vector

**6.** A cross-encoder cannot perform first-pass retrieval because:
- A) It's less accurate
- B) Query and document are encoded jointly, so nothing can be precomputed
- C) It only handles short text
- D) It requires normalised vectors

**7.** On normalised vectors, cosine, dot product, and L2 distance:
- A) Give completely different rankings
- B) Give identical rankings
- C) Only agree for 768 dimensions
- D) Cannot be computed

**8.** RAG is an example of:
- A) Symmetric search
- B) Asymmetric search — short query against long document
- C) Exact search
- D) Late interaction

**9.** Getting instruction prefixes wrong on a model that expects them:
- A) Raises an error
- B) Silently degrades recall
- C) Only affects multilingual models
- D) Doubles latency

**10.** Matryoshka (MRL) embeddings allow:
- A) Compression to int8 without loss
- B) Truncating the vector while remaining usable
- C) Cross-lingual transfer
- D) Training without negatives

**11.** The main risk in choosing a model by MTEB score is:
- A) The leaderboard is paid
- B) Contamination, domain mismatch, and averages hiding per-task variance
- C) It only covers English
- D) Scores change hourly

**12.** After fine-tuning an embedding model you must:
- A) Retrain the LLM
- B) Re-embed the entire corpus
- C) Switch to L2 distance
- D) Rebuild the tokenizer

---

## Short answer

**13.** Explain contrastive learning, including anchor/positive/negative and what the loss optimises.

**14.** Explain why hard negatives matter most, and connect it to a Week 14 principle.

**15.** Explain why the bi-encoder/cross-encoder split is *forced* rather than chosen.

**16.** Explain symmetric vs asymmetric search and how this explains why HyDE (Week 9) works.

**17.** Explain why embeddings fail on exact identifiers, and why the fix is architectural.

**18.** Explain the dimension trade-off and what Matryoshka embeddings enable.

**19.** Explain when fine-tuning an embedding model beats upgrading to a bigger one, and where the pairs come from.

**20.** Your RAG system has poor recall on an internal engineering wiki. Walk through diagnosis and fixes, in priority order.

---
---

## Answer key

**1. B** — Contrastive objective, one fixed-length vector per input, versus an LLM's next-token objective and token-sequence output.

**2. B** — There is no next token to predict, so the causal mask that prevents cheating in a language model serves no purpose; every token may attend to every other.

**3. B** — Each anchor uses the other pairs' positives in the batch as negatives, yielding N−1 free negatives per anchor.

**4. B** — More negatives per anchor sharpens the contrastive signal, which is why embedding models train with unusually large batches.

**5. B** — Superficially similar but not relevant, e.g. "change your username" against a password-reset query.

**6. B** — Joint encoding means no document representation exists until the query arrives, so scoring a large corpus requires a forward pass per document.

**7. B** — Cosine is a dot product on unit vectors, and L2 is a monotonic function of that, so all three produce the same ordering.

**8. B** — A short interrogative query is compared against long declarative passages.

**9. B** — There is no error; you simply get measurably worse retrieval.

**10. B** — MRL front-loads information so a truncated prefix of the vector remains a usable embedding.

**11. B** — Public datasets may be contaminated, the benchmark is general while your corpus is specific, and an average hides that a model may be strong at classification and weak at retrieval.

**12. B** — Fine-tuning changes the vector space, so existing embeddings are no longer comparable to new ones.

**13.** Contrastive learning trains a model to arrange texts in a geometry rather than to generate anything: **pull related texts together and push unrelated texts apart.** The training unit is a triple — an **anchor** (for example a query, "How do I reset my password?"), a **positive** that should end up nearby (the actual reset instructions), and one or more **negatives** that should end up far away (an unrelated refund policy). The loss, typically **InfoNCE**, is a softmax over similarities: it maximises the similarity between anchor and positive *relative to* the similarities between the anchor and all the negatives. That relative framing is the important part — the model is not asked to hit an absolute similarity value, only to rank the positive above the negatives, which is exactly what retrieval needs. Practically, negatives usually come from **in-batch negatives**: with a batch of N (anchor, positive) pairs, every other pair's positive serves as a negative, giving N−1 negatives per anchor at no labelling cost — which is why embedding models train with unusually large batches.

**14.** In-batch negatives are overwhelmingly **easy** — a randomly chosen unrelated document is obviously irrelevant, so the model can separate it with almost no learning and gains little signal from the comparison. **Hard negatives** are texts that are superficially similar but genuinely not relevant: against "How do I reset my password?", the passage "To change your username, go to Account Settings" shares the domain, the register, and much of the vocabulary while being the wrong answer. Learning to separate *those* is what actually improves retrieval, because those are the documents that will compete for the top-k slots at query time. Mining them — retrieving top-k with the current model and taking the incorrect results — is the single biggest lever on embedding quality. **The Week 14 connection** is direct: that lecture distinguished easy preference pairs (where the quality gap is obvious) from hard ones (where the rejected response is factually impeccable and loses only on accessibility), and observed that **the hard pairs teach the most** because preferences are about **boundaries, not centres**. Contrastive learning is the same claim in a different setting — easy negatives confirm what the model already knows, hard negatives locate the actual decision boundary.

**15.** The constraint is **precomputation**, not accuracy preference. A **bi-encoder** encodes the query and the document through separate passes, so every document in the corpus can be embedded **once, at ingestion**, and stored; a query then costs one embedding plus a vector search, which scales to millions of documents. A **cross-encoder** feeds the query and document through the network **together**, so attention can align specific query terms against specific document spans — but that means no document representation exists until the query arrives, and nothing can be stored in advance. Scoring a million-document corpus would require a million transformer forward passes **per query**, which is not slow but infeasible. Conversely the bi-encoder's weakness is also structural: the document must be compressed to a fixed vector **before the query is known**, so the encoder has to guess in advance what future queries might care about, and information it discards is unrecoverable. Neither limitation can be engineered away, which is why the two-pass architecture — cheap recall over everything, expensive precision over a shortlist of tens — is forced rather than merely conventional. **Late interaction (ColBERT)** occupies the middle by storing a vector per token, trading storage for some of the cross-encoder's alignment ability.

**16.** **Symmetric search** compares two things of the same kind — two questions for duplicate detection, two documents for near-duplicate clustering — so a single embedding space treating both inputs identically is correct. **Asymmetric search** compares a **short query** against a **long document**, which are different in length, register, and grammar: "What's the stipend?" is interrogative, terse, and sparse, while the passage answering it is declarative, detailed, and full of domain vocabulary. Embedding both with the same treatment compares an interrogative vector against declarative ones, and the mismatch costs recall. This is why many embedding models take **instruction prefixes** that differ for queries and documents, and why getting them wrong degrades retrieval silently. **HyDE attacks the same mismatch at the pipeline level:** it asks an LLM to generate a hypothetical *answer* to the query and embeds that instead of the query, converting an asymmetric query→document comparison into a symmetric document→document one. That reframing also explains Week 9's otherwise puzzling observation that HyDE works even when the hypothetical answer is factually wrong — correctness is irrelevant, because the hypothetical is a **retrieval probe** whose job is to be *stylistically* document-like, not to be true.

**17.** Embedding models are trained on **semantic similarity**, so the geometry they learn encodes what a text is *about*. Every date is semantically "a date"; every error code is "an error code"; `RS256` and `HS256` are both "a cryptographic algorithm identifier." The information that distinguishes them is the **exact character sequence**, which is precisely what compression into a few hundred dimensions of semantic space throws away — the model is doing its job correctly, and its job is the wrong one for this query type. Week 9 demonstrated it empirically: the correct chunk for "What changed on 2024-09-15?" was retrieved at distance 0.551, barely separated from irrelevant chunks at 0.719 and 0.745. **The fix is architectural because no better embedding model solves it** — the failure follows from the objective, not from insufficient training. Hence **hybrid search**: BM25 matches exact strings by term statistics and has no notion of meaning at all, which makes it the exact complement to dense retrieval, and RRF fuses the two ranked lists. This also explains Week 11's observation that **code is close to the worst case for embeddings** and the best case for grep, since a function name is an exact identifier rather than a concept to approximate.

**18.** Dimension trades **expressiveness against storage and speed**. Smaller vectors (384) are cheap to store and fast to search but capture less nuance; larger ones (1536, 3072) capture more but cost proportionally more memory and comparison time, with diminishing returns at the top end. The arithmetic is not academic: one million documents at 1536 dimensions and 4 bytes per float is roughly **6 GB of vectors alone**, before the index structure — so dimension is an infrastructure decision, and quadrupling it quadruples the bill. **Matryoshka Representation Learning** changes the trade-off by training the model so information is **front-loaded**: the first 256 dimensions form a usable embedding on their own, the first 512 a better one, and the full vector the best. This enables **two-stage retrieval from a single embedding** — search cheaply over truncated 256-dimensional vectors to get a candidate set, then re-rank those candidates using the full-length vectors — capturing most of the speed and memory benefit of a small model with most of the accuracy of a large one, and requiring no second model. **The essential caveat** is that this only works on models explicitly trained for it: truncating an ordinary embedding degrades it badly, because information is distributed across all dimensions rather than concentrated in the prefix.

**19.** Fine-tuning wins when the distinctions that matter in your domain are ones a general model **never learned**. General embedding models place vectors according to general semantics, so in finance "liquidity" and "solvency" sit close together as near-synonyms, when technically they mean different things and a query about one should not retrieve the other. A bigger general model is still a *general* model — it has more capacity applied to the same broad distribution, so it does not necessarily separate your domain's fine distinctions any better. Fine-tuning reshapes the space around exactly those distinctions, and a few thousand good pairs typically produces meaningful improvement at low cost. **Where the pairs come from** — most products already have them: **search logs with clicks**, where the clicked result is the positive; **support tickets** paired with the article that resolved them; **Q&A pairs** from documentation; and **synthetic pairs** generated by prompting an LLM to write questions for each chunk, which is Week 13's synthetic-data pipeline pointed at retrieval instead of instruction tuning. **Then mine hard negatives** by retrieving top-k with the current model and treating the wrong results as negatives — the step that produces most of the gain. **The cost to plan for** is that fine-tuning changes the vector space, so the entire corpus must be **re-embedded**, and the index must be versioned against the model that produced it.

**20.** **Diagnose before changing anything, and inspect retrieved chunks rather than final answers** — Week 9's central debugging lesson, since the LLM papers over bad retrieval. Build a small eval set of real wiki questions with known correct chunks and measure **recall@k**, so every subsequent change is measurable (S3). **First, check the cheap structural causes.** Is the chunk size larger than the embedding model's **max sequence length**? If so, content past the limit is silently truncated and invisible to retrieval — a total failure that looks like a quality problem. Are **instruction prefixes** being used correctly for queries versus documents, or omitted on a model that expects them? Both are silent, both are one-line fixes, and both are common. Then check the **ANN search-effort parameter** (`ef_search`/`nprobe`): a too-low value drops recall in a way that looks exactly like a bad embedding model, and comparing against exact search on a sample will tell you immediately. **Second, classify the failures.** An engineering wiki is dense with **exact identifiers** — error codes, config flags, function names, ticket IDs, version numbers — which is the known worst case for dense retrieval. If failures cluster there, the fix is **hybrid search with BM25 and RRF**, and that is likely the single largest win available. If failures are instead cross-document or context-dependent ("which service owns this?"), the fix is **contextual retrieval** — prepending LLM-generated situating context before embedding — which Week 9 reported at 67% fewer failures. **Third, improve chunking**, which Week 9 called the place 80% of RAG systems fail: wiki pages have strong heading structure, so structure-aware chunking on section boundaries beats fixed-size splitting, and parent-child (embed small, retrieve large) suits reference material well. **Fourth, add reranking** — a cross-encoder over the top 10–20 candidates, which is the highest-value precision upgrade once recall is adequate. **Only then consider the embedding model itself**: try a domain fine-tune using search logs, ticket-to-article pairs, or synthetic questions per chunk with mined hard negatives, remembering it requires re-embedding everything. **Throughout, evaluate over the query set rather than individual examples** — Week 9's own hybrid-search demo improved on average while ranking one query *worse*, which is exactly why single-example testing misleads.

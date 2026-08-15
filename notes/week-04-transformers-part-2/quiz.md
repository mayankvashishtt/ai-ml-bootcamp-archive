# Week 4 — Quiz (20 questions)

**Topic:** Transformers Part 2 — Self-Attention, KV Cache, Multi-Head Attention
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** In the Q/K/V framework, the Key vector answers which question?
- A) "What am I looking for?"
- B) "What do I contain?"
- C) "What information do I offer?"
- D) "Where am I in the sequence?"

**2.** The attention formula is:
- A) `softmax(QK ᵀ · √d_k) V`
- B) `softmax(QKᵀ / √d_k) V`
- C) `softmax(QVᵀ / √d_k) K`
- D) `QKᵀV / softmax(d_k)`

**3.** Dividing by √d_k prevents:
- A) Negative attention weights
- B) Dot products growing large enough to saturate softmax and kill gradients
- C) The output dimension changing
- D) Attention weights summing to more than 1

**4.** Attention is best described as which kind of lookup?
- A) A hard lookup returning one exact match
- B) A soft lookup returning a weighted mixture across all keys
- C) A hash-table lookup with collision resolution
- D) A binary search over sorted keys

**5.** In the notebook, each row of the attention weight matrix sums to:
- A) 0
- B) 1
- C) d_k
- D) The sequence length

**6.** Q, K, and V are produced by:
- A) Three hand-designed transformations
- B) Multiplying the embedding by three separate learned matrices W_Q, W_K, W_V
- C) Slicing the embedding into three equal parts
- D) Three different tokenizers

**7.** During generation, which of these must be recomputed at every step?
- A) K for all previous tokens
- B) V for all previous tokens
- C) Q for the new token
- D) All of Q, K, V for every token

**8.** The KV cache stores:
- A) Queries for all tokens
- B) Keys and Values for previously processed tokens
- C) The final output logits
- D) The tokenizer vocabulary

**9.** "Time to first token" is slower than subsequent tokens because:
- A) The model loads from disk on the first token
- B) The entire prompt must be processed and the KV cache filled
- C) The first token requires more sampling steps
- D) Network latency only affects the first token

**10.** With `d_model = 768` and 8 heads, each head's Q/K/V dimension is:
- A) 768
- B) 96
- C) 8
- D) 6144

**11.** Which is NOT given as a reason a single attention head is insufficient?
- A) Syntactic relationships like subject → verb
- B) Coreference like pronoun → referent
- C) The need to process tokens sequentially
- D) Topical relationships between concepts

**12.** In the notebook's toy example, all attention weights came out near 0.33 because:
- A) The implementation has a bug
- B) The embeddings and projection matrices are random, not trained
- C) Softmax always produces uniform distributions
- D) The sequence was too short for attention to work

---

## Short answer

**13.** Define Query, Key, and Value, and explain how they interact to produce an attention weight.

**14.** Walk through all four steps of `softmax(QKᵀ/√d_k)V`, giving the matrix shape at each step for a 3-token sequence with d_k = d_v = 4.

**15.** Explain the database analogy for attention, contrasting hard and soft lookup.

**16.** Using "The bank was closed," explain precisely how self-attention resolves the ambiguity that static embeddings could not.

**17.** Explain the KV cache: what is redundant without it, what gets stored, and why Q is not cached.

**18.** Explain why long context is memory-expensive. Include the layer-count factor.

**19.** Explain multi-head attention: the problem it solves, how dimensions are split, and the trade-off in choosing head count.

**20.** A colleague sees uniform ~0.33 attention weights in the toy notebook and concludes attention doesn't work. Correct them, and say what a trained model's weights would look like.

---
---

## Answer key

**1. B** — The Key answers "What do I contain?" It is the label other tokens match against. (Query = "What am I looking for?", Value = "What information do I offer?")

**2. B** — `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`.

**3. B** — Dot products grow with dimensionality; large inputs make softmax saturate (one weight ≈ 1, the rest ≈ 0), which drives gradients toward zero and stalls learning.

**4. B** — A soft lookup: partial matches against **all** keys, returning a mixture of values weighted by match quality, rather than a binary hit/miss.

**5. B** — 1. Softmax over each row makes it a probability distribution, so every token distributes a fixed attention budget.

**6. B** — Each embedding is multiplied by three separate **learned** matrices. The model discovers useful projections during training; only the structure is designed.

**7. C** — Only the new token's Query. Keys and Values for previous tokens are unchanged, which is exactly what makes caching possible.

**8. B** — Keys and Values for tokens already processed.

**9. B** — The first token requires *prefill*: processing the whole prompt, computing K and V for every prompt token, and filling the cache. Later tokens reuse that cache and only compute for the new token.

**10. B** — 96, since 768 ÷ 8 = 96. Heads split the dimension rather than multiplying it.

**11. C** — Sequential processing is not a limitation of single-head attention; transformers are parallel by design. The other three are genuine relationship types one head cannot capture at once.

**12. B** — Random embeddings and random projection matrices produce near-uniform similarity scores. The demo illustrates mechanism, not learned behaviour.

**13.** **Query** is a token's search criteria — what it is looking for in the rest of the sequence. **Key** is a token's advertised label — what it contains, used by other tokens to judge relevance. **Value** is the content a token actually contributes once matched. All three are learned linear projections of the same embedding. **Interaction:** each Query is compared against every Key by dot product, producing similarity scores; those scores are scaled by √d_k and passed through softmax to become attention weights; each weight then determines how much of the corresponding Value is mixed into the output. Similarity between Q and K decides *how much*, and V supplies *what*.

**14.** (i) **Project:** `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V` → Q, K, V each **(3, 4)**. (ii) **Score:** `Q @ K.T` → **(3, 3)**, one score per query–key pair. (iii) **Scale and normalise:** divide by √d_k = 2, then softmax over each row → attention weights **(3, 3)**, rows summing to 1. (iv) **Mix:** `attention_weights @ V` → output **(3, 4)** — one contextualised vector per token.

**15.** A **hard lookup** in a traditional database matches a key exactly and returns a single value; the outcome is binary — either a row matches or it does not. **Soft lookup (attention)** compares the query against *every* key, scores how well each matches, and returns a **weighted mixture of all values** rather than one. Nothing is excluded; items merely contribute in proportion to relevance. This matters because the operation is **differentiable** — a hard match has no useful gradient, whereas a soft weighted mixture can be trained end to end. Softmax is what turns raw similarities into the mixing proportions.

**16.** In isolation, "bank" has one static embedding representing a **blended average** of its financial and geographic senses, so both readings are conflated and neither is well represented. With self-attention, the representation of "bank" is **recomputed from its context**: "bank" issues a Query, which is matched against the Keys of every surrounding token, and tokens like "closed," "money," or "loan" score highly while river-related tokens do not. The output for "bank" is then a weighted mixture of the Values of the tokens it attended to, so it absorbs financial features specifically. The same token in "the bank of the river" attends to different neighbours and yields a different vector — one token, two genuinely different context-dependent representations.

**17.** **What's redundant:** generation runs attention over the whole sequence at every step, recomputing K and V for *every* previous token each time — but those tokens haven't changed, so their Keys and Values are identical to the previous step. That work is pure waste and grows with sequence length. **What's stored:** the K and V vectors for every token already processed, at every layer. Each step computes K and V only for the newly added token and appends them to the cache. **Why Q isn't cached:** the Query is only ever needed for the token currently being generated — old tokens' queries are never reused, because attention at step *n* asks what the *new* token is looking for. So Q is always computed fresh and discarded.

**18.** The cache must hold a K and a V vector for **every token, at every layer**. Memory therefore scales as roughly *tokens × layers × 2 × head_dim × heads*, growing **linearly with context length** and multiplied by depth. With GPT-4 at ~120 layers, a 128K-token context means storing keys and values for 128,000 positions across 120 layers — a massive footprint that often exceeds the model weights themselves. Because this memory is per-request rather than shared, it limits how many concurrent users a GPU can serve, which is why long context is expensive and why context windows are capped at all.

**19.** **Problem:** a single head has one set of Q/K/V weights and therefore one perspective, but language contains many simultaneous relationship types — syntactic (subject→verb), semantic (pronoun→referent), positional (nearby words), and topical (related concepts). One head must compromise between them. **Solution:** run several attention mechanisms in parallel, each with its own learned projections, then concatenate the outputs and project back to `d_model`. **Dimension split:** heads *partition* rather than duplicate the representation — 8 heads × 96 dims = 768 — so the cost is roughly that of single-head attention at the same width. **Trade-off:** more heads give more perspectives but shrink each head's dimension, and too-small heads cannot represent anything useful; too few heads give too little diversity. Typical practice is 8–32 heads.

**20.** The uniformity is expected, not a defect: the notebook uses **random** embeddings and **random** projection matrices, so no Query has any learned reason to prefer one Key over another and all similarity scores come out near-identical — softmax over near-identical scores yields a near-uniform distribution. The code is demonstrating the *mechanism* (project → score → scale → softmax → mix), and its correctness is independently confirmed where the manual weighted sum for "cat" exactly reproduces the matrix result. **In a trained model,** `W_Q`, `W_K`, and `W_V` are learned so that related tokens produce high dot products, giving **sharply peaked** rows — most weight concentrated on a few genuinely relevant tokens (a pronoun attending strongly to its antecedent, a verb to its subject), often with substantial self-attention, and near-zero weight on irrelevant tokens.

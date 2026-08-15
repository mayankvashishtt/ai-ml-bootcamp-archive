# Week 7 — Quiz (20 questions)

**Topic:** Training Your First Model — building MiniLLM from scratch
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** In `get_batch()`, `y` is constructed as:
- A) A random unrelated chunk
- B) The same chunk as `x` shifted right by one character
- C) The reverse of `x`
- D) A one-hot encoding of `x`

**2.** A single training sequence of length N provides how many training examples?
- A) 1
- B) 2
- C) N
- D) N²

**3.** The causal mask exists to prevent:
- A) Attention weights from summing to more than 1
- B) A token from attending to future positions and seeing the answer
- C) Gradient explosion
- D) Overfitting to the training set

**4.** Masked positions are set to `-inf` rather than 0 because:
- A) 0 would cause a division error
- B) The mask is applied before softmax, and `e^(-inf) = 0`
- C) `-inf` is faster to compute
- D) It preserves the sign of the scores

**5.** RoPE in this implementation is applied to:
- A) Q, K, and V
- B) Q and K only
- C) V only
- D) The token embeddings before the blocks

**6.** With 8 query heads and 2 KV heads, the `k_proj` layer maps:
- A) 256 → 256
- B) 256 → 64
- C) 64 → 256
- D) 256 → 680

**7.** `expand()` inside `repeat_kv()` is preferred because it:
- A) Runs only on GPU
- B) Creates a view without copying the underlying data
- C) Automatically applies RoPE
- D) Sorts the heads by relevance

**8.** Weight tying in `MiniLLM` means:
- A) All transformer blocks share weights
- B) The token embedding and the LM head share the same weight matrix
- C) Q and K projections share weights
- D) Dropout is applied to the weights

**9.** For a 65-character vocabulary, the expected initial loss is approximately:
- A) 0.0
- B) 1.0
- C) 4.17
- D) 65.0

**10.** `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)` prevents:
- A) Vanishing gradients
- B) Exploding gradients
- C) Overfitting
- D) Dropout during evaluation

**11.** In `generate()`, `logits[:, -1, :]` is taken because:
- A) The first position is always padding
- B) Only the last position predicts the actual next token
- C) It reduces memory usage
- D) The model outputs are reversed

**12.** Raising the temperature from 0.3 to 1.5:
- A) Sharpens the distribution toward the top choice
- B) Flattens the distribution, making output more random
- C) Has no effect on sampling
- D) Disables sampling entirely

---

## Short answer

**13.** Explain why `y` is `x` shifted by one, and why this makes language-model pretraining scale so well.

**14.** Draw the 4×4 causal mask and explain what would go wrong during training without it.

**15.** Explain why RoPE is applied to Q and K but not V.

**16.** Explain exactly where GQA's memory saving comes from in this implementation, referencing the projection shapes.

**17.** Explain weight tying: what is shared, why it is sensible, and what two benefits it gives.

**18.** Why should the initial loss equal `ln(vocab_size)`? Explain the derivation and how you'd use it to debug.

**19.** Contrast the two sublayers in a transformer block in terms of how information moves, and explain the pre-norm residual pattern `x = x + sublayer(norm(x))`.

**20.** The `generate()` function has no KV cache. Explain what this costs and how you would add one.

---
---

## Answer key

**1. B** — `y` is the same chunk shifted right by one, so the target at each position is the next character. `x = "To be o"` → `y = "o be or"`.

**2. C** — N. Every position is a supervised example simultaneously.

**3. B** — Without it, position 3 could attend to position 4, which *is* the target, and the model would learn to copy rather than predict.

**4. B** — The mask is applied to the scores **before** softmax; since `e^(-inf) = 0`, masked positions get exactly zero weight while the remaining weights still normalise to 1.

**5. B** — Q and K only. Position affects matching, not content.

**6. B** — 256 → 64, i.e. `n_kv_heads (2) × head_dim (32)`. Four times smaller than `q_proj`.

**7. B** — `expand()` produces a view with stride 0 on the repeated axis, so no data is duplicated in memory — which is what preserves the actual 4× saving.

**8. B** — `self.lm_head.weight = self.token_emb.weight`.

**9. C** — ln(65) ≈ 4.17.

**10. B** — Exploding gradients, which transformers are prone to; it rescales gradients whose total norm exceeds 1.0.

**11. B** — The model produces a prediction at every position, but during generation only the final position's distribution corresponds to the next token to emit.

**12. B** — Higher temperature flattens the distribution (91/6/2/1 at T=0.3 versus 33/27/22/18 at T=1.5), producing more varied, less predictable output.

**13.** `y` is `x` shifted by one so that at every position the target is simply the **next character**: given `T` predict `o`, given `To` predict `␣`, given `To␣` predict `b`, and so on. This means one sequence of length N yields **N training examples** in a single forward pass, and — crucially — **no human labelling is required**, since the labels come from the text itself. That is what makes pretraining scale: any raw text corpus is instantly a supervised dataset, so the approach extends to trillions of tokens without annotation cost, and each forward pass extracts N supervision signals rather than one.

**14.**
```
      T0  T1  T2  T3
T0    ✓   ✗   ✗   ✗
T1    ✓   ✓   ✗   ✗
T2    ✓   ✓   ✓   ✗
T3    ✓   ✓   ✓   ✓
```
Each token attends only to itself and earlier tokens (lower triangle including the diagonal). **Without it**, training would be worthless: because targets are the inputs shifted by one, the answer for position *i* sits at position *i+1* in the very same sequence. An unmasked model would learn the trivial solution of attending to the next position and copying it, achieving near-zero training loss while learning nothing about language — and would then fail completely at generation, where future tokens do not yet exist.

**15.** RoPE encodes position by rotating vectors so that **the dot product between a rotated Q and a rotated K depends only on their relative distance**. Position therefore needs to enter only where dot products are computed — the Q·K score that decides *how much* one token attends to another. **V is not involved in any dot product**; it carries the *content* that gets mixed into the output once attention weights are decided. Rotating V would corrupt that content with positional information for no benefit, distorting what is passed along rather than how strongly it is selected. The clean split is: **position affects matching, not content.**

**16.** The saving is in the **projection output sizes**. `q_proj` maps 256 → 256 (8 heads × 32 dims), because every query head needs its own unique query. But `k_proj` and `v_proj` map 256 → **64** (2 heads × 32 dims) — four times smaller — because only two KV heads exist. Consequently only two heads' worth of K and V are ever computed, stored, or cached, which is a 4× reduction in KV cache memory. `repeat_kv()` then expands `[b, 2, seq, 32]` to `[b, 8, seq, 32]` so shapes match for the batched matmul, but it uses `expand()`, which creates a **stride-0 view rather than copying data** — so the expansion is free and the memory saving is preserved.

**17.** **What is shared:** the token embedding matrix `token_emb.weight` and the output projection `lm_head.weight` are literally the same tensor. **Why it is sensible:** the embedding maps token → vector while the LM head maps vector → per-token scores; they are inverse operations over the same vocabulary in the same representation space, so a token's input and output representations ought to agree. **Benefits:** (i) it removes an entire `vocab_size × d_model` parameter matrix, a large saving when vocabularies run to 100k entries; (ii) it typically *improves* quality by forcing consistency between how a token is read and how it is predicted, acting as a regulariser and giving each token's vector twice the gradient signal.

**18.** An untrained model has no reason to prefer any character, so it should output a roughly uniform distribution assigning probability **1/V** to each of V classes. Cross-entropy loss is `−ln(p_correct)`, which for `p = 1/V` gives `−ln(1/V) = ln(V)`. With V = 65 that is **ln(65) ≈ 4.17**. **As a debugging tool** this is exceptionally cheap: run one forward pass before training and compare. A much *higher* initial loss means the model is confidently wrong — usually bad initialisation with weights scaled too large. A much *lower* one means it is already confident, suggesting label leakage (for instance a missing causal mask) or a bug in the loss. Either way you catch a fundamental wiring error before spending 20 minutes of GPU time.

**19.** **Attention mixes information *between* tokens** — every token gathers a weighted combination of other tokens' values, moving information sideways across the sequence. **The FFN processes each token independently** — the same SwiGLU is applied position-wise with no cross-token interaction, transforming each representation in place. So a block alternates *gather context* and *think about it*. **The pattern `x = x + sublayer(norm(x))`** combines pre-norm and residual: `norm(x)` stabilises the sublayer's input, the sublayer computes a **change** rather than a replacement, and `x +` adds that change back. The consequence is that the sublayer only has to learn a *delta*, which is easier than reproducing the whole representation, and the addition leaves an unnormalized identity path from input to output down which gradients flow freely — which is precisely what makes deep stacks trainable.

**20.** **The cost:** each generation step passes the entire context through all four blocks again, recomputing K and V for every previous token even though those values are unchanged. Work per step grows with context length, so producing n tokens is roughly **O(n²)** instead of O(n), and generation slows noticeably as output lengthens. For a 256-token teaching model this is tolerable; in production it is unacceptable. **How to add one:** maintain a per-layer buffer of K and V tensors. On the first call (prefill), run the full prompt and store each layer's K and V. On each subsequent step, feed **only the newly generated token**, compute its Q, K, and V, append the new K and V to that layer's buffer, and attend the single new query against the full cached K/V. Because only one query position is processed, the causal mask becomes unnecessary — the new token legitimately attends to everything already in the cache. Note the cache must store the **pre-`repeat_kv`** tensors (2 heads, not 8) to retain GQA's memory advantage, and it must be trimmed alongside the sliding window so it never exceeds `max_seq_len`.

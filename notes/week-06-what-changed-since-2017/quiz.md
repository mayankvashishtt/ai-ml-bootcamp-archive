# Week 6 — Quiz (20 questions)

**Topic:** What Changed Since 2017 — RMSNorm, SwiGLU, RoPE, GQA
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The four modern upgrades to the classic transformer are:
- A) BatchNorm, GELU, ALiBi, MQA
- B) RMSNorm, SwiGLU, RoPE, GQA
- C) LayerNorm, ReLU, Sinusoidal, MHA
- D) Dropout, Softmax, Learned position, Flash Attention

**2.** Compared with LayerNorm, RMSNorm removes:
- A) The scale parameter
- B) Mean subtraction and the bias/shift term
- C) The epsilon term
- D) Variance computation and scaling

**3.** In pre-norm architectures, normalization is applied:
- A) After the sublayer, as in the original transformer
- B) Before the sublayer, leaving the residual path unnormalized
- C) Only at the final output layer
- D) Only during inference

**4.** Pre-norm is described as essential for models with:
- A) More than 5 layers
- B) More than 30 layers
- C) Fewer than 10 layers
- D) Exactly 12 layers

**5.** SiLU is defined as:
- A) `max(0, x)`
- B) `x · σ(x)`
- C) `tanh(x)`
- D) `1 / (1 + e⁻ˣ)`

**6.** SwiGLU's structure consists of:
- A) A single ReLU path
- B) Two paths — one creating a gate via SiLU, one carrying the raw signal
- C) Three stacked linear layers
- D) An attention mechanism inside the FFN

**7.** The two stated problems with sinusoidal position encoding are:
- A) It is too slow and requires many parameters
- B) It is absolute, and does not naturally capture relative distance
- C) It cannot handle sequences under 100 tokens
- D) It requires GPU support

**8.** RoPE encodes position by:
- A) Adding a positional vector to the embedding
- B) Rotating query and key vectors according to position
- C) Concatenating a learned position embedding
- D) Multiplying the attention scores by position indices

**9.** In RoPE, the dot product between rotated vectors depends on:
- A) Absolute positions only
- B) Relative distance only
- C) The batch size
- D) The number of heads

**10.** For a 32-layer model at 8192 tokens, the total KV cache is approximately:
- A) 128 MB
- B) 4 GB
- C) 64 GB
- D) 512 GB

**11.** GQA reduces KV cache memory by roughly:
- A) 2×
- B) 4×
- C) 16×
- D) 100×

**12.** The final slide contrasts the two eras' focus as:
- A) Speed (2017) vs Accuracy (2024)
- B) Correctness (2017) vs Stability & Efficiency (2024)
- C) Research (2017) vs Products (2024)
- D) Small models (2017) vs Large models (2024)

---

## Short answer

**13.** Write out the operations in LayerNorm and RMSNorm side by side. What was removed, and what was the justification?

**14.** Write the `RMSNorm` forward pass in code and explain each term, including `eps`.

**15.** Explain why pre-norm enables much deeper models, connecting it to the vanishing gradient problem from Week 2.

**16.** Explain SwiGLU's gating and why SiLU's smoothness and non-monotonicity are advantages over ReLU.

**17.** Explain RoPE's core insight. Why is rotating Q and K a better fit for attention than adding position to embeddings?

**18.** Work through the KV cache calculation for a 32-layer model at 8192 tokens, explaining each factor, then explain what changes at 128K.

**19.** Explain GQA, including what is shared and what is *not*, and why that preserves quality.

**20.** All four upgrades are framed as stability/efficiency rather than capability improvements. Explain what that says about how progress in this field actually happens.

---
---

## Answer key

**1. B** — RMSNorm, SwiGLU, RoPE, and GQA, used by Llama, Mistral, Gemma, DeepSeek, and virtually every modern open-weight LLM.

**2. B** — Mean subtraction and the bias (shift) term. RMSNorm keeps only RMS normalization and a learned scale.

**3. B** — Before the sublayer, so the residual connection carries the raw, unnormalized signal.

**4. B** — 30+ layers.

**5. B** — `SiLU(x) = x · σ(x)`, also known as Swish.

**6. B** — Two paths: one produces a gate via SiLU, the other carries the raw signal, and they are combined so the gate controls information flow.

**7. B** — It is absolute (encoding "I am token #47") and does not naturally capture how far apart two tokens are.

**8. B** — By rotating the query and key vectors based on their position in the sequence.

**9. B** — Relative distance only. This is RoPE's defining property.

**10. B** — ~4 GB (about 128 MB per layer × 32 layers). 128K tokens would require roughly 64 GB.

**11. B** — 4× less KV cache memory, with almost no drop in model quality.

**12. B** — Correctness (2017) versus Stability & Efficiency (2024).

**13.** **LayerNorm:** Mean → Variance → Normalize → Scale → Shift. **RMSNorm:** RMS → Normalize → Scale. **Removed:** mean subtraction (centering) and the bias/shift term, leaving one learnable parameter instead of two. **Justification:** *"Centering wasn't earning its keep"* — RMSNorm deliberately maintains a non-zero mean and controls only the *scale* of activations, and empirically this stabilises training just as well while being simpler and faster. Since normalization runs twice per layer across dozens of layers, the saving compounds, which matters when scaling to billions of parameters.

**14.**
```python
def forward(self, x):
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return (x / rms) * self.weight
```
`x.pow(2)` squares each element; `.mean(-1, keepdim=True)` averages over the **feature dimension** (the last axis), keeping the dimension so the result broadcasts back against `x`; `torch.sqrt` takes the root, completing root-mean-square. Dividing `x` by `rms` rescales the vector to unit RMS magnitude without shifting its mean. `self.weight` is a learned per-dimension scale initialised to ones, letting the model recover any scaling it needs. **`eps`** (1e-6) is added inside the square root to prevent division by zero when an activation vector is all zeros, and to avoid numerical instability for very small magnitudes.

**15.** With **post-norm**, every sublayer's output — including the residual path — passes through a normalization operation, so on the backward pass gradients are repeatedly rescaled as they travel toward earlier layers. Stacking dozens of such transformations compounds the effect, and gradients reaching early layers become unreliably small or distorted — the vanishing gradient problem from Week 2, where multiplying many small factors drives the signal toward zero. With **pre-norm**, normalization is moved *inside* the block before the sublayer, leaving the **residual connection as a clean, unnormalized identity path** from input to output. Gradients can flow backward along that path essentially unimpeded regardless of depth. This is why 30+ layer models train reliably under pre-norm, whereas post-norm transformers past roughly 20 layers were notoriously fragile and required careful learning-rate warmup.

**16.** **Gating:** SwiGLU splits the feed-forward computation into two parallel paths — one passed through SiLU to produce a **gate**, the other carrying the raw linear signal — and multiplies them element-wise, so the gate controls how much of each feature is allowed through. Crucially the gate is **learned**, letting the network modulate information flow per feature and per token rather than applying one fixed non-linearity. **Smoothness:** SiLU is differentiable everywhere, unlike ReLU which has a kink at zero, giving better-behaved gradients during optimisation. **Non-monotonicity:** SiLU dips slightly below zero for small negative inputs instead of clamping flat, so those inputs still carry gradient — avoiding the "dying ReLU" failure where a unit stuck in the negative region receives zero gradient and never recovers.

**17.** **The insight:** attention uses positional information *only* through the Q·K dot product, so rather than injecting position into the input and hoping the model learns to use it, RoPE rotates Q and K by an angle determined by position such that **their dot product depends only on relative distance**. Relative position becomes a structural property of the operation instead of something inferred. **Why it beats addition:** sinusoidal encoding *adds* a position signal into the embedding, where it occupies the same dimensions as meaning and must be disentangled during training — and it encodes absolute indices, so relative distance must be learned indirectly. RoPE never perturbs the embedding's semantic content; it applies a rotation at the point where position is actually used. The clock analogy captures it: the angle between 3 and 5 equals the angle between 7 and 9 — only the gap matters, which is exactly what attention needs.

**18.** `2 × 32 heads × 8192 tokens × 128 head_dim × 2 bytes ≈ 128 MB per layer`. The leading **2** accounts for storing both **K and V**; **32 heads** is the number of attention heads each with its own K/V under MHA; **8192 tokens** is the sequence length, since every past token's K and V must be retained; **128 head_dim** is the size of each head's vector; **2 bytes** reflects fp16/bf16 precision. Multiplying by **32 layers** gives roughly **4 GB total**, because every layer maintains its own cache. **At 128K tokens**, only the token count changes — it grows 16× — so the cache scales linearly to roughly **64 GB**. That exceeds the total memory of most GPUs *before accounting for the model weights themselves*, which is precisely the "wall" described: the cache exhausts memory before the model starts generating.

**19.** In standard MHA, **every query head has its own dedicated Key and Value projections**, so the KV cache must store one K and one V per head per token per layer. **GQA shares K and V across groups of query heads** — several query heads read from the same K/V pair — cutting cache memory roughly 4× at typical group sizes. **What is not shared: the query heads themselves.** Every query head keeps its own `W_Q` and therefore its own perspective on the sequence, so the multi-head benefit from Week 4 — different heads capturing syntax, coreference, locality, topic — is preserved. Quality holds up because the diversity that matters most lives in the queries (what each head looks *for*), while the keys and values represent shared content that is largely redundant across heads. GQA sits between MHA (maximum quality and memory) and MQA (all heads sharing one K/V, minimum memory with some quality loss), and won because the quality-per-byte curve is very favourable at moderate group sizes.

**20.** It shows that progress here is often **engineering rather than invention**: the 2017 architecture was already capable in principle, and what blocked bigger, better models was that they could not be trained deep enough or served cheaply enough. Each upgrade removes a practical constraint — pre-norm and RMSNorm make deep stacks trainable, RoPE makes long relative context natural, GQA makes long context affordable to serve — and capability then arrives via **scale**, exactly as Week 1 described with emergent abilities. It also reveals a distinctive research style: RMSNorm was found by *deleting* a step and observing nothing broke, and GQA by *sharing* something previously duplicated. Much of the field's advance comes from discovering which parts of an accepted design were never load-bearing, which is why reading primary sources (Week 20) matters more than memorising any current architecture — today's four upgrades will themselves be superseded.

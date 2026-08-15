# Week 6 — What Changed Since 2017 (and Let's Build It)

**Subtitle:** Building & Training a Modern LLM
**Date:** 21/02/2026
**Source:** `downloads/week-06-what-changed-since-2017.pdf` (14 slides) — *no Colab notebook for this week*
**Notion page:** https://100xschool.notion.site/30effffa33e5803e911bdf08de015b5d

**Referenced links:**
- [Excalidraw board](https://excalidraw.com/#json=VcgqSnBC74dLEWeqhmFP0,THeOoMUaZIrRjNGuFCOvOg) — live diagrams from the session
- [syhya.github.io — Group Query Attention](https://syhya.github.io/posts/2025-01-16-group-query-attention/) — deep dive on GQA

---

## The premise

Weeks 3–4 built the **2017 transformer**. That architecture — token embedding + sinusoidal position, post-norm LayerNorm, multi-head attention, ReLU feed-forward — is **essentially GPT-2 (2019)**.

Modern open-weight LLMs still use that skeleton, but **four components were swapped out**. This week is the diff.

| Component | Classic (2017–2019) | Modern (2023–2025) |
|---|---|---|
| **Normalization** | LayerNorm | **RMSNorm** |
| **Feed-Forward** | ReLU FFN | **SwiGLU** |
| **Position Encoding** | Sinusoidal | **RoPE** |
| **Attention** | Multi-Head (MHA) | **Grouped Query (GQA)** |

> Used by **Llama, Mistral, Gemma, DeepSeek**, and virtually every modern open-weight LLM.

**The through-line, stated on the final slide:** the 2017 focus was **correctness** — proving attention works at all. The 2024 focus is **stability & efficiency** — making it trainable at 100+ layers and servable at scale. Every one of the four changes is a stability or efficiency win, not a capability win. That framing is the actual lesson of the week.

---

## Upgrade 1 — RMSNorm (replacing LayerNorm)

### What changed

```
LayerNorm:  Mean → Variance → Normalize → Scale → Shift
RMSNorm:    RMS  →            Normalize → Scale
```

**What's gone:** mean subtraction, and the bias term (shift).

> **Centering wasn't earning its keep.**

That's the whole justification, and it's a good example of how the field actually progresses — someone removed a step, found nothing broke, and the simpler version won.

### The benefit

- **Simpler and faster** — fewer operations per call, and normalization runs twice per layer across ~100 layers, so it adds up
- **Stabilizes training just as well** — no loss in stability despite the removal
- **Crucial for scaling** to billions of parameters

RMSNorm **maintains a non-zero mean** — it doesn't center the distribution at all. It only rescales by root-mean-square magnitude. The empirical finding is that *scale* control is what stabilises training; *centering* was incidental.

### The code

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Square, Mean, Root
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

Note there is **one** learnable parameter (`weight`) where LayerNorm has two (weight and bias). `eps` prevents division by zero.

---

## Pre-Norm vs Post-Norm

A separate change, orthogonal to *which* norm you use:

| | Placement |
|---|---|
| **Original (Post-Norm)** | Normalize **after** the sublayer |
| **Modern (Pre-Norm)** | Normalize **before** the sublayer |

**Why it matters:** *essential for deep models with 30+ layers.* With pre-norm, the **residual connection carries the raw signal** — it forms a clean, unnormalized path from input to output through the entire network, so gradients flow to early layers without being repeatedly rescaled.

This directly addresses the vanishing-gradient problem from Week 2. Post-norm transformers past ~20 layers were notoriously difficult to train and needed careful learning-rate warmup; pre-norm largely removed that pain and is what made very deep stacks routine.

---

## Upgrade 2 — SwiGLU (replacing the ReLU FFN)

> A smarter Feed-Forward Network that uses **learned gating** to control information flow.

**Two paths:**
- One path creates a **gate** using SiLU
- The other carries the **raw signal**

The two are multiplied element-wise, so the gate decides *how much of each feature passes through* — and the gating itself is learned.

**SiLU (Sigmoid Linear Unit), also called Swish:**

```
SiLU(x) = x · σ(x)
```

A **smooth, non-monotonic** alternative to ReLU's hard cutoff. Two properties matter:

- **Smooth** — differentiable everywhere, unlike ReLU's kink at zero. Better-behaved gradients.
- **Non-monotonic** — dips slightly below zero for small negative inputs rather than clamping flat to zero, so slightly-negative inputs still carry a gradient (no "dying ReLU").

This continues the activation lineage from Week 2: **ReLU → GELU → Swish/SiLU**, each smoother than the last, with SwiGLU adding the gating structure on top.

---

## Upgrade 3 — RoPE (replacing sinusoidal position encoding)

### The problem with sinusoidal positions

1. **Absolute encoding** — it encodes *"I am token #47"* rather than context-aware relationships.
2. **No relative distance** — it doesn't naturally capture *how far apart* two tokens are.

Week 3 claimed sinusoidal encoding makes relative position *learnable* (p+k is a linear function of p). True — but it must be **learned**, and it's added to the embedding where it competes for the same dimensions as meaning. RoPE makes relative distance **structural** instead.

### The idea

> Instead of **adding** position to embeddings, **rotate** the query and key vectors based on their position in the sequence.

**The key property:** the **dot product between rotated vectors depends only on their relative distance**, not their absolute positions.

This is elegant, because attention *only ever uses Q·K dot products*. If the dot product is naturally relative, then relative position is baked into the mechanism rather than inferred from a signal added to the input.

### The clock analogy

> Like hands on a clock: the angle between **3 and 5** is the same as between **7 and 9**. The gap is what matters.

### Mechanics

- **Dimension pairing** — for each pair of dimensions `(2i, 2i+1)`, apply a **2D rotation** by angle `θ_m`, based on the token's position
- **Frequency scaling** — **low dims rotate fast** (capturing nearby relationships); **high dims rotate slowly** (capturing distant relationships)

Same multi-scale principle as sinusoidal encoding — fast and slow frequencies covering short and long range — but applied as a *rotation of Q and K* rather than an *addition to the embedding*. Position never competes with meaning for representational space.

---

## Upgrade 4 — GQA (replacing multi-head attention)

### The memory bottleneck

The KV cache problem from Week 4, now quantified.

- **The KV cache** — during generation, all past Key and Value vectors are stored to avoid recomputation
- **Linear scaling** — memory grows linearly with sequence length *and* number of heads
- **The wall** — for long contexts, **the cache exceeds GPU memory before the model even starts "thinking"**

**The maths, for a 32-layer model:**

```
2 × 32 heads × 8192 tokens × 128 head_dim × 2 bytes
  ≈ 128 MB per layer
  ≈ 4 GB TOTAL CACHE
```

> *Just for 8K tokens. **128K would require 64 GB.***

Reading the formula: the leading `2` is K *and* V; `2 bytes` is fp16/bf16 precision. 64 GB for a single request's cache is more memory than most GPUs have in total — and that's *before* the model weights. This is the concrete reason long context was hard.

### The solution

> Instead of every query head having its own key and value, **Grouped Query Attention shares KV pairs across groups of query heads.**

**The sweet spot: 4× less KV cache memory than standard MHA, with almost no drop in model quality.**

The spectrum runs: **MHA** (every query head has its own K/V — maximum quality, maximum memory) → **GQA** (groups share K/V — the practical middle) → **MQA** (all heads share one K/V — minimum memory, some quality loss). GQA won because the quality/memory curve is very favourable at moderate group sizes.

Note what is *not* reduced: the number of **query** heads stays the same, so the model keeps its multiple perspectives (Week 4). Only the K/V projections are shared, and those are what the cache stores.

---

## The modern transformer block (2024 standard)

| Component | Role |
|---|---|
| **RMSNorm** | Pre-normalization for extreme training stability |
| **SwiGLU** | Gated feed-forward paths for richer representation |
| **RoPE** | Rotary embeddings for infinite relative context |
| **GQA** | KV-cache sharing for memory-efficient inference |

### Classic vs Modern

| Feature | Classic (2017) | Modern (2024) |
|---|---|---|
| **Normalization** | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| **Position** | Absolute Sinusoidal | Rotary (RoPE) |
| **Attention** | Multi-Head (MHA) | Grouped-Query (GQA) |
| **Activation** | ReLU | SwiGLU |
| **Focus** | **Correctness** | **Stability & Efficiency** |

---

## Key takeaways

1. **The 2017 skeleton survived.** Attention, residuals, and the overall block structure are unchanged — only four components were swapped.
2. **RMSNorm dropped mean-centering** because it "wasn't earning its keep." Simpler, faster, equally stable.
3. **Pre-norm beats post-norm** for deep stacks: the residual path carries the raw signal, so 30+ layers train reliably.
4. **SwiGLU adds learned gating** with a smooth, non-monotonic SiLU activation — better gradients than ReLU's hard cutoff.
5. **RoPE rotates Q and K instead of adding to embeddings**, making the dot product depend on *relative* distance structurally rather than by learned inference.
6. **The KV cache is the binding constraint on long context** — 4 GB at 8K tokens, ~64 GB at 128K for a 32-layer model.
7. **GQA shares K/V across groups of query heads**, cutting cache memory ~4× with almost no quality loss, while keeping all query heads.
8. **The theme is stability and efficiency, not raw capability.** Modern architecture research is largely about making the same idea trainable deeper and servable cheaper.

---

## Glossary

| Term | Meaning |
|---|---|
| **LayerNorm** | Normalizes using mean and variance, with learned scale and shift. |
| **RMSNorm** | Normalizes by root-mean-square only — no mean subtraction, no bias. |
| **Pre-Norm** | Normalizing *before* the sublayer, leaving the residual path unnormalized. |
| **Post-Norm** | The original placement, normalizing *after* the sublayer. |
| **Residual connection** | A skip path adding a sublayer's input to its output, easing gradient flow. |
| **FFN** | Feed-Forward Network — the per-token MLP inside each transformer block. |
| **SiLU / Swish** | `x · σ(x)` — a smooth, non-monotonic activation. |
| **SwiGLU** | A gated FFN variant: one SiLU path gates another linear path. |
| **Gating** | Multiplying a signal by a learned value that controls how much passes. |
| **RoPE** | Rotary Position Embedding — encodes position by rotating Q and K vectors. |
| **Relative position** | How far apart two tokens are, as opposed to their absolute indices. |
| **KV cache** | Stored Keys and Values for past tokens, avoiding recomputation during generation. |
| **MHA** | Multi-Head Attention — each query head has its own K and V. |
| **GQA** | Grouped Query Attention — groups of query heads share K and V projections. |
| **MQA** | Multi-Query Attention — all query heads share a single K and V. |
| **head_dim** | Dimensionality per attention head, e.g. 128. |

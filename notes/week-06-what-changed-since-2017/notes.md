# Week 6 — What Changed Since 2017 (and Let's Build It)

**Subtitle:** Building & Training a Modern LLM
**Date:** 21/02/2026
**Source:** `downloads/week-06-what-changed-since-2017.pdf` (14 slides) — *no Colab notebook for this week*
**Notion page:** https://100xschool.notion.site/30effffa33e5803e911bdf08de015b5d

**Referenced links:**
- [Excalidraw board](https://excalidraw.com/#json=VcgqSnBC74dLEWeqhmFP0,THeOoMUaZIrRjNGuFCOvOg) — live diagrams from the session
- [syhya.github.io — Group Query Attention](https://syhya.github.io/posts/2025-01-16-group-query-attention/) — deep dive on GQA

---

## 0. The idea in plain language

Weeks 3–4 built the **2017 transformer**. That architecture is essentially **GPT-2 (2019)**.

Modern LLMs still use that skeleton. Attention is unchanged. Residual connections are unchanged. The block structure is unchanged. **Four components were swapped**, and this week is the diff:

| Component | Classic (2017–2019) | Modern (2023–2025) | Why |
|---|---|---|---|
| **Normalization** | LayerNorm | **RMSNorm** | Simpler and faster, equally stable |
| **Feed-Forward** | ReLU FFN | **SwiGLU** | Learned gating, smoother gradients |
| **Position Encoding** | Sinusoidal | **RoPE** | Relative distance becomes structural |
| **Attention** | Multi-Head (MHA) | **Grouped Query (GQA)** | ~4× less KV cache memory |

> Used by **Llama, Mistral, Gemma, DeepSeek**, and virtually every modern open-weight LLM.

**The through-line, and the actual lesson of the week:** the 2017 goal was **correctness** — proving attention works at all. The 2024 goal is **stability and efficiency** — making it trainable at 100+ layers and servable at scale.

**Not one of these four changes makes the model smarter.** They make it cheaper to train and cheaper to run. That's what "architecture research" mostly means now, and it's a genuinely useful thing to understand about the field: the headline capability gains come from scale and data (S10, S11), while the architecture work quietly removes the constraints that would otherwise make that scale unaffordable.

> ⚠️ **A significant omission worth knowing about.** This deck's list of "what changed" doesn't include **Mixture-of-Experts (MoE)**, which is arguably the defining architectural change in current frontier models — it decouples how many parameters a model *has* from how many *run per token*. Everything in this lecture is an efficiency gain *within* a dense architecture; MoE leaves that frame entirely. **S10 covers it.**

---

## 1. Background you need first: residuals and normalization

*(Added — the deck uses both concepts throughout without ever explaining them, and two of the four upgrades are meaningless without them.)*

### Residual connections

Inside every transformer block, each sublayer is wrapped like this:

```
output = x + Sublayer(x)
```

That `x +` is a **residual connection** (or skip connection). Instead of the sublayer replacing its input, it computes an *adjustment* that gets **added** to the input.

**Why this matters enormously:** recall Week 2's vanishing gradient problem — gradients shrink as they multiply backward through layers, so early layers stop learning. A residual creates a path where the gradient flows through the `+` **completely unchanged**. Addition passes gradient through untouched. So no matter how deep the stack, there's always a clean highway from the loss back to the earliest layers.

**This is the single reason 100-layer networks are trainable at all.** Without residuals, the depth that makes modern models powerful would be unreachable — and it's why the idea survived unchanged from 2015 (ResNet) through every architecture since.

A second, softer benefit: each block only has to learn a small *refinement* to the representation rather than reconstruct it wholesale, which is an easier learning problem.

### Normalization

**The problem:** as data flows through many layers, the numbers can drift — growing enormous or shrinking to nothing. Enormous values saturate activations (Week 2 §10) and kill gradients; tiny values vanish. Either way, training destabilises.

**Normalization rescales the numbers back to a sane range at each step.** For a vector of activations:

```
LayerNorm:  subtract the mean, divide by the standard deviation,
            then apply a learned scale and shift
```

The learned scale and shift exist so the network can undo the normalization if it turns out to need to — you're standardising by default, not forcing it.

**Note this normalizes across the features of a single token**, independently for each token. That's what makes it "Layer" norm as opposed to "Batch" norm (which normalizes across the batch and behaves badly for variable-length sequences — which is why transformers use LayerNorm).

With those two concepts in place, the four upgrades make sense.

---

## 2. Upgrade 1 — RMSNorm (replacing LayerNorm)

### What changed

```
LayerNorm:  Mean → Variance → Normalize → Scale → Shift
RMSNorm:    RMS  →            Normalize → Scale
```

**What's gone:** mean subtraction, and the bias term (shift).

> **Centering wasn't earning its keep.**

That's genuinely the whole justification, and it's a nice example of how the field actually progresses: someone removed a step, found nothing broke, and the simpler version won on speed.

### Why it works

RMSNorm divides by the **root-mean-square** of the vector — a measure of magnitude — rather than by the standard deviation, and it doesn't subtract the mean first. So it **maintains a non-zero mean**; it doesn't centre the distribution at all.

The empirical finding is that **scale control is what stabilises training, and centering was incidental.** Keeping the numbers in a reasonable *magnitude* range is what prevents saturation and explosion; where that range sits relative to zero turns out not to matter much.

### The benefit

- **Simpler and faster** — fewer operations, and normalization runs **twice per layer across ~100 layers**, so a small per-call saving compounds into a real one
- **Just as stable** — no measured loss in training stability
- **One learned parameter instead of two** — a small parameter saving, repeated everywhere

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

Reading it: square every element, average them, take the square root — that's the RMS. Divide the vector by it, then apply the learned per-dimension scale. `eps` is a tiny constant preventing division by zero if the vector is all zeros. Note there's **one** learnable parameter where LayerNorm has two.

---

## 3. Pre-Norm vs Post-Norm

A separate change, **orthogonal to which norm you use** — it's about *where* you put it.

| | Placement |
|---|---|
| **Original (Post-Norm)** | `x + Sublayer(x)` → then normalize |
| **Modern (Pre-Norm)** | normalize → `x + Sublayer(Norm(x))` |

**Why it matters:** *essential for deep models with 30+ layers.*

With **pre-norm**, the normalization happens *inside* the residual branch, so **the residual path itself carries the raw, unnormalized signal** all the way from input to output. That gives gradients a completely clean highway through the entire network.

With **post-norm**, the normalization sits *on* the residual path, so the signal gets rescaled at every single layer on the way back. Through 50 layers that's 50 rescalings, and the gradient signal degrades.

**The practical history:** post-norm transformers past ~20 layers were notoriously difficult to train and needed careful learning-rate warmup schedules to converge at all. Pre-norm largely removed that pain and is what made very deep stacks routine. It's the same vanishing-gradient story from Week 2, solved by protecting the path rather than by changing the activation.

---

## 4. Upgrade 2 — SwiGLU (replacing the ReLU FFN)

> A smarter Feed-Forward Network using **learned gating** to control information flow.

### What the FFN is doing at all

Quick recap, since it's easy to forget amid all the attention machinery: each transformer block has **two** parts. Attention mixes information *between* tokens. The **feed-forward network** then processes each token *independently* — same small network applied to every position. Roughly: attention decides what's relevant, the FFN decides what to do with it. It's also where most of a transformer's parameters live.

### The gating idea

**Two parallel paths:**
- One path creates a **gate** using SiLU
- The other carries the **raw signal**

The two are multiplied element-wise, so the gate decides **how much of each feature passes through** — and the gating itself is learned.

The intuition is a set of learned valves. For each feature, the network can learn "let this through fully," "suppress this," or anything in between — and crucially the decision is *input-dependent*, computed fresh for each token rather than fixed.

**SiLU (Sigmoid Linear Unit), also called Swish:**

```
SiLU(x) = x · σ(x)
```

A **smooth, non-monotonic** alternative to ReLU's hard cutoff. Two properties matter:

- **Smooth** — differentiable everywhere, unlike ReLU's sharp kink at zero. Better-behaved gradients.
- **Non-monotonic** — it dips slightly *below* zero for small negative inputs rather than clamping flat, so slightly-negative inputs still carry a gradient. This avoids **"dying ReLU,"** where a neuron whose output is always negative gets zero gradient forever and is permanently dead.

This continues the activation lineage from Week 2: **ReLU → GELU → Swish/SiLU**, each smoother than the last, with SwiGLU adding the gating structure on top.

> **A caveat the deck doesn't mention:** SwiGLU needs *three* weight matrices where a plain FFN needs two (gate, up, down). Implementations compensate by shrinking the hidden dimension — commonly to ⅔ of what it would otherwise be — so parameter count stays comparable. If you ever wonder why modern FFN dimensions are odd numbers like 11008, this is why.

---

## 5. Upgrade 3 — RoPE (replacing sinusoidal position encoding)

### The problem with sinusoidal positions

1. **It's absolute.** It encodes *"I am token #47"* rather than any relationship.
2. **It competes for space.** Week 3 §8 noted position is *added* to the embedding, sharing the same dimensions as meaning. The model must learn to disentangle them.
3. **Relative distance is learnable, not structural.** Week 3 correctly noted that `p+k` is a linear function of `p`, so relative position *can* be learned — but "can be learned" is weaker than "is built in."

### The idea

> Instead of **adding** position to embeddings, **rotate** the query and key vectors according to their position.

**The key property: the dot product between two rotated vectors depends only on their *relative* distance**, not their absolute positions.

This is elegant, and here's why it's more than a trick: **attention only ever uses Q·K dot products** (Week 4 §3). Nothing else about Q and K matters. So if you arrange for the dot product to be naturally relative, relative position is baked into the *mechanism* rather than inferred from a signal bolted onto the input. Position stops competing with meaning for representational space entirely.

### The clock analogy

> Like hands on a clock: the angle between **3 and 5** is the same as between **7 and 9**. Only the gap matters.

Rotate both vectors by an amount proportional to their positions, and the *angle between them* — which is what the dot product measures — reflects only the difference in position. Absolute position cancels out.

### Mechanics

- **Dimension pairing** — for each pair of dimensions `(2i, 2i+1)`, apply a **2D rotation** by an angle determined by the token's position
- **Frequency scaling** — **low dimensions rotate fast** (capturing nearby relationships); **high dimensions rotate slowly** (capturing distant ones)

Same multi-scale principle as sinusoidal encoding — fast and slow frequencies covering short and long range — but applied as a *rotation of Q and K* rather than an *addition to the embedding*.

> **Why this matters for long context:** because the encoding is a rotation with well-understood mathematics, you can **interpolate** it to handle sequences longer than the model trained on — scaling the rotation frequencies so that a 4K-trained model can be extended to 32K or beyond with a short continued-pretraining phase. This "RoPE scaling" is how most long-context open models were actually produced. Sinusoidal encoding does not extend nearly as gracefully. (S10 §8 covers this.)

---

## 6. Upgrade 4 — GQA (replacing multi-head attention)

### The memory bottleneck

The KV cache problem from Week 4 §6, now quantified.

- **The KV cache** stores all past Key and Value vectors during generation to avoid recomputation
- **It scales linearly** with sequence length *and* number of heads *and* number of layers
- **The wall:** for long contexts, **the cache exceeds GPU memory before the model even starts "thinking"**

**The maths, for a 32-layer model:**

```
2 × 32 heads × 8192 tokens × 128 head_dim × 2 bytes
  ≈ 128 MB per layer
  ≈ 4 GB TOTAL CACHE
```

> *Just for 8K tokens. **128K would require ~64 GB.***

**Reading the formula:** the leading `2` is because you store both K *and* V. `2 bytes` is fp16/bf16 precision. And note this is **per request** — 64 GB is more than most GPUs have in total, *before* you've loaded any model weights. Serving ten concurrent users at long context is then flatly impossible.

**This is the concrete reason long context was hard**, and why it's priced the way it is.

### The solution

> Instead of every query head having its own Key and Value, **Grouped Query Attention shares K/V pairs across groups of query heads.**

Concretely, with 32 query heads and 8 KV groups, every 4 query heads share one set of K and V. The cache stores 8 sets instead of 32 — a **4× reduction**.

**The sweet spot: ~4× less KV cache memory with almost no drop in quality.**

The spectrum:

| | K/V per query head | Memory | Quality |
|---|---|---|---|
| **MHA** | Every head its own | Highest | Best |
| **GQA** | Groups share | Middle | Nearly as good |
| **MQA** | All heads share one | Lowest | Measurable loss |

GQA won because the quality/memory curve is very favourable at moderate group sizes — you get most of MQA's savings for almost none of its cost.

**Note what is *not* reduced: the number of query heads stays the same.** The model keeps all its distinct "perspectives" (Week 4 §7). Only the K/V projections are shared — and those are precisely what the cache stores. That's why the quality loss is small: you've reduced what's *stored*, not how many ways the model can *look*.

> **What came next:** DeepSeek's **MLA (Multi-head Latent Attention)** compresses K/V into a lower-dimensional latent space for a larger reduction still. Same problem, more aggressive solution. (S10.)

---

## 7. The modern transformer block (2024 standard)

| Component | Role |
|---|---|
| **RMSNorm (Pre-Norm)** | Training stability at extreme depth |
| **SwiGLU** | Gated feed-forward for richer representation |
| **RoPE** | Relative position, structurally, extensible to long context |
| **GQA** | KV-cache sharing for memory-efficient inference |

### Classic vs Modern

| Feature | Classic (2017) | Modern (2024) |
|---|---|---|
| **Normalization** | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| **Position** | Absolute Sinusoidal | Rotary (RoPE) |
| **Attention** | Multi-Head (MHA) | Grouped-Query (GQA) |
| **Activation** | ReLU | SwiGLU |
| **Focus** | **Correctness** | **Stability & Efficiency** |

### Which change targets what

Worth organising them by the problem they solve, because it makes the pattern obvious:

| Change | Fixes | Helps |
|---|---|---|
| RMSNorm | Wasted computation | Training speed |
| Pre-Norm | Vanishing gradients at depth | Training stability |
| SwiGLU | Hard gradients, dying units | Training quality |
| RoPE | Position competing with meaning; poor extrapolation | Long context |
| GQA | KV cache memory | **Inference** cost |

Note the last row is the only one about *serving* — and it's the one driven by the training/inference tension from S10 §1.

---

## Common confusions

**"Do these changes make the model smarter?"** No. Every one is a stability or efficiency win. Capability came from scale and data. This is the week's central point and it's easy to miss.

**"Is RMSNorm the same as removing normalization?"** No — it still rescales by magnitude, which is the part that matters. It only drops the mean-centering and the bias.

**"Pre-norm vs post-norm — is that about RMSNorm?"** No, they're independent choices. You could have post-norm RMSNorm or pre-norm LayerNorm. Modern models happen to use pre-norm RMSNorm.

**"Does RoPE replace the embedding?"** No. Embeddings still carry meaning (Week 3). RoPE only rotates Q and K *inside* the attention computation. Nothing is added to the embedding at all.

**"Does GQA reduce the number of attention heads?"** No — query heads stay the same. Only the K and V projections are shared across groups, and only those are cached.

**"Why is the KV cache per-request?"** Because it holds *that conversation's* tokens. Ten users means ten caches. This is why concurrency, not model size, is often the binding constraint on a serving GPU (S5).

**"Is the 2017 architecture obsolete then?"** No — the skeleton is intact. Attention, residuals, the block structure, and next-token prediction are unchanged. Four parts were swapped.

---

## Key takeaways

1. **The 2017 skeleton survived.** Attention, residuals, and the block structure are unchanged — only four components were swapped.
2. **Residual connections (`x + Sublayer(x)`) are why deep networks train at all** — addition passes gradient backward untouched, creating a clean highway to early layers.
3. **RMSNorm dropped mean-centering** because it wasn't earning its keep. Scale control is what stabilises training; centering was incidental.
4. **Pre-norm beats post-norm for deep stacks** because the residual path stays unnormalized, so gradients aren't rescaled at every layer.
5. **SwiGLU adds learned, input-dependent gating** with a smooth non-monotonic activation — better gradients than ReLU's hard cutoff, and no dying units.
6. **RoPE rotates Q and K instead of adding to embeddings**, making the dot product depend on *relative* distance structurally — and it extends to long context by interpolation.
7. **The KV cache is the binding constraint on long context** — ~4 GB at 8K tokens, ~64 GB at 128K for a 32-layer model, *per request*.
8. **GQA shares K/V across groups of query heads**, cutting cache memory ~4× with almost no quality loss, while keeping every query head.
9. **The theme is stability and efficiency, not capability.** Modern architecture research mostly removes the constraints that would make scale unaffordable.
10. **MoE is the big omission from this list** — see S10.

---

## Glossary

| Term | Meaning |
|---|---|
| **Residual / skip connection** | `x + Sublayer(x)`; passes gradient backward unchanged, enabling depth. |
| **Normalization** | Rescaling activations to a stable range so training doesn't diverge. |
| **LayerNorm** | Normalizes using mean and variance, with learned scale and shift. |
| **RMSNorm** | Normalizes by root-mean-square only — no mean subtraction, no bias. |
| **Pre-Norm** | Normalizing *inside* the residual branch, leaving the skip path raw. |
| **Post-Norm** | The original placement, normalizing after the residual addition. |
| **FFN** | Feed-Forward Network — the per-token MLP inside each transformer block. |
| **SiLU / Swish** | `x · σ(x)` — a smooth, non-monotonic activation. |
| **SwiGLU** | A gated FFN variant: one SiLU path gates another linear path. |
| **Gating** | Multiplying a signal by a learned, input-dependent value controlling how much passes. |
| **Dying ReLU** | A unit stuck always-negative, receiving zero gradient forever. |
| **RoPE** | Rotary Position Embedding — encodes position by rotating Q and K. |
| **RoPE scaling** | Interpolating the rotation frequencies to extend context beyond training length. |
| **Relative position** | How far apart two tokens are, as opposed to their absolute indices. |
| **KV cache** | Stored Keys and Values for past tokens, avoiding recomputation during generation. |
| **MHA** | Multi-Head Attention — each query head has its own K and V. |
| **GQA** | Grouped Query Attention — groups of query heads share K and V projections. |
| **MQA** | Multi-Query Attention — all query heads share a single K and V. |
| **MLA** | Multi-head Latent Attention — compresses K/V into a latent space (see S10). |
| **head_dim** | Dimensionality per attention head, e.g. 128. |

# S10 — Scale: Mixture-of-Experts, Distributed Training, and Scaling Laws

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 7 trains a MiniLLM on one GPU and Weeks 12–13 fine-tune with LoRA on one GPU. Week 6 covers what changed since 2017 — RMSNorm, SwiGLU, RoPE, GQA — but **Mixture-of-Experts isn't in that list**, and it is arguably the defining architectural choice of every current frontier model. Nothing explains how the models you call through an API are actually built, why "just make it bigger" stopped being the answer, or why a 7B model won't fit on a 24GB GPU for training when its weights are only 14GB.

**Fills the gap after:** Week 7 (Training Your First Model), Week 6 (What Changed Since 2017)
**Prerequisites:** Weeks 2, 5, 6, 7 (training loops, PyTorch, architecture, end-to-end training)

---

## 0. The idea in plain language

Three questions, and this lecture answers them in order.

**"How big should I make it?"** — There's an actual formula. Model quality follows a predictable curve as you add parameters, data, and compute, and for a fixed budget there's a *right* place to spend it. Getting this wrong wasted enormous amounts of money for several years (§2).

**"Why won't it fit?"** — Because training a model needs roughly **6–8× more memory than the weights alone**. Understanding where that multiplier comes from explains every distributed-training technique that follows (§3).

**"How do you train something that doesn't fit on any single GPU?"** — You split it. There are three fundamentally different ways to split, they solve different problems, and large runs use all three at once (§5–6).

Then the twist: **Mixture-of-Experts** breaks the assumption underlying all of it — that every parameter must participate in every token (§7).

---

## 1. Two different scaling questions

Keep these separate, because conflating them causes most of the confusion:

- **Training scale** — how do you build a very large model? (§3–7)
- **Inference scale** — how do you serve it cheaply? (S5's subject)

They pull in opposite directions. A design that trains efficiently may serve terribly. This tension is the single most important thing driving modern architecture decisions — MoE, GQA, and the "over-train a small model" strategy are all consequences of it.

---

## 2. Scaling laws: how big, trained on how much?

### The empirical finding

Model loss falls as a **power law** in model size, dataset size, and compute. Plotted on log–log axes it's a straight line, over many orders of magnitude. That's a remarkable regularity, and it means you can **predict** the loss of a model you haven't trained by fitting the curve on smaller ones — which is exactly how frontier labs derisk multi-million-dollar runs.

Two caveats stated up front, because they're routinely lost:
- **Loss is not capability.** A smooth loss curve can sit underneath abrupt changes in downstream task performance.
- **Power laws have no cliff, but they do have diminishing returns.** Each successive halving of loss costs exponentially more.

### Kaplan → Chinchilla: the correction that mattered

**Kaplan et al. (2020)** established the power-law relationships and were read as implying that, given more compute, you should mostly make the **model** bigger. The field did that. GPT-3 was 175B parameters trained on roughly 300B tokens.

**Hoffmann et al. (2022) — the Chinchilla paper** — reran the experiment more carefully and found the field had been systematically wrong. For a **compute-optimal** run, parameters and training tokens should scale **roughly equally**, giving a rule of thumb around:

```
tokens ≈ 20 × parameters
```

By that rule, GPT-3's 175B parameters wanted ~3.5 trillion tokens, not 300 billion. It was **severely under-trained**. Chinchilla demonstrated the point directly: a 70B model trained on 1.4T tokens outperformed the 280B Gopher trained on 300B tokens, using the same compute.

**The lesson is that a smaller model trained on more data beat a larger model trained on less.** Everyone had been buying parameters when they should have been buying tokens.

### The inference correction: why nobody is compute-optimal any more

Chinchilla optimises **training** compute. But you train once and serve billions of times, so total cost is dominated by **inference** — and inference cost scales with parameter count, not with how many tokens you trained on.

That flips the calculus. If a 7B model trained on 10× the Chinchilla-optimal tokens gets close to a 30B Chinchilla-optimal model, the 7B wins overwhelmingly in production: cheaper per token, less memory, faster, fits on smaller hardware.

So modern open models are deliberately **"over-trained"** far past the compute-optimal point — trillions of tokens for models in the single-digit billions of parameters. **Chinchilla is not wrong; it answers a question that stopped being the one that matters.** This is the training/inference tension from §1 in its purest form.

### What this means for you

You will not train a frontier model. But the reasoning transfers directly to fine-tuning (Weeks 12–13): **the same total budget spent on a smaller model with more and better data usually beats a bigger model with less.** That's the same conclusion, one tier down.

---

## 3. The memory arithmetic — the "aha" that explains everything else

Here is where the intuition usually breaks. A 7B model in fp16 is **14 GB** of weights. So it fits on a 24GB GPU, right?

For **inference**, yes. For **training**, absolutely not. Training needs to store four things:

| What | Size (7B model, fp16 weights, Adam) | Why |
|---|---|---|
| **Parameters** | 14 GB | The weights themselves |
| **Gradients** | 14 GB | One gradient per parameter |
| **Optimizer states** | ~56 GB | Adam keeps momentum + variance, typically in fp32, plus an fp32 master copy of the weights |
| **Activations** | Varies — often large | Every intermediate value needed for the backward pass |

**That's ~84 GB before activations**, for a model whose weights are 14 GB. The multiplier is roughly **6× the parameter count in bytes** for mixed-precision Adam training, and it is why an 80GB A100 cannot train a 7B model naively.

**Two of these have standard fixes:**

**Mixed precision.** Compute in **bf16** (or fp16) while keeping an fp32 master copy of the weights for the optimizer update. bf16 is preferred over fp16 in modern training because it has the same exponent range as fp32 — it trades precision for range, which means it doesn't overflow, so you don't need loss scaling. This roughly halves compute memory and roughly doubles throughput on hardware with tensor cores.

**Gradient checkpointing (activation recomputation).** Don't store every activation for the backward pass — store a subset ("checkpoints") and **recompute** the rest when needed. This is a direct, tunable **memory-for-compute trade**: typically ~30% more compute time for a large reduction in activation memory. It is the single most useful knob when you hit OOM, and it's available in one line in most frameworks — including for the LoRA fine-tuning of Weeks 12–13.

**The remaining problem is the optimizer state**, which is the biggest single item and doesn't compress the same way. That's what ZeRO attacks (§5).

> **This is why QLoRA (Week 12) works.** It quantises the frozen base weights to 4-bit and trains only small adapters — so parameters shrink 4×, and **gradients and optimizer states only exist for the tiny adapter**, eliminating the dominant term entirely. Week 12 showed you the technique; this is the arithmetic explaining why it's transformative rather than merely convenient.

---

## 4. Three ways to split a model

When one GPU isn't enough, you split. There are exactly three axes, and they answer different questions:

| Approach | What you split | Each GPU holds | Solves |
|---|---|---|---|
| **Data parallel** | The batch | The **whole model** | Throughput |
| **Tensor parallel** | Individual layers | A **slice of every layer** | Model too big for one GPU |
| **Pipeline parallel** | The layer stack | A **contiguous block of layers** | Model too big for one node |

The critical distinction: **data parallelism does not help if the model doesn't fit**, because every GPU still holds a full copy. That's the confusion to avoid.

---

## 5. Data parallelism, ZeRO, and FSDP

**Plain data parallelism (DDP)** — every GPU holds a full model copy, processes a different slice of the batch, computes gradients, and then all GPUs **all-reduce** the gradients so everyone applies an identical update. Simple, well-supported, near-linear scaling for models that fit. The waste is obvious: 8 GPUs hold 8 identical copies of the parameters, gradients, and optimizer states.

**ZeRO (Zero Redundancy Optimizer)** removes that duplication in three cumulative stages:

| Stage | Shards | Memory effect |
|---|---|---|
| **ZeRO-1** | Optimizer states | Removes the largest single item |
| **ZeRO-2** | + gradients | Further large saving |
| **ZeRO-3** | + parameters | Each GPU holds only a slice of the model itself |

At **ZeRO-3**, no GPU holds a complete model. Parameters are gathered on demand for each layer's forward pass, used, then released. This trades communication for memory, and it is what makes very large models trainable on commodity interconnects.

**FSDP (Fully Sharded Data Parallel)** is PyTorch's native implementation of essentially the ZeRO-3 idea, and it's the default choice in the PyTorch ecosystem you learned in Week 5. **DeepSpeed** is the other major library implementing ZeRO.

**The mental model:** data parallelism replicates everything, ZeRO shards everything and reassembles just-in-time. You pay in network traffic to save memory.

---

## 6. Tensor and pipeline parallelism

**Tensor parallelism** splits individual matrices *within* a layer. An attention head or an MLP's weight matrix is divided column-wise or row-wise across GPUs; each computes a partial result and they're combined with a collective operation. Because this happens **inside every layer**, it demands very high bandwidth — it's used **within a node** over NVLink, essentially never across a slow network.

**Pipeline parallelism** assigns contiguous blocks of layers to different GPUs. GPU 0 gets layers 1–8, GPU 1 gets 9–16, and activations flow forward through the chain. The problem is the **pipeline bubble**: while GPU 0 works on the first microbatch, GPUs 1–3 sit idle waiting. The fix is to split the batch into **microbatches** so stages stay busy, which shrinks but never eliminates the bubble. Pipeline parallelism needs much less bandwidth than tensor parallelism, so it's what you use **across nodes**.

**3D parallelism** combines all three, matched to the hardware topology:
- **Tensor parallel** within a node (fast NVLink)
- **Pipeline parallel** across nodes (slower network)
- **Data parallel** across the whole cluster (for throughput)

Plus ZeRO/FSDP sharding on top. This is how frontier training runs are actually configured, and the configuration is chosen by the physical network, not by preference.

**One more piece of reality:** at thousands of GPUs running for months, **hardware failures are routine, not exceptional**. Frequent checkpointing and automatic restart aren't operational niceties — they're a load-bearing part of the training system.

---

## 7. Mixture-of-Experts — the assumption that broke

### The idea

Every architecture so far assumes **every parameter participates in processing every token**. A 70B dense model does 70B parameters' worth of work per token, always.

MoE questions that. Intuitively, why should the parameters that handle Python syntax fire when the model is processing Portuguese poetry?

**The mechanism:** replace the feed-forward network in a transformer block with **N separate FFNs ("experts")** plus a small **router**. For each token, the router picks the **top-k** experts (commonly k = 1 or 2), and only those run.

```
token → router → picks 2 of 64 experts → only those 2 FFNs execute → weighted combination
```

### Total vs active parameters — the key number

This is the distinction that makes MoE make sense:

- **Total parameters** — everything stored. Determines **memory**.
- **Active parameters** — what actually runs per token. Determines **compute**.

A model might have **400B total** parameters but only **35B active** per token. It has the knowledge capacity of a very large model with the per-token compute cost of a medium one.

**That's the whole pitch: decouple capacity from compute.** For the same training FLOPs, an MoE can hold far more knowledge; for the same knowledge, it costs far less compute per token.

### What it costs you

MoE isn't free, and the costs are what make it an engineering problem rather than a free lunch:

**Memory doesn't shrink.** All experts must be *resident* even though only a couple run. So MoE saves compute, not VRAM — which is why MoE models are demanding to *host* despite being cheap to *run*. This directly contradicts the intuition most people form on first hearing the idea.

**Load balancing is a genuine failure mode.** Left alone, the router collapses onto a few favourite experts; the rest are never selected, never receive gradient, and never learn — wasted capacity. Training therefore adds an **auxiliary load-balancing loss** encouraging even expert utilisation. This is a real, tuned, sometimes-fiddly part of MoE training.

**Expert parallelism.** Experts are distributed across GPUs, so tokens must be **routed across the network** to wherever their expert lives, then routed back. This makes MoE communication-heavy and adds a fourth axis to the parallelism strategy.

**Serving is harder.** Batching is complicated when different tokens in the same batch need different experts (a direct complication of S5's continuous batching), and it can produce uneven load across devices.

**Fine-tuning is trickier.** The routing behaviour can shift during fine-tuning, and the model may become unbalanced in ways a dense model simply cannot.

### Why it won anyway

Because the economics are decisive at scale. If you can get large-model quality at medium-model per-token cost, the engineering complexity is worth it — especially when serving cost dominates lifetime spend (§2). This is why MoE is now the norm at the frontier, and why "how many parameters?" has become an ambiguous question that requires you to ask **total or active?**

**Week 6 connection:** the deck's story of what changed since 2017 — RMSNorm, SwiGLU, RoPE, GQA — is a story of **efficiency improvements within a dense architecture**. MoE is the change that leaves that frame entirely, altering not how well each parameter works but **how many of them run at all**.

---

## 8. Long context, at the architecture level

Week 4 established attention's quadratic cost in sequence length. Getting to million-token contexts required work on several fronts:

**FlashAttention** — not an approximation but an **exact** attention algorithm that is far more memory-efficient. The insight is that standard attention is **memory-bandwidth bound**, not compute bound: it materialises the huge N×N attention matrix in slow GPU memory. FlashAttention tiles the computation to keep it in fast on-chip SRAM and never writes the full matrix. Same mathematical result, dramatically less memory traffic. It is one of the highest-impact systems contributions to the field, and it's why long-context training became feasible at all.

**GQA / MLA — attacking the KV cache.** Week 4 covered KV cache and Week 6 covered Grouped-Query Attention. The motivation is inference memory: KV cache grows linearly with context length and batch size, and at long contexts it dominates. GQA shares key/value heads across query heads to shrink it; **Multi-head Latent Attention (MLA)** compresses KV into a lower-dimensional latent representation for a larger reduction. Both trade a small amount of quality for a large amount of memory — and both matter for *serving*, not training.

**RoPE scaling.** Position extrapolation beyond the trained context length (Week 6's rotary embeddings) via interpolation and frequency-scaling methods, usually with a short continued-pretraining phase at the longer length.

**The honest caveat**, and Week 10 made this empirically: **a large context window is not the same as usable context.** Architecture gives you the window; context rot describes what actually happens inside it. Both facts are true simultaneously.

---

## 9. What you can actually do with this

You will not run a 3D-parallel job. But this material has direct, immediate applications:

**When you hit OOM fine-tuning (Weeks 12–13), you now know the levers, in order:** gradient checkpointing → smaller batch with gradient accumulation → bf16 → LoRA → QLoRA → 8-bit optimizer → FSDP across GPUs. Each targets a specific term in §3's table, and knowing which term tells you which lever.

**Read model cards correctly.** "MoE, 8×7B" doesn't mean 56B active — it means ~47B total (experts share attention layers) with ~13B active. Total determines what hardware you need to *host* it; active determines what it costs to *run*.

**Reason about the Chinchilla ratio when fine-tuning.** More, better data on a smaller model usually beats less data on a bigger one — the same conclusion at your scale.

**Know that gradient accumulation is virtual batch size**, not a free lunch: it trades wall-clock time for memory. Week 5's own notebook had a bug in its accumulation demo, which is worth revisiting now that you know what it should show.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 2** (training loop) | Gradients, optimizer states, and activations are the same objects — now counted in gigabytes. |
| **Week 5** (PyTorch) | FSDP is PyTorch-native; gradient checkpointing and accumulation are one-line changes. The W5 accumulation bug is worth rereading. |
| **Week 6** (RMSNorm/SwiGLU/RoPE/GQA) | Those are efficiency gains *within* dense architectures; MoE changes how many parameters run at all. GQA's motivation is KV-cache memory. |
| **Week 7** (training MiniLLM) | The same loop, scaled by six orders of magnitude. Everything here is what breaks along the way. |
| **Week 4** (KV cache) | KV cache memory is *why* GQA and MLA exist, and why long context is a systems problem. |
| **Week 10** (context rot) | Architecture provides the window; context rot describes what's usable inside it. |
| **Weeks 12–13** (LoRA/QLoRA) | §3's memory table is the explanation for why QLoRA works — it eliminates the dominant term. |
| **S5** (inference optimisation) | The training/inference tension drives everything: over-training small models, MoE, GQA, quantisation. |
| **S3** (evaluation) | Scaling laws predict *loss*, not capability. Downstream evaluation is a separate measurement. |
| **S11** (data curation) | Chinchilla says you need enormous token counts; S11 is where they come from and how they're cleaned. |

---

## Key takeaways

1. **Loss follows power laws** in parameters, data, and compute — predictable enough to derisk enormous runs by extrapolating from small ones.
2. **Chinchilla: scale parameters and tokens together**, roughly 20 tokens per parameter. GPT-3 was badly under-trained, and the field was buying parameters when it should have been buying data.
3. **Nobody is compute-optimal any more**, because inference cost dominates lifetime spend. Modern models are deliberately over-trained so a smaller model can serve cheaply.
4. **Training needs ~6× the memory of the weights** — parameters, gradients, optimizer states, activations. This one table explains every technique that follows, including QLoRA.
5. **bf16 over fp16** for training: same exponent range as fp32, so no overflow and no loss scaling.
6. **Gradient checkpointing trades compute for memory** and is the first lever to reach for on OOM.
7. **Three axes of splitting:** data (throughput), tensor (within a node, high bandwidth), pipeline (across nodes, lower bandwidth). Data parallelism alone never solves "doesn't fit."
8. **ZeRO/FSDP shard the redundancy** — optimizer states, then gradients, then parameters — trading communication for memory.
9. **MoE decouples capacity from compute**: many total parameters, few active per token. Ask "total or active?" whenever someone quotes a size.
10. **MoE saves compute, not memory** — all experts stay resident. And load balancing is a real, tuned failure mode, not a footnote.
11. **FlashAttention is exact, not approximate** — it wins by respecting memory bandwidth rather than by cutting corners.
12. **A big context window is not usable context** (Week 10). Architecture and behaviour are different claims.

---

## Glossary

| Term | Definition |
|---|---|
| **Scaling law** | Power-law relationship between loss and model size, data, or compute |
| **Chinchilla-optimal** | Compute-optimal ratio of ~20 training tokens per parameter |
| **Over-training** | Deliberately training past compute-optimal so a smaller model serves cheaply |
| **Mixed precision** | Computing in bf16/fp16 while keeping an fp32 master copy for updates |
| **bf16** | Brain float 16: fp32's exponent range with reduced precision; no loss scaling needed |
| **Gradient checkpointing** | Storing a subset of activations and recomputing the rest; memory-for-compute trade |
| **Gradient accumulation** | Summing gradients over several small batches to simulate a large batch |
| **Data parallel (DDP)** | Full model on every GPU, different batch slices, all-reduce gradients |
| **Tensor parallel** | Splitting individual weight matrices across GPUs; needs very high bandwidth |
| **Pipeline parallel** | Assigning contiguous layer blocks to different GPUs; suffers pipeline bubbles |
| **Microbatch** | Batch subdivision used to keep pipeline stages busy |
| **ZeRO** | Sharding optimizer states (1), gradients (2), and parameters (3) across GPUs |
| **FSDP** | PyTorch's native fully-sharded data parallel, essentially ZeRO-3 |
| **3D parallelism** | Combining tensor, pipeline, and data parallelism, matched to network topology |
| **MoE** | Mixture-of-Experts: multiple FFNs with a router selecting top-k per token |
| **Router** | Small network choosing which experts process each token |
| **Total vs active parameters** | Everything stored (memory) vs what runs per token (compute) |
| **Load-balancing loss** | Auxiliary loss preventing router collapse onto a few experts |
| **Expert parallelism** | Distributing experts across GPUs, requiring token routing over the network |
| **FlashAttention** | Exact attention algorithm avoiding materialisation of the N×N matrix |
| **MLA** | Multi-head Latent Attention: compressing KV cache into a latent space |

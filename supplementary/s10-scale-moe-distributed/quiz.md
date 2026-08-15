# S10 — Quiz (20 questions)

**Topic:** Scale — MoE, Distributed Training, and Scaling Laws
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The Chinchilla rule of thumb is roughly:
- A) 20 parameters per training token
- B) 20 training tokens per parameter
- C) 100 training tokens per parameter
- D) Equal bytes of data and parameters

**2.** GPT-3 (175B params, ~300B tokens) was, by Chinchilla's finding:
- A) Compute-optimal
- B) Severely under-trained — it wanted roughly 3.5T tokens
- C) Over-trained
- D) Correctly sized but under-parameterised

**3.** Modern models are deliberately trained past the compute-optimal point because:
- A) Chinchilla was proven wrong
- B) Inference cost dominates lifetime spend, and it scales with parameters, not training tokens
- C) More tokens always improves loss without limit
- D) It reduces training time

**4.** Training a 7B model in fp16 with Adam requires roughly:
- A) 14 GB
- B) ~84 GB before activations — about 6× the weights
- C) 28 GB
- D) 7 GB

**5.** The largest single memory item in that list is:
- A) Parameters
- B) Gradients
- C) Optimizer states
- D) The KV cache

**6.** bf16 is preferred over fp16 for training because:
- A) It's more precise
- B) It has fp32's exponent range, so it doesn't overflow and needs no loss scaling
- C) It uses less memory
- D) It's faster on all hardware

**7.** Gradient checkpointing trades:
- A) Accuracy for speed
- B) Compute for memory — recomputing activations instead of storing them
- C) Memory for accuracy
- D) Bandwidth for latency

**8.** Data parallelism alone does **not** solve:
- A) Low throughput
- B) A model that doesn't fit on one GPU — every GPU still holds a full copy
- C) Slow convergence
- D) Small batch sizes

**9.** Tensor parallelism is used within a node rather than across nodes because:
- A) Nodes have different GPU types
- B) It splits matrices inside every layer, demanding very high bandwidth
- C) It requires shared memory
- D) Pipeline bubbles would form

**10.** In a "400B total, 35B active" MoE, the number determining what hardware can host it is:
- A) 35B active
- B) 400B total — all experts must be resident
- C) Neither; only context length matters
- D) The router size

**11.** MoE load-balancing loss exists to prevent:
- A) Gradient explosion
- B) Router collapse onto a few experts, leaving the rest untrained
- C) Overfitting
- D) Communication bottlenecks

**12.** FlashAttention is:
- A) An approximation that trades accuracy for speed
- B) An exact algorithm that avoids materialising the N×N matrix in slow memory
- C) A sparse attention pattern
- D) A KV-cache compression scheme

---

## Short answer

**13.** Explain scaling laws, what Chinchilla corrected, and the two caveats about interpreting them.

**14.** Explain why inference economics made compute-optimal training obsolete, and how the same reasoning applies to fine-tuning.

**15.** Work through the memory arithmetic of training a 7B model and explain what each term is for.

**16.** Explain, using that arithmetic, why QLoRA is transformative rather than merely convenient.

**17.** Compare data, tensor, and pipeline parallelism — what each splits, what each solves, and why the combination is chosen by network topology.

**18.** Explain ZeRO's three stages and what FSDP is.

**19.** Explain MoE: the mechanism, the total-vs-active distinction, and the four real costs.

**20.** You're fine-tuning a 13B model for a domain task on 4×A100 40GB and hitting OOM. Walk through your options in priority order, explaining what each targets.

---
---

## Answer key

**1. B** — Parameters and tokens should scale together, roughly 20 tokens per parameter.

**2. B** — 175B × 20 ≈ 3.5T tokens; it received under a tenth of that.

**3. B** — You train once and serve billions of times, so per-token serving cost dominates.

**4. B** — 14 GB params + 14 GB gradients + ~56 GB optimizer states, before activations.

**5. C** — Adam's momentum and variance plus an fp32 master copy dominate, at roughly 4× the fp16 weights.

**6. B** — Same exponent range as fp32; it trades precision for range.

**7. B** — Store a subset of activations, recompute the rest; typically ~30% more compute for a large memory saving.

**8. B** — Every GPU holds a full model copy, so the fit problem is untouched.

**9. B** — Splitting happens inside every layer, so the communication is frequent and heavy; it needs NVLink-class bandwidth.

**10. B** — All experts must be resident in memory; only compute is sparse.

**11. B** — Unselected experts receive no gradient and never learn, wasting capacity.

**12. B** — Same mathematical result; it wins on memory bandwidth by tiling into on-chip SRAM.

**13.** Scaling laws are the empirical finding that model loss falls as a **power law** in model size, dataset size, and compute — a straight line on log–log axes across many orders of magnitude. The practical value is **predictability**: you can fit the curve on small models and forecast the loss of a model you haven't trained, which is exactly how frontier labs derisk multi-million-dollar runs. **Kaplan et al. (2020)** established the relationships and were widely read as implying that extra compute should go mostly into **model size**; the field followed, and GPT-3 was 175B parameters on roughly 300B tokens. **Hoffmann et al. (2022) — Chinchilla** — reran the experiments more carefully and found this systematically wrong: for a compute-optimal run, **parameters and training tokens should scale roughly equally**, about **20 tokens per parameter**. GPT-3 therefore wanted ~3.5 trillion tokens and was severely under-trained. Chinchilla proved the point directly — a **70B model on 1.4T tokens beat the 280B Gopher on 300B tokens at equal compute** — meaning everyone had been buying parameters when they should have been buying data. **Two caveats:** first, **loss is not capability** — a smooth loss curve can sit underneath abrupt changes in downstream task performance, so scaling laws predict the training objective, not usefulness (S3's distinction). Second, **power laws have no cliff but do have diminishing returns**: each successive halving of loss costs exponentially more, so "no wall" and "economically sensible" are different claims.

**14.** Chinchilla optimises **training** compute — the cost of building the model once. But a deployed model is trained once and served billions of times, so **total lifetime cost is dominated by inference**, and inference cost scales with **parameter count**, not with how many tokens you trained on. Training tokens are a one-off; parameters are a tax on every request forever. That inverts the objective. If a 7B model trained on 10× the Chinchilla-optimal token count approaches the quality of a 30B compute-optimal model, the 7B wins overwhelmingly in production — cheaper per token, less memory, faster, and it fits on smaller hardware. So modern open models are deliberately **"over-trained"** far past the compute-optimal point, with trillions of tokens poured into models of single-digit billions of parameters. **Chinchilla isn't wrong; it answers a question that stopped being the one that matters** — it is the correct answer to "cheapest way to reach loss L" and the wrong answer to "cheapest model to operate at quality Q." **The same reasoning applies one tier down at fine-tuning** (Weeks 12–13): for a fixed budget, **a smaller model with more and better data usually beats a bigger model with less**, and the smaller model is also cheaper to serve afterwards. The instinct to reach for the largest base model you can afford to train is the same instinct that produced GPT-3's under-training.

**15.** A 7B model in fp16 has **14 GB of weights**, which fits comfortably on a 24 GB GPU — for **inference**. Training must additionally store three things. **Gradients: ~14 GB** — one gradient value per parameter, needed to apply the update. **Optimizer states: ~56 GB** — Adam maintains both first-moment (momentum) and second-moment (variance) estimates per parameter, typically kept in fp32, plus an fp32 master copy of the weights so that small updates aren't lost to fp16 rounding; that's roughly 4× the fp16 parameter size and it is the **largest single item**. **Activations: variable but often large** — every intermediate value produced in the forward pass must be retained for the backward pass, and this scales with batch size, sequence length, and depth. The total is **~84 GB before activations**, roughly **6× the parameter size in bytes**, which is why an 80 GB A100 cannot naively train a 7B model despite holding its weights five times over. **Two terms have standard fixes:** mixed precision (compute in bf16, keep an fp32 master copy) roughly halves compute memory and roughly doubles throughput on tensor-core hardware, and **gradient checkpointing** stores only a subset of activations and recomputes the rest, trading roughly 30% more compute for a large activation-memory reduction. **The optimizer state is what remains**, and it is precisely what ZeRO stage 1 shards.

**16.** The memory table shows that **parameters are the smallest of the four terms** — gradients match them, and optimizer states are roughly four times larger. So the naive framing of QLoRA as "quantisation to save memory on weights" badly understates it. **Quantising the base model to 4-bit shrinks the parameter term ~4×**, from 14 GB to roughly 3.5 GB for a 7B model — useful, but only one term. **The transformative part is that the base weights are frozen.** Frozen parameters produce no gradients and have no optimizer states, so the **~14 GB gradient term and the ~56 GB optimizer-state term simply cease to exist for the base model.** They exist only for the LoRA adapters, which are a tiny fraction of the parameter count — often well under 1% — making both terms effectively negligible. What was an ~84 GB problem becomes a few gigabytes plus activations. **That is why a technique that sounds like an incremental memory optimisation instead changes what hardware class can fine-tune a model at all**, putting 7B–13B fine-tuning on a single consumer GPU. Week 12 taught the technique; this arithmetic is why it mattered rather than merely helped. It also explains the residual costs honestly: activations still scale with batch and sequence length, so gradient checkpointing remains useful, and 4-bit quantisation of the frozen base does cost some quality.

**17.** **Data parallelism** splits the **batch**: every GPU holds a complete model copy, processes a different slice, computes gradients, and all-reduces them so every GPU applies an identical update. It solves **throughput** and scales near-linearly — but every GPU still holds the full model, so **it does not help at all if the model doesn't fit**, which is the most common misconception. **Tensor parallelism** splits **individual weight matrices within a layer**: an MLP or attention projection is divided across GPUs, each computes a partial result, and a collective combines them. It solves "the model is too big for one GPU," but because the split-and-combine happens **inside every layer**, communication is extremely frequent — so it demands NVLink-class bandwidth and is used **within a node**, essentially never across a network. **Pipeline parallelism** splits the **layer stack**: GPU 0 takes layers 1–8, GPU 1 takes 9–16, and activations flow forward. It solves "the model is too big for one node," and it communicates only at stage boundaries, so it tolerates slower links. Its cost is the **pipeline bubble** — downstream stages idle while the first microbatch works through — mitigated but never eliminated by splitting into **microbatches**. **The combination is dictated by physical network topology, not preference:** tensor parallel inside a node where NVLink is fast, pipeline parallel across nodes where the network is slower, and data parallel across the whole cluster for throughput, with ZeRO/FSDP sharding layered on top. That's **3D parallelism**. And at thousands of GPUs over months, **hardware failure is routine**, so checkpointing and automatic restart are load-bearing components rather than operational polish.

**18.** Plain data parallelism is wasteful in a specific way: 8 GPUs hold **8 identical copies** of the parameters, gradients, and optimizer states, even though only the batch differs. **ZeRO (Zero Redundancy Optimizer)** eliminates that duplication in three cumulative stages. **ZeRO-1 shards the optimizer states** across GPUs — the largest single memory item (§15), so this alone is a major win and costs little extra communication. **ZeRO-2 additionally shards the gradients**, so each GPU holds gradients only for its slice. **ZeRO-3 additionally shards the parameters themselves**, meaning **no GPU holds a complete model**: parameters are gathered on demand for each layer's forward pass, used, and released immediately after. The consistent trade is **communication for memory** — you move more data over the network to avoid storing redundant copies, which is why the practical stage choice depends on interconnect quality. **FSDP (Fully Sharded Data Parallel)** is PyTorch's native implementation of essentially the ZeRO-3 idea, and it's the default in the PyTorch ecosystem from Week 5; **DeepSpeed** is the other major library implementing ZeRO. **The mental model:** plain data parallelism replicates everything, ZeRO shards everything and reassembles just-in-time.

**19.** Every architecture up to this point assumes **every parameter participates in processing every token** — a 70B dense model does 70B parameters' worth of work per token, always. **MoE rejects that assumption.** The feed-forward network in a transformer block is replaced by **N separate FFNs ("experts")** plus a small **router**; for each token the router selects the **top-k** experts (commonly 1 or 2) and only those execute, their outputs combined by weight. Intuitively: the parameters handling Python syntax needn't fire on Portuguese poetry. **The key distinction is total vs active parameters.** Total is everything stored and determines **memory**; active is what runs per token and determines **compute**. A model with 400B total and 35B active has a large model's knowledge capacity at a medium model's per-token compute cost — **capacity decoupled from compute**, which is the entire pitch. **Four real costs.** **(1) Memory doesn't shrink** — all experts must be resident even though few run, so MoE saves compute, not VRAM, and MoE models are demanding to *host* while cheap to *run*. This contradicts most people's first intuition. **(2) Load balancing is a genuine failure mode** — left alone the router collapses onto favourite experts, and unselected experts receive no gradient and never learn, wasting capacity, so training adds an **auxiliary load-balancing loss** that is real, tuned, and sometimes fiddly. **(3) Expert parallelism** distributes experts across GPUs, so tokens must be routed over the network to their expert and back, making MoE communication-heavy and adding a fourth parallelism axis. **(4) Serving and fine-tuning are harder** — batching is complicated when tokens in one batch need different experts (complicating S5's continuous batching) and load across devices becomes uneven, while fine-tuning can shift routing behaviour and unbalance the model in ways dense models cannot. **It won anyway because the economics are decisive at scale**, and it means "how many parameters?" is now an ambiguous question requiring "total or active?".

**20.** **First, identify which memory term is actually the problem**, because 4×A100 40GB is 160 GB total and a 13B model needs roughly 26 GB of fp16 weights — so the weights are not the issue, and blind lever-pulling wastes time. Full fine-tuning of 13B needs roughly **6× the weights ≈ 156 GB** before activations, which is why it OOMs on this hardware despite the apparently generous total: naive **data parallelism replicates everything**, so each 40 GB card is trying to hold the full ~156 GB, not a quarter of it. **In priority order, cheapest and most effective first: (1) Switch to LoRA, and QLoRA if needed.** This is the single biggest lever and should be the default rather than a fallback. Freezing the base eliminates the gradient and optimizer-state terms entirely — the two that dominate — leaving weights plus activations plus tiny adapters. QLoRA additionally quantises the frozen base to 4-bit, cutting ~26 GB to ~7 GB, at which point a 13B fine-tune fits comfortably on a **single** card and the other three become throughput rather than necessity. For a domain task this is also usually the *right* method, not just the affordable one (Week 12): full fine-tuning risks catastrophic forgetting, and LoRA's adapters are far easier to version, swap, and roll back. **(2) Enable gradient checkpointing** — one line, targets the activation term specifically, costs roughly 30% wall-clock, and is worth enabling almost unconditionally at this scale. **(3) Reduce per-device batch size and use gradient accumulation** to keep the effective batch unchanged; this trades time for memory and directly shrinks activations, which scale with batch and sequence length. **(4) Confirm bf16 mixed precision** is actually on — it halves compute memory and roughly doubles throughput on A100 tensor cores, and it is the correct choice over fp16 because its fp32 exponent range removes overflow and loss scaling. **(5) Check sequence length**, since activation memory scales with it and a max_length inherited from a config is a common and invisible cause of OOM — truncating to what your data actually needs is often a large, free saving. **(6) Use an 8-bit optimizer** (e.g. bitsandbytes) if any optimizer state remains, cutting that term substantially. **(7) Only then reach for FSDP** to shard whatever is left across the four GPUs — it's the most complex option and, after steps 1–6, usually unnecessary for a 13B model on this hardware. **Finally, verify rather than assume the fix worked:** re-check peak allocated memory, confirm the loss curve looks sane before committing to a long run, and hold out an eval set with confidence intervals (S3) so you can tell whether the smaller batch actually cost you quality.

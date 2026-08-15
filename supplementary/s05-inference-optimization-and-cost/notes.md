# S5 — Inference Optimization & Serving Economics

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 4 teaches the KV cache *theory* and Week 6 quantifies its memory cost, but nothing covers production serving: batching, speculative decoding, latency metrics, or the build-vs-buy cost calculation.

**Fills the gap after:** Week 4 (KV cache), Week 6 (GQA), Week 23 (deployment surfaces)
**Prerequisites:** Weeks 4, 6, 12

---

## 1. The two phases, and why everything follows from them

Week 4 introduced prefill and decode. Almost every serving decision follows from one fact about them:

| | **Prefill** | **Decode** |
|---|---|---|
| Processes | The whole prompt, in parallel | One token at a time |
| Bottleneck | **Compute-bound** (GPU FLOPs) | **Memory-bandwidth-bound** |
| Scales with | Prompt length | Output length |
| Metric | **TTFT** — time to first token | **TPOT** — time per output token |

**Why decode is memory-bound and not compute-bound:** to generate one token you must read *every* model weight from GPU memory — billions of parameters — and then do a tiny amount of arithmetic with them. The GPU spends its time waiting on memory, not calculating. The arithmetic units sit largely idle.

**This single fact explains batching, quantization, and speculative decoding.** All three are attempts to get more useful work out of each pass over the weights.

---

## 2. Batching — the highest-leverage optimization

If decode is memory-bound, and you've already paid to read all the weights, **processing one sequence or thirty-two costs nearly the same time.** The weights are read once and used for the whole batch.

**Batching is close to free throughput.** This is why hosted inference is cheaper than self-hosting for most workloads: providers batch across all their customers.

### Three generations of batching

| Approach | Behaviour | Problem |
|---|---|---|
| **Static** | Wait for N requests, run them together, return together | Everyone waits for the *longest* generation; the GPU idles as sequences finish |
| **Dynamic** | Group whatever arrived within a time window | Better, same tail problem |
| **Continuous (in-flight)** | Sequences enter and leave the batch **per token**; a finished slot is refilled immediately | The current standard |

**Continuous batching is the big one.** Because generation lengths vary enormously — one request wants 20 tokens, another 4,000 — static batching wastes most of the GPU waiting for the long one. Continuous batching removes that waste entirely and typically produces a multiple-fold throughput improvement over static.

**The trade-off:** larger batches mean higher throughput and *worse* per-request latency. You are choosing where on that curve to sit, and the right answer differs for a chat UI (favour latency) versus an overnight batch job (favour throughput).

---

## 3. The KV cache is the real constraint

Week 6 did the arithmetic: a 32-layer model at 8K context needs roughly 4 GB of KV cache, and 128K context needs roughly 64 GB.

**The consequence for serving:** the KV cache — not the weights — usually determines **how many concurrent users a GPU can serve.** Weights are loaded once and shared across the whole batch; the KV cache is **per request**.

```
GPU memory = weights (shared, fixed) + KV cache (per request) + activations
                                        ↑
                            this is what limits concurrency
```

### PagedAttention

The problem: naive implementations preallocate a contiguous KV block for each sequence's *maximum possible* length. A request that might generate 4,000 tokens but actually generates 50 wastes 98% of its allocation. Across a batch, most of your memory is reserved and unused — internal fragmentation, the same problem operating systems had before virtual memory.

**PagedAttention** (the technique behind vLLM) applies the OS solution: split the KV cache into fixed-size **blocks** and maintain a per-sequence block table. Memory is allocated as tokens are actually generated, non-contiguously.

**Results:** near-elimination of fragmentation waste, substantially larger effective batch sizes, and — a bonus — **prefix sharing**, where multiple requests with a common prompt prefix physically share the same KV blocks instead of duplicating them. That last part is the same insight as prompt caching (S2), implemented at the memory-allocation layer.

### The architectural lever

Week 6's **GQA** attacks this from the model side: sharing K/V projections across groups of query heads cuts KV cache memory ~4× with almost no quality loss. Fewer cache bytes per request means more concurrent requests per GPU.

**Serving and architecture are solving the same problem from two ends.** That's why GQA is in every modern open-weight model.

---

## 4. Quantization at serving time

Week 12 covered quantization for *training* (QLoRA: 4-bit frozen base, 16-bit adapters). Serving quantization is a different decision with a different trade-off.

| Format | Size vs fp16 | Notes |
|---|---|---|
| **fp16 / bf16** | 1× | Baseline |
| **fp8** | 0.5× | Hardware-supported on recent GPUs; minimal quality loss |
| **int8** | 0.5× | Widely supported |
| **int4** (AWQ, GPTQ) | 0.25× | Noticeable quality cost on some tasks; test on *your* workload |

**Two distinct things get quantized, and people conflate them:**

- **Weight quantization** shrinks the model. Since decode is memory-bandwidth-bound, **halving the weights roughly halves the bytes read per token** — so weight quantization is a *speed* optimization as much as a memory one.
- **KV cache quantization** shrinks the per-request cache, which directly increases concurrency. Often the higher-value option for long-context serving, and less discussed.

**The honest caveat:** quality loss from int4 is workload-dependent. It's frequently negligible on chat and noticeable on precise reasoning or code. **Measure on your own eval set (S3)** — do not accept a general claim.

---

## 5. Speculative decoding

The cleverest trick in serving, and it follows directly from decode being memory-bound.

**The idea:** a small fast "draft" model proposes the next several tokens. The large model then verifies all of them **in a single forward pass** — which is possible because verification is a parallel operation, exactly like prefill. Tokens matching what the large model would have produced are accepted; the first mismatch and everything after it is discarded.

```
Draft model:   proposes  [the, cat, sat, on, the]
Target model:  verifies all 5 in ONE pass
Result:        accepts [the, cat, sat], rejects [on, the]
               → 3 tokens for the cost of ~1 forward pass
```

**Why it's free quality-wise:** the accepted tokens are exactly what the large model would have generated. **The output distribution is unchanged.** This is not an approximation — it's a pure latency optimization.

**Why it works at all:** you had spare compute. Decode leaves the arithmetic units idle while waiting on memory; verifying five tokens costs barely more than verifying one.

**Speedups** are typically in the 2–3× range, depending on how well the draft model predicts the target — which is why it works better on predictable text (code, structured output) than on genuinely creative generation.

**Variants:** a smaller model from the same family as the drafter; *n*-gram or prompt-lookup drafting with no model at all (surprisingly effective when the output copies from the input, as in RAG or summarisation); or self-speculation using the model's own early layers.

---

## 6. Latency metrics — say which one you mean

| Metric | Meaning | Governed by |
|---|---|---|
| **TTFT** | Time to first token | Prefill — scales with **prompt** length |
| **TPOT / ITL** | Time per output token | Decode — roughly constant per token |
| **Total latency** | `TTFT + TPOT × output_tokens` | Both |
| **Throughput** | Tokens/second across *all* requests | Batching |
| **Goodput** | Throughput of requests that met their SLA | The one that matters |

**Two things people get wrong:**

**"Latency" is ambiguous.** A long prompt with a short answer is dominated by TTFT; a short prompt with a long answer is dominated by TPOT. Optimising the wrong one wastes effort. Week 4 already gave you the diagnosis: *"time to first token and tokens per second are different metrics."*

**Throughput and latency trade against each other.** Bigger batches raise throughput and hurt per-request latency. A dashboard showing only throughput will look great while users experience a slow product — which is what **goodput** exists to capture.

**The streaming trick:** perceived latency is dominated by TTFT because streaming lets users start reading immediately. A response that takes 8 seconds total but starts in 0.4s feels far faster than one that takes 5 seconds and arrives all at once. **Stream by default.**

---

## 7. Cost modelling

### API pricing

Priced per million tokens, with input much cheaper than output. Across the Claude family the input:output ratio is consistently about **1:5**, and there's roughly a 5× spread between the cheapest and most capable tiers.

Three consequences:

1. **Output tokens dominate cost** on generation-heavy workloads. Reducing verbosity has ~5× the effect of reducing prompt length, per token.
2. **Model choice is the biggest single lever** — bigger than most optimizations. Routing simple work to a cheaper tier can cut spend several-fold.
3. **Batch APIs** typically offer a large discount for latency-insensitive work. If it can wait an hour, it should.

### Prompt caching (S2) is the cheapest optimization available

Cache reads cost roughly a tenth of normal input price. For any workload with a large shared prefix — a system prompt, a document, few-shot examples — this is a large saving for a one-line change.

**The economics:** cache writes cost *more* than a normal request (a modest premium for the short-lived cache, more for the long-lived one), so it pays off from about the second or third identical-prefix request onward. Below that, it costs you.

### Where the money actually goes in agents

Week 8's context-engineering problem is a **cost** problem:

```
Turn 1: 1,000 input tokens
Turn 2: 1,500  (turn 1 + result)
Turn 3: 2,400
...
Turn 10: ~15,000
Total input billed ≈ 60,000 tokens for a 10-step task
```

**Agent cost is roughly quadratic in step count** because every step resends the whole transcript. This is why Week 8's truncation caps, Week 17's compaction, and prompt caching all matter more for agents than for single calls.

### Self-host vs API

Roughly, self-hosting becomes competitive when you have:
- **High, sustained utilisation** — an idle GPU costs the same as a busy one, and APIs charge nothing when idle
- **Predictable load** — you can't autoscale a GPU reservation quickly
- **A specific model** — a fine-tune nobody hosts (Week 23)
- **Data-residency requirements** that rule out third-party inference

**What people forget to count:** engineering time (the largest line item, usually), GPU idle time, ops and on-call, and the fact that providers batch across *all* their customers and you can only batch across your own — so their effective utilisation is structurally higher than yours.

**Default to the API.** Self-host when you've measured that it's cheaper, not when it feels cheaper.

---

## 8. Model routing and cascades

**Routing:** classify the request, send it to an appropriately-sized model. Simple lookups to a small model, hard reasoning to a large one.

**Cascade:** try the cheap model first; escalate only if a confidence check or validator fails.

Both exploit the fact that request difficulty is **highly skewed** — most requests are easy, a few are hard, and paying frontier prices for all of them is waste.

**The catch:** the router itself costs something and can be wrong. A misrouted hard question is worse than an over-served easy one. Keep the router cheap (a classifier or heuristic, not another frontier call), and measure end-to-end quality (S3), not just the savings.

**Effort as a routing dial:** on models with an effort parameter, adjusting effort per route is a cheaper, lower-risk version of the same idea — same model, less thinking, no routing logic to get wrong.

---

## 9. Serving engines

| Engine | Character |
|---|---|
| **vLLM** | The default. PagedAttention, continuous batching, broad model support. |
| **SGLang** | Strong on structured generation and prefix-heavy workloads (RAG). |
| **TensorRT-LLM** | Fastest on NVIDIA hardware; heaviest to set up. |
| **llama.cpp** | CPU and consumer hardware; the local/edge option. |

For most teams: **vLLM unless you have a specific reason.** Week 23 noted the same default in the managed-endpoint context.

---

## 10. A practical optimization order

Cheapest and highest-impact first:

```
1. Stream           → perceived latency, one line of code
2. Right-size the model → the biggest single cost lever
3. Prompt caching   → ~10× cheaper on the shared prefix
4. Trim the prompt  → especially in agent loops
5. Cap output length → output tokens cost ~5× input
6. Batch API        → large discount for anything latency-insensitive
7. Continuous batching → if self-hosting
8. Quantization     → after measuring quality impact
9. Speculative decoding → free latency, more moving parts
10. Custom kernels  → almost never worth it
```

**Steps 1–6 need no infrastructure and cover most of the available win.** Most teams jump to step 8 and skip steps 2 and 3.

---

## 11. How this connects to the course

| Course moment | What this adds |
|---|---|
| **W4** — prefill vs decode | Why decode is memory-bound, and that batching, quantization, and speculative decoding all follow |
| **W4** — "TTFT and TPS are different metrics" | The full metric set, plus goodput |
| **W4/W6** — KV cache arithmetic | It's the *concurrency* limit; PagedAttention is the fix |
| **W6** — GQA | Architecture and serving attacking the same bottleneck |
| **W8** — context grows every iteration | The same fact, priced |
| **W12** — QLoRA quantization | Training vs serving quantization are different decisions |
| **W23** — three deployment surfaces | The cost model behind choosing between them |

---

## Key takeaways

1. **Prefill is compute-bound; decode is memory-bandwidth-bound.** Almost everything follows.
2. **Batching is nearly free throughput** — and continuous batching is the version that matters.
3. **The KV cache, not the weights, limits concurrency.** PagedAttention removes the fragmentation waste; GQA removes the bytes.
4. **Weight quantization is a speed win**, not just a memory one, because decode is bandwidth-bound.
5. **KV-cache quantization is the underrated one** for long-context serving.
6. **Speculative decoding is free** — identical output distribution, 2–3× latency.
7. **Say which latency you mean.** TTFT is prompt-driven; TPOT is output-driven.
8. **Stream by default** — perceived latency is dominated by TTFT.
9. **Output tokens cost ~5× input.** Verbosity is expensive.
10. **Agent cost is roughly quadratic in steps.**
11. **Model choice is the biggest cost lever**; caching is the cheapest optimization.
12. **Default to the API.** Self-host on measurement, not intuition.

---

## Glossary

| Term | Meaning |
|---|---|
| **Prefill** | Processing the prompt in parallel; compute-bound; sets TTFT. |
| **Decode** | Generating tokens one at a time; memory-bandwidth-bound; sets TPOT. |
| **Memory-bandwidth-bound** | Limited by reading weights from memory, not by arithmetic. |
| **Static / dynamic / continuous batching** | Fixed groups / time-windowed groups / per-token slot refill. |
| **PagedAttention** | Block-based KV cache allocation eliminating fragmentation. |
| **Prefix sharing** | Multiple requests sharing KV blocks for a common prompt prefix. |
| **Weight quantization** | Lower-precision weights — smaller and faster to read. |
| **KV cache quantization** | Lower-precision cache — more concurrent requests. |
| **AWQ / GPTQ** | Post-training quantization methods. |
| **Speculative decoding** | Draft model proposes, target model verifies in one pass. |
| **Draft model** | The small proposer in speculative decoding. |
| **TTFT** | Time to first token. |
| **TPOT / ITL** | Time per output token / inter-token latency. |
| **Throughput** | Tokens per second across all requests. |
| **Goodput** | Throughput of requests meeting their latency SLA. |
| **Batch API** | Asynchronous processing at a large discount. |
| **Routing / cascade** | Sending requests to a right-sized model / escalating on failure. |
| **vLLM / SGLang / TensorRT-LLM / llama.cpp** | Serving engines: default / structured & prefix-heavy / NVIDIA-optimised / CPU-local. |

# S5 — Quiz (20 questions)

**Topic:** Inference Optimization & Serving Economics
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The decode phase is:
- A) Compute-bound
- B) Memory-bandwidth-bound
- C) Network-bound
- D) Disk-bound

**2.** Batching is nearly free throughput because:
- A) GPUs have spare memory
- B) The weights are read once and reused across the whole batch
- C) Requests are compressed
- D) Attention is quadratic

**3.** Continuous (in-flight) batching differs from static batching by:
- A) Using larger batches
- B) Letting sequences enter and leave the batch per token
- C) Running on CPU
- D) Requiring quantization

**4.** What usually limits how many concurrent users a GPU can serve?
- A) Model weights
- B) The KV cache, which is per-request
- C) Tokenizer throughput
- D) Network bandwidth

**5.** PagedAttention solves:
- A) Attention's quadratic cost
- B) KV cache memory fragmentation from preallocating maximum lengths
- C) Slow tokenization
- D) Quantization error

**6.** Weight quantization improves decode *speed* because:
- A) Smaller numbers multiply faster
- B) Decode is bandwidth-bound, so halving weight bytes halves what must be read per token
- C) It enables larger batches only
- D) It removes the KV cache

**7.** Speculative decoding changes the output distribution:
- A) Slightly, in exchange for speed
- B) Not at all — accepted tokens are exactly what the target model would produce
- C) Only at high temperature
- D) Only for the first token

**8.** Speculative decoding is possible because:
- A) Draft models are more accurate
- B) Verification is parallel, and decode leaves compute idle
- C) The KV cache is shared
- D) It skips attention

**9.** TTFT is primarily driven by:
- A) Output length
- B) Prompt length
- C) Batch size only
- D) Model precision

**10.** Across the Claude family, the input:output price ratio is roughly:
- A) 1:1
- B) 1:5
- C) 5:1
- D) 1:100

**11.** Agent cost grows roughly:
- A) Linearly with step count
- B) Quadratically with step count, because each step resends the transcript
- C) Logarithmically
- D) Independently of step count

**12.** Prompt caching pays off from about:
- A) The first request
- B) The second or third identical-prefix request
- C) The hundredth request
- D) Never, for short prompts

---

## Short answer

**13.** Explain why decode is memory-bandwidth-bound and name three optimizations that follow from it.

**14.** Explain the three generations of batching and why continuous batching matters most.

**15.** Explain why the KV cache limits concurrency, and how PagedAttention and GQA attack it from different ends.

**16.** Explain speculative decoding, including why it's quality-neutral and when it works best.

**17.** Distinguish TTFT, TPOT, throughput, and goodput. Why is "latency" ambiguous?

**18.** Explain why agent cost is quadratic in steps and name three mitigations from the course.

**19.** When is self-hosting genuinely cheaper than an API, and what do people forget to count?

**20.** Your chat product has 3s TTFT and users complain it's slow. Prompts are ~8,000 tokens (a large system prompt plus retrieved context); responses are ~200 tokens. Diagnose and prioritise fixes.

---
---

## Answer key

**1. B** — Generating one token requires reading every weight from memory while doing very little arithmetic.

**2. B** — Once the weights have been read, using them for one sequence or thirty-two costs nearly the same time.

**3. B** — Finished slots are refilled immediately rather than waiting for the whole batch, removing the tail-latency waste.

**4. B** — Weights are shared across the batch; the KV cache is per request and scales with context length.

**5. B** — Naive allocation reserves each sequence's maximum possible length contiguously, wasting most of it; PagedAttention allocates fixed-size blocks on demand.

**6. B** — Fewer bytes to read per token directly reduces the bottleneck.

**7. B** — Accepted tokens are exactly those the target model would have produced; it's a pure latency optimization.

**8. B** — Verification of several proposed tokens is a parallel operation like prefill, and the idle arithmetic units make it nearly free.

**9. B** — TTFT is the prefill phase, which scales with prompt length.

**10. B** — Roughly 1:5, consistently across tiers.

**11. B** — Every step resends the full transcript, so total input billed grows with the sum of transcript lengths.

**12. B** — Cache writes cost more than a normal request, so you need roughly two or three reads to break even depending on the cache lifetime.

**13.** Generating a single token requires reading **every parameter of the model** out of GPU memory and performing a comparatively tiny amount of arithmetic with each one. The arithmetic units finish almost immediately and then wait for the next block of weights to arrive, so the wall-clock time is set by memory bandwidth rather than FLOPs — the GPU's compute capacity is largely idle. **Three optimizations that follow.** **Batching:** since the weights have already been paid for, running thirty-two sequences through that same pass costs barely more than one, making batching nearly free throughput. **Weight quantization:** halving the bytes per parameter halves the data that must be read per token, so it is a *speed* optimization as much as a memory one — the opposite of the intuition that lower precision is merely a space saving. **Speculative decoding:** because compute is idle, verifying five proposed tokens in one forward pass costs almost the same as verifying one, which is precisely the spare capacity the technique monetises.

**14.** **Static batching** waits for N requests, runs them together, and returns them together — so every request waits for the *longest* generation in the group, and GPU slots sit idle as shorter sequences finish. **Dynamic batching** groups whatever arrived within a short time window, which improves arrival flexibility but leaves the tail problem untouched: the batch still completes as a unit. **Continuous (in-flight) batching** lets sequences enter and leave the batch **per token**, refilling a finished slot immediately with a waiting request. **Why it matters most:** generation lengths vary enormously in real traffic — one request wants 20 tokens and another 4,000 — so under static batching the great majority of the batch's slot-time is spent waiting on one long generation. Continuous batching eliminates that waste entirely and typically yields a multiple-fold throughput improvement, which is why it is now standard in every serious serving engine. The trade-off to remember is that larger batches always raise throughput and worsen per-request latency, so the batch size is a position on a curve rather than a value to maximise.

**15.** The **model weights are loaded once and shared** across every sequence in the batch, so they are a fixed cost. The **KV cache is per request** and grows with that request's context length, so as you add concurrent users the cache is what consumes the remaining memory — making it, not the weights, the binding constraint on concurrency. Week 6's arithmetic makes the scale concrete: roughly 4 GB at 8K context for a 32-layer model, and roughly 64 GB at 128K. **PagedAttention attacks the waste.** Naive implementations preallocate a contiguous block sized to each sequence's *maximum possible* length, so a request that might generate 4,000 tokens but produces 50 wastes almost all of its reservation — internal fragmentation, the problem operating systems solved with virtual memory. PagedAttention splits the cache into fixed-size blocks with a per-sequence block table, allocating on demand and non-contiguously, which reclaims that waste and additionally enables **prefix sharing** between requests with a common prompt. **GQA attacks the bytes.** By sharing K/V projections across groups of query heads it cuts the cache roughly fourfold with almost no quality cost, so every request simply needs less. One is a memory-management fix and the other an architectural one, and they compose.

**16.** A small, fast **draft model** proposes the next several tokens; the large **target model** then verifies all of them in a **single forward pass**, accepting the prefix that matches what it would itself have generated and discarding from the first mismatch onward. In the worked example the draft proposes five tokens, three are accepted, and you have produced three tokens for roughly the cost of one forward pass. **Why it's quality-neutral:** the accepted tokens are by construction exactly those the target model would have produced, so the output distribution is unchanged — this is not an approximation trading quality for speed but a pure latency optimization. **Why it's possible:** verifying several candidate tokens is a *parallel* operation, structurally like prefill, and decode leaves the arithmetic units idle waiting on memory, so the extra verification is nearly free. **Where it works best:** on predictable text, because the speedup depends entirely on the draft model's acceptance rate. Code, structured output, and text that copies heavily from the input (RAG, summarisation) accept at high rates — so well that prompt-lookup drafting with no model at all can work — while genuinely creative or high-entropy generation accepts less and gains less. Typical speedups land in the 2–3× range.

**17.** **TTFT** (time to first token) measures the prefill phase and scales with **prompt** length. **TPOT** (time per output token, also called inter-token latency) measures the decode phase and is roughly constant per token, so total generation time scales with **output** length. **Total latency** is `TTFT + TPOT × output_tokens`. **Throughput** is tokens per second across *all* concurrent requests — a fleet-level metric driven by batching. **Goodput** is the throughput of requests that actually met their latency SLA, which is the metric that corresponds to user experience. **"Latency" is ambiguous** because the two components are driven by completely different inputs and respond to completely different fixes: a request with an 8,000-token prompt and a 200-token answer is dominated by TTFT and is fixed by shrinking or caching the prompt, while a request with a short prompt and a 4,000-token answer is dominated by TPOT and is fixed by speculative decoding, quantization, or a smaller model. Optimising the wrong half wastes the effort entirely — which is Week 4's point that time-to-first-token and tokens-per-second are different metrics. **Goodput exists** because throughput and latency trade against each other: a dashboard reporting only throughput looks excellent precisely when large batches are making individual users wait.

**18.** The model is **stateless**, so every step of an agent loop resends the entire conversation so far — system prompt, every prior thought, every action, every observation. Input tokens billed at step *n* are proportional to the transcript length at step *n*, and total cost is the **sum** of those lengths, which grows roughly with the square of the step count. A ten-step task whose first call is 1,000 tokens can bill on the order of 60,000 input tokens in total. Costs are asymmetric too: this growth is all *input* tokens, which are the cheap side, but the volume compensates. **Three mitigations from the course.** **Truncate observations at the source** (Week 8) — the notebook caps Wikipedia summaries at 800 characters, file reads at 3,000, and directory listings at 50 entries, so no single tool result can flood the transcript. **Compaction** (Week 17) — summarise older turns once context passes a threshold, producing that lecture's saw-tooth context graph. **Prompt caching** (S2) — since the system prompt and early turns form a stable prefix, cache reads bring that portion down to roughly a tenth of its price, which is the single highest-value change for a long agent loop.

**19.** Self-hosting becomes competitive under **high, sustained utilisation**, because a reserved GPU bills whether or not it is busy while an API bills only for what you use — so the comparison turns on how full you can keep the hardware. It also wins with **predictable load**, since GPU capacity cannot be autoscaled quickly and bursty traffic forces you to provision for the peak; with a **specific model** nobody hosts, such as a fine-tune or an adapter you own (Week 23's Inference Endpoints case); and under **data-residency or compliance requirements** that rule out third-party inference regardless of cost. **What people forget to count:** engineering time, which is usually the largest line item and is invisible in a per-token comparison; **idle GPU hours**, which are the whole ballgame and are systematically underestimated because people model peak rather than average load; ops burden and on-call; and — the structural one — that **providers batch across all their customers while you can only batch across your own**, so their effective utilisation is inherently higher than anything you can achieve at your scale. The practical default is therefore to use the API and self-host only on measurement, not intuition.

**20.** **Diagnose first: this is a TTFT problem, not a TPOT problem.** The prompt is 8,000 tokens and the response is 200, so the request is dominated by prefill; the entire 3 seconds is essentially the model reading the prompt before it says anything. That immediately rules out the decode-side optimizations — speculative decoding, weight quantization, and a smaller model would all target the wrong phase and deliver almost nothing here. **Fix in this order.** **(1) Stream, if you aren't already.** This does not reduce TTFT but it is the only fix that helps the 200 tokens *after* the first, and if the product currently waits for the full response before rendering, that alone is a large perceived-latency win for a one-line change. **(2) Prompt caching.** With a large system prompt and a stable prefix this is the highest-value change available: cached prefill is dramatically faster as well as roughly ten times cheaper, and it attacks precisely the phase that is slow. Verify it works by checking cache-read token counts — and audit for the silent invalidators from S2, since a `datetime.now()` or a per-user ID in the system prompt would prevent any of it from caching. **(3) Shrink the prompt.** Ask how much of the 8,000 tokens is earning its place: is retrieval returning ten chunks where three would do (Week 10 — more chunks is a trade-off, not a free win), and has the system prompt accumulated stale scaffolding (S2)? Fewer prefill tokens is a direct linear reduction in TTFT. **(4) Order the prompt for caching**, putting stable content first and volatile content last, so the cacheable prefix is as long as possible. **(5) Only then consider serving-side work** — continuous batching if self-hosting, or checking whether an over-large batch size is inflating queueing delay ahead of prefill. **What I'd measure before any of it:** the split between queueing, prefill, and first-token emission, because "3s TTFT" could partly be queueing under load rather than prefill at all, and that would point at capacity rather than prompt size.

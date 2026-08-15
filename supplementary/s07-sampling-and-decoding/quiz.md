# S7 — Quiz (20 questions)

**Topic:** Sampling and Decoding
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The direct output of a transformer's final layer is:
- A) Text
- B) A logit per vocabulary token
- C) A probability distribution
- D) A single token ID

**2.** Temperature is applied:
- A) After softmax, to the probabilities
- B) To the logits, before softmax
- C) To the attention weights
- D) To the embedding matrix

**3.** Softmax is shift-invariant but not scale-invariant. This is why:
- A) Attention works
- B) Temperature changes the distribution's sharpness
- C) Logits can be negative
- D) Top-p is adaptive

**4.** The main weakness of top-k relative to top-p is:
- A) It's slower
- B) A fixed k ignores whether the model is confident or uncertain at that position
- C) It cannot be combined with temperature
- D) It requires sorting

**5.** With `min_p = 0.1` and a top token probability of 0.9, the cutoff is:
- A) 0.1
- B) 0.09
- C) 0.9
- D) 0.01

**6.** A **presence** penalty differs from a **frequency** penalty in that it is:
- A) Applied to logits rather than probabilities
- B) Flat and applied once, regardless of how many times the token appeared
- C) Larger in magnitude
- D) Applied only to the first token

**7.** Beam search is largely absent from LLM chat serving mainly because:
- A) It's mathematically incorrect
- B) Maximum-likelihood sequences are bland, and it's expensive and hostile to streaming
- C) It requires a special model architecture
- D) It cannot handle long contexts

**8.** On Claude Opus 5, sending `temperature=0`:
- A) Produces deterministic output
- B) Returns a 400 error — the parameter is removed
- C) Is silently ignored
- D) Enables greedy decoding

**9.** Constrained decoding guarantees valid JSON by:
- A) Retrying until it parses
- B) Masking illegal tokens' logits to −∞ using a schema state machine
- C) Post-processing the string
- D) Lowering the temperature

**10.** A stop reason of `max_tokens` means:
- A) The model finished naturally
- B) The output was truncated mid-generation
- C) A stop sequence matched
- D) The request failed

**11.** Generating synthetic training data (Week 13) generally wants:
- A) Temperature 0, for consistency
- B) Higher temperature, because coverage is the point
- C) Beam search
- D) Greedy decoding with a large k

**12.** Self-consistency (majority voting over sampled reasoning paths) requires:
- A) Temperature 0
- B) Non-zero temperature, so the paths actually differ
- C) Beam search with B = 5
- D) Constrained decoding

---

## Short answer

**13.** Trace what happens between the final hidden state and a printed token.

**14.** Explain what temperature actually does, and state the most common misconception about it.

**15.** Compare top-k and top-p using a concrete example of a peaked and a flat distribution.

**16.** Explain the three penalty types and why none should be used with structured output.

**17.** Explain why `temperature=0` never guaranteed identical outputs, and what to do instead.

**18.** Explain constrained decoding and why it is strictly stronger than prompting plus a retry loop.

**19.** Explain what changed on current Claude models regarding sampling parameters, why, and what a developer should do instead.

**20.** You're building a RAG-based support agent that must return JSON with an answer, a citation list, and an escalation flag. Users report occasional parse failures, occasional invented citations, and non-reproducible outputs. Diagnose and fix, in priority order.

---
---

## Answer key

**1. B** — A vector of unnormalised logits, one per vocabulary token; softmax and decoding happen afterwards, outside the model.

**2. B** — Logits are divided by T before softmax, which is why it changes sharpness rather than merely rescaling probabilities.

**3. B** — Adding a constant to every logit changes nothing, but multiplying (equivalently, dividing by T) changes the distribution, which is exactly the mechanism temperature exploits.

**4. B** — k is fixed while the distribution's true width varies by position.

**5. B** — 0.1 × 0.9 = 0.09; the threshold scales with the model's confidence.

**6. B** — Presence is a one-time flat penalty for having appeared at all; frequency scales with the count.

**7. B** — Bland output (likelihood ≠ quality for open-ended text), B× compute and KV cache, and no token can be emitted until a beam wins.

**8. B** — `temperature`, `top_p`, and `top_k` are removed and return 400 on Opus 5, Fable 5, Opus 4.8, and Opus 4.7.

**9. B** — Illegal tokens receive −∞ logits, so no sampled sequence can be malformed.

**10. B** — The output was cut off at the limit; treating it as complete is a common silent bug.

**11. B** — Diversity is the product; low temperature collapses the dataset into near-duplicates.

**12. B** — Identical samples cannot vote; the method depends on path diversity.

**13.** The final transformer layer produces a **hidden state** — a vector of size `d_model` (e.g. 4096) for the current position. That vector is multiplied by the **unembedding matrix** of shape `[d_model, vocab_size]` (often the tied transpose of the input embedding matrix), producing one **logit** per vocabulary token — say 128,000 unnormalised real-valued scores with no probabilistic meaning individually; only their relative values matter. **Softmax** then converts them into a probability distribution: `p_i = exp(z_i) / Σ exp(z_j)`, non-negative and summing to 1. **At this point the model's work is finished.** Everything after is the **decoding strategy**, which is a separate algorithm that is not learned and can be swapped at inference time: optionally mask illegal tokens to −∞ (constrained decoding), optionally divide logits by a temperature, optionally truncate to top-k / top-p / min-p, optionally apply repetition penalties, then either take the argmax (greedy) or sample from the surviving distribution. The chosen token is appended to the context and the entire process repeats for the next position.

**14.** Temperature rescales the logits **before** softmax: `p_i = exp(z_i / T) / Σ exp(z_j / T)`. Because softmax is not scale-invariant, dividing by a small T amplifies the gaps between logits and produces a **sharper, more peaked** distribution (as T→0, one token approaches probability 1 and behaviour becomes greedy); dividing by a large T compresses the gaps and produces a **flatter** distribution (as T→∞, it approaches uniform). At T=1 the distribution is the model's own. **The misconception to kill is that temperature makes the model "more creative," "smarter," or "more careful."** It does none of those things — it only changes *how sharply the model's existing preferences are expressed*. The ranking of tokens is unchanged at every temperature. If the model's knowledge is wrong, no temperature setting fixes it; if the correct answer is ranked second, raising temperature makes it more likely to be sampled but simultaneously makes a dozen wrong answers more likely too. It is a blunt scalar with no semantics, applied uniformly to every token regardless of what the model is doing at that position — which is precisely why the frontier is replacing it with prompt-level and effort-level controls.

**15.** **Top-k** keeps the `k` highest-probability tokens and discards the rest; **top-p (nucleus)** sorts descending and keeps the smallest set whose cumulative probability reaches `p`. The difference shows up when the distribution's true width changes. After **"The capital of France is"**, the distribution is extremely peaked — `" Paris"` might hold 0.98. Top-p at 0.9 keeps **one token**, correctly reproducing greedy behaviour where the model is certain. Top-k at 50 drags in 49 irrelevant candidates and, after renormalisation, hands them probability mass they never earned, so there is a real chance of sampling something wrong at a position where the model was not actually uncertain. After **"She opened the door and saw a"**, the distribution is genuinely flat — hundreds of nouns are plausible. Top-p at 0.9 might keep 200 tokens, preserving the real diversity; top-k at 50 arbitrarily amputates 150 valid options. **Top-p adapts to the model's own uncertainty and top-k does not**, which is why top-p became the default. One interaction worth knowing: temperature is applied to the logits *before* the nucleus is computed, so raising temperature also widens the nucleus — the knobs are not independent, and tuning both at once is a reliable way to confuse yourself.

**16.** **Repetition penalty** is the bluntest: it penalises the logit of any token already present, discouraging repeats regardless of how many. **Frequency penalty** scales with the **count** — each additional use increases the penalty, so pressure builds against phrases the model keeps reaching for. **Presence penalty** is **flat and applied once** if the token has appeared at all, which pushes the model toward new topics rather than new phrasings. Rule of thumb: frequency penalty for repeated *phrases*, presence penalty for refusing to leave a *topic*, and keep values small (roughly 0.1–1.0) because these penalties are indiscriminate. **They must not be used with structured output**, because valid structure is inherently repetitive: JSON keys repeat across array elements, field names repeat, delimiters repeat, and a schema may legitimately require the same token dozens of times. A penalty tuned for prose will actively suppress the tokens the schema requires — producing a model that is fighting its own output format. This is a real production bug: penalties are configured once for a chat use case, then inherited by a JSON-generating endpoint, where they cause intermittent malformed output that looks like model flakiness.

**17.** Greedy decoding is deterministic **given identical logits**, and that qualifier is where the trouble lives — real serving stacks do not reliably produce identical logits. **Floating-point arithmetic is not associative**, so `(a+b)+c ≠ a+(b+c)`, and GPU reductions sum in whatever order the scheduler produces. **Optimised kernels are batch-shape-dependent**, choosing different code paths and tiling depending on batch composition — and because continuous batching (S5) groups your request with whatever traffic arrives at that instant, your numerics depend on **other users' requests**. **Near-ties amplify all of this**: when the top two logits differ by 1e-7 any of these effects can flip the argmax, and a single flipped token changes the context for every subsequent step, so outputs diverge completely. **MoE routing** can also depend on batch composition, and **silent infrastructure changes** — kernel upgrades, hardware migration, model point-releases — shift numerics without changing the model name. **What to do instead:** stop relying on decoding-level determinism entirely. Pin the *shape* with **structured outputs and schemas** so format is guaranteed even when tokens vary; **cache responses** keyed on the full request where true reproducibility is required; **pin dated model snapshots** where the provider offers them; and **write evals that tolerate variance** — run n samples and report a confidence interval (S3) rather than asserting exact string equality, since an eval that breaks on a synonym was measuring the wrong thing.

**18.** Constrained decoding maintains a **grammar or schema state machine alongside generation**. At each step it computes which tokens are legal given the current state, sets the logits of all illegal tokens to **−∞**, and then samples normally — illegal tokens carry zero probability by construction. It is **strictly stronger than prompting plus retries** for a structural reason: prompting makes valid output *likely*, and a retry loop makes failure *less frequent*, but neither makes failure *impossible*, and both cost latency and tokens on every failure. Masking makes malformed output **unreachable** — there is no sequence of sampled tokens that produces invalid JSON, so the parse cannot fail. It also beats **assistant prefill**, which only constrains the beginning of the output and offers no guarantee thereafter (and which now returns 400 on current Claude models anyway). It composes cleanly with the rest of the pipeline, since masking happens before temperature and truncation. **The caveat is that constraining the format does not constrain the content**: a schema guarantees a well-formed `{"confidence": 0.93}` and guarantees nothing about whether 0.93 is meaningful. Over-tight schemas can even hurt quality by forcing a field the model has no basis for, which it will then fabricate — the S3 calibration lesson reappearing at the decoding layer.

**19.** On current frontier Claude models the sampling parameters have been **removed from the API**: `temperature`, `top_p`, and `top_k` **return a 400 error** on **Opus 5, Fable 5, Opus 4.8, and Opus 4.7**, and non-default values are rejected on **Sonnet 5**. **Assistant prefill** — seeding the start of the assistant turn to force a format — also returns 400 on these models. **The rationale** is that on models with extended thinking and strong instruction-following, steering belongs in the **prompt and in the effort/thinking configuration**, not in a scalar multiplier on the logit distribution. Temperature has no semantics: it applies uniformly to every token regardless of whether the model is at a position of genuine choice or one of certainty. Instructions, schemas, and reasoning budgets carry meaning and compose with the model's reasoning. **What a developer should do instead:** ask for the behaviour directly ("be terse", "give three distinct approaches"); raise the thinking/effort setting when more deliberation is wanted; use **structured outputs** (`output_config.format`) or strict tool schemas instead of prefill or temperature-0-plus-hope for format control; and for reproducibility, cache and pin rather than assume determinism. **Migration is deletion, not tuning** — code carrying `temperature=0` "for determinism" does not merely fail to help against these models, it fails the request. The concepts remain fully live for open-weight models served via vLLM, TGI, llama.cpp, or Ollama, for older API versions, and for other providers.

**20.** **Fix the parse failures first — they are the cheapest and most complete fix.** Replace prompt-level JSON requests with **constrained/structured decoding**: `output_config.format` with a JSON Schema, or a tool with `strict: true`. This makes malformed output structurally impossible rather than merely unlikely, and it removes the retry loop's latency and token cost. **Simultaneously check the stop reason on every response.** A `max_tokens` stop truncates mid-JSON, and it is the one failure mode schema enforcement cannot save you from — a citation list is unbounded in length, so this is a live risk here. Raise `max_tokens`, cap the citation count in the schema, and treat a `max_tokens` stop as an explicit error rather than a parseable result. **Also verify no frequency or presence penalty is configured**, since the citation array repeats keys and penalties will fight the schema intermittently. **Next, the invented citations — note that this is a grounding problem, not a decoding problem**, and no sampling setting fixes it. The schema guarantees a well-formed citation object and says nothing about whether the ID exists. The fixes are: **validate every returned citation ID against the retrieved chunk set in code**, rejecting or nulling any that was not actually retrieved; **require the model to quote the supporting span** so a citation is verifiable rather than merely present; **instruct it to answer only from the provided context and to say so when the context is insufficient**; and check whether retrieval is simply failing (S6, Week 9 — inspect retrieved chunks, not final answers, and measure recall@k). If the escalation flag is also unreliable, treat it the same way — it is a model judgement wearing a boolean's clothing, and it needs its own eval. **Finally, the non-reproducibility — reframe rather than fix.** Identical outputs were never guaranteed: floating-point ordering, batch-dependent kernels, and near-ties break it, and on current Claude models `temperature=0` returns a 400 anyway. Pin what actually matters — the schema guarantees the shape, citation validation guarantees the references, and a **response cache keyed on the full request** gives literal reproducibility where a caller truly needs it. Then **rewrite the evals to tolerate token-level variance** (S3): run n samples per case, assert on the validated fields and citation correctness rather than exact strings, and report a confidence interval. An eval that fails because the model chose a synonym was measuring the wrong thing, and chasing it would have sent you tuning decoding parameters that are not the cause of any of the three symptoms.

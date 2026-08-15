# S7 — Sampling and Decoding: How Tokens Actually Get Chosen

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 7 introduced temperature in a single line while covering transformer architecture, and Week 11 mentioned structured output as a harness concern. Neither explained what happens between the model's final layer and the token that appears on screen. That gap matters, because decoding is where a large fraction of "the model is bad at this" problems actually live.

**Fills the gap after:** Week 7 (Transformer Architecture), Week 11 (Harness/Context/Evals)
**Prerequisites:** Weeks 6–7 (neural networks, transformers, softmax), Week 8 (tool calling)

---

## 0. Why this lecture exists

A transformer does not output text. It outputs a **vector of numbers, one per token in the vocabulary**, for every position. Turning that vector into a token you can print is a separate algorithm that lives *outside* the model, is not learned, and can be changed at inference time without touching a single weight.

That algorithm is called the **decoding strategy**, and it is the last mile of every LLM. It determines whether your model is boring or unhinged, whether two identical requests return identical answers, whether JSON parses, and whether the model can be forced to stay inside a schema. It is also the part of the stack most commonly misunderstood — "just turn the temperature down" is the LLM equivalent of "have you tried turning it off and on again."

**One important framing before we start.** The concepts in this lecture are permanent — they describe how autoregressive generation works and apply to every open-weight model you will ever run locally, every older API, and every inference server (vLLM, TGI, llama.cpp, Ollama). But the *interface* has changed on the frontier: **current Claude models have removed sampling parameters from the API entirely** (§9). Learn the mechanics anyway. You need them to reason about model behaviour, to run open models, and to understand why the frontier moved away from these knobs.

---

## 1. From hidden state to probability distribution

At the final layer, for each position, the transformer produces a hidden state — a vector of size `d_model` (say 4096). To predict the next token, that vector is multiplied by an **unembedding matrix** (often the tied transpose of the input embedding matrix) of shape `[d_model, vocab_size]`.

```
hidden state:  [4096]
           ×   unembedding [4096, 128000]
           =   logits [128000]
```

The result is a **logit** per vocabulary token — an unnormalised score. Logits are real numbers, positive or negative, with no probabilistic meaning on their own. Only their *relative* values matter.

To turn logits into probabilities, apply **softmax**:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

This is the same softmax from attention (Week 7), used for a different purpose: there it produced attention weights over positions, here it produces a probability distribution over the vocabulary. The output is non-negative and sums to 1.

**Key property:** softmax is *shift-invariant* (adding a constant to every logit changes nothing) but **not scale-invariant**. Multiplying all logits by a constant changes the distribution's sharpness dramatically. That single fact is the entire basis for temperature.

At this point the model's job is finished. Everything that follows is the decoder's job.

---

## 2. Greedy decoding

The simplest strategy: **always take the highest-probability token.**

```python
next_token = argmax(logits)
```

Greedy decoding is deterministic in the mathematical sense — the same logits always produce the same token. It is the right default for tasks with one correct answer: classification, extraction, arithmetic, structured output, translation of a fixed phrase.

**Its weakness is that it is locally optimal and globally myopic.** Choosing the best token *now* can walk the model into a region where every continuation is poor, whereas a slightly worse token now might have led somewhere much better. There is no backtracking; the token is committed and becomes part of the context for every subsequent step.

**Its other weakness is repetition.** Greedy decoding is notorious for falling into loops — "I think that I think that I think that…" — because once a repetitive pattern begins, the pattern itself becomes the most probable continuation. This is a real, well-documented failure mode of purely deterministic decoding, and it is one of the reasons pure greedy is rarely used for long-form generation.

---

## 3. Temperature

Temperature rescales the logits **before** softmax:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

Work through what `T` does:

| T | Effect on logits | Effect on distribution | Behaviour |
|---|---|---|---|
| `T → 0` | Divided by a tiny number → gaps explode | One token approaches probability 1 | Equivalent to greedy |
| `T = 1` | Unchanged | The model's own distribution | "Raw" model |
| `T` between 0 and 1 | Gaps amplified | Sharper, peaked | More focused, more repetitive |
| `T > 1` | Gaps compressed | Flatter | More diverse, more errors |
| `T → ∞` | All logits → 0 | Uniform | Random token soup |

**The crucial misconception to kill:** temperature does not make the model "more creative" or "smarter" or "more careful." It only changes *how sharply the model's existing preferences are expressed*. If the model's knowledge is wrong, no temperature fixes it. If the model ranks the right answer second, a higher temperature makes it more likely to be *sampled*, but also makes twelve wrong answers more likely too.

**Temperature 0 is not truly deterministic in practice**, and this trips up almost everyone. See §10.

A worked example. Suppose three candidate tokens have logits `[3.0, 2.0, 1.0]`:

- `T = 1.0` → probabilities ≈ `[0.665, 0.245, 0.090]`
- `T = 0.5` → logits become `[6, 4, 2]` → ≈ `[0.867, 0.117, 0.016]`
- `T = 2.0` → logits become `[1.5, 1.0, 0.5]` → ≈ `[0.506, 0.307, 0.186]`

Same model, same knowledge, same ranking — only the confidence changes.

---

## 4. Top-k sampling

Temperature alone has a nasty property: it rescales the **entire** vocabulary, including the 100,000 tokens with near-zero probability. Raise the temperature enough and complete garbage becomes reachable, because the tail is enormous. Even a tiny probability, summed over 100,000 tokens, is a meaningful chance of sampling nonsense.

**Top-k truncation** fixes this: keep only the `k` highest-probability tokens, zero out everything else, renormalise, and sample from what remains.

```
k = 50  →  discard all but the top 50 candidates
```

This guarantees the model never samples from the long tail. Its weakness is that `k` is **fixed while the distribution is not**. Consider two positions:

- After `"The capital of France is"` the distribution is extremely peaked — `" Paris"` might hold 0.98. A `k` of 50 drags in 49 irrelevant candidates and hands them probability mass they never earned.
- After `"She opened the door and saw a"` the distribution is genuinely flat — hundreds of nouns are plausible. A `k` of 50 arbitrarily amputates valid options.

Top-k applies the same width to both cases. That mismatch motivates the next method.

---

## 5. Top-p (nucleus) sampling

**Top-p** keeps the smallest set of tokens whose cumulative probability reaches `p`, then renormalises and samples.

```
p = 0.9  →  sort descending, accumulate until the running total ≥ 0.9, keep those
```

The candidate set is now **adaptive**. On `"The capital of France is"`, `" Paris"` alone reaches 0.9, so the set is one token and behaviour is effectively greedy. On `"She opened the door and saw a"`, reaching 0.9 might take 200 tokens, and all 200 stay in play. The width follows the model's own uncertainty, which is exactly what you want.

Top-p is the most widely used sampling method in open models, and typical values sit between 0.9 and 0.95.

**Temperature and top-p interact, and order matters.** In the standard pipeline, temperature is applied to the logits *first*, then top-p truncation is computed on the resulting distribution. This means raising the temperature also widens the nucleus — the two knobs are not independent. Tuning both simultaneously is a common way to get confusing results; the usual advice is **change one, hold the other at its default**.

---

## 6. Min-p, and the other tail cutters

**Min-p** is a newer method that sets the threshold *relative to the top token*: keep every token whose probability is at least `min_p × p_max`.

With `min_p = 0.1`:
- If the top token has probability 0.9, the threshold is 0.09 — only genuinely strong candidates survive, so a confident model stays confident.
- If the top token has probability 0.15, the threshold is 0.015 — many tokens survive, so an uncertain model stays open.

The claimed advantage over top-p is better behaviour at high temperature: because the threshold scales with the model's confidence, min-p keeps the distribution coherent even when temperature is pushed up for creative work. It is supported in llama.cpp, vLLM, and most local-model tooling. It is not available on the major closed APIs.

Other variants you may encounter in open-model tooling — typical sampling, tail-free sampling, eta/epsilon cutoff, Mirostat — are all attempts at the same problem: **cut the unreliable tail without amputating genuine options.** You do not need them for production work, but recognise them as members of one family rather than unrelated magic.

---

## 7. Repetition, frequency, and presence penalties

Even with good truncation, models loop. Three penalty mechanisms address this, and they are routinely confused with one another.

| Penalty | Applied to | Scaling | Effect |
|---|---|---|---|
| **Repetition penalty** | Logits of tokens already present | Multiplicative/divisive on the logit | Blunt; discourages any repeat |
| **Frequency penalty** | Logits of already-used tokens | Proportional to **how many times** used | Increasing pressure with each reuse |
| **Presence penalty** | Logits of already-used tokens | Flat, applied **once** if used at all | Pushes toward new topics |

Use **frequency penalty** when the model repeats *phrases*. Use **presence penalty** when the model refuses to leave a *topic*. Values are typically small (0.1–1.0 for the OpenAI-style penalties); large values cause visible damage, because these penalties are indiscriminate and will suppress function names, required keywords, and legitimately repeated structure like JSON keys.

**Do not use penalties with structured output.** If the model must emit `{"name": ..., "name": ...}` inside a repeated array, a frequency penalty actively fights the schema. This is a real production bug: penalties tuned for prose break JSON generation.

---

## 8. Beam search — and why LLMs don't use it

**Beam search** maintains `B` partial sequences ("beams") at every step, expands each, and keeps the `B` highest-scoring continuations by total sequence log-probability. It is a partial fix for greedy decoding's myopia: a token that looks bad now can survive if it leads somewhere good.

Beam search dominated **machine translation** and **summarisation** for years, and still works well there. So why is it essentially absent from modern LLM serving?

1. **It produces bland text.** The highest-total-probability sequence is, almost by definition, the most generic one. Human text is *not* maximum-likelihood — real writing constantly takes locally improbable turns. Optimising for total probability optimises for cliché. This is the well-known observation that likelihood and quality diverge for open-ended generation.
2. **It is expensive.** `B` beams means roughly `B×` the compute and `B×` the KV cache. In a continuous-batching server (S5), holding `B` divergent sequences per request destroys the throughput model.
3. **It fights streaming.** You cannot emit a token until you know which beam wins, so the first-token latency advantage of streaming disappears.
4. **Tasks with one right answer don't need it.** Where beam search would help most — deterministic extraction — greedy plus a good prompt is usually already correct.

The rule of thumb: **beam search for constrained sequence-to-sequence tasks, sampling for open-ended generation.** Modern LLM APIs are built for the second case.

---

## 9. What the frontier APIs actually expose today

Everything above describes the mechanics of decoding. But the interface on frontier models has changed, and stating this accurately matters more than teaching a knob that now returns an error.

**On current Claude models, sampling parameters have been removed from the API.**

- `temperature`, `top_p`, and `top_k` are **removed and return a 400 error** on **Claude Opus 5, Fable 5, Opus 4.8, and Opus 4.7**.
- On **Claude Sonnet 5**, non-default values for these parameters are rejected.
- **Assistant prefill** — the trick of seeding the start of the assistant turn to force a format — also returns a 400 on these models.

This is not a bug or a temporary regression. The reasoning is that on models with extended thinking and strong instruction-following, **steering belongs in the prompt and in effort/thinking configuration, not in the logit distribution.** If you want terse output, ask for terse output. If you want more exploration, describe the exploration you want. If you want more deliberation, raise the thinking/effort setting. These are semantic controls, and they compose with the model's reasoning in a way that a scalar multiplier on the logits never could.

**Practical consequences:**

- Code that sets `temperature=0` "for determinism" against a current Claude model does not merely fail to help — it **fails the request**. Migrating older code means deleting these parameters, not tuning them.
- Prompt-level format control replaces prefill. Where you would once have prefilled `{`, you now use **structured outputs** (`output_config.format`) or a tool schema with `strict: true`, which is a stronger guarantee anyway.
- The concepts remain fully live everywhere else: open-weight models served through vLLM, TGI, llama.cpp, or Ollama expose the full parameter set, as do older API versions and most non-Anthropic providers. If you fine-tune and self-host (Weeks 13–16), decoding parameters are yours to set and you need to understand them.

**The transferable lesson** is that decoding parameters were always a weak, indirect steering mechanism — a scalar with no semantics, applied uniformly to every token regardless of what the model was doing at that position. The frontier is replacing them with mechanisms that carry meaning: instructions, schemas, and reasoning budgets.

---

## 10. Determinism, and why `temperature=0` was never a guarantee

A recurring production surprise: even at temperature 0, on APIs that supported it, identical requests could return different outputs. Greedy decoding is deterministic *given identical logits*, and the qualifier is where the trouble lives.

**Sources of non-determinism in a real serving stack:**

1. **Floating-point non-associativity.** `(a + b) + c ≠ a + (b + c)` in floating point. GPU reductions sum in whatever order the scheduler produces, which varies with block scheduling and with batch composition.
2. **Batch-dependent kernels.** Optimised kernels select different code paths and tiling depending on batch shape. Since continuous batching (S5) means your request is batched with whatever else arrives at that moment, the numerics depend on **other users' traffic**.
3. **Near-ties.** When the top two logits differ by 1e-7, any of the above can flip the argmax. One flipped token changes the context for everything after it, and the outputs diverge completely — a genuine butterfly effect.
4. **Mixture-of-Experts routing.** In MoE models, expert selection can depend on batch composition, so the computation itself differs.
5. **Silent infrastructure changes.** Kernel upgrades, hardware migrations, and model point-releases all shift numerics without changing the model name you requested.

**How to actually get reproducibility:**

- **Don't rely on decoding-level determinism at all.** Pin behaviour with **structured outputs** and schemas, so the *shape* is guaranteed even when tokens vary.
- **Cache responses** keyed on the full request when you truly need identical results.
- **Write evals that tolerate variance** — this is S3's point: run n samples and report a confidence interval instead of asserting exact string equality. An eval that breaks on a synonym was measuring the wrong thing.
- **Pin model versions** where the provider offers dated snapshots.

---

## 11. Constrained and structured decoding

This is the most practically valuable idea in the lecture, and it is *not* a sampling parameter — it is a **mask on the logits**.

The mechanism: maintain a grammar or schema state machine alongside generation. At each step, compute which tokens are *legal* given the state, set the logits of every illegal token to `-∞`, then sample normally. Illegal tokens have zero probability by construction.

Consequences worth internalising:

- **The output is valid by construction, not by luck.** If a JSON schema is enforced this way, a parse failure is impossible — there is no sequence of sampled tokens that produces malformed JSON.
- **It is strictly better than "please respond in JSON" plus a retry loop.** Retries cost latency and tokens and still fail sometimes. Masking cannot fail in that way.
- **It also beats prefill**, which only constrains the *beginning* of the output and offers nothing thereafter.
- **It composes with everything else.** Masking happens before temperature and truncation, so the two are orthogonal.

In practice you meet this as:
- **Structured outputs** on the Messages API (`output_config.format` with a JSON Schema), and `messages.parse()` for validated responses.
- **Strict tool schemas** (`strict: true`), which apply the same guarantee to tool-call arguments — directly relevant to Week 8's function calling and Week 21's LangGraph state machines.
- **Grammar-constrained generation** in open-model tooling: GBNF in llama.cpp, Outlines, XGrammar, and guidance, which accept arbitrary context-free grammars, not just JSON.

**The caveat worth stating:** constraining the format does not constrain the *content*. A schema guarantees you get a well-formed `{"confidence": 0.93}`; it guarantees nothing about whether 0.93 is meaningful. Over-tight schemas can also hurt quality by forcing the model into a shape it would not naturally produce — if the model must emit a field it has no basis for, it will fabricate one. This is the S1 lesson about calibration reappearing: a number in the right slot is not a number you can trust.

---

## 12. Stop sequences, length control, and logprobs

**Stop sequences** terminate generation when a specified string appears. They matter because generation otherwise runs until the EOS token or `max_tokens`, and a model that starts hallucinating a fresh `Human:` turn will happily continue the conversation with itself. Note that the stop string is typically **not included** in the returned text, and that stop-sequence matching operates on the decoded string — so a stop sequence that does not align with token boundaries may behave unexpectedly.

**`max_tokens`** is a hard cut, not a request for brevity. Hitting it truncates mid-sentence and, worse, mid-JSON. Always check the stop reason: `end_turn` means the model finished, `max_tokens` means you cut it off. Treating a truncated response as a complete one is a classic silent bug — and in a structured-output pipeline, it is the one failure mode masking cannot save you from.

**Logprobs** — where an API exposes them — return the log-probability of the chosen token and often the top alternatives. They are the raw material for:
- **Confidence estimation** (with the caveat that token probability is not calibrated truth — see S3),
- **Detecting uncertainty** as an escalation signal in an agent loop,
- **Perplexity** computation for evaluation,
- **Cheap classification**, by comparing the logprob of `" yes"` against `" no"` at a single position instead of generating and parsing text.

Availability varies sharply by provider and model; treat logprobs as a capability to check for, not assume.

---

## 13. Choosing settings, when you have the choice

For open models and any API that still exposes these parameters:

| Task | Suggested setting | Reasoning |
|---|---|---|
| Extraction, classification, structured output | Greedy (`T→0`), plus schema constraint | One correct answer; diversity is pure risk |
| Code generation | Low `T` (0.0–0.2) | Syntax is unforgiving; small deviations break compilation |
| Factual Q&A / RAG answering | Low `T` (0.0–0.3) | You want the grounded token, not an interesting one |
| Summarisation | Low–mid `T` (0.3–0.5) | Slight variation reads better; content must stay faithful |
| Conversational assistant | Mid `T` (0.7) with `top_p` 0.9 | Natural variation without incoherence |
| Brainstorming, creative writing | High `T` (0.9–1.2) with `top_p` 0.95 or min-p | Diversity is the product |
| Generating synthetic training data (W13) | Higher `T` deliberately | Low temperature collapses diversity and produces a degenerate dataset |
| Self-consistency / majority voting | Non-zero `T`, n samples | Determinism defeats the entire method — identical samples cannot vote |

Two entries deserve emphasis because they invert the usual instinct. **Synthetic data generation wants high temperature**: Week 13 showed that a synthetic dataset's value lies in its coverage, and greedy decoding produces thousands of near-identical examples that teach the model nothing. **Self-consistency wants non-zero temperature** for the same reason — sampling several reasoning paths and taking the majority answer requires the paths to actually differ.

And on frontier Claude models, none of this is configurable — you express the same intent in the prompt and in the effort setting.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 6–7** (neural nets, transformers) | Softmax reappears: attention weights over positions, then probabilities over vocabulary. Same function, different axis. |
| **Week 7** (temperature, one line) | This lecture is the full version of that line, including why it is disappearing from frontier APIs. |
| **Week 8** (function calling) | Strict tool schemas are constrained decoding — the argument JSON is valid by construction, not by prompt-begging. |
| **Week 9** (RAG) | Low temperature for answer generation; the retrieved context is the source of truth, and creativity is a liability. |
| **Week 11** (harness, evals) | `max_tokens` truncation and non-determinism are harness bugs that masquerade as model quality problems. Evals must tolerate token-level variance (S3). |
| **Week 13** (synthetic data) | Diversity is generated by decoding. Low temperature yields a degenerate dataset. |
| **Weeks 15–16** (RL, GRPO) | RL fine-tuning requires *sampling multiple rollouts* per prompt — deterministic decoding makes group-relative advantage meaningless because every rollout would be identical. |
| **Week 21** (LangGraph) | Structured output is what makes graph routing reliable; a router that must parse free text is a router that will eventually misroute. |
| **S3** (evaluation) | Non-determinism is why evals need n samples and confidence intervals, not exact-match assertions. |
| **S5** (inference optimisation) | Continuous batching is *why* determinism fails — your numerics depend on co-batched traffic. Speculative decoding operates on this same logit pipeline. |

---

## Key takeaways

1. **The model outputs logits, not text.** Decoding is a separate, unlearned algorithm that converts a distribution into tokens, and it can be changed without touching the weights.
2. **Temperature rescales logits before softmax.** It changes how sharply existing preferences are expressed. It does not add knowledge, care, or creativity.
3. **Top-k truncates to a fixed count; top-p truncates to a fixed probability mass.** Top-p adapts to the model's own uncertainty, which is why it won.
4. **Penalties come in three flavours** — repetition (blunt), frequency (scales with count), presence (flat, once). Never apply them to structured output.
5. **Beam search lost** because maximum-likelihood sequences are bland, expensive, and hostile to streaming. It survives in translation, not in chat.
6. **On current Claude models, `temperature`/`top_p`/`top_k` are removed and return 400**, and prefill does too. Steering moved to prompting and effort. The concepts still apply everywhere else.
7. **`temperature=0` never guaranteed determinism** — floating-point ordering, batch-dependent kernels, and near-ties all break it. Design for variance instead of assuming it away.
8. **Constrained decoding masks illegal tokens to `-∞`**, making invalid output structurally impossible. It is strictly stronger than prompting plus retries, and it composes with everything else.
9. **Always check the stop reason.** `max_tokens` truncation is a silent, common, expensive bug.
10. **Match the strategy to the task.** Deterministic for extraction and code, diverse for brainstorming, synthetic data, and RL rollouts — and note that the last two *require* diversity to function at all.

---

## Glossary

| Term | Definition |
|---|---|
| **Logit** | Unnormalised score per vocabulary token, output by the final layer before softmax |
| **Softmax** | Function converting logits to a probability distribution summing to 1 |
| **Decoding strategy** | The algorithm converting a probability distribution into a chosen token |
| **Greedy decoding** | Always select the argmax token |
| **Temperature** | Divisor applied to logits before softmax; controls distribution sharpness |
| **Top-k** | Keep only the k highest-probability tokens before sampling |
| **Top-p / nucleus** | Keep the smallest set of tokens with cumulative probability ≥ p |
| **Min-p** | Keep tokens with probability ≥ min_p × (probability of the top token) |
| **Repetition penalty** | Blunt logit penalty on any already-used token |
| **Frequency penalty** | Penalty proportional to how often a token has been used |
| **Presence penalty** | Flat one-time penalty on any token already used |
| **Beam search** | Maintain B partial sequences, keep the highest total log-probability |
| **Constrained decoding** | Masking illegal tokens to −∞ using a grammar or schema state machine |
| **Structured output** | API feature guaranteeing responses conform to a supplied JSON Schema |
| **Stop sequence** | String that terminates generation when produced |
| **Stop reason** | Why generation ended — `end_turn` (natural) vs `max_tokens` (truncated) |
| **Logprob** | Log-probability of a token; used for confidence, perplexity, cheap classification |
| **Self-consistency** | Sample multiple reasoning paths and take the majority answer; requires non-zero temperature |

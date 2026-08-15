# Decision Cheat Sheet

One page of the decisions you'll actually face, with the reasoning compressed. Each row points to the lecture that explains it.

**How to use this:** find your question, read the rule, follow the pointer if you need the why. Every rule here is defended somewhere in `notes/` or `supplementary/` — this file is a lookup table, not an argument.

---

## 1. Should I use an LLM at all?

| Situation | Do this | Why | Ref |
|---|---|---|---|
| Tabular data, structured features | **Gradient boosting** (XGBoost/LightGBM/CatBoost) | Still wins on tabular. An LLM is slower, costlier, and worse | S1 |
| Well-defined classification with labels | **Train a small classifier** | Cheaper, faster, more accurate, calibrated | S1 |
| Text with no labels, fuzzy task | LLM | This is what it's for | W8 |
| Deterministic transformation | **Write code** | An LLM is a nondeterministic way to do a solved problem | S1 |
| High volume, low complexity | Small model, or code | Cost scales with volume | S5 |

**The failure to avoid:** reaching for an LLM because it's available, when a 200-line scikit-learn script would be better on every axis.

---

## 2. Prompt, RAG, or fine-tune?

| You need to change… | Use | Why | Ref |
|---|---|---|---|
| **Knowledge** (facts, docs, data) | **RAG** | Knowledge changes; weights are stale the moment you train | W9, W12 |
| **Behaviour** (format, style, refusals) | **Fine-tune (SFT)** | Behaviour is what imitation learning teaches | W13 |
| **Both** | RAG + light SFT on grounded answers | Retrieval supplies facts; SFT supplies house style | W12–13 |
| **Neither yet** | **Better prompt** | Always try this first; it's free and instant | S2 |
| Preferences / subjective quality | DPO or RLHF | Needs comparisons, not demonstrations | W14 |
| Verifiable correctness (code, maths) | RLVR / GRPO | Verifiable reward beats a learned one | W15 |

**Order of escalation, always:** prompt → few-shot → RAG → SFT → preference tuning → RL.
**Never skip ahead.** Each step is 10× the effort of the previous and most problems stop at step 2 or 3.

---

## 3. Fine-tuning method

| Constraint | Method | Ref |
|---|---|---|
| One consumer GPU, 7–13B model | **QLoRA** | W12, S10 |
| Multiple GPUs, want quality | **LoRA** | W12 |
| Need to change deep behaviour, have a cluster | Full fine-tune | W12 |
| New domain vocabulary/knowledge at scale | **CPT** then SFT | W12 |
| Multiple customers/tasks, one base | **LoRA adapters**, swapped at serve time | W12 |

**Why QLoRA is transformative, not merely convenient:** frozen base weights produce **no gradients and no optimizer states** — the two terms that dominate training memory. An ~84 GB problem becomes a few GB. (S10 §3)

---

## 4. Fixing a bad RAG system — in priority order

| # | Check | Why it's here | Ref |
|---|---|---|---|
| 0 | **Build an eval set. Inspect retrieved chunks, not answers** | The LLM papers over bad retrieval. Without this you're guessing | W9, S3 |
| 1 | Chunk size > embedding model max length? | Silent truncation; total failure that looks like quality | S6 |
| 2 | Query/document **instruction prefixes** correct? | Silent recall loss, one-line fix | S6 |
| 3 | ANN `ef_search`/`nprobe` too low? | Looks exactly like a bad model | S6 |
| 4 | Failures on **exact identifiers** (codes, IDs, names)? | → **Hybrid search + BM25 + RRF**. Architectural, not tunable | W9, S6 |
| 5 | Failures need cross-document context? | → **Contextual retrieval** (~67% fewer failures) | W9 |
| 6 | Chunking strategy | Where 80% of RAG systems fail | W9 |
| 7 | Add **reranking** (cross-encoder over top 10–20) | Highest-value precision upgrade | S6 |
| 8 | Only now: fine-tune the embedding model | Requires re-embedding the whole corpus | S6 |

**Evaluate over the query set, not one example** — W9's own hybrid demo improved on average while ranking one query *worse*.

---

## 5. Agent, workflow, or single call?

| Task shape | Use | Ref |
|---|---|---|
| One input → one output | **Single call** | S2 |
| Fixed known steps | **Workflow** (you write the control flow) | W21 |
| Open-ended, model decides the path | **Agent** | W8 |
| Complex + high value + errors recoverable | Agent | W8 |
| Any of those four missing | Drop a tier | W8 |

**The four criteria for "should this be an agent":** complexity, value, viability, cost of error. **If any answer is no, use something simpler.**

---

## 6. Multi-agent: yes or no?

**Only three valid reasons.** If none apply, you're paying the context tax for nothing.

| Reason | Test | Ref |
|---|---|---|
| **Parallelism** | Can subtask B start before A finishes? If not, it's a workflow | S12 |
| **Context isolation** | Would a single agent's context fill with exploration noise? | S12, W10 |
| **Independent perspective** | Does the checker need to *not* be anchored by the author? | S12 |

**Before any of it: fix the harness** — tool descriptions, error text, context curation, tool count. (W17)

| Rule | Why |
|---|---|
| Shallow and wide > deep and narrow | 0.9^5 = 59%; telephone game compounds |
| Pass constraints **verbatim** | Summaries drop exactly the constraints that matter |
| Make **uncertainty a required field** | Hedging is the first casualty of compression |
| Cap depth **in code**, not the prompt | Prompts don't stop runaway loops |
| Orchestrator–worker by default | Debate is the most appealing, least reliable pattern |
| Budget 15–20×, not 10×, for a 10-worker fan-out | Per-agent prompts + returned results + billed failures |

---

## 7. Decoding settings

⚠️ **On Claude Opus 5, Fable 5, Opus 4.8/4.7, `temperature`/`top_p`/`top_k` return 400** — they're removed. Non-default values rejected on Sonnet 5. Prefill also 400s. Steer via **prompt + effort** instead. The table below applies to open models and other APIs. (S7)

| Task | Setting | Why |
|---|---|---|
| Extraction, classification | Greedy + **schema constraint** | One right answer; diversity is pure risk |
| Code | T 0.0–0.2 | Syntax is unforgiving |
| RAG answering | T 0.0–0.3 | Want the grounded token |
| Summarisation | T 0.3–0.5 | Faithful, slightly varied |
| Chat | T 0.7, top_p 0.9 | Natural without incoherent |
| Brainstorming | T 0.9–1.2 | Diversity *is* the product |
| **Synthetic training data** | **High T** | Low T collapses the dataset into duplicates |
| **Self-consistency / voting** | **Non-zero T** | Identical samples can't vote |

**Never use frequency/presence penalties with structured output** — JSON keys repeat legitimately.
**`temperature=0` never guaranteed determinism** — FP non-associativity, batch-dependent kernels, near-ties.
**Always check the stop reason.** `max_tokens` = truncated, and it's the one thing schemas can't save you from.

---

## 8. Cutting inference cost

In order of leverage:

| # | Lever | Typical effect | Ref |
|---|---|---|---|
| 1 | **Prompt caching** | Huge, if the prefix is stable | S5 |
| 2 | **Order context: stable first, volatile last** | Makes caching work at all | S5 |
| 3 | **Route by difficulty** — small model for easy cases | Often the biggest single win | S5 |
| 4 | **Batch API** for non-interactive work | Large discount | S5 |
| 5 | Shorten prompts / trim few-shot examples | Linear | S2 |
| 6 | Cap `max_tokens`; ask for terse output | Output tokens cost most | S5 |
| 7 | Downscale images; crop don't resize | Cost scales with **area** | S9 |
| 8 | Quantisation (self-hosted) | Memory + throughput | S10 |

**Metrics vocabulary:** TTFT (time to first token), TPOT (time per output token), **goodput** (useful throughput under latency SLO — the one that matters).

---

## 9. Which eval do I trust?

| Signal | Trust level | Why | Ref |
|---|---|---|---|
| Your own held-out private data | **High** | The only thing you can verify is clean | S3, S11 |
| Recent benchmark, documented pipeline | Medium | Less time for leakage | S11 |
| Older public benchmark | **Low** | Contamination is near-guaranteed at web scale | S11 |
| Vendor-reported numbers | **Low** | Missing baselines, chosen benchmarks | W18, W20 |
| A single example that worked | **Zero** | This is not evidence | S3 |

**Statistics that matter:** run **n samples**, report a **confidence interval**, use **McNemar's test** for paired comparisons. A 2-point gain on 100 examples is noise. Never assert exact string equality — an eval that breaks on a synonym was measuring the wrong thing.

---

## 10. Safety checklist

**The lethal trifecta** — if a system has all three, it's exploitable:
1. Access to private data
2. Exposure to untrusted content
3. Ability to communicate externally

| Control | Note | Ref |
|---|---|---|
| **Break the trifecta** | Separate sessions/agents/credentials. The strongest control | S4 |
| **Least-privilege credentials** | Enforcement in the credential beats enforcement in the prompt | S4, S8 |
| **Approval on consequential actions** | Show **actual arguments**, not just the tool name | S4 |
| **Treat all tool output as untrusted** | Issue bodies, web pages, DB rows, screenshots, PDFs | S4, S9 |
| **Pin MCP server versions** | Approval at install ≠ approval of future versions (rug pulls) | S8 |
| **Log everything** | Arguments and results, not just calls | W23 |

**Computer-use agents assemble the trifecta by default.** Sandbox them. (S9, W25)

---

## 11. OOM while fine-tuning — in order

| # | Lever | Targets |
|---|---|---|
| 1 | **LoRA / QLoRA** | Eliminates gradient + optimizer terms entirely |
| 2 | **Gradient checkpointing** | Activations; ~30% slower, one line |
| 3 | Smaller batch + gradient accumulation | Activations |
| 4 | Confirm **bf16** is on | Compute memory; bf16 > fp16 (no overflow) |
| 5 | **Check max sequence length** | Activations scale with it; often an inherited config bug |
| 6 | 8-bit optimizer | Optimizer states |
| 7 | FSDP across GPUs | Everything; most complex, usually last |

**The arithmetic:** training needs **~6× the weights** — params + gradients + optimizer states + activations. A 7B model = 14 GB weights but ~84 GB to train. (S10 §3)

---

## 12. Reading a model card

| Term | What it actually means |
|---|---|
| "8×7B MoE" | ~47B **total** (experts share attention), ~13B **active** |
| **Total params** | What you need to **host** it |
| **Active params** | What it costs to **run** per token |
| Context window | Architectural capacity — **not** usable context (W10) |
| Benchmark scores | See §9. Ask if the data pipeline is documented |

---

## 13. Data — the rules invert by stage

| Stage | Rule | Ref |
|---|---|---|
| **Pretraining** | More is better — *after* dedup and filtering | S11 |
| **Instruction tuning (SFT)** | **Quality ≫ quantity.** ~1,000 great examples beat 100,000 mediocre | S11 |
| **Preference data** | Hard pairs teach most; agreement is a hard ceiling | W14, S11 |

**Universal:** deduplicate (improves models *at fixed compute*), read 100 examples by hand, hold out an eval set before you start, track provenance.

---

## 14. Universal debugging order

Applies to almost any LLM system that's underperforming:

1. **Look at the actual inputs and outputs.** Not metrics — the raw text. Most bugs are visible in ten examples.
2. **Check the boring things first** — truncation, wrong stop reason, missing prefix, max length, wrong model name.
3. **Is it retrieval or generation?** Inspect the retrieved chunks (W9).
4. **Is it the model or the harness?** Bad tool descriptions and opaque errors look exactly like a dumb model (W17).
5. **Is it context rot?** Does quality degrade *through* the trajectory? (W10)
6. **Only then** consider a different model or fine-tuning.

**Grounding beats planning depth** — compounding error means long plans are built on drifted state. Act, observe, re-plan. (S13, W17)

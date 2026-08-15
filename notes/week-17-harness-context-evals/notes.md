# Week 17 — Harness, Context, and Evals

**Subtitle:** Engineering systems around model intelligence
**Date:** 09/05/2026
**Sources:** `downloads/week-17-harness-context-evals.pdf` (28 slides) · `downloads/week-17-harness-context-evals.ipynb` (61 cells)
**Notion page:** https://100xschool.notion.site/362ffffa33e58071a92fdc19d999eb77
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=Ipy3w_uTl5O62C70bK_0P,5ZCPqYMnoHVsV_ur8uyaMg)

> **Three sides of the same problem** — what wraps the model · what goes in the window · how we know it works.

---

## 0. The idea in plain language

Weeks 1–16 were about the **model**. This week is about **everything around it** — and the claim is that the surroundings matter more than people assume.

**The evidence, and it's a good hook:** take one model, hold it completely constant, and run it inside three different agent scaffolds. The scores differ by several points. **Same model, same weights, same everything — different wrapper, different capability.**

Which means two things. If you're comparing models by benchmark scores, you're partly comparing *harnesses*. And if you want your product to get better, the harness is a lever **you control**, unlike the model.

The lecture is three angles on one problem:

**1. Harness — what wraps the model.** The tools it has, how errors are reported back, how the loop is structured. The recurring finding: an agent's failures are usually **harness failures wearing a model costume**. A vague tool description means the model picks the wrong tool. An opaque stack trace means it can't recover. Fix those and the "dumb model" gets noticeably smarter.

**2. Context — what goes in the window.** Week 10 showed context rot is real and follows from attention being a fixed budget. The response here is a discipline: **just-in-time context** rather than pre-loading everything, at the **right altitude** — enough detail to act, not so much that it drowns. Every token you add is a bet against accuracy.

**3. Evals — how you know any of it worked.** And this is where most teams are weakest.

**The eval section is the part worth reading twice**, because it names three specific ways people fool themselves:

- **Generic metrics are a mirage.** "Helpfulness: 7.2/10" tells you nothing actionable. It doesn't say what's broken or what to fix.
- **The 5-star lie.** Ask a model to rate something 1–5 and you get numbers that look like data and aren't. The scale isn't calibrated, and small differences are noise.
- **`pass@k` vs `pass^k`** — the distinction that matters most in practice. `pass@k` asks *"did it succeed at least once in k tries?"* — which flatters a model wildly. `pass^k` asks *"did it succeed **every** time?"* Production users get one try. **If you're reporting pass@k, you're reporting a number your users will never experience.**

The constructive answer is **critique shadowing**: instead of inventing a rubric in the abstract, have a domain expert review real outputs, write down *why* each is good or bad, and build the rubric from what they actually said. Then check your LLM judge against their labels — an unaligned judge is a random number generator with good grammar.

**The single sentence to carry out of this week:** when an agent underperforms, the first question isn't "is the model good enough?" — it's **"can the model see what it needs, and can it tell what went wrong?"**

---

## The motivating fact

**Terminal-Bench 2.0 — same model, different harness:**

| Harness | Score | Model |
|---|---|---|
| **ForgeCode** | **79.8%** | Claude Opus 4.6 |
| **Capy** | **75.3%** | Claude Opus 4.6 |
| **Claude Code** | lower | Claude Opus 4.6 |

> **Why don't they get the same score?**

A ~4.5-point spread (and more) from the **scaffolding alone**, with the model held constant. If you're evaluating models by benchmark scores, you're partly evaluating harnesses — and if you want to improve your product, the harness is a lever you actually control.

---

# BEAT 1 — Harness

## The premise

> ### **Agent = Model + Harness**
>
> **If you're not the model, you're the harness.**

That second line is the career-relevant one. Almost nobody reading this trains frontier models; **everybody** builds harnesses. This is where the addressable work is.

## Harness primitives

> **Each one patches a specific model deficiency.**

| Primitive | Deficiency it patches |
|---|---|
| **1. Filesystem** | No durable state |
| **2. Bash + code** | No general-purpose action |
| **3. Sandbox** | Unsafe execution at scale |
| **4. Sub-agents** | Context contamination |
| **5. Middleware** | No deterministic interventions |

**This is the right way to think about tools.** Every primitive exists because of something the model *cannot do*, not because it seemed useful. Filesystem because the model is stateless (Week 8); sub-agents because context is finite and gets polluted (Week 10); middleware because a model can't be relied upon to do something *every single time* — for that you need code.

## Model-harness-fit

> **The harness is part of the model's effective parameters.**

| CODEX / OpenAI | CLAUDE / Anthropic |
|---|---|
| `apply_patch` (diff) | `Edit` (old/new string) |
| `<oai-mem-citation>` | `<system-reminder>` |
| `exec_command_tool` | `Bash` + `Monitor` |

> **Same operations. Different vocabularies. Baked in by post-training.**

The models were *trained* on these specific formats. A diff-based edit tool and a string-replacement edit tool accomplish the same thing, but each model is fluent in its own — using the other is out-of-distribution.

### Three consequences

| | |
|---|---|
| **1. No model-agnostic agent.** | Honest version: per-model harness. **You pick a product, not a model.** |
| **2. Mid-chat model swaps break.** | Transcript OOD, cache miss, tool-surface mismatch — **all at once.** |
| **3. The matched pair shifts.** | **Yesterday's load-bearing scaffold is today's dead weight.** |

Consequence 2 is worth unpacking: swapping models mid-conversation means the transcript contains tool calls in the *other* model's vocabulary (out of distribution), the prompt cache is invalidated (latency spike), and the tool surface may not even match. Three failure modes at once.

Consequence 3 has a direct implication: **as models improve, scaffolding built to compensate for old weaknesses becomes overhead.** A retry loop for a model that no longer fails, a formatting nudge for a model that now formats correctly — pure cost.

> **Beat 1 takeaways:**
> — **Primitives are derived, not invented.**
> — **Match the pair: harness ↔ model.**
> — **Scaffolding goes stale — delete code.**

## Hands-on: add primitives, watch failures fix

1. **Tool-result clearing** — fix output bloat
2. **Loop detection** — break out of doom loops
3. **Sub-agent dispatch** — protect parent context

All three are context-management fixes, which is exactly why Beat 2 follows.

---

# BEAT 2 — Context

## The problem: context rot

> **Frontier models break on a trivial copy task as input grows.**

Reprising Week 10, with a sharper demonstration: the task is *copying words*, which requires no reasoning at all — and models still degrade with length. This isolates the variable completely: it's not task difficulty, it's **input length**.

## Attention budget

> **Context is finite — in performance, not just tokens.**
>
> **n tokens → n² pairwise relationships.** Every token costs attention budget.
>
> ### **THE PRINCIPLE: Smallest set of high-signal tokens.**

The n² framing is the mechanism from Week 4. The principle is the operational rule — note it says *smallest*, not *smallest sufficient*. The default should be exclusion.

## Right altitude

> **Specific behavior, flexible judgment.**

The prompt-writing sweet spot. Too specific and you get brittle rules that fail on unanticipated cases; too vague and behaviour is inconsistent. Specify **what to do**, leave room for **how to judge**.

## Pre-loaded vs just-in-time

| PRE-LOADED / RAG | JUST-IN-TIME / Claude Code |
|---|---|
| Stuff context up front | **Lightweight identifiers** |
| Fast on small data | **Agent navigates via tools** |
| **Drowns on large data** | **Progressive disclosure** |
| **Throws away metadata** | **Mirrors human cognition** |

This is Week 11's Claude-Code-vs-Cursor debate restated as a context strategy. "Throws away metadata" is the subtle criticism of RAG: chunking discards file paths, section headers, modification dates, and directory structure — all of which are *signal* a navigating agent could use.

**Progressive disclosure** is the key idea: hand the agent identifiers (file paths, IDs, section names) and let it fetch the contents it actually needs. Cheap to include, and the agent decides.

## Long-horizon strategies

| Strategy | Use when | Example |
|---|---|---|
| **Compaction** | Conversational flow | Summary-then-continue |
| **Note-taking** | Iterative milestones | `NOTES.md`, scratchpad |
| **Sub-agents** | Parallel exploration | Clean context per branch |

Note-taking is the underrated one: it moves state **out of context and onto disk**, so the agent can drop history and re-read what matters. It's the Week 11 RLM idea — context as an environment you query — applied to the agent's own memory.

## Hands-on

1. **Reproduce context rot live** — Chroma's repeat-words task
2. **Compare pre-load vs JIT** — on a research task
3. **Add compaction** — **watch the saw-tooth**

*The "saw-tooth" is the context-length graph under compaction: it grows, gets compressed, grows again.*

---

# BEAT 3 — Evals

## The premise

> ### **Evals are training data for harness improvement.**
>
> **harness + evals + harness engineering → better agent**
>
> **Same loop as supervised learning. Gradient flows into the harness.**

This is a genuinely good reframe. In model training, loss flows into weights. In agent engineering, **eval failures flow into harness changes** — and *you* are the optimizer. Evals aren't a report card; they're the training signal for the part of the system you can actually change.

## Vocabulary

| Term | Meaning |
|---|---|
| **task** | Single test, defined inputs + success criteria |
| **trial** | One attempt at a task (**run multiple**) |
| **grader** | Scoring logic — code, LLM, or human |
| **transcript** | Full record — outputs, tool calls, reasoning |
| **outcome** | Final state in the environment |
| **eval harness vs agent harness** | **Different layers — don't conflate** |

Note the trinity from Week 16 reappears: task ≈ dataset, trial ≈ rollout, grader ≈ rubric. Same artifact, as promised.

## The mirage of generic metrics

> **MMLU, HELM, BERTScore — these don't tell you if your product works.**

| FOUNDATION EVAL | PRODUCT EVAL |
|---|---|
| Is the model generally capable? | **Does YOUR pipeline do its job?** |
| Standardized | **Domain-specific** |
| Cross-model | **Failure-mode-driven** |
| Easy to game | **Custom criteria** |
| **Weak signal for products** | **Strong signal for shipping** |

The distinction matters because foundation evals are the ones that are *public and easy to reach for* — and they tell you almost nothing about whether your specific system works.

## The 5-star lie

> **What does a 3.7 in "helpfulness" mean?**
>
> ### **Binary > Likert**
> Forces clarity · Faster to label · More consistent · **Actionable**
>
> **Decompose nuance into multiple binary checks, not a fuzzier scale.**

The argument is strong. A Likert score hides disagreement — two labellers averaging 3.7 may agree on nothing. And 3.7 is **unactionable**: you can't fix a 3.7. But "did it cite a source? no" tells you precisely what to build. Nuance isn't lost — it's *decomposed* into several binary checks, each independently fixable.

## Critique shadowing (Hamel Husain)

The process for building product evals:

```
01  Find THE principal domain expert
02  Build diverse dataset (features × scenarios × personas)
03  Pass / fail + detailed critique
04  Fix obvious agent errors
05  Build LLM judge, iterate to alignment
06  Error analysis at scale
```

**Note "THE principal domain expert"** — singular and definite. Consistency of judgment beats averaging across many labellers; committee-labelled data is noisy in a way that makes judge alignment impossible.

Note also step 4: **fix the obvious errors before building a judge.** No point aligning a judge against failures you already know how to fix.

## LLM judge alignment

> **Different splits than ML — small train, big test.**

| Split | Size | Purpose |
|---|---|---|
| **TRAIN** | **20%** | Few-shot examples |
| **DEV** | **40%** | Iterate prompt |
| **TEST** | **40%** | **Touched ONCE** |

> **Measure TPR + TNR — not raw agreement.**

The inverted split makes sense: you only need a handful of few-shot examples, but you need a *lot* of held-out data to trust the judge, since you'll iterate on the dev set many times and overfit it.

**Why TPR + TNR rather than raw agreement:** if 90% of examples pass, a judge that says "pass" every time scores 90% agreement while being completely useless. **True Positive Rate** and **True Negative Rate** are computed separately, so the degenerate judge scores TPR 100% / TNR 0% and is immediately exposed.

## pass@k versus pass^k

> **Two metrics. Opposite stories.**

| `pass@k` | `pass^k` |
|---|---|
| **Any** path to success in k tries? | **All** k trials succeed? |
| **↑ with k** | **↓ with k** |
| First-try problems | **Reliability-critical** |

Both computed from the same trials, telling opposite stories. `pass@k` rises with k because more attempts mean more chances; `pass^k` falls because every additional trial is another chance to fail. **Which you report should follow from what you're shipping** — an assistant a human reviews wants `pass@k`; an autonomous pipeline wants `pass^k`. Quoting `pass@k` for a system that runs unattended is a way to look good while shipping something unreliable.

## Hands-on: build an aligned LLM judge

```
01  Synthesize 20 diverse queries for a company-research agent
02  Hand-label pass / fail with critiques
03  Train / dev / test split — write judge v1 (no few-shot)
04  Iterate to v2 with few-shot, lock and test
05  Cluster failure modes → harness backlog
```

**Step 5 closes the loop:** failure clusters become the harness backlog. That's the gradient flowing into the harness.

---

## Three closing takeaways

1. **Agent = Model + Harness.** The harness is part of the model's effective parameters.
2. **Context is finite — in performance.** Smallest set of high-signal tokens; **JIT > pre-load**.
3. **Evals are training data.** Binary > Likert, aligned LLM judge, **the flywheel never ends.**

---

## Common confusions

**"What is a 'harness', concretely?"** Everything around the model call: the tool definitions, how tool results and errors are formatted back, the loop structure, what context gets assembled, retries, and stopping conditions. If the model is the engine, the harness is the entire rest of the car.

**"How can the same model score differently?"** Because the model only ever sees what the harness gives it. Different tools, different error messages, different context assembly → different information → different decisions. The weights are identical; the *situation* isn't.

**"My agent seems dumb — should I upgrade the model?"** Check the harness first, in this order: are tool **descriptions** clear enough to choose from? Do **errors** come back as readable text it can act on, or as opaque traces? Is context **curated or accumulated**? Are there **too many tools** (past ~20–30, selection accuracy degrades)? A bigger model with the same bad feedback makes the same mistakes more expensively.

**"What does 'right altitude' mean?"** The level of detail that lets the model act without drowning it. Too low: raw dumps of everything, and the signal is lost in noise. Too high: a summary so abstract it can't act on it. Most teams err toward too much, because adding context feels safe — Week 10 shows it isn't.

**"Pre-loaded vs just-in-time context?"** Pre-loading stuffs everything the model *might* need into the window up front. Just-in-time fetches it *when* it's needed, via tools. JIT keeps the window small and relevant, at the cost of extra round-trips. Given context rot, JIT is usually right for long-horizon work.

**"Why are generic metrics a mirage?"** Because "helpfulness: 7.2" is unactionable. It doesn't tell you what broke, which cases fail, or what to change. Useful evals are **specific and binary**: did it cite a real source? Did the code run? Did it refuse when it should have? You can act on those.

**"What's wrong with asking a model to rate 1–5?"** The scale isn't calibrated — the model has no consistent internal notion of what separates a 3 from a 4, and the numbers drift with phrasing and ordering. You get output that *looks* quantitative and carries far less information than it appears to. Binary or few-category rubrics with explicit criteria are far more reliable.

**"pass@k vs pass^k — what's the practical difference?"** `pass@k` = succeeded **at least once** in k attempts. `pass^k` = succeeded **every** time in k attempts. Your users get one attempt, so `pass@k` systematically overstates what they'll experience — and the gap widens as the task gets harder. A model at 80% pass@5 might be at 30% pass^5. Report the one that matches how the system is actually used.

**"What is critique shadowing?"** Rather than inventing a rubric from your imagination, have a **domain expert** review real outputs and write down *why* each is good or bad, in their own words. Build the rubric from their reasoning. It surfaces criteria you'd never have guessed, and it's grounded in what actually matters for the task.

**"Do I need to align an LLM judge?"** Yes, and this isn't optional. An unaligned judge produces confident scores uncorrelated with what you care about — a random number generator with good grammar. Label a sample by hand, measure agreement with the judge, and iterate on the judge prompt until they agree. Then keep spot-checking (S3).

**"Isn't a judge model just moving the problem?"** Partly — you're substituting one model's judgement for a human's, and inheriting its biases (notably length preference). It's justified because it scales, and it's only safe once calibrated against human labels on a sample you keep.

---

## Key takeaways

1. **The same model scores differently across harnesses** (79.8% vs 75.3% on Terminal-Bench 2.0) — the harness is a real, controllable lever.
2. **If you're not the model, you're the harness.**
3. **Harness primitives are derived from model deficiencies**, not invented: filesystem ← no durable state, sub-agents ← context contamination, middleware ← no deterministic interventions.
4. **Harness and model are a matched pair** baked in by post-training — hence no truly model-agnostic agent, and broken mid-chat swaps.
5. **Scaffolding goes stale.** Delete code as models improve.
6. **Context rot appears even on a trivial copy task** — it's length, not difficulty.
7. **n tokens → n² relationships.** Smallest set of high-signal tokens.
8. **Just-in-time beats pre-loading** on large data: lightweight identifiers, progressive disclosure, metadata preserved.
9. **Compaction / note-taking / sub-agents** are the three long-horizon strategies.
10. **Evals are the training signal for the harness** — you're the optimizer.
11. **Foundation evals ≠ product evals.** MMLU won't tell you if your pipeline works.
12. **Binary beats Likert** — actionable, consistent, faster to label.
13. **Judge splits are inverted** (20/40/40) and measured by **TPR + TNR**, never raw agreement.
14. **`pass@k` vs `pass^k` tell opposite stories** — pick the one matching your reliability requirement.

---

## Glossary

| Term | Meaning |
|---|---|
| **Harness** | Everything wrapping the model — tools, prompts, loop, middleware. |
| **Harness primitive** | A capability added to patch a specific model deficiency. |
| **Middleware** | Deterministic code intervening in the agent loop. |
| **Sub-agent** | A child agent with clean context, protecting the parent's. |
| **Model-harness-fit** | The post-trained match between a model and its expected tool vocabulary. |
| **OOD (out of distribution)** | Input unlike anything the model was trained on. |
| **Context rot** | Performance degradation as input length grows. |
| **Attention budget** | The finite, softmax-normalised attention distributed across tokens. |
| **Right altitude** | Prompting at the level of specific behaviour with flexible judgment. |
| **Pre-loaded context** | Stuffing retrieved content up front (RAG-style). |
| **Just-in-time (JIT)** | Supplying identifiers and letting the agent fetch what it needs. |
| **Progressive disclosure** | Revealing detail only as required. |
| **Compaction** | Summarising conversation history to reclaim context. |
| **Saw-tooth** | The context-length pattern produced by repeated compaction. |
| **Task / trial / grader / transcript / outcome** | Eval vocabulary: one test / one attempt / scoring logic / full record / final state. |
| **Foundation eval** | Standardized cross-model benchmark (MMLU, HELM). |
| **Product eval** | Domain-specific, failure-mode-driven evaluation of your pipeline. |
| **Likert scale** | Multi-point rating (e.g. 1–5); discouraged in favour of binary. |
| **Critique shadowing** | Hamel Husain's expert-driven process for building product evals. |
| **TPR / TNR** | True Positive Rate / True Negative Rate — judge quality on each class separately. |
| **`pass@k`** | Success if **any** of k trials succeeds; rises with k. |
| **`pass^k`** | Success only if **all** k trials succeed; falls with k. |
| **The flywheel** | evals → failure clusters → harness changes → better agent → new evals. |

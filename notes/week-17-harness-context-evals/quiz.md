# Week 17 — Quiz (20 questions)

**Topic:** Harness, Context, and Evals
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** On Terminal-Bench 2.0 with Claude Opus 4.6 held constant, ForgeCode and Capy scored:
- A) Identically
- B) 79.8% and 75.3%
- C) 60.1% and 88.4%
- D) Both below Claude Code

**2.** The lecture's core equation is:
- A) Agent = Model + Prompt
- B) Agent = Model + Harness
- C) Agent = Model + RAG
- D) Agent = Model + Fine-tuning

**3.** Which harness primitive patches "context contamination"?
- A) Filesystem
- B) Bash + code
- C) Sub-agents
- D) Sandbox

**4.** Which primitive patches "no durable state"?
- A) Filesystem
- B) Middleware
- C) Sandbox
- D) Sub-agents

**5.** Codex uses `apply_patch` while Claude uses `Edit`. This illustrates:
- A) A performance difference between the tools
- B) Model-harness-fit — same operations, different vocabularies baked in by post-training
- C) A licensing constraint
- D) Different programming languages

**6.** Mid-chat model swaps break because of:
- A) Transcript OOD, cache miss, and tool-surface mismatch — simultaneously
- B) API rate limits
- C) Tokenizer incompatibility only
- D) Temperature differences

**7.** The context principle stated is:
- A) Retrieve as much as possible
- B) Smallest set of high-signal tokens
- C) Always use the full context window
- D) Prefer pre-loading over JIT

**8.** Which is listed as a weakness of pre-loaded/RAG context?
- A) It is too slow on small data
- B) It drowns on large data and throws away metadata
- C) It cannot cite sources
- D) It requires a GPU

**9.** The three long-horizon context strategies are:
- A) Compaction, note-taking, sub-agents
- B) Chunking, embedding, reranking
- C) Caching, batching, streaming
- D) SFT, DPO, RLVR

**10.** The judge alignment split recommended is:
- A) 80% train / 10% dev / 10% test
- B) 20% train / 40% dev / 40% test
- C) 50/25/25
- D) 33/33/33

**11.** Judge quality should be measured with:
- A) Raw agreement percentage
- B) TPR and TNR separately
- C) F1 only
- D) Perplexity

**12.** `pass^k` (as opposed to `pass@k`):
- A) Increases with k
- B) Decreases with k and suits reliability-critical systems
- C) Is identical to pass@k
- D) Measures latency

---

## Short answer

**13.** Explain why the same model scores differently across three harnesses, and what this implies about benchmark interpretation.

**14.** List the five harness primitives with the deficiency each patches, and explain why "derived, not invented" is the right framing.

**15.** Explain model-harness-fit and its three consequences.

**16.** Explain "scaffolding goes stale — delete code" with a concrete example.

**17.** Contrast pre-loaded and just-in-time context. Explain "throws away metadata" and "progressive disclosure."

**18.** Explain "evals are training data for harness improvement." What is the optimizer, and what is the gradient?

**19.** Argue for binary over Likert scoring. Address the objection that binary loses nuance.

**20.** Explain why judge quality must be measured by TPR and TNR rather than raw agreement, with a numeric example.

---
---

## Answer key

**1. B** — ForgeCode 79.8%, Capy 75.3%, with Claude Code lower — all on the same model.

**2. B** — Agent = Model + Harness.

**3. C** — Sub-agents, which give a clean context per branch so exploration doesn't pollute the parent.

**4. A** — Filesystem. The model is stateless, so durable state must live outside it.

**5. B** — Model-harness-fit: the same operations expressed in different vocabularies, each baked into its model by post-training.

**6. A** — All three at once: the transcript contains the other model's tool vocabulary (OOD), the prompt cache is invalidated, and the tool surfaces may not match.

**7. B** — Smallest set of high-signal tokens.

**8. B** — It drowns on large data and throws away metadata.

**9. A** — Compaction, note-taking, and sub-agents.

**10. B** — 20/40/40, inverted relative to typical ML splits, with the test set touched once.

**11. B** — TPR and TNR separately, not raw agreement.

**12. B** — It requires all k trials to succeed, so it falls with k and is the right metric when reliability matters.

**13.** Because **the agent is the model plus the harness**, and only the model was held constant. The harness determines which tools exist and how they are described, how tool results are formatted and truncated, how the loop detects and recovers from failure, how context is managed as the run grows, and what deterministic middleware intervenes. Each of those changes what the model actually sees and can do, so a spread of several points is unsurprising. **Implication for benchmarks:** a leaderboard entry measures a **model-harness pair**, not a model. Comparing two models evaluated under different harnesses conflates two variables, and a model can appear better simply because its scaffolding was better. Practically this cuts both ways — you cannot read a benchmark as a pure model ranking, but you also have a real lever, since the harness is the part you control without training anything.

**14.** **Filesystem** ← no durable state; **Bash + code** ← no general-purpose action; **Sandbox** ← unsafe execution at scale; **Sub-agents** ← context contamination; **Middleware** ← no deterministic interventions. **Why "derived, not invented" matters:** it gives you a principled way to decide what belongs in a harness. Each primitive exists to compensate for something the model genuinely cannot do — it is stateless, so state goes on disk; it cannot act on the world, so give it a shell; it cannot be relied on to do something *every* time, so encode that in middleware rather than the prompt. Starting from deficiencies produces a minimal, justified tool surface, whereas starting from "what would be useful?" produces sprawl — and every extra tool costs context and adds a way for the model to choose wrongly. It also tells you when to *remove* a primitive: when the deficiency stops existing.

**15.** **Model-harness-fit** is the observation that a model is post-trained on a specific tool vocabulary, so the harness becomes part of its effective parameters. Codex expects `apply_patch` (diff-shaped edits), `<oai-mem-citation>`, and `exec_command_tool`; Claude expects `Edit` (old/new string), `<system-reminder>`, and `Bash`. The operations are equivalent; the surface forms are not, and each model is fluent only in its own. **Consequence 1 — no model-agnostic agent:** the honest version is a per-model harness, which means in practice you pick a *product*, not a model. **Consequence 2 — mid-chat model swaps break:** the existing transcript contains tool calls in the other model's vocabulary and is therefore out of distribution, the prompt cache is invalidated causing a latency spike, and the tool surface may not even match — three failure modes triggered simultaneously. **Consequence 3 — the matched pair shifts:** as models are retrained the fit moves, so a harness tuned to last year's model may actively hurt this year's.

**16.** Harness code is written to compensate for model weaknesses, so when a weakness disappears the compensating code becomes pure overhead — "yesterday's load-bearing scaffold is today's dead weight." **Concrete example:** suppose an earlier model frequently emitted JSON with trailing commas or wrapped in markdown fences, so the harness gained a repair layer that strips fences, fixes commas, retries on parse failure, and re-prompts with a formatting reminder. A newer model emits valid JSON reliably. The repair layer now contributes nothing but still costs: the formatting reminder consumes context on every call, the retry path adds latency whenever it misfires, and the repair logic can *corrupt* otherwise-valid output in edge cases it was never designed for. Worse, it is now untested against current behaviour, so it silently rots. The discipline is to periodically re-run evals with scaffolding removed and delete whatever no longer earns its place.

**17.** **Pre-loaded (RAG-style)** stuffs retrieved content into the context up front. It is fast on small data but drowns on large data and throws away metadata. **Just-in-time (Claude Code-style)** supplies lightweight identifiers and lets the agent navigate with tools, using progressive disclosure and mirroring human cognition. **"Throws away metadata"** refers to what chunking destroys: file paths, directory structure, section headers, modification dates, authorship, and the relationships between documents. A chunk arrives as bare text stripped of the context that would tell an agent what it is, where it came from, or whether it is current — information that is cheap to carry and highly informative for deciding what to read next. **"Progressive disclosure"** means giving the agent a cheap map — file paths, IDs, section names — and letting it fetch full contents only for what it actually needs, so context grows in proportion to relevance rather than to corpus size. It mirrors how a person uses a document: consult the table of contents, then open one chapter.

**18.** The claim is that eval results play the same role for harness engineering that labelled data plays for supervised learning: `harness + evals + harness engineering → better agent`, with the gradient flowing into the harness. **The optimizer is you** — the engineer reading failures and deciding what to change. **The gradient is the failure signal**: each failed trial, with its transcript of outputs, tool calls, and reasoning, indicates the direction in which the harness should move. Clustering those failures into modes converts a diffuse signal into a concrete backlog, which is exactly step 5 of the hands-on. **Why the reframe is useful:** it changes evals from a report card — a number you look at after the fact — into the input to an iterative improvement loop over the component you can actually change. You are not training weights, so loss cannot flow there; but the harness is fully under your control, and eval failures are precisely the information needed to improve it. Hence "the flywheel never ends."

**19.** **The argument:** binary forces clarity, is faster to label, is more consistent across labellers, and — most importantly — is **actionable**. A Likert score obscures rather than captures nuance: "3.7 in helpfulness" is unfalsifiable and un-fixable, and two labellers averaging 3.7 may agree on nothing at all, since the scale gives them no shared definition of the gap between 3 and 4. Binary questions force the criterion to be stated explicitly, which makes disagreement visible and resolvable, and makes the result directly usable — "did it cite a source? no" names a specific thing to build. **The objection — that binary loses nuance — is addressed by decomposition:** you do not replace a 5-point helpfulness score with one yes/no, you replace it with several independent binary checks (did it answer the question asked? did it cite sources? did it avoid unsupported claims? did it respect the length constraint?). Nuance is preserved and *localised*, because you now know which dimension failed rather than only that the average dipped. This also makes LLM-judge alignment tractable, since a judge can be evaluated on each binary criterion with TPR and TNR, which is impossible for a fuzzy scalar.

**20.** **Raw agreement is dominated by class imbalance.** Suppose an eval set of 100 examples where the agent passes 90 and fails 10 — a realistic distribution once obvious errors have been fixed. A degenerate judge that outputs "pass" unconditionally agrees with the human labels on all 90 passes and none of the 10 failures, scoring **90% raw agreement** while being completely worthless: it has never once identified a failure, which is the only thing you built it for. **Splitting the metric exposes this immediately:** TPR (of the true passes, how many did the judge call pass?) is **90/90 = 100%**, while TNR (of the true failures, how many did the judge call fail?) is **0/10 = 0%**. The 0% TNR makes the judge's uselessness unmissable. Reporting both keeps you honest in the other direction too — a judge that fails everything scores TNR 100% and TPR 0% — and the pair tells you *which* way the judge is biased, so you know whether to add few-shot examples of subtle passes or of subtle failures.

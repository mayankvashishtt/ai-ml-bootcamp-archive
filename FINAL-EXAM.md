# Cumulative Final Exam

The per-lecture quizzes test one lecture each. This tests **across** them — which is the actual skill. Most questions here require connecting two or more lectures, and several ask you to spot a flawed premise rather than answer the question as asked.

**Format:** 50 questions
- **Part A** — 20 multiple choice, cross-cutting
- **Part B** — 15 short answer, connecting lectures
- **Part C** — 9 "what's wrong with this?" — diagnose a flawed plan or claim
- **Part D** — 6 open design problems, interview-style

**Answer key at the bottom.** Cover it and attempt everything first. Part D has no single right answer — the key gives the reasoning a strong answer would contain.

**Suggested conditions:** 3 hours, closed book for Parts A–C, `CHEATSHEET.md` allowed for Part D.

---

# Part A — Multiple choice (20)

**1.** Softmax appears in both attention and decoding. The difference is:
- A) Different formulas
- B) In attention it normalises over positions; in decoding, over the vocabulary
- C) Attention uses temperature, decoding doesn't
- D) Decoding applies it before the final layer

**2.** Both S6 (embeddings) and W14 (preference tuning) converge on the same principle:
- A) Bigger batches are better
- B) Hard examples teach most, because learning is about boundaries, not centres
- C) Contrastive losses beat cross-entropy
- D) More data beats better data

**3.** A model that scores 94% on a public benchmark and 61% on your internal test set most likely indicates:
- A) Your test set is wrong
- B) Contamination and/or domain mismatch — the benchmark isn't measuring your task
- C) The model is broken
- D) You need a larger model

**4.** RAG uses low temperature, synthetic data generation uses high temperature. The reason is:
- A) RAG is more expensive
- B) RAG wants the grounded token; synthetic data's product *is* diversity
- C) Synthetic data needs longer outputs
- D) RAG requires deterministic output for caching

**5.** Which pair describes the same underlying idea at different scales?
- A) LoRA and quantisation
- B) MinHash+LSH and ANN indexes — avoiding N² comparison
- C) RoPE and RMSNorm
- D) Beam search and speculative decoding

**6.** An agent works on 5-file tasks and fails on 50-file tasks. The *first* thing to investigate is:
- A) Splitting it into multiple agents
- B) Whether context is curated or merely accumulating
- C) Switching to a larger model
- D) Raising max_tokens

**7.** Week 14's reward hacking and S13's model exploitation are:
- A) Unrelated phenomena
- B) The same Goodhart failure against a reward model vs a world model
- C) Both solved by RLVR
- D) Both caused by high temperature

**8.** For a 7B model, weights are 14 GB in fp16 but training needs ~84 GB. The dominant extra term is:
- A) Activations
- B) Gradients
- C) Optimizer states
- D) The KV cache

**9.** Which is true of both `temperature=0` and a large context window?
- A) Both are free
- B) Both promise something the system doesn't actually deliver — determinism, and usable context
- C) Both are deprecated
- D) Both only apply to open models

**10.** A tool description is best understood as:
- A) Documentation for developers
- B) Prompt engineering — the model selects tools from it alone
- C) A schema constraint
- D) Optional metadata

**11.** MCP's relationship to Week 8 function calling parallels:
- A) PyTorch's relationship to NumPy
- B) LSP's relationship to editor plugins — a protocol replacing N×M bespoke integrations
- C) RAG's relationship to fine-tuning
- D) LoRA's relationship to full fine-tuning

**12.** Why does HyDE work even when its generated answer is factually wrong?
- A) The LLM corrects it later
- B) It's a retrieval probe — it needs to be document-*like*, not true
- C) Errors cancel out across chunks
- D) It doesn't; that claim is false

**13.** Which cannot be fixed by a better embedding model?
- A) Poor recall on paraphrased questions
- B) Failure on exact identifiers like `RS256` or `2024-09-15`
- C) Domain-specific term confusion
- D) Poor multilingual retrieval

**14.** Reading "8×7B MoE," the number that tells you what GPU you need to *host* it is:
- A) 7B
- B) ~47B total
- C) ~13B active
- D) 56B

**15.** Constrained decoding, strict tool schemas, and structured outputs share:
- A) They all lower cost
- B) They make invalid output structurally impossible rather than merely unlikely
- C) They all require temperature 0
- D) They only work on open models

**16.** An agent's tool call is best described as:
- A) A performance optimisation
- B) Replacing prediction with observation, correcting an unreliable internal world model
- C) A way to extend the context window
- D) A safety mechanism

**17.** Which combination is the lethal trifecta?
- A) High temperature + long context + tools
- B) Private data + untrusted content + external communication
- C) Fine-tuning + RAG + agents
- D) MCP + computer use + memory

**18.** Deduplication is unusual among data interventions because it:
- A) Costs nothing
- B) Improves models at *fixed compute* — less data, better model
- C) Only helps large models
- D) Is required by law

**19.** A 3% quality improvement at 12× the cost should be reported as:
- A) A success
- B) A trade-off, with cost and latency stated alongside quality
- C) A failure
- D) Inconclusive

**20.** Which statement about current frontier Claude models is correct?
- A) `temperature=0` gives deterministic output
- B) `temperature`, `top_p`, and `top_k` are removed and return 400 on Opus 5
- C) Prefill is the recommended way to force JSON
- D) `top_k` is supported but `temperature` isn't

---

# Part B — Short answer (15)

**21.** Trace one idea — "context is a finite budget" — through at least four lectures, showing how it changes form.

**22.** Explain why "just use a bigger model" is the wrong first response to each of: bad RAG recall, an agent failing on long tasks, poor tabular predictions, and a chart-reading error.

**23.** Both S6 and S9 involve contrastive learning. Explain the mechanism they share and what differs.

**24.** Explain the training/inference tension and name four architectural or strategic decisions it drives.

**25.** Compare the three "who controls it" axes: MCP primitives (tools/resources/prompts), and explain why that axis is the design.

**26.** Explain why an eval that asserts exact string equality is usually measuring the wrong thing, drawing on S3 and S7.

**27.** Week 9's hybrid search demo improved average performance while ranking one query worse. Explain what this teaches about evaluation methodology.

**28.** Explain the relationship between Week 10's context rot, Week 11's RLM, and S12's context isolation.

**29.** Give the full escalation ladder from "prompt" to "RL," state what each rung is *for*, and explain why skipping rungs is a mistake.

**30.** Explain how tokenization decisions in S11 produce failures people attribute to reasoning.

**31.** Explain why grounding beats planning depth, connecting S13's compounding error to W17's harness lessons and W25's computer use.

**32.** Explain what "quality" means for pretraining data vs instruction data, and why the rules invert.

**33.** Compare LoRA, QLoRA, and full fine-tuning through S10's memory arithmetic.

**34.** Explain three distinct ways a system can produce confident, plausible, wrong output, drawn from different lectures.

**35.** Explain the difference between a benchmark score being *dishonest* and being *uninformative*, and why the distinction matters.

---

# Part C — What's wrong with this? (9)

Each describes a plan or claim containing at least one significant error. Identify it and give the correction.

**36.** *"We'll fine-tune a 7B model on our product documentation so it can answer customer questions accurately. We have 50,000 doc pages, so we'll generate 200,000 Q&A pairs with GPT and train on all of them."*

**37.** *"Our agent is unreliable, so we're splitting it into five specialist agents — a planner, a researcher, a coder, a tester, and a reviewer — passing results down the chain."*

**38.** *"We set temperature to 0 for reproducibility, so our eval asserts exact string matches. Our test suite catches any regression."*

**39.** *"Our RAG system has poor recall, so we're upgrading from a 768-dim to a 3072-dim embedding model. Most failures are on queries containing error codes and version numbers."*

**40.** *"We got 91% on MMLU, up from 88%, on 200 test examples. This confirms our fine-tuning approach works."*

**41.** *"We're connecting our agent to MCP servers for GitHub issues, our production Postgres, and Slack. Each is from a reputable source, so the setup is safe."*

**42.** *"We'll send full-resolution 4K screenshots at every step so the vision model has maximum detail, and keep all of them in context so the agent has full history."*

**43.** *"Our reward model gives high scores, our training loss is dropping, and the policy's reward is climbing steadily. Training is going well."*

**44.** *"We benchmarked our multi-agent research system against a single agent and got 15% better results, so we're shipping it."*

---

# Part D — Design problems (6)

Open-ended. Strong answers state assumptions, sequence work by leverage, name trade-offs, and say what they'd measure.

**45.** Design a customer support system for a SaaS product. 2,000 tickets/day, an existing docs site, a ticket history with resolutions. Cover architecture, cost, evaluation, safety, and what you'd build *first*.

**46.** Your company wants an internal "ask anything about our codebase" tool. 4M lines across 60 repos. Design it, and justify every retrieval decision.

**47.** You have a 3-month budget to make an open 8B model excel at your domain (legal contract analysis). You have 500 labelled examples and 40,000 unlabelled contracts. Plan the three months.

**48.** Design the evaluation strategy for an agent that files pull requests autonomously. What do you measure, how do you avoid fooling yourself, and when would you let it run unsupervised?

**49.** Your LLM feature costs $80k/month and finance wants it at $20k with no quality loss. Work through the options in order of leverage, with expected savings and risks.

**50.** A colleague proposes replacing your RAG system with a fine-tuned model "so it just knows the answers." Write the response — including where they might be right.

---
---
---

# ANSWER KEY

## Part A

**1. B** — Same function, different axis: attention weights over positions, probabilities over vocabulary. (W4, S7)

**2. B** — S6's hard negatives and W14's hard preference pairs are the same claim: easy examples confirm what the model knows; hard ones locate the decision boundary.

**3. B** — The two live explanations are contamination (benchmarks are on the web and in Common Crawl many times over) and domain mismatch. Both mean the benchmark isn't measuring your task. (S3, S11)

**4. B** — RAG's context is the source of truth and creativity is a liability; synthetic data's value is coverage, and greedy decoding produces near-duplicates. (S7, W13)

**5. B** — Both avoid all-pairs comparison by bucketing plausible candidates. (S6, S11)

**6. B** — Context rot is the leading hypothesis and the cheapest to check. Multi-agent is a later step, and only if isolation is genuinely the need. (W10, W17, S12)

**7. B** — Optimise hard against an imperfect proxy and you get the proxy's flaws, amplified. Same law, different proxy.

**8. C** — Adam's momentum + variance + fp32 master copy ≈ 56 GB, versus 14 GB gradients.

**9. B** — Neither delivers what its name implies: FP non-associativity and batch-dependent kernels break determinism; context rot means the window isn't usable end to end. (S7, W10)

**10. B** — The model selects tools from names, descriptions, and parameter docs alone; it cannot read the implementation. (S2, S8)

**11. B** — Explicitly modelled on LSP; N×M becomes N+M.

**12. B** — HyDE converts asymmetric query→document into symmetric document→document. Correctness is irrelevant; stylistic document-likeness is the point. (W9, S6)

**13. B** — Follows from the training objective, not insufficient training. The fix is architectural: hybrid search with BM25.

**14. B** — All experts must be resident; only compute is sparse.

**15. B** — All three mask illegal tokens or constrain generation so malformed output is unreachable, rather than merely improbable.

**16. B** — The agent can't reliably predict a command's output, so it runs it. (S13)

**17. B** — Each individually reasonable; jointly exploitable.

**18. B** — A strictly better outcome rather than a trade-off, which is rare.

**19. B** — Reporting it as a trade-off with cost and latency stated is the honest framing. (S3, S12)

**20. B** — Removed and 400 on Opus 5, Fable 5, Opus 4.8/4.7; non-default rejected on Sonnet 5. Prefill also 400s. (S7)

---

## Part B

**21.** **W4 — the origin.** Attention is a fixed budget: every token attends to every other, so adding tokens dilutes attention across more competitors. This is architecture, not policy. **W10 — measured.** Context rot demonstrated empirically that quality degrades as the window fills with noise, and that a large context window is not usable context. The budget isn't just theoretical; it has an observable cost curve. **W11 — the escape.** RLM: rather than loading a large space into context, give the model a way to *explore* it through code, keeping the window small while the reachable space stays large. **W17 — the discipline.** JIT context and the attention-budget principle: assemble context deliberately per step rather than accumulating it, because everything in the window competes. **W18 — persistence.** Memory architectures exist because the budget is finite across time as well as within one request. **S12 — delegation.** A subagent is a context firewall: its 50,000 tokens of exploration never enter the parent's window; only the conclusion does — the same insight as W11, applied to agents rather than code. **S9 — new units.** Screenshots are tokens too, so image resolution is a context-budget decision. **S5 — money.** The budget has a price: every token in the window is billed on every request, so context discipline is cost discipline. The idea starts as a property of attention and ends as an operational constraint on architecture, cost, and agent design.

**22.** **Bad RAG recall** — retrieval quality is not the generator's job. A bigger model papers over bad retrieval more convincingly, which makes the problem *harder* to see, since W9's central lesson is to inspect retrieved chunks rather than final answers. The real causes are chunk size exceeding the embedding max length, missing instruction prefixes, low ANN search effort, exact-identifier queries needing hybrid search, or bad chunking. **Agent failing on long tasks** — this is usually context rot (W10) or a harness problem (W17): opaque errors, vague tool descriptions, unmanaged context accumulation. A bigger model with the same degraded context still reads mostly noise by step 40. **Poor tabular predictions** — the correct model class is gradient boosting, not a language model (S1). Scaling up the wrong model class is expensive and still worse than a 200-line scikit-learn script. **Chart-reading error** — a systematic property of the representation (S9): small text falls below patch resolution and is *absent*, not misread, while precise position-to-value mapping is a known weakness. A bigger VLM has the same patching. The fix is higher resolution, cropping, or extracting the underlying data. **The common thread:** "bigger model" treats every failure as insufficient capability, when most failures are in retrieval, context, model class, or representation — and a bigger model often *hides* the failure rather than fixing it.

**23.** **Shared mechanism:** both train a geometry rather than a generator. An **anchor** is pulled toward a **positive** and pushed away from **negatives**, using **InfoNCE** — a softmax over similarities that maximises anchor–positive similarity *relative to* anchor–negative similarities. Both rely on **in-batch negatives**, where every other pair's positive in the batch serves as a negative, giving N−1 free negatives per anchor — which is why **batch size is a quality parameter** in both, not just a throughput knob. Both produce a space where distance means relatedness, and both are evaluated by retrieval rather than by loss. **What differs:** S6's pairs are **within one modality** (query–document, both text), so a single encoder can handle both sides, though asymmetric search motivates instruction prefixes to distinguish query from document. S9's CLIP pairs are **across modalities** (image–caption), requiring **two separate encoders** projecting into a shared space. The negatives differ in character too: text retrieval benefits enormously from **mined hard negatives** (superficially similar but irrelevant passages), whereas CLIP trained largely on scale — hundreds of millions of web pairs — with in-batch negatives, and its batch size was famously enormous for exactly the reason above. Finally the output differs in use: S6's embeddings feed a retrieval index; CLIP's feed zero-shot classification and cross-modal search, and its image encoder became the vision backbone for the adapter-style multimodal models that followed.

**24.** **The tension:** training happens once, inference happens billions of times. A design that trains efficiently may serve terribly, and lifetime cost is dominated by serving. **Decision one — over-training small models.** Chinchilla gives the compute-optimal ratio (~20 tokens/parameter) for *training*, but inference cost scales with parameters, not with training tokens. So modern models are deliberately trained far past compute-optimal: a 7B on 10× the optimal tokens beats a 30B compute-optimal model in production on every axis that matters after launch. (S10) **Decision two — MoE.** Decouples capacity from per-token compute: many total parameters for knowledge, few active per token for cost. Note it saves compute but *not* memory, which is itself a serving trade-off. (S10) **Decision three — GQA and MLA.** Both exist to shrink the **KV cache**, which is an inference-memory problem, not a training one. They trade a small quality loss for a large serving-memory win. (W6, S10) **Decision four — quantisation and serving-time optimisations** (PagedAttention, continuous batching, speculative decoding, prompt caching). None affect training; all exist purely because serving economics dominate. (S5) A fifth if you want it: **distillation**, training a small model to imitate a large one, which spends training compute specifically to reduce inference compute.

**25.** **Tools are model-controlled** — the model decides to invoke them, they take a JSON Schema, and they have side effects, which is why they need approval gates. **Resources are application-controlled** — the *host* decides to fetch them, they're identified by URI, and they should be side-effect free; they're context, not action. **Prompts are user-controlled** — a human deliberately selects them, typically as a slash command; they are neither injected automatically nor chosen by the model. **Why the axis *is* the design:** the three primitives aren't distinguished by what they do technically — a read-only tool and a resource can return identical data — but by **who holds the authority to trigger them**. That determines everything downstream: where the approval gate belongs, what the security model must guard, what the host UI needs to expose, and what can be abused. A model-controlled primitive can be steered by prompt injection in tool output; a user-controlled one cannot. Getting the axis right means the host can enforce policy at the correct layer, which is only possible because the host is the sole component with a complete view. The practical evidence that this is the real distinction: many servers expose read-only *tools* rather than resources, precisely because tools are model-driven and therefore work without host-side UI — a legitimate design choice that trades away host control for convenience.

**26.** **From S7:** identical outputs were never guaranteed even at temperature 0. Floating-point non-associativity means GPU reductions sum in scheduler-dependent order; optimised kernels take different code paths by batch shape, so **your numerics depend on other users' co-batched traffic**; near-ties flip the argmax; MoE routing can vary with batch composition; and infrastructure changes shift numerics silently. On current frontier Claude models, `temperature=0` doesn't merely fail to help — it returns a 400. **From S3:** an eval is a measurement instrument, and an instrument that reports failure when the model chose a synonym is measuring **token identity, not capability**. It produces false alarms on harmless variation and gives no signal on partial correctness, so it is simultaneously too strict and uninformative. **What to do instead:** assert on the *properties* that matter — schema validity (enforced by structured outputs, so it's guaranteed rather than tested), correctness of extracted fields, presence of required citations, absence of forbidden content. Run **n samples** and report a **confidence interval** rather than a single pass/fail, because the system is stochastic and one run is one draw. Use **McNemar's test** for paired comparisons. And where literal reproducibility is genuinely required, get it by **caching responses**, not by hoping the decoder is deterministic.

**27.** It teaches that **an improvement is a distributional claim, not a per-example one**, and that single-example testing actively misleads. Hybrid search shifted the ranking function; for most queries the exact-match signal from BM25 helped, and for at least one it hurt — which is exactly what "improves the average" means, and it is not a defect in the method. **Three methodological consequences.** First, **evaluate over a query set, never a favourite example**: had the demo been judged on that one query, a genuine improvement would have been rejected. Second, **the converse is equally dangerous** — a change validated on one example that happened to improve can be a regression overall, and this is the more common direction in practice because people test on the case that motivated the change. Third, **report the distribution, not just the mean**: knowing that a change improves 80% of queries and degrades 5% is far more actionable than a single average, since it tells you whether the regressions are tolerable and lets you inspect them. This is also why W9's real lesson — inspect retrieved chunks rather than final answers — pairs with it: aggregate metrics tell you *whether* something changed, and inspection tells you *why*. Add S3's statistics on top: with a small query set, a small average improvement may not survive a confidence interval at all.

**28.** All three are responses to the same fact: **the context window is finite and fills with noise.** **W10 established the problem empirically** — context rot, where quality degrades measurably as the window accumulates content, and where a large advertised window is not usable context. This reframed "just use a longer context" from a solution into a partial one. **W11's RLM is the first escape route.** Rather than loading a large corpus into the window, give the model a way to **explore** it programmatically — write code that searches, filters, and inspects, so the reachable space is enormous while the window stays small. The model chooses what enters its own context, moving from passive receipt to active retrieval. **S12's context isolation is the same insight applied to agents.** A subagent explores in **its own window** — reading 30 files, trying things, failing, retrying — and returns only a summary. The parent's context never sees the 50,000 tokens of exploration. A subagent is, precisely, a **context firewall**. **The unifying principle:** separate *the space you can reach* from *the space you must hold*. RLM does it with code; multi-agent does it with process boundaries. Both accept a cost — RLM's code can be wrong, and multi-agent handoffs lose information the sender didn't know to include — and in both cases the design work is in **what comes back**, which is why S12 insists that a subagent's return value is the product and deserves as much care as a tool description.

**29.** **1. Prompt.** Clearer instructions, structure, explicit constraints, stated output format. *For:* the model already has the capability and just needs to be told what you want. Free and instant. **2. Few-shot examples.** Demonstrate the pattern. *For:* format and style that are easier to show than describe. Costs context on every request. **3. RAG.** Retrieve relevant context at query time. *For:* **knowledge** — facts, documents, anything that changes. **4. SFT.** Fine-tune on demonstrations. *For:* **behaviour** — format, register, refusal patterns, house style. Not for knowledge, which goes stale in weights. **5. Preference tuning (DPO/RLHF).** *For:* subjective quality where you can compare but not demonstrate — you know B is better than A but can't write the ideal answer. **6. RL with verifiable rewards (RLVR/GRPO).** *For:* tasks with a programmatic correctness check — code that runs, maths that checks out. **Why skipping is a mistake:** each rung is roughly an order of magnitude more effort, cost, and risk than the one below, and **most problems genuinely stop at rung 2 or 3**. Skipping also destroys your diagnosis — if you fine-tune first and it works, you never learn whether a better prompt would have sufficed, and you've taken on a training pipeline, a dataset to maintain, and a model to version for no established reason. Worse, the higher rungs can't fix lower-rung problems: no amount of SFT fixes retrieval that returns the wrong chunks, and no amount of RL fixes an ambiguous task specification. And the diagnostic value runs the other way too — if a careful prompt can't do it, you've learned something real about what the model lacks.

**30.** BPE merges are **learned from a corpus**, so the tokenizer inherits that corpus's distribution, and because the embedding table is indexed by token ID, the choice is **permanent after pretraining**. **Arithmetic.** If "12345" is split inconsistently — "123"+"45" here, "1"+"2345" there, depending on which sequences were frequent — then the same numeric value has different token decompositions in different contexts, and the model must learn digit relationships through an unstable representation. Modern tokenizers often split digits individually precisely to fix this. **So a meaningful share of "LLMs can't do maths" is a tokenization artifact, not a reasoning limit** — and someone concluding "the model can't reason" from arithmetic errors has misattributed a data-pipeline decision to cognition. **Non-English performance.** A tokenizer trained mostly on English fragments other scripts heavily, so a Hindi or Thai sentence can cost 3–5× the tokens of its English equivalent. That produces three compounding penalties — higher cost, less content fitting in the window, and worse quality because each concept is spread across many poorly-trained tokens. Observers describe this as "the model is worse at Hindi," which is true but locates the cause in the wrong place. **Code.** Whitespace handling determines whether Python indentation is efficiently represented; a tokenizer trained without code wastes tokens on runs of spaces and handles structure badly. **Domain terms.** Medical, legal, and chemical vocabulary fragments if absent from tokenizer training, making the model least efficient in exactly the domains people most want to specialise in — and a fine-tune (W12) **inherits the tokenizer along with the weights**, so this cannot be fixed downstream.

**31.** **S13's mechanism:** a world model predicts one step with small error; feed that slightly-wrong prediction back in and the error compounds — roughly correct at step 1, drifting by step 5, plausible fiction by step 20. The dangerous property is that **the rollout keeps looking coherent the whole time**; there's no point where the simulation announces it has become fiction. This bounds useful planning horizon regardless of how good the one-step model is. **W17's harness lesson** arrives at the same conclusion empirically: agents that act, observe, and re-plan outperform agents that produce long plans up front, and long trajectories drift. W17 found the behaviour; S13 explains why it had to be true — the agent's plan is built on an internal rollout of consequences, and that rollout compounds error exactly like a learned world model's does. **W25's computer use is the sharpest case.** The agent must predict what a click does, using a model of the UI learned from training data that is frequently wrong — dialogs appear, pages load slowly, state changes unexpectedly. **The fix isn't better prediction, it's verification:** screenshot after every action and check the result. **The unifying statement:** every tool call replaces prediction with observation. Tools aren't conveniences; they're **corrections to an unreliable simulator**, which is why tool quality and feedback quality dominate agent performance. It also explains the design rule — short imagined horizons plus frequent re-grounding — and why "let the agent plan twenty steps then execute" is structurally the wrong architecture rather than merely ambitious.

**32.** **Pretraining data quality is about what to throw away at absurd scale.** You start with petabytes of crawl and retain a low single-digit percentage. "Quality" here means: not boilerplate, not machine-generated spam, not duplicated, in the target language, and not benchmark data. Volume genuinely matters because Chinchilla-scale training needs trillions of tokens, so the goal is **breadth with the junk removed** — more is better *after* deduplication and filtering. The trap is that "quality" encodes an opinion: filtering for Wikipedia-likeness produces an encyclopedic writer, and aggressive toxicity filtering measurably reduces the model's ability to *recognise* toxicity. Filters must be **ablated**, not assumed. **Instruction data quality is about what to keep at small scale, and the rules invert.** Pretraining already installed the knowledge; SFT teaches **format and behaviour**, which is a much smaller thing to learn. Because **SFT is imitation learning**, a mediocre example isn't neutral filler that averages away — it is a demonstration of mediocrity the model faithfully learns to reproduce. Hence LIMA: ~1,000 curated, diverse, consistent examples beat 100,000 mediocre ones. Quality means correctness, response quality you'd ship, **diversity across task types**, consistency (contradictory examples teach inconsistency directly), and format consistency. **Why they invert:** at pretraining you're estimating a broad distribution and noise partially averages out across trillions of tokens; at instruction tuning you're demonstrating a target behaviour on a small set, where every example is a direct instruction about how to behave and nothing averages away.

**33.** **The arithmetic (S10 §3):** training memory ≈ parameters + gradients + optimizer states + activations, roughly **6× the weights** for mixed-precision Adam. For a 7B model in fp16: 14 GB params, 14 GB gradients, ~56 GB optimizer states (Adam's momentum and variance plus an fp32 master copy), plus activations. **~84 GB before activations.** **Full fine-tuning** pays all four terms — every parameter has a gradient and optimizer state — which is why it needs a multi-GPU cluster for a 7B model and why it risks catastrophic forgetting: you're moving every weight. **LoRA** freezes the base and trains small low-rank adapters. The insight is that **frozen parameters produce no gradients and no optimizer states**, so the two dominant terms — ~14 GB and ~56 GB — collapse to near-nothing, existing only for the adapters (often well under 1% of parameters). You still hold 14 GB of frozen weights plus activations. Roughly 84 GB becomes roughly 15 GB, and the adapters are small, versionable, and swappable at serve time. **QLoRA** adds 4-bit quantisation of the frozen base, cutting the remaining parameter term ~4× — 14 GB to ~3.5 GB — so a 7B fine-tune fits on a single consumer GPU. **The key point is that QLoRA's quantisation is the *smaller* half of its benefit;** framing it as "quantisation saves memory" understates it badly, because freezing eliminates the terms that actually dominate. Residual costs remain honest: activations still scale with batch and sequence length (so gradient checkpointing still helps), and 4-bit quantisation of the base costs some quality.

**34.** **1. Retrieval failure masked by fluent generation (W9).** The retriever returns the wrong chunks; the LLM writes a confident, well-structured answer from them. The output looks authoritative and is wrong, and **inspecting the answer cannot reveal this** — you must inspect the retrieved chunks. Week 9's exact-identifier failure is the canonical case: the correct chunk at distance 0.551 barely separated from irrelevant ones at 0.719. **2. Sub-patch information loss in vision (S9).** Small text falls below patch resolution and is **absent** from the representation, not merely blurry. The model then fills the gap from context and priors — reporting a plausible value for a chart axis it never read. This fails *confidently* precisely because the model has no signal that anything is missing, and it's why chart-derived numbers must always be verified. **3. Compounding error in agent planning (S13, W17).** A multi-step plan is built on an internal rollout of consequences; per-step errors accumulate until the assumed world state is fiction, yet the plan remains internally coherent and reads as reasonable throughout. **A fourth, if you want it — lost hedging across agent handoffs (S12):** uncertainty is the first thing summaries drop, so a tentative finding from Agent A arrives at Agent B stripped of its qualifications and is built upon as fact. **The system's confidence increases as the information degrades** — the most insidious version, because the usual signal for "be careful" points the wrong way. **The common structure across all four:** the model produces output shaped by a representation that is missing something, and **nothing in the pipeline carries a signal that the something is missing** — so the absence of doubt is not evidence of correctness.

**35.** **Dishonest** means the number misrepresents what was measured — cherry-picked benchmarks, an unstated tuning budget for one side, a strawman baseline, or a test set that leaked deliberately. It's a claim about the reporter's conduct. **Uninformative** means the number is accurately reported and still tells you nothing useful about your decision: the benchmark was contaminated in the pretraining corpus without anyone knowing, the task distribution doesn't match yours, the sample was too small for the difference to survive a confidence interval, or the metric measures something correlated with but distinct from what you need. **Most misleading benchmark numbers are the second kind, and this is why the distinction matters.** Contamination is nearly guaranteed at web scale — benchmarks are published on the web, discussed in papers, copied into repos, and duplicated throughout Common Crawl — so a model's authors may genuinely not know their score is inflated, and decontamination only works against benchmarks you know about and misses paraphrases. Assuming dishonesty leads to the wrong response (distrust the source, trust a different published number), whereas recognising uninformativeness leads to the right one: **held-out evaluation on your own private data is the only measurement you can verify**, and it's necessary even when everyone involved is scrupulous. It also sets your expectations correctly — you shouldn't expect a well-run lab to produce numbers that predict your use case, because that isn't what a general benchmark is for. Finally, the distinction directs scepticism productively: ask whether the data pipeline is documented, whether the baseline was tuned (W20's missing-baseline critique), and whether the benchmark postdates the training cutoff, rather than asking whether the authors are trustworthy.

---

## Part C

**36.** **Three errors.** **(a) Wrong tool for the job.** Product documentation is **knowledge**, and knowledge changes. W12's decision table is explicit: **RAG adds knowledge; fine-tuning changes behaviour.** A model fine-tuned on today's docs is stale the moment the docs change, and it will answer confidently from memorised outdated content — worse than not knowing. The default should be RAG over the docs, with SFT reserved for teaching *behaviour* (house format, citation style, saying "not documented"). **(b) Quantity over quality.** 200,000 unfiltered synthetic pairs is exactly the anti-pattern S11 and LIMA warn about: SFT is imitation learning, so mediocre examples actively teach mediocrity. ~1,000–2,000 curated, diverse, verified examples would beat 200,000 raw ones. **(c) No filtering, dedup, or eval.** Synthetic generation suffers **diversity collapse** toward the generator's favourite phrasings, so semantic dedup is essential; and no held-out eval set was mentioned, so there's no way to know if it worked. **Also missing:** no refusal/abstention examples, so the model will never learn to say "that isn't documented" and will fabricate instead — the single most damaging failure mode for a docs assistant.

**37.** **The core error is treating multi-agent as a fix for unreliability.** S12: a single agent with good tools beats a multi-agent system with bad ones. Adding agents to an unreliable system produces **five unreliable agents plus coordination overhead**, and makes debugging much harder. Fix the harness first (W17): tool descriptions, readable error feedback, context curation, tool count. **Second error: it's a deep sequential chain, the worst topology.** Passing results "down the chain" maximises both failure modes — the **telephone game** (each hop loses context the receiver doesn't know is missing, so the original requirements are unrecognisable by the reviewer) and **error compounding** (five stages at 90% reliability each = 0.9^5 ≈ **59%**). Prefer shallow and wide. **Third error: no valid reason given.** The three legitimate reasons are parallelism, context isolation, and independent perspective. This design has no parallelism (each stage needs the previous one's output — it's a **workflow**, which W21 already gives you, not a multi-agent system), and the reviewer at the end of the chain has received the whole story and is therefore **anchored**, destroying the one benefit a reviewer could have provided. **What might work:** fix the harness first; if the tasks genuinely parallelise, use orchestrator–worker one level deep; and give the reviewer the artifact and requirements only, never the author's reasoning.

**38.** **Both halves are wrong, and they compound.** **Temperature 0 never guaranteed determinism** (S7): floating-point non-associativity in GPU reductions, batch-shape-dependent kernel paths — meaning your numerics depend on **other users' co-batched traffic** — near-ties flipping the argmax, MoE routing varying with batch composition, and silent infrastructure changes. On current frontier Claude models it's worse than ineffective: `temperature` is **removed and returns a 400**. **Exact string matching is the wrong assertion** (S3). It fails on harmless paraphrase, gives no partial credit, and produces false alarms that erode trust in the suite until people stop reading it — while simultaneously giving no signal on *how* wrong a genuine failure is. The claim "catches any regression" is precisely backwards: it catches **variation**, and it will miss real regressions that happen to preserve the string. **The fix:** assert on **properties** — schema validity (guaranteed by structured outputs rather than tested), correctness of extracted fields, required citations present, forbidden content absent. Run **n samples** and report a **confidence interval**, using McNemar's test for paired comparisons. Where literal reproducibility is genuinely needed, **cache responses** keyed on the request; don't try to get it from the decoder.

**39.** **The diagnosis contradicts the fix.** Error codes and version numbers are **exact identifiers**, and S6/W9 are explicit that this is the one failure dense retrieval cannot solve at any dimension. Embedding models are trained on **semantic similarity**, so every version number is semantically "a version number" and every error code "an error code" — the distinguishing information is the exact character sequence, which is precisely what compression into semantic space discards. **This follows from the objective, not from insufficient capacity**, so 3072 dimensions has the same blind spot as 768. W9 demonstrated it: the correct chunk at distance 0.551 barely separated from irrelevant chunks at 0.719 and 0.745. **The correct fix is architectural: hybrid search.** BM25 matches exact strings by term statistics with no notion of meaning at all, making it the exact complement to dense retrieval, with RRF fusing the two ranked lists. **The proposed change is also actively costly:** 4× the storage and memory (1M docs at 3072 dims × 4 bytes ≈ 12 GB of vectors alone), slower search, and **re-embedding the entire corpus** — real money and downtime for no improvement on the stated failures. **Before any of it:** check the cheap structural causes (chunk size vs max sequence length, instruction prefixes, ANN `ef_search`) and build an eval set measuring recall@k so the change is measurable at all.

**40.** **The sample is far too small for a 3-point difference to mean anything.** With 200 examples, the confidence interval on 91% is roughly ±4 points, so 88% and 91% overlap substantially — this is entirely consistent with **noise**. Report a confidence interval, and use **McNemar's test** for the paired comparison since both models saw the same examples (which is more powerful than comparing independent proportions). **Second problem: MMLU may not measure your task at all.** It's a broad general-knowledge benchmark; if you fine-tuned for a specific domain, improving on MMLU is close to irrelevant to whether the fine-tune worked — and you should be evaluating on a **held-out set from your own distribution**. **Third: contamination.** MMLU has been public for years and appears throughout Common Crawl; scores on it are compromised for any model whose pretraining data you can't audit (S11). **Fourth, and most likely to matter in practice: MMLU is the wrong thing to watch during fine-tuning.** It's more useful as a **regression check for catastrophic forgetting** (W12) than as evidence of improvement — and read that way, a *rise* is surprising and slightly suspicious. **What to do:** build a domain eval set, run n samples, report CIs, test paired significance, and compare against a well-tuned prompt-only baseline (W20's missing-baseline critique) before concluding fine-tuning was responsible.

**41.** **"Reputable sources" addresses only one of several risks, and not the main one.** The combination assembles the **lethal trifecta** (S4) perfectly: **private data** (production Postgres), **untrusted content** (GitHub issue bodies — anyone can open an issue and write anything in it), and **external communication** (Slack). Each server is individually reasonable; jointly they are a data-exfiltration pipeline triggerable by any stranger who can file an issue. **Nothing in the protocol notices you have assembled this.** **Other unaddressed risks:** **prompt injection through tool results** — issue text enters context looking exactly like legitimate content; **tool poisoning** — instructions hidden in tool *descriptions*, which hosts typically don't display, so the model reads what the human never sees; **rug pulls** — dynamic discovery means a server can change tools after you approved it, so install-time approval isn't approval of future versions; and **cross-server interference** — connection-layer isolation doesn't isolate the shared context window, so compromised data from one server influences how the model uses another. **What to actually do:** **break the trifecta** — don't give the issue-reading session both database access and Slack; separate agents, sessions, and credentials. **Least-privilege credentials** — a read-only Postgres role makes a whole class of outcomes impossible rather than unlikely, and enforcement in the credential beats enforcement in the prompt. **Pin server versions.** **Human approval with visible arguments** on consequential actions. **Log every call with arguments and results** (W23).

**42.** **Both halves are wrong, in the same direction.** **Full-resolution 4K is usually wasteful.** Image token cost scales with **area**, so 4K versus 1024px is roughly 16× the tokens — and for most UI tasks that detail is not used. The right approach is to establish the **resolution floor empirically** for your task (there *is* a floor, and it's task-dependent — small text does need resolution), and to **crop rather than resize** when only part of the screen matters, since a cropped region at full resolution beats the whole screen downscaled on both cost and accuracy. **Keeping every screenshot is worse.** Twenty steps of full-resolution images is simultaneously a **context-limit** problem, a **cost** problem, and a **context-rot** problem (W10) — the agent rarely needs step 3's screenshot at step 19, and the old images actively crowd out the current task. Keep the most recent at good resolution and downscale or drop older ones. **It also defeats prompt caching** (S5): screenshots change every turn, so they must sit at the *end* of the context with stable instructions first; interleaving fresh images throughout invalidates the cache repeatedly. **And "maximum detail" misdiagnoses the real problem anyway.** The hard part of computer use is **grounding** — mapping "click Submit" to coordinates — which is weak at any resolution because it's spatial regression the model was never trained to do well. The fixes are architectural: **set-of-mark prompting** (overlay numbered markers so the model picks a number) and **accessibility trees / DOM** as the source of truth with the screenshot as a supplement. **Finally, a screenshot is untrusted content** (S4) — rendered text can carry prompt injection.

**43.** **Every listed signal measures the proxy, not the goal — this is the textbook shape of reward hacking (W14).** "Reward model gives high scores" and "policy's reward is climbing" are the *same* observation, not two independent confirmations, and a policy optimising against a learned reward model will find and exploit that model's flaws. **Rising reward is exactly what both success and reward hacking look like from the inside**, which is why it cannot distinguish them. Goodhart's Law: optimise hard against an imperfect proxy and you get the proxy's flaws, amplified in proportion to how hard you optimised. **"Training loss is dropping" adds nothing** — it says the model is fitting its objective, which is true of a hacked policy too. **What's missing is any measurement outside the optimisation loop.** You need: **held-out human evaluation** on real outputs; **eval on tasks the reward model never scored**; **KL divergence from the reference policy**, since a large drift is a warning sign; **regression checks on general capability** to catch narrowing; and **direct inspection of actual generations**, which is where hacking is usually obvious to a human in ten examples — degenerate patterns, length inflation, sycophancy, or exploitation of a scoring quirk. **Watch specifically for length bias** (S11) — the most reliably observed pathology, where the model learns "longer is better" because annotators preferred longer responses. **The general rule:** never let the thing being optimised be the only thing being measured, and note that S13's model exploitation is this same failure one level deeper — a policy gaming a learned *world* model instead of a learned *reward* model.

**44.** **The critical question is unanswered: what was the single-agent baseline?** W20's missing-baseline critique applies directly. If the comparison was against an **untuned** single agent — default prompt, unrefined tools, no context management — then the result shows that **effort helps**, not that multi-agent helps. A fair comparison gives the single agent the same improved tools, prompts, and context curation, and it frequently closes most of the gap. **Second: cost and latency are unreported, and for multi-agent they are the whole trade.** A 10-worker fan-out typically runs **15–20×** a single agent once per-agent system prompts, tool definitions, returned results, retries, and billed failed work are counted. 15% better at 18× cost is a **trade-off to state explicitly**, not a shipping decision — and it might still be right, but only if said out loud. **Third: "15% better" on what, and with what uncertainty?** No sample size, no confidence interval, no significance test. Multi-agent systems have **more stochastic components and therefore higher variance**, so they need *more* samples for the same confidence, and a 15% difference on a small eval set may not survive a CI. **Fourth: which metric?** If it's an aggregate, check the distribution — a mean improvement can hide regressions on a query class you care about (W9's own hybrid demo did exactly this). **What to do before shipping:** rerun against a well-tuned single agent, report quality with CIs alongside cost and latency, inspect the per-case distribution, and consider routing — using multi-agent only for the cases where it actually wins is usually the right answer.

---

## Part D — Design problems

*These are graded on reasoning, not on matching an answer. A strong response states assumptions, sequences by leverage, names trade-offs, and specifies measurement. Below is what a strong answer contains.*

**45. Support system.** **Start with the eval and a triage baseline, not the model.** Build a labelled set from historical tickets with known resolutions; measure deflection rate, escalation accuracy, and harm rate separately. **Architecture: RAG over docs + ticket history, not fine-tuning** — support knowledge changes constantly (W12). Ticket history is the higher-value corpus, since it contains real problems and real resolutions; index resolved tickets alongside docs. **Retrieval:** hybrid search is mandatory here, because tickets are full of exact identifiers — error codes, plan names, API endpoints, version numbers (W9, S6). Contextual retrieval helps for tickets that only make sense with surrounding thread context. Rerank the top 20. **Behaviour via light SFT** if needed: house tone, citation format, and — critically — **abstention**, since a support bot that fabricates is worse than one that escalates. **Routing by difficulty is the main cost lever** (S5): most tickets are repetitive and a small model handles them; escalate hard ones. Prompt caching on the stable system prompt and doc prefix. **Safety:** ticket content is **untrusted input** (S4) — customers can write anything, including injection attempts, so never let the bot take account actions (refunds, plan changes, password resets) directly; those go through approval. **Build first:** retrieval + eval + a human-in-the-loop draft-suggestion mode, where agents accept/edit/reject. That ships value immediately, is safe by construction, and **generates exactly the preference and SFT data** you'd need later. Full automation only for ticket classes where measured accuracy justifies it.

**46. Codebase Q&A.** **The central retrieval fact is that code is close to the worst case for embeddings** (W11, S6): function names, config keys, and error strings are **exact identifiers**, which is precisely what semantic compression discards — and it's the best case for grep. So: **hybrid search is not optional, it's the foundation**, and for many queries plain ripgrep beats vector search outright. **Strongly consider the RLM/agentic approach over classic RAG** (W11): 4M lines will never fit in context, and an agent that can grep, read files, follow imports, and traverse the call graph explores the space instead of pre-chunking it. This also sidesteps the hardest chunking problem — code has structure that fixed-size chunking destroys. **If you do embed:** chunk on **syntactic boundaries** (function/class) using an AST parser, not fixed windows; use contextual retrieval to prepend file path, module purpose, and enclosing symbols before embedding, since a function body alone is often meaningless. Index docs, ADRs, and commit messages too — the *why* usually isn't in the code. **Cross-repo is the hard part:** a query about a service may need code from three repos, so include dependency metadata and route by repo when the query names one. **Freshness:** re-index on merge; a stale index answering confidently about deleted code is a real failure mode. **Evaluate on real developer questions** with known-correct files, measuring recall@k on files, not on chunks.

**47. Legal domain, 3 months, 500 labelled + 40k unlabelled.** **Month 1 — establish the baseline and the eval, and try not to train at all.** Split the 500 labelled examples into ~150 eval and ~350 train — the eval set is the most valuable asset you have and must be protected. Baseline a strong prompt with few-shot examples on a frontier API model *and* on the base 8B, so you know what you're improving on and whether the open model is even necessary. Many projects end here; finding that out in month 1 is a win. **Month 2 — data, which is where the leverage actually is (S11).** The 40k unlabelled contracts are the real asset. Use **CPT** (W12) on them to teach legal register and vocabulary — this is exactly the case CPT is for, since legal language is genuinely out-of-distribution. Then build SFT data: generate synthetic labelled examples using a strong model over the unlabelled corpus, filter hard (rule-based checks, verification against contract structure, model-as-judge with calibration against your human labels), deduplicate semantically to fight diversity collapse, and **read 100 by hand**. Target ~2,000–5,000 high-quality examples, not 50,000 — LIMA's finding. Reserve human labelling effort for the *hard* cases the filters flag as uncertain. **Month 3 — train, evaluate, iterate.** QLoRA (S10 memory arithmetic makes this trivially affordable), evaluate with CIs and McNemar against the month-1 baseline, and iterate on **data** rather than hyperparameters, because that's where the returns are. **Reserve the last two weeks** for failure analysis and a clean writeup of where it's unreliable — for legal work, knowing the failure boundary is part of the deliverable. **Flag throughout:** legal analysis is high-stakes, so abstention and calibration matter more than raw accuracy, and this should be a drafting aid with human review, not an autonomous system.

**48. Evaluating an autonomous PR agent.** **Measure outcomes, not activity.** PRs opened is not a metric; **merged without modification**, **merged after minor edits**, **rejected**, and **caused a revert or incident** are. Weight by severity — one production incident outweighs fifty merged typo fixes. **Build a retrospective eval set** from historical issues with known-good fixes, so you can measure offline before ever running live. **Trace everything** (W23): with agent trajectories, aggregate metrics can't tell you *which* step failed, and you'll need per-run traces to debug. **Avoid fooling yourself** in specific ways: (a) don't evaluate on the tasks you tuned on; (b) beware **survivorship bias** — if a human reviews and fixes every PR, "merge rate" measures the reviewer, not the agent, so track edit distance from proposed to merged; (c) watch for **easy-task drift**, where the agent gets good at trivial changes and the aggregate improves while hard-task performance doesn't; (d) run **n samples** on the same task, since agent runs are high-variance and one success proves little; (e) check whether it games the metric — tests that pass because they were weakened is the canonical reward-hack here (W14), so review test diffs specifically. **When to let it run unsupervised:** only per task class, never globally. Reversibility is the axis (S4, S13) — a PR is inherently reviewable, so the gate is really *merge* authority, not *open* authority. Candidate classes for autonomy: dependency bumps with green CI, lint fixes, mechanical refactors with strong test coverage. Never: anything touching auth, payments, migrations, infra, or security. Require CI green as a hard precondition, cap blast radius by file count and path allowlist, and keep a human on merge for anything else — with the honest recognition that a rubber-stamping reviewer is worse than no reviewer, because it creates false assurance.

**49. $80k → $20k.** **First, measure where the money actually goes** — by feature, by model, by input vs output tokens, and by request class. Optimising before profiling is guessing, and the distribution is usually far more skewed than people expect. **Then, in order of leverage: (1) Prompt caching** (S5) — if there's a stable system prompt, tool set, or document prefix, this is often the single largest win and requires no quality trade. **Reorder context so stable content is first and volatile content last**; this alone can be the difference between a cache that works and one that never hits. **(2) Route by difficulty** — the classic finding is that most traffic is easy. Send it to a smaller model, escalate the rest, and measure the escalation threshold against your eval set. This is usually the biggest *structural* saving. **(3) Batch API** for anything non-interactive — nightly jobs, backfills, classification pipelines — at a large discount. **(4) Cut output tokens**, which cost several times input: ask for terse output, cap `max_tokens` sensibly, and stop generating prose when a schema would do. **(5) Trim the prompt** — few-shot examples are billed on every single request forever, so removing three that aren't earning their place is a permanent saving; measure the quality effect rather than assuming. **(6) Cache full responses** for repeated identical requests, which in support/FAQ workloads is a surprising share. **(7) Downscale images** if any (S9) — cost scales with area, so this is a 4–16× lever where it applies. **Risks to state:** routing is the only item with a real quality risk, so it needs an eval gate and a monitored fallback; over-trimming prompts degrades quality silently; and aggressive caching can serve stale results. **Report honestly** — if 4× isn't reachable without quality loss, say so with numbers and let the business decide, rather than quietly shipping a worse product.

**50. RAG vs fine-tuning response.** **Lead with where they're right**, because they partly are. Fine-tuning genuinely wins on **behaviour**: response format, domain register, house style, when to refuse, and consistent structure. It also removes retrieval latency and the retrieval failure mode, and can make a smaller model viable — a real cost argument. If the current RAG system's problems are *style* problems, they're identifying something true. **Then the core distinction (W12):** **RAG adds knowledge; fine-tuning changes behaviour.** Weights are a snapshot. If the content changes — and for almost any real corpus it does — a fine-tuned model is **confidently wrong** the moment it's stale, with no signal that it's out of date, whereas RAG picks up new content on the next index update. Worse, you can't cite: RAG returns sources, which for most business uses is a requirement rather than a nicety, and "it just knows" means "it can't show its work." **The practical points:** every knowledge update means a retraining cycle plus an eval cycle; you lose the ability to remove a document on request (a real problem if any content is customer- or licence-restricted); and hallucination gets *harder* to detect, because a fine-tuned model produces plausible domain-shaped answers with no retrieval trace to inspect. **The likely right answer is both:** RAG for knowledge, light SFT for behaviour — including teaching the model to say "not in the documentation," which is the highest-value behaviour to train. **Close by making it empirical rather than a debate:** propose an eval set of real questions and a head-to-head — well-tuned RAG vs fine-tuned vs both — measured on accuracy, staleness after a doc update, and cost. If the corpus genuinely never changes and citations aren't needed, they may well win, and the test will show it.

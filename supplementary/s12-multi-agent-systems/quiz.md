# S12 — Quiz (20 questions)

**Topic:** Multi-Agent Systems
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The three things multi-agent systems buy you are:
- A) Speed, accuracy, and cost savings
- B) Parallelism, context isolation, and independent perspective
- C) Specialisation, memory, and tool access
- D) Reliability, observability, and scale

**2.** The unavoidable cost of multi-agent design is:
- A) Latency
- B) Context does not survive handoffs, and neither agent knows what was lost
- C) Tool duplication
- D) Model version drift

**3.** If your single agent is failing, the first thing to check is:
- A) Whether to split it into specialists
- B) The harness — tool descriptions, error feedback, context curation, tool count
- C) The model size
- D) The temperature

**4.** A subtask is genuinely parallel only if:
- A) It uses a different tool
- B) It can start before the other subtask finishes
- C) It runs on a different model
- D) It returns structured output

**5.** The strongest argument for delegation is:
- A) Specialists outperform generalists
- B) Context isolation — the subagent's exploration never enters the parent's window
- C) Lower cost
- D) Easier debugging

**6.** For a subagent, the actual product is:
- A) Its full transcript
- B) What it returns — its exploration is discarded
- C) The tools it called
- D) Its token usage

**7.** A code reviewer agent given the author's reasoning and justifications is:
- A) Better informed and more accurate
- B) Anchored — you've paid for an echo rather than independence
- C) Unable to function
- D) Equivalent to a fresh reviewer

**8.** Five sequential agents each 90% reliable give end-to-end reliability of about:
- A) 90%
- B) 59%
- C) 45%
- D) 81%

**9.** Compared with wide fan-outs, deep chains are worse because:
- A) They're slower
- B) They compound both context loss and unreliability
- C) They cost more tokens
- D) They can't be cached

**10.** Debate as a pattern is risky mainly because:
- A) It's slow
- B) Agents converge through agreeableness rather than evidence
- C) It requires many models
- D) It can't be evaluated

**11.** A 10-worker fan-out typically costs:
- A) Exactly 10×
- B) Roughly 15–20×, once orchestration and returned results are counted
- C) Less than 10× due to caching
- D) 100×

**12.** Delegation depth should be capped:
- A) In the system prompt
- B) In code — prompts are not a control mechanism for runaway loops
- C) By the model's judgement
- D) By token budget alone

---

## Short answer

**13.** Explain the three reasons to go multi-agent and the one cost, and why that framing predicts which designs fail.

**14.** Explain why "fix the harness first" precedes any multi-agent decision, and what specifically to check.

**15.** Explain context isolation, its connection to Weeks 10 and 11, and the design implication most implementations miss.

**16.** Explain the context tax in detail — the failure shapes it produces and five practical mitigations.

**17.** Compare orchestrator–worker, pipeline, and debate topologies. Why is orchestrator–worker the default?

**18.** Explain the cost model honestly — what multiplies, what justifies it, what doesn't, and what controls work.

**19.** Explain how to evaluate a multi-agent system, and name the comparison people most often skip.

**20.** You have a working single-agent code reviewer. It does well on small PRs and degrades badly on PRs touching 40+ files. Design a fix, justifying whether multi-agent is warranted.

---
---

## Answer key

**1. B** — Everything else follows from these three; if you need none of them, you're paying the tax for nothing.

**2. B** — Every handoff is a lossy compression step, and the omissions are invisible to the receiver.

**3. B** — Adding agents to a bad harness gives you several confused agents plus coordination overhead.

**4. B** — If B needs A's output, it's sequential — a workflow, not a multi-agent system.

**5. B** — A subagent is a context firewall; it's the same insight as Week 11's RLM.

**6. B** — 50,000 tokens of exploration are thrown away; only the return value survives.

**7. B** — Independence must be real, or the second look is anchored by the first.

**8. B** — 0.9^5 ≈ 0.59.

**9. B** — The telephone game plus reliability multiplication both worsen with depth.

**10. B** — Models are trained to be agreeable, so consensus can be confident and wrong.

**11. B** — Per-agent system prompts and tool definitions, results flowing back, retries, and billed failed work.

**12. B** — A prompt cannot stop a runaway loop; a structural limit can.

**13.** **Three things you can buy.** **Parallelism** — genuinely independent subtasks run concurrently, cutting wall-clock time to roughly the slowest one rather than the sum. **Context isolation** — a subagent's exploration happens in its own window, so 30 files read never enter the parent's context; only the conclusion does. **Independent perspective** — a fresh agent, unanchored by the first agent's committed approach and baked-in assumptions, catches things the author structurally cannot. **The one cost is that context does not survive the boundary.** Every handoff is a lossy compression step: Agent A knows a hundred things, passes a paragraph, and Agent B works from the paragraph. Everything omitted is gone, and — the dangerous part — **B does not know what was omitted**, so it cannot ask about a constraint it doesn't know exists. **This framing predicts failure precisely:** if a problem isn't solved by parallelism, isolation, or independent perspective, then adding agents pays the context tax and buys nothing, making the system worse while making the architecture diagram look more sophisticated. Most failed multi-agent designs are exactly this — someone splits a coherent task into "specialists" that all needed the full context anyway, so each one now works from a lossy summary of what a single agent had in full.

**14.** Because **a single agent with good tools beats a multi-agent system with bad ones, every time**, and because adding agents to a broken harness produces several confused agents plus coordination overhead — a strictly worse system that is also harder to debug. Week 17's lessons fix far more failures than delegation does, and the multi-agent instinct usually arrives as a misdiagnosis. **Specifically check four things.** **Tool descriptions** — the model selects tools from names, descriptions, and parameter docs alone, and cannot read the implementation, so a vague description produces a tool that's never called or called wrongly (S2, S8). **Error feedback** — do errors return as readable text the model can act on ("table 'user' not found, did you mean 'users'?"), or as opaque stack traces that end the run? Week 17's point is that an agent's ability to recover is bounded by the quality of the feedback it gets. **Context curation** — is context deliberately managed or merely accumulated? By step 40 an unmanaged agent's window is mostly dead ends and stale tool output, and that's context rot (Week 10), not a reasoning deficiency. **Tool count** — beyond roughly 20–30 tools, selection accuracy degrades, and the fix may be consolidating overlapping tools rather than adding an agent to choose between them. Only once these are genuinely good is "the task needs parallelism or isolation" a diagnosis rather than a guess.

**15.** A single agent working a large job accumulates everything — every file read, every failed attempt, every tool result, every dead end. By step 40 the context is mostly noise, and **context rot** (Week 10, measured there rather than asserted) degrades performance in a way that looks like the model getting dumber. **Context isolation** means dispatching a subagent with a **fresh context window**: it reads 30 files, does its work, and returns a summary. The 30 files never touch the parent's context; only the conclusion does. This is the strongest argument for delegation, and it is **the same insight as Week 11's Recursive Language Model** — give the model a way to explore a large space without dragging the entire space into its window. A subagent is, precisely, a **context firewall**. **The design implication most implementations miss is that the return value is the product.** The subagent's 50,000 tokens of exploration are discarded by design; only what it returns survives. That means the **return format deserves the same care as a tool description** — specify what to return, in what structure, at what length, and with which fields required. A subagent that dumps its raw transcript back into the parent has defeated the entire purpose, reconstructing the exact context problem it was dispatched to avoid, and this is where naive implementations most reliably fail. The corollary is that structured returns (S7) and a required uncertainty field are not polish; they're what makes the pattern work at all.

**16.** When A hands to B, everything B knows arrives through one message. A knew what it tried, what failed, what surprised it, which sources were shaky, and where it was unsure; B receives a summary and **cannot know what was left out**. **Four failure shapes follow.** **The telephone game** — each hop degrades fidelity, so A→B→C→D leaves the original requirements unrecognisable by D. **Lost constraints** — the user said "don't touch the auth module," the orchestrator's brief didn't repeat it, and the subagent modifies auth *correctly given what it was told*. **Duplicated and conflicting work** — overlapping scopes produce two different solutions to the same problem, and merging becomes a third unplanned problem. **Confident propagation of error** — this is the most insidious: A's plausible-but-wrong finding reaches B stripped of hedging, because **uncertainty is the first thing summaries drop**, so B builds on it as fact and the system's confidence *increases* as the information degrades. **Five mitigations.** **Pass requirements and constraints verbatim**, repeated in every subagent brief — cheap, and it prevents the worst failures. **Prefer shallow and wide over deep and narrow**, since one orchestrator with ten workers loses far less than a five-level chain. **Structure the handoff with a schema** (S7), forcing important fields to be present rather than hoping they survive prose compression. **Make uncertainty and open questions required fields**, so hedging survives the boundary instead of being compressed away. **Give subagents a path back to the source** — one that can re-read the original file is far more robust than one working only from a description.

**17.** **Orchestrator–worker** has one coordinator dispatching N workers and synthesising their results. It suits parallel independent subtasks, is **shallow** (a single hop, so minimal telephone effect), is naturally parallel, and is easy to reason about and debug. Its main risk is the **orchestrator becoming the bottleneck**: ten workers returning 5,000 tokens each hands the orchestrator 50,000 tokens of results, reconstructing the context problem you were solving — which is why short structured returns matter. **Pipeline** runs fixed stages A→B→C, suiting well-understood staged work, but it is the topology most exposed to the **telephone game** and to **error compounding** (0.9^N), and each stage is a barrier if implemented naively. Note that a pipeline isn't really multi-agent in the interesting sense — it's a workflow, which Week 21 already gave you, and it should be built as one. **Debate** has agents argue toward consensus, and it's the most intuitively appealing and least reliably useful pattern: models are trained to be **agreeable**, so they frequently converge through mutual accommodation rather than evidence, leaving you paying N× for confident consensus no better than one agent's answer. If used, force disagreement structurally — assign opposing positions, require citations, and have a **separate judge** score the arguments rather than letting debaters decide. **Orchestrator–worker is the default** because it maximises the benefits (parallelism and isolation both apply) while minimising the cost (one hop means minimal context loss, and no sequential reliability multiplication).

**18.** **What multiplies:** every agent pays for its own system prompt and full tool definitions (S5), the orchestrator pays again for every subagent's returned result, retries and error handling multiply per agent, and **failed subagent work is billed in full**. The result is **superlinear**: a 10-worker fan-out commonly runs **15–20×** a single agent rather than 10×. **What justifies it:** latency genuinely matters and the work genuinely parallelises, so you are **deliberately converting money into wall-clock time**; isolation prevents a failure that would occur at any cost, meaning a single agent simply cannot do this task; or the task's value is high enough that correctness dominates cost. **What doesn't justify it:** "it's a more sophisticated architecture"; "specialists beat generalists" — frequently untrue when the generalist has full context and the specialist has a lossy summary; and "it's how a human team would do it" — human teams are shaped by humans' inability to hold everything in one head and their inability to be cloned, and neither constraint transfers cleanly. **Controls that work:** cap subagent count and recursion depth **structurally in code** rather than by prompting; set explicit token budgets per subagent; keep shared prefixes stable across workers so **prompt caching** applies (S5); and **route by difficulty**, using cheaper models for mechanical subtasks and reserving expensive ones for hard judgement. Reporting cost alongside quality is part of doing this honestly — a 3% quality gain for 12× cost is a finding, not a success.

**19.** S3's principles apply, with multi-agent-specific additions. **Evaluate end-to-end outcomes, not agent-level behaviour** — every subagent can behave sensibly while the system as a whole fails, so per-agent metrics can look healthy on a broken system. **Trace the full trajectory** (Week 23): a failure now lives somewhere across N transcripts, and without tracing you're guessing which agent introduced the error; instrument *before* scaling, because this becomes acutely painful exactly when the system is complex enough to need it. **Measure cost and latency alongside quality**, and report them together — a 3% quality improvement at 12× cost is a real result that should be stated as a trade rather than a win. **Expect higher variance:** more stochastic components means noisier results, so you need more samples for the same confidence interval, and a difference that looks meaningful on 20 runs may well be noise. **The comparison people most often skip is the well-tuned single-agent baseline.** It is the only comparison that answers the actual question — "does this architecture help?" — and it must be *well-tuned*, not a strawman, meaning the single agent gets the same improved tools, prompts, and context management. This is Week 20's missing-baseline critique applied directly: a multi-agent system compared only against an untuned single agent has demonstrated that tuning helps, not that multi-agent does.

**20.** **First, diagnose why 40-file PRs fail, because the fix depends on which cause it is.** Two candidates. If the reviewer runs out of context or degrades as it accumulates 40 files of diffs plus tool output, that's **context rot** (Week 10) — a real isolation problem. If instead it reviews everything but produces shallow findings, or misses cross-file issues, that's a different problem and delegation won't fix it. Establish this by tracing a failing run: check whether quality degrades monotonically through the review (rot) or is uniformly shallow (harness/prompt). **Also do the Week 17 harness pass before anything else** — are diffs being passed efficiently or as whole files, are the review criteria specific, does the agent get the surrounding code it needs, is output structured? A meaningful share of "large PR" failures are just wasteful context usage. **Assuming it's genuine context rot, multi-agent is warranted here, and specifically for isolation and parallelism**, which is the honest justification: 40 files is too much for one window, the per-file reviews are **genuinely independent** (file B's review doesn't need file A's result), and the exploration — reading surrounding code, checking call sites — is exactly the noise that should never reach the synthesiser. **Design: orchestrator–worker, shallow and wide.** One worker per file, or better, per **coherent change unit** (a file plus its test, or a module) so related changes are reviewed together rather than artificially split — this matters because splitting a coherent change across two workers guarantees each sees half the picture. **Pass constraints verbatim** to every worker: the PR description, the repo conventions, the review criteria, and any explicit "don't flag X" instructions. Summarising these is where the lost-constraints failure comes from. **Require a structured return** (S7): findings as a list with file, line, category, severity, a concrete failure scenario, and — critically — a **confidence field**, so hedging survives the boundary instead of being the first casualty of compression. Bound the return length so the orchestrator doesn't accumulate 40 × 5,000 tokens and rebuild the original problem. **Add a verification stage with genuine independence.** Findings from a fan-out are noisy, and false positives destroy trust in a review tool faster than misses do. Dispatch a fresh agent per finding whose explicit job is to **refute** it, defaulting to "unproven" when uncertain, and give it the code and the claim but **not the finder's reasoning** — otherwise it's anchored and you've bought an echo. Drop findings that don't survive. **Handle what per-file workers structurally cannot see:** cross-file and architectural issues are invisible to a worker scoped to one file, so add a separate pass over the PR's overall diff summary and file list specifically for interface consistency, cross-module contracts, and whether the change set is coherent. Without this, the fan-out trades one failure mode for another. **Keep it one level deep** — no worker delegates further, capped in code — since deep chains compound both context loss and unreliability. **Cost controls:** cheaper models for the per-file pass, stronger ones for verification and synthesis; stable shared prefixes across workers so prompt caching applies (S5); a hard cap on worker count with explicit logging when a PR exceeds it, because silent truncation reads as "reviewed everything" when it didn't. **Then evaluate honestly:** build a set of real PRs with known issues, and compare the new system against the **well-tuned single agent** — not the current broken one — on precision, recall, cost, and latency. Confirm it actually wins on the 40-file case, and check it hasn't regressed on small PRs, where the single agent should probably still be used. Routing by PR size is a legitimate and likely outcome.

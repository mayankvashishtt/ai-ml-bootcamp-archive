# S12 — Multi-Agent Systems: When More Agents Help, and When They Don't

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 8 builds a single ReAct agent and Week 21 orchestrates one with LangGraph. Neither addresses what happens when you use several agents together — which is the first thing most people try after their single agent works, and the point at which a lot of projects quietly get worse while appearing more sophisticated.

**Fills the gap after:** Week 21 (LangGraph)
**Prerequisites:** Weeks 8, 11, 17, 21 (agents, RLM, harness, orchestration), S5 (cost), S8 (MCP) helpful

---

## 0. The idea in plain language

The appeal is obvious. One agent gets confused doing a big job, so you split the job across specialists — a researcher, a writer, a reviewer — like a team. Teams outperform individuals, so agents should too.

The reality is more specific: **multi-agent systems buy you three things, and cost you one thing, and most failed designs bought something they didn't need while paying the cost anyway.**

**What you can buy:**
1. **Parallelism** — genuinely independent work happening at once
2. **Context isolation** — keeping one job's clutter out of another job's window
3. **Independent perspective** — a second look that isn't anchored by the first

**What you always pay:**
**Context does not survive the boundary.** Every handoff between agents is a lossy compression step. Agent A knows a hundred things; it passes a paragraph; Agent B works from the paragraph. Everything not in that paragraph is gone — and neither agent knows what was lost.

That single trade is the whole lecture. **If your problem isn't solved by parallelism, isolation, or independent perspective, adding agents makes it worse**, because you're paying the context tax for nothing.

---

## 1. Start with the strong default: one agent

Before anything else, exhaust the single-agent option. Week 17's lessons — better tools, better context, better feedback — fix far more problems than delegation does.

**A single agent with good tools beats a multi-agent system with poor ones, every time.** If your agent is failing, the first question is not "how do I split this up?" but "which of these is broken?":

- Are the **tool descriptions** clear? (S2, S8 — the model picks tools from descriptions alone)
- Do **errors come back as readable text** the model can act on? (Week 17)
- Is the context **curated or accumulated**? (Week 10)
- Are there **too many tools**? (Beyond roughly 20–30, selection accuracy degrades — and the fix might be tool consolidation, not another agent)

Adding agents to a system with bad tools gives you several confused agents instead of one, plus coordination overhead. This is the most common failure in the space, and it's expensive: multi-agent designs look like progress on an architecture diagram while making the actual failure worse.

---

## 2. Reason 1 to go multi-agent: parallelism

The clearest win, and the easiest to evaluate honestly.

If a task decomposes into **genuinely independent subtasks**, running them concurrently cuts wall-clock time roughly to the slowest one instead of the sum. Reviewing 30 files, researching 8 competitors, checking a claim against 5 sources — these parallelise cleanly.

**The test for "genuinely independent":** can subtask B start before subtask A finishes? If B needs A's output, it isn't parallel, it's sequential — and a pipeline is not a multi-agent system in any interesting sense, it's just a workflow, which Week 21 already gave you.

**The honest cost:** parallelism multiplies token spend. Ten agents means roughly ten times the tokens, and they run whether or not their results turn out to matter. You're buying latency with money (S5). Often worth it; always a choice, and worth stating explicitly rather than discovering on the bill.

---

## 3. Reason 2: context isolation

The subtler and often more valuable reason, and it comes straight from Week 10.

A single agent doing a large job accumulates everything: every file read, every failed attempt, every tool result, every dead end. By step 40 the context is mostly noise, and **context rot** degrades performance measurably.

Subagents give you a **fresh context per subtask.** A subagent reads 30 files, does its work, and returns a summary. The 30 files never enter the parent's context — only the conclusion does.

**This is the single strongest argument for delegation**, and it's the same insight as Week 11's Recursive Language Model: give the model a way to explore a large space without dragging the whole space into its window. A subagent is a **context firewall**.

**The design implication is important and frequently missed:** what the subagent *returns* is the product. Its 50,000 tokens of exploration are discarded. So the return format deserves the same care as a tool description — specify what to return, in what structure, at what length. A subagent that returns a raw transcript has defeated its own purpose, and this is exactly where most naive implementations fail.

---

## 4. Reason 3: independent perspective

Some tasks genuinely benefit from a second look that **isn't anchored by the first**.

An agent that wrote code is a poor reviewer of that code — it has already committed to its approach, its assumptions are baked into the context, and it's essentially being asked to disagree with itself. A **fresh agent with only the code and the requirements** will catch things the author cannot.

This generalises into a useful pattern family:
- **Adversarial verification** — an agent whose explicit job is to *refute* a finding, defaulting to "unproven" when uncertain
- **Perspective-diverse review** — several reviewers each given a distinct lens (correctness, security, performance, maintainability) rather than several identical reviewers
- **Generate-then-judge** — N independent attempts scored by separate judges, synthesising from the winner

**The important qualifier:** independence must be real. If the reviewer receives the author's reasoning and justifications, it is anchored and you have paid for an echo. Give it the artifact and the requirements, not the story.

---

## 5. The context tax: why handoffs lose information

Now the cost, in detail, because this is what kills designs.

When Agent A hands off to Agent B, everything B knows arrives through a message. A knew: what it tried, what failed, what surprised it, which sources were shaky, which parts it was unsure about. B receives a summary — and **critically, B does not know what was omitted.** It cannot ask about a constraint it doesn't know exists.

Failure shapes that follow:

**The telephone game.** Each hop degrades fidelity. A → B → C → D and the original requirements are unrecognisable by D. Deep chains are far worse than wide fan-outs for exactly this reason.

**Lost constraints.** The user said "don't touch the auth module." The orchestrator's task description didn't repeat it. The subagent modifies the auth module — correctly, given what it was told.

**Duplicated and conflicting work.** Two agents given overlapping scopes solve the same problem differently. Merging is now a third problem nobody planned for.

**Confident propagation of error.** Agent A's plausible-but-wrong finding arrives at B stripped of its uncertainty, because summaries lose hedging first. B builds on it as fact. This is the most insidious failure, because the system's confidence *increases* as the information degrades.

**Practical mitigations:**
- **Pass requirements verbatim, not summarised.** Constraints especially — repeat them in every subagent's brief. Cheap; prevents the worst failures.
- **Prefer shallow and wide over deep and narrow.** One orchestrator with ten workers loses far less than a five-level chain.
- **Structure the handoff.** A schema (S7) forces the important fields to be present rather than hoping they survive prose compression.
- **Make uncertainty a required field.** If a subagent must report confidence and open questions explicitly, hedging survives the boundary instead of being the first thing dropped.
- **Give subagents a way back to the source.** A subagent that can re-read the original file is more robust than one working purely from a description.

---

## 6. Topologies

| Pattern | Shape | Good for | Main risk |
|---|---|---|---|
| **Orchestrator–worker** | One coordinator, N workers | Parallel independent subtasks | Orchestrator context grows with results |
| **Pipeline** | A → B → C, fixed stages | Well-understood staged work | Telephone game; a barrier at each stage |
| **Debate** | Agents argue to consensus | Genuinely contested judgements | Expensive; often converges on confident nonsense |
| **Reviewer / critic** | Worker + independent checker | Quality gates, verification | Reviewer must be genuinely independent |
| **Blackboard** | Shared state, agents read/write | Loosely coupled collaboration | Races, conflicting writes |
| **Hierarchical** | Orchestrators of orchestrators | Very large decompositions | Compounding context loss per level |

**Orchestrator–worker is the workhorse** and should be your default when you go multi-agent at all. It's shallow (one hop, so minimal telephone effect), naturally parallel, and easy to reason about.

**A note on debate:** it's the most intuitively appealing and the least reliably useful. Agents frequently converge through mutual agreement rather than through evidence — they're trained to be agreeable — so you can pay N× for confident consensus that's no better than one agent's answer. If you use it, force genuine disagreement structurally: assign opposing positions, require citations, and score arguments with a separate judge rather than letting the debaters decide.

---

## 7. The cost model

Be blunt about this, because it's where multi-agent designs surprise people.

**Multi-agent multiplies tokens, and often superlinearly:**
- Every agent pays for its own system prompt and tool definitions (S5)
- The orchestrator pays for every subagent's returned result
- Retries and error handling multiply per agent
- Failed subagent work is still billed

A 10-worker fan-out can easily cost 15–20× a single agent, not 10×, once you count orchestration overhead and the results flowing back.

**What makes it worth paying:**
- **Latency matters and the work parallelises** — you're converting money into wall-clock time, deliberately
- **Isolation prevents failure** — a single agent would simply fail on this task at any cost
- **The task's value is high** and correctness dominates cost

**What doesn't justify it:**
- "It's a more sophisticated architecture"
- "Specialists should be better than a generalist" — often untrue when the generalist has the full context and the specialist doesn't
- "It's how a human team would do it" — human teams are structured around humans' inability to hold everything in one head and their inability to be cloned; neither constraint applies the same way here

**Cost controls that work:** cap subagent count and recursion depth structurally (not by prompting); set token budgets per subagent; make prompt caching effective by keeping shared prefixes stable across workers (S5); use cheaper models for mechanical subtasks and expensive ones only for hard judgement.

---

## 8. Failure modes

**Coordination overhead exceeding the benefit.** Below a certain task size, orchestration costs more than it saves. Small tasks should not be delegated.

**Infinite delegation.** An agent with a delegate tool delegates to an agent with a delegate tool. **Cap depth in code**, not in the prompt — prompts are not a control mechanism for a runaway loop.

**Conflicting writes.** Two agents editing the same file concurrently corrupt it. Either partition the work so scopes cannot overlap, or isolate execution (separate directories, git worktrees, or transactional application of changes at the end).

**Error compounding.** With N sequential agents each 90% reliable, end-to-end reliability is 0.9^N — 59% at five stages. **Sequential chains are reliability multipliers in the wrong direction**, which is another argument for shallow-and-wide.

**The orchestrator becoming the bottleneck.** If ten workers return 5,000 tokens each, the orchestrator now holds 50,000 tokens of results — you've reconstructed the context problem you were solving. Force short, structured returns.

**Diffusion of responsibility.** Every agent assumes another handled the edge case. Nobody did. Assign explicit ownership in the task description.

**Debugging difficulty.** A failure now lives somewhere across N transcripts. Without tracing (Week 23), you are guessing. **Instrument before scaling** — this becomes acutely painful precisely when the system is complex enough to need it.

---

## 9. Evaluating multi-agent systems

S3's principles apply, with additions:

- **Evaluate end-to-end outcomes, not agent-level behaviour.** Every subagent can behave sensibly while the system fails. Outcome is the only thing that matters.
- **Trace the whole trajectory** (Week 23). Which agent introduced the error? Aggregate metrics can't tell you.
- **Compare against the single-agent baseline.** This is the comparison people skip, and it's the only one that answers the actual question. Week 20's "missing baseline" critique applies squarely.
- **Measure cost and latency alongside quality.** A 3% quality gain for 12× cost is a finding, not a success — and reporting it that way is the honest thing.
- **Expect higher variance.** More stochastic components means noisier results, so you need more samples for the same confidence (S3).

---

## 10. Where it genuinely wins

Being concrete, since the lecture has been mostly cautionary:

**Parallel research.** Independent questions, each requiring lots of reading, results synthesised. Both parallelism *and* isolation apply, and the reading never pollutes the synthesiser's context. This is the strongest general case.

**Large-scale code review.** One agent per file or per dimension, findings aggregated, then independently verified. Parallel, isolated, and the verification step wants genuine independence.

**Verification and adversarial checking.** A separate agent trying to refute a finding, defaulting to "unproven." Real independence is the whole point.

**Broad sweeps and migrations.** Hundreds of similar transformations across a codebase — parallelism is the entire benefit, and worktree isolation solves the conflicting-writes problem cleanly.

**Long-horizon work exceeding one context window.** When the job genuinely cannot fit, isolation isn't an optimisation, it's the enabling condition.

**Where it usually loses:** conversational assistants, anything needing tight coherence across the whole task, small tasks, latency-sensitive interactive work, and anything where the "specialists" would all need the same full context anyway.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 8** (ReAct agents) | The unit of composition. Master one agent first — a multi-agent system is N of them plus coordination overhead. |
| **Week 10** (context rot) | The core justification for delegation: subagents are context firewalls. |
| **Week 11** (RLM) | The same insight — explore a large space without dragging it all into one window. |
| **Week 17** (harness/evals) | Fix tools, context, and feedback before adding agents. Most "needs multi-agent" diagnoses are harness problems. |
| **Week 20** (reading papers) | The missing-baseline critique: multi-agent claims must be compared against a well-tuned single agent. |
| **Week 21** (LangGraph) | The orchestration layer. Multi-agent is one thing you can build with it, not a replacement for it. |
| **Week 23** (observability) | Tracing goes from useful to mandatory. A failure spans N transcripts. |
| **S3** (evaluation) | More stochastic components means higher variance and more samples needed. |
| **S5** (inference cost) | Token cost multiplies superlinearly; caching and model routing are the controls. |
| **S7** (structured output) | Schemas on handoffs are the main defence against lossy compression at boundaries. |
| **S8** (MCP) | Servers are shared across agents, so integrations are written once; also N× the tool-definition tokens. |

---

## Key takeaways

1. **Multi-agent buys three things** — parallelism, context isolation, independent perspective. If you need none of them, you are paying a tax for nothing.
2. **The tax is that context doesn't survive handoffs**, and neither agent knows what was lost.
3. **A single agent with good tools beats a multi-agent system with bad ones.** Fix the harness first (Week 17).
4. **Context isolation is the strongest argument** — a subagent is a context firewall, and the return value is the product, not the exploration.
5. **Independence must be real.** A reviewer given the author's reasoning is anchored, and you bought an echo.
6. **Prefer shallow and wide.** Deep chains compound both context loss and unreliability — 0.9^5 = 59%.
7. **Pass constraints verbatim** into every subagent brief. Summarised requirements lose exactly the constraints that matter.
8. **Make uncertainty a required field** in handoffs, or hedging will be the first casualty of compression.
9. **Orchestrator–worker is the default.** Debate is the most appealing and least reliable pattern.
10. **Cost multiplies superlinearly** — a 10-worker fan-out often runs 15–20× a single agent.
11. **Cap depth and concurrency in code**, never in the prompt.
12. **Always compare against a well-tuned single-agent baseline**, and report cost and latency alongside quality.

---

## Glossary

| Term | Definition |
|---|---|
| **Orchestrator** | Agent that decomposes a task, dispatches subagents, and synthesises results |
| **Subagent / worker** | Agent executing a scoped subtask and returning a result |
| **Context firewall** | Using a subagent so its exploration never enters the parent's context |
| **Handoff** | Transfer of task and context between agents; inherently lossy |
| **Telephone game** | Cumulative fidelity loss across a chain of handoffs |
| **Fan-out / fan-in** | Dispatching N parallel subagents, then aggregating their results |
| **Barrier** | A synchronisation point where all parallel work must finish before proceeding |
| **Pipeline** | Items flowing through stages independently, with no barrier between stages |
| **Debate** | Agents arguing toward consensus; prone to agreeable convergence |
| **Adversarial verification** | An agent tasked with refuting a finding, defaulting to "unproven" |
| **Perspective diversity** | Giving each reviewer a distinct lens rather than duplicating reviewers |
| **Blackboard** | Shared mutable state that multiple agents read and write |
| **Error compounding** | Reliability multiplying down a sequential chain (0.9^N) |
| **Diffusion of responsibility** | Each agent assuming another handled an edge case; none did |
| **Worktree isolation** | Giving each agent its own copy of a repo to prevent conflicting writes |

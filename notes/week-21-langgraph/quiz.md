# Week 21 — Quiz (20 questions)

**Topic:** LangGraph — agents as state machines
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The three LangGraph primitives are:
- A) Chains, tools, memory
- B) State, nodes, edges
- C) Prompts, models, parsers
- D) Agents, supervisors, workers

**2.** A node's return value is:
- A) The complete new state
- B) A partial state update that the graph merges
- C) The next node's name
- D) A string response

**3.** The `add_messages` reducer:
- A) Overwrites the messages list
- B) Appends and updates by ID
- C) Deletes old messages
- D) Sorts messages by timestamp

**4.** Multiple out-edges from a single node cause the targets to:
- A) Run in sequence
- B) Run in parallel
- C) Raise an error
- D) Be chosen randomly

**5.** `Command(update={…}, goto="next")` replaces:
- A) The checkpointer
- B) The router plus the edge
- C) The state schema
- D) The compile step

**6.** The checkpointer stores state:
- A) Across all threads globally
- B) Within a thread, keyed by `thread_id`
- C) Only in memory, never on disk
- D) Only for the final output

**7.** Which is required for `interrupt()` to work?
- A) A store
- B) A checkpointer
- C) A subgraph
- D) Streaming enabled

**8.** On resume after an interrupt, the node:
- A) Continues from the interrupt line
- B) Re-runs from the top
- C) Is skipped entirely
- D) Runs in a new thread

**9.** `interrupt_before` / `interrupt_after` are described as:
- A) The recommended production HITL mechanism
- B) Static breakpoints for debugging only
- C) Equivalent to `interrupt()`
- D) Deprecated

**10.** `Send(node, slice)` from a conditional edge provides:
- A) Streaming output
- B) Runtime fan-out to N copies — map-reduce
- C) Human approval
- D) Structured output

**11.** Which is the store's role (as opposed to the checkpointer)?
- A) State within a thread
- B) Memory across threads, namespaced, with optional semantic search
- C) Compiled graph structure
- D) Tool definitions

**12.** The lecture says NOT to use LangGraph for:
- A) Loops and state
- B) Human-in-the-loop
- C) One-shot prompts and linear chains
- D) Durable execution

---

## Short answer

**13.** Explain the reframe from Week 8's while-loop to a graph, mapping each construct.

**14.** Explain reducers and why they matter for keeping nodes simple.

**15.** Explain what persistence unlocks and why all four capabilities follow from one mechanism.

**16.** Explain the interrupt gotchas, especially the interaction between node re-runs and side effects.

**17.** Contrast the checkpointer and the store, connecting to Week 18.

**18.** Explain time travel and give a concrete debugging scenario where it earns its keep.

**19.** Explain how LangGraph middleware implements a primitive named in Week 17, with two examples.

**20.** You're building a research agent that runs for 20 minutes, calls paid APIs, and needs human approval before publishing. Design it in LangGraph, justifying each feature.

---
---

## Answer key

**1. B** — State (shared snapshot), nodes (functions reading state and returning updates), edges (deciding what runs next).

**2. B** — A partial state update; the graph merges it according to the reducers.

**3. B** — Appends new messages and updates existing ones by ID.

**4. B** — In parallel.

**5. B** — The router function plus the edge, combining update and routing in one return.

**6. B** — Within a thread, keyed by `thread_id`, snapshotting at every super-step.

**7. B** — A checkpointer; without persisted state there is nothing to resume from.

**8. B** — It re-runs from the top, which is why prior side effects must be idempotent.

**9. B** — Static breakpoints for debugging only; `interrupt()` is real HITL.

**10. B** — Runtime fan-out to N node copies, i.e. map-reduce.

**11. B** — Cross-thread memory, namespaced, with optional semantic search.

**12. C** — One-shot prompts (just call the model) and linear chains (a function is fine).

**13.** In Week 8 the agent was `while not done: think → pick tool → run tool → observe`, with **state in local variables** and **control flow in `if`/`break`**. In LangGraph the same program becomes: the loop body splits into **two nodes** (an agent node that calls the model and a tool node that executes calls); the `while` **condition becomes a router function** on a conditional edge asking "are there `tool_calls`?"; **`break` becomes an edge to `END`**; the **tools → agent edge closes the loop**; and the local variables become **explicit state** merged by reducers. The semantics are identical — the picture is literally `agent ⇄ tools, agent → END`. What changes is that control flow is now a **data structure rather than Python control flow**, which is what makes it inspectable, checkpointable, pausable, and forkable.

**14.** A reducer defines how a node's partial update merges into existing state. The default **overwrites**; `Annotated[list, add]` **appends**; `add_messages` **appends and updates by ID**. **Why they matter:** without reducers, a node returning `{"messages": [new_msg]}` would replace the entire conversation history, so every node would have to read the current state, copy it, append, and return the whole thing — making nodes stateful, verbose, and easy to get wrong, especially with parallel edges where two nodes might clobber each other. With reducers, a node returns only what it produced and the graph handles accumulation correctly and consistently. This keeps nodes **pure and local** — they see state, emit a delta, and never worry about merge semantics — which is what makes parallel out-edges safe and the whole model composable.

**15.** Persistence means the checkpointer **snapshots state at every super-step, keyed by `thread_id`**. From that single mechanism you get: **conversational memory** — re-invoking with the same `thread_id` restores prior state, so turn two remembers turn one; **human-in-the-loop** — execution can pause because there is a durable snapshot to resume from; **time travel** — every past step is retained, so `get_state_history()` can list checkpoints and `update_state()` can fork from any of them; and **fault tolerance** — a crash loses nothing, since resuming the same `thread_id` with `None` input continues from the last checkpoint. **Why all four follow from one mechanism:** each is fundamentally about *state existing outside the running process at a known point in time*. Once execution is decomposed into discrete super-steps with a durable snapshot after each, pausing, resuming, replaying, and branching are all the same operation applied differently. This is why the slides say you get all of it "from compile with a checkpointer."

**16.** The rules are: the **node re-runs from the top on resume**; **no bare `try`/`except` around `interrupt`**; interrupts must occur in the **same order every run** (they are index-matched); payloads must be **JSON-serializable**; and **side effects before an interrupt must be idempotent**. **The dangerous interaction** is between the first and last. `interrupt()` does not suspend a function mid-execution the way a coroutine would — on resume the **entire node body re-executes**, and everything before the interrupt runs a second time. If that code sent an email, charged a card, posted to an API, or wrote a file, it happens again on every resume. The fix is to make those operations idempotent — use an idempotency key, check-then-act, or wrap them in `@task` so they are not replayed. The `try`/`except` rule follows from the same mechanism: `interrupt` signals control flow by raising, so a bare `except` swallows it and the graph never pauses. Index-matching explains the ordering rule — resume values are matched positionally, so conditional interrupts that appear in different orders across runs will resume with the wrong values.

**17.** The **checkpointer** persists state **within a thread**, keyed by `thread_id`, snapshotting at every super-step — this is the state of one conversation or one run, enabling resume, HITL, and time travel for that specific thread. The **store** holds memory **across threads**, is namespaced, and optionally supports semantic search — this is what the system knows about a user or domain regardless of which conversation it learned it in. **Connection to Week 18:** this is precisely that lecture's distinction between conversation state and memory. The checkpointer is the current session's working state, analogous to the sliding window of an active conversation; the store is long-term memory, analogous to the extracted facts and preferences that persist across sessions. The practical consequence is the same too: things belonging in the store need write-time decisions about what is worth keeping and what supersedes what, whereas checkpointer state is simply everything that happened in this thread.

**18.** Time travel comprises `get_state_history()` to list past checkpoints, replay from any of them, and `update_state()` to **fork a new branch** from a chosen point. **A concrete scenario:** a research agent runs fifteen steps, and at step nine the model picks the wrong tool — it searches the web when it should have queried the internal database — after which every subsequent step builds on that bad result and the final answer is wrong. Without time travel you re-run from scratch, paying for all fifteen steps again and, because sampling is stochastic, possibly not even reproducing the failure. With time travel you list the checkpoints, rewind to step eight, use `update_state()` to correct the state — inject the right tool call, or edit the message that misled the router — and **fork** an alternate path, replaying only the six steps after the divergence while keeping the original branch intact for comparison. It turns debugging a long stochastic run from "re-run and hope" into something closer to editing and re-executing a notebook cell.

**19.** Middleware implements Week 17's **"deterministic interventions"** primitive — the harness capability that exists because a model cannot be relied upon to do something *every single time*, so it must be encoded in code rather than in a prompt. LangGraph exposes it through `create_agent` with hooks: `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`, plus `before_agent` and `after_agent`. **Example one — PII redaction:** a `before_model` hook strips emails, phone numbers, and account identifiers from anything about to be sent to the provider. Instructing the model not to leak PII is a probabilistic request; a redaction pass is a guarantee, which matters when the requirement is legal rather than aspirational. **Example two — summarization**, which is Week 17's **compaction** by another name: an `after_model` or `before_model` hook checks context length and compresses older turns once a threshold is crossed, producing that lecture's characteristic saw-tooth context graph. Both are things that must happen unconditionally, on a schedule the model does not control — hence code, not prompt. (Guardrails, retries, and dynamic model/tool selection are equally valid answers.)

**20.** **Framework choice is justified up front:** the task needs loops, durable state, human-in-the-loop, and fault tolerance — exactly the four conditions the lecture gives for reaching for LangGraph, and none of the anti-patterns (it is neither a one-shot prompt nor a linear chain). **State and structure:** subclass `MessagesState` to add fields for accumulated findings and sources, using an `add`-annotated list so parallel research nodes append rather than overwrite. The core is the standard agent ⇄ tools loop with a router on `tool_calls`, plus a final publish node. **Persistence — non-negotiable at 20 minutes:** compile with a **`PostgresSaver`**, not `InMemorySaver`, since a 20-minute run will eventually span a deploy, a restart, or a crash. A persistent checkpointer *is* fault tolerance: on crash, resume the same `thread_id` with `None` input and continue from the last super-step rather than repeating twenty minutes of paid API calls. Set `durability="sync"` for the segments where losing a step is expensive. **Paid API calls — idempotency is the real hazard:** wrap every paid call in **`@task`** so it is not re-executed on replay, and give each an idempotency key. This matters most around the interrupt, because **the node re-runs from the top on resume** — an unprotected paid call placed before `interrupt()` will be charged again on every approval. **Human approval:** use a **dynamic `interrupt()`** immediately before the publish node — not `interrupt_before`, which is a debugging breakpoint — passing a JSON-serializable payload containing the draft and its sources so the reviewer can inspect and edit. Resume with `Command(resume=…)`, and keep the interrupt as the **first** statement in the publish node so nothing side-effecting precedes it. **Parallelism:** use **`Send`** to fan out one research worker per identified subtopic, since the number of subtopics is only known at runtime. **Observability:** stream with `stream(version="v2")` in `updates` mode so a 20-minute run is not a black box, and keep `get_state_history()` available so a bad step can be rewound and forked rather than re-run from zero.

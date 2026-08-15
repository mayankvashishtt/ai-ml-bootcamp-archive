# Week 21 — LangGraph

**Subtitle:** Agents as state machines
**Date:** 06/06/2026
**Sources:** `downloads/week-21-langgraph.pdf` (43 slides, 49-slide deck) · `downloads/week-21-langgraph.ipynb` (27 cells)
**Notion page:** https://100xschool.notion.site/379ffffa33e580e99c51cf3d26190ca4

**Referenced links:**
- [docs.langchain.com — DeepAgents overview](http://docs.langchain.com/oss/python/deepagents/overview)
- [github.com/langchain-ai/open-swe](https://github.com/langchain-ai/open-swe)

---

## 0. The idea in plain language

Week 8's agent was a `while` loop:

```python
while not done:
    think → pick tool → run tool → observe
```

That works, and for many agents it's genuinely all you need. **This week is what to do when it stops being enough.**

**Where the plain loop breaks down:**

- You need **different paths** — if the question is about billing, do X; if it's technical, do Y
- You need **a human to approve** something mid-run, then resume where it left off
- The process **crashes at step 7** and you'd like not to redo steps 1–6
- You need to **see what happened**, and a loop with mutated local variables is opaque
- Several steps could run **in parallel**

Retrofitting all that into a `while` loop produces a tangle of flags and nested conditionals that nobody can reason about.

**LangGraph's reframe: stop writing a loop, and describe a state machine.**

You define three things:

- **State** — one object holding everything the run knows, passed from step to step
- **Nodes** — functions that take state and return an updated state ("call the model," "run the tool," "check the answer")
- **Edges** — which node runs next, either fixed or **conditional** on the state

```
        ┌──────────┐
        │  agent   │◄──────────┐
        └────┬─────┘           │
             │ conditional     │
      ┌──────┴──────┐          │
      ▼             ▼          │
  ┌───────┐    ┌────────┐      │
  │ tools │    │  END   │      │
  └───┬───┘    └────────┘      │
      └────────────────────────┘
```

**Once the flow is data rather than control flow, you get things for free** — and this is the actual argument for the framework:

- **Persistence** — state is a serialisable object, so you can checkpoint it and resume after a crash
- **Human-in-the-loop** — pause at a node, wait for approval, resume. Trivial with checkpoints, painful in a loop
- **Observability** — every node transition is a recordable event, so you can see exactly what path a run took
- **Time travel** — rewind to an earlier state and try a different branch
- **Parallelism** — independent branches are visible in the graph, so they can run concurrently

**The honest framing, and the lecture makes it:** this is **machinery, and machinery has a cost.** A three-step agent in LangGraph is more code and more concepts than the same agent as a loop. Reach for it when the complexity is *real* — branching, approval gates, resumability — not in anticipation of complexity that may never arrive. Week 8's loop remains the right default.

**One distinction worth fixing now, because it confuses people later:** LangGraph is about **control flow** — what runs next. MCP (S8) is about **integration** — how a tool is exposed and consumed across a process boundary. They're different layers and they compose: a LangGraph node can call an MCP tool. Asking "LangGraph or MCP?" is like asking "state machine or USB?"

---

## Where we are

- **Week 8** — agent from scratch: `while`-loop + tool dispatch
- Everything since — RAG, fine-tuning, RL, memory
- **Today — the loop, formalized**

### The callback to Week 8

```python
while not done:
    think → pick tool → run tool → observe
```

- **State lived in local variables.**
- **Control flow lived in `if` / `break`.**

### The reframe

> ### **nodes = the work · edges = what's next · state = the whiteboard**

**Three primitives:**

| Primitive | Definition |
|---|---|
| **State** | Shared snapshot |
| **Nodes** | Functions that read state, return updates |
| **Edges** | Decide the next node |

That's the entire conceptual model. Everything else is convenience on top.

---

## The primitives

> **Six lines is the whole API surface.**

### State

- A **TypedDict** schema
- Nodes return **partial updates**
- **The graph merges them**

**Reducers** — how updates merge into existing state:

| Reducer | Behaviour |
|---|---|
| default | **overwrite** |
| `Annotated[list, add]` | **append** |
| `add_messages` | **append + update by ID** |

**`MessagesState`** — prebuilt state with one `messages` key and the `add_messages` reducer baked in. **Subclass to add fields.**

Reducers are the design detail that makes this work. A node returning `{"messages": [new_msg]}` doesn't clobber the history — the reducer appends. So nodes stay pure and local while state accumulates correctly.

### Nodes

```python
def node(state) -> dict:
    ...
```

- Optional: `config`, `runtime`
- **Return is a state update — not the full state**

### Edges

| Call | Behaviour |
|---|---|
| `add_edge(A, B)` | Normal |
| `add_conditional_edges(A, router)` | Conditional |
| `START` / `END` | The entry and exit |
| **Multiple out-edges** | **Run in parallel** |

### Compile

```python
builder.compile()
```

- Runs **structure checks**
- Attaches the **checkpointer / store**
- **Must compile before use**

### The hello-world graph

```python
g = StateGraph(MessagesState)
g.add_node("chat", chat_node)
g.add_edge(START, "chat")
g.add_edge("chat", END)
app = g.compile()
app.invoke({"messages": [...]})
```

> **Six lines. This is the whole API surface.**

---

## Runtime

### Command

```python
Command(update={…}, goto="next")
```

> **Update and route in one return — replaces the router + the edge.**

### Persistence = the checkpointer

- **Snapshot at every super-step**
- **Keyed by `thread_id`**
- **Backends:** `InMemorySaver` → `SqliteSaver` → `PostgresSaver`

**What persistence unlocks:**
- Conversational memory
- Human-in-the-loop
- **Time travel**
- **Fault tolerance**

This is the payoff of the whole framework. Because state is explicit and snapshotted at every step, four capabilities that are painful to build by hand come from a single constructor argument.

### Threads vs store

| | Scope |
|---|---|
| **checkpointer** | State **within** a thread |
| **store** | Memory **across** threads — namespaced, optional semantic search |

The Week 18 distinction, in API form: the checkpointer is conversation state; the store is memory.

### Interrupts (human-in-the-loop)

```python
interrupt("approve?")      # pauses
Command(resume="yes")      # continues
```

- **Needs a checkpointer**
- `invoke(version="v2")` → `GraphOutput.interrupts`

**⚠️ Interrupt rules — the gotchas:**

- **Node re-runs from the top on resume**
- **No bare `try`/`except` around `interrupt`**
- **Same order every run** (index-matched)
- **JSON-serializable payloads only**
- **Side effects before `interrupt` → make them idempotent**

The first and last are the dangerous pair. `interrupt()` doesn't suspend mid-function — on resume the **whole node re-executes**, replaying anything before the interrupt. If that included sending an email or charging a card, it happens twice.

| Static vs dynamic | |
|---|---|
| `interrupt()` | **Dynamic — real HITL** |
| `interrupt_before` / `interrupt_after` | Static breakpoints — **debugging only** |

### Time travel

- `get_state_history()`
- **Replay from any checkpoint**
- `update_state()` → **fork a new branch**

---

## Hands-on — building up in stages

| Stage | What |
|---|---|
| **1. Hello world** | `StateGraph(MessagesState)` · one node · two edges · `.invoke()` |
| **2. Bind a tool** | `@tool` · `llm.bind_tools([...])` · a tool node + an agent node |
| **3. The loop** | Router: `tool_calls?` → tools : END. The **tools → agent** edge closes the loop |
| **4. Memory** | Compile with `InMemorySaver`, invoke with a `thread_id` — **the second turn remembers the first** |
| **5. HITL** | `interrupt()` before the tool fires — inspect, approve, edit, resume with `Command` |
| **6. Time travel** | List checkpoints, rewind before a bad step, **fork an alternate path** |
| **7. Subgraphs + multi-agent** | A subgraph is a graph used as a node |
| **8. Streaming** | `stream(version="v2")` |

### Stage 3 — the loop

```
agent ⇄ tools,  agent → END
```

> **This is your Class 6 while-loop, as a graph.**

The `while` condition became a **router function**; `break` became an **edge to END**; the loop body became **two nodes**. Same semantics, now inspectable.

### Stage 7 — subgraphs and multi-agent

- A **subgraph is a graph used as a node**
- **Shared keys** → add as a node; **different schemas** → invoke inside a node
- **Persistence:** `None` (per-call) · `True` (per-thread) · `False` (stateless)
- **A supervisor routes to workers**
- **Hand-offs via `Command(goto, graph=PARENT)`**

### Stage 8 — streaming

- `stream(version="v2")` — unified `StreamPart {type, ns, data}`
- **Modes:** `values` · `updates` · `messages` · `custom` · `checkpoints` · `tasks` · `debug`
- `stream_events(version="v3")` — typed projections: `.messages` · `.values` · `.interrupts` · `.output`

### Extras

**`Send` — map-reduce:** fan out to **N copies at runtime** via `Send(node, slice)` from a conditional edge. This is dynamic parallelism — the fan-out width is decided at runtime from state, so you can spawn one worker per retrieved document without knowing the count in advance.

**Durable execution + fault tolerance:**
- **A persistent checkpointer *is* fault tolerance**
- **Crash → resume same `thread_id` with `None` input**
- `durability = "exit" / "async" / "sync"`
- **Wrap side-effects in `@task` — idempotent on replay**

**LangGraph Platform:** Agent Server (checkpointing & store, automatic) · **Studio** (visual debugging) · deploy long-running, stateful agents.

---

## Where LangGraph sits

### Graph API vs Functional API

| | |
|---|---|
| **Graph** | Explicit nodes & edges |
| **Functional** | `@entrypoint` + `@task`, plain control flow |
| **Both** | **Same runtime, persistence, streaming, HITL** |

Useful to know: you can get durability and HITL without adopting the graph mental model at all.

### Middleware (the LangChain layer)

- `create_agent` from `langchain.agents`
- **Hooks:** `before_model` / `after_model` / `wrap_model_call` / `wrap_tool_call`, plus `before_agent` / `after_agent`
- **Uses:** guardrails, summarization, PII redaction, retries, dynamic model/tool

This is Week 17's **middleware primitive** — "no deterministic interventions" — implemented. Summarization here is Week 17's **compaction**; guardrails and PII redaction are things you cannot trust a prompt to do every time.

### Structured output

- `create_agent(response_format=Schema)`
- **`ToolStrategy`** (any model) vs **`ProviderStrategy`** (native)
- Result in `["structured_response"]`
- v1: agent state must be a TypedDict

### The three layers

| Layer | What |
|---|---|
| **LangChain** | The prebuilt agent — `create_agent` |
| **LangGraph** | **Low-level orchestration** |
| **Deep Agents** | Planning / subagent harness on top |

### When NOT to use LangGraph

> - **One-shot prompt** — just call the model
> - **Linear chain** — a function is fine
> - **Reach for it when you need: loops, state, HITL, durability**

A refreshingly honest slide. The framework is justified by *durability and control-flow complexity*, not by being a framework.

### What you got for free

> Durable state · pause/resume · inspect any step · rewind + fork —
> **all from "compile with a checkpointer."**

---

## The one idea

> ### **Make control flow visible. The graph is the program.**

In Week 8 the agent's control flow was implicit in Python — you couldn't inspect it, visualise it, pause it, or resume it. Making the graph an explicit data structure is what enables everything else: you can draw it, checkpoint it, interrupt it, replay it, and fork it, precisely because the structure is data rather than code.

---

## Common confusions

**"Do I need LangGraph to build an agent?"** No. Week 8's `while` loop is a complete agent and remains the right default. LangGraph earns its cost when you need branching, approval gates, resumability, or visible parallelism — not before.

**"What's the actual difference from a `while` loop?"** The loop encodes control flow in **code**; the graph encodes it in **data**. Once "what happens next" is a data structure, tooling can inspect it, checkpoint it, replay it, and visualise it. That's the whole payoff — everything else follows from making flow inspectable.

**"What exactly is 'state'?"** One object carrying everything the run knows — messages so far, tool results, intermediate values, flags. Every node receives it and returns an updated version. Because it's a plain serialisable object, you can save it, which is what makes persistence and resumption possible.

**"Why do people say nodes should return updates rather than mutate?"** Because a returned update can be **merged, logged, and replayed**. Mutating shared state in place destroys the history that makes checkpointing and time travel work. It's the same reason functional updates are preferred in front-end state management.

**"What's a conditional edge?"** A function that inspects the state and returns the name of the next node. It's the `if` statement, lifted out of the loop body and made part of the graph — which is precisely what makes routing visible rather than buried.

**"How is this different from a normal workflow engine like Airflow?"** Traditional engines run **fixed DAGs** decided in advance. LangGraph supports **cycles** and lets the *model* decide the route at runtime via conditional edges. The graph is a constraint on what's possible, not a script of what will happen.

**"LangGraph or MCP?"** Different layers, and the question is malformed. **LangGraph is control flow** — what runs next. **MCP is integration** — how a capability is exposed across a process boundary. A LangGraph node can call an MCP tool. They compose.

**"LangGraph or multi-agent?"** Also different layers. Multi-agent (S12) is a *pattern* — several agents with separate contexts. LangGraph is *orchestration machinery* you might use to implement it. You can build multi-agent systems with or without it.

**"When is it the wrong choice?"** Simple linear tasks; anything where a single well-tuned prompt suffices; latency-critical paths where framework overhead matters; and small teams who'd spend more time learning the abstraction than the problem warrants. Complexity you don't need is still complexity.

**"Does it lock me in?"** Somewhat — your control flow is expressed in its abstractions. Mitigate by keeping tool implementations and prompts as plain functions and strings that the graph merely *calls*, so the portable parts stay portable.

---

## Key takeaways

1. **nodes = the work · edges = what's next · state = the whiteboard.** Three primitives, six lines.
2. **Nodes return partial updates**; reducers decide how they merge (`overwrite` / `add` / `add_messages`).
3. **Week 8's while-loop is a graph**: the condition is a router, `break` is an edge to END.
4. **Persistence is the whole payoff** — memory, HITL, time travel, and fault tolerance from one checkpointer.
5. **checkpointer = within a thread; store = across threads** (Week 18's distinction in code).
6. **`interrupt()` re-runs the node from the top on resume** — make prior side effects idempotent.
7. **Static breakpoints are for debugging; `interrupt()` is real HITL.**
8. **Time travel = `get_state_history()` + `update_state()`** to fork a branch.
9. **`Send` gives runtime map-reduce** — fan-out width decided from state.
10. **Middleware implements Week 17's "deterministic interventions"** primitive.
11. **Don't use it for one-shot prompts or linear chains** — reach for it when you need loops, state, HITL, or durability.
12. **Make control flow visible. The graph is the program.**

---

## Glossary

| Term | Meaning |
|---|---|
| **StateGraph** | The graph builder — nodes, edges, and a state schema. |
| **State** | A TypedDict snapshot shared across nodes. |
| **Reducer** | Rule merging a node's partial update into state (overwrite / add / add_messages). |
| **`MessagesState`** | Prebuilt state with a `messages` key and the `add_messages` reducer. |
| **Node** | `def node(state) -> dict` returning a partial state update. |
| **Edge** | A transition; conditional edges route via a router function. |
| **START / END** | The graph's entry and exit sentinels. |
| **Super-step** | One round of graph execution; the checkpoint granularity. |
| **`compile()`** | Validates structure and attaches checkpointer/store. |
| **`Command`** | Return value combining a state update with a routing decision. |
| **Checkpointer** | Persistence layer snapshotting state per `thread_id`. |
| **`thread_id`** | Key identifying one conversation/run. |
| **Store** | Cross-thread memory, namespaced, with optional semantic search. |
| **`interrupt()`** | Dynamic pause for human input; requires a checkpointer. |
| **`interrupt_before` / `after`** | Static breakpoints for debugging. |
| **Time travel** | Replaying from a past checkpoint, optionally forking via `update_state()`. |
| **Subgraph** | A compiled graph used as a node. |
| **Supervisor** | A routing node dispatching to worker agents. |
| **`Send`** | Runtime fan-out to N node copies — map-reduce. |
| **Durable execution** | Crash-resume by re-invoking the same `thread_id` with `None`. |
| **`@task`** | Wrapper making side effects idempotent on replay. |
| **Middleware** | LangChain hooks around model/tool calls for guardrails, summarization, retries. |
| **`ToolStrategy` / `ProviderStrategy`** | Structured-output via tool calling vs native provider support. |
| **Functional API** | `@entrypoint` + `@task` alternative with the same runtime guarantees. |
| **Deep Agents** | Planning/subagent harness layered above LangGraph. |

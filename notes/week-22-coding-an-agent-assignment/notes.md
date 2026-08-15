# Week 22 — Coding an Agent (Assignment)

**Date listed:** 13/12/2026 *(almost certainly a typo for **13/06/2026** — it sits between Week 21 on 06/06 and Week 23 on 20/06, and the linked repo is named `13-june-assignment`)*
**Sources:** *No slides and no Colab notebook this week* — the content is three links.
**Notion page:** https://100xschool.notion.site/397ffffa33e580febdecc04c91c54067

**The three links:**
1. **[Build a Mini Workflow Orchestration Engine](https://brindle-goal-102.notion.site/Build-a-Mini-Workflow-Orchestration-Engine-37d46b36b2e980b4aa95df09bfa31019)** — the assignment spec (reproduced below)
2. [github.com/rahul-MyGit/contest-2](https://github.com/rahul-MyGit/contest-2)
3. [github.com/100xdevs-bootcamp-1/13-june-assignment](https://github.com/100xdevs-bootcamp-1/13-june-assignment)

> ⚠️ **Scope note:** despite the syllabus title "Coding an Agent," the linked assignment is a **distributed workflow orchestration engine** — DAG resolution, Redis queues, Kubernetes pod leasing. There is no LLM in it. It's a systems-engineering exercise, and a good one; just don't expect agent code. The connection to the course is real but indirect (see "Why this belongs in an AI course" below).

---

## The goal

Build a mini workflow orchestration system where:

- A **backend accepts workflows via HTTP**
- Steps are executed inside **leased Kubernetes runner pods**
- A **step queue** manages execution order
- A **result queue** sends results back to the backend **asynchronously**
- Dependencies between steps are respected via a **DAG resolver**
- **The backend owns all state** — users can query step status at any time

> **Read this first:** complete Section 1 fully before moving to Section 2. **A working Section 1 is worth more than a broken Section 3.**

---

## System architecture

```
Client
   ↓  POST /workflow
Main Backend
   ↓  enqueue ready steps
Step Queue
   ↓  dequeue + dispatch
Pod Manager
   ↓  acquire lease → execInPod()
Runner Pod
   ↓  command output returned to
Pod Manager
   ↓  pushes StepResult event
Result Queue
   ↓  consumed by
Main Backend
   ↓  re-runs DAG resolver
   ↓  enqueues newly unblocked steps
Step Queue  (loop continues)
```

**The shape to notice:** it's a **closed loop**, and the only feedback channel from executor back to brain is the **result queue**. That single constraint is what makes the design clean.

---

## What's given in the boilerplate

You do **not** need to write Kubernetes code or Redis queue plumbing.

```ts
await podPool.acquirePod(): Promise<Pod>
await podPool.releasePod(podId: string): void
await podPool.execInPod(podId: string, command: string): Promise<string>
await podPool.getPods(): Promise<Pod[]>
podPool.getPoolStatus(): PoolStatus
```

Also provided: `src/k8s/pod-pool.ts` (pod discovery, leasing, release, exec) · `src/queue/redis-client.ts` · `src/queue/step-queue.ts` (FIFO) · `src/queue/result-queue.ts` (push + `BRPOP` consume loop) · `src/backend/workflow-store.ts` · `src/types/workflow.ts`

> **Your job is to build the orchestration logic on top of these.**

---

## The example workflow

```json
{
  "workflowId": "wf-123",
  "steps": [
    { "id": "A", "command": "sleep 2" },
    { "id": "B", "command": "ls /",  "dependsOn": ["A"] },
    { "id": "C", "command": "pwd",   "dependsOn": ["A"] },
    { "id": "D", "command": "date",  "dependsOn": ["B", "C"] }
  ]
}
```

```
    A
   / \
  B   C
   \ /
    D
```

- A runs first (no dependencies)
- **B and C wait for A** — and can run in parallel
- **D waits for both B and C**

**Dependency rules:** a step can only run after **all** its dependencies are `COMPLETED`; a step with no `dependsOn` is immediately runnable.

---

## Step status lifecycle

```
PENDING
   ↓  (backend enqueues step into step queue)
QUEUED
   ↓  (pod manager acquires lease + publishes RUNNING event)
RUNNING
   ↓
COMPLETED   or   FAILED
```

### Who owns each transition

| Transition | Owner | When |
|---|---|---|
| `PENDING` | **Backend** | On `POST /workflow` |
| `QUEUED` | **Backend** | When the step is pushed into the step queue |
| `RUNNING` | **Backend** | When it consumes the pod manager's `RUNNING` event from the result queue |
| `COMPLETED` / `FAILED` | **Backend** | When it consumes the pod manager's result event |

> **Pod manager owns execution, not workflow state.** It acquires/releases pods and emits `StepResult` events. **Backend consumes those events and writes all statuses.**

**Every single transition is owned by the backend.** The pod manager — the component doing the actual work — never writes state. That's the central design decision, and it's what makes the system debuggable: there is exactly one writer.

### The core invariant

> ### **A step is RUNNING if and only if a pod lease exists for it.**
>
> If the lease is gone, the step must not be `RUNNING`.

An **iff** — both directions. No lease without RUNNING (leak), no RUNNING without a lease (a stuck step, which Section 3 Task 3 exists to detect).

---

## Types

```ts
type StepStatus = "PENDING" | "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED"

type WorkflowStep = {
  id: string
  command: string
  dependsOn?: string[]
  retries?: number
}

type Workflow = { workflowId: string; steps: WorkflowStep[] }

type QueuedStep = {
  stepId: string; workflowId: string; command: string; enqueuedAt: number
}

type StepResult = {
  stepId: string; workflowId: string; podId: string
  status: "RUNNING" | "COMPLETED" | "FAILED"
  stdout?: string; exitCode?: number; error?: string
}
```

## Folder structure

```
src/
├── backend/
│    ├── server.ts          ← Express routes
│    ├── orchestrator.ts    ← ties queue + pod manager + DAG together
│    ├── dag.ts             ← getReadySteps()
│    └── workflow-store.ts  ← in-memory state
├── queue/
│    ├── redis-client.ts    ← given
│    ├── step-queue.ts      ← FIFO queue of ready steps
│    └── result-queue.ts    ← receives results from pod manager
├── pod-manager/
│    └── pod-manager.ts     ← lease, exec, release, publish events
├── k8s/
│    └── pod-pool.ts        ← given
└── types/
     └── workflow.ts
```

---

# Section 1 — Core Orchestration (Mandatory)

### Task 1 — The DAG resolver

```ts
function getReadySteps(
  steps: WorkflowStep[],
  stepStatus: Record<string, StepStatus>
): WorkflowStep[]
```

**Rules:**
- Return steps whose `dependsOn` are **all** `COMPLETED`
- Return steps with no `dependsOn` immediately
- **Do not return steps already `QUEUED`, `RUNNING`, or `COMPLETED`**

```
stepStatus = { A: "COMPLETED" }  →  getReadySteps() → [B, C]
```

> **Test this in isolation.** This function is pure logic — no pods, no queues.

That third rule is the one that bites: `getReadySteps()` is called repeatedly, so without the already-dispatched filter you re-enqueue running steps and execute them twice.

### Task 2 & 3 — Use the provided queues

```ts
await stepQueue.enqueue(step) / dequeue() / peek() / size()
await resultQueue.push(result)
await resultQueue.consume(async (result) => { /* update workflow state */ })
```

> Students implement **what happens when a result is received**, not how Redis blocking pop works.

### Task 4 — `POST /workflow`

1. Store the workflow in memory
2. Mark all steps `PENDING`
3. ⚠️ **Save the workflow state *before* enqueueing any ready steps** — result events may come back quickly, and the backend must be able to find the workflow when consuming them
4. Run `getReadySteps()`
5. Push them into the step queue
6. Mark those steps `QUEUED`
7. **Return `{ workflowId, status: "accepted" }` immediately**

Step 3 is a genuine race condition and the spec calls it out explicitly. `sleep 2` is slow enough to hide it; `echo` is not.

### Task 5 — The pod manager

```
1. acquirePod()                    ← boilerplate
2. publish { status: "RUNNING" }   → result queue
3. execInPod(podId, command)       ← boilerplate
4. publish { status: "COMPLETED" } → result queue
5. releasePod(podId)               ← boilerplate

On failure:
3. execInPod() throws
4. publish { status: "FAILED" }    → result queue
5. releasePod(podId)               ← ALWAYS release
```

> **The pod manager never touches workflow state directly. It only pushes to the result queue.**

Note the ordering: `RUNNING` is published **after** the lease is acquired — which is exactly what upholds the core invariant. And `releasePod` must be in a `finally`.

### Task 6 — The orchestrator loop

```
On startup:
  Start consuming from result queue

On StepResult received:
  1. Update step status in workflow-store
  2. If COMPLETED:
       - Re-run getReadySteps()
       - Push newly unblocked steps into step queue
       - Mark them QUEUED
  3. Check if all steps COMPLETED → mark workflow "completed"

On step dequeued from step queue:
  1. Send step to pod manager for execution
```

> For Section 1 this can be **sequential** — dequeue one step, wait, dequeue next.

### Task 7 — `GET /workflow/:id`

```json
{
  "workflowId": "wf-123",
  "status": "running",
  "steps": [
    { "id": "A", "status": "COMPLETED", "podId": "runner-0" },
    { "id": "B", "status": "RUNNING",   "podId": "runner-1" },
    { "id": "C", "status": "QUEUED",    "podId": null },
    { "id": "D", "status": "PENDING",   "podId": null }
  ]
}
```

| Workflow status | When |
|---|---|
| `"pending"` | No steps running yet |
| `"running"` | At least one step is RUNNING or QUEUED |
| `"completed"` | All steps COMPLETED |
| `"failed"` | Any step FAILED (Section 3 refines this) |

### Test commands

```bash
bun run dev

curl -X POST http://localhost:3000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "workflowId": "wf-section-1",
    "steps": [
      { "id": "A", "command": "sleep 2 && echo A" },
      { "id": "B", "command": "echo B", "dependsOn": ["A"] },
      { "id": "C", "command": "echo C", "dependsOn": ["B"] }
    ]
  }'

curl http://localhost:3000/workflow/wf-section-1
```

> If you see **`501 Not implemented`**, the route isn't wired to `orchestrator.submitWorkflow()` yet.

---

# Section 2 — Parallel Execution

> Start only after Section 1 is fully working.

```
A completes
   ↓
getReadySteps() → [B, C]
   ↓
Enqueue both → dequeue both → two pods acquired simultaneously
   ↓
B and C run in parallel
```

**What to change:** after a `COMPLETED` event, immediately enqueue all newly unblocked steps; **dequeue and dispatch *all* available steps** (up to available pods); **do not wait for one to finish before starting another.**

**Concurrency rules:**
- **Never assign the same pod to two steps simultaneously**
- **Lease map must be checked atomically before acquiring**
- **If no pods are free, steps stay in the queue** until a pod is released

**Expected timeline:**
```
t=0   A starts
t=2   A completes → B and C enqueue simultaneously
t=2   B starts on runner-0, C starts on runner-1 (parallel)
t=?   Both complete → D enqueues
t=?   D starts
```

> **The important thing to verify** is that B and C are both `RUNNING` at the same time, **with different `podId` values.**

The Section 2 test uses `sleep 5` for B and C — deliberately, so the parallel window is wide enough to catch by polling.

---

# Section 3 — Production Hardening (BONUS)

### Task 1 — Failure handling

Add `SKIPPED` to the status type. When a step fails:
1. Mark it `FAILED`
2. Find all steps depending on it **directly or transitively**
3. Mark those `SKIPPED`
4. Mark the workflow `FAILED`

```
B fails
D depends on B → D becomes SKIPPED
C still runs (it only depends on A, which completed)
```

> **Think carefully: does C still run? Does D run if C completes but B failed? Implement and justify your decision.**

An open design question, deliberately. The defensible answer: **C still runs** — its dependencies are satisfied and cancelling it discards useful work — but **D never runs**, because `getReadySteps()` requires *all* dependencies COMPLETED and B never will be. Hence `SKIPPED` rather than `PENDING`: it distinguishes "will never run" from "not yet ready," which matters for reporting and for knowing the workflow is finished.

### Task 2 — Retries

```ts
type WorkflowStep = { id; command; dependsOn?; retries?: number }
```

- If a step fails and **`retries > 0`**, re-enqueue with **`retries - 1`**
- Only mark `FAILED` permanently when **`retries === 0`**
- Retried steps go **back into the step queue**

### Task 3 — Lease timeout

**The problem:**
```
Step marked RUNNING
Pod manager crashes
execInPod() never resolves
Step stuck as RUNNING forever
```

**The fix** — record a `leasedAt` timestamp on the RUNNING transition, then:

```ts
setInterval(() => {
  for each step in RUNNING state:
    if (now - leasedAt > LEASE_TIMEOUT_MS):
      mark step FAILED
      releasePod(podId)
      re-enqueue if retries remain
}, CHECK_INTERVAL_MS)
```

This is what **enforces** the core invariant. Without it, "RUNNING iff a lease exists" is an aspiration; the checker is the mechanism that reconciles reality when the executor dies.

### Task 4 — Heartbeats

```ts
type Heartbeat = { stepId: string; workflowId: string; heartbeatAt: number }
```

The pod manager pushes heartbeats to the result queue every few seconds while `execInPod()` runs; **the backend resets the lease timer on each heartbeat.** If heartbeats stop → the step is stuck → mark `FAILED`.

Heartbeats fix a real flaw in Task 3: a fixed lease timeout must exceed the longest legitimate step, so it's either too short (killing slow-but-healthy work) or too long (slow crash detection). Heartbeats decouple the two — the timeout now measures *liveness*, not *duration*, so a 30-second heartbeat window detects a crash in 30 seconds regardless of whether the step takes 5 seconds or 5 hours.

### Task 5 — Final workflow status

| Status | Condition |
|---|---|
| `"pending"` | All steps PENDING |
| `"running"` | At least one step QUEUED or RUNNING |
| `"completed"` | All steps COMPLETED |
| `"failed"` | Any step **permanently** FAILED (retries exhausted) |

---

## Evaluation criteria

| Level | Criteria |
|---|---|
| **Section 1** | DAG resolves correctly · steps execute in order · status transitions accurate · GET returns live state |
| **Section 2** | B and C run in parallel · no pod assigned twice · steps queue correctly when pods are full |
| **Section 3** | Failure propagates · retries work · stale leases recovered · heartbeat resets timeout |

## Key rules to remember

1. **Backend owns all state.** Pod manager never writes to `stepStatus` directly.
2. **Pod manager only pushes events** to the result queue.
3. **A step is RUNNING iff a pod lease exists for it.** No exceptions.
4. **Always release the pod** — on success, failure, and timeout.
5. **Result queue is the only feedback channel** from pod manager to backend.

---

## Why this belongs in an AI course

The assignment has no LLM in it, but the architecture is exactly what an agent runtime needs — and it maps onto earlier weeks:

| This assignment | The course |
|---|---|
| **DAG of steps with dependencies** | LangGraph's nodes and edges (**Week 21**) |
| **Backend owns all state; workers emit events** | LangGraph's state + reducers — one merge point, pure nodes |
| **Result queue as the only feedback channel** | The agent loop's OBSERVATION step (**Week 8**) — the executor reports, it doesn't decide |
| **Lease timeout + heartbeats** | Durable execution and fault tolerance (**Week 21**) |
| **Retries with a counter** | Loop detection and recovery primitives (**Week 17**) |
| **Idempotency on re-enqueue** | `@task` idempotency on replay (**Week 21**) |

Swap `execInPod(podId, command)` for `callModel(prompt)` and you have a multi-agent orchestrator. The point of the exercise is that **the hard parts of agent infrastructure are distributed-systems problems**, not model problems — which is Week 17's "if you're not the model, you're the harness," made concrete.

---

## Key takeaways

1. **One writer for state.** The backend owns every transition; the executor only emits events. This is what makes the system debuggable.
2. **The result queue is the sole feedback channel** — a hard architectural boundary, not a convention.
3. **`getReadySteps()` must exclude already-dispatched steps**, or you double-execute.
4. **Persist state before enqueueing** — results can return before your write lands.
5. **`releasePod` belongs in a `finally`.** Always release, on every path.
6. **"RUNNING iff a lease exists" is an invariant that needs a mechanism** — the lease-timeout checker is that mechanism.
7. **Heartbeats decouple crash detection from step duration**, which a fixed timeout cannot.
8. **`SKIPPED` ≠ `PENDING`** — it distinguishes "will never run" from "not yet ready."
9. **Ship Section 1 before touching Section 2.** A working core beats a broken feature set.

---

## Glossary

| Term | Meaning |
|---|---|
| **DAG** | Directed Acyclic Graph — steps with dependency edges and no cycles. |
| **`getReadySteps()`** | Pure function returning steps whose dependencies are all COMPLETED. |
| **Step queue** | Redis FIFO holding steps ready to execute. |
| **Result queue** | Redis queue carrying `StepResult` events back to the backend. |
| **`BRPOP`** | Redis blocking right-pop — how the consume loop waits for results. |
| **Pod lease** | Exclusive claim on a runner pod for the duration of one step. |
| **`acquirePod` / `releasePod` / `execInPod`** | The pod-pool API: claim, return, execute. |
| **Orchestrator** | The component tying DAG, queues, and pod manager together. |
| **Pod manager** | Executes steps and emits events; never writes workflow state. |
| **`StepResult`** | Event carrying status, stdout, exit code, and pod id. |
| **`PENDING` / `QUEUED` / `RUNNING` / `COMPLETED` / `FAILED` / `SKIPPED`** | The step lifecycle states. |
| **`SKIPPED`** | A step that can never run because a dependency failed. |
| **Lease timeout** | Backend-side deadline reclaiming steps whose executor died. |
| **`leasedAt`** | Timestamp recorded on the RUNNING transition. |
| **Heartbeat** | Periodic liveness event from the pod manager that resets the lease timer. |
| **Idempotency** | Property allowing a retried operation to run safely more than once. |

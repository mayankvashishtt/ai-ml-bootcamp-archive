# Week 22 — Quiz (20 questions)

**Topic:** Build a Mini Workflow Orchestration Engine (assignment)
**Format:** Q1–12 multiple choice · Q13–20 short answer / design
**Answer key at the bottom.**

---

## Multiple choice

**1.** Who owns the `COMPLETED` status transition?
- A) The pod manager
- B) The backend
- C) The runner pod
- D) Redis

**2.** The core invariant of the system is:
- A) Every step must eventually complete
- B) A step is RUNNING if and only if a pod lease exists for it
- C) The step queue is always non-empty
- D) Pods are never released early

**3.** `getReadySteps()` must NOT return steps that are:
- A) PENDING
- B) Already QUEUED, RUNNING, or COMPLETED
- C) Missing a `dependsOn` field
- D) Longer than 100 characters

**4.** The only feedback channel from pod manager to backend is:
- A) A shared database
- B) The result queue
- C) Direct function calls
- D) The Kubernetes API

**5.** In the pod manager sequence, the `RUNNING` event is published:
- A) Before acquiring the pod
- B) After acquiring the pod, before `execInPod`
- C) After `execInPod` completes
- D) After releasing the pod

**6.** `releasePod(podId)` must be called:
- A) Only on success
- B) Only on failure
- C) On success, failure, and timeout — always
- D) Only when the workflow completes

**7.** In `POST /workflow`, the spec warns that you must:
- A) Validate the DAG for cycles first
- B) Save workflow state before enqueueing ready steps
- C) Wait for the first step to finish before responding
- D) Acquire all pods up front

**8.** In the DAG `A → (B, C) → D`, if B fails, D becomes:
- A) FAILED
- B) SKIPPED
- C) PENDING forever
- D) COMPLETED

**9.** With `retries: 2`, a step that fails is:
- A) Marked FAILED immediately
- B) Re-enqueued with `retries - 1`, only permanently FAILED at 0
- C) Retried instantly in the same pod without re-queuing
- D) Marked SKIPPED

**10.** The lease timeout mechanism exists to handle:
- A) Slow network connections
- B) A pod manager crash leaving a step stuck as RUNNING forever
- C) Redis running out of memory
- D) Invalid workflow JSON

**11.** Heartbeats are pushed by:
- A) The backend, to the pod
- B) The pod manager, while `execInPod()` is running
- C) Redis, on a timer
- D) The client, while polling

**12.** The recommended completion order is:
- A) Section 3 → 2 → 1
- B) Section 1 → 2 → 3, with Section 1 fully working first
- C) All three in parallel
- D) Section 2 first, since it's the interesting part

---

## Short answer

**13.** Explain why the backend owns every state transition while the pod manager owns execution. What does this buy you?

**14.** Explain the third rule of `getReadySteps()` and what breaks without it.

**15.** Explain the race condition in Task 4 and why `sleep 2` might hide it.

**16.** State the core invariant and explain what enforces it in each direction.

**17.** The spec asks: does C still run if B fails? Does D? Answer both and justify.

**18.** Explain why heartbeats are needed when a lease timeout already exists.

**19.** Explain what changes between Section 1 and Section 2, and the concurrency rules that become necessary.

**20.** This assignment contains no LLM. Explain how its architecture maps onto agent infrastructure from earlier weeks.

---
---

## Answer key

**1. B** — The backend. It owns every transition; the pod manager only emits events.

**2. B** — A step is RUNNING if and only if a pod lease exists for it.

**3. B** — Steps already QUEUED, RUNNING, or COMPLETED, since returning them would cause double dispatch.

**4. B** — The result queue.

**5. B** — After `acquirePod()` and before `execInPod()`, which is what keeps the RUNNING status aligned with lease existence.

**6. C** — Always, on every path, which in practice means a `finally` block.

**7. B** — Save the workflow state before enqueueing, because result events can return before the write lands.

**8. B** — SKIPPED, since a dependency can never reach COMPLETED.

**9. B** — Re-enqueued with the counter decremented; permanent failure only at `retries === 0`.

**10. B** — A crashed pod manager whose `execInPod()` never resolves, leaving the step RUNNING forever.

**11. B** — The pod manager, periodically while `execInPod()` runs.

**12. B** — Section 1 → 2 → 3. "A working Section 1 is worth more than a broken Section 3."

**13.** **The split:** the pod manager acquires leases, runs commands, releases pods, and pushes `StepResult` events; the backend consumes those events and performs every write to `stepStatus`. **What it buys you: a single writer.** With one component authorised to mutate state, there are no write races between concurrent workers, no ordering ambiguity about which update wins, and exactly one place to look when a status is wrong. It also makes the pod manager **stateless and disposable** — it holds no authoritative information, so it can crash, be restarted, or be scaled horizontally without risking state corruption, and the backend can reconcile whatever it left behind via the lease timeout. Finally it makes the system **auditable**: since every transition flows through the result queue, the event stream is a complete history of what happened, which is what makes debugging a distributed run tractable at all.

**14.** The rule is: **do not return steps that are already QUEUED, RUNNING, or COMPLETED.** It matters because `getReadySteps()` is called **repeatedly** — once on submission and again after every `COMPLETED` event — and it recomputes readiness from scratch each time. **Without the rule:** consider `A → (B, C) → D`. When B completes, the resolver runs and sees that C's dependency (A) is COMPLETED, so it returns C — even though C was already enqueued and may be mid-execution. C gets pushed into the step queue a second time, dispatched to a second pod, and its command runs twice. For `pwd` that's harmless; for anything with side effects it is not. The bug also compounds: each subsequent completion re-enqueues everything still in flight.

**15.** In `POST /workflow`, the backend must store the workflow, mark steps PENDING, compute ready steps, enqueue them, and mark them QUEUED. **The race:** the moment a step is enqueued, the pod manager may dequeue it, acquire a pod, and push a `RUNNING` event — and the backend's result-queue consumer may process that event **before** the workflow has been written to the store. The consumer then looks up the workflow by id, finds nothing, and either throws or silently drops the event, leaving the step permanently stuck. Hence the spec's instruction to **save state before enqueueing**. **Why `sleep 2` hides it:** the test workflow's first step sleeps two seconds before doing anything, so even if the RUNNING event is emitted immediately, the window is wide and the store write almost certainly lands first. Replace it with a bare `echo` and the timing tightens to milliseconds, at which point the bug surfaces intermittently — the worst kind.

**16.** **The invariant: a step is RUNNING if and only if a pod lease exists for it.** **Forward direction (RUNNING ⟹ lease exists)** is enforced by the pod manager's ordering: `acquirePod()` runs *first*, and only then is the RUNNING event published, so the status can never be set before a lease is held. **Reverse direction (lease exists ⟹ RUNNING)** is enforced by always calling `releasePod()` — on success, on failure, and on timeout — so a lease never outlives the work. **But neither guarantees the invariant when the pod manager crashes**, because then `execInPod()` never resolves, no terminal event is ever published, and the step sits RUNNING with a lease nobody is using. That gap is exactly what **Section 3 Task 3's lease timeout** closes: a background checker scans RUNNING steps, and any whose `leasedAt` exceeds the timeout is marked FAILED, has its pod released, and is re-enqueued if retries remain. The invariant is a statement of intent; the checker is the mechanism that makes it true again after a failure.

**17.** **C still runs.** Its only dependency is A, which completed, so it is legitimately ready — and cancelling it would discard useful work for no reason, since B's failure tells us nothing about C's viability. **D never runs, and is marked SKIPPED.** `getReadySteps()` requires *all* of a step's dependencies to be COMPLETED, and B will never reach COMPLETED, so D can never become ready. The important part is *why SKIPPED rather than leaving it PENDING*: PENDING means "not yet ready, might run later," whereas D's situation is permanent. Without a distinct state the workflow could never be reported as finished — a poller would see a PENDING step and reasonably assume work is ongoing. SKIPPED encodes "will never run," which lets the workflow terminate as `failed` and tells the user precisely which steps were abandoned and why. Note also that C completing does **not** unblock D, since D depends on both.

**18.** A **lease timeout alone forces an impossible tuning choice**, because a fixed deadline must exceed the duration of the longest legitimate step. If `LEASE_TIMEOUT_MS` is set generously — say an hour, to accommodate a slow build — then a crashed pod manager leaves a step stuck for a full hour before recovery. If it is set tightly — say 30 seconds, for fast crash detection — then any healthy step running longer than 30 seconds is killed mid-execution and retried, wasting work and potentially looping forever. The timeout is conflating two different quantities: **how long the step takes** and **whether the executor is alive**. **Heartbeats separate them.** The pod manager pushes a liveness event every few seconds while `execInPod()` runs, and the backend resets the lease timer on each one. The timeout now measures only the gap *between heartbeats*, so a 30-second window detects a crash within 30 seconds regardless of whether the step takes five seconds or five hours. Crash detection becomes fast **and** long steps become safe.

**19.** **Section 1 is deliberately sequential** — dequeue one step, wait for it to finish, dequeue the next — so at most one pod is ever in use and correctness is easy to reason about. **Section 2 makes execution concurrent:** after a `COMPLETED` event, immediately enqueue *all* newly unblocked steps, and dequeue and dispatch *all* available steps up to the number of free pods, without waiting for any to finish. In the canonical DAG this means B and C run simultaneously on different pods after A completes. **The concurrency rules this makes necessary:** (i) **never assign the same pod to two steps simultaneously** — the whole point of leasing; (ii) **the lease map must be checked atomically before acquiring**, because two dispatches racing on a check-then-acquire can both observe the same pod as free and claim it; (iii) **if no pods are free, steps stay in the queue** until one is released, rather than being dropped or failed — backpressure, not error. The verification is behavioural: poll `GET /workflow/:id` while B and C run and confirm both show `RUNNING` **with different `podId` values**, which is why the Section 2 test gives them `sleep 5`.

**20.** The assignment is pure distributed systems, but its architecture is precisely what an agent runtime requires. The **DAG of steps with dependency edges** is LangGraph's nodes-and-edges model from Week 21, and `getReadySteps()` plays the role of the router deciding what runs next. **The backend owning all state while workers emit events** mirrors LangGraph's state-plus-reducers design, where nodes are pure and return deltas that a single merge point applies. **The result queue as the sole feedback channel** is the OBSERVATION step of the Week 8 agent loop: the executor reports what happened and never decides what happens next, which is the same asymmetry as "the model requests, your code executes." **Lease timeouts and heartbeats** are Week 21's durable execution and fault tolerance; **retries with a counter** are the loop-detection and recovery primitives from Week 17; and **idempotency on re-enqueue** is exactly the `@task` replay-safety concern, since a retried step may run twice. Substituting `callModel(prompt)` for `execInPod(podId, command)` turns this into a multi-agent orchestrator essentially unchanged. That is the pedagogical point: **the hard parts of agent infrastructure are distributed-systems problems, not model problems** — a concrete instance of Week 17's "if you're not the model, you're the harness."

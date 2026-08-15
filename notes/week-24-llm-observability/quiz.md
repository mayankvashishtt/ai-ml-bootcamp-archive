# Week 24 — Quiz (20 questions)

**Topic:** LLM Observability — traces, spans, OpenTelemetry, Langfuse
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** A trace represents:
- A) One operation, such as a single LLM call
- B) One request's full lifecycle
- C) A batch of requests
- D) A single log line

**2.** A span represents:
- A) A whole request
- B) One operation inside a trace
- C) A time window
- D) A user session

**3.** In a trace tree, the parent → child relationship encodes:
- A) Chronological order only
- B) Causality
- C) Cost allocation
- D) Retry counts

**4.** `print()` breaks down for agent systems primarily because:
- A) It is too slow
- B) Flat logs lose the tree structure of nested, concurrent calls
- C) It cannot write to disk
- D) It has no timestamps

**5.** The three properties needed from telemetry are:
- A) Fast, cheap, small
- B) Structured, causal, queryable
- C) Encrypted, compressed, replicated
- D) Real-time, batched, archived

**6.** OTLP is:
- A) An LLM provider API
- B) The OpenTelemetry wire format that Jaeger, Tempo, Datadog, and Langfuse all speak
- C) A prompt templating language
- D) A quantization format

**7.** A "generation" span carries which first-class fields?
- A) CPU, memory, disk
- B) Model, prompt/completion, token usage
- C) User, session, tag
- D) Latency, retries, status

**8.** Which is NOT listed as a gap in raw OTel for LLM work?
- A) No cost
- B) No prompt/completion rendering
- C) No place for a score
- D) No support for async code

**9.** Generic APM tools like Datadog and Jaeger fall short because they have:
- A) No distributed tracing
- B) No token concept and no prompt ↔ completion pairing
- C) No UI
- D) No sampling

**10.** Observability providers are best described as:
- A) Alternatives to OpenTelemetry
- B) OTel backends with LLM-aware storage and UI on top
- C) Replacements for evals
- D) Proxy servers only

**11.** Which provider is characterised as "proxy · near-zero setup"?
- A) Langfuse
- B) LangSmith
- C) Phoenix
- D) Helicone

**12.** The three ways to instrument with Langfuse are:
- A) Config file, CLI flag, environment variable
- B) Drop-in import, `@observe()` decorator, context manager
- C) Proxy, sidecar, agent
- D) Webhook, polling, streaming

---

## Short answer

**13.** Explain the difference between what evals tell you and what observability tells you.

**14.** Explain why flat logs fail for agent systems, and what structure is lost.

**15.** Explain why attributes must be added at instrumentation time, with a concrete consequence.

**16.** Explain "learn the concept, swap the tool" and why OTel is the durable thing to learn.

**17.** Explain precisely why generic APM cannot serve LLM observability.

**18.** Explain the significance of "`LangfuseSpanProcessor` = the span processor from Part 2."

**19.** Contrast Week 17's static eval set with scoring live traffic. Why is the latter an upgrade?

**20.** A user reports "the assistant was slow and gave a wrong answer this morning." Walk through how you'd diagnose this with an instrumented system.

---
---

## Answer key

**1. B** — One request's full lifecycle.

**2. B** — One operation inside a trace: an LLM call, a retrieval, a tool call, or custom logic.

**3. B** — Causality. Chronology alone is what flat logs already give you.

**4. B** — Agent runs are nested trees, and with concurrency the lines interleave, so the parent-child structure is lost.

**5. B** — Structured, causal, queryable.

**6. B** — The OpenTelemetry wire format spoken by all major backends.

**7. B** — Model, prompt/completion, and token usage (the cost driver).

**8. D** — Async support is not among the listed gaps; the four are cost, prompt/completion rendering, scores, and sessions/users/UI.

**9. B** — No token concept and no prompt ↔ completion pairing, since they were built for generic microservice spans.

**10. B** — OTel backends with LLM-aware storage and UI layered on top — not alternatives.

**11. D** — Helicone.

**12. B** — `from langfuse.openai import openai`, `@observe()`, and `start_as_current_observation`.

**13. Evals tell you *that* something failed; observability tells you *where and why*.** An eval scores the **output** of a run against a criterion, producing a verdict — pass/fail, or a judge's score — but treats the system as a black box, so a failure tells you nothing about which step caused it. Observability records the **process**: the tree of spans, each with its latency, token usage, inputs, and outputs, so you can see that step 4's retrieval returned irrelevant chunks and that the model's answer followed from them. The three motivating questions make this concrete — "why did this take 9 seconds," "why did it cost 11k tokens," "why did step 4 return garbage" — none of which are answerable from the final output alone, and all of which are trivially answerable from a trace. They are complements: evals detect, observability diagnoses.

**14.** An agent run is a **tree**: an agent calls a tool, which calls a retriever, which calls an embedding model, and each of those may branch further. `print()` produces a **flat list of lines**, which preserves order but not nesting — so you can see that an embedding call happened without knowing which retrieval requested it or which agent step that retrieval served. **Concurrency makes it worse:** with async or parallel steps, output from independent branches interleaves arbitrarily, so even the ordering becomes misleading, and two simultaneous tool calls produce lines you cannot attribute. **What is lost is causality.** You retain a record of everything that happened and lose the ability to say what caused what — which is precisely the question you are debugging. Hence the requirement for telemetry that is structured (machine-readable), causal (parent → child), and queryable (filterable across runs).

**15.** Attributes are key/value metadata attached to a span — user, session, cost, latency, tags — and they can only be recorded by the code that creates the span, at the moment it runs. **Filtering later depends entirely on what was captured then**, because a trace is an immutable record of a past execution; there is no way to retroactively decorate historical traces with information that was never written. **Concrete consequence:** suppose you instrument without a `user_id` attribute. A month later a customer reports intermittent failures, and you want to inspect every trace from that user — but you cannot, because nothing distinguishes their traces from anyone else's. You can neither isolate their sessions, nor compute their cost, nor check whether the problem correlates with their account tier. Recovering the capability requires adding the attribute and waiting a month for new data. The practical rule is to over-tag at instrumentation time, since attributes are cheap to add and impossible to backfill.

**16.** OpenTelemetry is a **vendor-neutral standard** defining the vocabulary (trace, span, attribute) and the wire format (OTLP), and **Jaeger, Tempo, Datadog, and Langfuse all speak it**. Because your application emits OTel spans rather than vendor-specific calls, switching backends means changing an **exporter endpoint**, not rewriting instrumentation. **Why OTel is the durable thing to learn:** the tooling layer moves quickly — Langfuse, LangSmith, Phoenix, and Helicone are all recent, and the LLM observability market is unsettled — but the underlying concepts are a CNCF standard with years of adoption across the broader software industry. Learning "what a span is and how parent-child encodes causality" transfers to any backend and to non-LLM systems, whereas learning one vendor's SDK does not. It is the same principle Week 6 drew about architectures and Week 20 about papers: learn the mechanism, since the surface will be replaced.

**17.** Generic APM was designed for **microservices**, where a span means "an HTTP call to service X that took 40 ms and returned 200." Its data model has no concept of the two things that matter for LLM work. **First, tokens:** it cannot record that a span consumed 8,000 input tokens and produced 400 output tokens, and it has **no price tables**, so it cannot convert usage into cost — yet token spend is the dominant cost driver and the answer to "why did this cost 11k tokens." **Second, prompt ↔ completion pairing:** it has no notion that a span has a text input and a text output that belong together and must be rendered side by side for a human to read. Debugging an LLM failure almost always means reading the exact prompt sent and the exact completion returned; a generic span viewer shows JSON blobs and hex parent IDs. Raw OTel is genuinely structured and causal — it just is not **readable for LLM work** — which is why an LLM-native layer must exist on top rather than replacing it.

**18.** It reveals that Langfuse is **not a separate system** but the very OTel pipeline taught in Part 2, pointed at a different backend. `LangfuseSpanProcessor` implements the standard OpenTelemetry span-processor interface — the component that batches spans and hands them to an exporter — so the same instrumentation, the same span tree, and the same attributes flow through unchanged, and only the destination differs. **Why the staging matters pedagogically:** the lecture teaches the vendor-neutral architecture first and then shows the product slotting into it, so the reveal lands as *"Part 2 was the architecture, not an analogy."* **Practically**, it means adopting Langfuse is not a lock-in decision: you can emit to Langfuse and Jaeger simultaneously, or replace the processor later, without touching application code. It is the concrete demonstration of "providers are OTel backends, not alternatives to OTel."

**19.** Week 17's evals operate on a **static set** — a curated collection of tasks with known inputs, built once by a domain expert through critique shadowing, and run repeatedly against changes. Scoring live traffic instead attaches scores — from an LLM judge, a heuristic, or human review — **to production traces as they happen**. **Why this is an upgrade:** a static set is a fixed *sample* of the world, chosen in advance by someone guessing at what users will do, so it inevitably misses real distribution shift, novel query shapes, and the long tail; production traffic **is** the world, so the evaluation distribution matches the actual distribution by construction. It also **grows automatically** rather than requiring manual curation, and because scores attach to traces you can filter to "traces scored bad last week" and inspect the full span tree of each failure — connecting the verdict to its cause. Most importantly it makes **regression visible before users hit it**, since a drop in scores across live traffic surfaces immediately. It closes Week 17's flywheel with production supplying the failures, and the scored traces themselves become new eval cases.

**20.** **Start from the report and find the traces.** "This morning" and "the assistant" are enough to filter, provided you instrumented attributes — filter traces by `user_id` and a time window. This is the moment the Week 17 lesson about attributes pays off; without `user_id` you are reduced to guessing, since nothing distinguishes this user's traces. **Diagnose the latency from the tree.** Open the trace and read per-span durations. Latency in an agent run is almost always concentrated, and the tree tells you where: a slow model call (check whether the prompt was unusually long, since prefill scales with input — Week 4), a slow retrieval, a tool that timed out and retried, or — commonly — a doom loop where the agent repeated the same step several times, which is visible immediately as repeated sibling spans. **Diagnose the wrong answer from the inputs, not the output.** Walk the tree to the generation span and read the **prompt** alongside the **completion**. The usual finding is that the model answered reasonably given bad inputs: check the retrieval span's returned chunks, since Week 9's lesson applies — inspect the retrieved chunks, not the final answer, because the model papers over bad retrieval. If retrieval was fine, look at whether context was truncated or compacted (Week 17) and whether the relevant fact survived. **Check the token usage** on each span, which frequently explains both symptoms at once: a bloated context is slow *and* degrades quality through context rot (Week 10), so 11k tokens where 2k were expected is a single root cause for both complaints. **Close the loop.** Attach a score to the trace marking it a failure, then look for other traces sharing the same signature — same tool, same retrieval pattern, same length profile — to distinguish a one-off from a systematic regression. Cluster the failures into a mode and turn that into a harness change, which is exactly Week 17's flywheel with production supplying the failures.

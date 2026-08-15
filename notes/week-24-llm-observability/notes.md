# Week 24 — LLM Observability

**Subtitle:** Seeing inside a running system
**Date:** 01/07/2026
**Sources:** `downloads/week-24-llm-observability.pdf` (31 slides, 39-slide deck) · `downloads/week-24-llm-observability.ipynb` (33 cells)
**Notion page:** https://100xschool.notion.site/397ffffa33e58038a07ccd5afeada74d

---

## Where we are

> - **You can build the system.**
> - **You can score the output (evals).**
> - **You cannot see what happened inside a run.**

Week 17 gave you evals — a verdict on the *output*. This week is about the *process*: evals tell you **that** something failed, observability tells you **where and why**.

### Three questions

> - **Why did this take 9 seconds?**
> - **Why did it cost 11k tokens?**
> - **Why did step 4 return garbage?**

Each is unanswerable from the final output alone. All three are trivially answerable from a trace.

---

## Part 1 — The problem

### `print()` breaks down

- **Nested calls**
- **Async / concurrent**
- **Multi-step agent loops**
- **Flat logs lose the tree**

The core failure: an agent run is a **tree** — an agent calls a tool which calls a retriever which calls an embedding model — but `print()` emits a **flat list**. With concurrency the lines interleave, and the parent-child structure is gone. You can see everything that happened and still be unable to say what caused what.

### What we actually need

| Property | Why |
|---|---|
| **Structured** | Machine-readable, not free text |
| **Causal** | **parent → child** — the tree preserved |
| **Queryable** | Filter and aggregate across runs |

---

## Part 2 — OpenTelemetry

> **The vocabulary that outlives the vendor.**

| Concept | Definition |
|---|---|
| **Trace** | **One request's full lifecycle** |
| **Span** | **One operation inside a trace** — LLM call · retrieval · tool · custom logic |
| **The tree** | **Parent → child = causality** |
| **Attributes** | Key/value metadata on a span — user · session · cost · latency · tags |

> **Filtering later depends on this.**

That's the practical warning about attributes: they're cheap to add at instrumentation time and impossible to add retroactively. If you don't tag `user_id` now, you cannot ask "what did this customer experience?" later — the data simply isn't there.

### The pipeline

> **OpenTelemetry is the wire format.**

**Why this matters:**
- **Jaeger · Tempo · Datadog · Langfuse**
- **All speak OTLP**
- **Learn the concept, swap the tool**

This is the durable takeaway. OTel is a CNCF standard, so instrumenting against it means you're not locked to a vendor — you change an exporter endpoint, not your application code.

### The LLM-specific piece

**Generation** — a span with first-class fields:

| Field | Meaning |
|---|---|
| **model** | Which model ran |
| **prompt / completion** | The text in and out |
| **token usage** | **The cost driver** |

---

## Part 3 — What raw OTel can't do

After running raw OTel:

| Property | Verdict |
|---|---|
| Structured? | **yes** |
| Causal? | **yes** |
| **Readable for LLM work?** | **no** — JSON blobs, hex parent IDs |

### The gaps

- **No cost** — there are no price tables
- **No prompt / completion rendering**
- **No place for a score**
- **No sessions, users, or UI**

### So why doesn't APM just work?

- **Datadog / Jaeger — generic spans**
- **No token concept**
- **No prompt ↔ completion pairing**
- **→ an LLM-native layer must exist**

The argument is precise. Generic APM was built for microservices, where a span is "an HTTP call that took 40 ms." It has no notion that a span consumed 8,000 input tokens at $3/M and produced 400 output tokens at $15/M, and no way to render a prompt next to its completion so a human can read the exchange. Those aren't cosmetic gaps — **cost and prompt/completion inspection are the two things you actually look at.**

---

## Part 4 — Why providers exist

> ### **Not alternatives to OTel — backends for it.**
> Providers are **OTel backends** with **LLM-aware storage & UI** on top.

### The landscape

| Provider | Character |
|---|---|
| **Langfuse** | **OSS · OTel-native · self-hostable** |
| **LangSmith** | **Tightest LangChain fit** |
| **Phoenix (Arize)** | **Eval / experiment focus** |
| **Helicone** | **Proxy · near-zero setup** |

Note the different integration models: Langfuse and Phoenix are SDK-based, LangSmith is coupled to LangChain, and **Helicone is a proxy** — you change your `base_url` and it observes traffic passing through, requiring no code changes but seeing only what crosses the wire.

### Why Langfuse here

- **Open source**
- **OTel-native — `LangfuseSpanProcessor`**
- **Broad framework support**
- **The concepts transfer to all of them**

---

## Part 5 — Langfuse, live

### The reveal

> **`LangfuseSpanProcessor` = the span processor from Part 2.**
>
> **Part 2 was the architecture, not an analogy.**

Nicely staged: the lecture teaches OTel generically, then shows that Langfuse *is literally that pipeline* with a different exporter. Not an alternative system — the same one, pointed elsewhere.

### Three ways to instrument

| # | Method | Code |
|---|---|---|
| **1** | **Drop-in** | `from langfuse.openai import openai` |
| **2** | **Decorator** | `@observe()` |
| **3** | **Context manager** | `start_as_current_observation` |

Increasing control for increasing effort: the drop-in requires changing one import and captures every LLM call automatically; `@observe()` wraps your own functions so business logic appears as spans; the context manager gives fine-grained control over span boundaries and attributes. **Use all three together** — drop-in for model calls, decorators for functions, context managers where you need precision.

### The trace UI

- **Tree view**
- **Per-span latency + cost**
- **Filter by user / session / tag**

This answers the three opening questions directly: the tree shows *where* nine seconds went, per-span cost shows *what* burned 11k tokens, and the prompt/completion rendering shows *why* step 4 returned garbage.

---

## Part 6 — Closing the loop

### Evals, again

| Week 17 | Now |
|---|---|
| **A static set** | **Scores on live traffic** |

The upgrade that matters. A static eval set is a fixed sample of the world; production traffic is the world. Scoring live traces means your evaluation distribution matches your actual distribution, and it grows automatically.

### Score the trace

```
LLM-as-judge  → attach to trace
Heuristic     → attach to trace
Human review  → attach to trace
```

All three attach to the same object. A trace carries its own quality signal, so you can filter to "traces scored bad last week" and inspect exactly what happened inside each one.

### Why observability matters

> ### **Regression becomes visible — before users hit it.**

This closes Week 17's flywheel: evals → failure clusters → harness changes → better agent. Observability supplies the failures **continuously from production** rather than from a hand-built set, and the scored traces *are* the new eval cases.

---

## Takeaways

> - **Trace · span · generation.**
> - **OTel is the substrate.**
> - **Observability + evals = a feedback loop.**

---

## Key takeaways

1. **Evals tell you *that* something failed; observability tells you *where and why*.**
2. **`print()` loses the tree** — agent runs are nested and concurrent, flat logs are neither.
3. **You need structured, causal, queryable data.** Parent → child *is* causality.
4. **Trace = one request's lifecycle; span = one operation inside it.**
5. **Attributes must be added at instrumentation time** — you cannot filter by a tag you never recorded.
6. **OTel is the wire format**, and Jaeger, Tempo, Datadog, and Langfuse all speak OTLP. **Learn the concept, swap the tool.**
7. **A "generation" is a span with model, prompt/completion, and token usage** as first-class fields.
8. **Raw OTel is structured and causal but unreadable for LLM work** — no cost, no prompt rendering, no scores, no UI.
9. **Generic APM fails because it has no token concept and no prompt ↔ completion pairing.**
10. **Providers are OTel backends, not alternatives** — `LangfuseSpanProcessor` is the span processor from Part 2.
11. **Three instrumentation styles** — drop-in import, `@observe()` decorator, context manager — used together.
12. **Scoring live traces beats a static eval set**, because the evaluation distribution matches production.
13. **Observability + evals = the feedback loop** that makes regressions visible before users hit them.

---

## Glossary

| Term | Meaning |
|---|---|
| **Observability** | Ability to see what happened inside a running system. |
| **Trace** | One request's full lifecycle. |
| **Span** | One operation inside a trace — LLM call, retrieval, tool, or custom logic. |
| **Parent → child** | The span relationship encoding causality. |
| **Attributes** | Key/value metadata on a span (user, session, cost, latency, tags). |
| **Generation** | An LLM-specific span with model, prompt/completion, and token usage. |
| **OpenTelemetry (OTel)** | Vendor-neutral standard for traces, metrics, and logs. |
| **OTLP** | OpenTelemetry Protocol — the wire format all backends speak. |
| **Span processor** | Component batching and exporting spans to a backend. |
| **Exporter** | Sends spans to a specific destination. |
| **APM** | Application Performance Monitoring — generic tracing built for microservices. |
| **Langfuse** | Open-source, OTel-native, self-hostable LLM observability backend. |
| **LangSmith / Phoenix / Helicone** | LangChain-tight / eval-focused / proxy-based alternatives. |
| **`LangfuseSpanProcessor`** | Langfuse's OTel span processor. |
| **`@observe()`** | Decorator instrumenting a function as a span. |
| **Score** | A quality signal attached to a trace — from a judge, heuristic, or human. |
| **LLM-as-judge** | Using a model to grade outputs (Week 17). |
| **Session / user tagging** | Attributes enabling per-user and per-conversation filtering. |
| **Regression** | A quality drop detectable from scored traces before users report it. |

# S2 — Quiz (20 questions)

**Topic:** Prompt Engineering, Systematically
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The correct mental model for a prompt is:
- A) An instruction executed by a program
- B) A conditioning signal that shifts a probability distribution
- C) A configuration file
- D) A compiled query

**2.** `CRITICAL: You MUST use this tool` on a current model tends to cause:
- A) More reliable tool use
- B) Over-triggering and a hedging output tone
- C) A parse error
- D) No effect at all

**3.** Which is a *hedge* that current models read literally as permission to skip?
- A) "Include a summary."
- B) "Try to include a summary if possible."
- C) "You are a support agent."
- D) "Answer from the knowledge base."

**4.** Prompt caching works by:
- A) Storing responses keyed by question
- B) Prefix match — any earlier byte change invalidates the rest
- C) Compressing the system prompt
- D) Caching per user session only

**5.** The prompt renders in which order?
- A) messages → system → tools
- B) tools → system → messages
- C) system → tools → messages
- D) Order is arbitrary

**6.** Putting `datetime.now()` in the system prompt:
- A) Improves temporal reasoning at no cost
- B) Invalidates the cache on every request
- C) Is required for date-aware answers
- D) Only affects the first request

**7.** "Think step by step" is largely obsolete because:
- A) It never worked
- B) Reasoning depth is now an API parameter on thinking models
- C) It violates content policy
- D) It breaks structured outputs

**8.** The modern replacement for "Output ONLY valid JSON" plus a retry loop is:
- A) A stop sequence
- B) Structured outputs with a JSON Schema
- C) Assistant prefill
- D) A lower temperature

**9.** Negative instructions like "never mention pricing" are discouraged because:
- A) They're too long
- B) They put the prohibited concept into context where attention can reach it
- C) Models cannot parse negation
- D) They break caching

**10.** Which is the best defence against prompt injection from a retrieved chunk?
- A) Asking the model to ignore instructions in the data
- B) Delimiting the untrusted content and keeping authorisation in your code
- C) Lowering temperature
- D) Using a longer system prompt

**11.** Few-shot examples are described as dangerous when:
- A) There are more than two
- B) They're stale — written for an older model
- C) They're labelled as illustrative
- D) They cover edge cases

**12.** "A prompt is a per-model artifact" implies:
- A) Prompts should be versioned in git
- B) You must re-baseline prompts when you change models
- C) Each model needs a different schema
- D) Prompts cannot be shared across teams

---

## Short answer

**13.** Explain the "conditioning a distribution" model and give three consequences that follow from it.

**14.** Explain both failure directions of emphasis, and why marking many things critical fails.

**15.** Explain the prefix-match invariant and list four silent cache invalidators.

**16.** Explain why the "output only JSON" stack is obsolete, naming everything that gets deleted with it.

**17.** Explain why negative instructions can backfire, and give the recommended alternative.

**18.** Explain why a prompt is a per-model artifact, and what maintenance practice follows.

**19.** Explain the difference between context and length, and why too-short prompts produce generic output.

**20.** Rewrite this prompt and justify every change:
```
You are a helpful AI assistant.
CRITICAL: You MUST think step by step.
IMPORTANT: Always use the search tool. If unsure, search.
Do not hallucinate. Do not make up facts. Be accurate.
Try to be brief if you can.
Output ONLY JSON: {"answer": "...", "confidence": 0-1}
No markdown, no preamble.
```

---
---

## Answer key

**1. B** — The model computes `p(next token | everything so far)`, and the prompt is "everything so far."

**2. B** — Current models follow the system prompt closely, so emphasis written to overcome older models' reluctance now over-applies; the anxious register also propagates into the output tone.

**3. B** — "Try to… if possible" attached to an actual requirement is read literally as optional.

**4. B** — Prefix match; any byte change anywhere in the prefix invalidates everything after it.

**5. B** — tools → system → messages, which is why a marker on the last system block covers tools and system together.

**6. B** — It changes the prefix bytes on every request, so nothing after it can ever be cached.

**7. B** — Thinking is a trained behaviour on reasoning models and exposed as an adaptive-thinking plus effort parameter; the incantation is redundant.

**8. B** — Structured outputs constrain the response to a supplied schema, removing parse failures entirely.

**9. B** — Stating the prohibition puts the concept into context, where attention can reach it; describing success is safer.

**10. B** — Delimit the untrusted content explicitly and keep authorisation in code. Asking the model to ignore embedded instructions is not reliable.

**11. B** — Examples written for an older model freeze that model's length, tone, and structure into the new one.

**12. B** — Instructions that were load-bearing on one generation become dead weight or actively harmful on the next, so prompts must be re-baselined at each model change.

**13.** The model computes `p(next token | everything so far)`, and your prompt **is** that conditioning context — so prompting shifts the distribution toward the region where good outputs are likely, rather than issuing a command to a program. **Consequence one: everything in context is signal.** The model has no built-in distinction between "instructions" and "background material"; attention treats it all as tokens, so an offhand aside gets applied where it does not belong. **Consequence two: style is contagious.** A terse prompt yields terse output, a bulleted prompt yields bulleted output, and a prompt full of anxious all-caps warnings yields cautious, hedging prose — the register of the prompt becomes the register of the response. **Consequence three: examples dominate.** Concrete examples pin length, tone, and structure more strongly than any adjective, which is why few-shot works so well and simultaneously why stale examples are so damaging.

**14.** **Inflated emphasis** — `CRITICAL:`, `MUST`, `NEVER`, "if in doubt, use X" — was written for older, less steerable models that genuinely needed forcing. Current models follow the system prompt closely, so the same text now **over**-applies: tools fire when they shouldn't, behaviour becomes rigid in cases that call for judgment, and the prompt's anxious register bleeds into a hedging output tone. **Leftover hedges** fail in the opposite direction: "try to", "if possible", and "ideally" attached to genuine requirements are now read **literally** as permission to skip, so a required summary becomes optional. **Why marking many things critical fails:** emphasis is a *relative* signal. If five instructions are each marked `IMPORTANT`, none of them stands out and the marker carries no information — you have raised the volume of everything, which is the same as raising the volume of nothing, while adding tokens and an anxious tone. Emphasis is properly a tested, scoped fix for one demonstrably underweighted instruction, not a default writing register.

**15.** **The invariant:** prompt caching is a **prefix match** — the cache key derives from the exact bytes up to each cache breakpoint, so a single byte difference at position N invalidates the cache for everything at or after N. The prompt renders as tools → system → messages, so anything volatile placed early poisons everything downstream. **Four invalidators:** (i) `datetime.now()` or any timestamp in the system prompt — the prefix differs on every request; (ii) a UUID or per-request ID interpolated early in the content — same effect; (iii) serialising a dict without sorting keys, so the byte order varies between runs even when the content is identical; (iv) building the tool list per user, which changes position zero and prevents anything from caching at all. (Also valid: a user ID in the system prompt, which makes the prefix per-user and unshareable; or conditional system sections, where every flag combination becomes a distinct prefix.) **How to detect it:** the API reports cache read and write token counts — if reads stay at zero across repeated identical-prefix calls, one of these is present.

**16.** The old stack existed because there was no way to *guarantee* a response shape: you asked for JSON in prose ("Output ONLY valid JSON, no preamble"), then parsed it, and when parsing failed you retried, added stop sequences, and stripped markdown fences the model added anyway. **Structured outputs replace the guarantee, not just the instruction** — you pass a JSON Schema and the response is constrained to conform, so parse failure stops being a possible outcome. **Everything in the surrounding stack goes with it:** the "output only JSON" instruction, the "no preamble / no markdown fences" instruction, the stop sequences guarding the JSON, the regex or fence-stripping extraction step, the try/except parse block, and the retry-on-parse-failure loop — along with any tests asserting the old behaviour. This is why the audit is worth doing at the *request-builder* level rather than only in the prompt string: most of the dead code lives outside the prompt. The related fossil is **assistant prefill** (starting the model's turn with `{` to force a shape), which now errors on current Claude models — finding it is a migration task, not a style preference.

**17.** Stating a prohibition **puts the prohibited concept into the context window**, where attention can reach it — "never mention pricing" makes pricing a live token in the model's working set, and "never use the word 'delve'" makes 'delve' present. The instruction is understood, but the concept has been made salient, which is the opposite of the intent. A related problem is that a list of prohibitions describes the shape of failure without ever describing the shape of success, leaving the model to infer what you actually want from the negative space. **The alternative is to describe success positively:** instead of "don't mention pricing," write "discuss features and support options"; instead of a list of banned phrasings, give a positive example of the desired voice. **Prohibitions still earn their place** when they encode a real constraint — a compliance rule, a promise the business must not make, a genuinely destructive action — and those should carry their reason, because a rule with a reason generalises to cases you didn't anticipate while a bare rule does not.

**18.** Prompts are written **against a specific model's behaviour**, and much of their content consists of workarounds for that model's weaknesses: emphasis added because it under-triggered, step-by-step scaffolding added because it planned poorly, format-forcing because there was no schema support, verbosity caps tuned against how much it padded. **When the model changes, the weaknesses change** — and a workaround for a weakness that no longer exists does not become neutral, it becomes actively harmful. Emphasis that was necessary now over-triggers; scaffolding that helped now over-constrains; a verbosity cap tuned for a padded model now starves reasoning. The direction also runs the other way: a new model may have *new* failure modes that need guidance the old prompt never contained. **The maintenance practice that follows** is to treat every model migration as a prompt audit: re-baseline the prompt against the new model with old scaffolding removed, test the removals rather than only testing additions, and periodically delete lines nobody can justify — because prompts rot by accretion, with each incident adding a line and nobody ever removing one.

**19.** **Context** is information only you have — who the audience is, what the product does, what the quality bar is, what environment the output lands in, and *why* each constraint exists. **Length** is bytes. They are commonly confused because good prompts are often long, but the causation runs the wrong way: prompts are long *because* real context takes words, not because length itself helps. A long prompt made of generic virtues — "be accurate, be thorough, be helpful, be clear" — restates trained defaults, adds tokens, dilutes the instructions that matter, and inflates thinking spend. **Why too-short prompts produce generic output:** the model must produce *something* for every unspecified dimension — tone, length, depth, audience assumptions — and in the absence of information it fills those gaps with the safest, most average choice available. Generic input specifies nothing, so you get the distribution's centre of mass. The asymmetry is what makes the advice actionable: real context is essentially never wasted, while padding always costs something — so give more context than feels necessary, and cut anything the model could already know.

**20.** **The rewrite:**
```
You are a support agent for [Product], helping [audience] with [domain].
Users typically contact us when [common situation].

Answer from the knowledge base. Call search when the answer depends on
product specifics — versions, limits, pricing, configuration — rather than
general knowledge.

If the knowledge base doesn't cover it, say so and offer to escalate.
A confident wrong answer costs us more than an escalation.

Keep answers to what the question needs.
```
...with the JSON shape moved to a **structured output schema**.

**Justifying each change.** *"You are a helpful AI assistant"* → replaced with real context: a one-line role is fine, but here it was the *only* context, so it was substituting for the audience, product, and domain the model cannot infer. *"CRITICAL: You MUST think step by step"* → **deleted**; reasoning depth is now an adaptive-thinking plus effort parameter, so the incantation is redundant, and the `CRITICAL:` marker over-triggers. *"Always use the search tool. If unsure, search"* → made **conditional** on the kind of question; the unconditional form plus "if unsure" causes the model to search when it doesn't need to. *"Do not hallucinate. Do not make up facts. Be accurate"* → replaced with an **escape hatch that carries its reason**; the originals are negative instructions restating a trained default, whereas "say so and offer to escalate, because a confident wrong answer costs more" tells the model what to *do* and generalises to situations the rule didn't anticipate. *"Try to be brief if you can"* → *"Keep answers to what the question needs"*; the hedge made a real requirement optional, and the replacement gives a calibration rule rather than an unenforceable adjective. *The JSON block* → moved to a **schema**, which deletes the "output ONLY JSON" instruction, the "no markdown, no preamble" instruction, and — outside the prompt — the parse/retry/fence-stripping code that existed to serve it. **Also worth noting:** the `confidence: 0-1` field should be treated with suspicion; self-reported confidence is poorly calibrated, and if it drives a downstream decision it needs validating against outcomes (see S3).

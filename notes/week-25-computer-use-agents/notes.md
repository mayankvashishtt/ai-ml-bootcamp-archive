# Week 25 — Computer Use Agents

**Subtitle:** screenshot → decide → act → repeat · **Pixels in → actions out**
**Date:** 08/07/2026 — **the final lecture**
**Sources:** `downloads/week-25-computer-use-agents.pdf` (21 slides, 23-slide deck) · `downloads/week-25-computer-use-agents.ipynb` (29 cells)
**Notion page:** https://100xschool.notion.site/397ffffa33e5807fa564ee50ed97b074

---

## The task

> **"File an expense: United Airlines, $450, Travel. Submit."**
>
> - **Trivial for you.**
> - **Six steps for a model.**
> - **Today: build the thing that does it.**

### What the model sees

- **One image. 1024 × 768.**
- **No DOM. No element IDs. No API.**
- **Pixels in → actions out.**

That constraint is the whole subject. Every other agent in this course received *text* — tool results, retrieved chunks, file contents. This one receives **an image** and must produce **coordinates**.

### The universal pattern

```
screenshot → understand → decide → act → repeat
```

Structurally identical to Week 8's ReAct loop — OBSERVATION is a screenshot, ACTION is a click. What changes is that the observation is now **unstructured pixels**.

---

## Build it by hand (Stage 2)

- **A general VLM over OpenRouter**
- **We invent the action JSON**
- **We parse it. We execute it.**

**The result:**

> ### **It misses. Loop: correct. Grounding: wrong.**

A precise diagnosis, and the pivot of the lecture. The *architecture* was right — the loop ran, the JSON parsed, the actions executed. What failed was the model's ability to say **where** to click.

---

## Grounding — the through-line

> **Referring expression → precise location, well enough to act.**
>
> - **"the Submit button" → (743, 373)**
> - **Perception ≠ grounding**
> - **Errors compound across steps**

**"Perception ≠ grounding"** is the key distinction. A VLM can describe a screenshot fluently — "there's a blue Submit button at the bottom right" — while being unable to produce accurate pixel coordinates for it. Understanding *what* is on screen and knowing *where* it is are separate capabilities, and only the second lets you act.

**"Errors compound"** is the practical consequence: a six-step task with 90% per-step grounding accuracy succeeds only ~53% of the time. Long-horizon GUI work demands per-step accuracy far above what feels "good enough."

### The lineage

| Year | Development |
|---|---|
| **2016** | **Universe** — too early |
| **2023** | **GPT-4V-Act · Set-of-Mark** |
| **2024** | **UFO · OmniParser** |
| **2024** | **Claude computer use** |
| **2025** | **Operator · UI-TARS** |

### Set-of-Mark

> **Pixel regression → multiple choice. The detector is the grounding.**

The trick: run a detector over the screenshot, draw numbered boxes on every interactive element, and ask the model to pick a **number** instead of coordinates. This converts an open-ended regression problem (predict x, y from a continuous plane) into a **classification** problem (choose element 7 of 23) — which VLMs are far better at. The accuracy then comes from the detector, not the model.

### Where the detector lives

| | |
|---|---|
| **Out of model** | **OmniParser · the a11y tree** |
| **Into model** | **UI-TARS · Claude computer use** |

> **The frontier went "into the model."**

Same arc as Weeks 9–11: an external scaffold compensating for a model weakness, eventually absorbed into the model once training closed the gap. Week 17's "scaffolding goes stale — delete code," demonstrated on a two-year timescale.

---

## The native tool (Stage 4)

- **`computer_20251124`**
- **Grounding trained in**
- **Fixed action space — screenshot / click / type / scroll / zoom**
- **A documented agent loop**

> **Same browser. Same task. Only difference: grounding moved into the model — plus self-verification.**

### The vendor's own guidance

| Guidance | Why |
|---|---|
| **Text before screenshot → better clicks** | Let the model reason in text first; the plan improves the grounding |
| **`zoom` for small targets** | Resolution is the binding constraint — zooming buys effective pixels |
| **A self-check prompt cuts "assumed it worked"** | The characteristic failure: the model believes its click landed and proceeds on a false premise |
| **Isolate — treat prompt injection as a given** | A web page is untrusted input, and this agent *reads and acts on* whatever is on screen |

That last point deserves emphasis. A computer-use agent reads the screen and acts — so any text on any page it visits is effectively an instruction channel. **Prompt injection isn't a risk to mitigate here; it's an assumption to design around.** Hence isolation and least privilege rather than filtering.

---

## Benchmarks

**OSWorld: 12% → 38% → 61% → 72%+ in about two years.**

### Read the footnote

- **OSWorld vs OSWorld-Verified**
- **`pass@1` vs `pass@10`**
- **Step budget · vendor self-report**

> ### **Skepticism is the skill.**

Week 20's red flags applied to a live number. `pass@1` versus `pass@10` is exactly Week 17's distinction — and for an agent that acts autonomously on a real computer, `pass@10` is close to meaningless, since you cannot un-submit a form nine times. **Step budget** matters too: an agent allowed 100 steps will beat one allowed 15, so the numbers aren't comparable without it.

---

## The open frontier

| Project | Character |
|---|---|
| **UI-TARS 1.5 → 2** | Screenshot-only, RL |
| **UGround** | A learned detector |
| **Qwen3-VL** | Top open model, **~66%** |
| **OmniParser** | The parser |

### Is it even a GUI problem?

- **CoAct-1 — code-as-action (hybrid)**
- **CLI-as-computer-use** (geohot / Rauch)
- **The ceiling: not everything is text**

The sharpest question in the lecture. If a task can be done with a shell command or an API call, **doing it through the GUI is absurd** — slow, brittle, and dependent on grounding that might miss. This is Week 11's Claude Code argument again: the CLI is a better interface for a model than a GUI, because it's text.

But **not everything is text.** Legacy desktop software, proprietary applications, and anything without an API leave the GUI as the only surface.

### Cua-Bench / KiCad

- **25 expert EDA tasks**
- **Best frontier agent: 6 / 25**
- **Fails: budget + hallucinated screen state**
- **"Stage 3 — alive at the frontier"**

A useful corrective to the 72% OSWorld figure. On genuinely expert software the best agent manages **under a quarter** of tasks, and the failure modes are the same ones the hand-rolled Stage 2 build hit — running out of steps, and **hallucinating what's on screen**. Progress is real; it is not solved.

### The synthesis

> ### **Code when efficient. Vision when you must. Know which is which.**

---

## Building for real

- **Isolated VM, least privilege**
- **Human-in-the-loop for consequential actions**
- **Log every action + screenshot**
- **Cua · a reference impl · browser-use**

Three of the four are things this course has already argued for: sandboxing (Week 8's tool safety), HITL (Week 21's `interrupt()`), and logging every step (Week 24's observability — here with screenshots, since the observation *is* an image).

---

## One frame

| | **HAND-ROLLED** | **NATIVE TOOL** |
|---|---|---|
| **Grounding** | **misses** | **lands** |
| **Actions** | **invented JSON** | **tool schema** |
| **Loop** | **improvised** | **contract** |
| **Verify** | **none** | **self-check** |

> ### **The API is shaped by the pain of Stage 3.**

The best closing line of the course. Every field in `computer_20251124` — the fixed action space, the `zoom` action, the self-verification step — exists because someone hit exactly the failure you hit building it by hand. **Build it badly first, and the API stops looking arbitrary.** That's the pedagogical argument for the whole course: Week 2's NumPy backprop before PyTorch, Week 8's raw ReAct loop before LangGraph, and now a hand-rolled computer-use agent before the native tool.

---

## Key takeaways

1. **Pixels in → actions out.** No DOM, no element IDs, no API — one 1024×768 image.
2. **The loop is easy; grounding is hard.** The hand-rolled build had a correct loop and wrong coordinates.
3. **Perception ≠ grounding** — describing a screen and locating things on it are different capabilities.
4. **Errors compound across steps** — 90% per-step accuracy is ~53% over six steps.
5. **Set-of-Mark turns pixel regression into multiple choice**, moving accuracy into the detector.
6. **The frontier moved the detector "into the model"** — external scaffolding absorbed once training caught up.
7. **The native tool's value is trained-in grounding plus self-verification**, not a better loop.
8. **Text before screenshot improves clicks**; `zoom` for small targets.
9. **Treat prompt injection as a given** — a computer-use agent reads and acts on untrusted screen content.
10. **OSWorld 12% → 72%+ in two years, but read the footnote**: verified split, `pass@1` vs `pass@10`, step budget, vendor self-report.
11. **On expert software (KiCad), the best agent scores 6/25** — failures are step budget and hallucinated screen state.
12. **Code when efficient. Vision when you must.** GUI automation is a last resort, not a default.
13. **The API is shaped by the pain of building it by hand.**

---

## Glossary

| Term | Meaning |
|---|---|
| **Computer use agent** | An agent operating a computer via screenshots and synthetic input. |
| **Grounding** | Mapping a referring expression ("the Submit button") to a precise location. |
| **Referring expression** | A natural-language description of an on-screen element. |
| **Perception vs grounding** | Describing what is on screen vs knowing where it is. |
| **Error compounding** | Multiplication of per-step accuracy across a multi-step task. |
| **Set-of-Mark** | Overlaying numbered boxes so the model selects an index instead of coordinates. |
| **Detector** | The component identifying interactive elements; may be external or trained in. |
| **a11y tree** | The accessibility tree — a structured view of UI elements. |
| **OmniParser** | An external screen parser providing element detection. |
| **UI-TARS** | Screenshot-only agent model trained with RL, with grounding built in. |
| **`computer_20251124`** | The native computer-use tool with a fixed action space. |
| **Action space** | The allowed actions: screenshot / click / type / scroll / zoom. |
| **Self-verification** | A model check that an action produced the expected result. |
| **OSWorld / OSWorld-Verified** | Computer-use benchmark and its cleaned variant. |
| **`pass@1` / `pass@10`** | Success on the first attempt vs within ten (Week 17). |
| **Step budget** | The cap on actions an agent may take per task. |
| **Prompt injection** | Malicious instructions embedded in content the agent reads. |
| **CoAct-1** | Hybrid agent using code-as-action alongside GUI control. |
| **CLI-as-computer-use** | The argument that a shell is a better model interface than a GUI. |
| **Cua-Bench / KiCad** | Expert EDA benchmark; best frontier agent scores 6/25. |
| **Hallucinated screen state** | Believing the screen shows something it does not — a top failure mode. |
| **browser-use / Cua** | Open-source computer-use implementations. |

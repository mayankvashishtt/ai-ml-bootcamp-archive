# Week 0 — AI & ML Orientation

**Date:** 09/01/2026
**Instructor:** Rishabh (Research Engineer, ex-AI Lab London — LLM & LLM-RL; Twitter [@rishabh10x](https://twitter.com/rishabh10x))
**Source:** `downloads/week-00-orientation.pdf` (6 slides)
**Notion page:** https://100xschool.notion.site/2e3ffffa33e5807da538db58eeb3aa52

> **Note on scope:** this is a 6-slide housekeeping session, not a technical lecture. The notes below cover everything on the slides; the real material starts in Week 1. Treat this page as the course contract — what you're signing up for and what you need installed.

---

## 1. What the course is trying to do

Four stated goals, and they're worth reading as a statement of philosophy rather than a syllabus:

1. **Understand what's really going on in this field, and how to adapt.** The emphasis is on adaptation, not memorisation. The field's half-life is short; the course is built around that fact.
2. **Understand how LLMs actually work.** Mechanism, not API surface. This is why Weeks 2–6 grind through neural networks and transformers from scratch before you ever touch a production API.
3. **New trends in AI Engineering.** The applied layer — agents, RAG, fine-tuning, evals, observability.
4. **How to stay up to date when new things ship every day.** This goal is why Week 20 is an entire lecture on *how to read research papers*. It's a skill the course treats as a first-class deliverable.

**The arc this implies:** fundamentals (W1–7) → building with models (W8–11) → adapting models (W12–16) → engineering discipline around them (W17–25).

---

## 2. Prerequisites and setup

| Requirement | Detail |
|---|---|
| **Basic Python** | Assumed, not taught. Functions, loops, classes, list comprehensions, `pip`. |
| **A Colab account** | Have it working *before* Week 2. Every hands-on session ships a Colab notebook — 18 of the 26 weeks have one. |
| **Model exploration** | Go and actually use the models on the market. Form your own opinions on how they differ. |

---

## 3. The stance on mathematics

> "Will try to not keep as much math, if any will be done with intuitions."

This is the single most important line on the slides for setting expectations. The course is **intuition-first**. You will meet the equations that matter — `output = activation(Σ wᵢxᵢ + b)`, MSE, attention — but they arrive as explanations of a mechanism you've already understood in words, not as derivations.

**What this means practically:**
- Don't stall waiting to feel mathematically "ready." You aren't expected to derive backprop.
- **Do** get comfortable reading notation, because the papers in Week 20 won't soften it for you.
- If you want the rigour, add it yourself alongside — the course won't block you, but it won't supply it either.

---

## 4. Tools mentioned

Two free coding agents flagged as worth trying:

- **Opencode** — open-source terminal coding agent.
- **Gemini CLI** — Google's terminal agent, free tier available.

These aren't incidental. Weeks 22 and 25 have you *building* agents; using a mature one first gives you a reference for what good looks like — how it plans, when it calls tools, how it recovers from errors.

---

## 5. Key takeaways

- The course optimises for **mechanism + adaptability** over breadth of tools.
- **Intuition over derivation** — but you still need to be able to *read* maths by Week 20.
- Get **Python and Colab working now**. Week 2 assumes both.
- Start **using agents early** so you have intuition for what you'll later build.
- The instructor's background is **LLM + reinforcement learning**, which explains the unusually heavy RL block (Weeks 15–16, RLVR and RL environments) — that's specialist territory most courses skip entirely.

---

## Glossary

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model — a large neural network trained to predict the next token. |
| **LLM-RL** | Applying reinforcement learning to language models to shape behaviour after pretraining. See Weeks 15–16. |
| **Colab** | Google Colaboratory — hosted Jupyter notebooks with free GPU access. |
| **AI Engineering** | The applied discipline of building products on top of existing models, as opposed to training them from scratch. |

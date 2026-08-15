# Week 16 — RL Environments for LLMs

**Date:** 08/05/2026 *(marked "Offline" in the syllabus)*
**Sources:** `downloads/week-16-rl-environments-for-llms.pdf` (25 slides) · `downloads/week-16-rl-environments-for-llms.ipynb` (45 cells)
**Notion page:** https://100xschool.notion.site/362ffffa33e580a09dd8ca5d257b3c63

**Referenced links:**
- [Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Excalidraw board](https://excalidraw.com/#json=j9gliuL2jj0a0JeNILMTj,F7W5Q_6i93-Tb3xCmzWm7g)

> **The thesis, on slide 1:** *"Environments are synthetic data engines, RL trainers, and eval harnesses — the same artifact, three jobs."*

---

## 0. The idea in plain language

Week 15 said: *reward the model when its answer verifiably checks out.* Fine — but **something has to actually do the checking**, hand the model the problem, catch its answer, run the verifier, and hand back a score.

**That something is an environment**, and this week is about building one.

**Concretely, an environment is three functions:**

```
1. give me a task        →  "What is 17 × 23?"
2. take the model's reply →  "...so the answer is 391."
3. score it              →  parse out 391, compare to 391, reward = 1.0
```

That's genuinely all it is. It's the "gym" the model trains in — the thing that poses problems, watches what happens, and keeps score.

**The insight that makes the week worth its time** is on slide 1, and it's a good one:

> **Environments are synthetic data engines, RL trainers, and eval harnesses — the same artifact, three jobs.**

Think about what those three things need. An **eval harness** needs tasks, a way to run the model, and a scorer. An **RL trainer** needs tasks, a way to run the model, and a scorer. A **synthetic data generator** needs tasks, a way to run the model, and a scorer — keeping the high-scoring outputs as training data.

**They're the same object.** Most teams build all three separately, in three codebases, encoding the same task definition three times — and then they drift apart, so your eval no longer measures what your training optimises. Build it once, use it four ways (add agent-harness experimentation as the fourth).

**Two things to watch for, because they're where people actually get hurt:**

**"Verifiability = unhackability" is the design principle.** The more your scorer *computes* rather than *judges*, the less room the model has to game it. Executing tests is unhackable in a way an LLM judge is not. Every step you take away from computation and toward judgement reopens Goodhart's Law (Week 14).

**The #1 footgun is parsing, not scoring.** §7 covers this and it's the most practically useful thing in the lecture. If your model answers correctly but your regex fails to extract the answer, you score it zero — and you've just taught it that correct answers are bad. **Parse first, score second**, and verify your parser separately, because a broken parser doesn't fail loudly. It silently trains the wrong behaviour.

**And a connection worth holding:** an environment is a **hand-built world model** — you're supplying the rules of a world the model can act in and learn from. S13 explores what happens when models try to *learn* that world model instead, and why building one is hard for exactly the same reasons.

---

## 1. The thesis

Weeks 12–15 covered CPT, SFT, RLHF/DPO, and RLVR. **This week is the thing all of those had in common.**

An environment is the same artifact used for:

1. **Evaluation**
2. **RL training**
3. **Synthetic data generation**
4. **Agent harness experimentation**

> **Build it once. Use it four ways.**

This is the practical insight of the week. Most teams build an eval harness, then separately build a training pipeline, then separately build a data generator — three codebases encoding the same task definition, drifting apart. They're the same object.

---

## 2. Foundations — classical RL → LLM-RL

| Classical RL | LLM-RL |
|---|---|
| **State** | The conversation so far |
| **Action** | Next token (or full response) |
| **Environment** | Whatever sends back the next observation + reward |
| **Reward** | Rubric score on the rollout |
| **Policy** | The LLM itself |

> **Everything in modern LLM RL training is a special case of this loop.**

Note the Week 14 mapping (agent/environment/state/action/policy) is repeated with one addition: the environment is now defined *by what it returns*, which is exactly what you build.

---

## 3. The trinity

> **Environment = Dataset + Rollout + Rubric.** Three pieces.

### 1. Dataset

Task inputs, **usually with ground truth or rubric metadata attached.**

| Dataset | Domain |
|---|---|
| **GSM8K** | Grade-school maths |
| **HumanEval** | Coding problems with tests |
| **τ-bench** | Customer service dialogues |
| **SWE-bench** | Real GitHub issues |
| **BrowseComp** | Web research tasks |

> **The closer the dataset to real work, the more it teaches the model.**

### 2. Rollout

**How the model interacts with the dataset.**

| Type | Description |
|---|---|
| **Single-turn** | One prompt, one answer |
| **Multi-turn** | Conversation, feedback loops |
| **Tool-use** | Model calls APIs, environment returns results |
| **Sandboxed** | Model writes code, environment executes it |

### 3. Rubric

**The scoring function.**

| Type | Description |
|---|---|
| **Verifiable** | Exact match · test passes · regex hit |
| **LLM judge** | Another model grades the output |
| **Hybrid** | Multiple rubrics, weighted |

**⚠️ Reward hacking — a story:**

> Train a model to maximize an LLM judge's helpfulness score. **1,000 steps later, every answer starts with "Great question!"**
>
> **Score went up. Capability didn't.**

Goodhart's Law again (Weeks 14–15), now with a memorable failure. LLM judges are *especially* hackable because they inherit human raters' biases toward flattery, confidence, and length.

---

## 4. Why "verifiability = unhackability"

The unification argument: a verifiable rubric is a **program**, and a program cannot be flattered. There is no phrasing that makes `int(answer) == expected` return True when the answer is wrong. Every other rubric type — LLM judge especially — is a **model** with exploitable failure modes.

> **Verifiable rewards are unhackable. Everything else, you fight.**

(As Week 15 noted, "unhackable" applies to the *scoring* — you can still game a weak verifier with insufficient test coverage.)

---

## 5. Landscape — why now

**Closed labs are racing to build proprietary environments.** The receipt:

| **INTELLECT-3** | |
|---|---|
| Released | **26 November 2025** |
| Architecture | **106B Mixture-of-Experts** |
| Base model | GLM-4.5-Air |
| Compute | **512 NVIDIA H200 GPUs · ~2 months** |
| Stack | **Verifiers + Environments Hub + prime-rl** |

An open model trained on an open environment stack — the existence proof that this tooling works at frontier scale.

### The library — Verifiers

**Prime Intellect, by Will Brown** (the same author as Week 15's GSM8K reward stack).
`github.com/PrimeIntellect-ai/verifiers` · 4.1k★ · v0.1.11

| Abstraction | Purpose |
|---|---|
| **`SingleTurnEnv`** | One prompt, one response |
| **`MultiTurnEnv`** | Iterative dialogue |
| **`ToolEnv`** | Native tool calling |
| **`CodeEnv`** | Code execution |
| **`Rubric`** | Composable scoring |
| **`Parser`** | Structured extraction |

> Used to train INTELLECT-3. **The whole library fits in your head.**

Note that `Rubric` and `Parser` are **separate abstractions** — that separation is the week's central lesson (§7).

### The alternative — OpenEnv

**Meta + Hugging Face**, announced at the PyTorch Conference, **23 October 2025**. `github.com/meta-pytorch/OpenEnv`

| | |
|---|---|
| **API style** | **Gymnasium-shaped**: `step()` · `reset()` · `close()` |
| **Isolation** | **Docker container per environment** |
| **Integrates** | TorchForge · TRL · SkyRL · verl |
| **The bet** | Heavier infra, deeper sandboxing |

> **Verifiers added OpenEnv interop — they're not competing.**

The Gymnasium API is deliberate: it's the standard interface from classical RL, so the ecosystem inherits a decade of convention.

### The commons — the Environments Hub

**A public registry. Pip-for-environments.**

```bash
prime env install primeintellect/math-python
prime env install primeintellect/alphabet-sort
prime env install primeintellect/wiki-search
```

> **Every published env can be evaluated, used as a data engine, or trained on** — the four-uses thesis, operationalised.

### The trainer layer

Environments are **portable**; these consume them:

| Trainer | Purpose | From |
|---|---|---|
| **prime-rl** | Async RL training at scale | Prime Intellect |
| **TRL** | General RL & fine-tuning | Hugging Face |
| **SkyRL** | Distributed RL training | Berkeley |
| **verl** | Production RL framework | Volcano Engine |
| **TorchForge** | RL training | Meta-PyTorch |
| **Tinker** | Hosted LoRA RL (remote GPU) | Thinking Machines |

**The separation matters:** environment (task definition) and trainer (optimization) are decoupled, so you can swap either independently — which is what makes an environment shareable at all.

---

## 6. GRPO in three lines

From Week 15:

```
01  Sample N rollouts for the same prompt.
02  Score them all with the rubric.
03  Update the policy to favor rollouts above the group's average.
```

> **No reward model. No baseline network. Just rollouts and ranks.**

Note how cleanly this maps onto the trinity: **dataset** supplies the prompt, **rollout** generates the N attempts, **rubric** scores them. GRPO is the natural consumer of an environment.

---

## 7. The #1 footgun — parse first, score second

**What happened in the notebook:**

> The model gave a **correct answer** wrapped in markdown fences. **The rubric scored it zero.**
>
> **The bug wasn't the model.** The rubric was doing **two jobs at once: extract and score.**
>
> **Separate them. Always. Conflating them is the #1 footgun in RL training.**

This deserves emphasis because of how it fails: **silently and catastrophically.** A brittle rubric doesn't error — it returns 0, which looks exactly like a model that can't do the task. You then spend days tuning hyperparameters, adding data, or changing models, when the actual bug is a regex that didn't anticipate ```` ``` ```` fences.

It also connects directly to Week 15's zero cascade: rewards of zero across a group mean zero variance, zero advantage, and **no gradient**. A parsing bug doesn't just mis-score — it kills training outright.

**The fix is architectural:** a `Parser` extracts structure from raw output; a `Rubric` scores the extracted structure. Two components, independently testable. You can unit-test the parser against messy real outputs without running the model at all.

---

## 8. What the library buys you

| Built ourselves | Get from Verifiers |
|---|---|
| Loop over examples | **`env.evaluate(client, model, n)`** |
| One reward function | **`Rubric(funcs=[…], weights=[…])`** |
| Code-as-string scoring | **`Parser` classes** |
| No retry / no async | **Async, retries, timeouts** |
| No way to share | **`prime env push` → Hub** |
| No way to train | **Plug into prime-rl** |

> **Same env. Same data. ⅓ the code. Same complexity tax — none.**

The honest framing (as with Unsloth in Week 12): the library isn't doing anything conceptually new. It's the *engineering* — async, retries, timeouts, composable weighted rubrics, shareability — that you'd otherwise rebuild badly.

---

## 9. Multi-turn is where real RL lives

| Single-turn | Multi-turn |
|---|---|
| **Benchmarks** | **Agents, real RL training** |

> **The trajectory the model writes — and the rubric scores — is what GRPO actually optimizes.**

Single-turn RL optimizes one response. Multi-turn optimizes a **strategy**: when to use a tool, when to ask a clarifying question, when to backtrack after an error. That's the behaviour an agent needs, and it only exists if the rollout gives the model somewhere to act.

---

## 10. Five takeaways

1. **Environment = dataset + rollout + rubric.** Three pieces.
2. **Same artifact: eval, training, data, harness.** Build once, use four ways.
3. **Rubric brittleness is the #1 footgun.** Parse first, score second.
4. **Verifiable rewards are unhackable.** Everything else, you fight.
5. **Multi-turn is where real RL lives.** Single-turn is for benchmarks.

### The bridge to next week

> **Same model. Different harness. Different result.**
>
> **Class 12:** harness engineering · context engineering · **how to evaluate a harness (≠ model eval)**.

---

## Further reading

**Primary:** [Verifiers docs](https://docs.primeintellect.ai/verifiers) · [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) · [Anakin87 walkthrough](https://hf.co/blog/anakin87/environments-hub)
**Deeper:** [Prime Intellect blog](https://primeintellect.ai/blog/environments) · INTELLECT-3 report (arxiv.org/abs/2512.16144)
**Adjacent:** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) · [Lakshyaag essay](https://lakshyaag.com/blogs/rl-environments)

---

## Discussion questions (from the slides)

1. What domain would you build an environment for?
2. How do you handle **non-verifiable rewards** (creative writing, design, taste)?
3. What changes if your rubric must run in **100 ms vs 10 s**?
4. When do you reach for **LLM-as-judge vs rule-based scoring**?

*Question 3 is the sharpest: rubric latency directly bounds group size. GRPO scores G=8 rollouts per prompt per step, so a 10-second rubric makes G=8 cost 80 seconds of pure scoring — which is exactly why Week 15 said cheap verifiers and GRPO were "designed for each other."*

---

## Common confusions

**"What *is* an environment, minimally?"** Three functions: produce a task, accept the model's response, return a score. Everything else — datasets, harnesses, multi-turn state — is elaboration on those three.

**"Why is it the same thing as an eval harness?"** Because both need tasks, a way to run the model, and a scorer. The only difference is what you do with the score: an eval **reports** it, RL **trains on** it, and a data generator **filters by** it. Building them separately guarantees they drift, and then your eval stops measuring what your training optimises.

**"What does 'verifiability = unhackability' actually mean?"** The more your scorer **computes** and the less it **judges**, the less room the model has to game it. Executing a test suite is deterministic and hard to fool. An LLM judge has preferences and blind spots, so optimising against it reopens Goodhart's Law (Week 14). Every step from computation toward judgement widens the attack surface.

**"Why is parsing the #1 footgun?"** Because a parsing failure is **indistinguishable from a wrong answer** at the reward level. Model answers correctly, regex misses it, reward = 0 — and you've just taught the model that correct answers are bad. It doesn't crash and it doesn't warn you. **Parse first, score second**, and test your parser against real outputs independently of the scorer.

**"How is this different from just writing an eval?"** Mostly it isn't — that's the point. The difference is design intent: an environment is built to be *run repeatedly at volume* by a training loop, so it must be fast, deterministic, and safe to execute untrusted model output against.

**"Why is multi-turn where 'real RL lives'?"** Single-turn is close to supervised learning with a computed label — one action, one reward. Multi-turn introduces genuine **credit assignment**: the model took eight actions and got a reward at the end, and it must work out which actions mattered. That's the actual RL problem, and it's much harder.

**"Can the model game a deterministic environment?"** Yes. Known patterns: writing code that special-cases the test inputs, exploiting a lenient parser, or finding a degenerate output that scores well. Verifiable rewards **shrink** the attack surface; they don't close it. Read actual high-scoring rollouts periodically — exploits are usually obvious to a human in ten examples and invisible in aggregate metrics.

**"Do I need an RL library for this?"** Not to understand it. GRPO fits in three lines (§6) and the environment interface is three functions. Libraries buy you batching, distributed rollout collection, checkpointing, and logging — real engineering value, but not conceptual machinery you're missing.

**"Is an environment just a simulator?"** Effectively yes — it's a **hand-built world model** supplying the rules of a world the model acts in. Environment design is hard for exactly the reasons learned world models are hard (S13): capturing what matters, in a form that supports learning, without exploitable flaws.

---

## Key takeaways

1. **An environment is one artifact with four jobs** — eval, RL training, synthetic data, harness experimentation.
2. **The trinity is dataset + rollout + rubric**; every RL setup in Weeks 12–15 decomposes this way.
3. **Rollout type determines what you can learn** — single-turn for benchmarks, multi-turn/tool/sandboxed for agents.
4. **LLM-judge rubrics are highly hackable** — "Great question!" raised the score without raising capability.
5. **Verifiable rubrics are programs**, and programs can't be flattered.
6. **Parse first, score second** — conflating extraction and scoring is the #1 footgun, and it fails silently as a zero.
7. **A zero rubric kills the gradient**, not just the metric (Week 15's zero cascade).
8. **Environments are portable across trainers** — prime-rl, TRL, SkyRL, verl, TorchForge, Tinker.
9. **The Environments Hub makes environments installable**, like packages.
10. **INTELLECT-3** (106B MoE, 512 H200s, ~2 months) is the existence proof at scale.

---

## Glossary

| Term | Meaning |
|---|---|
| **Environment** | Dataset + rollout + rubric — the artifact defining a task. |
| **Dataset** | Task inputs with ground truth or rubric metadata. |
| **Rollout** | How the model interacts: single-turn, multi-turn, tool-use, or sandboxed. |
| **Rubric** | The scoring function: verifiable, LLM-judge, or hybrid. |
| **Parser** | Component extracting structured output before scoring. |
| **Trajectory** | The full sequence of actions/observations a model produces in a rollout. |
| **LLM-as-judge** | Using a model to grade outputs; flexible but hackable. |
| **Verifiable reward** | Programmatic, deterministic scoring. |
| **Verifiers** | Prime Intellect's environment library (`SingleTurnEnv`, `ToolEnv`, `Rubric`, `Parser`). |
| **OpenEnv** | Meta + HuggingFace environment standard with Gymnasium-style `step()/reset()/close()`. |
| **Gymnasium API** | The classical RL interface convention. |
| **Environments Hub** | Public registry making environments installable via `prime env install`. |
| **prime-rl / TRL / SkyRL / verl / TorchForge / Tinker** | Trainers that consume environments. |
| **INTELLECT-3** | 106B MoE model trained on the Verifiers stack, released Nov 2025. |
| **GSM8K / HumanEval / τ-bench / SWE-bench / BrowseComp** | Benchmark datasets for maths, code, dialogue, real issues, and web research. |

# Week 15 — RLVR: Reasoning Models

**Subtitle:** From o1 to R1 — How RL Made LLMs Think (RLVR + GRPO + Emergent Self-Correction)
**Date:** 25/04/2026
**Sources:** `downloads/week-15-rlvr.pdf` (52 slides) · `downloads/week-15-rlvr.ipynb` (30 cells)
**Notion page:** https://100xschool.notion.site/34effffa33e580dd8139f0d521288360

**Referenced links:**
- [OpenAI — Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [Fireworks — DeepSeek R1 deep dive](https://fireworks.ai/blog/deepseek-r1-deepdive)
- [DeepSeek R1 paper (arXiv 2501.12948)](https://arxiv.org/pdf/2501.12948) — the same paper used in Week 10's PageIndex demo
- [Shared Google Sheet](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit?gid=650541192)

---

## 0. The idea in plain language

Week 14 ended with a problem. RLHF needs a **reward model** — a learned stand-in for human judgement — and because it's learned, it has flaws, and because it has flaws, the policy learns to exploit them. Goodhart's Law, unavoidable.

**RLVR's move: for some tasks, you don't need a learned reward at all. You can just *check*.**

- Maths problem? **Run the arithmetic.** Right or wrong.
- Code? **Execute the tests.** Pass or fail.
- Logic puzzle? **Verify the constraints.** Satisfied or not.

No reward model. No human labelling. No approximation. The reward is **computed**, and it's correct by construction. That's the whole idea, and the name says it: **R**einforcement **L**earning with **V**erifiable **R**ewards.

**Why this unlocked reasoning.** Here's the thing that makes it powerful: you don't have to tell the model *how* to solve the problem. You only check whether the final answer is right. So the model is free to try any approach — and what it discovers, entirely on its own, is that **thinking longer helps**.

Nobody trained it to say "wait, let me reconsider." Nobody wrote a backtracking module. The model tried long chains of reasoning, those chains produced correct answers more often, so the reward pushed it toward generating them. **Self-correction emerged from the incentive**, which is genuinely striking and is the headline result of the DeepSeek-R1 work.

**The new scaling axis.** Until this point, "better model" meant more parameters and more training data. RLVR added a third dimension: **more compute at inference time**. A model that thinks for 30 seconds outperforms the same model answering instantly. You can now buy capability with *thinking time* rather than only with model size — which is why "reasoning models" behave so differently from what came before.

The results were not subtle: **GPT-4o scored 13% on AIME maths; o1 scored 83%.** Same architecture, same scale, different training.

**GRPO is the algorithm that made it practical.** Standard RL (PPO) needs a separate "critic" model estimating how good each state is — expensive and fiddly. GRPO's simplification is neat: **generate a group of answers to the same question, and score each one relative to the group average.** Better than your peers → reinforce. Worse → discourage. The group *is* the baseline, so the critic disappears entirely. That's much cheaper and much simpler, and it's what let DeepSeek do this at scale.

**The catch, stated honestly, because it bounds the whole technique:** RLVR only works where **verification is cheap and unambiguous**. Maths, code, and formal logic qualify. "Write a good essay," "is this email polite," and "is this medical advice sound" do not — and for those you're back to Week 14's learned reward models and their flaws. This is why reasoning models are dramatically better at maths and code specifically, rather than uniformly better at everything.

**And Goodhart doesn't fully disappear.** Even with a deterministic verifier, models find exploits — passing tests by special-casing the test inputs, or reaching the right answer through invalid reasoning. Verifiable rewards *shrink* the attack surface; they don't eliminate it. Week 16 covers what happens when you try to build environments around this.

---

## 1. The day everything changed

**September 12, 2024** — OpenAI releases **o1-preview**.

| Model | AIME math |
|---|---|
| GPT-4o | **13%** |
| o1-preview | **56%** |
| o1 | **83%** |

> **Same architecture. Same scale. Different training.**

| Old way | o1's way |
|---|---|
| Prompt → instant answer | Prompt → **[thinks 30s]** → answer |

> *"We trained these models to spend more time thinking through problems before they respond, much like a person would."* — OpenAI
>
> **Thinking time isn't padding — it's the model exploring, backtracking, double-checking.**

### System 1 vs System 2 (Kahneman)

| System 1 | System 2 |
|---|---|
| Fast, intuitive, automatic | Slow, deliberate, analytical |
| "What's 2+2?" → 4 (instant) | "What's 17 × 23?" → pause, calculate, 391 |
| **Standard LLM behaviour** | **o1's mode** |

> **Pre-o1 LLMs were ALL System 1. o1 added System 2 — for the first time.**

### The mystery

OpenAI said what o1 could do, **not how they trained it.**

| What we knew | What we DIDN'T know |
|---|---|
| Used reinforcement learning · generated long chains of thought · CoT was hidden ("safety + competitive advantage") | The algorithm · the reward function · the training data · whether SFT was needed first · **whether reasoning was learned or imitated** |

> For 4 months the community had a working reasoning model and **zero understanding of how.**

### Then — January 20, 2025

**DeepSeek releases R1.** Same reasoning capability as o1. **Open weights. Open paper. With the recipe.**

| o1 said | R1 said |
|---|---|
| *"Reasoning models exist."* | *"Here's how to build one."* |

**Three breakthroughs in the R1 paper:**
1. **Pure RL works** (no SFT needed for R1-Zero)
2. **Reasoning EMERGES** (was not taught)
3. **The algorithm: GRPO** (simpler than PPO)

---

## 2. The new scaling axis

**Old scaling laws** (Kaplan, Hoffmann): more parameters → better · more data → better · more training compute → better. *For 5 years, "make it bigger" was the recipe.*

**o1 introduced a fourth axis:**

> **More TEST-TIME compute → better.** Let the model think longer at inference, get a better answer.

| The three curves | Era |
|---|---|
| 1. Model size (parameters) | pre-training |
| 2. Training compute (FLOPs) | pre-training |
| **3. Test-time compute (think tokens)** | **NEW** |

> **For the first time, scaling continues at inference, not just training.**
> **Implication: even small models can be made smart by giving them more thinking time.**

### The trade-off

| Pros | Cons |
|---|---|
| Better answers on hard problems | **10–100× more tokens** per response |
| Smaller models can match bigger ones | **10–100× slower** |
| Self-correction catches errors | **10–100× more expensive** · latency hurts UX |

> **Use reasoning models only when reasoning matters.**

---

## 3. The DeepSeek R1 story

> *"This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning."* — DeepSeek R1 paper

### R1-Zero — the pure RL experiment

**The hypothesis:** can a base model learn to reason with **NO supervised fine-tuning**?

```
Standard pipeline:      Pre-train  →  SFT  →  RLHF/DPO
DeepSeek's R1-Zero:     Pre-train  →  SKIP SFT  →  straight to RL
                                        ↑ This was supposed to be impossible
```

> Recall Week 14: **you MUST start RL from an SFT'd model.** R1-Zero violated that rule deliberately.

**The setup:**
- **Base model:** DeepSeek-V3-Base (671B MoE)
- **Algorithm:** GRPO
- **Reward:** ONLY answer correctness + format
- **No reward model. No human labels. No SFT.** Just *"did you get the math problem right?"*
- **Tasks:** math (AIME, MATH-500), coding (LiveCodeBench, Codeforces), logic
- **Reward = 1 if correct, 0 if wrong.** That's it.

**What happened — AIME 2024 pass@1:**

```
Step 0 (base):   15.6%
Step 1000:       30%
Step 5000:       55%
Step 8000:       71%
Final:           77.9%

For comparison:  GPT-4o 13%  ·  o1-0912 74%
```

> **Pure RL on math correctness — the model matches o1 on AIME.**

### The "Aha Moment"

Around **training step 8000**, the model started writing:

> *"Wait, let me reconsider..."*
> *"Actually, that's not right..."*
> *"Let me try a different approach..."*

> **These behaviours WERE NOT IN THE TRAINING DATA.** The reward function never said "write 'wait'." The model learned to **backtrack, reflect, and self-correct** — purely because doing so led to correct final answers.

**What it means:**

> **RL doesn't just polish behaviours. It can create NEW behaviours not in the data.**
> **Reasoning ≠ memorization of reasoning patterns. Reasoning IS reward-seeking search.**
> **You don't need to train the model to "think." You need a reward that rewards correctness. Thinking emerges as the means to the end.**

This is the deepest result in the course. Compare Week 1's *emergent capabilities* from scale — this is emergence from **optimization pressure**. Self-correction wasn't designed, demonstrated, or requested; it was *discovered* by the model because it raised the reward.

### The full R1 recipe (4 stages)

| Stage | What |
|---|---|
| **1. Cold-start SFT (small)** | A few thousand high-quality reasoning examples — a "head start" |
| **2. RL on reasoning (GRPO)** | Like R1-Zero but from the SFT'd model. **More readable.** |
| **3. Rejection sampling + SFT** | Generate, filter for quality, retrain. Adds general task capability. |
| **4. Final RL with RLHF + RLVR** | Combine human preference + verifiable rewards. |

**Result: R1 — readable AND reasoning-capable.**

### The two lessons

| Lesson 1: Pure RL works (R1-Zero proved it) | Lesson 2: But pure RL alone isn't enough (R1 showed it) |
|---|---|
| Reasoning is not just memorization · **emergence is real** | For production, mix RL + SFT · **use SFT for format, RL for capability** |

> **The breakthrough was DEMONSTRATING #1. The practical recipe is #2.**

Note the honest framing: R1-Zero was a *scientific* result, not a product. The shipped model uses SFT — because pure-RL output, while capable, was hard to read.

---

## 4. RLVR — Reinforcement Learning with Verifiable Rewards

### The reward problem from Week 14

RLHF rewards come from a **model** — approximate, hackable, effortful to train, subjective. **Goodhart's Law: optimize a proxy, the proxy gets corrupted.**

### The insight

> **For SOME tasks, we don't need a model to score. We can write a PROGRAM.**

```python
Math:   if int(answer) == expected:  reward = 1
Code:   if all_tests_pass(code):     reward = 1
JSON:   if parses_validly(output):   reward = 1
Logic:  if proof_checker(out):       reward = 1
```

> **Verifiable rewards = programmatic checks = deterministic, cheap, perfect.**

### RLHF vs RLVR

| | RLHF | RLVR |
|---|---|---|
| **Reward source** | Reward model | **Program/verifier** |
| **Reward signal** | Continuous | **Binary (or stepped)** |
| **Reward training** | Needs data | **Just write code** |
| **Reward hacking** | Common | **Rare** (verifier is exact) |
| **Domain** | Subjective tasks | **Objective tasks** |

> **You can do BOTH on the same model** — R1 used both in stage 4. Different rewards for different tasks.

### When RLVR works

| ✅ WORKS | ❌ DOESN'T WORK |
|---|---|
| Math (compare numerical answers) | Creative writing (no "correct") |
| Code (run tests, count passes) | Helpfulness (subjective) |
| Logic (formal proof checker) | Translation (multiple valid) |
| Format compliance (regex) | Conversation (no ground truth) |
| JSON validity (parser) | Style (preference-based) |
| Game playing (game state) | |

> **The split is roughly objective vs subjective. If you can write the verifier, RLVR works.**

### The R1 reward function (concrete)

```python
def reward(response, ground_truth):
    score = 0
    # Format reward — encourage <think>...</think>
    if "<think>" in response and "</think>" in response:
        score += 0.1
    # Correctness reward — does final answer match?
    final = extract_answer(response)
    if final == ground_truth:
        score += 1.0
    return score
```

> **That's it. No reward model. No labels. Just rules.**

**Why format reward matters:** without it, correct answers arrive in random formats, hard to extract and hard to use downstream. With it, reasoning is wrapped in `<think>` tags — parseable, visible, reliably evaluable.

> **Format reward is small (0.1) so it's a NUDGE, not the main signal. Correctness is the main signal.**
> **Format helps the engineering. Correctness drives the capability.** Get the ratio wrong and you get **format-only optimization without learning.**

### The RLVR limitation

> **RLVR can ONLY improve what the base model can do.** If the base model never produces the correct answer, RL can never reward it, and never reinforce it.

**Recent research (2025):** RLVR mostly improves **SAMPLING EFFICIENCY**, not raw capability — it teaches the model to **find its existing knowledge faster**, not to learn new knowledge.

> **This is a hot debate. The capability/efficiency question is unresolved.**

Worth holding honestly: it sits in tension with the aha moment. Self-correction *looked* like a new behaviour, but the counter-argument is that the base model could already produce such text occasionally, and RL merely made it frequent. Both readings fit the evidence.

---

## 5. GRPO — Group Relative Policy Optimization

### The problem with PPO: the value model

PPO needs a **value model (critic)** to compute the advantage: `A = actual_reward − V(s)`. The critic is a **separate neural net, roughly the same size as the policy.**

| Cost | Impact |
|---|---|
| **Memory** | Doubles your training footprint |
| **Compute** | Doubles your gradient computation |
| **Stability** | Critic training is its own headache |

For a 7B model: policy 14 GB + reference 14 GB + reward 14 GB + **value 14 GB = 56+ GB just for weights.**

### The GRPO insight

> **What if we DON'T need a value model? What if we estimate "how good is this response" by COMPARING IT TO OTHER RESPONSES TO THE SAME PROMPT?**

```
Generate 8 responses to the same question.
Score each:  r_1, r_2, ..., r_8
Compute mean(r) and std(r)
Advantage_i = (r_i − mean) / std     "How much better than the AVERAGE response?"
```

**No value model needed.**

**Why "above average" is a useful signal:** the same prompt yields 8 different reasoning attempts; some reach correct answers, some don't. **Reward** the above-average ones, **punish** the below-average ones → the policy shifts toward what works *for this prompt*, with no need to predict global reward distributions.

> **The "group" in GRPO = the 8 generations from one prompt. The "relative" = each compared to its peers.**

The critic was only ever a *learned estimate* of expected reward. GRPO replaces it with an **empirical** estimate from actual samples — which is both cheaper and unbiased.

### The equation

```
L_GRPO = (1/G) Σ_i [ min( r_i · A_i,  clip(r_i, 1−ε, 1+ε) · A_i ) − β · KL(π_θ || π_ref) ]

G   = group size (typically 8)
r_i = π_θ(o_i|q) / π_θ_old(o_i|q)      policy ratio
A_i = (R_i − mean(R)) / std(R)          group-normalized advantage
KL  = anchor to reference model
```

> **Compare to PPO: same min/clip, same KL. DIFFERENT advantage — empirical mean vs critic.**

### Step by step

```
For each training prompt q:
1. Generate G responses: o_1 ... o_G   (using current policy)
2. Score each with the verifier:  R_1 ... R_G
3. Normalize within group:  A_i = (R_i − mean) / std
4. Update policy weights:  ↑ high-A, ↓ low-A
5. Apply KL penalty to stay near reference
6. Repeat with next prompt
```

### Group size

| G too small (2) | G too large (32) |
|---|---|
| Advantage is noisy · hard to distinguish good from average | Compute scales linearly · **diminishing returns past ~8** |

> **Sweet spot: G = 8** (used in R1). Small models with limited compute: **G = 4**.

### GRPO vs PPO

| | PPO | GRPO |
|---|---|---|
| Models in memory | 4 | **3** ← no value model! |
| Stability | Tricky | **Stable** |
| Hyperparameters | 12+ | **5** |
| Compute per step | High | **Lower** |
| Generation | Sequential | **Parallel (group)** |

### Why GRPO + RLVR matched perfectly

| GRPO needs | RLVR provides |
|---|---|
| A **cheap reward signal** (scoring G=8 per prompt) → can't afford a slow learned reward model | A **programmatic verifier** (microseconds per check) |

> **Designed for each other. PPO + RLHF was the old pair. GRPO + RLVR is the new pair.**
> **RLVR + GRPO is the foundation of every post-2025 reasoning model.**

### The KL penalty still matters

Without an anchor, the model exploits the verifier: writing `<think>I'll cheat by...` scores well on format, poorly on correctness, but the *pattern* works → drift, instability, gibberish. With KL, it stays near base behaviour and reasoning **emerges organically from the original linguistic capabilities.** β typically **0.001–0.01**.

---

## 6. The hands-on — a controlled failure

> Train SmolLM-135M with GRPO on GSM8K math. **This is going to fail. ON PURPOSE.**

**Why:** Unsloth's docs explicitly warn — *"Apply GRPO to a model at least 1.5B in parameters to correctly generate thinking tokens."* **SmolLM-135M is 11× smaller than that floor.**

**What we'll observe:** the algorithm runs perfectly · the reward functions fire perfectly · **the model learns NOTHING.**

> **The failure IS the lesson.**

Genuinely excellent pedagogy — a deliberately negative result teaches the precondition better than any successful run could.

### The 5 reward functions (Will Brown's GSM8K stack)

| Function | Max reward | Measures |
|---|---|---|
| `correctness_reward` | **+2.0** | Final answer correct? |
| `int_reward` | +0.5 | Answer is an integer? |
| `strict_format_reward` | +0.5 | Exact XML structure? |
| `soft_format_reward` | +0.5 | XML structure (lenient)? |
| `xmlcount_reward` | +0.5 | **Partial credit per tag** |
| | **Total max: +4.0** | |

> The **graduated** rewards (xmlcount especially) exist so the model gets **some** signal before producing full correct answers. **Watch xmlcount — it's the canary in the coal mine.**

### The diagnostic — three signals

| Signal | Question |
|---|---|
| **1.** `correctness_reward > 0` | Is the model getting math right? |
| **2.** `reward_std > 0` | Are the generations producing **different** rewards? **If all get 0, NO learning signal.** |
| **3.** `xmlcount_reward > 0` | Is the model producing **ANY** structure? If 0, the algorithm has nothing to amplify. |

> **ALL THREE must be > 0 for GRPO to learn. If any are 0, the run is dead.**

Signal 2 is the subtle one and follows directly from the algorithm: `A_i = (R_i − mean)/std`. If every generation scores identically, the advantage is zero for all of them, so the gradient is zero — **regardless of whether the score was 0 or 4.** GRPO learns from *variance*, not from level.

### What actually happened

```
Step | reward | reward_std | correctness | xmlcount | kl
─────┼────────┼────────────┼─────────────┼──────────┼──────
  1  |  0.00  |    0.00    |    0.00     |   0.00   | 0.00
  ...
 18  |  0.00  |    0.00    |    0.00     |   0.00   | 0.00
```

> **Every column. Every row. Exactly zero.**

The model generated 60–512 token completions; **none** contained a `<reasoning>` tag, **none** got the maths right, all generations per prompt received identical (zero) reward.

### The zero cascade

```
xmlcount = 0        (model produces no XML tags)
   ↓
soft_format = 0     (no <reasoning>...<answer> structure)
   ↓
strict_format = 0   (no exact format match)
   ↓
int_reward = 0      (extracted answer isn't even an integer)
   ↓
correctness = 0     (extracted answer never matches truth)
   ↓
ALL rewards = 0     (reward_std = 0, advantage = 0)
   ↓
NO gradient         (KL = 0, weights frozen)
```

> **The cascade ALL starts at the top: the model can't even produce the format.**

### The lesson

> **RL polishes existing behaviours. RL amplifies existing patterns. RL does NOT teach behaviours from scratch.**
>
> For RL to work, **the base model must occasionally produce the target behaviour — even by accident.** Then RL increases the frequency. **If the base model can't produce the target ONCE in 100 generations, RL cannot help.**

### When to reach for RL

You want to add capability X. **Is X already in the base model AT ALL?**

| Answer | Action |
|---|---|
| **NO** | **Use SFT first** — teach the pattern from imitation |
| **YES, rarely** | **Use RL** (RLHF, DPO, GRPO) — amplify the existing rare behaviour |
| **YES, often** | Maybe RL for refinement, maybe just better prompting |

**The R1 recipe sequences these correctly:**
```
Pre-train        →  Cold-start SFT   →  RL (GRPO)
"build capacity"    "teach the format"   "amplify the capability"
```

---

## 7. Limitations

**Reward hacking even in RLVR** — the model finds tricks that pass tests but aren't right; produces verbose `<think>` tags with empty content; exploits regex format rewards without reasoning.

**Defense:** multiple reward functions (correctness + format + length) · adversarial test sets (verifier coverage) · KL penalty to reference.

> **Goodhart's Law is undefeated. Even with deterministic rewards.**

**The six limitations:**
1. Only works for tasks with **verifiable answers**
2. **Capability ceiling** = base model's reasoning paths
3. Reward hacking still happens
4. Long chains = **expensive at inference**
5. **Hidden CoT raises safety concerns** (alignment debate)
6. Unclear if reasoning **generalizes outside the training domain**

> **Reasoning models are not omniscient. They are math/code/logic specialists with emergent self-correction.**

---

## 8. The complete cheat sheet

| Need | Use |
|---|---|
| Domain knowledge | **CPT** (Week 12) |
| Format/instructions | **SFT** (Week 13) |
| Subjective quality | **RLHF / DPO** (Week 14) |
| Verifiable correctness | **RLVR + GRPO** (Week 15) |

**Production reasoning model:** `CPT → SFT → DPO → RLVR`

| Build | Use |
|---|---|
| A math tutor | **RLVR** (math is verifiable) |
| A customer support bot | **RLHF / DPO** (helpfulness is subjective) |
| A code generator | **RLVR** (tests are verifiable) |
| A creative writing assistant | **DPO** (style is preference-based) |
| A research assistant | **BOTH** (RLHF for help, RLVR for facts) |

> **Match the reward type to the task.**

### The future after R1

**Already happening (2025–2026):** multi-turn RLVR (agent + environment loops) · process rewards (intermediate step verification) · self-improvement (AI-generated training problems) · tool use within reasoning · **reasoning distillation** (small model, big mind).

**Still unknown:** does reasoning generalize beyond math/code? · can RLVR scale to truly novel domains? · what's the limit of test-time compute scaling?

---

## Common confusions

**"What's actually different about a 'reasoning model'?"** Not the architecture — it's the same transformer. What differs is training: it was rewarded for producing long chains of reasoning that reach verifiably correct answers, so it *learned to think before answering*. The visible "thinking" is generated tokens like any other, just trained to be useful rather than presented.

**"Is the thinking real, or is it theatre?"** It's real in the sense that it measurably improves accuracy and the model's answer depends on it. It is *not* necessarily a faithful account of the computation — the stated reasoning can diverge from what actually drove the answer. Treat it as useful working, not as an audit trail.

**"How is RLVR different from RLHF?"** The **reward source**. RLHF learns a reward model from human preferences, so the reward is an approximation with exploitable flaws. RLVR *computes* the reward by executing or checking, so it's correct by construction. That's why RLVR resists reward hacking far better — though not perfectly.

**"Why doesn't RLVR just replace RLHF entirely?"** Because it requires **cheap, unambiguous verification**, and most things people want don't have it. You can execute code and check maths. You cannot programmatically verify that an essay is insightful or a reply is appropriately empathetic. RLVR is narrow and deep; RLHF is broad and approximate. Frontier models use both.

**"Was self-correction programmed in?"** No, and this is the interesting part. Nobody wrote a backtracking routine or trained on "wait, let me reconsider" examples. The model was rewarded only on final correctness, discovered that longer deliberation produced more correct answers, and the behaviour emerged from the incentive. That's why the R1 result got attention.

**"What does GRPO actually simplify?"** PPO needs a separate **critic/value model** estimating expected future reward — a second network to train, tune, and keep stable. GRPO replaces it by generating a **group** of answers to the same prompt and scoring each against the group average. The group is the baseline. One less model, much less tuning.

**"Why does GRPO need non-zero temperature?"** Because the whole method depends on the group's answers *differing*. If every sample is identical, they all score the same, every advantage is zero, and there's no gradient. Deterministic decoding makes group-relative advantage meaningless (S7).

**"Does RLVR teach the model new knowledge?"** No. It teaches it to *use* what it already has more effectively — to explore, check, and backtrack. Capability comes from pretraining; RLVR surfaces and organises it.

**"Can the model still cheat with a deterministic verifier?"** Yes, and it does. Known patterns: writing code that special-cases the test inputs rather than solving the problem, and reaching a correct answer through invalid reasoning that happens to land right. Verifiable rewards **shrink** the attack surface rather than closing it — Goodhart's Law is stubborn, and S13 shows the same failure appearing again against learned world models.

**"Does more thinking always help?"** No. There are diminishing returns, and on easy problems extended reasoning wastes tokens and can talk the model out of a correct first instinct. Thinking time is a cost as well as a capability — which is precisely why modern APIs expose it as a *tunable effort setting* rather than always maximising it.

---

## Key takeaways

1. **Reasoning emerged from RL — not designed in** (R1's aha moment at step 8000).
2. **Test-time compute is the new scaling axis** — the first that operates at inference.
3. **RLVR replaces reward models with verifiers** — deterministic, cheap, and much harder to hack.
4. **GRPO replaces the value model with group statistics** — advantage from peers, not a critic.
5. **Pure RL works (R1-Zero) — IF the base model is large enough.**
6. **The model size floor for GRPO is ~1.5B parameters.**
7. **RL amplifies and polishes. RL does NOT teach from scratch.**
8. **GRPO learns from reward *variance*** — identical rewards across a group means zero gradient, even if all are perfect.
9. **Format reward must stay a nudge (0.1)**; correctness must dominate.
10. **Goodhart's Law survives even deterministic rewards.**
11. **Reasoning models are specialists**, not general improvements — use them only when reasoning matters.

> **The algorithm we ran today was identical to R1's. The only thing different was the base model.**
> **Capacity is the floor. Algorithms work above it, not below it.**

---

## Glossary

| Term | Meaning |
|---|---|
| **RLVR** | Reinforcement Learning with Verifiable Rewards — rewards from programs, not models. |
| **GRPO** | Group Relative Policy Optimization — advantage from group statistics, no critic. |
| **Test-time compute** | Inference-time "thinking" tokens; a scaling axis independent of model size. |
| **System 1 / System 2** | Kahneman's fast-intuitive vs slow-deliberate thinking. |
| **Chain of thought (CoT)** | Intermediate reasoning tokens produced before the answer. |
| **R1-Zero** | DeepSeek's pure-RL model trained with no SFT. |
| **Aha moment** | Step ~8000, where self-correction ("Wait, let me reconsider…") emerged untaught. |
| **Emergence** | Behaviour arising from optimization pressure without being trained for. |
| **Verifier** | A program that deterministically checks correctness. |
| **Format reward** | Small bonus (0.1) for correct output structure, e.g. `<think>` tags. |
| **Value model / critic** | PPO's network predicting expected reward; eliminated by GRPO. |
| **Advantage** | How much better an action was than expected/average. |
| **Group-normalized advantage** | `(R_i − mean(R)) / std(R)` across G generations of one prompt. |
| **Group size (G)** | Generations per prompt; sweet spot 8, workable at 4. |
| **`reward_std`** | Spread of rewards in a group; zero means no learning signal. |
| **Zero cascade** | Failure chain from no format → no extractable answer → no reward → no gradient. |
| **Sampling efficiency** | Finding existing knowledge faster, as distinct from gaining new capability. |
| **Process rewards** | Rewarding intermediate reasoning steps, not just final answers. |
| **Reasoning distillation** | Transferring a large reasoning model's behaviour into a smaller one. |
| **pass@1** | Accuracy of the first sampled answer. |
| **AIME / GSM8K / MATH-500** | Maths benchmarks used to evaluate reasoning. |

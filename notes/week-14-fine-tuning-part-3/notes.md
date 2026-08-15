# Week 14 — Fine-tuning Part 3: RLHF & DPO

**Subtitle:** Teaching the Model What "Good" Means — From Preference Pairs to Aligned Models
**Date:** 17/04/2026
**Sources:** `downloads/week-14-fine-tuning-part-3.pdf` (47 slides) · `downloads/week-14-fine-tuning-part-3.ipynb` (29 cells)
**Notion page:** https://100xschool.notion.site/345ffffa33e580329b21d1f37245d947

**Referenced links:**
- [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
- [Y Combinator companies, W/S/S 2026 batches](https://www.ycombinator.com/companies?batch=Summer%202026&batch=Spring%202026&batch=Winter%202026)

---

## The ChatGPT origin story

> GPT-3 was released in **June 2020**. For 2.5 years, almost nobody outside AI cared.
> **November 2022:** ChatGPT launches. **100 million users in 2 months** — the fastest-growing product in history.
>
> Same base model (GPT-3.5). Same architecture. Same parameters. **What changed? Four letters: R-L-H-F.**

The best possible motivation for this lecture. The capability was sitting there for two and a half years; **alignment** is what made it usable.

---

## Where we are

| Stage | Signal | Teaches |
|---|---|---|
| Pre-training | predict next token | Language |
| CPT | same loss, domain data | Domain language |
| SFT | per-token, on responses | Instruction following |
| **RLHF / DPO** | **per-RESPONSE, contrastive** | **What humans prefer** ← TODAY |
| RLVR | verifiable rewards | Reasoning |

---

## 1. Why SFT isn't enough

**Week 13's three failures:**

1. **Sycophancy** — user says "Revenue was $100B, right?", context says $53B → model agrees
2. **Hallucination** — asked about employees with only revenue in context → "The company had 47,000 employees"
3. **Format overfitting** — works with the template, breaks without it

> **SFT couldn't fix these because it only shows GOOD examples. It never shows BAD examples and says "don't do this."**

### The style problem

**User:** *"Help, I have to give a presentation tomorrow and I'm panicking."*

| Response A (blunt) | Response B (supportive) |
|---|---|
| *"You'll be fine. Practice your slides, get some sleep, and stop overthinking it."* | *"That sounds really stressful. Do one quick practice run, pick the 1-2 points you want people to remember, and give yourself permission to rest tonight."* |

> Both are "correct." Neither is wrong. **But most humans strongly prefer B.**
> **SFT has no mechanism to express "B is better than A."**

### The fundamental difference

| SFT signal | RLHF signal |
|---|---|
| *"Here is a good response. Copy it."* | *"A is BETTER than B."* |
| One example at a time · **per-token** · **imitation** | Two examples compared · **per-response** · **contrastive** |

> **SFT = learning from a textbook** (here's the right answer)
> **RLHF = learning from feedback** (this was better than that)

### The two loss functions

```
SFT  (per-token):     L = − Σ_t log π_θ(y_t | x, y_<t)
                      Sum over every token. Imitate the training sequence.

DPO  (per-response):  L = − log σ( β·Δ_chosen − β·Δ_rejected )
                      where Δ(y) = log [π_θ(y|x) / π_ref(y|x)]
                      One comparison per response.
```

> **The structural difference IS the per-token vs per-response shift.**

### Why negative examples matter

Teaching someone to draw a circle:

| SFT approach | RLHF approach |
|---|---|
| Show 1000 perfect circles: *"Make it look like this."* | Show pairs: *"This circle > this oval."* |
| **Problem:** they don't know what MISTAKES look like | **Learns the BOUNDARIES** of what's good |

> **Contrastive learning teaches boundaries. SFT only teaches the center.**
> Preferences are inherently about **boundaries** — what's acceptable vs not.

### SFT memorizes, RL generalizes

Proven empirically (**Chu et al., 2025**):

```
SFT on Task A:  better at A.  WORSE at Task B.
RL  on Task A:  better at A.  Task B stays INTACT.
```

| SFT says | RL says |
|---|---|
| *"Your outputs should look EXACTLY like this data."* | *"Find outputs that score well."* |
| Stretches to match training data → **disrupts other capabilities** | Improves in one direction → **preserves unrelated capabilities** |

This is the mechanistic explanation for the alignment tax from Week 13 — and the reason RL is the preferred tool once you can afford it.

---

## 2. RL for LLMs

### The objective in one line

```
max_π  E[ R(τ) ]  −  β · KL(π_θ || π_ref)
       └─ maximize ─┘  └── don't drift from where you started ──┘
```

> **Every RL algorithm solves a version of this. PPO, DPO, GRPO — different math, same goal.**

Worth memorising: **maximise reward, stay close to the starting point.** Everything else is implementation.

### The dog training analogy

You can't **explain** what "fetch" means (no SFT). You can't **write down the rules** (no loss function on behaviour).

```
1. Dog tries something        (EXPLORATION)
2. Dog brings back the ball   (ACTION)
3. You give it a treat        (REWARD)
4. Dog does more of that      (POLICY UPDATE)
```

> **Try → feedback → adjust.** The dog never sees the rules; it discovers them through rewards.

This is Week 1's definition of learning, third appearance — after the Week 2 training loop and the Week 8 agent loop.

### Vocabulary mapped to LLMs

| RL term | Dog training | LLM |
|---|---|---|
| **Agent** | The dog | The language model |
| **Environment** | The park | The conversation |
| **State** | Where the ball is | The prompt |
| **Action** | Run, fetch, sit | Generate a response |
| **Policy** | Dog's learned habits | **The model's weights** |
| **Reward** | Treat (or no treat) | Preference score |
| **Episode** | One fetch attempt | One prompt-response pair |

> **The POLICY is what RL optimizes. For LLMs, the policy IS the model.**
>
> *"Optimizing the policy" = "changing the weights" = "fine-tuning the model" = what we've been doing all along!*

| SFT | RL |
|---|---|
| "make **this** response more likely" (imitation) | "make **high-reward** responses more likely" (optimization) |

### What IS a reward?

In robotics, reward is objective and automatic (+1/−1: did the robot reach the target?). **In LLM RLHF, "how good was this response?" — but who decides "good"?**

| Option 1 | Option 2 |
|---|---|
| Ask a human every time → **too slow, too expensive** | **Train a MODEL to predict what humans would say** |
| | ▸ This is the **reward model**<br>▸ A learned approximation of human judgment<br>▸ Fast, scalable, **but imperfect (it's a proxy)** |

That parenthetical is the seed of everything that goes wrong later (§5, Goodhart's Law).

---

## 3. The RLHF pipeline (InstructGPT recipe)

OpenAI, 2022 — *"Training Language Models to Follow Instructions with Human Feedback"*

```
Step 1: SFT                    [you know this]
Step 2: Train a REWARD MODEL   collect human preference pairs, train a scorer
Step 3: Optimize with PPO      generate → score → update → constrain, ×1000s
```

### You MUST start from an SFT'd model

| Base model | After SFT |
|---|---|
| Random autocomplete, no instruction following | Follows instructions, produces structured responses |

> **RLHF needs REASONABLE responses so it can compare them.** If the model generates gibberish, there's nothing to compare.
>
> **Pipeline:** Base → SFT → RLHF/DPO. **Never:** Base → RLHF/DPO.

### Collecting preference data

**Prompt:** *"Explain what a stock dividend is."*

| Response A | Response B |
|---|---|
| *"A stock dividend is when a company gives shareholders additional shares instead of cash."* | *"A stock dividend means you get more shares instead of cash. E.g., 5% on 100 shares = 5 extra shares. Total value unchanged — each share just costs less."* |

**Human says: B > A** (more complete, has an example).

> **Not about right vs wrong — about BETTER vs WORSE.**

**Easy vs hard pairs:**

| EASY (obvious gap) | HARD (subtle gap) |
|---|---|
| Chosen: clear explanation with example<br>Rejected: vague, wrong, or off-topic | Chosen: *"Inflation means your money buys less over time. A coffee that cost $3 last year costs $3.50 now."*<br>Rejected: *"Inflation is a sustained increase in the general price level of goods and services in an economy."* |

> **The HARD pairs teach the MOST.** Easy pairs just confirm the obvious.

Note the hard pair: the rejected response is **factually impeccable** — it's a textbook definition. It loses on *accessibility*. That's the kind of judgment no rule could encode, which is precisely why preference learning exists.

### Training the reward model

- Take an SFT'd language model
- **Remove the language modelling head**
- **Add a single linear layer that outputs ONE number**

```
Input: [prompt + response] → Reward Model → score: 0.73
```

Trained so `score(prompt, chosen) > score(prompt, rejected)`.

**Features it learns:** accuracy ↑ · concrete examples ↑ · empathy ↑ · hallucination ↓ · verbosity ↓ — **all compressed into a single number.**

### Bradley-Terry — pairs to scores

From **chess Elo (1952)**. Pairwise wins → a single score per item.

```
P(chosen ≻ rejected) = σ( r(chosen) − r(rejected) )

Reward model loss:
L = − log σ( r_φ(chosen) − r_φ(rejected) )
```

- Same shape as **binary cross-entropy**
- **Only DIFFERENCES matter** — the absolute scale is meaningless

> **Why not just label chosen=1, rejected=0?** Because preferences are **probabilistic** — even the best response isn't preferred 100% of the time. Bradley-Terry models this naturally.

### PPO optimization

```
1. GENERATE   model produces responses to a batch of prompts
2. SCORE      reward model scores each response
3. UPDATE     adjust model so high-scoring responses become more likely
4. CONSTRAIN  don't drift too far from the SFT model
5. REPEAT     thousands of times
```

> **This is ONLINE training:** the model generates NEW data during training. It doesn't train on a fixed dataset — **it learns from its OWN outputs.**

**The objective:**

```
L_PPO = E[ min( r_t(θ)·A_t ,  clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]
```

| Piece | Plain English |
|---|---|
| `r_t(θ) = π_θ(a\|s) / π_old(a\|s)` | The **policy ratio** — how much more likely under the new model vs the old |
| `A_t` = advantage | **How much better this action was than average** = reward − expected reward |
| `clip(r, 1−ε, 1+ε)` | Cap the ratio at [0.8, 1.2] for ε=0.2 — **don't let any single update change probabilities by more than 20%** |

> **PPO = policy gradient, but CLIPPED so you never take a step too big.** Clip = the *"proximal."*

### The total reward

```
R_total = R_φ(x, y) − β · KL(π_θ || π_ref)
          └ reward model ┘  └ leash to reference ┘
```

| β | Effect |
|---|---|
| small | loose leash → faster learning, **risk of collapse** |
| large | tight leash → less drift, slower learning |
| **0** | no leash → **guaranteed reward hacking** |

**Typical: β = 0.01 to 0.1.**

> **This subtraction is the entire defense against reward hacking.**

**The reference model** is a frozen copy of the SFT model — your safety net, defining "what the model was like before RLHF." The KL penalty says: *you can improve, but stay CLOSE.*

---

## 4. Why PPO is hard

**Four models in memory at once:**

1. **Active model** (training) ← gradients, optimizer states
2. **Reference model** (frozen) ← KL anchor
3. **Reward model** (frozen) ← scores responses
4. **Value model / critic** ← predicts expected reward for the advantage calculation

For a **7B model in fp16**: 4 × 14 GB = **56 GB just for weights**, + gradients + optimizer ≈ **150+ GB total → need an 8×A100 cluster.**
For **70B**: 4 × 140 GB = **560+ GB VRAM minimum.**

**Plus:** online generation every step (slow) · **12+ hyperparameters** · training collapses if any are wrong · **reward hacking is rampant** · human labelling **$100K–$1M**.

> **This is why for 2 years only OpenAI and Anthropic did RLHF. Everyone else: SFT.**

---

## 5. Goodhart's Law — the fundamental problem

> **"When a measure becomes a target, it ceases to be a good measure."**

The reward model is a **proxy** for human preferences. Optimize too hard and:

- *"Adding 'I hope this helps!' gets +0.3 reward"* → adds it to **every** response
- *"Longer responses score higher"* → **walls of text** for simple questions
- *"Confident language scores higher"* → **confidently wrong** instead of honestly uncertain

> **This is REWARD HACKING.** The KL penalty, β tuning, and early stopping fight it.

Note the third example is genuinely alarming: optimizing for human approval **selects against calibrated uncertainty**, because humans rate confident answers more highly. That's a structural incentive toward the exact behaviour you don't want.

---

## 6. DPO — the simplification

**Rafailov et al., NeurIPS 2023 — "Direct Preference Optimization"**

> **Key insight:** the optimal policy under RLHF can be expressed **DIRECTLY** in terms of the policy and reference model. **No reward model needed.**

```
RLHF:  preferences → reward model → RL → better model
DPO:   preferences → better model              (one step!)
```

| | Change |
|---|---|
| Models in memory | **4 → 2** (policy + reference only) |
| Online generation | **not needed** (works on a fixed dataset) |
| Human labelling | **same data as RLHF** |
| Stability | **much easier** — it's basically SFT with a new loss |

### The intuition

For each `(prompt, chosen, rejected)`, DPO asks:
- How much does the **POLICY** like chosen vs rejected?
- How much does the **REFERENCE** like chosen vs rejected?

| Situation | Update |
|---|---|
| Policy already prefers chosen (matching reference) | **Small update.** Already aligned. |
| Policy prefers rejected (disagreeing with the data) | **Large update.** Fix this. |

> **The "relative to reference" part IS the KL constraint.** It's baked into the loss function, not a separate penalty.

That's the elegant part: PPO needs an explicit KL term because reward and constraint are separate objects. DPO's loss is *already expressed* as a ratio against the reference, so the constraint is structural.

### The loss, decoded

```
L_DPO = − log σ( β · log[π_θ(y_c|x) / π_ref(y_c|x)]
                − β · log[π_θ(y_r|x) / π_ref(y_r|x)] )
```

Read inside-out:

| Piece | Meaning |
|---|---|
| `log[π_θ(y\|x) / π_ref(y\|x)]` | *"how much MORE likely to generate y now vs where we started"* = **IMPLICIT REWARD** (no reward model needed!) |
| `chosen_reward − rejected_reward` | **preference margin** |
| `σ(margin)` | probability of the correct preference |
| `−log(...)` | standard cross-entropy |

### Why contrasting beats imitation

| SFT approach | DPO approach |
|---|---|
| Train on **chosen only** | Push chosen **UP** |
| **Ignore rejected** | Push rejected **DOWN** |
| Works OK, **misses half the signal** | Learns **what to do AND what to avoid** |

The rejected response contains information: *"this phrasing is too verbose," "this tone is too blunt," "this answer is too vague."*

> **Throwing away the rejected = throwing away lessons. DPO uses the FULL signal. SFT uses half.**

### Offline vs online

| RLHF (PPO): **ONLINE** | DPO: **OFFLINE** |
|---|---|
| Model generates **during** training | Works on a **fixed dataset** |
| New data every step | Same data every epoch |
| More powerful, expensive, unstable | Simpler, cheaper, more stable |

> **Trade-off:** online → the model can **explore and discover new good behaviours**. Offline → it can only learn from what's in the dataset.
> **For most use cases, offline is enough. For frontier performance, online still wins slightly.**

### Side by side

| | RLHF (PPO) | DPO |
|---|---|---|
| Models in memory | 4 | **2** |
| Training | Online (generates) | **Offline (fixed dataset)** |
| Reward model | Required | **Not needed** |
| Stability | Tricky | **Stable (like SFT)** |
| Compute | Very expensive | **Moderate** |
| Human data | Preference pairs | **Same pairs!** |
| Performance | Best at frontier | **Very close (90–95%)** |
| Learning rate | ~1e-6 | **~5e-7 to 5e-6** |

> **DPO trades a small performance gap for massive simplicity. This is why DPO exploded in 2024.**

---

## 7. The learning rate breakthrough

> **THE single most important DPO hyperparameter.**

| SFT learning rate | DPO learning rate |
|---|---|
| **2e-4** | **5e-7 to 5e-6** (10–100× LOWER!) |

**The history:** early DPO papers (2023) reported *"DPO doesn't work that well"* — **they used SFT-scale learning rates.** The **Zephyr team** (late 2023) tried 10–100× lower, and **suddenly DPO worked beautifully.**

A great cautionary tale: a technique was nearly written off because of one misconfigured hyperparameter.

### Why the maths demands it

DPO's gradient hits **harder** than SFT's per step:
- Each update affects **BOTH** chosen and rejected probabilities
- Both measured **relative to reference** (amplified)
- Net update ≈ **2×** what SFT would produce
- Plus the policy/reference ratio is **exponential** — small weight changes → large probability swings

**Practical rules:**
- **If loss bounces → halve LR.**
- **If loss is flat → double LR.**
- **Never above 1e-5** unless you know what you're doing.

---

## 8. Preference datasets

DPO needs `(prompt, chosen_response, rejected_response)`.

| Dataset | Detail |
|---|---|
| **UltraFeedback** | 64K pairs, diverse tasks (used in the notebook) |
| **Nectar** | 183K pairs, multi-turn |
| **HH-RLHF** | Anthropic's helpfulness/harmlessness data |

```json
{
  "prompt":   "Explain inflation in simple terms.",
  "chosen":   "Inflation means prices go up over time, so your money buys less.
               A $3 coffee this year might cost $3.50 next year.",
  "rejected": "Inflation is a sustained increase in the general price level of
               goods and services in an economy..."
}
```

> Both correct. **Chosen is simpler, uses a concrete example.**

---

## 9. Rejection sampling — the underrated middle ground

Even simpler than DPO:

```
1. Generate 10–30 responses per prompt
2. Score them with a reward model
3. Keep only the best ones
4. SFT on the best responses
```

> **No RL. No contrastive loss. Just "generate, filter, retrain."**
> Used by **Llama 3, Tulu 3, and most production systems.** Often combined with DPO in multi-stage pipelines.
>
> Think of it as: **curating your own SFT dataset using a scorer.**

Genuinely underrated — it captures much of RL's benefit (the model learns from *its own* outputs, which is what makes RL generalize) using only SFT machinery.

---

## 10. The DPO variant landscape

| Variant | Year | What it adds |
|---|---|---|
| **DPO** | 2023 | The original, most tested |
| **IPO** | 2024 | Fixes overconfidence |
| **KTO** | 2024 | Works with **thumbs-up/down** (no pairs needed!) |
| **ORPO** | 2024 | Combines SFT + DPO in **one step** |
| **SimPO** | 2024 | **No reference model** needed |

> **Start with DPO.** Most documented and supported. Branch out only for specific needs.

KTO is worth noting practically — thumbs-up/down data is *far* cheaper to collect than pairwise comparisons, and most products already log it.

---

## 11. The complete cheat sheet

| Symptom | Fix |
|---|---|
| Model doesn't know the domain | **CPT** |
| Model doesn't follow instructions | **SFT** |
| Model's answers lack style/quality | **DPO** |
| Model agrees when it shouldn't | **DPO + pushback pairs** |
| Model hallucinates confidently | **DPO + refusal pairs** |
| Model needs to reason better | **RLVR** (Week 15) |

**Production pipeline:**
```
Base → CPT → SFT → Rejection Sampling → DPO → (RLVR)
```
> **Each step fixes what the previous couldn't.**

Note rows 4 and 5 close the loop on Week 13's failure modes: sycophancy and hallucination are fixed here, with **pushback pairs** and **refusal pairs**.

---

## 12. The one equation to remember

```
L_DPO = − log σ( β · Δ̂_chosen − β · Δ̂_rejected )
where  Δ̂(y) = log [ π_θ(y|x) / π_ref(y|x) ]
```

| Three knobs | Three failure modes |
|---|---|
| **β** — preference strength (0.1) | **Reward hacking** → check length, formatting |
| **LR** — step size (5e-6) | **Overfitting** → use eval set, early stop |
| **data** — pair quality (**matters most**) | **Drift** → β too low OR LR too high |

---

## Key takeaways

1. **SFT shows good examples. RLHF/DPO shows good vs bad.**
2. **SFT memorizes, RL generalizes** — proven empirically (Chu et al., 2025), not just theory.
3. **RLHF pipeline:** SFT model → reward model → PPO. **Four models in memory.**
4. **DPO: same result, no reward model, 2 models, offline.**
5. **You MUST start from an SFT'd model.** Never DPO a base model.
6. **Reward hacking / Goodhart's Law is the fundamental challenge**; the KL penalty is the defense.
7. **Rejection sampling is underrated** — generate, filter, retrain.
8. **DPO learning rate: 5e-7 to 5e-6** — 10–100× lower than SFT.
9. **The RL objective is one line:** maximise reward, stay near the reference.
10. **The policy IS the weights.** Optimizing the policy = fine-tuning.
11. **Hard preference pairs teach the most** — the rejected response can be factually perfect.
12. **Contrastive learning teaches boundaries; imitation teaches only the center.**

---

## Glossary

| Term | Meaning |
|---|---|
| **RLHF** | Reinforcement Learning from Human Feedback. |
| **DPO** | Direct Preference Optimization — RLHF's result without a reward model. |
| **Policy (π_θ)** | The strategy mapping states to actions; for an LLM, its weights. |
| **Reference model (π_ref)** | Frozen SFT copy anchoring how far the policy may drift. |
| **Reward model** | A learned scorer approximating human judgment, outputting one number. |
| **Preference pair** | `(prompt, chosen, rejected)` — the training unit for RLHF/DPO. |
| **Bradley-Terry** | 1952 pairwise-comparison model (chess Elo) converting wins into scores. |
| **KL penalty** | `β · KL(π_θ ‖ π_ref)` — the leash preventing drift. |
| **β (beta)** | Strength of the KL constraint / preference sharpness; typically 0.01–0.1. |
| **PPO** | Proximal Policy Optimization — clipped policy gradient. |
| **Policy ratio** | `π_θ(a\|s) / π_old(a\|s)` — how much likelihood changed. |
| **Advantage** | How much better an action was than average. |
| **Clipping** | Capping the policy ratio (e.g. [0.8, 1.2]) so no update is too large. |
| **Online training** | The model generates new data during training (PPO). |
| **Offline training** | Training on a fixed dataset (DPO). |
| **Implicit reward** | `log[π_θ(y\|x)/π_ref(y\|x)]` — DPO's reward-model replacement. |
| **Reward hacking** | Exploiting the proxy reward without improving true quality. |
| **Goodhart's Law** | "When a measure becomes a target, it ceases to be a good measure." |
| **Rejection sampling** | Generate many responses, score, keep the best, SFT on those. |
| **Pushback pairs** | Preference pairs teaching the model to correct a mistaken user. |
| **Refusal pairs** | Preference pairs teaching the model to decline when context is insufficient. |
| **IPO / KTO / ORPO / SimPO** | DPO variants: overconfidence fix / binary feedback / SFT+DPO combined / no reference model. |

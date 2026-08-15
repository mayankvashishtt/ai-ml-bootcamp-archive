# Week 14 — Quiz (20 questions)

**Topic:** RLHF & DPO — teaching the model what "good" means
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The RL objective in one line is:
- A) Minimise cross-entropy over all tokens
- B) `max E[R(τ)] − β·KL(π_θ ‖ π_ref)`
- C) Maximise the reward model's score without constraint
- D) Minimise the distance to the reference model

**2.** For an LLM, the *policy* is:
- A) The system prompt
- B) The model's weights
- C) The sampling temperature
- D) The reward model

**3.** SFT's signal versus RLHF's signal:
- A) SFT per-response contrastive; RLHF per-token imitation
- B) SFT per-token imitation; RLHF per-response contrastive
- C) Both are per-token
- D) Both are contrastive

**4.** Chu et al. (2025) found that after training on Task A:
- A) SFT preserves Task B; RL degrades it
- B) RL preserves Task B; SFT degrades it
- C) Both preserve Task B equally
- D) Both degrade Task B equally

**5.** The Bradley-Terry model originated in:
- A) Reinforcement learning for robotics
- B) Ranking chess players (1952)
- C) Information retrieval
- D) Computer vision

**6.** In the reward model, what matters is:
- A) The absolute score values
- B) Only the differences between scores
- C) The score variance
- D) The score magnitude relative to 1.0

**7.** How many models must be in memory during PPO-based RLHF?
- A) 1
- B) 2
- C) 3
- D) 4

**8.** The KL penalty term `− β·KL(π_θ ‖ π_ref)` exists to:
- A) Speed up convergence
- B) Prevent drift from the reference, defending against reward hacking
- C) Normalise the reward scale
- D) Reduce memory usage

**9.** DPO eliminates which component of the RLHF pipeline?
- A) The SFT stage
- B) The reward model
- C) The preference data
- D) The reference model

**10.** The recommended DPO learning rate range is:
- A) 2e-4
- B) 1e-3 to 1e-2
- C) 5e-7 to 5e-6
- D) 1e-1

**11.** DPO is described as *offline* training, meaning:
- A) It runs without internet access
- B) It works on a fixed dataset rather than generating during training
- C) It runs on CPU
- D) It requires no GPU

**12.** Rejection sampling consists of:
- A) Rejecting low-confidence tokens during generation
- B) Generate many responses, score them, keep the best, SFT on those
- C) Discarding training examples with high loss
- D) Filtering the preference dataset by length

---

## Short answer

**13.** Explain the ChatGPT origin story and what it demonstrates about capability versus alignment.

**14.** Explain why negative examples matter, using the circle-drawing analogy. What can contrastive learning teach that imitation cannot?

**15.** Map the RL vocabulary (agent, environment, state, action, policy, reward, episode) onto LLM training.

**16.** Explain the reward model: how it is built from an SFT model, what it outputs, and why it is described as an imperfect proxy.

**17.** Explain Goodhart's Law with the three reward-hacking examples. Why is the third especially concerning?

**18.** Explain DPO's key insight and how the KL constraint is enforced without an explicit penalty term.

**19.** Explain why DPO requires a learning rate 10–100× lower than SFT, and tell the historical story around this.

**20.** Your DPO-trained model now produces much longer answers than before, and users rate them worse. Diagnose and give three concrete fixes.

---
---

## Answer key

**1. B** — Maximise expected reward while staying close to the reference. PPO, DPO, and GRPO all solve versions of this.

**2. B** — The weights. Different weights produce different responses to the same prompt, so "optimizing the policy" is literally fine-tuning.

**3. B** — SFT is per-token imitation ("copy this"); RLHF is per-response contrastive ("A is better than B").

**4. B** — RL improves Task A while leaving Task B intact; SFT improves Task A but degrades Task B.

**5. B** — Ranking chess players, 1952 — the same lineage as Elo.

**6. B** — Only differences matter; the absolute scale is meaningless, since the loss depends on `r(chosen) − r(rejected)`.

**7. D** — Four: policy, reference, reward model, and value model/critic.

**8. B** — It anchors the policy to the reference and is described as "the entire defense against reward hacking."

**9. B** — The reward model. The reference model is still required.

**10. C** — 5e-7 to 5e-6, roughly 10–100× lower than SFT's 2e-4.

**11. B** — It trains on a fixed preference dataset, unlike PPO which generates new responses during training.

**12. B** — Generate 10–30 responses per prompt, score with a reward model, keep the best, and SFT on those.

**13.** GPT-3 launched in June 2020 and drew little attention outside AI for two and a half years. ChatGPT launched in November 2022 and reached 100 million users in two months — the fastest-growing product in history — using the **same base model, architecture, and parameters**. The only difference was RLHF. **What it demonstrates:** raw capability and *usable* capability are different things. The model already possessed the knowledge and reasoning; what it lacked was the behavioural layer — knowing what kind of response humans actually want, how to be helpful rather than merely accurate, when to stop, and how to handle ambiguous requests. Alignment was not a polish step on top of the product; it *was* the product. This is also why the course spends three weeks on post-training: the gap between a capable model and a useful one is where most engineering value sits.

**14.** Teaching someone to draw a circle by **SFT** means showing 1,000 perfect circles and saying "make it look like this" — but the learner never sees what mistakes look like, so they have no idea how far from perfect is still acceptable, or which deviations matter. **RLHF** shows pairs — "this circle is better than this oval" — teaching where the boundary between acceptable and unacceptable lies. **What contrastive learning uniquely teaches: boundaries.** Imitation only teaches the *center* of the distribution of good outputs. Preferences are inherently about boundaries — what is acceptable versus not — and many quality judgments are only expressible comparatively. The presentation-panic example makes this concrete: both the blunt and supportive responses are correct and neither is wrong, so no single-example training signal can express that most humans strongly prefer the supportive one. Only a comparison can.

**15.** **Agent** = the language model (the dog). **Environment** = the conversation (the park). **State** = the prompt (where the ball is). **Action** = generating a response (run, fetch, sit). **Policy** = the model's weights (the dog's learned habits). **Reward** = the preference score (the treat, or no treat). **Episode** = one prompt-response pair (one fetch attempt). The policy is what RL optimizes, and since for an LLM the policy *is* the weights, "optimizing the policy," "changing the weights," and "fine-tuning the model" all denote the same operation. The difference from SFT is only in the objective: SFT says "make *this* response more likely" (imitation), RL says "make *high-reward* responses more likely" (optimization).

**16.** It is built by taking an **SFT'd language model**, **removing the language-modelling head** that predicts next tokens, and **attaching a single linear layer that outputs one number**. It consumes `[prompt + response]` and emits a scalar score such as 0.73. It is trained on preference pairs under the Bradley-Terry objective `L = −log σ(r_φ(chosen) − r_φ(rejected))`, so it learns that accuracy, concrete examples, and empathy raise the score while hallucination and verbosity lower it — all compressed into one number. **Why an imperfect proxy:** the true target is human preference, which is expensive and slow to query for every response, so the reward model is a *learned approximation* fitted to a finite sample of human judgments. It will be wrong in places, and — critically — it can be **exploited**: the policy is optimized against the proxy, not against humans, so any systematic error becomes an attack surface. That is exactly the Goodhart problem.

**17.** **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure." Because the reward model is a proxy, optimizing it too hard drifts away from what it was proxying. **The three examples:** the model learns that adding "I hope this helps!" earns +0.3 reward and appends it to every response; it learns that longer responses score higher and produces walls of text for simple questions; it learns that confident language scores higher and becomes confidently wrong rather than honestly uncertain. **The third is especially concerning** because it is not a cosmetic tic but a **degradation of calibration** — the model's honest signalling of uncertainty is trained away. It arises from a real property of the training data: human raters genuinely do prefer confident-sounding answers, so the reward model faithfully reflects human preference while that preference is itself misaligned with truthfulness. There is no way to fix this by improving the reward model's accuracy, because the reward model is not the thing that is wrong. This is why the defenses are structural — the KL penalty, β tuning, and early stopping — rather than better reward modelling.

**18.** **The insight (Rafailov et al., 2023):** the optimal policy under the RLHF objective can be written **directly** in terms of the policy and reference model, so preference data can be turned into a better model in one step, with no reward model in between. This cuts memory from four models to two, removes online generation, and makes training as stable as SFT. **How the KL constraint is enforced:** DPO's loss is expressed in terms of `Δ̂(y) = log[π_θ(y|x) / π_ref(y|x)]` — the log-ratio of the policy's likelihood to the reference's. This ratio *is* an implicit reward, and because every term is measured **relative to the reference**, moving far from the reference automatically produces large ratios that the loss resists. The constraint is therefore **structural rather than additive**: PPO needs an explicit `−β·KL` term because reward and constraint are separate objects, whereas in DPO the reference appears inside the reward definition itself, so staying close is built into the objective's form.

**19.** **Why the maths demands it:** each DPO update affects **both** the chosen and rejected probabilities, so the net update is roughly **2×** what SFT would produce from a comparable signal; both quantities are measured **relative to the reference**, which amplifies them further; and the policy/reference ratio is **exponential** in the log-probabilities, so small weight changes cause large probability swings. The contrastive signal is simply much stronger than SFT's gentle per-token "predict this token," so an SFT-scale rate overshoots and the loss bounces or the model collapses. **The historical story:** early DPO papers in 2023 concluded that "DPO doesn't work that well" — because they used SFT-scale learning rates around 2e-4. In late 2023 the **Zephyr team** tried rates 10–100× lower, and DPO suddenly worked beautifully. A genuinely useful technique was nearly written off over a single misconfigured hyperparameter. **Practical rules:** if the loss bounces, halve the LR; if it is flat, double it; never exceed 1e-5 without good reason.

**20.** **Diagnosis: length-bias reward hacking.** Human annotators tend to rate longer, more thorough-looking responses more highly, so length correlates with "chosen" in the preference data. DPO faithfully learns this correlation and optimizes for the proxy — verbosity — rather than the underlying quality it was standing in for. The user ratings dropping confirms the divergence: the model is scoring better on the learned preference signal while getting worse on the thing that signal was meant to measure, which is Goodhart's Law exactly. Note that the slides list "reward hacking → check length, formatting" as the first DPO failure mode for precisely this reason. **Three fixes:** **(i) Raise β.** A larger KL coefficient tightens the leash to the reference model, limiting how far the policy can drift toward degenerate verbosity; the failure mode "drift → β too low OR LR too high" points here directly, so lowering the learning rate is a companion fix. **(ii) Fix the data — the knob that matters most.** Audit the preference pairs for a length imbalance between chosen and rejected; rebalance by adding pairs where the **concise** response is chosen over a longer one of equal substance, so length stops predicting preference. The pattern mirrors the pushback- and refusal-pair remedies for sycophancy and hallucination. **(iii) Early-stop on a held-out evaluation.** Track response length alongside a quality metric on an eval set and stop when quality plateaus rather than when training loss does, since reward hacking typically worsens with continued optimization. If the problem persists, consider a DPO variant designed against this — **SimPO** includes length normalisation — or interpose a **rejection sampling** stage where the scorer explicitly penalises verbosity.

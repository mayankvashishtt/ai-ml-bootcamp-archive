# Week 15 — Quiz (20 questions)

**Topic:** RLVR, GRPO, and reasoning models (o1 → R1)
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** On AIME math, GPT-4o scored 13% while o1 scored:
- A) 34%
- B) 56%
- C) 71%
- D) 83%

**2.** The "fourth scaling axis" introduced by o1 is:
- A) More parameters
- B) More training data
- C) More test-time compute
- D) More attention heads

**3.** R1-Zero deliberately violated which prior rule?
- A) Always use a reward model
- B) You must start RL from an SFT'd model
- C) Never use LoRA for RL
- D) Group size must exceed 16

**4.** R1-Zero's reward function consisted of:
- A) A trained reward model on human preferences
- B) Answer correctness plus a small format bonus
- C) Human labels collected per response
- D) Perplexity on a held-out set

**5.** The "aha moment" occurred around training step:
- A) 100
- B) 1000
- C) 8000
- D) 50000

**6.** What made the aha moment significant?
- A) The model reached 100% accuracy
- B) Self-correction behaviours appeared that were never in the training data
- C) Training loss suddenly dropped to zero
- D) The model started using fewer tokens

**7.** RLVR differs from RLHF principally in that the reward comes from:
- A) A larger reward model
- B) A program or verifier
- C) Multiple human annotators
- D) The model scoring itself

**8.** GRPO eliminates which component of PPO?
- A) The reference model
- B) The KL penalty
- C) The value model / critic
- D) The clipping term

**9.** In GRPO, the advantage for generation i is computed as:
- A) `R_i − V(s)` using a learned critic
- B) `(R_i − mean(R)) / std(R)` across the group
- C) `R_i / max(R)`
- D) `log(R_i)`

**10.** The recommended GRPO group size used in R1 is:
- A) 2
- B) 4
- C) 8
- D) 32

**11.** Unsloth warns that GRPO requires a model of at least:
- A) 135M parameters
- B) 500M parameters
- C) 1.5B parameters
- D) 70B parameters

**12.** If all generations in a group receive identical rewards, the result is:
- A) Maximum learning signal
- B) Zero advantage and therefore no gradient
- C) A KL penalty spike
- D) Automatic early stopping

---

## Short answer

**13.** Explain the System 1 / System 2 framing and what o1 changed.

**14.** Explain why test-time compute is a genuinely new scaling axis and state the trade-off.

**15.** Describe the aha moment and explain precisely why it matters for our understanding of reasoning.

**16.** Explain the RLVR insight and give three tasks where it works and three where it doesn't, with the reason for the split.

**17.** Explain why format reward is 0.1 while correctness is 1.0, and what goes wrong if the ratio is reversed.

**18.** Explain the GRPO insight. Why can group statistics replace a learned critic?

**19.** Walk through the zero cascade in the failed run and explain what each of the three diagnostic signals tells you.

**20.** You want a model that writes SQL queries correctly against your schema. Design a training approach, justifying each stage.

---
---

## Answer key

**1. D** — 83%. o1-preview scored 56%, GPT-4o 13%.

**2. C** — Test-time compute: letting the model think longer at inference.

**3. B** — The rule that RL must start from an SFT'd model. R1-Zero went straight from pre-training to RL.

**4. B** — Answer correctness (+1.0) plus a small format bonus (+0.1). No reward model, no human labels, no SFT.

**5. C** — Around step 8000.

**6. B** — Behaviours like "Wait, let me reconsider…" appeared spontaneously; the reward function never asked for them.

**7. B** — A program or verifier that deterministically checks correctness.

**8. C** — The value model (critic), reducing models in memory from 4 to 3.

**9. B** — Group-normalized advantage across the G generations of the same prompt.

**10. C** — 8. Four is workable for small models with limited compute.

**11. C** — At least 1.5B parameters to correctly generate thinking tokens.

**12. B** — `A_i = (R_i − mean)/std` is zero for every generation, so there is no gradient — regardless of whether the shared reward was 0 or maximal.

**13.** Kahneman's **System 1** is fast, intuitive, and automatic — "What's 2+2?" answered instantly — while **System 2** is slow, deliberate, and analytical, as in pausing to compute 17 × 23. **Pre-o1 LLMs were entirely System 1:** prompt in, answer out, with a fixed amount of computation regardless of difficulty. **o1 added System 2** by spending inference-time compute before answering — thinking for around 30 seconds on hard problems. Crucially, that thinking time is not padding: it is the model exploring approaches, backtracking, and double-checking. The architecture and scale were unchanged; only the training differed.

**14.** The old scaling laws all operated at **training time** — more parameters, more data, more training FLOPs — so improving a model meant retraining a bigger one, and a deployed model's capability was fixed. Test-time compute is different because performance improves by letting an **already-trained** model generate more thinking tokens per question, meaning **scaling continues at inference**. The implication is significant: even small models can be made smart by giving them more thinking time, decoupling capability from parameter count. **The trade-off:** thinking tokens cost real money and time — 10–100× more tokens, 10–100× slower, 10–100× more expensive — and the latency harms user experience. Hence the guidance to use reasoning models only when reasoning actually matters.

**15.** Around step 8000 of R1-Zero's pure-RL training, the model spontaneously began producing text like "Wait, let me reconsider…", "Actually, that's not right…", and "Let me try a different approach…" — **backtracking, reflection, and self-correction that were never in the training data.** The reward function only checked whether the final answer was correct; it never mentioned these phrases or behaviours. **Why it matters:** it shows that RL can *create* new behaviours rather than merely polish existing ones, and it reframes what reasoning is — not memorised reasoning patterns copied from demonstrations, but **reward-seeking search**. The practical consequence is striking: you do not need to teach a model to think, you need a reward that rewards correctness, and thinking emerges as the means to that end. It parallels Week 1's emergent capabilities from scale, but here emergence comes from **optimization pressure** rather than size.

**16.** **The insight:** for some tasks we do not need a model to score outputs — we can write a **program**. `if int(answer) == expected: reward = 1`, or run the test suite, or parse the JSON. Verifiable rewards are deterministic, cheap, and exact, which removes the reward model's approximation error and most of its hackability. **Works:** maths (compare numerical answers), code (run tests and count passes), and formal logic (a proof checker) — also format compliance, JSON validity, and game playing. **Doesn't work:** creative writing (there is no correct output), helpfulness (inherently subjective), and translation (many valid renderings) — also conversation and style. **The split is objective versus subjective:** RLVR needs a ground truth that a program can check, so wherever correctness is a matter of judgment or taste, no verifier can be written and RLHF/DPO remains the tool. The rule of thumb is simply: if you can write the verifier, RLVR works.

**17.** **Correctness is the capability signal; format is only an engineering convenience.** The format bonus exists so that reasoning is wrapped in `<think>` tags — making the chain of thought parseable, visible, and reliably extractable for evaluation — but wrapping text in tags is not the behaviour you actually want to develop. Keeping it at 0.1 against correctness at 1.0 makes it a **nudge** rather than the main objective, so a correct answer always outweighs a well-formatted one. **If the ratio were reversed**, the model would optimize the cheapest available reward: it would learn to emit perfectly-formed `<think></think>` tags — potentially with empty or meaningless content — while never improving at the underlying task, since correctness would contribute a negligible fraction of total reward. That is **format-only optimization without learning**, and it is a textbook instance of reward hacking: the model maximises the measurable proxy while the thing being proxied stagnates. As the slides put it, format helps the engineering; correctness drives the capability.

**18.** **The insight:** rather than training a network to predict expected reward, estimate "how good is this response" by **comparing it against other responses to the same prompt**. Generate G responses (typically 8), score them all, compute the mean and standard deviation, and set `A_i = (R_i − mean)/std`. **Why this can replace a critic:** the value model's only job was to answer "what reward should I expect from this state?" so that the advantage — how much better an action was than expected — could be computed. A learned critic is an *estimate* of that expectation, carrying its own approximation error and requiring its own training. Sampling multiple responses to the same prompt gives an **empirical** estimate of the same quantity, drawn from actual behaviour, which is both unbiased and free of separate training. **The saving is substantial:** models in memory drop from 4 to 3, hyperparameters from 12+ to about 5, training becomes stable, and the group can be generated in parallel. It works precisely when reward is cheap to compute, which is why GRPO and RLVR are described as designed for each other.

**19.** **The cascade:** the model produced no XML tags, so `xmlcount = 0`; without tags there was no `<reasoning>…<answer>` structure, so `soft_format = 0`; with no structure there was certainly no exact match, so `strict_format = 0`; with no parseable answer field the extracted value was not even an integer, so `int_reward = 0`; and with nothing extractable, the answer never matched ground truth, so `correctness = 0`. All rewards being zero made `reward_std = 0`, hence zero advantage for every generation, hence **no gradient** and frozen weights — `kl` stayed at 0.00 for all 18 steps because the policy never moved. **The three signals:** `correctness_reward > 0` asks whether the model is getting maths right at all; `reward_std > 0` asks whether generations differ from each other, which is the actual requirement for a learning signal, since GRPO learns from *variance* rather than level; and `xmlcount_reward > 0` asks whether the model produces any structure at all, making it the earliest warning because it is the most forgiving reward in the stack. **All three must be positive**, and `xmlcount` is the canary — if the most lenient reward is zero, everything downstream is guaranteed to be zero too.

**20.** **Stage 1 — establish the capability, probably with SFT.** First check whether the base model ever produces valid SQL against your schema, even occasionally: sample 100 generations and count. If it essentially never does, RL cannot help, because there is nothing to amplify — this is exactly the SmolLM lesson. So begin with **SFT** on a few thousand (natural-language question, correct SQL) pairs over your actual schema, teaching both the query patterns and your table and column names. **Stage 2 — RLVR + GRPO, because SQL is beautifully verifiable.** Unlike helpfulness or style, correctness here is programmatic: execute the generated query against a test database and compare the result set to the expected one. Build a graduated reward stack rather than a single binary check, so the model receives signal before it is fully correct — for example, does it parse as valid SQL, does it execute without error, does it reference only real tables and columns, and does the result set match — with **result-set correctness weighted highest** and syntactic rewards kept as small nudges, mirroring the 1.0-versus-0.1 ratio. Use GRPO with G=8, since running a query takes milliseconds and cheap verification is precisely what makes G=8 affordable. **Stage 3 — guard against reward hacking**, which survives even deterministic rewards. A model can pass a weak verifier without being right: returning a hardcoded constant that matches on the test row, or `SELECT *` where a projection was wanted. Defend with adversarial test cases, multiple databases with different data so the answer cannot be memorised, and a KL penalty to the reference. **Optionally add DPO** if the queries are correct but poorly written — unreadable, unnecessarily nested, or ignoring house conventions — since style is subjective and needs preference pairs rather than a verifier. The overall sequence follows the course's own recipe: build the capacity, teach the format, then amplify the capability.

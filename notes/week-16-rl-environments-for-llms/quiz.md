# Week 16 — Quiz (20 questions)

**Topic:** RL Environments for LLMs — the dataset/rollout/rubric trinity
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** An environment consists of:
- A) Model + optimizer + dataset
- B) Dataset + rollout + rubric
- C) Policy + reward model + critic
- D) Prompt + response + score

**2.** The four uses of a single environment artifact are:
- A) Training, inference, deployment, monitoring
- B) Evaluation, RL training, synthetic data generation, agent harness experimentation
- C) Pretraining, CPT, SFT, RLHF
- D) Parsing, scoring, logging, sharing

**3.** In the LLM-RL mapping, the *policy* is:
- A) The rubric
- B) The dataset
- C) The LLM itself
- D) The rollout strategy

**4.** Which is NOT a listed rollout type?
- A) Single-turn
- B) Multi-turn
- C) Tool-use
- D) Gradient-accumulated

**5.** The reward hacking story involved a model that learned to:
- A) Return empty responses
- B) Start every answer with "Great question!"
- C) Copy the prompt verbatim
- D) Always refuse

**6.** INTELLECT-3 was trained using:
- A) OpenEnv + TorchForge
- B) Verifiers + Environments Hub + prime-rl
- C) TRL + PEFT
- D) A proprietary closed stack

**7.** OpenEnv's API style follows:
- A) The OpenAI chat completions format
- B) Gymnasium: `step()`, `reset()`, `close()`
- C) REST with JSON schemas
- D) gRPC streaming

**8.** The "#1 footgun in RL training" is:
- A) Learning rate too high
- B) A rubric that conflates extraction and scoring
- C) Group size too small
- D) Missing KL penalty

**9.** In the notebook's failure, the model's answer was:
- A) Wrong, and correctly scored zero
- B) Correct but wrapped in markdown fences, and scored zero
- C) Correct and scored correctly
- D) Empty

**10.** GRPO in three lines is:
- A) Train reward model, run PPO, apply KL
- B) Sample N rollouts, score with the rubric, favor those above the group average
- C) Generate, filter, retrain
- D) Compute advantage from a critic, clip, update

**11.** Multi-turn rollouts are described as where:
- A) Benchmarks live
- B) Real RL training and agents live
- C) Evaluation is cheapest
- D) Rubrics are unnecessary

**12.** Which claim about verifiable rewards does the lecture make?
- A) They are slower than LLM judges
- B) They are unhackable, whereas everything else you must fight
- C) They only work for single-turn tasks
- D) They require a reward model

---

## Short answer

**13.** Explain the "build it once, use it four ways" thesis and why it matters in practice.

**14.** Map the classical RL vocabulary onto LLM-RL, and explain what "environment" concretely means here.

**15.** Describe the three rubric types with a use case for each, and explain the trade-off.

**16.** Explain the "Great question!" story and why LLM judges are especially vulnerable to this.

**17.** Explain "parse first, score second." Why does conflating them fail *silently*, and how does it connect to Week 15's zero cascade?

**18.** Explain why environments being portable across trainers matters.

**19.** Discussion question 3 from the slides: what changes if your rubric must run in 100 ms versus 10 s? Connect this to GRPO.

**20.** Design an environment for "the model should write good commit messages for a diff." Specify dataset, rollout, and rubric, and be honest about the hard part.

---
---

## Answer key

**1. B** — Dataset + rollout + rubric. Three pieces.

**2. B** — Evaluation, RL training, synthetic data generation, and agent harness experimentation.

**3. C** — The LLM itself. Its weights are what optimization changes.

**4. D** — Gradient accumulation is a training technique, not a rollout type. The four are single-turn, multi-turn, tool-use, and sandboxed.

**5. B** — Every answer began with "Great question!" The judge's score rose while capability did not.

**6. B** — Verifiers + Environments Hub + prime-rl, on 512 H200s over roughly two months.

**7. B** — Gymnasium-shaped: `step()`, `reset()`, `close()`, with a Docker container per environment.

**8. B** — A rubric doing two jobs at once, extraction and scoring.

**9. B** — Correct, but wrapped in markdown fences that the rubric's extraction did not anticipate.

**10. B** — Sample N rollouts for the same prompt, score them all with the rubric, and update the policy to favour those above the group average. No reward model, no baseline network.

**11. B** — Agents and real RL training; single-turn is for benchmarks.

**12. B** — Verifiable rewards are unhackable; everything else you fight.

**13.** The claim is that the artifact defining a task — its inputs, how the model interacts with it, and how outputs are scored — is **identical** whether you are evaluating a model, training it with RL, generating synthetic data, or experimenting with an agent harness. All four need the same three pieces, so they should be one object rather than four. **Why it matters in practice:** most teams build an eval harness, a separate training pipeline, and a separate data generator, each encoding the same task definition in different code. Those definitions **drift**: the eval's answer-extraction diverges from the trainer's, so eval numbers stop predicting training behaviour and neither reflects what the data generator produced. Unifying them means one definition to maintain and debug, one place to fix a parsing bug, and — via the Environments Hub — an artifact that can be shared and reused by others.

**14.** **State** = the conversation so far; **Action** = the next token or full response; **Environment** = whatever sends back the next observation and reward; **Reward** = the rubric score on the rollout; **Policy** = the LLM itself. **What "environment" concretely means:** it is not a simulated world in the classical sense but the **code you write** that receives the model's output and returns the next observation plus a score. For a single-turn maths task it may just be a scorer; for a tool-use task it executes the requested API call and returns the result; for a sandboxed coding task it runs the code and returns stdout, stderr, and test results. It is defined entirely by what it hands back to the model, which is why it decomposes into dataset (what to present), rollout (how interaction proceeds), and rubric (how to score).

**15.** **Verifiable** — exact match, test passes, or a regex hit; use for maths, code, JSON validity, or anything with programmatic ground truth. **LLM judge** — another model grades the output; use where correctness is a matter of judgment, such as tone, helpfulness, or writing quality. **Hybrid** — several rubrics combined with weights; use where a task has both an objective core and subjective qualities, for instance code that must pass tests *and* be readable. **The trade-off is coverage versus hackability.** Verifiable rubrics cannot be gamed by rhetoric — no phrasing makes `int(answer) == expected` true when the answer is wrong — but they only apply where a program can decide correctness. LLM judges apply almost anywhere but are models with exploitable failure modes, so optimizing against them invites reward hacking. Hybrids buy coverage at the cost of reintroducing a hackable component, which is why verifiable terms should carry the dominant weight.

**16.** A model was trained to maximise an LLM judge's helpfulness score. After 1,000 steps, **every answer began with "Great question!"** — the score had risen substantially while actual capability had not moved. The model found a cheap token pattern that reliably increased the judged score without improving the response, a textbook instance of Goodhart's Law. **Why LLM judges are especially vulnerable:** they are trained on, or prompted to imitate, human preference judgments, and therefore **inherit human raters' biases** toward flattery, confident phrasing, length, and superficial markers of effort. Those biases are systematic rather than random, so gradient descent finds them reliably and exploits them consistently. A verifiable rubric has no such surface: it evaluates a property of the output, not an impression of it. This is the concrete meaning of "verifiable rewards are unhackable; everything else, you fight."

**17.** **The principle:** extracting the answer from raw model output and judging whether that answer is correct are two distinct jobs, and they belong in two components — a `Parser` that pulls structure out of messy text, and a `Rubric` that scores the extracted value. In the notebook a **correct** answer wrapped in markdown fences scored **zero**, because the rubric's extraction logic did not anticipate ``` fences. The bug was in the harness, not the model. **Why it fails silently:** a brittle rubric does not raise an error — it returns 0, which is indistinguishable from a model that genuinely cannot do the task. You then chase the wrong problem, tuning hyperparameters, adding data, or swapping models, while the real fault is a regex. **Connection to the zero cascade:** in GRPO the advantage is `(R_i − mean)/std`, so if every rollout in a group scores zero there is no variance, no advantage, and therefore **no gradient at all**. A parsing bug does not merely mis-report the metric; it freezes training exactly as the SmolLM run's all-zero table did. Separating parser from rubric makes the parser independently unit-testable against real messy outputs, without running the model.

**18.** Because it **decouples task definition from optimization**. The environment specifies what the task is — the data, the interaction pattern, and how success is measured — while the trainer implements how gradients are computed and applied, and these change for entirely different reasons. Portability means you can swap the trainer as your compute situation changes, moving from TRL on a single GPU to prime-rl or verl for distributed training, or to Tinker for hosted LoRA RL on remote GPUs, **without rewriting the task**. It also means the environment becomes a **shareable artifact**: because it does not embed one framework's assumptions, it can be published to the Environments Hub and installed with `prime env install`, then evaluated, used as a data engine, or trained on by anyone. That is what makes "pip-for-environments" possible, and why Verifiers adding OpenEnv interop is described as complementary rather than competitive — a common interface makes the commons work.

**19.** Rubric latency **directly bounds group size and therefore training throughput**. GRPO scores G rollouts per prompt per step, so with the sweet-spot G=8 a **100 ms** rubric costs 0.8 seconds of scoring per prompt — negligible beside generation — whereas a **10 s** rubric costs **80 seconds per prompt**, and scoring, not generation or gradient computation, becomes the bottleneck. The consequences compound: you are pushed toward smaller groups such as G=4 or G=2, which makes the group-normalized advantage noisier and harder to distinguish good rollouts from average ones, degrading the learning signal itself. **This is exactly why Week 15 described RLVR and GRPO as "designed for each other":** GRPO needs a cheap reward because it scores G rollouts per prompt, and a programmatic verifier answers in microseconds. A slow rubric — typically an LLM judge, or a test suite that spins up containers — pushes you toward caching results, parallelising scoring, using a cheap rubric during training and an expensive one only for evaluation, or accepting a smaller G with correspondingly worse gradients.

**20.** **Dataset:** pairs of `(diff, reference commit message)` mined from repositories with genuinely good commit hygiene — the Linux kernel, well-maintained OSS projects — filtered to exclude "fix", "wip", and merge commits. Store metadata alongside each: repository conventions, whether the project uses Conventional Commits, and the diff's size and file types. **Rollout:** single-turn is the obvious start — diff in, message out. But **multi-turn is more valuable**, letting the model request additional context such as the file's history, the surrounding code, or the linked issue, since a human writing a good message does exactly that. This is the difference between benchmarking and training an agent. **Rubric — hybrid, and this is where the honesty is required.** The verifiable components are genuinely easy: subject line under 72 characters, imperative mood detectable by a rule, Conventional Commits prefix validity, a blank line before the body, and no mention of files the diff does not touch. These are cheap regex checks, they run in milliseconds, and they are unhackable — but **they only measure form**. **The hard part is that "good" is mostly about content, which is not verifiable.** A good commit message explains *why* the change was made, information that frequently does not appear in the diff at all. Similarity to the reference message is a poor proxy, since many different messages are equally good and the reference may itself be mediocre. That leaves an LLM judge for the substantive dimension — and, per the "Great question!" story, that judge is hackable, likely rewarding verbosity or confident phrasing. **The defensible design** weights the verifiable form checks as a small nudge, roughly the 0.1-versus-1.0 ratio from Week 15, keeps a judge for content but audits it against held-out human ratings, and accepts that this is fundamentally a DPO-shaped problem — preference pairs over commit messages — rather than a clean RLVR one. Slide 25's second discussion question, on handling non-verifiable rewards, is precisely this problem, and it does not have a tidy answer.

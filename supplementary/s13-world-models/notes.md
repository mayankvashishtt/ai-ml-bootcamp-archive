# S13 — World Models: Learning to Imagine What Happens Next

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. The course teaches models that predict **tokens**. A world model predicts **what happens to a world** — and that difference sits underneath several things the course does touch: Week 16's RL environments are hand-built worlds, Week 25's computer-use agent is an agent acting in a world it must understand, and the whole question of whether LLMs "really understand anything" is, at bottom, a question about world models.

**Fills the gap after:** Week 16 (RL Environments), Week 25 (Computer Use Agents)
**Prerequisites:** Weeks 14–16 (RL, rewards, environments). S9 (vision) helps for §7.

---

## 0. The idea in plain language

Here's the thing your brain does constantly and effortlessly.

You're holding a glass. You let go. **You know it falls** — you didn't have to drop a glass to find out. You ran a little simulation in your head: glass, no support, gravity, floor, probably shatters. You can also simulate outcomes you've never experienced: what if you dropped it onto a mattress? Into water? From a plane?

That internal simulator is a **world model**: a compressed, predictive representation of how things work, good enough to answer **"what happens if…?"** without doing it.

**Why this matters enormously for AI:** an agent with a world model can *think before acting*. It can try ten plans in imagination, discard the nine that fail, and execute the one that works. An agent without one has to try things in reality — which is slow, expensive, and sometimes irreversible.

Compare:

| | No world model | With world model |
|---|---|---|
| How it learns | Try in the real world, observe, repeat | Try in imagination, then act |
| Cost per attempt | Real time, real money, real damage | Nearly free |
| Novel situations | Must experience them | Can simulate them |
| Planning | Reactive — one step at a time | Can look ahead |

**Formally:** a world model learns the **transition function** — given the current state and an action, what's the next state?

```
next_state  ≈  f(current_state, action)
```

That's it. Everything in this lecture is about how you learn `f`, what "state" should mean, and what goes wrong.

---

## 1. Where this fits with what you already know

**Week 14–16 gave you reinforcement learning.** An agent takes actions, receives rewards, and learns a policy. But there are two fundamentally different ways to do that:

**Model-free RL** — learn the policy directly from experience. "In this situation, this action tends to work." No understanding of *why*, no ability to predict consequences. This is what most LLM RL does (PPO, GRPO in Week 15).

**Model-based RL** — first learn how the world works, then use that model to plan. Slower to set up, dramatically more **sample-efficient**, and capable of planning for situations never encountered.

The trade-off is sharp and worth memorising:

| | Model-free | Model-based |
|---|---|---|
| Learns | Policy directly | Dynamics, then plans |
| Sample efficiency | Poor — needs millions of trials | Much better |
| Compute per decision | Cheap | Expensive (planning) |
| Fails when | Data is scarce or costly | The learned model is wrong |
| Example | PPO, GRPO (W15) | Dreamer, MuZero |

**Why sample efficiency is the whole point:** in a video game you can run a million episodes overnight. On a real robot, a million trials means years and a destroyed robot. In a real production system, a million trials means a million user-visible mistakes. **Whenever real experience is expensive, model-based methods win** — and that's most of the interesting world.

---

## 2. What "state" should mean — the key design decision

Naively, the state is what you observe: the raw pixels of the screen, the full sensor readout.

That's a terrible representation to predict. Pixels are enormous, mostly redundant, and full of detail that doesn't matter. Predicting the exact texture of a wall is wasted effort if you only need to know that the wall is there.

So essentially every serious world model learns a **latent state**: a compact vector capturing what matters for prediction, and discarding what doesn't.

```
observation (huge, noisy)  →  encoder  →  latent state (small, meaningful)
                                              ↓
                          latent state + action  →  next latent state
```

**This is the central design insight, and the source of the field's biggest disagreement:** what should the latent state be *sufficient for*?

- **Sufficient for reconstruction** — you can regenerate the observation from it. Intuitive, easy to train, but forces the model to encode irrelevant detail like wall textures.
- **Sufficient for prediction and control** — you can predict future rewards, values, and what your actions will do, but *not* reconstruct the image. Much more efficient; you keep only what affects outcomes.

The field has moved decisively toward the second. Which brings us to the actual systems.

---

## 3. The classic: Ha & Schmidhuber's "World Models" (2018)

The paper that named the field, and it's beautifully simple — three components:

**V (Vision)** — a variational autoencoder compressing each video frame into a small latent vector. "What does the screen look like right now, in 32 numbers?"

**M (Memory)** — a recurrent network (MDN-RNN) that predicts the *next* latent given the current latent and action. "Given where I am and what I do, where will I be?" It predicts a *distribution*, not a point, which matters because the world is stochastic.

**C (Controller)** — a tiny policy mapping state to action. Deliberately tiny — most of the intelligence lives in V and M.

**The famous result:** they trained the controller **entirely inside the model's own dream.** The agent never touched the real environment during policy learning — it learned to drive a car and play a game by practising in its own hallucinated simulation, then transferred that policy back to reality successfully.

That's the thesis of the whole field in one experiment: **if your model of the world is good enough, you can learn in imagination.**

---

## 4. Dreamer: learning by imagining

The **Dreamer** line (Hafner et al., building on PlaNet) made this practical and general.

The core component is the **RSSM (Recurrent State-Space Model)**, which splits the latent state in two:
- A **deterministic** part carrying reliable history
- A **stochastic** part capturing genuine uncertainty about what happens next

That split matters because some things are predictable and some genuinely aren't, and forcing one representation to do both jobs does neither well.

**The training loop:**
1. Act in the real environment, collect experience
2. Train the world model to predict transitions and rewards
3. **Imagine** long rollouts entirely inside the latent space — no environment interaction
4. Train the policy on those imagined trajectories
5. Repeat

Step 3 is why this is cheap. Imagined rollouts happen in a small latent space — no rendering, no physics engine, no real robot. You can generate enormous amounts of training experience from a modest amount of real experience.

**DreamerV3** (2023) is the headline result: **one algorithm with fixed hyperparameters** across wildly different domains — continuous control, Atari, 3D navigation — and notably, it **collected diamonds in Minecraft from scratch without human data**, a long-standing challenge requiring a long chain of dependent subgoals. The "fixed hyperparameters" part is the underrated achievement, because RL's reputation for fragility is largely a hyperparameter-tuning story.

---

## 5. MuZero: don't model the world, model what matters

**MuZero** (Schrittwieser et al., 2020) makes the sharpest version of §2's argument.

Predecessor **AlphaZero** mastered chess and Go using a *given* simulator — the rules were programmed in. That's fine for board games and useless for the real world, where nobody hands you the rules.

MuZero learns the dynamics itself — but with a crucial twist: **it never tries to predict observations at all.** It learns a latent dynamics model trained only so that it correctly predicts three things:

1. The **reward**
2. The **value** (expected future reward)
3. The **policy** (which actions are good)

The latent state has **no requirement to correspond to anything visible.** It doesn't reconstruct the board; it doesn't have to. It only has to be sufficient for planning.

**Why that's a big idea:** the model is free to encode exactly what's decision-relevant and ignore everything else. It learns a representation shaped by usefulness rather than by appearance — and it worked, matching AlphaZero on Go, chess, and shogi *without being told the rules*, while also handling Atari.

**JEPA** (Yann LeCun's Joint Embedding Predictive Architecture, with I-JEPA and V-JEPA as instances) pushes the same argument philosophically: predicting in **representation space** rather than pixel space is the right move, because pixel-level generative prediction spends enormous capacity on unpredictable detail that doesn't matter. Whether this is *the* path forward is genuinely debated — but the underlying critique is sound and worth understanding.

---

## 6. Where world models actually get used

**Robotics** — the flagship case. Real-robot trials are slow, expensive, and break hardware. Learn a world model from limited real data, train policies in imagination, deploy. The persistent difficulty is the **sim-to-real gap**: policies exploit inaccuracies in the model that don't exist in reality (§8).

**Autonomous driving** — predicting how the scene evolves is the core task, and dangerous scenarios cannot be collected by driving into them. Simulation is a safety requirement, not an optimisation.

**Game playing** — MuZero's domain, and the classic proving ground because ground truth is cheap.

**Video generation as world simulation** — the most-hyped and most-contested application (§7).

---

## 7. Video models: are they world models?

Modern video generators produce strikingly coherent footage, and they've been described — including by their makers — as **world simulators**. This claim deserves care, because it's partly right and frequently overstated.

**The case for:** to generate temporally consistent video, a model must encode a great deal about how things move, occlude, persist, and interact. Objects staying solid when the camera pans is not free — something in there represents object permanence. **Genie** (DeepMind) is a sharper version of the claim: it learned **interactive** environments from unlabelled internet video, inferring a latent action space without action labels — you can *act* in the generated world, which is much closer to a real world model than passive generation.

**The case against, stated plainly:**

- **Visual plausibility is not physical correctness.** Generated video routinely violates physics in ways that look fine at a glance — objects merging, appearing, changing count, or passing through each other. The model was trained to produce plausible *pixels*, and it optimises for exactly that.
- **A world model you can't query is of limited use.** Prediction alone isn't enough; usefulness comes from being able to ask "what if I do X?" and act on the answer. That requires an action-conditioned interface, which passive video generation doesn't have (and which is exactly what Genie adds).
- **Consistency degrades over long horizons** — the compounding-error problem (§8).
- **Appearance-level modelling wastes capacity**, which is JEPA's critique from §5.

**A fair summary:** video models clearly learn *something* about world dynamics — that's not in serious dispute. Whether that constitutes a world model in the useful sense — a queryable, action-conditioned, physically-reliable simulator you can plan with — is **an open research question, not a settled fact**. Treat confident claims in either direction with suspicion.

---

## 8. What goes wrong

This section is the practical heart of the lecture, because these failures are why world models remain hard.

### Compounding error

A world model predicts one step with small error. Feed that slightly-wrong prediction back in to predict step two, and the error grows. Iterate 50 times and you're simulating a world that doesn't exist.

```
step 1:  ~correct
step 5:  drifting
step 20: plausible-looking fiction
step 50: unrelated to reality
```

This is the fundamental limit on planning horizon, and it's why **short imagined rollouts followed by real observation** beats long imagined rollouts. Mitigations include training on multi-step prediction directly (not just single steps), keeping horizons short, and re-grounding in reality frequently.

**You have already seen this exact pattern**: it's why long agent trajectories drift (Week 17), and why an agent that verifies against reality every few steps outperforms one that plans twenty steps ahead. **Grounding beats planning depth.** The world-model literature just states it mathematically.

### Model exploitation — the Goodhart problem again

This one is delicious and important. When you train a policy inside a learned model, **the policy will find the model's bugs.**

If the learned physics has a glitch where pushing a wall at a certain angle yields infinite reward, the policy finds it — reliably. It becomes superb at exploiting a simulation flaw and useless in reality.

**This is exactly Week 14's reward hacking**, one level deeper. There, the agent gamed a flawed *reward model*. Here it games a flawed *world model*. Same Goodhart's Law: **optimise hard against an imperfect proxy and you get the proxy's flaws, amplified.**

The standard defences are instructive: use **ensembles** of world models and trust only what they agree on; be **pessimistic** where the model is uncertain; and keep imagined horizons short so there's less room to find exploits.

### Partial observability

The agent sees observations, not states. A closed box's contents affect the future but aren't visible. This is what recurrent/memory components are for — accumulating history into something closer to a true state — and it's the same reason agents need memory (Week 18).

### Stochasticity

Some things are genuinely unpredictable. A model that predicts a single future is wrong about a world that has many. This is why serious world models predict **distributions** — Ha & Schmidhuber's MDN, Dreamer's stochastic latent component.

### The exploration problem

You can only learn dynamics for states you've visited. The model is confidently wrong about everything else — and a planner that trusts it will happily plan straight into the unknown region where the model is fiction.

---

## 9. Do LLMs have world models?

The question everyone actually wants answered, and it deserves a careful answer rather than a slogan.

**The sceptical position:** an LLM is trained to predict the next token. It has never seen, touched, or acted in a world. It learns statistical regularities of text, and apparent understanding is sophisticated pattern-matching over a very large corpus.

**The other position:** predicting text well *requires* modelling what the text is about. To predict the end of a murder mystery you need to track who was where. To predict the next line of a chess game you benefit from representing the board. The claim is that **world models emerge as a side effect of the prediction objective**, because they're the efficient way to do the job.

**What the evidence actually shows — this is where it gets interesting:**

**Othello-GPT** (Li et al., 2023) is the cleanest result. Train a GPT purely on sequences of Othello moves — no board, no rules, no images, just move lists. Then:
- **Linear probes recover the board state** from internal activations. The model built a representation of the board nobody gave it.
- More importantly, **interventions on that representation causally change the model's outputs** in the predicted way. Edit the internal board, and the model's move predictions change to match the edited board.

That second part is what elevates it from correlation to something stronger: the representation isn't a bystander, it's **used**. (A follow-up by Neel Nanda found the representation is linear in terms of "my pieces vs their pieces" rather than "black vs white" — a nice reminder that these representations are real but not necessarily in the form we'd expect.)

**Gurnee & Tegmark (2023)** found LLMs contain linear representations of **space and time** — probes recover geographic coordinates and historical dates from internal activations.

**What this justifies concluding:** models trained on next-token prediction **do** build internal representations of the domains their data describes, and those representations are **causally used**, not decorative. "Just statistics" is too dismissive as a description of what's happening mechanically.

**What it does not justify:** concluding that LLMs have rich, reliable, general world models. Othello is a tiny closed domain with perfect information. The representations found in LLMs are **partial, inconsistent, and frequently wrong** — which is precisely why models confidently produce physically impossible scenarios, lose track of object state across a long story, and fail at multi-step spatial reasoning. A representation that exists and is used can still be bad.

**The honest position — and this is the one to hold:** LLMs have *fragments* of world models — real, causally-used internal representations that are patchy and unreliable. This is not the same as having none, and not the same as having a robust simulator. Anyone claiming either extreme with confidence is ahead of the evidence.

---

## 10. Why this matters for the agents you build

Concrete implications, not philosophy:

**Agents plan with an implicit, unreliable world model.** When an agent decides "I'll edit this file, then run the tests, then commit," it's predicting consequences. Those predictions come from its patchy internal model — which is why plans that look coherent fail on contact with reality.

**Therefore: ground frequently, plan shallowly.** This is §8's compounding error in practice, and it's Week 17's harness lesson with a mechanism attached. An agent that acts and observes every step outperforms one that plans ten steps ahead, because the ten-step plan is built on a rollout that drifted.

**Tools are the substitute for a good world model.** An agent can't reliably predict what a command outputs — so it runs the command. **Every tool call is a step where the agent replaces prediction with observation.** That reframing explains why good tools matter so much: they're not conveniences, they're corrections to an unreliable simulator.

**Week 16's RL environments are hand-built world models.** When you write an environment with explicit state, transitions, and rewards, you're supplying the world model the agent doesn't have to learn. The reason environment design is hard is the same reason learned world models are hard — capturing what matters, in a form that supports planning, without exploitable flaws.

**Week 25's computer-use agents are the sharpest case.** The agent must predict what a click does. Its model of the UI is learned from training data and is often wrong — dialogs appear, pages load slowly, state changes unexpectedly. **The fix is not better prediction, it's verification**: screenshot after every action. Replace the world model with observation.

**And the safety angle (S4):** an agent that plans in imagination and acts in reality is only as safe as its model. Where the model is wrong, the plan is wrong, and the actions are real. Approval gates on consequential actions exist precisely because the agent's world model cannot be trusted at the moment it matters most.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 14** (RLHF, reward hacking) | Model exploitation is reward hacking one level deeper — Goodhart against a world model instead of a reward model. |
| **Week 15** (RLVR, GRPO) | These are model-free: they learn policies directly from sampled outcomes without a dynamics model. |
| **Week 16** (RL environments) | A hand-built environment *is* a world model. Environment design is hard for exactly the reasons learned models are hard. |
| **Week 17** (harness, evals) | Compounding error explains trajectory drift, and why grounding beats planning depth. |
| **Week 18** (memory) | Partial observability is why agents need memory — accumulating history toward a usable state. |
| **Week 25** (computer use) | Agents predict what a click does, often wrongly. Verification substitutes for a reliable model. |
| **Week 20** (reading papers) | "Video models are world simulators" is the perfect case for the scepticism this lecture trains — an interesting claim, weaker evidence than the phrasing implies. |
| **S4** (safety) | An agent plans in imagination and acts in reality; where the model is wrong, real harm follows. |
| **S9** (vision) | Video world models rest on the visual representations covered there. |
| **S12** (multi-agent) | Ensembles and disagreement-as-uncertainty is the same defence as adversarial verification. |

---

## Key takeaways

1. **A world model predicts what happens next** — `next_state ≈ f(state, action)`. It's the internal simulator that lets you know a dropped glass falls without dropping one.
2. **Model-based RL learns dynamics then plans; model-free learns the policy directly.** Model-based is far more sample-efficient, which matters whenever real experience is expensive — robots, production systems, anything irreversible.
3. **Latent state is the key design decision.** Predict in a compact learned space, not in pixels.
4. **The field moved from "sufficient for reconstruction" to "sufficient for control."** MuZero's model predicts reward, value, and policy — and never reconstructs an observation.
5. **Ha & Schmidhuber trained a policy entirely inside the model's dream** and transferred it to reality — the field's thesis in one experiment.
6. **DreamerV3 generalised it**: one algorithm, fixed hyperparameters, many domains, including Minecraft diamonds from scratch.
7. **Compounding error is the fundamental limit.** Small per-step errors iterate into fiction, which is why short rollouts plus frequent re-grounding beat deep planning.
8. **Model exploitation is Goodhart's Law again** — a policy trained inside a learned model will find and exploit the model's bugs. Ensembles, pessimism, and short horizons are the defences.
9. **Video models learn *something* about dynamics**, but visual plausibility isn't physical correctness, and "world simulator" is a contested claim, not a settled one.
10. **LLMs have real but patchy world models.** Othello-GPT showed internal board representations that are **causally used**, not decorative — while remaining partial and often wrong. Both extremes of the debate overreach.
11. **Agents plan using that unreliable model**, which is why grounding beats planning depth and why tools matter: **every tool call replaces prediction with observation.**
12. **Week 16's environments are hand-built world models** — and they're hard to build for the same reasons learned ones are hard to learn.

---

## Glossary

| Term | Definition |
|---|---|
| **World model** | Learned predictive model of environment dynamics: state + action → next state |
| **Transition function** | The dynamics mapping current state and action to the next state |
| **Model-free RL** | Learning a policy directly from experience, without modelling dynamics |
| **Model-based RL** | Learning dynamics, then planning or training a policy against that model |
| **Sample efficiency** | How much real experience is needed to reach a given performance level |
| **Latent state** | Compact learned representation replacing raw observations for prediction |
| **RSSM** | Recurrent State-Space Model; Dreamer's latent with deterministic + stochastic parts |
| **Imagination / latent rollout** | Simulating trajectories inside the model with no environment interaction |
| **MDN-RNN** | Mixture-density recurrent network predicting a distribution over next latents |
| **MuZero** | Model-based agent learning dynamics sufficient for reward, value, and policy only |
| **JEPA** | Joint Embedding Predictive Architecture; predicting in representation space, not pixels |
| **Genie** | Model learning interactive environments from unlabelled video via inferred latent actions |
| **Compounding error** | Per-step prediction errors accumulating over a rollout until it's fiction |
| **Model exploitation** | A policy discovering and exploiting flaws in a learned world model |
| **Sim-to-real gap** | Discrepancy between simulated dynamics and reality, breaking transferred policies |
| **Partial observability** | Observations not fully determining the underlying state; motivates memory |
| **Othello-GPT** | Model trained on move sequences shown to build a causally-used internal board representation |
| **Linear probe** | Simple classifier testing whether information is linearly recoverable from activations |

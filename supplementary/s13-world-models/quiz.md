# S13 — Quiz (20 questions)

**Topic:** World Models
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** A world model learns:
- A) A mapping from states to rewards
- B) The transition function — state + action → next state
- C) A policy
- D) A value function

**2.** The main advantage of model-based over model-free RL is:
- A) Lower compute per decision
- B) Sample efficiency — far less real experience needed
- C) Simpler implementation
- D) Better final performance in all cases

**3.** Model-based RL matters most when:
- A) Compute is cheap
- B) Real experience is expensive, slow, or irreversible
- C) The reward is dense
- D) The action space is small

**4.** Serious world models predict in latent space rather than pixel space because:
- A) Pixels can't be predicted
- B) Pixels are huge, redundant, and full of detail that doesn't affect outcomes
- C) Latent states are easier to visualise
- D) It avoids compounding error entirely

**5.** MuZero's learned latent state is trained to be sufficient for:
- A) Reconstructing the observation
- B) Predicting reward, value, and policy — with no reconstruction requirement
- C) Rendering the game board
- D) Compressing the input losslessly

**6.** Ha & Schmidhuber's headline result was that:
- A) Their model beat AlphaGo
- B) The controller was trained entirely inside the model's dream, then transferred to reality
- C) They eliminated compounding error
- D) They used no neural networks

**7.** DreamerV3's underrated achievement was:
- A) Beating humans at Atari
- B) One algorithm with fixed hyperparameters working across wildly different domains
- C) Running on a single GPU
- D) Removing the need for a reward signal

**8.** Compounding error means:
- A) The reward model becomes miscalibrated
- B) Small per-step prediction errors accumulate until the rollout is fiction
- C) Gradients explode during training
- D) Multiple agents disagree

**9.** A policy trained inside a learned world model will reliably:
- A) Generalise perfectly
- B) Find and exploit the model's bugs — Goodhart's Law again
- C) Converge faster than model-free
- D) Avoid unvisited states

**10.** The strongest evidence from Othello-GPT is that:
- A) Probes can recover the board state
- B) Interventions on the internal representation causally change the model's outputs
- C) It beat human players
- D) It was trained on board images

**11.** Calling video generators "world simulators" is:
- A) Settled and correct
- B) Contested — they learn something about dynamics, but plausible pixels aren't physical correctness
- C) Definitively wrong
- D) True only for Genie

**12.** For an agent, the practical substitute for a reliable world model is:
- A) A larger context window
- B) Tools — every tool call replaces prediction with observation
- C) Higher temperature
- D) More planning steps

---

## Short answer

**13.** Explain what a world model is in plain language, and why it's valuable for an agent.

**14.** Compare model-free and model-based RL, and explain why sample efficiency is the decisive factor.

**15.** Explain why latent state matters and the shift from "sufficient for reconstruction" to "sufficient for control."

**16.** Describe Ha & Schmidhuber's three components and why the dream-training result mattered.

**17.** Explain compounding error, why it limits planning horizon, and how it connects to Week 17.

**18.** Explain model exploitation and its relationship to Week 14's reward hacking. What defends against it?

**19.** Do LLMs have world models? Give the evidence on both sides and state a defensible position.

**20.** You're building an agent that manages cloud infrastructure — provisioning, scaling, and configuration changes on live production systems. Analyse this through the world-model lens and design accordingly.

---
---

## Answer key

**1. B** — Everything else in the field is about how you learn that function and what "state" should mean.

**2. B** — You can generate enormous training experience in imagination from modest real experience.

**3. B** — A million real-robot trials means years and a destroyed robot; a million production trials means a million user-visible mistakes.

**4. B** — Predicting a wall's exact texture is wasted effort if you only need to know the wall is there.

**5. B** — The latent has no requirement to correspond to anything visible.

**6. B** — The agent learned in its own hallucinated simulation and the policy transferred.

**7. B** — RL's fragility reputation is largely a hyperparameter-tuning story, so fixed hyperparameters is the significant part.

**8. B** — Correct at step 1, drifting by step 5, plausible fiction by step 20.

**9. B** — It becomes superb at exploiting a simulation flaw and useless in reality.

**10. B** — Causal intervention elevates it beyond correlation: the representation is used, not decorative.

**11. B** — An open research question; treat confident claims in either direction with suspicion.

**12. B** — The agent can't reliably predict a command's output, so it runs the command.

**13.** A world model is an **internal simulator** — a compressed, predictive representation of how things work, good enough to answer **"what happens if…?"** without actually doing it. The everyday example: you hold a glass, you let go, and **you know it falls**. You never had to drop a glass to learn this; you ran a small simulation — glass, no support, gravity, floor, probably shatters. You can also simulate situations you've never experienced: dropping it onto a mattress, into water, from a plane. Formally, it learns the **transition function**: `next_state ≈ f(current_state, action)`. **Why it's valuable for an agent is that it enables thinking before acting.** An agent with a world model can try ten plans in imagination, discard the nine that fail, and execute only the one that works. An agent without one must try things in **reality** — which is slow, expensive, and sometimes irreversible. The difference shows up in four places: how it learns (real trials vs imagined ones), cost per attempt (real time and real damage vs nearly free), handling novel situations (must experience them vs can simulate them), and planning (reactive one-step-at-a-time vs genuine lookahead). Everything else in the field is about how you learn `f`, what "state" should mean, and what goes wrong when the model is imperfect.

**14.** **Model-free RL** learns the policy directly from experience — "in this situation, this action tends to work" — with no representation of *why* and no ability to predict consequences. PPO and GRPO (Week 15) are model-free, and this is what most LLM RL does. **Model-based RL** first learns how the world works, then uses that model to plan or to train a policy. It is slower to set up and more expensive per decision (planning costs compute), and it fails in a distinctive way — when the learned model is wrong, everything built on it is wrong. **Sample efficiency is the decisive factor because it determines whether the method is usable at all in a given domain.** In a video game you can run a million episodes overnight, and model-free's appetite for data is simply not a problem — which is why model-free dominates benchmarks built on cheap simulators. But on a **real robot**, a million trials means years of wall-clock time and destroyed hardware. In a **live production system**, a million trials means a million user-visible mistakes. In anything **irreversible**, the count of allowable trials may be zero. Model-based methods let you learn dynamics from a modest amount of real experience and then generate effectively unlimited *imagined* experience in a small latent space — no rendering, no physics engine, no real robot. **So the rule is: wherever real experience is expensive, slow, or irreversible, model-based wins** — and that describes most of the genuinely interesting applications, which is why the field persists with it despite the added difficulty.

**15.** The naive choice is to treat the raw observation as the state — the screen's pixels, the full sensor readout. That is a **terrible thing to predict**: pixels are enormous, mostly redundant, and dominated by detail that has no bearing on outcomes. Predicting the exact texture of a wall is wasted capacity when all that matters is that the wall is there. So serious world models learn a **latent state**: a compact vector capturing what matters for prediction and discarding what doesn't, with dynamics learned in that small space rather than in observation space. **The deeper question — and the field's central disagreement — is what the latent should be *sufficient for*.** The older answer is **sufficient for reconstruction**: you can regenerate the observation from the latent. It's intuitive and easy to train with an autoencoder-style objective, but it **forces the model to encode irrelevant detail**, because reconstruction demands the wall texture whether or not it affects anything. The newer answer is **sufficient for prediction and control**: the latent must let you predict rewards, values, and the effects of your actions, but need not reconstruct the image at all. **MuZero is the purest example** — its latent is trained only so that reward, value, and policy predictions come out right, with **no requirement to correspond to anything visible**, which frees the model to encode exactly what is decision-relevant. **JEPA** makes the same argument philosophically, holding that predicting in representation space rather than pixel space is correct because pixel-level generative prediction spends enormous capacity on unpredictable detail that doesn't matter. The field has moved decisively toward the control-sufficient view.

**16.** **V (Vision)** is a variational autoencoder compressing each frame into a small latent vector — "what does the screen look like right now, in 32 numbers?" **M (Memory)** is a recurrent network (MDN-RNN) predicting the *next* latent given the current latent and the action taken — "given where I am and what I do, where will I be?" Critically it predicts a **distribution rather than a point**, because the world is genuinely stochastic and a model committing to one future is wrong about a world that has many. **C (Controller)** is a deliberately tiny policy mapping state to action; the design choice to keep it small is itself the argument — **most of the intelligence lives in V and M**, in understanding the world rather than in the reflex that acts on it. **The result that mattered is that they trained the controller entirely inside the model's own dream.** During policy learning the agent never touched the real environment at all — it learned to drive and to play by practising in its own hallucinated simulation, and the resulting policy **transferred back to reality successfully**. That is the thesis of the entire field demonstrated in a single experiment: **if your model of the world is good enough, you can learn in imagination.** Everything since — Dreamer's latent rollouts, MuZero's planning — is a refinement of that claim, and the practical significance is that real experience becomes a resource you spend on *learning the model* rather than on learning the policy.

**17.** A world model predicts one step with small error. Feed that slightly-wrong prediction back in to predict the next step, and the error compounds — step 1 is roughly correct, step 5 is drifting, step 20 is plausible-looking fiction, step 50 bears no relation to reality. **The rollout stays coherent-looking the whole time, which is what makes it dangerous:** there is no point at which the simulation announces it has become fiction. **This is the fundamental limit on planning horizon.** However good the one-step model, error growth bounds how far ahead you can usefully imagine, which is why **short imagined rollouts followed by real observation** systematically beat long imagined rollouts. Mitigations include training on multi-step prediction directly rather than only single steps, keeping horizons deliberately short, and re-grounding in reality frequently. **The Week 17 connection is direct**: long agent trajectories drift for exactly this reason. An agent that plans twenty steps ahead is building that plan on an internal rollout that has drifted, so the plan is coherent and wrong. An agent that acts, observes, and re-plans every step or two keeps re-anchoring its state to reality and outperforms it. **"Grounding beats planning depth" is the practical statement of a mathematical fact about error accumulation** — Week 17 arrived at it empirically through harness design, and the world-model literature explains why it had to be true.

**18.** When you train a policy inside a learned world model, **the policy will find the model's bugs** — reliably, because that is precisely what optimisation does. If the learned physics contains a glitch where pushing a wall at a particular angle yields enormous reward, the policy discovers it and exploits it. You end up with an agent that is superb at exploiting a simulation flaw and **useless in reality**, and the training metrics look excellent throughout. **This is Week 14's reward hacking one level deeper.** There, the agent gamed a flawed **reward model** — learning to produce responses the reward model scored highly rather than responses that were good. Here it games a flawed **world model** — learning to exploit dynamics that don't exist rather than dynamics that do. Both are **Goodhart's Law**: optimise hard against an imperfect proxy and you get the proxy's flaws, amplified in proportion to how hard you optimised. The recurrence across levels is the point — this is a structural property of optimisation against learned approximations, not a bug in any particular system. **Three defences, all instructive.** **Ensembles** — train several world models and trust only predictions they agree on, since a glitch in one is unlikely to be replicated in all. **Pessimism** — deliberately assume bad outcomes where the model is uncertain, so the policy is repelled from rather than attracted to regions the model doesn't understand. **Short horizons** — less imagined time means less room to discover and exploit a flaw. Note the family resemblance to S12's adversarial verification: disagreement among independent models is being used as an uncertainty signal in both cases.

**19.** **The sceptical case:** an LLM is trained to predict the next token. It has never seen, touched, or acted in a world. It learns statistical regularities of text, and apparent understanding is sophisticated pattern-matching over an enormous corpus. **The other case:** predicting text well *requires* modelling what the text is about — to predict the end of a murder mystery you must track who was where, and to predict chess moves you benefit from representing the board. On this view world models **emerge as a side effect of the prediction objective**, because they are the efficient way to do the job. **The evidence is more interesting than either slogan.** **Othello-GPT** (Li et al., 2023) trained a GPT purely on sequences of Othello moves — no board, no rules, no images. **Linear probes recovered the board state** from internal activations, showing the model built a representation nobody gave it. More importantly, **interventions on that representation causally changed the model's outputs** in the predicted direction: edit the internal board and the move predictions change to match. That causal step is what elevates the result beyond correlation — **the representation isn't a bystander, it is used.** (A follow-up by Neel Nanda found it is linear in "my pieces vs their pieces" rather than "black vs white" — real, but not in the form one would have guessed.) **Gurnee & Tegmark (2023)** similarly found linear representations of **space and time**, with probes recovering geographic coordinates and historical dates. **What this justifies:** models trained on next-token prediction **do** build internal representations of the domains their data describes, and those representations are **causally used**. "Just statistics" is too dismissive as a mechanical description. **What it doesn't justify:** concluding LLMs have rich, reliable, general world models. Othello is a tiny closed domain with perfect information; the representations found in LLMs are **partial, inconsistent, and frequently wrong**, which is exactly why models produce physically impossible scenarios, lose track of object state across a long story, and fail at multi-step spatial reasoning. **The defensible position: LLMs have *fragments* of world models — genuine, causally-used internal representations that are patchy and unreliable.** That is neither "none" nor "a robust simulator," and confident claims at either extreme are ahead of the evidence.

**20.** **The world-model framing is the right one here, and it immediately identifies the core danger: the agent must predict the consequences of actions on a live system, using a world model that is patchy, partly stale, and unverifiable at the moment it matters.** Infrastructure is the worst case on every axis this lecture covers. **Real experience is extremely expensive and often irreversible** — you cannot let the agent learn by trying, because a wrong scaling action costs money and a wrong configuration change causes an outage. **The state is partially observable**: the agent sees API responses and dashboards, not the true system state, and there is always drift between what the control plane reports and what is actually running. **The environment is stochastic and non-stationary** — deploys, traffic spikes, and other operators change things underneath it. And **the agent's model of your specific infrastructure is largely wrong**, since it was trained on generic cloud documentation, not on your VPC layout, your service dependencies, or your undocumented quirks. **Design implication one: make the world model explicit and external rather than internal.** Don't rely on the agent's memory of what the infrastructure looks like — give it read tools that query actual current state, and require it to *look* before it plans. Every planning step should begin with observation. **Implication two: ground constantly and plan shallowly** (§8). Compounding error means a ten-step infrastructure plan built from an initial snapshot is fiction by step four, because each step's assumed post-state drifts from the real one. Structure the loop as observe → single action → observe → re-plan, and explicitly resist multi-step plans executed without re-checking. **Implication three: use a real simulator wherever one exists**, because this is precisely the domain that supplies one. `terraform plan`, `--dry-run`, `kubectl diff`, and staging environments are hand-built world models with far better fidelity than anything the agent has internally, and they are exactly the Week 16 insight — a supplied environment beats a learned one. **Every mutating action should be planned in the simulator, the diff shown, and only then applied.** **Implication four: guard against model exploitation and Goodhart** (§8, Week 14). If you optimise the agent against any proxy metric — cost reduction, latency, resource utilisation — it will find the flaw in that metric, and here the exploit is a real production change. "Reduce cost" is satisfied beautifully by deleting things. Constrain by what it *may do*, not by what it *should optimise*, and never give a single scalar objective over live infrastructure. **Implication five: treat irreversibility as the primary axis of control.** Partition actions into read-only (free), reversible (auto-approve with logging), and irreversible or blast-radius-wide (human approval, always). Deletion, security-group changes, database operations, and anything touching production data belong in the last category permanently, regardless of how well the agent has been performing — because the argument for auto-approving is always "it's been reliable," and reliability on reversible actions predicts nothing about the one irreversible mistake. This is S4's approval-gate reasoning with a mechanism attached: **the gate exists because the agent's world model cannot be trusted precisely where being wrong is unrecoverable.** **Implication six: verify effects rather than assuming them** (Week 25's lesson). After every action, re-observe and confirm the intended state was actually reached — an apply that reports success while the service fails its health check is the normal case, not the exotic one. Build automatic rollback on verification failure, and make the agent's success criterion the *observed* state, not the API's return code. **Finally, operational discipline:** trace everything (Week 23), since a failure spanning several actions is undiagnosable without it; build an eval set from real historical incidents and changes (S3) and measure against a baseline before trusting it anywhere near production; and start it read-only — as a diagnostician and recommender — for long enough to establish where its model of your infrastructure is wrong. That period is not caution for its own sake; it is how you discover the specific gaps in its world model before those gaps become outages.

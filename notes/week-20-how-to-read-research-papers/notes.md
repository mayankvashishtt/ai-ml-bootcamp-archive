# Week 20 — How to Read Research Papers

**Subtitle:** The math, the triage, the bullshit detection
**Date:** 31/05/2026
**Source:** `downloads/week-20-how-to-read-research-papers.pdf` (64 slides) — *no Colab notebook this week*
**Notion page:** https://100xschool.notion.site/379ffffa33e580f491e7e437d87f8ce1

> This is the lecture Week 0 promised: *"how to be up to date, with new things coming everyday."* The course's answer is a **skill**, not a reading list.

---

## 0. The idea in plain language

Week 0 stated four goals, and the fourth was *"how to stay up to date when new things ship every day."* This is the lecture that answers it — and the answer is a **skill**, not a reading list. Reading lists expire; this doesn't.

**The framing that makes it tractable:**

> **90% of papers don't matter. 10% change how you build. The skill is identifying the 10% fast.**

You are not trying to read papers thoroughly. You're trying to **triage** them — spend two minutes deciding whether a paper deserves twenty. Most don't, and that's fine.

**Three things this lecture teaches, in increasing difficulty:**

**1. Triage — read in passes, not linearly.** Nobody reads a paper front to back on the first pass. You read the abstract, look at the figures, read the conclusion, and *then* decide whether to go deeper. Most papers stop at pass one. This alone changes your throughput by an order of magnitude.

**2. Bullshit detection.** The most valuable section, and it's specific rather than cynical. The recurring red flags:

- **Missing baselines** — the new method is compared against a weak or untuned version of the alternative. If someone reports a big win, ask *what exactly did they compare against, and was it tuned as carefully as their method?*
- **Cherry-picked benchmarks** — strong results on three tasks, silence on the ten where it lost.
- **No ablations** — a paper with five components and no analysis of which one mattered hasn't shown you what works.
- **Unreproducible setups** — no code, no hyperparameters, no seeds.

You've already met this pattern twice in the course without it being named: Week 10's PageIndex result (54% vs 98.7%) is against **naive** RAG, not advanced RAG. Week 11's RLM figures come from a citation that doesn't resolve. **Neither is dishonest; both need the question asked.**

**3. Surviving the maths.** The section that unblocks most people. The key reframe is that **notation is vocabulary, not intelligence** — you don't need to derive anything, you need to *read* it. Learn what Σ, 𝔼, ∇, argmax, and subscript conventions mean, and most ML equations become sentences rather than walls. The 5-step method in §6 is: identify what each symbol *is* (scalar? vector? distribution?), read the equation aloud in words, work a tiny numeric example, check the edge cases, and only then worry about why it's the right equation.

**Why this is a first-class deliverable rather than a bonus.** The specific techniques in this course have a half-life measured in months. LangGraph may be superseded; the decoding parameters in S7 already have been. **The ability to read a paper, extract the mechanism, and judge whether the evidence supports the claim is the thing that doesn't expire.** It's also, bluntly, what separates people who use AI tools from people who build them.

---

## Part 1 — Why papers, why now

- **The field moves in weeks, not years.**
- **Tweets are downstream of papers.**
- **The 10% rule.**

### The 10% rule

> - **90% of papers don't matter.**
> - **10% change how you build.**
> - **The skill is identifying the 10% fast.**

### What you're optimizing for

> - **Not "read every paper."**
> - **Triage > comprehension.**
> - **Intuition > rigor.**

This reframes the whole activity. The bottleneck isn't reading speed, it's **deciding what not to read** — and that's a different skill, closer to editorial judgment than to study.

---

## Part 2 — Anatomy of an ML paper

```
01 Abstract        05 Experiments / Results
02 Introduction    06 Ablations
03 Related work    07 Conclusion
04 Method          08 Appendix
```

| Section | What it is | How to treat it |
|---|---|---|
| **Abstract** | The whole paper in 200 words — claims + the headline number | **Read it first, always** |
| **Introduction** | Motivation, the gap, "here's what we did," contribution bullets | Read |
| **Related work** | **Author signaling** | **Usually skippable** — read only if you don't know the field |
| **Method** | The actual technique; **the math lives here** | **Read it after the results** |
| **Experiments** | Setup, datasets, baselines — **where setups get vague** | **Watch for what's missing** |
| **Results & ablations** | Headline tables and figures | **Ablations are the honest part**; cherry-picking lives here too |
| **Conclusion** | **Throwaway** | Skip |
| **Appendix** | **Where the real details hide** — hyperparameters, failure cases, extra ablations | Check |

**"Read the method after the results"** is the counter-intuitive instruction and it's right: the method is only worth understanding if the results justify the effort. Reading linearly means investing in the maths before knowing whether the paper matters.

### Where authors hide what

| What | Where |
|---|---|
| **Bad results** | Buried in the appendix |
| **Cherry-picked examples** | The main figures |
| **Missing baselines** | Silently absent |
| **Caveats** | The footnotes |

Note the asymmetry: **the appendix hides bad news, the main figures oversell good news.** So the appendix is where you look for problems, and the main figures are what you discount.

---

## Part 3 — The triage pass

> **Linear reading is for textbooks. Papers are reference docs. Jump to the signal.**

### The order

```
01 Abstract
02 Figures + captions
03 Results table
04 Method overview (skim)
05 Ablations
06 Then decide if you go deeper
```

Figures come second because a good paper's central claim is usually visible in one figure, and captions are written to be self-contained.

### The three-pass method (Keshav)

| Pass | Depth | Time |
|---|---|---|
| **1** | Skim | **10 minutes** |
| **2** | Details | **1 hour** |
| **3** | Reimplement | **4+ hours** |
| **Reality** | **Most papers stop at pass 1** | |

### The decision

> - **Does it change how I build?**
> - **Is the method novel, or repackaged?**
> - **Are the claims verifiable?**
> - **If no — close it.**

Explicit permission to stop. The failure mode for conscientious readers is finishing papers out of obligation.

---

## Part 4 — Red flags

> Single benchmark · Missing baselines · No code · Cherry-picked examples · Vague setup · **"We achieve SOTA."**

| Red flag | Why it matters | The question to ask |
|---|---|---|
| **Single-benchmark wins** | One leaderboard ≠ general capability | **Does it work on 3+ unrelated tasks?** |
| **Missing baselines** | No comparison to obvious priors. *"We outperform Llama-3"* — **at what size? what budget?** | **Apples-to-apples, or it doesn't count** |
| **No code** | **Cannot be verified** | Treat claims as **conjectures** — especially in industry papers |
| **Cherry-picked examples** | Main-figure prompts ≠ random samples | Look for *"we report random samples in the appendix."* **If absent — assume curation** |
| **Vague setup** | Missing hyperparameters; *"trained on a large corpus"* — **which one?** | **Reproducibility is an honesty signal** |
| **"We achieve SOTA"** | **SOTA is a marketing word** | **What's the actual delta?** Ablations matter more than the headline |

Two of these connect directly to earlier weeks. **"Single benchmark"** is exactly the caution to apply to Week 10's FinanceBench numbers (54% vs 98.7% on one benchmark of one document type). **Missing baselines** is the same critique: PageIndex was compared to *naive* RAG, not advanced RAG.

### What to skip without guilt

> Related work (usually) · Most proofs · Hyperparameter appendices · Author bios · Acknowledgements

### Skim / skip / study

| Skim | Skip | Study |
|---|---|---|
| Intro, method overview | Related work, most proofs | **Results, ablations, the key formula** |

---

## Part 5 — Math survival kit: notation

> **Pattern-match before parsing.**

### The ten notations

| Symbol | Meaning |
|---|---|
| **𝔼** | Expectations |
| **D_KL** | KL divergence |
| **∇** | Gradients |
| **argmax / argmin** | The input that maximizes / minimizes |
| **π / π_old** | Ratios |
| **p(y \| x)** | Conditionals |
| **Σ, Π** | Sums and products |
| **𝟙** | Indicators |
| **O( · )** | Big-O |
| **ℒ** | Lagrangians |

### Translations

| Notation | Plain English |
|---|---|
| `𝔼[X]` | *"the average of X over the distribution"* |
| `D_KL(P ‖ Q)` | *"how different are P and Q"* |
| `∇L` | *"direction of steepest increase of L"* |
| `argmaxₓ f(x)` | *"the x that maximizes f"* — **optimization is just chasing these** |
| `π(a\|s) / π_old(a\|s)` | *"how much has the policy changed"* |
| `p(y \| x)` | *"distribution of y given we saw x"* — **language modeling is all `p(token \| context)`** |
| `Σᵢ` | *"loop"* |
| `Πᵢ` | *"product loop"* |
| `𝟙[cond]` | 1 if true, 0 otherwise |
| `O(n²)` | *"scales quadratically with n"* |
| `ℒ = f + λ·g` | *"optimize f subject to g"* — **λ is the price of the constraint** |

> **All of them are for-loops in disguise.**

### Form matters — pattern-match first

| Shape | What it is |
|---|---|
| `θₜ₊₁ = θₜ − α·(·)` | **An optimization step** |
| `L = −log p(y\|x)` | **Cross-entropy loss** |
| `π / π_old` | **Policy ratio (PPO family)** |

**This is the highest-value idea in the section.** You don't need to parse an equation to recognise its *shape*. Seeing `θₜ₊₁ = θₜ − α·(…)` tells you "this is gradient descent, and the interesting part is whatever's in the parentheses" — which is 80% of the understanding for 5% of the effort.

Note `λ` as "the price of the constraint" — the same intuition as β in the KL penalty (Weeks 14–15). Every constrained optimization has a knob representing how much you're willing to pay.

---

## Part 6 — The 5-step method for math

> **About 30 minutes per formula. Worth it.**

### Step 0 — Breathe

> - **Overwhelm is normal.**
> - **You're not supposed to glance and get it.**
> - **Seniors feel this too.**

Worth stating explicitly. Most people conclude they're not smart enough when the actual issue is that *nobody* reads dense notation at a glance.

### Step 1 — Identify

- All formulas in the paper
- **Including ones referenced but not shown**
- Group them by logical block

### Step 2 — Move to paper

- **Off the screen**
- **Real pen, real paper**
- **Degrees of freedom matter**

Not a nostalgia point: you need to annotate, cross out, redraw, and write in the margins — operations a PDF viewer makes awkward.

### Step 3 — Translate

- **Read meaning, not symbol**
- **"Learning rate," not "alpha"**
- **Name every variable in your head**

### Step 4 — Edge cases

- **Set the knobs to 0 and 1**
- **What known algorithm do you recover?**
- **"Just X with one extra knob" is the insight**

**This is the single best technique in the lecture.** New methods are almost always old methods plus a term. Zeroing that term reveals the ancestor, which converts an unfamiliar equation into a familiar one plus a delta — and the delta is the actual contribution.

### Step 5 — Distill

- **One-sentence intuition per formula**
- **Not rigorous — logical**
- **Rubber-duck test: can you explain it?**

### Primitives → novelty

> - **Papers build vocabulary.**
> - **Section 5 ≠ Section 0, for a reason.**
> - **Lost in the method? Back up to the primitives.**

Confusion in section 5 usually means a definition from section 2 didn't land. Going *back* is faster than pushing forward.

---

## Part 7 — R1 + GRPO deep dive (worked example)

**Triage discussion questions:** What did the accuracy curve show? · The "aha moment" — **real, or marketing?** · Red flags you spotted? · What did you skip?

That second question is the right instinct to train. Week 15 presented the aha moment as a landmark result; here you're asked to interrogate it.

### R1-Zero vs R1

| | |
|---|---|
| **R1-Zero** | Pure RL on the base model |
| **R1** | Cold-start SFT → RL → SFT → RL |
| **The point** | **R1-Zero is the experiment, not the product** |

### The reward design

- Accuracy reward (**a verifier**)
- Format reward (`<think>...</think>`)
- **No reward model**
- **Why this is the whole story**

### Cold-start data

- Why R1-Zero alone wasn't shipped: **readability, language mixing**
- A few thousand examples
- **The SFT–RL sandwich**

*("Language mixing" — R1-Zero would switch languages mid-reasoning, since nothing in the reward penalised it. A detail Week 15 didn't mention, and a nice example of why reading the primary source pays.)*

### The GRPO formula — the 5-step method, live

**Steps 1–2:** pull the objective off the page, onto the board, list every symbol.

**Step 3 — Name every symbol:**

| Symbol | Meaning |
|---|---|
| **π_θ** | Current policy |
| **π_old** | Snapshot of the policy before the update |
| **Aᵢ** | Advantage for sample i |
| **ε** | Clip bound |
| **β** | KL coefficient |

**Step 4 — Edge cases:**

```
β = 0                       →  no KL anchor
Aᵢ = reward directly        →  no group baseline
what you recover            →  PPO
so GRPO is                  →  PPO with group-relative advantage
```

**Step 5 — Distill:**

> **"Sample a group. Advantage = z-score within the group. Clipped policy gradient, KL anchor."**
> **No value model needed. That's it.**

That's a 52-slide lecture (Week 15) compressed into two sentences — and it demonstrates the method working. Note how step 4 does the heavy lifting: zeroing β and replacing the advantage recovers PPO, so GRPO is *PPO plus one idea*.

---

## Part 8 — Building a habit

| | |
|---|---|
| **Cadence** | **Read 1, skim 5** |
| **Slot** | Tuesday morning, hour blocked |
| **Cumulative** | **50 a year** |

**Feeds:** alphaXiv · Hugging Face Daily Papers · arXiv-sanity — **don't subscribe to everything.**

**Paper clubs:**
- Three to five people
- One presenter, weekly
- **Forces you to read deeper**
- **Best ROI in the field**

**X as a filter:**
- Trusted accounts curate for you
- **A good thread is pass-1, done for you**
- Read the thread first, the paper if interested

Note this closes the loop with Part 1's *"tweets are downstream of papers."* Twitter isn't a substitute for reading — it's a **triage layer**. Use it to select, not to learn.

---

## Closing

> - **Most papers don't matter.**
> - **The ones that do deserve the five-step method.**
> - **This is a skill. Practice is the only path.**

---

## Common confusions

**"Do I need to understand every equation?"** No — and trying to is the main reason people give up. Most equations are restating something the prose already said. Identify what the paper *does* first; return to the maths only for the one or two equations that carry the actual novelty.

**"I don't have the maths background."** The blocker is almost always **notation**, not mathematics. Σ, 𝔼, ∇, argmax, and subscript conventions are vocabulary. Learn maybe fifteen symbols and most ML papers become readable. §5 is that list.

**"How do I know if a result is real?"** Ask three questions in order. **What's the baseline, and was it tuned as hard as the proposed method?** (This catches most inflated results.) **Which benchmarks are shown, and which are conspicuously absent?** **Are there ablations** showing which component actually did the work? A paper failing all three is a press release with citations.

**"Why does 'missing baseline' keep coming up?"** Because it's the easiest way to produce a large number honestly. Compare your tuned method against someone else's untuned one and you'll win by a lot without anyone lying. You've seen it twice already in this course — Week 10's PageIndex comparison is against *naive* RAG, and multi-agent claims (S12) routinely compare against an untuned single agent.

**"Should I distrust everything?"** No — that's as useless as believing everything. The goal is **calibrated** reading: separate the **mechanism** (usually sound, and the transferable part) from the **reported numbers** (often optimistic, and setup-dependent). Week 11's RLM is a good case: the mechanism is well-motivated and Week 10's evidence independently supports the direction, while the specific figures come from a citation that doesn't resolve.

**"How much time should one paper take?"** Two minutes to triage. Twenty for a paper that passes. Two hours only for the rare paper you intend to implement or build on. If everything is taking two hours, your triage is broken.

**"Where do I even find the 10%?"** Signals that a paper is worth a look: it's being implemented rather than just retweeted; the authors released code; people are reporting *reproductions* rather than reactions; and it's cited by work you already trust. Volume of discussion is a weak signal — **tweets are downstream of papers**, and downstream of hype.

**"What about the maths in the appendix?"** Usually safe to skip on the first two passes. Appendices carry proofs and hyperparameters. Read the hyperparameters if you plan to reproduce; read the proofs almost never.

**"Do I need to read the related-work section?"** Skim it for what the paper is positioning itself *against* — that tells you what baseline it considers fair, which is directly useful for judging the results.

---

## Key takeaways

1. **The 10% rule** — 90% of papers don't matter; the skill is identifying the 10% fast.
2. **Triage > comprehension. Intuition > rigor.**
3. **Read non-linearly:** abstract → figures → results → method skim → ablations → decide.
4. **Read the method *after* the results** — only invest in the maths if the results justify it.
5. **Ablations are the honest part**; conclusions are throwaway; **the appendix is where bad news hides.**
6. **Six red flags:** single benchmark, missing baselines, no code, cherry-picked examples, vague setup, "SOTA."
7. **Reproducibility is an honesty signal.**
8. **Pattern-match equation shapes before parsing symbols** — `θₜ₊₁ = θₜ − α·(…)` is gradient descent, and the parentheses are the contribution.
9. **Step 4 is the killer move:** set knobs to 0 and 1, see which known algorithm you recover.
10. **GRPO distilled:** sample a group, advantage = z-score within group, clipped policy gradient, KL anchor. **PPO with group-relative advantage.**
11. **Read 1, skim 5, weekly → 50 papers a year.**
12. **Paper clubs are the best ROI**; X is a triage layer, not a substitute.

---

## Glossary

| Term | Meaning |
|---|---|
| **Triage** | Deciding quickly whether a paper deserves further reading. |
| **The 10% rule** | Only about a tenth of papers change how you build. |
| **Three-pass method (Keshav)** | Skim (10 min) → details (1 hr) → reimplement (4+ hrs). |
| **Ablation** | Removing a component to isolate its contribution — the honest part of a paper. |
| **Baseline** | The comparison method; a missing or weak one invalidates the claim. |
| **SOTA** | State of the art — treated here as a marketing word. |
| **Cherry-picking** | Presenting selected favourable examples as representative. |
| **𝔼[X]** | Expectation — the average of X over a distribution. |
| **D_KL(P ‖ Q)** | KL divergence — how different two distributions are. |
| **∇L** | Gradient — direction of steepest increase. |
| **argmax / argmin** | The input that maximizes / minimizes a function. |
| **Policy ratio (π/π_old)** | How much a policy changed after an update. |
| **𝟙[cond]** | Indicator function: 1 if the condition holds, else 0. |
| **O(n²)** | Big-O — quadratic scaling. |
| **Lagrangian (ℒ = f + λ·g)** | Constrained optimization; **λ is the price of the constraint**. |
| **The 5-step method** | Breathe → identify → move to paper → translate → edge cases → distill. |
| **Edge-case reduction** | Setting parameters to 0/1 to recover a known algorithm. |
| **Primitives → novelty** | Papers build vocabulary; back up when lost in the method. |
| **alphaXiv / HF Daily Papers / arXiv-sanity** | Paper discovery feeds. |
| **Paper club** | 3–5 people, one weekly presenter — described as the best ROI in the field. |

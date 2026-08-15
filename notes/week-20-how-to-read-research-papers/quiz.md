# Week 20 — Quiz (20 questions)

**Topic:** How to Read Research Papers — triage, notation, the 5-step method
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The "10% rule" states that:
- A) Only 10% of a paper is worth reading
- B) 90% of papers don't matter; 10% change how you build
- C) You should read 10% of arXiv weekly
- D) Papers should be skimmed in 10% of the time

**2.** The recommended triage order is:
- A) Abstract → introduction → related work → method → results
- B) Abstract → figures/captions → results table → method skim → ablations → decide
- C) Method → results → abstract
- D) Conclusion → abstract → appendix

**3.** The method section should be read:
- A) First, since everything depends on it
- B) After the results
- C) Only in pass 3
- D) Never

**4.** Which section is described as "the honest part"?
- A) The abstract
- B) Related work
- C) Ablations
- D) The conclusion

**5.** Which section is described as "throwaway"?
- A) Abstract
- B) Method
- C) Conclusion
- D) Appendix

**6.** Where do authors typically bury bad results?
- A) The main figures
- B) The appendix
- C) The abstract
- D) Related work

**7.** In Keshav's three-pass method, pass 2 takes roughly:
- A) 10 minutes
- B) 1 hour
- C) 4+ hours
- D) A full day

**8.** "We outperform Llama-3" is flagged as a red flag because:
- A) Llama-3 is outdated
- B) It omits at what size and what budget — it may not be apples-to-apples
- C) Llama-3 is closed source
- D) Benchmarks are always unreliable

**9.** In `ℒ = f + λ·g`, λ represents:
- A) The learning rate
- B) The price of the constraint
- C) The number of layers
- D) A regularisation dropout rate

**10.** Step 4 of the 5-step method is:
- A) Translate symbols into plain English
- B) Move to pen and paper
- C) Set the knobs to 0 and 1 and see what known algorithm you recover
- D) Write a one-sentence intuition

**11.** Applying step 4 to GRPO (β = 0, advantage = raw reward) recovers:
- A) SFT
- B) DPO
- C) PPO
- D) Rejection sampling

**12.** The recommended reading habit is:
- A) Read 5, skim 1
- B) Read 1, skim 5
- C) Read everything on arXiv daily
- D) Read only survey papers

---

## Short answer

**13.** Explain why "triage > comprehension" and what skill this actually demands.

**14.** Explain why the method should be read after the results, and what this implies about linear reading.

**15.** Describe the asymmetry in where authors hide information, and how it should change your reading.

**16.** List the six red flags and apply two of them to a claim from an earlier week of this course.

**17.** Explain "pattern-match before parsing" with the three equation shapes given.

**18.** Explain step 4 (edge cases) and why it is the most powerful step. Use GRPO as the example.

**19.** Explain "primitives → novelty" and the practical advice that follows when you get lost.

**20.** You have one hour a week for papers. Design your routine, justifying each element.

---
---

## Answer key

**1. B** — 90% of papers don't matter, 10% change how you build, and the skill is identifying the 10% fast.

**2. B** — Abstract, then figures and captions, then the results table, then a method skim, then ablations, then decide whether to go deeper.

**3. B** — After the results, so you only invest in the maths if the results justify it.

**4. C** — Ablations, because they isolate what each component actually contributes.

**5. C** — The conclusion.

**6. B** — The appendix. Cherry-picked examples go in the main figures, and caveats in the footnotes.

**7. B** — About one hour. Pass 1 is ten minutes, pass 3 is four or more hours, and most papers stop at pass 1.

**8. B** — It omits the comparison's size and compute budget, so it may not be apples-to-apples.

**9. B** — The price of the constraint, the same intuition as β in a KL penalty.

**10. C** — Set the knobs to 0 and 1 and identify which known algorithm you recover.

**11. C** — PPO. GRPO is therefore PPO with a group-relative advantage.

**12. B** — Read one paper properly and skim five, weekly, accumulating around 50 a year.

**13.** Because the binding constraint is **attention, not reading speed**. The field moves in weeks, far more is published than anyone can read, and by the 10% rule most of it will not change how you build — so the value is created by *deciding what not to read*, and reading a mediocre paper thoroughly is worse than skimming five and finding the one that matters. **The skill this demands is closer to editorial judgment than to study:** recognising from an abstract and a results table whether a claim is novel or repackaged, whether the evidence is the kind that could be wrong, and whether it touches anything you actually build. It also requires the discipline to **stop** — the failure mode for conscientious readers is finishing papers out of obligation, which is why the lecture makes "if no, close it" an explicit instruction.

**14.** The method section is where the maths lives and where the reading is most expensive, but it is **only worth understanding if the results justify the effort**. Reading it first means you invest heavily *before* knowing whether the paper's claims hold up, whether the baselines are fair, or whether the improvement is meaningful — so on the ~90% of papers that don't matter, that investment is wasted. Reading results first lets you ask "is this delta real and does it matter to me?" and only then, if the answer is yes, spend the time to understand how it was achieved. **The implication is that linear reading is wrong for papers:** a paper is a **reference document**, not a textbook, and its section order reflects the conventions of academic presentation rather than the optimal order for a reader deciding whether to care. You jump to the signal.

**15.** The asymmetry is that **bad news hides in the appendix while good news is oversold in the main figures**: bad results are buried in the appendix, cherry-picked examples appear in the main figures, missing baselines are silently absent, and caveats sit in the footnotes. **How this changes your reading:** you go to the appendix specifically *looking for problems* — extra ablations that didn't make the cut, failure cases, hyperparameters that reveal an unfair comparison — and you **discount the main figures**, treating them as a curated best case rather than a representative sample, unless the paper explicitly says it reports random samples. Missing baselines are the hardest to spot because absence leaves no trace, so you must actively ask "what obvious comparison should be here and isn't?" The general principle is that the most informative parts of a paper are the ones the authors had least incentive to polish.

**16.** **The six:** single benchmark, missing baselines, no code, cherry-picked examples, vague setup, and "we achieve SOTA." **Applied to Week 10's PageIndex claim (54% naive RAG vs 98.7% PageIndex on FinanceBench):** **(i) Single benchmark** — this is one leaderboard on one document type, long structured financial filings, which is precisely where structure preservation should help most; the rule is to ask whether it works on three or more unrelated tasks, and no such evidence was presented, so it may not transfer to unstructured corpora like chat logs or scraped web pages. **(ii) Missing baselines** — the comparison is against **naive** RAG, which Week 9 established is a deliberately weak starting point, beaten substantially by recursive chunking, contextual retrieval, hybrid search with RRF, and cross-encoder reranking. An apples-to-apples comparison would pit PageIndex against that full advanced pipeline. (Also applicable: the numbers appear to be vendor-reported, which is the same caution the memory-benchmark section of Week 18 raises.)

**17.** The idea is that an equation's **shape** carries most of its meaning, and shape can be recognised without parsing every symbol. **`θₜ₊₁ = θₜ − α·(·)`** is an **optimization step**: new parameters equal old parameters minus a step in some direction, which immediately tells you this is gradient descent and that the *only* interesting content is whatever sits in the parentheses. **`L = −log p(y|x)`** is **cross-entropy loss**, the standard objective for predicting y given x, so seeing it means "they're doing ordinary supervised likelihood training" and you can move on. **`π / π_old`** is a **policy ratio**, which places the paper firmly in the PPO family and tells you to expect clipping and a KL anchor. **Why this matters:** recognising the family gets you roughly 80% of the understanding for about 5% of the effort, and it converts an intimidating unfamiliar equation into a familiar template plus a modification — which is exactly what step 4 then exploits.

**18.** **Step 4 is to set the knobs to 0 and 1 and ask what known algorithm you recover**, on the premise that new methods are almost always old methods plus a term. **Applied to GRPO:** setting **β = 0** removes the KL anchor, and replacing the group-normalized advantage **Aᵢ with the raw reward** removes the group baseline — and what remains is exactly **PPO**. So GRPO is **PPO with a group-relative advantage**, and its entire contribution is replacing the learned value model's estimate with a z-score computed within a group of sampled responses. **Why it is the most powerful step:** it converts an unfamiliar formula into *a familiar formula plus a delta*, and the delta **is** the paper's contribution. Instead of understanding an equation from scratch you only need to understand one change against something you already know, which is far less work and also tells you precisely what to be sceptical about. It reliably produces the "just X with one extra knob" insight, which is the single most compressible form of understanding a method — and it feeds directly into step 5's one-sentence distillation.

**19.** **Papers build vocabulary**: section 5 depends on definitions and notation introduced in section 2, which is why the ordering exists and why section 5 is genuinely harder than section 0. When you find yourself lost in the method, the usual cause is **not** that the method is beyond you but that a **primitive from earlier didn't land** — a symbol you never named, a quantity whose meaning you skimmed, an assumption stated once in passing. **The practical advice is to back up rather than push forward**: return to where the confusing term was first defined, name it properly in plain English (step 3), and re-read the method from there. Pushing forward feels like progress but compounds the deficit, because every subsequent line uses the thing you don't understand. This also explains why step 1 says to identify formulas "including ones referenced but not shown" — a paper frequently builds on an equation from prior work that it assumes you know, and that unstated primitive is often exactly what is blocking you.

**20.** **Structure the hour as triage-heavy, not depth-heavy — one deep read, five skims.** *Before the hour:* let feeds and trusted accounts do the first filter, since a good thread is pass-1 already done for you; subscribe to one or two sources (alphaXiv, Hugging Face Daily Papers) rather than everything, and maintain a short queue of candidates during the week so the hour isn't spent hunting. *First ~25 minutes — five triage passes at five minutes each:* for each candidate, read the abstract, look at the figures and captions, check the results table, and apply the red-flag checklist — single benchmark, missing baselines, no code, curated examples, vague setup, "SOTA." Then answer the three decision questions: does it change how I build, is the method novel or repackaged, are the claims verifiable? Most will fail and should be closed without guilt. *Remaining ~35 minutes — one deep read:* take the single paper that survived and skim the method, study the ablations, and if it is math-heavy apply the **5-step method** to its one key formula — pen and paper, name every symbol, run the edge cases to see which known algorithm it reduces to, then write a **one-sentence distillation**. *Keep the distillations in a running note*, since the point is cumulative vocabulary, not individual papers. **The multiplier is a paper club** — three to five people, one presenter weekly — described as the best ROI in the field, because presenting forces depth you will not reach alone and you effectively get four other people's triage for free. At read-1-skim-5 this compounds to roughly **50 papers a year**, which is the actual goal: not comprehension of any single paper but a maintained map of the field.

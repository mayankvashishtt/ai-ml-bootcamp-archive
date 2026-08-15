# S3 — Quiz (20 questions)

**Topic:** Evaluation & Statistics Fundamentals
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The test set exists (rather than just train/validation) because:
- A) It is larger
- B) Repeated validation-set decisions leak information, so an untouched set is needed
- C) It contains harder examples
- D) Regulators require three splits

**2.** Data leakage typically presents as:
- A) A crash during training
- B) Suspiciously good offline results
- C) High training loss
- D) Slow convergence

**3.** Fitting a scaler on the full dataset before splitting is:
- A) Standard practice
- B) Train-test contamination
- C) Required for stratified CV
- D) Only a problem for neural networks

**4.** On a dataset with 1% positives, a model predicting "negative" always achieves:
- A) 50% accuracy
- B) 99% accuracy and zero usefulness
- C) An undefined metric
- D) 1% accuracy

**5.** Recall answers:
- A) Of what I flagged, how much was real?
- B) Of what was real, how much did I catch?
- C) Of what was fine, how much did I clear?
- D) How calibrated are my probabilities?

**6.** For spam filtering, the metric to prioritise is generally:
- A) Recall — never miss spam
- B) Precision — a lost legitimate email is worse than a spam that gets through
- C) Accuracy
- D) ROC-AUC

**7.** When positives are rare, prefer:
- A) ROC-AUC, because it is threshold-free
- B) PR-AUC, because ROC-AUC can look high due to the large negative class
- C) Raw accuracy
- D) MAPE

**8.** A model is *calibrated* if:
- A) It ranks positives above negatives
- B) Among predictions of 0.7, roughly 70% are actually positive
- C) Its accuracy exceeds 70%
- D) It has been trained with early stopping

**9.** Moving from 82% to 85% on **100** examples:
- A) Is a clear improvement worth shipping
- B) Falls within the ~±8% confidence interval — indistinguishable
- C) Proves the change is harmful
- D) Requires a Bonferroni correction to interpret

**10.** Halving a confidence interval requires:
- A) 2× the data
- B) 4× the data
- C) 10× the data
- D) The same data with more iterations

**11.** Evaluating two prompts on the *same* eval set rather than independent samples:
- A) Introduces bias
- B) Gives more statistical power by cancelling per-example difficulty
- C) Is only valid for regression
- D) Requires a larger sample

**12.** Testing 20 prompt variants at p < 0.05:
- A) Is standard and needs no adjustment
- B) Would be expected to produce roughly one false positive by chance
- C) Guarantees the winner is real
- D) Halves the required sample size

---

## Short answer

**13.** Explain why three splits are needed, and why this applies to prompt engineering.

**14.** Define data leakage and give four distinct types with a fix for each.

**15.** Explain benchmark contamination and connect it to Week 17's foundation-vs-product distinction.

**16.** Explain why accuracy fails on imbalanced data, and connect it to Week 17's degenerate judge.

**17.** Explain precision/recall trade-off and give two situations with opposite answers, justifying each.

**18.** Explain calibration, why it's distinct from accuracy, and why it matters for LLM confidence fields.

**19.** Explain the bootstrap and why it should be the default for reporting eval results.

**20.** A colleague reports: "Our new RAG pipeline hits 91% on our eval set, up from 84%." What do you ask before believing it?

---
---

## Answer key

**1. B** — Every validation-set look leaks information into your decisions; after many iterations validation performance measures fit to that set, not generalisation.

**2. B** — Leakage is silent; the symptom is unusually good results, which people rarely investigate.

**3. B** — Statistics computed on validation and test rows influence the training transform, contaminating the split.

**4. B** — 99% accuracy, entirely useless.

**5. B** — Of the genuine positives, the fraction caught. Precision answers the first option.

**6. B** — A false positive (legitimate mail marked spam) is worse than a false negative, so precision dominates.

**7. B** — The large negative class makes FPR move slowly, so ROC-AUC can look strong while precision at useful thresholds is poor.

**8. B** — Calibration is about the probabilities matching observed frequencies, not about ranking.

**9. B** — At n=100 and p≈0.82 the 95% interval is roughly ±7.7%, so the two overlap heavily.

**10. B** — Standard error scales as 1/√n, so halving it needs 4× the data.

**11. B** — Paired comparison cancels per-example difficulty; only disagreements carry information (McNemar's test).

**12. B** — At a 5% threshold, roughly one in twenty tests produces a false positive by chance even with no real effect.

**13. Three splits exist because tuning consumes a set's independence.** Train fits the model; **validation** is looked at repeatedly to choose hyperparameters, prompts, and stopping points; and each of those decisions transfers a little information from the validation set into the system. After enough iterations, validation performance measures **how well you fitted the validation set** rather than how the system generalises — so a third, untouched **test** set is the only unbiased estimate available, and it stays unbiased only if you touch it once. **This applies identically to prompt engineering.** Iterating a prompt against a fixed set of examples is a search over prompt-space guided by that set's feedback, which is structurally the same as hyperparameter search; fifty prompt revisions burn a validation set exactly as fifty learning-rate trials do. Week 17's 20/40/40 judge split is this discipline applied to LLM judges, with the sizes inverted because few few-shot examples are needed but a lot of held-out data is required to trust the result.

**14.** **Leakage is when information unavailable at prediction time influences training.** **(i) Target leakage** — a feature that exists only *because* the outcome occurred, such as a `cancellation_date` column when predicting churn. *Fix:* ask of every feature whether it is knowable before the outcome. **(ii) Train-test contamination** — fitting scalers, imputers, or encoders on the full dataset before splitting, so validation and test statistics shape the training transform. *Fix:* split first; fit every transform on train only and apply it to the others. **(iii) Temporal leakage** — randomly splitting time-ordered data, letting the model learn from the future to predict the past. *Fix:* split by time. **(iv) Group leakage** — the same customer, patient, or document appearing in both train and test, so the model recognises the entity rather than generalising. *Fix:* group-aware splitting. (Also valid: duplicate leakage from near-identical rows straddling the split, fixed by deduplicating first; and target-encoding leakage, fixed by computing the encoding inside CV folds only.)

**15.** **Benchmark contamination** is the presence of benchmark questions and answers in a model's pretraining data. Public benchmarks are published on the open web, scraped into training corpora, and thereafter a model may reproduce answers from **memory** rather than solving the task — so a high score can measure recall of the test rather than the capability the test was designed to probe. It is hard to detect and impossible to fully rule out for any public benchmark. **The connection to Week 17** is that this is a second, sharper reason for its foundation-versus-product distinction. Week 17 argued that foundation evals like MMLU are standardised, cross-model, easy to game, and a weak signal for whether *your* pipeline works. Contamination strengthens that: a foundation eval may not be measuring capability at all on a model whose training data absorbed it. A **product** eval built from your own domain data cannot be contaminated in the same way, because the model has never seen it — which makes it both a better proxy for your use case and a more trustworthy measurement in principle.

**16.** Accuracy is the fraction of predictions that are correct, which weights every example equally — so on imbalanced data the majority class dominates it completely. With 1% positives, a model that always predicts "negative" is correct on 99% of examples and scores 99% accuracy while never once identifying the thing you built it to find. The metric is technically correct and practically meaningless, because the interesting cases are a rounding error in the denominator. **Week 17's degenerate judge is exactly this shape.** With an eval set where 90 of 100 cases pass, a judge that outputs "pass" unconditionally agrees with the human labels 90 times and disagrees 10 times, scoring **90% raw agreement** while identifying zero failures — the only thing it was built for. Week 17's remedy generalises directly: split the metric by class. **TPR** (of the true passes, how many did the judge catch?) is 100%, while **TNR** (of the true failures, how many did it catch?) is 0%, and the 0% makes the uselessness unmissable. Precision and recall do the same job for imbalanced classification.

**17.** Precision is TP/(TP+FP) — of what you flagged, how much was real. Recall is TP/(TP+FN) — of what was real, how much you caught. **They trade off through the decision threshold:** lowering it flags more cases, catching more true positives (recall rises) at the cost of more false alarms (precision falls); raising it does the reverse. You cannot maximise both, so you must decide **which error is more expensive** — a business question, not a technical one. **Cancer screening favours recall:** a missed cancer can be fatal, while a false positive costs an anxious patient and a follow-up test, so you accept many false alarms to avoid missing a case. **Spam filtering favours precision:** a spam message reaching the inbox is a minor annoyance, while a legitimate email silently routed to spam can lose a contract, so you tolerate letting spam through to avoid ever misfiling real mail. Same mathematics, opposite answers, decided entirely by the asymmetry of the costs. This is also why F1 is a default rather than an answer — it weights the two equally, which is almost never what the business wants.

**18.** A model is **calibrated** if its probabilities mean what they say: among predictions assigned 0.7, roughly 70% turn out positive. **It is distinct from accuracy because accuracy and AUC only care about ranking** — a model that outputs 0.51 for every positive and 0.49 for every negative ranks perfectly and achieves perfect accuracy at a 0.5 threshold, while its probabilities are almost meaningless. Conversely a well-calibrated model can be mediocre at ranking. The two properties are independent, and which you need depends on whether you consume the *decision* or the *number*. **Calibration matters whenever the probability feeds a downstream computation** — expected-value calculations, cost-weighted thresholds, risk pricing, or deciding when to escalate to a human — because those multiply the probability by something, and a miscalibrated input silently corrupts the result. Check it with a **reliability diagram** plotting predicted probability against observed frequency in buckets; fix it with Platt scaling or isotonic regression. **For LLMs specifically**, a model asked to emit a `confidence: 0–1` field is typically poorly calibrated: it is producing a plausible-looking number, not a measured frequency. If that field routes cases — auto-answer above 0.8, escalate below — it must be validated against actual outcomes first, or you have built a threshold on a number that does not mean what it appears to.

**19.** The **bootstrap** estimates the uncertainty of any metric by resampling: draw a sample of the same size from your eval set **with replacement**, compute the metric, repeat around a thousand times, and take the 2.5th and 97.5th percentiles of the resulting distribution as a 95% confidence interval. It works because resampling simulates the variability you would see across different eval sets drawn from the same population. **It should be the default for three reasons.** First, **it works for any metric** — F1, AUC, an LLM-judge pass rate, a bespoke composite score — including all the ones with no clean closed-form standard error, which is most of what people actually report in LLM work. Second, it is **assumption-light**: it makes no distributional claim, unlike formulas that assume normality. Third, it is **trivial to implement** — a resampling loop around code you already have — so the usual excuse for reporting a bare point estimate disappears. The payoff is that it converts "85%" into "85% (95% CI: 81–89%)", which is the difference between a number that invites a shipping decision and a number that supports one.

**20.** **On the eval set itself:** how many examples, and were the two pipelines evaluated on the **same** set? Paired comparison is both more sensitive and the only fair basis for the claim. Where did the cases come from — hand-built, sampled from production, or generated? If generated by the pipeline being tested, the eval is grading the system on its own homework. **On statistics:** what is the confidence interval? At n=100, ±8% makes 84 and 91 barely distinguishable; at n=400, ±4% makes the gap credible. How many variants were tried before this one won — twenty attempts at a 5% threshold produces a winner by chance. Has this eval set been iterated against many times, so that it is now effectively a validation set rather than a test set? **On leakage:** are any eval questions drawn from the same documents in the retrieval index in a way that makes them trivially findable, and were the eval cases fixed *before* the pipeline was tuned or added afterwards? **On the metric:** what does "91%" measure — answer correctness, retrieval hit rate, or a judge's pass rate? If a judge, has it been aligned and reported with TPR and TNR rather than raw agreement (Week 17)? **On baselines and coverage:** what does keyword search alone score, since Week 9 showed hybrid beats naive on a distribution but not on every query? Does the eval include the failure modes that matter — exact identifiers, dates, cross-document questions — or only the easy cases? **The framing that matters most:** a 7-point jump is large, and per Week 20 an unusually good result deserves *more* scrutiny than a bad one, not less.

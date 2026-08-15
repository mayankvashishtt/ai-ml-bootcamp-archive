# S3 — Evaluation & Statistics Fundamentals

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 17 covers *product* evals well — binary criteria, critique shadowing, judge alignment, TPR/TNR. What's missing is the layer underneath: train/test discipline, data leakage, metric selection, and whether a measured difference is real.

**Fills the gap after:** Week 7 (train/val loss), Week 17 (Harness, Context, Evals)
**Prerequisites:** Weeks 2, 7, 17

---

## 1. What Week 17 left out

Week 17 taught you to build product evals. It didn't teach you to **trust a number**.

| Week 17 covered | This lecture covers |
|---|---|
| Binary > Likert | When accuracy is the wrong metric entirely |
| Judge alignment, TPR/TNR | Confidence intervals — is the difference real? |
| Foundation vs product evals | Train/val/test discipline and data leakage |
| `pass@k` vs `pass^k` | Sample size — how many examples do you need? |
| The flywheel | Overfitting to the test set |

Both halves matter. Week 17 stops you measuring the wrong thing; this stops you believing a number that isn't there.

---

## 2. Train / validation / test — and why three

Week 7 used train and validation. The third split is the one people skip.

| Split | Purpose | How often you touch it |
|---|---|---|
| **Train** | Fit the model | Constantly |
| **Validation** | Tune — hyperparameters, prompts, early stopping | Many times |
| **Test** | Final unbiased estimate | **Once** |

**Why three, not two:** every time you look at the validation set and change something, you leak a little information from it into your decisions. After fifty prompt iterations, validation performance reflects *how well you fit the validation set*, not how the system generalises. The test set is your only defence — and it only works if you leave it alone.

**This applies to prompt engineering exactly as it applies to model training.** Week 17's 20/40/40 judge split is this same discipline with the sizes inverted because you need few few-shot examples and a lot of held-out data.

**Time-series data must split by time.** Random splitting lets the model learn from the future to predict the past — which is leakage, and produces beautiful offline numbers that collapse in production.

---

## 3. Data leakage — the silent killer

**Leakage is when information that won't be available at prediction time influences training.** It's the most common cause of a model that scores brilliantly offline and fails in production, and it is *silent* — the symptom is unusually good results, which nobody investigates.

| Type | Example | Fix |
|---|---|---|
| **Target leakage** | A `cancellation_date` column when predicting churn — only populated *because* they churned | Ask of every feature: is this known **before** the outcome? |
| **Train-test contamination** | Scaling, imputing, or encoding fitted on the full dataset before splitting | Split first. Fit every transform on train only, apply to val/test |
| **Temporal leakage** | Random split on time-ordered data | Split by time |
| **Group leakage** | The same patient/customer in both train and test | Group-aware splitting |
| **Duplicate leakage** | Near-duplicate rows straddling the split | Deduplicate before splitting |
| **Target-encoding leakage** | Category → mean-target computed on the full data (S1) | Compute inside CV folds only |

### Leakage in LLM evaluation

The same problem, in two forms the course touches:

- **Benchmark contamination** — public benchmarks are in the pretraining data. A model may have *memorised* the answers, so a high MMLU score can measure recall rather than capability. This is a specific, sharp reason for Week 17's insistence on **product** evals over foundation evals.
- **Eval-set contamination from your own pipeline** — if you build eval cases from production traces and those traces were themselves generated with the prompt you're testing, you're grading the model on its own homework.

**The heuristic that catches most leakage:** if a result seems too good, it probably is. **Investigate wins as hard as you investigate failures.**

---

## 4. Cross-validation

With limited data, a single train/val split wastes most of it and gives a noisy estimate.

**k-fold cross-validation:** split into k parts; train on k−1, validate on the held-out one; rotate; average the k scores.

- **Every row is used for both training and validation**, at different times
- You get **k scores**, so you can compute a standard deviation — a variance estimate for free
- Costs k× the training time

**Variants:** *stratified* k-fold preserves class balance in each fold (essential for imbalanced data); *grouped* k-fold keeps related rows together (prevents group leakage); *time-series* CV uses expanding windows so you only ever train on the past.

**Keep the test set out of CV entirely.** Cross-validate over train+validation; the test set stays untouched.

---

## 5. Metrics — accuracy is usually wrong

### The imbalanced-data trap

Fraud detection with 1% fraud. A model that predicts "not fraud" every single time achieves **99% accuracy** and is completely worthless.

This is the same shape as Week 17's degenerate judge scoring 90% raw agreement while never identifying a failure — and it's why that lecture insisted on TPR and TNR separately.

### The confusion matrix

```
                 Predicted
                 Pos     Neg
Actual  Pos      TP      FN     ← missed
        Neg      FP      TN
                  ↑
              false alarm
```

| Metric | Formula | Answers |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of what I flagged, how much was real? |
| **Recall (TPR)** | TP / (TP + FN) | Of what was real, how much did I catch? |
| **Specificity (TNR)** | TN / (TN + FP) | Of what was fine, how much did I correctly clear? |
| **F1** | harmonic mean of P and R | One number balancing both |

**Precision and recall trade off.** Lower the decision threshold and you catch more (recall ↑) at the cost of more false alarms (precision ↓).

**Which one you optimise is a business decision, not a technical one:**

| Situation | Optimise | Because |
|---|---|---|
| Cancer screening | **Recall** | A missed case is catastrophic; a false alarm costs a follow-up test |
| Spam filtering | **Precision** | A lost legitimate email is worse than a spam that gets through |
| Fraud with manual review | **Balance** | Bounded by how many cases reviewers can handle per day |

**F1 is a default, not an answer.** It weights precision and recall equally, which is almost never what the business wants. Say which error is worse and optimise that.

### ROC-AUC vs PR-AUC

Both summarise performance across all thresholds.

- **ROC-AUC** plots TPR against FPR. Interpretable as "the probability a random positive is ranked above a random negative." **But it can look impressively high on heavily imbalanced data**, because the huge negative class makes FPR move slowly.
- **PR-AUC** plots precision against recall. **Use this when positives are rare** — it's far more sensitive to the thing you actually care about.

### Calibration — the forgotten one

A model is **calibrated** if, among predictions it assigns 0.7, about 70% are actually positive.

Accuracy and AUC only care about *ranking*. Calibration cares whether the number means anything. It matters whenever the probability feeds a downstream decision — expected-value calculations, thresholding on cost, risk pricing.

Check with a **reliability diagram** (predicted probability vs observed frequency, bucketed). Fix with Platt scaling or isotonic regression.

**Relevant to LLMs:** a model asked for a `confidence: 0–1` field is usually poorly calibrated. If that number drives a decision, validate it against outcomes before trusting it.

### Regression metrics

| Metric | Behaviour |
|---|---|
| **MAE** | Mean absolute error — robust to outliers, in the units of the target |
| **RMSE** | Penalises large errors heavily (Week 2's MSE, square-rooted) |
| **MAPE** | Percentage error — breaks when actuals are near zero |
| **R²** | Fraction of variance explained; can be negative if worse than predicting the mean |

Choose by asking whether one large error is worse than several small ones. If yes, RMSE. If they're equivalent, MAE.

---

## 6. Is the difference real?

You change a prompt. Accuracy goes from 82% to 85% on 100 examples. **Ship it?**

**Probably not — that difference is well within noise.**

### Confidence intervals

For a proportion, the standard error is `√(p(1−p)/n)`, and a rough 95% interval is ±2 standard errors.

At p = 0.82:

| n | 95% interval | Detects a real difference of |
|---|---|---|
| 100 | ±7.7% | ~10%+ |
| 400 | ±3.8% | ~5%+ |
| 1,000 | ±2.4% | ~3%+ |
| 10,000 | ±0.8% | ~1%+ |

**With 100 examples, 82% and 85% have overlapping intervals.** You cannot distinguish them. Reporting "85%, up from 82%" without an interval is reporting noise as signal.

**The rule of four:** halving the interval requires **four times** the data. This is why good eval sets are expensive.

### Paired comparison — the cheap win

Comparing two systems on the **same** examples is far more sensitive than comparing independent samples, because it cancels per-example difficulty.

Instead of two independent accuracies, count: how many examples did A get right and B wrong, and vice versa? Only the disagreements carry information — this is **McNemar's test**, and it needs far fewer examples than an unpaired comparison.

**Always evaluate variants on the same eval set.** It's free statistical power.

### Bootstrap — the practical tool

When the metric has no clean formula (F1, AUC, an LLM-judge pass rate):

```
repeat 1000 times:
    resample the eval set with replacement
    compute the metric
take the 2.5th and 97.5th percentiles → 95% interval
```

Simple, assumption-light, works for any metric. **This should be your default** for reporting eval results.

### Multiple comparisons

Test 20 prompt variants at p < 0.05 and you'd expect **one false positive by chance** even if none of them is better. Iterating until something looks good is a form of overfitting to the validation set. Correct for it (Bonferroni: divide the threshold by the number of tests) or confirm the winner on held-out data.

---

## 7. How many eval examples do you need?

| Purpose | Rough size |
|---|---|
| Smoke test — does it run? | 5–10 |
| Catching obvious regressions | 30–50 |
| Detecting a ~10% difference | ~100 |
| Detecting a ~5% difference | ~400 |
| Detecting a ~1% difference | ~10,000 |

Week 17's hands-on used 20 examples — right for building and aligning a judge, **too few to compare systems**. Those are different jobs and the size requirement differs by an order of magnitude.

**For rare failure modes, the binding constraint is positives, not total size.** If a failure occurs 2% of the time, a 100-example set contains ~2 instances — you cannot measure a change in something you observe twice. Either enrich the set with known failure cases (and report on that slice separately) or make it much larger.

---

## 8. Baselines — the discipline

**Every result needs a baseline, or the number is meaningless.**

| Baseline | Catches |
|---|---|
| **Majority class** | A "90% accurate" classifier on 90/10 data |
| **Random** | Sanity floor |
| **Simplest reasonable model** | Logistic regression (S1); keyword search before RAG |
| **Previous system** | The only comparison the business cares about |
| **Human performance** | The ceiling; also reveals label noise |

**Human agreement is the most underused of these.** If two experts agree only 80% of the time, an 85% model score may be at the ceiling of what the labels can express — and chasing 95% is chasing noise.

---

## 9. Overfitting to the test set

The failure mode of a mature eval process.

You measure, tune, measure again — a hundred times. Each decision leaks a little information from the test set into your system. Eventually the number is optimistic and you don't know by how much.

**Symptoms:** offline scores that keep improving while user-facing quality doesn't; a big gap between your eval and production behaviour.

**Mitigations:**
- Keep a **held-out test set you touch once** per major release
- **Rotate in fresh eval cases** periodically — Week 24's scored production traces are the natural source
- Track **how many times** you've evaluated against a given set; treat a much-used set as a validation set, not a test set
- Prefer **live production metrics** as the final arbiter

This is also the leaderboard problem: a public benchmark, optimised against by the whole field for years, is a validation set for the community — which is a second reason (alongside contamination) that leaderboard scores overstate real-world capability.

---

## 10. Reporting honestly

A defensible eval report contains:

1. **The metric, and why that metric** — not just "accuracy"
2. **The sample size**
3. **A confidence interval or standard deviation**
4. **The baseline**
5. **How the data was split**, and confirmation that the test set is untouched
6. **Known limitations** — what the eval doesn't cover

**Compare these:**

> ❌ "The new prompt improves accuracy to 85%."

> ✅ "On 400 held-out support tickets, the new prompt scores 85% ±3.8% (95% CI) versus 82% ±3.8% for the current prompt — overlapping intervals, so we cannot conclude an improvement. A paired comparison shows the new prompt winning on 34 examples and losing on 21, which is suggestive but not conclusive at this sample size. Neither prompt was evaluated on multi-turn conversations."

The second one is longer and much more useful. It also tells you what to do next: get more data, or accept that the difference is too small to matter.

---

## 11. How this connects to the course

| Course moment | What this lecture adds |
|---|---|
| **W7** — "eval loss is your north star" | Why the *third* split exists, and what train/val gap means |
| **W7** — initial loss = ln(vocab_size) | A baseline check, generalised: always have a baseline |
| **W17** — binary > Likert | Sound; add sample size and intervals on top |
| **W17** — TPR + TNR, not raw agreement | The imbalanced-metric problem, stated generally |
| **W17** — 20-example hands-on | Right for judge alignment, too small for system comparison |
| **W17** — foundation vs product evals | Benchmark contamination is a second, sharper reason |
| **W10** — vendor-reported benchmarks | Leaderboard overfitting explains the pattern |
| **W20** — reading papers sceptically | Single benchmark, missing baseline — the same checklist |
| **W24** — scored production traces | The right source of fresh eval cases |

---

## Key takeaways

1. **Three splits, and the test set is touched once.** Prompt iteration burns validation sets exactly as training does.
2. **Leakage is silent and shows up as suspiciously good results.** Investigate wins as hard as failures.
3. **Split before preprocessing.** Fit every transform on train only.
4. **Accuracy is usually the wrong metric** — it hides in imbalance, exactly like Week 17's degenerate judge.
5. **Precision vs recall is a business decision.** F1 is a default, not an answer.
6. **Use PR-AUC, not ROC-AUC, when positives are rare.**
7. **Calibration matters** whenever a probability drives a decision — including LLM self-reported confidence.
8. **Report intervals.** 82% → 85% on 100 examples is noise.
9. **The rule of four:** halving the interval needs 4× the data.
10. **Paired comparison on the same eval set is free statistical power.**
11. **Bootstrap for any metric without a clean formula** — make it the default.
12. **Every result needs a baseline**, and human agreement reveals the label-noise ceiling.
13. **Test sets wear out.** Rotate them, and prefer production metrics as the final arbiter.

---

## Glossary

| Term | Meaning |
|---|---|
| **Train / validation / test** | Fit / tune / final unbiased estimate. |
| **Data leakage** | Information unavailable at prediction time influencing training. |
| **Target leakage** | A feature that exists only because of the outcome. |
| **Temporal leakage** | Training on future data to predict the past. |
| **Group leakage** | The same entity appearing in train and test. |
| **Benchmark contamination** | Benchmark data present in pretraining. |
| **k-fold CV** | Rotating held-out folds; gives a variance estimate. |
| **Stratified / grouped / time-series CV** | Preserving class balance / entity grouping / temporal order. |
| **Confusion matrix** | TP / FP / FN / TN table. |
| **Precision / Recall** | Correctness of flags / coverage of real positives. |
| **Specificity (TNR)** | Correctly identified negatives. |
| **F1** | Harmonic mean of precision and recall. |
| **ROC-AUC / PR-AUC** | Threshold-free summaries; PR-AUC for rare positives. |
| **Calibration** | Whether predicted probabilities match observed frequencies. |
| **Reliability diagram** | Plot of predicted probability vs observed frequency. |
| **MAE / RMSE / MAPE / R²** | Regression metrics. |
| **Confidence interval** | Range of plausible values for the true metric. |
| **Standard error** | Sampling variability of an estimate. |
| **Bootstrap** | Resampling with replacement to estimate an interval. |
| **McNemar's test** | Paired comparison using only the disagreements. |
| **Multiple comparisons** | Inflated false-positive rate from many tests. |
| **Bonferroni correction** | Dividing the significance threshold by the number of tests. |
| **Baseline** | The reference a result is meaningless without. |

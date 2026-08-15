# S1 — Classical ML & Tabular Data

> ⚠️ **Supplementary lecture.** This is not part of the 100xSchool course. It was written to fill a gap: the course is called "AI & ML" and contains **no machine learning outside deep learning**. It goes from "what is ML" (Week 1) straight to neural networks (Week 2) and never returns.

**Fills the gap after:** Week 1 (Fast-Tracking the Course of AI), Week 2 (Neural Networks from Scratch)
**Prerequisites:** Python, basic statistics
**Why it matters:** most industry ML is still tabular, and neural networks usually lose there.

---

## 1. The gap this fills

Week 1 traced the history: rules → machine learning → deep learning → transformers. Week 2 then built a neural network and the course never looked back.

But **"machine learning" is much larger than "deep learning."** The branch the course skipped is the one most working data scientists use daily:

| Data type | What wins | Covered in the course? |
|---|---|---|
| Text, images, audio, video | **Deep learning** | ✅ Weeks 2–25 |
| **Tabular** (rows and columns) | **Gradient-boosted trees** | ❌ Not at all |

A student finishing this course could build a RAG agent but could not build a churn model — and, more importantly, **wouldn't know that gradient boosting was the right answer.** That's the gap.

---

## 2. The tabular reality

**On tabular data, gradient-boosted decision trees generally beat neural networks** — on accuracy, on training time, and on the amount of tuning required. This has been shown repeatedly in benchmark studies and is the consistent experience of practitioners; it remains true even as deep-learning architectures designed specifically for tabular data have appeared.

**Why trees win on tables** — four structural reasons:

| Reason | Detail |
|---|---|
| **Heterogeneous features** | A table mixes age (0–100), income (0–10⁶), and country (categorical). Trees split on each feature independently and are **scale-invariant** — no normalisation needed. Neural networks need every input on a comparable scale. |
| **No smoothness prior** | Neural networks assume nearby inputs give nearby outputs. Tabular relationships are often genuinely discontinuous — a credit decision may flip hard at a threshold. Trees model step functions natively; networks must approximate them. |
| **Small data** | Deep learning needs a lot of examples (Week 1's first bottleneck). Most business datasets have thousands of rows, not millions. Trees are far more sample-efficient. |
| **Irrelevant features** | Real tables carry junk columns. Trees simply never split on them; networks must learn to zero out the weights. |

**The practical rule:** if your data is a table, **start with gradient boosting**. Reach for a neural network only when you've measured that it helps.

---

## 3. The supervised learning taxonomy

Everything the course covered is one corner of a larger map.

```
Machine Learning
├── Supervised (labelled data)
│   ├── Regression      → predict a number     (house price, demand)
│   └── Classification  → predict a category   (churn / no churn, spam)
├── Unsupervised (no labels)
│   ├── Clustering      → find groups          (customer segments)
│   └── Dim. reduction  → compress             (PCA, embeddings)
└── Reinforcement (reward signal)              → Weeks 14–16
```

**Where the course sits:** language modelling is *self-supervised* — a special case of supervised learning where the labels come free from the data itself (Week 7's "shift by one"). RL appeared in Weeks 14–16. Classification, regression, and clustering never did.

---

## 4. Linear and logistic regression — the baseline that's often enough

### Linear regression

Week 2's house-price example (`$200/sq ft`) **is linear regression** — the course used it as motivation and then moved on without naming it.

```
ŷ = w₁x₁ + w₂x₂ + ... + b
```

That's a single neuron with no activation function. Week 2's linear-collapse derivation applies: stacking these gains nothing.

### Logistic regression

For classification, wrap the same linear output in a sigmoid:

```
p(y=1) = σ(w·x + b)
```

Which is **exactly** Week 2's neuron. Logistic regression is a one-neuron network — the historical starting point of the whole field.

**Why start here, always:**
- **Fast** to train, trivial to deploy
- **Interpretable** — each weight is a per-unit effect on the outcome, which regulated industries often require
- **A baseline you must beat.** If your gradient-boosted model can't beat logistic regression, something is wrong with your setup — and you've learned that cheaply.

Week 3's caution applies to interpretation: the coefficients are only interpretable if features aren't strongly correlated with each other.

---

## 5. Trees → forests → boosting

### Decision tree

A sequence of yes/no questions:

```
              income > 50k?
             /            \
          yes              no
           |                |
    tenure > 2y?         churn: YES
      /      \
    yes       no
     |         |
  churn: NO  churn: YES
```

**Strengths:** interpretable, handles mixed types, no scaling, captures interactions automatically.
**Weakness:** a single deep tree **overfits badly** — it will happily memorise the training set (Week 2's overfitting, in a different model class).

### Random forest — bagging

Train **many** trees, each on a random subset of rows *and* a random subset of features at each split, then average their predictions.

**Why it works:** individual trees are high-variance — small data changes produce very different trees. Averaging many decorrelated high-variance models cancels the variance while preserving the signal. Feature subsampling is what keeps the trees from all making the same mistakes.

Robust, hard to misconfigure, and a strong default.

### Gradient boosting — the winner

Trees trained **sequentially**, each one fitting the *residual errors* of the ensemble so far:

```
1. Start with a constant prediction (e.g. the mean)
2. Compute the errors
3. Fit a small tree to predict those errors
4. Add it to the ensemble, scaled by a learning rate
5. Repeat, hundreds of times
```

**This is gradient descent (Week 2) in function space** — instead of stepping parameters downhill, you add a whole model that points downhill. The learning rate plays exactly the same Goldilocks role: too high overshoots, too low never arrives.

| Library | Character |
|---|---|
| **XGBoost** | The classic; huge ecosystem, extensive tuning surface |
| **LightGBM** | Fastest on large data; leaf-wise growth |
| **CatBoost** | Best native categorical handling; strong defaults, least tuning |

**Practical advice:** they're within noise of each other when tuned. Start with **CatBoost** if you have many categorical columns and want good results without tuning; **LightGBM** if training time matters.

### Bagging vs boosting

| | Random Forest (bagging) | Gradient Boosting |
|---|---|---|
| **Trees trained** | Independently, in parallel | Sequentially, each correcting the last |
| **Each tree is** | Deep, low-bias, high-variance | Shallow, high-bias, low-variance |
| **Reduces** | **Variance** | **Bias** |
| **Overfits if** | Rarely — more trees is safe | **Yes** — too many rounds overfits |
| **Tuning needed** | Little | More |

The asymmetry matters: adding trees to a forest is safe; adding rounds to a boosted model is not. Boosting needs early stopping on a validation set.

---

## 6. The bias-variance tradeoff

The concept underneath all of this, and the one the course never named.

| | Meaning | Symptom |
|---|---|---|
| **Bias** | Error from wrong assumptions — the model is too simple | **Underfitting** — bad on train *and* test |
| **Variance** | Error from sensitivity to the training sample — too complex | **Overfitting** — great on train, bad on test |

```
Total error = bias² + variance + irreducible noise
```

Week 2's experiments were this tradeoff without the vocabulary:
- **2 hidden neurons couldn't learn XOR** → high bias, underfitting, a *capacity* problem
- **Dropout in Week 7** → a variance-reduction technique

**How each model class moves along the curve:** more trees in a forest → less variance. More boosting rounds → less bias, eventually more variance. Deeper trees → less bias, more variance. Regularisation of any kind → more bias, less variance.

You are always trading one for the other. The best model is the one at the minimum of the sum, which you find empirically on held-out data — never on the training set.

---

## 7. Feature engineering

This is where most tabular accuracy actually comes from, and it's the sharpest contrast with the course's material.

**Deep learning's central promise (Week 1) was learning features automatically** — that's what replaced hand-crafted rules. On tabular data that promise is much weaker: there's no spatial or sequential structure for the network to exploit, so **domain-informed hand-built features usually beat learned ones.**

Common transforms:

| Type | Example |
|---|---|
| **Ratios and differences** | `debt / income` beats either column alone |
| **Date decomposition** | timestamp → day-of-week, month, is_holiday, days_since_signup |
| **Aggregations** | per-customer mean, count, max over a transactions table |
| **Categorical encoding** | one-hot (low cardinality), target encoding (high cardinality — leakage-prone, see S3) |
| **Binning** | age → age bracket, when the relationship is genuinely non-monotonic |

> ⚠️ **Target encoding is a leakage trap.** Replacing a category with the mean target for that category uses the label, so it must be computed **within cross-validation folds only** — never on the full dataset before splitting. See S3.

---

## 8. Unsupervised learning, briefly

**k-means clustering** — partition data into k groups by iteratively assigning points to the nearest centroid and recomputing centroids. Used for customer segmentation and anomaly detection. You must choose k yourself; it assumes roughly spherical, similarly-sized clusters.

**PCA** — find the directions of greatest variance and project onto the top few. The course used it in Week 3 to plot embeddings in 2D without explaining it. Also used for compression, denoising, and decorrelating features.

Both are worth knowing because they appear constantly as *preprocessing steps*, not as end goals.

---

## 9. When to actually reach for deep learning on tabular data

Genuine cases:
- **Very large datasets** (millions of rows) where the extra capacity pays off
- **Mixed modality** — a table *plus* free-text or image columns, where you need one model over both
- **Embedding very high-cardinality categoricals** (millions of user or product IDs), where learned embeddings beat one-hot
- **Transfer learning** from a pretrained tabular foundation model

Otherwise: gradient boosting.

---

## 10. The practical workflow

```
1. Understand the data      →  distributions, missingness, target balance
2. Split FIRST              →  train/val/test before any preprocessing (S3)
3. Baseline                 →  logistic/linear regression. Write down the number.
4. Gradient boosting        →  CatBoost or LightGBM with defaults
5. Feature engineering      →  usually the biggest single gain
6. Tune                     →  learning rate, depth, regularisation, early stopping
7. Evaluate honestly        →  right metric, held-out test set, confidence interval (S3)
8. Interpret                →  feature importance, SHAP values
```

**Step 2 is the one people get wrong**, and it's silent — see S3 on data leakage.

**Step 8 matters commercially:** SHAP values attribute a prediction to individual features, which is often required for lending, insurance, and hiring decisions, and is useful everywhere else for debugging.

---

## 11. How this connects to the course

| Course concept | Classical ML counterpart |
|---|---|
| Week 2 — house price example | **Linear regression**, unnamed |
| Week 2 — the neuron `σ(w·x+b)` | **Logistic regression** exactly |
| Week 2 — gradient descent | Generalises to **gradient boosting** in function space |
| Week 2 — 2-neuron capacity failure | **High bias / underfitting** |
| Week 3 — PCA embedding plots | **Dimensionality reduction**, unnamed |
| Week 7 — dropout | **Variance reduction / regularisation** |
| Week 12 — "RAG vs fine-tuning" decision | Same discipline: **pick the method that fits the problem** |

---

## Key takeaways

1. **The course has no ML outside deep learning** — this is the single largest gap in it.
2. **On tabular data, gradient-boosted trees generally beat neural networks** — and most industry ML is tabular.
3. **Four structural reasons trees win:** heterogeneous features, no smoothness prior, small data, irrelevant features.
4. **Logistic regression is Week 2's neuron.** It's also the baseline you must beat before believing any fancier result.
5. **Bagging reduces variance (safe to add trees); boosting reduces bias (needs early stopping).**
6. **Bias-variance is the frame** for every capacity decision — including Week 2's XOR failure.
7. **Feature engineering usually beats model choice** on tabular data — the opposite of the deep-learning story.
8. **Always split before preprocessing.** Leakage is silent and inflates every number you report.
9. **Reach for deep learning on tables only when you've measured that it helps.**

---

## Further reading

- *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) — free PDF, the standard entry text
- *The Elements of Statistical Learning* — the rigorous version of the same material
- scikit-learn user guide — excellent conceptual documentation, not just API reference
- Papers comparing tree ensembles and deep learning on tabular benchmarks (search "why do tree-based models still outperform deep learning on tabular data")

---

## Glossary

| Term | Meaning |
|---|---|
| **Supervised learning** | Learning from labelled input-output pairs. |
| **Regression / Classification** | Predicting a number / predicting a category. |
| **Self-supervised** | Labels derived from the data itself — e.g. next-token prediction. |
| **Linear regression** | `ŷ = w·x + b` — a neuron with no activation. |
| **Logistic regression** | `σ(w·x + b)` — a single neuron; the classification baseline. |
| **Decision tree** | Recursive yes/no splits; interpretable, overfits alone. |
| **Bagging** | Training models on random subsets and averaging — reduces variance. |
| **Random forest** | Bagged trees with random feature subsets per split. |
| **Boosting** | Sequential models each fitting the previous ensemble's residuals — reduces bias. |
| **XGBoost / LightGBM / CatBoost** | The three standard gradient-boosting libraries. |
| **Bias** | Error from oversimplification — underfitting. |
| **Variance** | Error from sensitivity to the training sample — overfitting. |
| **Scale invariance** | Property of trees: feature scaling doesn't change splits. |
| **Feature engineering** | Constructing informative input columns by hand. |
| **Target encoding** | Replacing a category with its mean target — leakage-prone. |
| **k-means** | Clustering by iterative centroid assignment. |
| **PCA** | Projection onto directions of greatest variance. |
| **SHAP** | Per-feature attribution of an individual prediction. |

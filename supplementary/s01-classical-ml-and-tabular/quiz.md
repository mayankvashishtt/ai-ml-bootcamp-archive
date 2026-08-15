# S1 — Quiz (20 questions)

**Topic:** Classical ML & Tabular Data
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** On tabular data, the model class that generally performs best is:
- A) Deep neural networks
- B) Gradient-boosted decision trees
- C) k-nearest neighbours
- D) Transformers

**2.** Which is NOT one of the structural reasons trees beat neural networks on tables?
- A) Heterogeneous features with different scales
- B) Tabular relationships are often discontinuous
- C) Tabular datasets are typically small
- D) Trees train faster on GPUs

**3.** Logistic regression is mathematically equivalent to:
- A) A decision tree of depth 1
- B) A single neuron with a sigmoid activation
- C) A two-layer network
- D) k-means with k=2

**4.** Bagging (random forests) primarily reduces:
- A) Bias
- B) Variance
- C) Irreducible noise
- D) Training time

**5.** Gradient boosting primarily reduces:
- A) Bias
- B) Variance
- C) Feature count
- D) Memory usage

**6.** Which failure mode is a *capacity* problem?
- A) Overfitting — great on train, bad on test
- B) Underfitting — bad on train and test
- C) Data leakage
- D) Class imbalance

**7.** Adding more trees to a random forest:
- A) Eventually causes overfitting
- B) Is generally safe
- C) Increases bias linearly
- D) Requires re-tuning the learning rate

**8.** Adding more boosting rounds:
- A) Is generally safe
- B) Can overfit, so needs early stopping
- C) Reduces variance monotonically
- D) Has no effect after 100 rounds

**9.** Trees are described as *scale-invariant*, meaning:
- A) They work on any dataset size
- B) Feature scaling doesn't change the splits chosen
- C) They can be scaled across GPUs
- D) They handle missing values automatically

**10.** Target encoding is flagged as dangerous because:
- A) It's slow to compute
- B) It uses the label, so it leaks unless computed within CV folds
- C) It only works for binary targets
- D) It requires one-hot encoding first

**11.** Compared with deep learning, feature engineering on tabular data is:
- A) Unnecessary — the model learns features
- B) Usually the biggest single source of accuracy gain
- C) Only useful for linear models
- D) Handled automatically by XGBoost

**12.** Which is a genuine reason to use deep learning on tabular data?
- A) The dataset has 5,000 rows
- B) The table includes free-text or image columns alongside numeric ones
- C) You want interpretability
- D) You have many irrelevant features

---

## Short answer

**13.** Explain the gap this lecture fills and why it matters commercially.

**14.** Give the four structural reasons trees outperform neural networks on tabular data.

**15.** Explain how gradient boosting relates to Week 2's gradient descent.

**16.** Contrast bagging and boosting across: how trees are trained, what each tree looks like, what error component is reduced, and overfitting risk.

**17.** Explain bias-variance and map Week 2's two-hidden-neuron XOR failure onto it.

**18.** Why is logistic regression worth running even when you intend to use gradient boosting?

**19.** Explain why "deep learning learns features automatically" is weaker on tabular data than on images or text.

**20.** You're given a 40,000-row table with 60 columns (mixed numeric and categorical) to predict customer churn. Walk through your approach.

---
---

## Answer key

**1. B** — Gradient-boosted decision trees. This holds across benchmark studies and practitioner experience, including against deep architectures designed for tabular data.

**2. D** — GPU training speed is not a reason trees win; trees are largely CPU-bound. The other three are the structural reasons.

**3. B** — `σ(w·x + b)` is exactly Week 2's neuron.

**4. B** — Variance. Averaging many decorrelated high-variance trees cancels variance while preserving signal.

**5. A** — Bias. Each sequential tree corrects the ensemble's remaining errors.

**6. B** — Underfitting means the model is too simple to represent the target function, which is a capacity problem — no amount of extra training fixes it.

**7. B** — Generally safe; more trees means more averaging.

**8. B** — Boosting can overfit as rounds accumulate, so early stopping on a validation set is standard.

**9. B** — Splits depend only on the ordering of values within a feature, not their magnitude, so no normalisation is needed.

**10. B** — It computes a statistic of the label, so computing it before the split (or outside CV folds) lets label information from validation rows influence training features.

**11. B** — On tables, domain-informed hand-built features usually beat learned ones.

**12. B** — Mixed modality genuinely needs a network that can process text or images alongside the numeric columns. Small data, interpretability needs, and irrelevant features all favour trees.

**13.** The course is called "AI & ML" but contains **no machine learning outside deep learning** — it goes from "what is ML" in Week 1 straight to neural networks in Week 2 and never returns. Missing entirely: regression, classification with classical models, decision trees, random forests, gradient boosting, clustering, dimensionality reduction, and the bias-variance framework. **Why it matters commercially:** most industry ML is **tabular** — churn, fraud, credit risk, demand forecasting, pricing, lead scoring — and on tabular data gradient boosting generally beats neural networks. A graduate of the course could build a RAG agent but not a churn model, and worse, would not know that gradient boosting was the correct tool. The gap is not a missing topic so much as a missing default.

**14.** **(i) Heterogeneous features** — a table mixes wildly different scales and types (age 0–100, income 0–10⁶, country categorical); trees split each feature independently and are scale-invariant, while networks need everything normalised onto comparable scales. **(ii) No smoothness prior** — networks assume nearby inputs produce nearby outputs, but tabular relationships are frequently discontinuous (a decision flips hard at a threshold); trees model step functions natively while networks must approximate them with many units. **(iii) Small data** — deep learning needs many examples, which was Week 1's first bottleneck, but most business datasets have thousands of rows, and trees are far more sample-efficient. **(iv) Irrelevant features** — real tables carry junk columns; a tree simply never splits on them, whereas a network must learn to drive their weights to zero, spending capacity and data doing so.

**15.** **Gradient boosting is gradient descent performed in function space rather than parameter space.** In Week 2, each step computed the gradient of the loss with respect to the *parameters* and nudged the weights downhill. In boosting, each step computes the residual errors of the current ensemble — which is the gradient of the loss with respect to the *predictions* — and then fits a whole new small tree to point in that direction, adding it to the ensemble scaled by a learning rate. The correspondence is exact enough that the hyperparameters carry over: the **learning rate** plays the same Goldilocks role Week 2 demonstrated (too high overshoots and oscillates, too low never arrives), and the number of boosting rounds plays the role of the number of training iterations. The difference is only in what gets added at each step — a parameter update versus an entire weak model.

**16.** **How trees are trained:** bagging trains them **independently and in parallel**, each on a bootstrap sample of rows with a random feature subset at each split; boosting trains them **sequentially**, each fitting the residual errors of the ensemble built so far. **What each tree looks like:** bagged trees are **deep** — individually low-bias but high-variance; boosted trees are **shallow** — individually high-bias but low-variance. **Error component reduced:** bagging reduces **variance** by averaging decorrelated models; boosting reduces **bias** by iteratively correcting what the ensemble still gets wrong. **Overfitting risk:** adding trees to a forest is **generally safe** — more averaging is more stabilisation; adding boosting rounds **can overfit**, because each round further fits the training residuals, so boosting requires early stopping on a validation set. This asymmetry is the practical reason random forests are a forgiving default and boosted models need tuning.

**17.** **Bias** is error from wrong or oversimplifying assumptions — the model cannot represent the target function, which shows up as **underfitting**: poor performance on training *and* test data. **Variance** is error from sensitivity to the particular training sample — the model fits noise, which shows up as **overfitting**: excellent training performance with poor test performance. Total error decomposes as bias² + variance + irreducible noise, and reducing one typically increases the other. **Mapping Week 2's XOR failure:** with only two hidden neurons, the network produced ~0.5 — total uncertainty — on two of the four inputs, and it did so *on the training data itself*. Bad on train and bad on test is the signature of **high bias / underfitting**, and the lecture correctly diagnosed it as a **capacity** problem rather than an optimisation one. No number of extra iterations would have helped; only adding neurons did. Week 7's dropout is the mirror image — a deliberate increase in bias to buy a reduction in variance.

**18.** Three reasons. **It is a baseline you must beat.** If a tuned gradient-boosted model cannot outperform logistic regression, something is wrong — a bug in the pipeline, leakage, a broken target, or a problem with no learnable signal — and finding that out from a model that trains in seconds is far cheaper than debugging the complex one. **It is fast and cheap**, so it costs almost nothing to run and gives you a number to anchor every subsequent result against. **It is interpretable** — each coefficient is a per-unit effect on the outcome, which regulated domains such as lending and insurance frequently require, and which is useful everywhere for sanity-checking that the model has learned something plausible rather than an artefact. A caveat worth carrying: coefficients are only interpretable when features are not strongly correlated with one another.

**19.** The promise that deep learning **learns features automatically** was Week 1's central story — it is what replaced hand-written rules, and it works because images and text have **exploitable structure**. In an image, nearby pixels are related, so convolution and attention can build edges into shapes into objects; in text, tokens have sequential and syntactic structure that attention can exploit. The architecture encodes a prior about how the input is organised, and that prior is correct. **A tabular row has no such structure.** Column 3 sitting next to column 4 means nothing — the ordering is arbitrary, and there is no locality, sequence, or hierarchy for an architecture to exploit. There is therefore no useful prior for the network to bring, and it must discover interactions purely from data, which is exactly what small tabular datasets do not provide enough of. This is why **domain-informed hand-built features** — a `debt/income` ratio, a `days_since_signup`, a per-customer aggregate — usually beat learned ones on tables, and why feature engineering is the biggest lever there while being nearly irrelevant in modern NLP.

**20.** **Understand the data first.** Check the target balance — churn is usually imbalanced, which decides the metric (see S3: accuracy is the wrong choice at 5% positives). Look at missingness patterns, cardinality of the categorical columns, and the distributions of the numerics. Check whether any column could not possibly be known at prediction time — a cancellation date or a churn-reason field is leakage, and at 60 columns something like that is likely present. **Split before anything else.** Train/validation/test, and if the data has a time dimension, split **by time** rather than randomly, since predicting the past from the future is leakage. All preprocessing gets fitted on train only. **Baseline with logistic regression.** Simple encoding, write the number down, and confirm the pipeline runs end to end. **Then gradient boosting** — with 60 mixed columns and 40,000 rows this is squarely tree territory, and CatBoost is the natural choice for its native categorical handling and strong defaults. Use early stopping on the validation set. **Invest in feature engineering**, which is where most of the gain will come from: ratios between related numerics, date decomposition, per-customer aggregates over any transaction history, and careful encoding of high-cardinality categoricals — with target encoding computed **inside CV folds only**. **Tune modestly** — learning rate, depth, and regularisation, with early stopping throughout. **Evaluate honestly** on the untouched test set with a metric matched to the business decision (precision/recall or PR-AUC rather than accuracy for imbalanced churn), and report a confidence interval rather than a bare point estimate. **Finally, interpret** with feature importance and SHAP values, both to explain the model to stakeholders and to sanity-check that the top features are plausible rather than leaked. **What I would not do:** start with a neural network. 40,000 rows of mixed tabular data is the case where trees win, and using a network here would be slower to build, harder to tune, and probably less accurate.

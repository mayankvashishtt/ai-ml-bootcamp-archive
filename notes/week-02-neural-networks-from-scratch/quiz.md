# Week 2 — Quiz (20 questions)

**Topic:** Neural Networks from Scratch — how machines actually learn
**Format:** Q1–12 multiple choice · Q13–20 short answer / code reasoning
**Answer key at the bottom.**

---

## Multiple choice

**1.** The anatomy of a neuron is best written as:
- A) `output = Σwᵢxᵢ`
- B) `output = activation(Σwᵢxᵢ + b)`
- C) `output = activation(Σxᵢ) + b`
- D) `output = w · activation(x) + b`

**2.** Stacking two linear layers `y = W₁x + b₁` and `z = W₂y + b₂` produces:
- A) A quadratic function of x
- B) A single equivalent linear function `z = W'x + b'`
- C) A non-linear function due to matrix multiplication
- D) An undefined operation

**3.** Which activation is described as "the breakthrough" that enabled deep networks?
- A) Sigmoid
- B) Tanh
- C) ReLU — `max(0, x)`
- D) Softmax

**4.** In MSE, errors are squared primarily so that:
- A) The computation runs faster
- B) All errors become positive and large errors are penalised more heavily
- C) The result is always an integer
- D) Gradients become constant

**5.** The maximum value of the sigmoid derivative is:
- A) 1.0
- B) 0.5
- C) 0.25
- D) 0.1

**6.** `0.25¹⁰ ≈ 9.5e-07` was computed in the notebook to demonstrate:
- A) Why sigmoid is the best activation
- B) The vanishing gradient problem in deep networks
- C) How to compute exponentials in NumPy
- D) The learning rate schedule

**7.** Minsky and Papert's 1969 result was that a single-layer perceptron:
- A) Cannot learn XOR
- B) Cannot learn AND
- C) Trains too slowly to be useful
- D) Requires too much memory

**8.** The XOR network in the notebook has how many total parameters?
- A) 8
- B) 17
- C) 20
- D) 4

**9.** In the weight update `weights -= learning_rate * gradient`, subtraction is used because:
- A) Gradients are always negative
- B) The gradient points uphill toward higher loss, so we move against it
- C) It prevents numerical overflow
- D) It is a convention with no mathematical meaning

**10.** With only 2 hidden neurons, the network output ~0.5 on two of four inputs. This is:
- A) An optimisation problem fixed by more iterations
- B) A capacity problem — the architecture cannot represent the function
- C) A bug in the loss function
- D) Correct behaviour for XOR

**11.** In PyTorch, `loss.backward()` corresponds to which part of the NumPy implementation?
- A) The `forward()` function
- B) The entire `backward()` function's gradient computation
- C) The loss calculation only
- D) Weight initialisation

**12.** `optimizer.zero_grad()` is required because PyTorch:
- A) Deletes the model between iterations
- B) Accumulates gradients by default rather than overwriting them
- C) Cannot compute gradients twice
- D) Needs to free GPU memory

---

## Short answer

**13.** Show algebraically why two stacked linear layers collapse into one, and state the consequence for a 100-layer linear network.

**14.** List the four steps of the training loop, and explain what stays the same between a 17-parameter XOR net and a 1-trillion-parameter GPT-4.

**15.** Explain the "blindfolded on a hilly landscape" analogy, mapping each element to its technical counterpart.

**16.** Why must weights be initialised randomly while biases can safely be zero?

**17.** Contrast the failure modes of a learning rate that is too high versus too low. Include the observed numbers from the notebook.

**18.** In the backward pass, `hidden_error = np.dot(output_delta, weights_hidden_output.T)`. Explain in words what this line does and why it is the essence of backpropagation.

**19.** A colleague reports high training loss. Describe how you would distinguish an optimisation problem from a capacity problem, and what you would change in each case.

**20.** The shipped notebook's training output begins at loss 0.000265 rather than ~0.25. What happened, why is it misleading, and how do you fix it?

---
---

## Answer key

**1. B** — `output = activation(Σwᵢxᵢ + b)`. Weighted sum, plus bias, passed through a non-linearity.

**2. B** — It collapses to `z = (W₂W₁)x + (W₂b₁ + b₂)`, i.e. `W'x + b'`, a single linear function.

**3. C** — ReLU, `max(0, x)`. Simple, efficient, and it made deep networks trainable.

**4. B** — Squaring makes all errors positive (so over- and under-prediction both count) and penalises large errors disproportionately.

**5. C** — 0.25, occurring at x = 0.

**6. B** — The vanishing gradient problem. Multiplying derivatives ≤ 0.25 through 10 layers leaves a gradient near zero, so early layers stop learning.

**7. A** — A single-layer perceptron cannot learn XOR. The notebook credits this result with triggering the first AI winter; hidden layers were the resolution.

**8. B** — 17: (2×4 = 8 weights + 4 biases) + (4×1 = 4 weights + 1 bias).

**9. B** — The gradient points in the direction of steepest *increase* in loss; descending means stepping opposite it.

**10. B** — A capacity problem. Two hidden neurons cannot represent XOR's decision boundary, so the network sits at maximum uncertainty on the cases it cannot separate. More iterations will not help; more neurons will.

**11. B** — `loss.backward()` performs all the gradient computation that the hand-written `backward()` function did explicitly. (`optimizer.step()` performs the weight updates.)

**12. B** — PyTorch accumulates gradients into `.grad` by default. Without zeroing, gradients from previous iterations sum into the current update and corrupt training.

**13.** Substituting layer 1 into layer 2: `z = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`. Setting `W' = W₂W₁` and `b' = W₂b₁ + b₂` gives `z = W'x + b'` — a single linear layer. **Consequence:** a 100-layer purely linear network has exactly the representational power of one linear layer. Depth buys nothing without non-linearity; that is why activation functions are mandatory rather than optional.

**14.** (i) **Forward pass** — data flows through to produce a prediction; (ii) **Calculate loss** — measure error against ground truth; (iii) **Backpropagate** — trace error backward to assign blame to each weight; (iv) **Update weights** — step downhill via gradient descent. **What stays the same:** the loop itself, in full. GPT-4 differs in scale (data, parameters, compute) and architecture, but is trained by the identical four-step cycle. Understanding this loop means understanding how frontier models are trained.

**15.** *Blindfolded person* = the training process, which cannot see the whole solution space. *Hilly landscape / loss surface* = every possible error the model can make across parameter settings. *Slope under your feet / gradient* = the local direction of steepest increase in loss, so its negative is "downhill." *Step size / learning rate* = how far you move each iteration. *Lowest valley / global minimum* = the parameter setting with the lowest achievable error. *Stepping repeatedly / gradient descent* = the iterative optimisation procedure.

**16.** Weights must be random to **break symmetry**. If all neurons in a layer began with identical weights they would compute identical outputs, receive identical gradients, and update identically — remaining clones forever, so the layer would have the representational power of a single neuron. Random initialisation makes each neuron learn a different feature. Biases can be zero because the weights already differ, so symmetry is broken without them; a zero bias is a neutral starting offset.

**17.** *Too high (lr = 100):* steps overshoot the minimum, so loss **oscillates or explodes** and may diverge entirely — like taking mile-long strides hunting for a small dip. *Too low (lr = 0.001):* after 10,000 iterations predictions were still 0.4977, 0.4852, 0.4828, 0.4716 against targets 0, 1, 1, 0 — essentially unmoved from the ~0.5 starting point. Learning is not wrong, merely so slow as to be useless. The learning rate is a Goldilocks parameter: too large diverges, too small never arrives.

**18.** It takes the error signal computed at the output layer (`output_delta`) and pushes it back through the **transpose of the same weights that carried the signal forward**, distributing responsibility to each hidden neuron in proportion to how strongly it influenced the output. That is backpropagation in one line: the forward path determines how blame flows backward, so a hidden neuron connected by a large weight receives a correspondingly large share of the error. Multiplying by `sigmoid_derivative(z_hidden)` then converts that blame into a gradient with respect to the pre-activation values.

**19.** **Diagnose by varying one thing at a time.** First attack optimisation: sweep the learning rate across orders of magnitude (e.g. 0.001 → 100) and check whether loss diverges (too high) or crawls (too low); watch the loss curve's shape. If a well-tuned learning rate still plateaus at high loss, and the model cannot even **overfit a tiny subset** of the training data, the problem is capacity. *Fix for optimisation:* adjust learning rate, change optimiser, tune initialisation. *Fix for capacity:* add neurons or layers so the architecture can represent the target function — as adding hidden neurons fixed the 2-neuron XOR failure.

**20.** The notebook cells were **executed out of order**: the training cell was re-run without first re-running the weight-reset cell, so training resumed from already-trained weights and the recorded "initial" loss reflects a converged network rather than a fresh one. It is misleading because the headline result — a loss curve descending from random guessing — is precisely what the notebook claims to demonstrate, and the saved output does not actually show it. **Fix:** restart the kernel and run all cells top to bottom (Runtime → Restart and run all), which reinitialises the weights under `np.random.seed(42)` and produces a genuine curve starting near 0.25.

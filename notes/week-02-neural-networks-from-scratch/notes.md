# Week 2 — Neural Networks from Scratch

**Subtitle:** How Machines Actually Learn
**Date:** 24/01/2026
**Sources:** `downloads/week-02-neural-networks-from-scratch.pdf` (17 slides) · `downloads/week-02-neural-networks-from-scratch.ipynb` (49 cells)
**Notion page:** https://100xschool.notion.site/2f2ffffa33e58020964cf052345cedf0

**Notebook goal, in its own words:** *"By the end of this notebook, you'll understand exactly what happens when someone says 'the model is training.' No magic, no hand-waving."*

The notebook builds a network in **pure NumPy**, trains it on a problem a single neuron provably cannot solve, then shows PyTorch doing the same thing in a fraction of the code. That progression — do it by hand, then see what the framework automates — is the whole point of the week.

---

## 1. Recap: rules vs patterns

| Traditional programming | Machine learning |
|---|---|
| **Rules don't scale** — humans can't write a rule for every scenario | **Pattern discovery** — find consistent mathematical relationships in large datasets |
| **Rigid logic** — fails on data that doesn't fit predefined rules | **The goal this week:** understand the mechanics of *how* that discovery happens |

---

## 2. The simplest prediction problem

| Size (sq ft) | Price |
|---|---|
| 1,000 | $200,000 |
| 1,500 | $300,000 |
| 2,000 | $400,000 |
| **2,500** | **?** |

You can see it instantly: $200/sq ft. **The challenge is teaching a machine to find that relationship automatically** — without being told the rule.

---

## 3. Machine learning in one sentence

> **Training is a simple, iterative loop of guessing and correcting.**

1. **Start with a guess** — initialise with random values
2. **Measure error** — how far is the guess from the truth?
3. **Adjust** — modify the guess to be slightly less wrong
4. **Repeat** — millions of times, until error is minimised

This is Week 1's definition of learning (try → feedback → adjust → repeat) rendered as an algorithm. Same shape, now mechanical.

---

## 4. The neuron

The building block is **a simple mathematical function, not a biological mystery.** Numbers in, arithmetic, signal out.

```
output = activation(Σ wᵢxᵢ + b)
```

| Part | Role |
|---|---|
| **Inputs (x)** | Numerical features — house size, pixel intensity |
| **Weights (w)** | The strength/importance of each input signal |
| **Bias (b)** | An offset letting the neuron shift its decision boundary |
| **Activation** | A non-linear function deciding whether the signal "fires" |

Weights and biases are **what training changes**. Everything else about the architecture stays fixed.

---

## 5. Why activation functions are mandatory

This is the most important derivation in the lecture. Without non-linearity, **depth is an illusion.**

Take two stacked linear layers:

```
Layer 1:  y = W₁x + b₁
Layer 2:  z = W₂y + b₂

Substitute:
z = W₂(W₁x + b₁) + b₂
z = (W₂W₁)x + (W₂b₁ + b₂)
z = W'x + b'          where W' = W₂W₁ and b' = W₂b₁ + b₂
```

**Mathematical collapse.** Two layers reduce to one layer with different numbers. So do a hundred. Stacking linear functions gives you a linear function — no matter how many you stack.

**The non-linear solution:**
- **Breaking linearity** — activations add "bends," preventing layers from collapsing into each other
- **Enabling depth** — this is the secret sauce that lets deep networks learn layered representations
- **Universal approximation** — with non-linearity, networks can model *any* continuous function

### The activation menu

| Function | Range | Notes |
|---|---|---|
| **Sigmoid** | (0, 1) | Classic S-curve. Historically the default; natural fit for probabilities. |
| **Tanh** | (−1, 1) | Zero-centred, often converges faster than sigmoid. |
| **ReLU** | [0, ∞) | `max(0, x)`. The breakthrough — simple, cheap, and what made deep networks trainable. |
| **GELU / Swish** | — | Smooth ReLU variants used in state-of-the-art LLMs. |

---

## 6. The loss function

> **Loss = a single number measuring how wrong we are.** Lower loss = better predictions. Training goal = minimise loss.

**Mean Squared Error (MSE):**

```
L = (1/n) Σ (prediction − target)²
```

- Squaring makes all errors positive (over- and under-shooting both count)
- Squaring penalises **big errors much more** than small ones
- Good for regression tasks

**Worked example from the slides:**
```
Predicted: $350,000
Actual:    $400,000
Error:     $50,000
Squared:   2,500,000,000
```

Collapsing all performance into one number is what makes optimisation possible — you can't descend a surface with no height.

---

## 7. Backpropagation — assigning blame

> **The question:** "The prediction was wrong. Which specific weights are responsible?"

- **The flow:** error signals travel **backward** from the output layer, through the hidden layers, to the input.
- **The adjustment:** each weight is adjusted **proportionally to its contribution** to the final mistake.

The notebook's framing — *"blame flows backward"* — is the intuition. The machinery is the chain rule from calculus, but you can hold the concept without the derivation.

**The chain of blame:**
1. Calculate error at the output
2. Work out how much each output weight contributed
3. Propagate the error back to the hidden layer
4. Work out how much each hidden weight contributed
5. Adjust all weights proportionally

---

## 8. Gradient descent — finding the valley

**The analogy:** you're blindfolded on a hilly landscape, trying to reach the lowest valley.

- **The landscape** = the *loss surface*, representing every error the model could make
- **The gradient** = the slope under your feet, telling you which way is "down"
- **The strategy** = feel the slope, step downhill, repeat
- **The goal** = reach the global minimum, where error is as low as possible

**Learning rate** = how big each step is. It's the single most consequential hyperparameter, and the notebook proves it experimentally (§11).

---

## 9. The training loop

```
01 Forward Pass      → data flows through the network, producing a prediction
02 Calculate Loss    → measure error against ground truth
03 Backpropagate     → trace error backward, assign blame to each weight
04 Update Weights    → step downhill via gradient descent
   ↺ repeat
```

**Scaling to the moon:** the XOR network in the notebook has **~20 parameters**. GPT-4 has **~1 trillion**. The loop is *identical* — forward pass, loss, backprop, update. Only the scale changes. This is the week's biggest idea: you have now seen, in full, the mechanism that trains frontier models.

---

## 10. The notebook — XOR from scratch

### Why XOR?

| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Different inputs → 1. Same inputs → 0.

**The historical weight:** in 1969 **Minsky and Papert proved a single-layer perceptron cannot learn XOR.** The notebook attributes the first AI winter to this result — people concluded neural networks were fundamentally limited. **The solution was hidden layers**, and this notebook demonstrates it directly.

Geometrically: plot the four points and try to separate the classes with **one straight line**. You can't. XOR is not linearly separable — you need a non-linear boundary, which is exactly what a hidden layer plus non-linear activation provides.

### Architecture

```
Input (2) → Hidden (4, sigmoid) → Output (1, sigmoid)
```

**Total parameters: 17** — (2×4 weights + 4 biases) + (4×1 weights + 1 bias).

Sigmoid is used throughout for pedagogy: output is naturally 0–1, the maths is clean, and it's historically important. *In practice you'd use ReLU for hidden layers.*

### Initialisation

```python
weights_input_hidden  = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden           = np.zeros((1, HIDDEN_SIZE))
weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output           = np.zeros((1, OUTPUT_SIZE))
```

Weights random, **biases zero**. Weights must be random to break symmetry — if every neuron started identical, they'd receive identical gradients and stay identical forever, making the hidden layer useless.

### Sigmoid and its derivative

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

⚠️ **The vanishing gradient problem, demonstrated numerically.** The maximum value of the sigmoid derivative is only **0.25**. Backprop multiplies these derivatives layer by layer, so through 10 layers:

```
0.25¹⁰ = 0.00000095   (9.54e-07)
```

The gradient reaching early layers is essentially zero — **they stop learning.** This is the concrete reason ReLU mattered so much: its derivative is 1 for positive inputs, so it doesn't shrink the signal.

### Forward pass

```python
def forward(X):
    z_hidden = np.dot(X, weights_input_hidden) + bias_hidden   # linear
    a_hidden = sigmoid(z_hidden)                                # activation
    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = sigmoid(z_output)
    return z_hidden, a_hidden, z_output, a_output
```

Note it returns **all intermediate values** — the pre-activation `z`s are needed during backprop.

Untrained, predictions hover around 0.41–0.45 for every input: the network is guessing, and the initial loss is **0.2557**.

### Backward pass

```python
# OUTPUT LAYER
output_error = a_output - y
output_delta = output_error * sigmoid_derivative(z_output)
grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m
grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

# HIDDEN LAYER  — propagate error backward
hidden_error = np.dot(output_delta, weights_hidden_output.T)
hidden_delta = hidden_error * sigmoid_derivative(z_hidden)
grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

# UPDATE  — move opposite the gradient
weights_hidden_output -= learning_rate * grad_weights_hidden_output
weights_input_hidden  -= learning_rate * grad_weights_input_hidden
```

Two things to internalise:

1. **`hidden_error = output_delta @ W_ho.T`** — *this line is backpropagation.* The error is pushed back through the same weights that carried the signal forward, transposed.
2. **`-=` not `+=`** — the gradient points *uphill* (toward higher loss), so you subtract to descend.

### Training and results

Hyperparameters: `learning_rate = 2.0`, `iterations = 10000`.

Final predictions:

| Input | Target | Prediction | ✓ |
|---|---|---|---|
| [0 0] | 0 | 0.0124 | ✅ |
| [0 1] | 1 | 0.9898 | ✅ |
| [1 0] | 1 | 0.9897 | ✅ |
| [1 1] | 0 | 0.0098 | ✅ |

**The network learned XOR from random weights** — no rule was ever written down.

> 🔍 **Gotcha in the shipped notebook:** the saved training output starts at `Iteration 0 | Loss: 0.000265`, which is already near-zero. A freshly initialised network should start around 0.25. The cells were evidently re-run without re-executing the reset cell, so training resumed from already-trained weights. **Re-run the notebook top to bottom** to see the real loss curve — otherwise the headline "loss decreased" plot is misleading.

---

## 11. Breaking it — the three experiments

> *"Understanding what breaks a network teaches you more than seeing it work."*

This section is the most valuable part of the notebook and the most likely to be examined.

### Experiment 1 — Learning rate too high (`lr = 100.0`)

Loss **explodes or oscillates**. Steps are so large you overshoot the valley and bounce between walls, sometimes diverging entirely. Back to the analogy: you're taking mile-long strides looking for a small dip.

### Experiment 2 — Learning rate too low (`lr = 0.001`)

After 10,000 iterations:

| Input | Prediction | Target |
|---|---|---|
| [0 0] | 0.4977 | 0 |
| [0 1] | 0.4852 | 1 |
| [1 0] | 0.4828 | 1 |
| [1 1] | 0.4716 | 0 |

Everything is still ~0.5 — barely moved from initialisation. Learning is **painfully slow**. Not wrong, just useless in practice.

**The lesson:** learning rate is a Goldilocks parameter. Too high diverges, too low never arrives.

### Experiment 3 — Not enough hidden neurons (2 instead of 4)

| Input | Prediction | Target |
|---|---|---|
| [0 0] | 0.0185 | 0 ✅ |
| [0 1] | 0.4993 | 1 ❌ |
| [1 0] | 0.9832 | 1 ✅ |
| [1 1] | 0.5005 | 0 ❌ |

It gets two cases right and **sits at 0.5 — total uncertainty — on the other two.** The network lacks the **capacity** to represent the function. This is *underfitting*: not a training failure, an architecture failure. No amount of extra iterations fixes it.

**Distinguish carefully:** bad learning rate = *optimisation* problem. Too few neurons = *capacity* problem. Same symptom (high loss), different cause, different fix.

---

## 12. The PyTorch version

```python
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2.0)

for i in range(10000):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()   # clear previous gradients
    loss.backward()         # compute gradients — THIS IS BACKPROP
    optimizer.step()        # update weights
```

**The mapping to your hand-written code:**

| Your NumPy code | PyTorch equivalent |
|---|---|
| The entire `backward()` function | `loss.backward()` |
| The `weights -= lr * grad` lines | `optimizer.step()` |
| Manually zeroing/recomputing grads | `optimizer.zero_grad()` |
| `np.dot(X, W) + b` | `nn.Linear(in, out)` |

**Why `zero_grad()` exists:** PyTorch *accumulates* gradients by default rather than overwriting them. Forget it and you sum gradients across iterations, corrupting every update. It's the classic beginner bug.

Both implementations reach essentially the same result (final loss ~1e-4). **The concepts are identical; PyTorch just automates the calculus.**

---

## Key takeaways

1. Neural networks are **layers of simple neurons doing weighted sums** plus a non-linearity.
2. **Training is the iterative adjustment of weights to minimise loss** — guess, measure, adjust, repeat.
3. **Backpropagation traces error backward** to assign blame to each weight.
4. **Gradient descent is the strategy** for stepping toward lower loss.
5. **Without non-linear activations, depth collapses** — 100 linear layers equal 1.
6. **Sigmoid's max derivative is 0.25**, which is why deep sigmoid networks suffer vanishing gradients and why ReLU won.
7. **XOR needs a hidden layer** — the 1969 Minsky/Papert result and its resolution.
8. **The same 4-step loop** trains a 17-parameter XOR net and a trillion-parameter GPT-4.

**Next class:** Transformers & Attention →

---

## Glossary

| Term | Meaning |
|---|---|
| **Neuron** | `output = activation(Σwᵢxᵢ + b)` — weighted sum plus non-linearity. |
| **Weight / bias** | Learned parameters: input importance, and an offset shifting the decision boundary. |
| **Activation function** | Non-linearity preventing layer collapse — sigmoid, tanh, ReLU, GELU, Swish. |
| **Linear collapse** | Stacked linear layers reduce algebraically to a single linear layer. |
| **Loss function** | One number quantifying wrongness; MSE = mean squared difference. |
| **Forward pass** | Data flowing input → output to produce a prediction. |
| **Backpropagation** | Propagating error backward to attribute blame to each weight. |
| **Gradient** | The slope of the loss surface; points uphill, so we step against it. |
| **Gradient descent** | Iteratively stepping downhill on the loss surface. |
| **Learning rate** | Step size. Too high → oscillation/divergence; too low → glacial progress. |
| **Loss surface** | The landscape of all possible errors over parameter settings. |
| **Global minimum** | The lowest point on that surface — minimum achievable error. |
| **Vanishing gradient** | Gradients shrinking toward zero through depth (sigmoid derivative ≤ 0.25). |
| **XOR** | Exclusive-or; the canonical non-linearly-separable problem. |
| **Linear separability** | Whether classes can be split by a single straight line. XOR cannot. |
| **Capacity** | How complex a function an architecture can represent; too little → underfitting. |
| **Symmetry breaking** | Random weight init so neurons learn different features instead of staying identical. |
| **`zero_grad()`** | Clears accumulated gradients in PyTorch; omitting it is a classic bug. |

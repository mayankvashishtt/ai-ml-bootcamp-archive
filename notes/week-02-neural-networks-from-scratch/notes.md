# Week 2 — Neural Networks from Scratch

**Subtitle:** How Machines Actually Learn
**Date:** 24/01/2026
**Sources:** `downloads/week-02-neural-networks-from-scratch.pdf` (17 slides) · `downloads/week-02-neural-networks-from-scratch.ipynb` (49 cells)
**Notion page:** https://100xschool.notion.site/2f2ffffa33e58020964cf052345cedf0

**Notebook goal, in its own words:** *"By the end of this notebook, you'll understand exactly what happens when someone says 'the model is training.' No magic, no hand-waving."*

---

## 0. The idea in plain language

This is the most important week in the course. Everything afterwards — transformers, fine-tuning, RLHF — is a variation on what happens here. If one week deserves a second pass, it's this one.

**The problem:** you want a machine to learn a relationship you never explicitly tell it.

**The solution, in four steps that never change:**

1. **Guess.** Start with completely random numbers. The output is garbage.
2. **Measure how wrong you were.** Reduce that wrongness to a single number.
3. **Work out which numbers to blame**, and by how much.
4. **Nudge every number slightly in the direction that reduces the wrongness.**

Repeat a few million times. That's it. That's training.

**The analogy that carries the whole week:** you're standing blindfolded on a hilly landscape and want to reach the lowest point. You can't see anything, but you can feel the slope under your feet. So you feel which way is downhill, take a small step that way, and repeat. Eventually you're at the bottom of a valley.

- The **landscape** is how wrong the model is, for every possible setting of its numbers.
- Your **position** is the current setting of those numbers.
- The **slope** is the gradient.
- **Step size** is the learning rate — and getting it wrong is how most training fails.

**The thing that should genuinely surprise you:** the network in this notebook has **17 numbers** in it. GPT-4 has roughly a trillion. The four steps above are *identical* for both. Not analogous — identical. When you finish this week you will have seen, in full and with nothing hidden, the mechanism that trains frontier models.

---

## 1. Recap: rules vs patterns

| Traditional programming | Machine learning |
|---|---|
| **Rules don't scale** — humans can't write a rule for every scenario | **Pattern discovery** — find consistent mathematical relationships in large datasets |
| **Rigid logic** — fails on data that doesn't fit predefined rules | Handles the unseen by generalising from what it saw |

Week 1 established *why* we abandoned rules. This week is *how* the alternative actually works.

---

## 2. The simplest prediction problem

| Size (sq ft) | Price |
|---|---|
| 1,000 | $200,000 |
| 1,500 | $300,000 |
| 2,000 | $400,000 |
| **2,500** | **?** |

You spotted it instantly: $200 per square foot, so $500,000.

**But notice what you just did.** You didn't apply a rule someone gave you. You looked at examples, inferred a relationship, and applied it to a new case. **The challenge is getting a machine to do that same inference automatically**, without anyone telling it "the rule is price = 200 × size."

In machine learning terms, the machine needs to discover that the correct value of one number — the multiplier — is 200. Right now it doesn't know that. It starts by guessing, say, 7.

---

## 3. Machine learning in one sentence

> **Training is a simple, iterative loop of guessing and correcting.**

1. **Start with a guess** — initialise with random values
2. **Measure error** — how far is the guess from the truth?
3. **Adjust** — modify the guess to be slightly less wrong
4. **Repeat** — millions of times, until error stops improving

This is Week 1's definition of learning (try → feedback → adjust → repeat) rendered as an algorithm. Same shape, now mechanical. You'll meet it again as the agent loop (Week 8) and the RL loop (Weeks 14–15).

---

## 4. The neuron

The building block is **a simple mathematical function, not a biological mystery.** Numbers go in, arithmetic happens, a number comes out. The name is historical baggage; don't let it intimidate you.

```
output = activation(Σ wᵢxᵢ + b)
```

**Reading that formula piece by piece**, since the notation may be new:

- `Σ` (sigma) means "add up all of these." `Σ wᵢxᵢ` means: multiply each input by its weight, then sum the results.
- If you have two inputs `x₁ = 3` and `x₂ = 5`, with weights `w₁ = 0.5` and `w₂ = 2`, then `Σ wᵢxᵢ = (0.5 × 3) + (2 × 5) = 1.5 + 10 = 11.5`.
- Then add the bias `b`. If `b = -4`, you get `7.5`.
- Then apply the activation function to `7.5` to get the final output.

| Part | Role | Plain meaning |
|---|---|---|
| **Inputs (x)** | Numerical features | The data — house size, pixel brightness |
| **Weights (w)** | Strength of each input | "How much does this input matter?" |
| **Bias (b)** | An offset | "Shift the whole thing up or down" |
| **Activation** | A non-linear function | "Bend the output so stacking layers means something" |

**Weights and biases are what training changes.** Everything else — how many neurons, how they're wired, which activation — stays fixed. When you hear "a model has 7 billion parameters," those parameters *are* the weights and biases.

**Why the bias is needed:** without it, when all inputs are zero the output must be zero. The bias lets the neuron have a baseline — like the intercept of a line. `y = wx` always passes through the origin; `y = wx + b` doesn't have to.

---

## 5. Why activation functions are mandatory

This is the most important derivation in the lecture, and it's short enough to follow completely. **Without non-linearity, depth is an illusion.**

Take two stacked linear layers:

```
Layer 1:  y = W₁x + b₁
Layer 2:  z = W₂y + b₂
```

Substitute the first into the second:

```
z = W₂(W₁x + b₁) + b₂
z = W₂W₁x + W₂b₁ + b₂
z = W'x + b'          where W' = W₂W₁  and  b' = W₂b₁ + b₂
```

**Look at what happened.** Two layers just collapsed algebraically into *one* layer with different numbers. And the same works for three layers, or a hundred. **Stacking linear functions gives you a linear function.** A 100-layer linear network has exactly the representational power of a 1-layer one — you've spent 100× the compute for nothing.

**Concretely:** a linear model can only ever draw a straight line (or in higher dimensions, a flat plane). No matter how many you stack, you can't bend it.

**The fix is to put a non-linear function between the layers.** Now the substitution doesn't work:

```
y = activation(W₁x + b₁)
z = W₂y + b₂            ← can't multiply W₂ into W₁ through the activation
```

The layers can no longer be merged, so depth genuinely buys you something. Each layer transforms the space, bends it, and passes it on.

**Three consequences:**
- **Breaking linearity** — activations add "bends," preventing collapse
- **Enabling depth** — layers can now learn progressively more abstract features
- **Universal approximation** — with non-linearity, a sufficiently large network can approximate *any* continuous function

### The activation menu

| Function | Formula | Range | Notes |
|---|---|---|---|
| **Sigmoid** | `1/(1+e⁻ˣ)` | (0, 1) | Classic S-curve. Squashes anything into 0–1, so it's a natural fit for probabilities. Historically the default. |
| **Tanh** | `tanh(x)` | (−1, 1) | Like sigmoid but zero-centred, which usually converges faster. |
| **ReLU** | `max(0, x)` | [0, ∞) | Absurdly simple: negative → 0, positive → unchanged. The breakthrough that made deep networks trainable. |
| **GELU / Swish** | — | — | Smooth ReLU variants used in modern LLMs. (Week 6 covers SwiGLU.) |

**Why ReLU beat the others** is explained properly in §10 once you've seen the vanishing gradient problem numerically — it's not obvious that the simplest function should win, and the reason is genuinely interesting.

---

## 6. The loss function

> **Loss = a single number measuring how wrong we are.** Lower is better. The goal of training is to make this number small.

**Why collapse everything into one number?** Because you cannot descend a landscape that has no height. The optimisation in §8 needs a single quantity to minimise. Every notion of "good" has to be compressed into one scalar — and choosing *how* to compress it is a real design decision with real consequences.

**Mean Squared Error (MSE):**

```
L = (1/n) Σ (prediction − target)²
```

Read it as: for each example, take the difference between what you predicted and what was correct, square it, then average across all examples.

**Why square it?**
- **All errors become positive.** Predicting $50k too high and $50k too low are both equally wrong; without squaring they'd cancel out and average to zero, which would falsely look perfect.
- **Big errors are punished much harder.** An error of 10 contributes 100; an error of 100 contributes 10,000. So the model prioritises fixing large mistakes — usually what you want.

**Worked example from the slides:**
```
Predicted: $350,000
Actual:    $400,000
Error:     $50,000
Squared:   2,500,000,000
```

That number is meaningless in isolation — what matters is whether it goes *down*.

---

## 7. Backpropagation — assigning blame

> **The question:** "The prediction was wrong. Which specific weights are responsible, and how much is each one to blame?"

This is the step people find hardest, so here's the intuition before any machinery.

**Imagine a company that shipped a bad product.** The CEO knows the outcome was bad. To fix it, blame has to be traced backwards: which department contributed, and how much? Then within each department, which team? The blame flows from the final outcome backwards through the organisation, and each person is corrected in proportion to their contribution.

Backpropagation is exactly that, for weights.

- **The flow:** error signals travel **backward** from the output layer, through hidden layers, to the input.
- **The adjustment:** each weight is adjusted **proportionally to its contribution** to the final mistake.

**Why proportionally matters:** a weight that barely influenced the output shouldn't be changed much — it wasn't the problem. A weight that dominated the output and pushed it the wrong way should be changed a lot. Backprop computes exactly how much each weight influenced the loss.

**The chain of blame:**
1. Calculate the error at the output
2. Work out how much each output-layer weight contributed
3. Push the error back to the hidden layer
4. Work out how much each hidden-layer weight contributed
5. Adjust all weights proportionally

The machinery underneath is the **chain rule** from calculus. You do not need to derive it. What you need is: *the chain rule lets you compute how a change deep inside the network affects the final loss, by multiplying together the local effects at each step.*

---

## 8. Gradient descent — finding the valley

### What a gradient actually is

If the word is new: **a gradient is a slope.** For a simple function `y = x²`, the gradient at any point tells you how steeply `y` changes as you nudge `x`. At `x = 3`, the gradient is 6 — meaning if you increase `x` slightly, `y` increases about 6× as fast.

In a neural network, the gradient of the loss with respect to a particular weight answers: **"if I increase this weight a tiny bit, does the loss go up or down, and how fast?"**

That's all you need. If the gradient is positive, increasing the weight increases the loss — so decrease it. If negative, do the opposite.

### The algorithm

**The analogy:** blindfolded on a hilly landscape, trying to reach the lowest valley.

- **The landscape** is the *loss surface* — the loss value for every possible combination of weights
- **The gradient** is the slope under your feet, telling you which way is "down"
- **The strategy** is: feel the slope, take a step downhill, repeat
- **The goal** is a minimum, where error is as low as you can get it

The update rule is one line:

```
new_weight = old_weight − learning_rate × gradient
```

**Note the minus sign.** The gradient points *uphill* (toward higher loss), so you move in the opposite direction. This is the single most common sign error in hand-written training code.

**Learning rate** is how big each step is. It is the most consequential hyperparameter in the whole process, and the notebook proves this experimentally in §11.

> **Do you actually reach the global minimum?** The slides say the goal is the global minimum. In practice, for networks of any real size, you almost never find it — and it turns out not to matter. High-dimensional loss surfaces have enormous numbers of local minima that are nearly as good as each other. This is a genuine and slightly surprising empirical finding: "good enough" minima are everywhere.

---

## 9. The training loop

```
01 Forward Pass      → data flows through the network, producing a prediction
02 Calculate Loss    → measure error against ground truth
03 Backpropagate     → trace error backward, assign blame to each weight
04 Update Weights    → step downhill via gradient descent
   ↺ repeat
```

**Scaling to the moon:** the XOR network below has **17 parameters**. GPT-4 has roughly a **trillion**. The loop is *identical* — forward, loss, backward, update. Only the scale changes.

This is the week's biggest idea, and it's worth pausing on. There is no additional secret at the top. Frontier models are this loop, with more numbers, more data, and a great deal of engineering to make it run on thousands of GPUs (which is S10's subject).

---

## 10. The notebook — XOR from scratch

### Why XOR?

| Input A | Input B | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The rule: **different inputs → 1, same inputs → 0.**

**Why this specific problem carries historical weight:** in 1969 **Minsky and Papert proved that a single-layer perceptron cannot learn XOR.** The result was widely read as showing neural networks were fundamentally limited, and the notebook attributes the first AI winter partly to it. **The resolution is hidden layers**, and this notebook demonstrates it directly — which is why an apparently trivial problem is the right one to start with.

**The geometric reason it's hard.** Plot the four points on a grid:

```
  B
  1 |  (0,1)=1      (1,1)=0
    |
  0 |  (0,0)=0      (1,0)=1
    +------------------------
       0             1        A
```

Try to separate the 1s from the 0s with **one straight line**. You can't — the 1s are diagonally opposite each other, and so are the 0s. XOR is **not linearly separable**. You need a curved or kinked boundary, which is exactly what a hidden layer plus a non-linear activation gives you.

### Architecture

```
Input (2) → Hidden (4, sigmoid) → Output (1, sigmoid)
```

**Total parameters: 17.** Count them: (2 inputs × 4 hidden = 8 weights) + (4 hidden biases) + (4 hidden × 1 output = 4 weights) + (1 output bias) = 17.

Sigmoid is used throughout for teaching reasons: the output is naturally between 0 and 1, the derivative is clean, and it's historically what people used. *In real work you'd use ReLU for the hidden layer.*

### Initialisation

```python
weights_input_hidden  = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden           = np.zeros((1, HIDDEN_SIZE))
weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output           = np.zeros((1, OUTPUT_SIZE))
```

Weights are random; **biases start at zero.**

**Why weights must be random — symmetry breaking.** Suppose every weight started at the same value. Then every hidden neuron computes the same thing, receives the same gradient, and updates identically. They stay identical forever. Four identical neurons have the representational power of one, so the hidden layer is useless. Randomness breaks the tie and lets each neuron specialise.

**Why biases can be zero:** the weights being random is already enough to break symmetry, so there's no need to randomise biases too.

**Why `* 0.5`:** scaling down the initial weights keeps the initial outputs in a range where sigmoid's gradient isn't tiny. Initialisation scale is a real topic in its own right.

### Sigmoid and its derivative

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

⚠️ **The vanishing gradient problem, demonstrated numerically — and this is why ReLU won.**

The sigmoid derivative `s(1−s)` is maximised when `s = 0.5`, giving `0.5 × 0.5 = 0.25`. So **the largest it can ever be is 0.25**, and it's usually much smaller.

Backpropagation *multiplies* these derivatives together as it moves backward through layers. Through 10 layers, the best case is:

```
0.25¹⁰ = 0.00000095   (9.54e-07)
```

The gradient reaching the early layers is essentially zero. **Those layers stop learning entirely.** You can train for a million iterations and the first layers barely move.

**This is the concrete reason ReLU mattered.** ReLU's derivative is exactly **1** for any positive input. Multiply 1 by itself ten times and you still have 1 — the signal passes back through depth undiminished. A function so simple it looks like a placeholder turned out to be the thing that made deep networks trainable.

### Forward pass

```python
def forward(X):
    z_hidden = np.dot(X, weights_input_hidden) + bias_hidden   # linear step
    a_hidden = sigmoid(z_hidden)                                # activation
    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = sigmoid(z_output)
    return z_hidden, a_hidden, z_output, a_output
```

**Reading `np.dot(X, W)` if matrix multiplication is unfamiliar:** it does the `Σ wᵢxᵢ` sum from §4, for every neuron and every example at once. If `X` is 4 examples × 2 features and `W` is 2 features × 4 neurons, the result is 4 examples × 4 neurons — every example scored by every neuron in a single operation. This batching is why GPUs help so much: it's one big parallel multiplication rather than a loop.

**Naming convention:** `z` is the value *before* the activation, `a` is *after*. The function returns **all four** intermediate values because backprop needs the `z`s — you can't compute the derivative without knowing what went into the activation.

Untrained, predictions hover around 0.41–0.45 for every input — the network is guessing. Initial loss: **0.2557**.

### Backward pass

```python
# ---- OUTPUT LAYER ----
output_error = a_output - y                                    # how wrong?
output_delta = output_error * sigmoid_derivative(z_output)     # scale by local slope
grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m
grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)

# ---- HIDDEN LAYER — propagate the error backward ----
hidden_error = np.dot(output_delta, weights_hidden_output.T)   # ← THIS is backprop
hidden_delta = hidden_error * sigmoid_derivative(z_hidden)
grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

# ---- UPDATE — move opposite the gradient ----
weights_hidden_output -= learning_rate * grad_weights_hidden_output
weights_input_hidden  -= learning_rate * grad_weights_input_hidden
```

**Walking through it line by line:**

1. `output_error = a_output - y` — the raw difference between prediction and truth.
2. `output_delta = output_error * sigmoid_derivative(z_output)` — scale the error by how sensitive the output was at this point. If the sigmoid was saturated (output near 0 or 1), its derivative is tiny, so even a large error produces a small update. **This is the vanishing gradient appearing in code.**
3. `grad_weights_hidden_output = np.dot(a_hidden.T, output_delta) / m` — a weight's blame is (what flowed into it) × (the error coming back). Divide by `m` (number of examples) to average.
4. `hidden_error = np.dot(output_delta, weights_hidden_output.T)` — **this single line is backpropagation.** The error is pushed back through the *same weights that carried the signal forward*, transposed. A hidden neuron connected by a large weight receives a large share of the blame; one connected weakly receives little. That's "proportional to contribution," implemented.
5. The update lines use **`-=` not `+=`** — the gradient points uphill, so subtract to descend.

**If you internalise two things from this week, make it lines 4 and 5.**

### Training and results

Hyperparameters: `learning_rate = 2.0`, `iterations = 10000`.

Final predictions:

| Input | Target | Prediction | ✓ |
|---|---|---|---|
| [0 0] | 0 | 0.0124 | ✅ |
| [0 1] | 1 | 0.9898 | ✅ |
| [1 0] | 1 | 0.9897 | ✅ |
| [1 1] | 0 | 0.0098 | ✅ |

**The network learned XOR starting from random numbers.** Nobody wrote the rule. It found a set of 17 values that implement exclusive-or, by nothing more than repeatedly measuring error and stepping downhill.

> 🔍 **Bug in the shipped notebook.** The saved training output starts at `Iteration 0 | Loss: 0.000265`, which is already essentially solved. A freshly initialised network should start around **0.25**. The cells were clearly re-run without re-executing the initialisation cell, so training resumed from already-trained weights. **Re-run the notebook top to bottom** (Runtime → Restart and run all) to see the real loss curve — otherwise the headline "loss decreased" plot is showing you almost nothing.

---

## 11. Breaking it — the three experiments

> *"Understanding what breaks a network teaches you more than seeing it work."*

This is the most valuable part of the notebook. Each experiment isolates one failure mode.

### Experiment 1 — Learning rate too high (`lr = 100.0`)

Loss **explodes or oscillates wildly**. Steps are so large you leap clean over the valley and land on the opposite slope, higher than where you started. Next step, you leap back over. Sometimes it diverges to infinity.

Back to the analogy: you're taking mile-long strides trying to find a small dip in the ground.

### Experiment 2 — Learning rate too low (`lr = 0.001`)

After the full 10,000 iterations:

| Input | Prediction | Target |
|---|---|---|
| [0 0] | 0.4977 | 0 |
| [0 1] | 0.4852 | 1 |
| [1 0] | 0.4828 | 1 |
| [1 1] | 0.4716 | 0 |

Everything is still ~0.5 — barely moved from where it started. Learning is happening, but so slowly it's useless. **Nothing is wrong; it just won't finish this century.**

**The lesson: learning rate is a Goldilocks parameter.** Too high diverges, too low never arrives. In practice people use schedules that start higher and decay, plus adaptive optimisers like Adam that tune per-parameter step sizes automatically.

### Experiment 3 — Not enough hidden neurons (2 instead of 4)

| Input | Prediction | Target | |
|---|---|---|---|
| [0 0] | 0.0185 | 0 | ✅ |
| [0 1] | 0.4993 | 1 | ❌ |
| [1 0] | 0.9832 | 1 | ✅ |
| [1 1] | 0.5005 | 0 | ❌ |

It gets two cases right and **sits at 0.5 — total uncertainty — on the other two.** That 0.5 is the network shrugging: it cannot represent a function that satisfies all four constraints, so it hedges.

This is **underfitting**, and the cause is **capacity**: the architecture is too small to express the required function. No amount of extra training fixes it, because nothing is wrong with the training.

**Distinguish these carefully, because they look identical from the outside:**

| Symptom | Cause | Fix |
|---|---|---|
| Loss stays high, oscillates | Learning rate too high | Lower it |
| Loss decreases glacially | Learning rate too low | Raise it |
| Loss plateaus above zero, predictions hedge | Not enough capacity | Bigger network |

All three present as "high loss." Diagnosing which one you have is the actual skill, and it's the same diagnostic discipline you'll need in Week 17 when an agent misbehaves.

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

**The mapping to the code you just wrote by hand:**

| Your NumPy code | PyTorch equivalent |
|---|---|
| The entire `backward()` function | `loss.backward()` |
| The `weights -= lr * grad` lines | `optimizer.step()` |
| Recomputing gradients from scratch each loop | `optimizer.zero_grad()` |
| `np.dot(X, W) + b` | `nn.Linear(in, out)` |

**The whole of your backward pass is now one line.** PyTorch tracks every operation you perform on tensors, builds a graph of them, and can therefore compute all the derivatives automatically. This is called **autograd**, and it's the single feature that makes deep learning frameworks worth using.

**Why `zero_grad()` exists — and why forgetting it is the classic beginner bug.** PyTorch *accumulates* gradients by default: calling `loss.backward()` twice adds the second set of gradients to the first rather than replacing them. If you forget to zero them, every iteration's update is contaminated by all previous iterations, and training silently degrades. It doesn't crash; it just quietly stops working. (Week 5's notebook has a bug of exactly this family.)

Both implementations reach essentially the same result (final loss ~1e-4). **The concepts are identical; PyTorch just automates the calculus.**

---

## Common confusions

**"Is a neuron like a brain cell?"** Only in loose inspiration. A neuron here is a weighted sum plus a bend. The biological analogy is historical and mostly unhelpful — treat it as arithmetic.

**"Why don't we just solve for the best weights directly?"** For a simple linear model you can (it's called least squares). For a network with non-linearities and millions of parameters, no closed-form solution exists, so iterative descent is the only practical option.

**"Do we reach the global minimum?"** Almost never, and it doesn't matter much. In high dimensions there are vast numbers of local minima of similar quality. See §8.

**"What's the difference between a parameter and a hyperparameter?"** A **parameter** is learned by training (weights, biases). A **hyperparameter** is chosen by you before training (learning rate, number of hidden neurons, number of iterations). Experiments 1–3 are all hyperparameter failures.

**"Loss went down — is the model good?"** Not necessarily. Loss going down means it fits the *training* data better. It could be memorising rather than generalising. This week never checks a held-out set, which is fine for XOR (there are only 4 possible inputs) but is a serious omission for anything real. S1 and S3 cover this properly.

**"Why sigmoid here if ReLU is better?"** Purely pedagogical — clean derivative, 0–1 output, historical importance. The notebook then uses it to *demonstrate* why ReLU is better, which needs sigmoid to be present.

**"17 parameters vs a trillion — surely something else is going on at that scale?"** More engineering, yes (S10). More conceptual machinery, no. The loop is the loop.

---

## Key takeaways

1. **Neural networks are layers of simple neurons** doing a weighted sum plus a non-linearity. No mystery, just arithmetic.
2. **Training is the iterative adjustment of weights to minimise loss** — guess, measure, assign blame, step downhill, repeat.
3. **Loss compresses all performance into one number**, because you cannot descend a landscape with no height.
4. **A gradient answers "if I nudge this weight, does loss go up or down, and how fast?"** — and you step in the opposite direction.
5. **Backpropagation assigns blame proportionally**, pushing error back through the same weights that carried the signal forward.
6. **Without non-linear activations, depth collapses** — 100 linear layers are algebraically equal to 1.
7. **Sigmoid's derivative maxes out at 0.25**, so gradients vanish through depth (0.25¹⁰ ≈ 1e-6). ReLU's derivative is 1, which is precisely why it won.
8. **Weights must be initialised randomly** to break symmetry, or every neuron learns the same thing forever.
9. **XOR needs a hidden layer** — the 1969 Minsky/Papert result, and its resolution, demonstrated in 17 parameters.
10. **Three failure modes look identical from outside:** learning rate too high, too low, and insufficient capacity. Telling them apart is the skill.
11. **The same 4-step loop** trains a 17-parameter XOR net and a trillion-parameter GPT-4. There is no extra secret at the top.

**Next class:** Transformers & Attention →

---

## Glossary

| Term | Meaning |
|---|---|
| **Neuron** | `output = activation(Σwᵢxᵢ + b)` — a weighted sum plus a non-linearity. |
| **Weight** | A learned number expressing how much an input matters. |
| **Bias** | A learned offset, letting the output be non-zero when inputs are zero. |
| **Parameter** | Any number learned by training — weights and biases collectively. |
| **Hyperparameter** | A setting you choose before training (learning rate, layer sizes). |
| **Activation function** | Non-linearity preventing layer collapse — sigmoid, tanh, ReLU, GELU. |
| **Linear collapse** | Stacked linear layers reduce algebraically to a single linear layer. |
| **Loss function** | One number quantifying wrongness; MSE = mean squared difference. |
| **Forward pass** | Data flowing input → output to produce a prediction. |
| **Backpropagation** | Propagating error backward to attribute blame to each weight. |
| **Gradient** | How fast loss changes as a weight changes; points uphill, so we step against it. |
| **Gradient descent** | Iteratively stepping downhill on the loss surface. |
| **Learning rate** | Step size. Too high → oscillation/divergence; too low → glacial progress. |
| **Loss surface** | The landscape of loss values over all possible weight settings. |
| **Local / global minimum** | A low point / the lowest point on that surface. |
| **Vanishing gradient** | Gradients shrinking toward zero through depth (sigmoid derivative ≤ 0.25). |
| **XOR** | Exclusive-or; the canonical non-linearly-separable problem. |
| **Linear separability** | Whether classes can be split by a single straight line. XOR cannot. |
| **Capacity** | How complex a function an architecture can represent; too little → underfitting. |
| **Underfitting** | Model too simple to capture the pattern; loss plateaus high. |
| **Symmetry breaking** | Random init so neurons learn different features instead of staying identical. |
| **Autograd** | PyTorch's automatic differentiation — why `loss.backward()` replaces your whole backward pass. |
| **`zero_grad()`** | Clears accumulated gradients in PyTorch; omitting it silently corrupts training. |

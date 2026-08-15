# Week 5 — Introduction to Tensors and PyTorch

**Subtitle:** The Math & Tools — Tensors, Matrices & PyTorch
**Date:** 14/02/2026
**Sources:** `downloads/week-05-tensors-and-pytorch.pdf` (39 slides) · `downloads/week-05-tensors-and-pytorch.ipynb` (75 cells)
**Notion page:** https://100xschool.notion.site/308ffffa33e580b596c6f790b1846ecf
**Extra link:** [youtu.be/f5liqUk0ZTw](https://youtu.be/f5liqUk0ZTw)

---

## 0. The idea in plain language

This week is the **retroactive maths lecture**. You've already built everything; now it gets named properly.

Here's the reveal. Look at what you wrote in Weeks 2–4:

```
Week 2:     weights × inputs + bias
Weeks 3–4:  Q = X × Wq
            attention = softmax(Q × K.T) × V
```

> **What were all of these, really? Matrix multiplication.** That's the secret. There isn't a second thing.

**Three ideas, and they're all simpler than they sound:**

**1. A tensor is just "a box of numbers with a shape."** A single number is a 0-dimensional tensor. A list is 1-D. A grid is 2-D. Stack grids and you get 3-D. That's the entire concept — the scary-sounding word means "array with any number of dimensions."

**2. A matrix is a *transformation*, not a filing cabinet.** This is the genuinely useful reframe. When you multiply a vector by a matrix, you're *moving it* — rotating, stretching, squishing, reflecting. And therefore: **training a neural network means searching for the right geometric transformation.** "Adjusting weights" and "finding the transformation that maps inputs to correct outputs" are the same sentence.

**3. PyTorch does the calculus for you.** In Week 2 you hand-wrote a backward pass and it was fiddly and easy to break. PyTorch records every operation you perform, then walks that recording backwards to compute every derivative automatically. One line: `loss.backward()`.

**The practical skill this week actually teaches** is reading **shapes**. `[8, 12, 3, 64]` should become as readable to you as a sentence. Most deep-learning debugging is shape debugging, and most deep-learning bugs are shape mismatches.

---

## Act 1 — Scalars → Vectors → Matrices → Tensors

| Name | What it is | Dimensions | Example |
|---|---|---|---|
| **Scalar** | A single number | **0D** | `temperature = 72`, `coffee_price = 4.99` |
| **Vector** | A list of numbers | **1D** | `location = [28.6139, 77.2090]`, `rgb = [255, 87, 51]` |
| **Matrix** | A grid of rows × columns | **2D** | `[[1,2,3],[4,5,6],[7,8,9]]` |
| **Tensor** | Cube / hypercube | **3D, 4D, 5D…** | batches of batches |

> **It's not a scary new concept. It's just: "What if we keep adding dimensions?"**

Strictly, "tensor" is the umbrella term — a scalar *is* a 0-D tensor, a vector *is* a 1-D tensor. In PyTorch everything is a `torch.Tensor` regardless of dimension count.

### You've already been using these

**Vectors:** the 768-dim word embedding from Week 3; the `[x₁, x₂, x₃]` features fed into a neuron in Week 2.

**Matrices:** the weight matrix `W` where `output = W × input` (Week 2); `Wq`, `Wk`, `Wv` and `scores = Q × K.T` (Week 4). **Weights are matrices. Attention is matrix maths.**

### Why tensors specifically — data has structure

Each extra dimension corresponds to a **real structural fact** about the data. Nothing is abstract here:

```
1. A single sentence: "The cat sat"
   sentence = Matrix(3 tokens, 768 dims)      Shape: [3, 768]           # 2D

2. A batch of 8 sentences processed together
   batch = Stack(8 sentences)                 Shape: [8, 3, 768]        # 3D

3. Split across 12 attention heads
   attention = Split(12 heads)                Shape: [8, 12, 3, 64]     # 4D
```

**Why process a batch at all?** Because GPUs are parallel machines (§Act 2). Running 8 sentences at once costs barely more than running 1, so you'd be wasting the hardware otherwise. Batching is why the first dimension of almost every tensor you'll meet is batch size.

### Shape is everything

```
[8, 12, 3, 64]
 ↑   ↑   ↑   ↑
 |   |   |   └── Head dimension (768 ÷ 12 = 64)
 |   |   └────── Number of tokens in each sentence
 |   └────────── Attention heads
 └────────────── Batch size — how many sentences at once
```

> **Pro-tip: if you understand shape, you can read any model.**

This is the single most practical sentence in the lecture, and it's worth taking literally. When you open unfamiliar model code, the fastest way to understand it is to trace what shape the data has at each line. When something crashes, the error is almost always a shape mismatch, and the fix is almost always a `view` or `permute`.

**Habit worth building now:** print `.shape` after every operation while learning. It feels tedious for a week and then becomes intuition.

---

## Act 2 — Matrix magic

### A matrix is a transformation, not storage

> **A matrix isn't just storage. A matrix is a TRANSFORMATION.** It takes a vector in → gives a different vector out.

Capabilities: **rotate, scale, stretch, reflect, project, transform meaning.**

**Rotation — verified in the notebook:**

```python
v = torch.tensor([1.0, 0.0])                 # a vector pointing right
rotate_90 = torch.tensor([[0., -1.],
                          [1.,  0.]])
rotate_90 @ v      # → [0.0, 1.0]  — now pointing up
rotate_45 @ v      # → [0.707, 0.707]  — pointing diagonally
```

The vector didn't change its *values* by magic — the matrix multiplication moved it. That's what multiplication by a matrix *does*.

**Scaling:**
```
[2 0]   [3]   [6]              Uniform 2× — every component doubled
[0 2] × [4] = [8]

[3  0 ]  [3]                   Stretch 3× horizontally,
[0 0.5] ×[4]                   squish 0.5× vertically
```

The notebook's `plot_transformation()` renders a grid before and after, with basis vectors drawn as arrows. **Run this cell and play with it** — it's the fastest route to the intuition.

| Matrix | Effect |
|---|---|
| `[[cos θ, −sin θ], [sin θ, cos θ]]` | Rotation |
| `[[2, 0], [0, 0.5]]` | Stretch horizontally, squish vertically |
| `[[1, 1], [0, 1]]` | Shear |
| `[[-1, 0], [0, 1]]` | Reflection (horizontal flip) |
| `[[0, 1], [1, 0]]` | Swap x and y |

> **Key insight: in a neural network, the model learns which matrix to use.**
> **Training is just finding the right transformation for your data.**

**This reframes everything from Week 2.** "Adjusting weights via gradient descent" is *searching the space of geometric transformations* for one that maps inputs to correct outputs. The loss surface from Week 2 is a landscape over possible transformations.

And it explains **why depth helps**: each layer applies a transformation, bends the space with a non-linearity, then hands it to the next layer. Ten layers is ten successive reshapings of the space — which is exactly why removing the non-linearity collapses them (Week 2 §5): without a bend between them, ten transformations compose into one.

### Why matrix multiplication is everything

| Operation | Where it appears |
|---|---|
| `X @ W` | Forward pass — learned transformations |
| `dL/dY @ W.T` | Backpropagation — computing gradients |
| `Q @ K.T` | Attention — measuring similarity |
| `one_hot @ W_embed` | Word embeddings — looking up meaning |

> **Every single computation in a modern AI model is essentially a series of matrix multiplications.**

Note that even the embedding *lookup* from Week 3 is a matmul in disguise — a one-hot vector (all zeros except a single 1) times the embedding matrix selects exactly one row. In practice it's implemented as a real lookup for speed, but mathematically it's the same operation.

### The dot product — the atomic operation

```python
v1 = [1, 2, 3];  v2 = [4, 5, 6]
v1 · v2 = (1*4) + (2*5) + (3*6) = 32
```

> **Dot product = similarity.** High → pointing the same direction. Low/zero → different directions. Negative → opposite.

Notebook verification:

| Vectors | Dot product | Meaning |
|---|---|---|
| `[1,2,3] · [1.1,1.9,3.1]` | **14.20** | Similar → high |
| `[1,0] · [0,1]` | **0.00** | Perpendicular → zero |
| `[1,2] · [-1,-2]` | **−5.00** | Opposite → negative |

**Matrix multiplication is just a lot of dot products.** Every cell of the output is one dot product between a row of the first matrix and a column of the second. That's why matmul is the workhorse: it computes thousands of similarity scores in one operation.

### Attention is just dot products

```
Q_sat · K_the = 0.1
Q_sat · K_cat = 0.8    ← high similarity!
Q_sat · K_sat = 0.9

score = softmax(Q @ K.T)
```

> **The W matrices rotate vectors into a space where "relevant" and "similar direction" mean the same thing.**
> **Attention scores are dot products. The model learns the right rotation.**

**This is the best one-sentence explanation of Q/K/V in the whole course**, and it's worth rereading. Raw embeddings don't have the property that "geometrically aligned" equals "contextually relevant." `W_Q` and `W_K` exist precisely to *rotate them into a space where it does*. Then a plain dot product measures relevance — because the transformation made it so.

### Why GPUs exist

| CPU | GPU |
|---|---|
| 8–16 **complex** cores | 5,000+ **simple** cores |
| Optimised for sequential tasks | Optimised for parallel tasks |
| Great for logic, branching, single-threaded speed | Great for the same simple maths thousands of times at once |

> **Matrix multiplication is "embarrassingly parallel"** — every cell of the output can be computed at the same time, because none of them depend on each other.

That last clause is the whole reason GPUs work here. If output cell 5 needed cell 4's result, you'd be stuck computing sequentially. It doesn't, so 5,000 cores can each grab a cell.

**Measured, Python loops vs PyTorch (100×100 matmul):**

```
Python loops: 18.4868 seconds
PyTorch @:     0.012059 seconds
Speedup:       1533x faster
```

**Measured, CPU vs GPU:**

| Size | CPU | GPU | Speedup |
|---|---|---|---|
| 100×100 | 0.09 ms | 0.07 ms | 1.3× |
| 500×500 | 5.17 ms | 0.13 ms | **40.3×** |
| 1000×1000 | 25.15 ms | 0.66 ms | 38.2× |
| 2000×2000 | 189.73 ms | 5.87 ms | 32.3× |
| 4000×4000 | 1474.08 ms | 39.04 ms | 37.8× |

> *"This is why NVIDIA is worth trillions."*

**Look carefully at the 100×100 row: only 1.3× faster.** At small sizes, the cost of shipping data across to the GPU and back dominates the actual computation. **GPUs win on scale, not on every operation** — which is a genuinely useful engineering lesson, and the reason moving a tiny model to GPU sometimes makes it *slower*.

---

## Act 3 — PyTorch

### What it is

Used by **OpenAI, Anthropic, Google DeepMind, Tesla.** Three core capabilities:

1. **Tensors on GPUs** — large speedups over CPU maths
2. **Autograd** — automatic differentiation, no more manual backprop
3. **`nn.Module`** — high-level building blocks

> *If deep learning is the car, PyTorch is the engine and dashboard.*

### Creating tensors

```python
x = torch.tensor(5.0)                  # scalar, 0D
v = torch.tensor([1.0, 2.0, 3.0])      # vector, 1D
m = torch.tensor([[1, 2], [3, 4]])     # matrix, 2D
t = torch.zeros(2, 3, 4)               # 3D, shape [2, 3, 4]

print(t.shape)                          # torch.Size([2, 3, 4])
```

> **Pro-tip: notice `.shape` — you will check this more than anything else.**

### How PyTorch works inside

A tensor stores **two things**: the actual numbers (a flat block of memory), and **metadata** (shape, stride, device). The metadata tells PyTorch how to *interpret* that flat block.

This matters more than it sounds, and the notebook proves it:

```python
x = torch.randn(3, 4)
y = x.view(4, 3)
x.data_ptr() == y.data_ptr()    # True — SAME memory!

x.stride()        # (4, 1)
x.is_contiguous() # True

z = x.T
z.is_contiguous() # False
z.stride()        # (1, 4)  — reversed!
```

**Reshaping doesn't move data — it changes how the same memory is read.**

**Stride explained:** stride `(4, 1)` means "to move one row down, jump 4 numbers forward; to move one column right, jump 1 forward." A transpose doesn't rearrange anything in memory; it just **swaps the strides** to `(1, 4)`, so the same numbers get read column-wise instead of row-wise.

That's why transposing is essentially free — and also why the result is **non-contiguous**, and why you sometimes need `.contiguous()` before `.view()`. `view` requires the data to be laid out in order; if it isn't, PyTorch refuses and asks you to make a real copy first. This error confuses everyone once.

### GPU acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)          # ← THE MAGIC LINE
```

**A rule that will save you an hour:** every tensor in an operation must be on the *same* device. Mixing a CPU tensor with a GPU tensor throws an error. The model must be moved too — `model.to(device)`, not just the data.

### Matrix multiply — the `@` operator

```python
A = torch.randn(3, 4)     # [3, 4]
B = torch.randn(4, 5)     # [4, 5]
C = A @ B                 # [3, 5]
C = torch.matmul(A, B)    # equivalent
```

> **The Rule: inner dimensions MUST match.** `(3,4) @ (4,5) → (3,5)`

The 4s meet in the middle and cancel; the 3 and 5 survive. If they don't match, you get the most common error message in deep learning.

### Reshaping

```python
# .view() — reinterpret data
x = torch.randn(8, 3, 768)
y = x.view(24, 768)                 # [8,3,768] → [24,768]   (8*3 = 24)

# .permute() — swap axes
x = torch.randn(8, 12, 3, 64)
y = x.permute(0, 2, 1, 3)           # [8,12,3,64] → [8,3,12,64]
```

> **The Rule: total number of elements must remain exactly the same.**

**`view` vs `permute` — the distinction that confuses people:** `view` keeps the numbers in the same order and just re-brackets them. `permute` genuinely reorders which axis is which, so the *same element* ends up at a different index. `permute` is exactly what's used to shuffle multi-head attention tensors between "heads-first" and "tokens-first" layouts.

### Broadcasting

```python
A = torch.ones(3, 4)                    # [3, 4]
B = torch.tensor([1, 2, 3, 4])          # [4]
C = A + B                               # [3, 4]  — works!
```

PyTorch "stretches" B to match A — effectively adding `[1,2,3,4]` to **every row**, without ever creating three copies in memory.

> **The Rule: dimensions must be equal, or one of them must be 1.**

This is how a bias vector gets added to an entire batch without writing a loop — and it's why `nn.Linear`'s bias is a single vector rather than one per example.

**Broadcasting is also a silent-bug source.** Two tensors with *compatible but wrong* shapes will happily broadcast and produce a result of unexpected shape rather than an error. If a tensor mysteriously grows a dimension, suspect broadcasting.

### Autograd — the big one

> **This is why PyTorch exists.**

How it works:
1. **Tracks every operation** — set `requires_grad=True` and PyTorch "starts rolling a tape," recording everything done to that tensor
2. **Builds a graph** — a dynamic computation graph linking all the tensors
3. **Computes gradients** — one call to `.backward()` finds every derivative

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 5
y.backward()
print(x.grad)        # tensor(4.0)     — dy/dx = 2x = 4
```

Check that by hand: the derivative of `x² + 5` is `2x`, and at `x = 2` that's 4. PyTorch got there without being told the rule.

**Inspecting the graph:**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y + 2 * x + 1

z.grad_fn           # <AddBackward0>
y.grad_fn           # <PowBackward0>
z.grad_fn.next_functions   # the chain backwards
```

Every tensor produced by an operation remembers **which operation made it**. `.backward()` walks that chain in reverse, applying the chain rule at each node.

**Backpropagation is literally a reverse traversal of this graph.** Week 2's hand-written backward pass was you doing this traversal manually for one small network.

### Manual vs automatic

```python
# Manual backprop (painful)
d_loss_y = 2 * (y_pred - y_true)
d_y_relu = 1 if z > 0 else 0
d_relu_w = x.T
d_loss_w = d_relu_w @ (d_loss_y * d_y_relu)
w -= lr * d_loss_w
# One mistake here = model never trains, and it fails silently.

# PyTorch autograd
loss = criterion(y_pred, y_true)
loss.backward()        # ← ALL OF THE ABOVE IN ONE LINE
optimizer.step()
```

> **Focus on the architecture, not the calculus.**

### `torch.no_grad()` and `zero_grad()`

**`torch.no_grad()`** tells PyTorch to **stop recording**. Use it during inference or evaluation: no graph is built, so it's faster and uses noticeably less memory. Forgetting it during evaluation is a common cause of unexplained memory growth.

**`optimizer.zero_grad()`** — PyTorch **accumulates gradients by default.** Call `.backward()` twice without clearing and they *add together*, silently corrupting every update.

> 🔍 **Bug in the shipped notebook.** Cell 43 is captioned *"After second backward: x.grad = 12.0! Accumulated!"* — but the code calls `x.grad.zero_()` immediately beforehand, so the printed output is actually **6.0, 6.0, 6.0**. The demonstration contradicts its own caption. **To see the real effect, delete the first `x.grad.zero_()`** and you'll get 6.0 then 12.0. Worth doing — accumulation is important and the notebook as shipped never shows it.
>
> *(Ironically, this bug is itself an example of the thing it's trying to teach: gradient state carried over from a previous run.)*

---

## Building models with `nn.Module`

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

> **Every model follows this exact pattern: declare layers in `__init__`, connect them in `forward`.**
> Each layer *is a matrix* — a transformation. That's the entire model.

You never call `forward()` yourself — you call `model(x)`, and `nn.Module` routes it. This matters because `nn.Module` also does bookkeeping around the call (hooks, training/eval mode), which calling `forward()` directly would skip.

### What's inside `nn.Linear`?

```python
layer = nn.Linear(3, 2)        # 3 inputs → 2 outputs

layer.weight    # shape [2, 3]
                # [[ 0.12, -0.45,  0.89],
                #  [-0.34,  0.22, -0.11]]
layer.bias      # shape [2]  → [0.05, -0.02]
```

> **It's just matrix multiplication and vector addition: `y = x @ W.T + b`**

Exactly the neuron equation from Week 2 (`output = Σwᵢxᵢ + b`), vectorised so it handles every neuron and every example in the batch at once. There is nothing else inside.

---

## The 5-line training pattern

```python
for epoch in range(10):
    y_pred = model(x_batch)              # 1. Forward pass
    loss   = criterion(y_pred, y_batch)  # 2. Compute loss
    optimizer.zero_grad()                # 3. Clear old gradients
    loss.backward()                      # 4. Backward pass
    optimizer.step()                     # 5. Update weights
```

| Step | Meaning |
|---|---|
| **1. Forward** | Predict using current weights. *"How did we do?"* |
| **2. Loss** | Measure the error. *"How wrong were we?"* |
| **3. zero_grad** | *"Flush the toilet."* Don't let old gradients leak in. |
| **4. Backward** | The calculus. Find how each weight caused the error. |
| **5. Step** | Nudge weights in the right direction. |

> **This pattern is the "heartbeat" of every DL model. From MNIST to GPT-4, this loop stays the same.**
> **Miss step 3 and gradients accumulate, and your model degrades in a way that doesn't crash.**

Same four-step loop as Week 2 — now with an explicit `zero_grad` because PyTorch accumulates by default.

> **Terminology, since it's used without definition:** an **epoch** is one full pass over the training dataset. A **batch** is the subset processed in one step. An **iteration/step** is one execution of the five lines above. So 1,000 examples with batch size 100 means 10 iterations per epoch.

---

## The demo — MNIST in under 60 seconds

**Goal: 98% accuracy in under 60 seconds.** MNIST is 28×28 greyscale images of handwritten digits 0–9 — the field's "hello world."

| Step | Shape change |
|---|---|
| **1. Flattening** | `[28, 28]` image grid → `[784]` list of pixels |
| **2. Transform 1** | `[784] @ [784, 128]` → `[128]` features |
| **3. Activation** | ReLU — filtering out the noise |
| **4. Transform 2** | `[128] @ [128, 10]` → `[10]` digit scores |

The 10 output numbers are scores for each digit; the highest wins. (Softmax turns them into probabilities — same function as Week 4's attention weights, again on a different axis.)

> **Training is just finding the perfect matrices to map "image pixels" → "correct number label".**

Note what's thrown away by flattening: the 2-D spatial structure of the image. It still hits 98% because handwritten digits are easy. Convolutional networks preserve that structure and do better — and Vision Transformers (S9) solve it a third way, by cutting the image into patches.

---

## Common confusions

**"Is a tensor different from a NumPy array?"** Conceptually almost identical. The differences that matter: PyTorch tensors can live on a GPU, and they carry autograd machinery. You can convert freely with `.numpy()` and `torch.from_numpy()`.

**"What's the difference between `view` and `reshape`?"** `view` requires contiguous memory and never copies. `reshape` will copy if it has to. When learning, use `reshape` and you'll hit fewer errors; use `view` when you want the guarantee of no copy.

**"Why does my transpose break `.view()`?"** Transposing changes strides without moving data, leaving the tensor non-contiguous. Call `.contiguous()` first.

**"Why is my GPU slower than my CPU?"** Probably a small tensor — see the 100×100 row (1.3×). Transfer overhead dominates below a certain size.

**"I moved my data to GPU and it still errors."** You probably didn't move the *model*. Both need `.to(device)`.

**"Why does my memory keep growing during evaluation?"** You're building a computation graph you never use. Wrap evaluation in `torch.no_grad()`.

**"What actually is `requires_grad`?"** A flag saying "track operations on this tensor so I can compute gradients for it later." Model parameters have it set automatically; your input data usually shouldn't.

**"Do I need to understand the calculus to use autograd?"** No — that's the point. You need to understand *what* gradients are (Week 2 §8) and that `.backward()` computes them. The chain rule is handled.

---

## Key takeaways

1. **Everything from Weeks 2–4 was matrix multiplication.** Weights, attention, embeddings, gradients. There is no second mechanism.
2. **Tensor = scalar/vector/matrix generalised** to any number of dimensions. Nothing exotic behind the word.
3. **Shape is the master skill** — `[batch, heads, tokens, head_dim]`. Read shapes and you can read models; most bugs are shape bugs.
4. **A matrix is a transformation** (rotate, scale, shear, reflect), so **training is a search for the right transformation.**
5. **Dot product = similarity**, matmul is many dot products at once, and attention scores are dot products — the W matrices rotate vectors so that alignment *means* relevance.
6. **Matmul is embarrassingly parallel** because output cells don't depend on each other — which is exactly what GPUs exploit. But it only pays off at scale (1.3× at 100×100, ~38× at 4000×4000).
7. **PyTorch tensors are data + metadata.** `.view()` and `.T` change interpretation, not memory — which is why transposes are free but non-contiguous.
8. **Broadcasting** expands smaller tensors automatically; it's how a bias reaches a whole batch, and it's a silent-bug source.
9. **Autograd records a tape** of operations, and `.backward()` traverses it in reverse — replacing Week 2's error-prone manual calculus.
10. **`nn.Linear` is `y = x @ W.T + b`** — the Week 2 neuron, vectorised. Nothing more.
11. **The 5-line loop is universal**, and forgetting `zero_grad()` corrupts training without crashing.

---

## Glossary

| Term | Meaning |
|---|---|
| **Scalar / Vector / Matrix / Tensor** | 0D / 1D / 2D / nD numerical containers. |
| **Shape** | The size of each dimension, e.g. `[8, 12, 3, 64]`. |
| **Stride** | How many memory positions to jump to move one step along a dimension. |
| **Contiguous** | Whether elements are laid out in order in memory. Transposes are not. |
| **Transformation** | The geometric action a matrix performs on a vector. |
| **Dot product** | `Σaᵢbᵢ` — measures directional similarity. |
| **Matmul / `@`** | Matrix multiplication; inner dimensions must match. |
| **Embarrassingly parallel** | A problem whose parts compute independently — ideal for GPUs. |
| **Batch** | A group of examples processed together in one step. |
| **Epoch** | One complete pass over the training dataset. |
| **`.view()`** | Reinterpret a tensor's shape without copying; needs contiguous data. |
| **`.reshape()`** | Like `view` but will copy if necessary. |
| **`.permute()`** | Reorder dimensions/axes, genuinely changing which axis is which. |
| **Broadcasting** | Automatic expansion of a smaller tensor to match a larger one. |
| **Autograd** | PyTorch's automatic differentiation engine. |
| **`requires_grad`** | Flag telling autograd to track operations on a tensor. |
| **Computation graph** | The dynamic DAG of operations autograd records. |
| **`grad_fn`** | The backward function recording how a tensor was produced. |
| **`.backward()`** | Traverses the graph in reverse, computing all gradients. |
| **`torch.no_grad()`** | Context manager disabling graph recording — for inference. |
| **`zero_grad()`** | Clears accumulated gradients before the next backward pass. |
| **`nn.Module`** | Base class for models: layers in `__init__`, dataflow in `forward`. |
| **`nn.Linear(in, out)`** | Fully connected layer computing `y = x @ W.T + b`. |

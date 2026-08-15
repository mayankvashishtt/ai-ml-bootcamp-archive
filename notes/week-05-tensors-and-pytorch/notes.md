# Week 5 — Introduction to Tensors and PyTorch

**Subtitle:** The Math & Tools — Tensors, Matrices & PyTorch
**Date:** 14/02/2026
**Sources:** `downloads/week-05-tensors-and-pytorch.pdf` (39 slides) · `downloads/week-05-tensors-and-pytorch.ipynb` (75 cells)
**Notion page:** https://100xschool.notion.site/308ffffa33e580b596c6f790b1846ecf
**Extra link:** [youtu.be/f5liqUk0ZTw](https://youtu.be/f5liqUk0ZTw)

---

## The premise: what's been hiding?

```
// Week 2: Neural Networks
weights * inputs + bias
backprop(gradients)

// Weeks 3–4: Transformers
Q = X * Wq
attention = softmax(Q * K.T) * V
```

> **What were ALL of these, really? Matrix multiplication. That's the secret.**

This week is the retroactive maths lecture — everything already built, now named properly. Three acts: **the containers** (tensors), **the operation** (matrix multiplication as transformation), **the tool** (PyTorch).

---

## Act 1 — Scalars → Vectors → Matrices → Tensors

| Name | What it is | Dimensions | Example |
|---|---|---|---|
| **Scalar** | A single number | **0D** | `temperature = 72`, `coffee_price = 4.99` |
| **Vector** | A list of numbers | **1D** | `location = [28.6139, 77.2090]`, `rgb = [255, 87, 51]` |
| **Matrix** | A grid of rows × columns | **2D** | `[[1,2,3],[4,5,6],[7,8,9]]` |
| **Tensor** | Cube / hypercube | **3D, 4D, 5D…** | batches of batches |

> **It's not a scary new concept. It's just: "What if we keep adding dimensions?"**

### You've already been using these

**Vectors:** the 768-dim `king_vector` word embedding from Week 3; the `[x₁, x₂, x₃]` features fed into a neuron in Week 2.

**Matrices:** the weight matrix `W` where `output = W × input` (Week 2); `Wq`, `Wk`, `Wv` and `scores = Q × K.T` (Week 4). **Weights are matrices. Attention is matrix maths.**

### Why tensors specifically — data has structure

```
1. A single sentence: "The cat sat"
   sentence = Matrix(3 tokens, 768 dims)      Shape: [3, 768]           # 2D

2. A batch of sentences
   batch = Stack(8 sentences)                 Shape: [8, 3, 768]        # 3D

3. Multi-head attention processing
   attention = Split(12 heads)                Shape: [8, 12, 3, 64]     # 4D
```

Each dimension you add corresponds to a real structural fact about the data — batching, heads, sequence.

### Shape is everything

```
[8, 12, 3, 64]
 ↑   ↑   ↑   ↑
 |   |   |   └── Head dimension
 |   |   └────── Number of tokens
 |   └────────── Attention heads
 └────────────── Batch size
```

> **Pro-tip: if you understand shape, you can read any model.**

This is the single most practical sentence in the lecture. Most deep-learning debugging is shape debugging.

---

## Act 2 — Matrix magic

### A matrix is a transformation, not storage

> **A matrix isn't just storage. A matrix is a TRANSFORMATION.** It takes a vector in → gives a different vector out.

Capabilities: **rotate, scale, stretch, reflect, project, transform meaning.**

**Rotation — verified in the notebook:**

```python
v = torch.tensor([1.0, 0.0])                 # pointing right
rotate_90 = torch.tensor([[0., -1.],
                          [1.,  0.]])
rotate_90 @ v      # → [0.0, 1.0]  — now pointing up
rotate_45 @ v      # → [0.707, 0.707]
```

**Scaling:**
```
[2 0]   [3]   [6]              Uniform 2× — every component doubled
[0 2] × [4] = [8]

[3  0 ]  [3]                   Stretch 3× horizontally,
[0 0.5] ×[4]                   squish 0.5× vertically
```

The notebook's `plot_transformation()` renders a grid before and after, with basis vectors drawn as arrows. Four transformations to try:

| Matrix | Effect |
|---|---|
| `[[cos θ, −sin θ], [sin θ, cos θ]]` | Rotation |
| `[[2, 0], [0, 0.5]]` | Stretch horizontally, squish vertically |
| `[[1, 1], [0, 1]]` | Shear |
| `[[-1, 0], [0, 1]]` | Reflection (horizontal flip) |
| `[[0, 1], [1, 0]]` | Swap x and y |

> **Key insight: in a neural network, the model learns which matrix to use.**
> **Training is just finding the right transformation for your data.**

This reframes everything from Week 2. "Adjusting weights" = *searching the space of geometric transformations* for the one that maps inputs to correct outputs.

### Why matrix multiplication is everything

| Operation | Where it appears |
|---|---|
| `X @ W` | Forward pass — learned transformations |
| `dL/dY @ W.T` | Backpropagation — computing gradients |
| `Q @ K.T` | Attention — measuring similarity |
| `one_hot @ W_embed` | Word embeddings — looking up meaning |

> **Every single computation in a modern AI model is essentially a series of matrix multiplications.**

Note that even the embedding *lookup* is a matmul in disguise — a one-hot vector times the embedding matrix selects a row.

### The dot product — the atomic operation

```python
v1 = [1, 2, 3];  v2 = [4, 5, 6]
v1 · v2 = (1*4) + (2*5) + (3*6) = 32
```

> **Dot product = similarity.** High → pointing the same direction. Low/zero → different directions.

Notebook verification:

| Vectors | Dot product | Meaning |
|---|---|---|
| `[1,2,3] · [1.1,1.9,3.1]` | **14.20** | Similar → high |
| `[1,0] · [0,1]` | **0.00** | Perpendicular → zero |
| `[1,2] · [-1,-2]` | **−5.00** | Opposite → negative |

### Attention is just dot products

```
Q_sat · K_the = 0.1
Q_sat · K_cat = 0.8    ← high similarity!
Q_sat · K_sat = 0.9

score = softmax(Q @ K.T)
```

> **The W matrices rotate vectors into a space where "relevant" and "similar direction" mean the same thing.**
> **Attention scores are dot products. The model learns the right rotation.**

This is the best one-sentence explanation of Q/K/V in the whole course. `W_Q` and `W_K` exist to *rotate embeddings into a space where geometric alignment equals semantic relevance.*

### Why GPUs exist

| CPU | GPU |
|---|---|
| 8–16 **complex** cores | 5,000+ **simple** cores |
| Optimised for sequential tasks | Optimised for parallel tasks |
| Great for logic, branching, single-threaded speed | Great for the same simple maths thousands of times at once |

> **Matrix multiplication is "embarrassingly parallel"** — every cell in the output can be calculated at the same time.

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

**Note the 100×100 row:** only 1.3× faster. At small sizes the overhead of moving data to the GPU dominates. **GPUs win on scale, not on every operation** — a genuinely useful engineering lesson.

---

## Act 3 — PyTorch

### What it is

Used by **OpenAI, Anthropic, Google DeepMind, Tesla.** Three core capabilities:

1. **Tensors on GPUs** — ~100× speedup over CPU maths
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

A tensor stores **two things**: the actual numbers, and **metadata** (shape, stride, device). The metadata tells PyTorch how to *interpret* the flat block of data.

The notebook proves this:

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

**Reshaping doesn't move data — it changes how the same memory is read.** A transpose just reverses the stride, which is why it's free but leaves the tensor non-contiguous (and why you sometimes need `.contiguous()` before `.view()`).

### GPU acceleration

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)          # ← THE MAGIC LINE
```

### Matrix multiply — the `@` operator

```python
A = torch.randn(3, 4)     # [3, 4]
B = torch.randn(4, 5)     # [4, 5]
C = A @ B                 # [3, 5]
C = torch.matmul(A, B)    # equivalent
```

> **The Rule: inner dimensions MUST match.** `(3,4) @ (4,5) → (3,5)`

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

`permute` is exactly the operation used to rearrange multi-head attention tensors between "heads-first" and "tokens-first" layouts.

### Broadcasting

```python
A = torch.ones(3, 4)                    # [3, 4]
B = torch.tensor([1, 2, 3, 4])          # [4]
C = A + B                               # [3, 4]  — works!
```

PyTorch "stretches" B to match A — like adding `[1,2,3,4]` to **every row**.

> **The Rule: dimensions must be equal, or one of them must be 1.**

This is how a bias vector gets added to a whole batch without writing a loop.

### Autograd — the big one

> **This is why PyTorch exists.**

How it works:
1. **Tracks every operation** — set `requires_grad=True` and PyTorch "starts rolling a tape," recording everything that happens to that tensor
2. **Builds a graph** — a dynamic computation graph of all tensors
3. **Computes gradients** — one call to `.backward()` finds all derivatives

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 5
y.backward()
print(x.grad)        # tensor(4.0)     — dy/dx = 2x = 4
```

**Inspecting the graph:**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y + 2 * x + 1

z.grad_fn           # <AddBackward0>
y.grad_fn           # <PowBackward0>
z.grad_fn.next_functions   # the chain backwards
```

**Backpropagation is just a reverse traversal of this graph**, applying the chain rule at every step.

### Manual vs automatic

```python
# Manual backprop (painful)
d_loss_y = 2 * (y_pred - y_true)
d_y_relu = 1 if z > 0 else 0
d_relu_w = x.T
d_loss_w = d_relu_w @ (d_loss_y * d_y_relu)
w -= lr * d_loss_w
# One mistake here = model never trains.

# PyTorch autograd
loss = criterion(y_pred, y_true)
loss.backward()        # ← ALL OF THE ABOVE IN ONE LINE
optimizer.step()
```

> **Focus on the architecture, not the calculus.**

### `torch.no_grad()` and `zero_grad()`

- **`torch.no_grad()`** — tells PyTorch to **stop recording**. Use during inference/evaluation: no graph is built, so it's faster and uses less memory.
- **`optimizer.zero_grad()`** — PyTorch **accumulates gradients by default.** Call `.backward()` twice without clearing and the gradients **add up.**

> 🔍 **Gotcha in the shipped notebook:** cell 43 is captioned *"After second backward: x.grad = 12.0! Accumulated!"* but the code calls `x.grad.zero_()` immediately before, so the actual printed output is **6.0, 6.0, 6.0** — the demonstration contradicts its own comment. To actually see accumulation, delete the first `x.grad.zero_()` and you'll get 6.0 then 12.0. Worth doing: the effect is important and the notebook doesn't show it.

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

### What's inside `nn.Linear`?

```python
layer = nn.Linear(3, 2)        # 3 inputs → 2 outputs

layer.weight    # shape [2, 3]
                # [[ 0.12, -0.45,  0.89],
                #  [-0.34,  0.22, -0.11]]
layer.bias      # shape [2]  → [0.05, -0.02]
```

> **It's just matrix multiplication and vector addition: `y = x @ W.T + b`**

Exactly the neuron equation from Week 2, vectorised.

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
| **3. zero_grad** | *"Flush the toilet."* Don't let old gradients leak. |
| **4. Backward** | The calculus. Find how each weight caused the error. |
| **5. Step** | Nudge weights in the right direction. |

> **This pattern is the "heartbeat" of every DL model. From MNIST to GPT-4, this loop remains the same.**
> **If you miss step 3, gradients accumulate and your model explodes.**

Same four-step loop as Week 2 — now with an explicit `zero_grad` because PyTorch accumulates.

---

## The demo — MNIST in under 60 seconds

**Goal: 98% accuracy in under 60 seconds.** What the transformations do:

| Step | Shape change |
|---|---|
| **1. Flattening** | `[28, 28]` image grid → `[784]` list of pixels |
| **2. Transform 1** | `[784] @ [784, 128]` → `[128]` features |
| **3. Activation** | ReLU — filtering out the noise |
| **4. Transform 2** | `[128] @ [128, 10]` → `[10]` digit scores |

> **Training is just finding the perfect matrices to map "image pixels" → "correct number label".**

---

## Key takeaways

1. **Everything from Weeks 2–4 was matrix multiplication.** Weights, attention, embeddings, gradients.
2. **Tensor = scalar/vector/matrix generalised** to any number of dimensions. Nothing exotic.
3. **Shape is the master skill** — `[batch, heads, tokens, head_dim]`. Read shapes, read models.
4. **A matrix is a transformation** (rotate, scale, shear, reflect); **training searches for the right one.**
5. **Dot product = similarity**, and attention scores are dot products — the W matrices rotate vectors so alignment means relevance.
6. **Matmul is embarrassingly parallel**, which is exactly what GPUs are built for — but only pays off at scale (1.3× at 100×100, ~38× at 4000×4000).
7. **PyTorch tensors are data + metadata**; `.view()` and `.T` change interpretation, not memory.
8. **Autograd records a tape** and `.backward()` traverses it in reverse — replacing error-prone manual calculus.
9. **`nn.Linear` is `y = x @ W.T + b`** — the Week 2 neuron, vectorised.
10. **The 5-line loop is universal**, and forgetting `zero_grad()` breaks it.

---

## Glossary

| Term | Meaning |
|---|---|
| **Scalar / Vector / Matrix / Tensor** | 0D / 1D / 2D / nD numerical containers. |
| **Shape** | The size of each dimension, e.g. `[8, 12, 3, 64]`. |
| **Stride** | Step size in memory per dimension; how metadata maps to a flat buffer. |
| **Contiguous** | Whether elements are laid out in order in memory. Transposes are not. |
| **Transformation** | The geometric action a matrix performs on a vector. |
| **Dot product** | `Σaᵢbᵢ` — measures directional similarity. |
| **Matmul / `@`** | Matrix multiplication; inner dimensions must match. |
| **Embarrassingly parallel** | A problem whose parts compute independently — ideal for GPUs. |
| **`.view()`** | Reinterpret a tensor's shape without copying data. |
| **`.permute()`** | Reorder dimensions/axes. |
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

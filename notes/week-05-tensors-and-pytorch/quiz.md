# Week 5 — Quiz (20 questions)

**Topic:** Tensors, Matrices & PyTorch
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** A tensor of shape `[8, 12, 3, 64]` in multi-head attention represents, in order:
- A) tokens, heads, batch, head_dim
- B) batch, heads, tokens, head_dim
- C) heads, batch, head_dim, tokens
- D) batch, tokens, heads, head_dim

**2.** The matrix `[[0, -1], [1, 0]]` applied to `[1, 0]` produces:
- A) `[1, 0]` — unchanged
- B) `[0, 1]` — rotated 90°
- C) `[-1, 0]` — reflected
- D) `[0, -1]`

**3.** The dot product of two perpendicular vectors is:
- A) 1
- B) 0
- C) −1
- D) Their magnitude product

**4.** For `A @ B` with `A` of shape `(3, 4)`, which shape must `B` have?
- A) `(3, 4)`
- B) `(5, 4)`
- C) `(4, 5)`
- D) `(4, 3)` only

**5.** Broadcasting requires that dimensions be:
- A) Identical in every position
- B) Equal, or one of them equal to 1
- C) Powers of two
- D) In ascending order

**6.** After `y = x.view(4, 3)` on a `(3, 4)` tensor, `x.data_ptr() == y.data_ptr()` evaluates to:
- A) True — same memory, different interpretation
- B) False — view always copies
- C) True only on GPU
- D) It raises an error

**7.** `x.T` on a contiguous `(3, 4)` tensor produces a tensor that is:
- A) Contiguous with stride (4, 1)
- B) Non-contiguous with stride (1, 4)
- C) A full copy in new memory
- D) Contiguous with stride (1, 4)

**8.** In the measured benchmark, the GPU speedup at 100×100 was only ~1.3× because:
- A) The GPU was faulty
- B) Data transfer and launch overhead dominate at small sizes
- C) PyTorch does not support small matrices on GPU
- D) CPUs are faster at floating point

**9.** `nn.Linear(3, 2)` has a weight of shape:
- A) `[3, 2]`
- B) `[2, 3]`
- C) `[3]`
- D) `[2]`

**10.** `torch.no_grad()` is used to:
- A) Zero out existing gradients
- B) Stop PyTorch recording operations onto the autograd tape
- C) Delete the computation graph permanently
- D) Force gradients to be computed twice

**11.** Omitting `optimizer.zero_grad()` causes:
- A) The model to train faster
- B) Gradients to accumulate across iterations, corrupting updates
- C) A runtime error on the first iteration
- D) The learning rate to double

**12.** "Matrix multiplication is embarrassingly parallel" means:
- A) It cannot be parallelised
- B) Every cell of the output can be computed independently and simultaneously
- C) It requires sequential dependency resolution
- D) It only runs on a single core

---

## Short answer

**13.** Explain the claim "a matrix is a transformation, not storage," and connect it to what training actually searches for.

**14.** Give the four operations listed as "why matrix multiplication is everything," with the formula for each.

**15.** Explain how the dot product connects to attention. What role do W_Q and W_K play geometrically?

**16.** Contrast CPU and GPU architecture and explain why matmul suits the GPU. Then explain the 100×100 result.

**17.** Explain what PyTorch stores when you create a tensor, and use `.view()` and `.T` to show why the distinction matters.

**18.** Describe the three steps of how autograd works, and explain what `grad_fn` tells you.

**19.** Write out the 5-line training loop, name each step, and explain what breaks if step 3 is omitted.

**20.** In notebook cell 43 the caption claims `x.grad = 12.0` after the second backward, but the output shows `6.0`. Explain the discrepancy and how to fix the demo.

---
---

## Answer key

**1. B** — batch size, attention heads, number of tokens, head dimension.

**2. B** — `[0, 1]`. The matrix rotates the vector 90° counterclockwise, turning "pointing right" into "pointing up."

**3. B** — 0. The notebook verifies `[1,0] · [0,1] = 0.00`. Similar vectors give a high positive value (14.20), opposite vectors negative (−5.00).

**4. C** — `(4, 5)`. Inner dimensions must match: `(3,4) @ (4,5) → (3,5)`.

**5. B** — Dimensions must be equal, or one must be 1, in which case it is stretched to match.

**6. A** — True. `.view()` reinterprets the same memory rather than copying, which is why it is nearly free.

**7. B** — Non-contiguous with reversed stride (1, 4). Transposing changes only the metadata, not the underlying buffer.

**8. B** — At small sizes, the cost of transferring data and launching the kernel dominates the actual arithmetic. GPUs win on scale — the same benchmark shows ~40× at 500×500 and above.

**9. B** — `[2, 3]`, i.e. `[out_features, in_features]`, because the layer computes `y = x @ W.T + b`.

**10. B** — It stops autograd from recording operations, so no graph is built. Used during inference to save time and memory.

**11. B** — PyTorch accumulates gradients into `.grad` by default, so without zeroing, gradients from previous iterations sum into the current update.

**12. B** — Every output cell is an independent dot product, so thousands can be computed simultaneously — exactly what 5,000+ simple GPU cores are for.

**13.** A matrix does not merely hold numbers; multiplying a vector by it **produces a different vector** — it rotates, scales, stretches, shears, reflects, or projects. `[[0,−1],[1,0]]` rotates by 90°; `[[2,0],[0,0.5]]` stretches horizontally and squishes vertically; `[[1,1],[0,1]]` shears. Since every layer's weights are a matrix, each layer is a geometric transformation of its input space. **Training is therefore the search for the right transformation** — gradient descent is exploring the space of possible geometric operations to find the one mapping inputs to correct outputs. "Adjusting weights" and "reshaping space" are the same statement.

**14.** (i) **Forward pass** — `X @ W`, applying learned transformations. (ii) **Backpropagation** — `dL/dY @ W.T`, computing gradients. (iii) **Attention** — `Q @ K.T`, measuring similarity. (iv) **Word embeddings** — `one_hot @ W_embed`, looking up meaning. Every computation in a modern model reduces to a series of these.

**15.** The dot product measures **directional similarity**: large and positive when vectors point the same way, zero when perpendicular, negative when opposed. Attention scores are exactly dot products — `Q_sat · K_cat = 0.8` means "sat" should attend strongly to "cat" — and `softmax(Q @ K.T)` simply normalises them into weights. **Geometrically, W_Q and W_K rotate embeddings into a space where "similar direction" and "relevant" mean the same thing.** Raw embeddings encode general meaning, in which alignment does not necessarily indicate task relevance; the learned projections reorient them so the cheap geometric test of a dot product becomes a meaningful relevance test. The model learns the right rotation.

**16.** A **CPU** has 8–16 complex cores optimised for sequential work — logic, branching, single-threaded speed. A **GPU** has 5,000+ simple cores optimised for doing the same simple maths thousands of times in parallel. Matrix multiplication fits because each output cell is an independent dot product with no dependency on any other cell, so all can be computed at once — "embarrassingly parallel." **The 100×100 result (1.3×)** shows the limit: at that size the actual arithmetic is trivial compared with the fixed cost of transferring data to the GPU and launching the kernel, so overhead dominates. Once matrices are large enough for arithmetic to dominate — 500×500 and beyond — the speedup jumps to roughly 32–40×.

**17.** PyTorch stores **two things**: the actual numbers as a flat block of memory, and **metadata** (shape, stride, device) describing how to interpret that block. Because interpretation is separate from data, many operations are free. `x.view(4, 3)` on a `(3, 4)` tensor returns a tensor whose `data_ptr()` is **identical** — the same memory read with different shape metadata, no copying. `x.T` similarly only reverses the stride from `(4, 1)` to `(1, 4)`, which makes the result **non-contiguous**: elements are no longer laid out in reading order. This matters in practice because operations such as `.view()` require contiguous memory, which is why you sometimes must call `.contiguous()` after a transpose or permute.

**18.** (i) **Tracks every operation** — setting `requires_grad=True` tells PyTorch to "start rolling a tape," recording each operation applied to that tensor. (ii) **Builds a graph** — those records form a dynamic computation graph linking every intermediate tensor to the operation that produced it. (iii) **Computes gradients** — one call to `.backward()` walks the graph in reverse, applying the chain rule at each node, and populates `.grad` on every leaf tensor. **`grad_fn`** names the backward function of the operation that created a tensor — `y.grad_fn` is `<PowBackward0>` for `y = x**2`, `z.grad_fn` is `<AddBackward0>` for an addition — and `grad_fn.next_functions` exposes the links to its inputs, letting you inspect the chain that `.backward()` will traverse.

**19.**
```python
y_pred = model(x_batch)              # 1. Forward pass
loss   = criterion(y_pred, y_batch)  # 2. Compute loss
optimizer.zero_grad()                # 3. Clear old gradients
loss.backward()                      # 4. Backward pass
optimizer.step()                     # 5. Update weights
```
*Forward* predicts with current weights; *loss* measures error; *zero_grad* flushes stale gradients; *backward* computes how each weight contributed; *step* nudges weights downhill. **Omitting step 3** means gradients are not cleared, and since PyTorch accumulates into `.grad` by default, each iteration's gradient **adds to every previous one**. Updates grow steadily larger and are computed from a mixture of stale and current signals, so training destabilises and typically diverges.

**20.** The caption describes accumulation, but the code calls `x.grad.zero_()` **immediately before** the second forward/backward pass, which clears the accumulated value — so the second `.backward()` writes a fresh gradient of 6.0 rather than adding to the existing 6.0 to give 12.0. All three printed values are therefore 6.0, and the cell fails to demonstrate the very behaviour it claims. **Fix:** delete the first `x.grad.zero_()` so the second backward accumulates onto the first, producing 6.0 then 12.0; keep the final `zero_()` before the third pass to show the reset restoring 6.0. This is worth repairing, since gradient accumulation is precisely the gotcha `optimizer.zero_grad()` exists to prevent.

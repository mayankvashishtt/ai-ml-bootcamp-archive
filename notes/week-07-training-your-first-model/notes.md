# Week 7 — Training Your First Model

**Subtitle:** Code Walkthrough — Building & Training a Modern LLM from Scratch (RMSNorm · RoPE · GQA · SwiGLU)
**Date:** 28/02/2026
**Sources:** `downloads/week-07-training-your-first-model.pdf` (40 slides) · `downloads/week-07-training-your-first-model.ipynb` (40 cells)
**Notion page:** https://100xschool.notion.site/316ffffa33e5804fa347e9b84506b93f

> This is the payoff week. Week 6 listed the four modern upgrades; **this week implements all of them** and trains the result on Shakespeare. The closing line of the deck is the point:
>
> **"This is not a toy — production LLMs use these exact same components, just bigger."**

---

## Section 0 — Setup & data

### Character tokenizer

Deliberately *not* BPE. Shakespeare has 65 unique characters, so a character tokenizer keeps the vocabulary tiny and the code readable.

```python
chars = sorted(set(text))                        # all unique chars
vocab_size = len(chars)                          # 65
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):   return [char_to_idx[c] for c in s]
def decode(ids): return "".join([idx_to_char[i] for i in ids])
```

`"Hello"` → `[20, 43, 50, 50, 53]`

**Trade-off, connecting back to Week 3:** character-level means a 65-entry vocabulary and no OOV ever, but every word costs many tokens and the model must learn spelling from scratch. Fine for a teaching model on one corpus; hopeless at scale — which is exactly the argument Week 3 made for subwords.

### `get_batch()` — grabbing training data

```python
def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i   : i+context_length  ] for i in ix])
    y = torch.stack([d[i+1 : i+context_length+1] for i in ix])
    return x.to(device), y.to(device)
```

- Pick **random** starting positions in the text
- `x` = input chunk; `y` = the same chunk **shifted right by 1**
- The target at each position is **the next character**

```
x = "To be o"  →  shifted by 1  →  y = "o be or"
```

### Why shift by 1?

| Pos | Given (context so far) | Predict |
|---|---|---|
| 0 | `T` | `o` |
| 1 | `To` | `␣` |
| 2 | `To␣` | `b` |
| 3 | `To␣b` | `e` |
| 4 | `To␣be` | `,` |

> **One sequence of length N gives us N training examples!**

This is the efficiency trick behind language-model pretraining. Every position in the sequence is a supervised example *simultaneously* — no labelling required, and a single forward pass produces N training signals. It's why "just predict the next token" scales to trillions of tokens: the data labels itself.

---

## Section 1 — RMSNorm

| LayerNorm (old) | RMSNorm (new) |
|---|---|
| Subtract mean | — |
| Divide by std | Divide by RMS |
| Scale + Shift | Scale only |
| **4 ops, 2 learned params** | **2 ops, 1 learned param** |

> Simpler, faster, works just as well. Used in LLaMA, Mistral, Gemma.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

**Worked example:** `[3, -1, 2, -4]` → mean of squares = (9+1+4+16)/4 = 7.5 → RMS = √7.5 ≈ 2.74 → `[1.09, -0.36, 0.73, -1.46]`

Note the mean of the output is **not** zero — RMSNorm only controls magnitude, never centres.

### Dropout — regularization

```python
DROPOUT = 0.2
```

- **During training:** randomly zeroes 20% of values
- Forces the model **not to rely on any single feature**
- **During inference:** all values active
- **Prevents memorization (overfitting)**

Dropout is the first *regularization* technique in the course. Weeks 2–6 were about making the model able to fit; this is about stopping it fitting *too well* to the training set.

---

## Section 2 — RoPE

| Sinusoidal (old) | RoPE (new) |
|---|---|
| **ADD** fixed vector to embedding | **ROTATE** Q and K vectors |
| Encodes absolute position | Dot product captures relative distance |
| Position 5 always looks the same | Only the **gap** between positions matters |
| Applied to the embedding | **Applied to Q and K only** |

**The clock analogy:** position 3 → 5 (gap 2) is the same rotation as 7 → 9 (gap 2). Different absolute positions, same relative distance.

### Precompute frequencies

```python
def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)
```

- Each **dimension pair** gets a different rotation speed
- **Low dims → fast rotation → short-range patterns**
- **High dims → slow rotation → long-range patterns**
- **Precomputed once, reused every forward pass** — RoPE costs no learned parameters and almost no compute

### Apply the rotation

```python
def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]      # even dims
    x2 = x[..., 1::2]     # odd dims
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

This is literally the 2D rotation matrix applied pairwise:

```
[ cos θ  -sin θ ]   [ x₁ ]   [ x₁·cos − x₂·sin ]
[ sin θ   cos θ ] × [ x₂ ] = [ x₁·sin + x₂·cos ]
```

`out1`/`out2` are exactly the two rows of that product. The `stack`+`flatten` interleaves the pairs back into the original layout.

---

## Section 3 — GQA

### The sharing scheme used here

| Standard MHA | GQA (this model) |
|---|---|
| Q0 ↔ K0,V0 | Q0 ↔ K0,V0 |
| Q1 ↔ K1,V1 | Q1 ↔ K0,V0 *(shared)* |
| Q2 ↔ K2,V2 | Q2 ↔ K0,V0 *(shared)* |
| Q3 ↔ K3,V3 | Q3 ↔ K0,V0 *(shared)* |
| Q4 ↔ K4,V4 | Q4 ↔ K1,V1 |
| Q5 ↔ K5,V5 | Q5 ↔ K1,V1 *(shared)* |
| Q6 ↔ K6,V6 | Q6 ↔ K1,V1 *(shared)* |
| Q7 ↔ K7,V7 | Q7 ↔ K1,V1 *(shared)* |
| **8 KV heads** | **2 KV heads → 4× savings** |

### Projections — where the saving lives

```python
self.q_proj = nn.Linear(d_model, n_heads    * head_dim)   # 256 → 256  (8 heads × 32)
self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim)   # 256 →  64  (2 heads × 32) ← 4× smaller
self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim)   # 256 →  64  (2 heads × 32) ← 4× smaller
self.o_proj = nn.Linear(n_heads * head_dim, d_model)      # 256 → 256
```

- **Q projection stays full size** — all 8 heads need unique queries
- **K and V projections are 4× smaller** — only 2 heads
- **All `bias=False`** (modern convention — the norm layers make biases redundant)

### `repeat_kv()` — expanding KV heads

```python
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, seq, hd = x.shape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))
```

`K: [b, 2, seq, 32]` → expand → `K: [b, 8, seq, 32]`

> **`expand()` is memory-efficient (no actual data copy)** — it creates a view with stride 0 on the repeated axis. The 4× saving is real: only 2 heads' worth of K/V is ever stored or cached.

### Forward — project, reshape, RoPE

```python
q = self.q_proj(x)      # [b, seq, 256]
k = self.k_proj(x)      # [b, seq,  64]
v = self.v_proj(x)      # [b, seq,  64]

q = q.view(b, seq, 8, 32).transpose(1, 2)   # [b, 8, seq, 32]
k = k.view(b, seq, 2, 32).transpose(1, 2)   # [b, 2, seq, 32]
v = v.view(b, seq, 2, 32).transpose(1, 2)   # [b, 2, seq, 32]

q = apply_rope(q, rope_cos, rope_sin)
k = apply_rope(k, rope_cos, rope_sin)

k = repeat_kv(k, self.n_rep)     # [b,2,seq,32] → [b,8,seq,32]
v = repeat_kv(v, self.n_rep)
```

- `.view()` splits the flat vector into separate heads
- `.transpose(1,2)` swaps seq and heads so matmul batches over heads
- ⚠️ **RoPE on Q and K only — not V!** Position affects *matching* (who attends to whom), not *content* (what gets passed along).

### Forward — attention scores

```python
scale = 1.0 / math.sqrt(self.head_dim)              # 1/√32
scores = (q @ k.transpose(-2, -1)) * scale          # [b, 8, seq, seq]

mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))

weights = F.softmax(scores, dim=-1)
weights = F.dropout(weights, p=0.2, training=self.training)
out = weights @ v                                   # [b, 8, seq, 32]
```

### The causal mask — the new idea this week

|  | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| **T0** | ✓ | ✗ | ✗ | ✗ |
| **T1** | ✓ | ✓ | ✗ | ✗ |
| **T2** | ✓ | ✓ | ✓ | ✗ |
| **T3** | ✓ | ✓ | ✓ | ✓ |

> **Each token can only attend to itself and previous tokens.**

**Why this is essential:** training predicts the next character at *every* position simultaneously. Without masking, position 3 could simply look at position 4 — the answer — and the model would learn nothing but copying. Masking future positions to `-inf` makes softmax assign them exactly 0.

`-inf` rather than 0 because the mask is applied *before* softmax: `e^(-inf) = 0`, so masked positions contribute nothing and the remaining weights still normalise to 1.

This is what makes the model a **decoder-only / causal** LM — the GPT family design.

### Merge heads & output

```python
out = out.transpose(1, 2).contiguous().view(b, seq, -1)   # [b,8,seq,32] → [b,seq,256]
return self.o_proj(out)
```

`.contiguous()` is required before `.view()` after a transpose — exactly the stride/contiguity issue from Week 5.

---

## Section 4 — SwiGLU

| Old FFN — single path | SwiGLU — gated dual path |
|---|---|
| `x → expand 256→680 → ReLU → contract 680→256` | `x →` **`W_gate → SiLU`** and **`W_up (raw)`** `→ × → W_down → out` |

> **The gate LEARNS which dimensions to keep and which to suppress.**

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        self.w_gate = nn.Linear(d_model, hidden_dim)   # 256 → 680
        self.w_up   = nn.Linear(d_model, hidden_dim)   # 256 → 680
        self.w_down = nn.Linear(hidden_dim, d_model)   # 680 → 256

    def forward(self, x):
        gate = F.silu(self.w_gate(x))    # gate path + activation
        up   = self.w_up(x)              # value path (no activation)
        return F.dropout(self.w_down(gate * up), p=DROPOUT, training=self.training)
```

Note **680**, not the classic 4×256 = 1024. SwiGLU uses *three* weight matrices instead of two, so the hidden dimension is shrunk to keep the parameter count comparable — a standard adjustment (≈ ⅔ × 4 × d_model).

---

## Section 5 — Assembly

### The transformer block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden_dim):
        self.attn_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, ffn_hidden_dim)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

```
x ─────────────────────────────────────┐
  └→ RMSNorm → GQA Attention ───────── + → x'
x' ────────────────────────────────────┐
  └→ RMSNorm → SwiGLU FFN ──────────── + → out
```

- **Pre-norm:** normalize BEFORE each sublayer
- **Residual:** `x + sublayer(norm(x))` — **the sublayer learns the CHANGE**, not the whole representation
- **Two sublayers:** attention **mixes between tokens**; FFN **processes each token independently**

That last distinction is the cleanest way to think about a transformer block: *attention moves information sideways, the FFN thinks about it in place.*

### The full model

```python
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, ...):
        self.token_emb = nn.Embedding(vocab_size, d_model)     # 65 → 256
        self.layers = nn.ModuleList([
            TransformerBlock(...) for _ in range(n_layers)      # ×4
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)          # 256 → 65

        # Weight tying!
        self.lm_head.weight = self.token_emb.weight
```

- **No positional embedding** — RoPE handles position inside attention
- **Weight tying:** the embedding matrix and output head **share the same weights**

**Why weight tying works:** the embedding maps token → vector; the LM head maps vector → token scores. They're inverse operations over the same vocabulary, so sharing is natural. It saves `vocab_size × d_model` parameters and usually *improves* quality by forcing consistency between input and output representations.

### Forward pass

```python
def forward(self, idx, targets=None):
    x = self.token_emb(idx)                # [b, seq, 256]
    for layer in self.layers:               # 4 blocks
        x = layer(x, self.rope_cos, self.rope_sin)
    x = self.final_norm(x)
    logits = self.lm_head(x)                # [b, seq, 65]

    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
    return logits, loss
```

`cross_entropy` replaces MSE from Week 2 — appropriate for classification over 65 characters rather than regression.

### Configuration

| Parameter | Value |
|---|---|
| `vocab_size` | 65 characters |
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_heads` | 8 query heads |
| `n_kv_heads` | 2 (GQA 4:1) |
| `ffn_hidden_dim` | 680 |
| `max_seq_len` | 256 |

> **Initial loss ≈ ln(65) ≈ 4.17 = random guessing → model is wired correctly!**

**This is a genuinely valuable debugging trick.** A randomly initialised classifier over V classes should assign probability 1/V to each, giving cross-entropy loss = −ln(1/V) = **ln(V)**. If your starting loss doesn't match `ln(vocab_size)`, something is wrong *before* you've wasted a single training step — bad initialisation, a bug in the loss, or mismatched shapes.

---

## Section 6 — Training

### Hyperparameters

- `BATCH_SIZE = 64` → 64 × 256 = **16,384 characters per step**
- `LEARNING_RATE = 3e-4` → standard for transformers
- `MAX_STEPS = 3000` → **~15–20 min on a T4 GPU**
- **AdamW** optimizer — Adam + weight decay

### The loop

```python
for step in range(MAX_STEPS):
    xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LEN)
    logits, loss = model(xb, yb)                    # forward
    optimizer.zero_grad()                           # clear old gradients
    loss.backward()                                 # backprop
    torch.nn.utils.clip_grad_norm_(                 # clip gradients
        model.parameters(), max_norm=1.0
    )
    optimizer.step()                                # update weights
```

> **Same 6-step dance from Class 2!** — with one addition.

**Gradient clipping** (`clip_grad_norm_`) is new. If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally. This **prevents exploding gradients**, which transformers are prone to: a single bad batch can produce an enormous gradient that destroys the weights in one step. Clipping caps the damage while preserving direction.

### Evaluation

```python
@torch.no_grad()
def estimate_loss():
    model.eval()                 # disable dropout
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb = get_batch(split, BATCH_SIZE, CONTEXT_LEN)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()                # re-enable dropout
    return out
```

- `@torch.no_grad()` — no gradients needed, saves memory
- `model.eval()` disables dropout for **fair** evaluation
- Average over 20 batches for a stable estimate
- **Train vs val gap → overfitting indicator**

Forgetting `model.eval()` / `model.train()` around evaluation is one of the most common real bugs in PyTorch code — dropout stays active and your validation loss is silently wrong.

---

## Section 7 — Generation

```python
@torch.no_grad()
def generate(model, prompt, max_new_tokens=500, temperature=0.8):
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context = tokens[:, -config["max_seq_len"]:]   # sliding window
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature        # LAST position only
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return decode(tokens[0].tolist())
```

- Feed prompt → get logits → **take the LAST position only** (the model predicts at every position, but only the final one is the actual next token)
- Divide by temperature → softmax → **sample** (`multinomial`, not `argmax` — sampling gives variety)
- Append and repeat — the autoregressive loop from Week 1
- **Sliding window**: only the last 256 tokens are used as context

> Note this implementation has **no KV cache** (Week 4) — it recomputes the whole context each step. Correct, but O(n²) work; fine for a teaching model, unacceptable in production.

### Temperature

| Setting | Distribution over A/B/C/D |
|---|---|
| **T = 0.3** (conservative) | 91% / 6% / 2% / 1% |
| **T = 0.8** (balanced) | 47% / 27% / 17% / 9% |
| **T = 1.5** (creative) | 33% / 27% / 22% / 18% |

Dividing logits by T *before* softmax: **low T sharpens** the distribution (more deterministic), **high T flattens** it (more random). T→0 approaches greedy argmax; T→∞ approaches uniform.

> Same concept used in ChatGPT and Claude — tune creativity vs consistency.

---

## The complete architecture

```
"ROMEO:" → encode → [44, 41, 37, 29, 41, 26]
         ↓
  Token Embedding (65 → 256)
         ↓
  ┌─────────────────────────────────────┐
  │  Transformer Block ×4               │
  │    RMSNorm → GQA (8Q, 2KV + RoPE)   │
  │    + residual                       │
  │    RMSNorm → SwiGLU (256→680→256)   │
  │    + residual                       │
  └─────────────────────────────────────┘
         ↓
  Final RMSNorm
         ↓
  LM Head (256 → 65)  [weight-tied]
         ↓
  logits → softmax → sample → next char
```

---

## Key takeaways

1. **Next-token prediction gives N training examples per sequence of length N** — self-supervision is why pretraining scales.
2. **The causal mask is what makes it a language model.** Without it, the model cheats by reading ahead.
3. **`-inf` before softmax** is how you zero out attention to masked positions.
4. **RoPE is applied to Q and K only, never V** — position affects matching, not content.
5. **GQA's saving is in the projection sizes** (256→64 for K/V vs 256→256 for Q); `repeat_kv` expands with `expand()`, which copies no data.
6. **Weight tying** shares the embedding and LM head — fewer parameters, usually better quality.
7. **Initial loss should equal ln(vocab_size)** — the cheapest correctness check you'll ever run.
8. **Gradient clipping** (`max_norm=1.0`) prevents a single bad batch from destroying training.
9. **`model.eval()` / `@torch.no_grad()`** during evaluation — disables dropout and saves memory.
10. **Temperature scales logits before softmax** — low sharpens, high flattens.
11. **Attention mixes across tokens; the FFN processes each token in place.**
12. **The same components scale to production LLMs — only bigger.**

---

## Glossary

| Term | Meaning |
|---|---|
| **Character tokenizer** | One token per character; tiny vocabulary, no OOV, poor efficiency. |
| **Self-supervision** | Labels derived from the data itself — here, the next character. |
| **Causal mask** | Upper-triangular mask preventing attention to future positions. |
| **Decoder-only** | Architecture using causal masking to generate left to right (GPT family). |
| **`masked_fill(mask, -inf)`** | Sets masked scores to −∞ so softmax gives them zero weight. |
| **Dropout** | Randomly zeroing activations during training to prevent overfitting. |
| **Overfitting** | Memorising training data; shows up as a growing train/val loss gap. |
| **`repeat_kv`** | Expands shared KV heads to match query head count without copying data. |
| **Weight tying** | Sharing weights between the token embedding and the output head. |
| **Cross-entropy loss** | Classification loss measuring divergence between predicted and true distributions. |
| **`ln(vocab_size)`** | Expected initial loss for a correctly-wired untrained classifier. |
| **AdamW** | Adam optimizer with decoupled weight decay — the transformer standard. |
| **Gradient clipping** | Rescaling gradients whose norm exceeds a threshold, preventing explosions. |
| **`model.eval()` / `model.train()`** | Toggles inference vs training behaviour for dropout and normalization. |
| **Temperature** | Divisor applied to logits before softmax, controlling randomness. |
| **`torch.multinomial`** | Samples from a probability distribution — the alternative to greedy argmax. |
| **Sliding window** | Truncating context to the last `max_seq_len` tokens during generation. |

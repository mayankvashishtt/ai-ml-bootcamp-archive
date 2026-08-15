# Week 4 — Transformers Part 2: Self-Attention, KV Cache, Multi-Head

**Date:** 06/02/2026
**Sources:** `downloads/week-04-transformers-part-2.pdf` (42 slides) · `downloads/week-04-transformers-part-2.ipynb` (69 cells)
**Notion page:** https://100xschool.notion.site/300ffffa33e580448dcaf3f127b52b22

> **Deck note:** slides 1–23 are an exact repeat of Week 3 (tokenization → embeddings → positional encoding). **The new material starts at slide 24.** Likewise, notebook Parts 1–3 are identical to Week 3's; **Part 4 (Self-Attention, cells 47–67) is the new content** — this is the attention implementation Week 3's notebook promised but didn't contain. These notes cover only the new material; see `week-03` for the recap.

---

## 1. Self-attention — the engine of context

> **The defining feature that allows the model to process information globally rather than sequentially.** It is the heart of the *"Attention is All You Need"* philosophy.

- **Mutual influence** — every token attends to every other token **simultaneously**, creating a rich map of relationships.
- **Contextualization** — words don't have a single fixed meaning; their meaning is *refined and updated* based on surrounding words.
- **Interconnectivity** — a dense web of connections lets the model resolve ambiguity and understand complex logical dependencies.

### The intuition

| Aspect | What it does |
|---|---|
| **Filtering noise** | Ignore irrelevant words, amplify signal from related tokens |
| **Dynamic weighting** | Assign a unique importance score to **every word pair** |
| **Reference resolution** | *"The bank was closed because it was a holiday"* — attention tells the model "it" refers to "bank" |
| **Semantic context** | Meaning is defined by relationships; attention is the tool that maps them |

---

## 2. Q, K, V — the three roles of every word

This is the central abstraction. **Every token plays three roles at once:**

| Role | Question it answers | Meaning |
|---|---|---|
| **Query (Q)** | *"What am I looking for?"* | The search criteria a token uses to find relevant information |
| **Key (K)** | *"What do I contain?"* | The label other tokens use to judge whether this token is relevant to their search |
| **Value (V)** | *"What information do I offer?"* | The actual semantic content passed along when Q and K match |

**The interaction:** the similarity between a Query and a Key determines the **attention weight** applied to the corresponding Value.

Three perspectives on the same word:
- **The Searcher (Q)** — the version used to find context in other words
- **The Label (K)** — the version other words use to find this one
- **The Content (V)** — the version that provides information once matched

### The database analogy

| Hard lookup (traditional DB) | Soft lookup (attention) |
|---|---|
| Finds an **exact match** for a key | Finds **partial matches across all keys** |
| Returns a **single value** | Returns a **mixture of values**, weighted by match quality |
| Binary: match or no match | Continuous: everything contributes, some more than others |

- **Weighted sum** — a word's final representation is a weighted combination of *all* words in the sequence.
- **Softmax** — converts raw similarity scores into probabilities summing to 1, determining the "mix."

This is genuinely the best mental model for attention: **a differentiable, fuzzy database query**, where every row returns something and the query decides how much of each.

### Computing Q, K, V

A single embedding is multiplied by **three separate learned matrices** — `W_Q`, `W_K`, `W_V` — producing its Query, Key, and Value vectors.

**These matrices are learned during training.** The model works out for itself how to project embeddings so that attention is maximally useful. Nothing about Q/K/V is hand-designed; only the *structure* is.

---

## 3. The attention formula

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V
```

| Component | Purpose |
|---|---|
| **QKᵀ (dot product)** | Measures similarity between every Query and every Key |
| **÷ √d_k (scaling)** | Prevents dot products becoming too large, keeping gradients stable |
| **softmax** | Converts raw scores into attention weights (probabilities summing to 1) |
| **× V (weighted sum)** | Applies the weights to Values, producing the context-aware representation |

**Why the √d_k scaling matters** (worth understanding, since it's the least obvious term): dot products of higher-dimensional vectors grow larger in magnitude. Feed large values into softmax and it *saturates* — one weight goes to ~1.0 and the rest to ~0. A saturated softmax has near-zero gradient, so learning stalls. Dividing by √d_k keeps the distribution soft enough to train through.

### The attention matrix

A **square grid** where every word is compared to every other word.

- **Heatmap logic** — each cell is an attention weight; brighter = stronger relationship
- **Self-focus** — words often attend strongly to themselves, preserving their own identity while absorbing context
- **Context-focus** — the model learns to link pronouns to antecedents, verbs to subjects

### Context-aware meaning — the payoff

> *"I went to the bank..."* vs *"The bank was closed..."*

Static embeddings give a **blended average** meaning of "bank," which is insufficient. Attention **updates the representation of "bank" based on surrounding tokens** — attending to "money," "loan," or "river" pulls in the features that resolve the ambiguity.

**This closes the loop opened in Week 1 and Week 3.** The one-vector-per-word limitation is now solved: representations are computed *from context* rather than looked up.

---

## 4. Self-attention coded from scratch (notebook Part 4)

### Setup

```python
sentence = ["The", "cat", "sat"]
d_model = 8   # embedding dimension
d_k = 4       # key/query dimension
d_v = 4       # value dimension

X = np.array([embeddings[word] for word in sentence])   # (3, 8)
```

### Step 1 — Project to Q, K, V

```python
W_Q = np.random.randn(d_model, d_k) * 0.1   # (8, 4)
W_K = np.random.randn(d_model, d_k) * 0.1   # (8, 4)
W_V = np.random.randn(d_model, d_v) * 0.1   # (8, 4)

Q = X @ W_Q    # (3,8) @ (8,4) = (3,4)
K = X @ W_K    # (3,4)
V = X @ W_V    # (3,4)
```

Each word now has three 4-dim vectors:

```
'The':  Q [-0.014 -0.380  0.058 -0.618]
        K [-0.071  0.146  0.687  0.154]
        V [ 0.422  0.105 -0.135 -0.289]
'cat':  Q [ 0.028  0.039 -0.032  0.260]
        K [-0.315  0.010 -0.497 -0.362]
        V [-0.446  0.126  0.161  0.405]
'sat':  Q [ 0.084  0.525  0.230  0.628]
        K [ 0.425  0.496 -0.649 -0.064]
        V [-0.050 -0.201  0.637  0.120]
```

### Step 2 — Raw scores

```python
scores = Q @ K.T    # (3,4) @ (4,3) = (3,3)
```

|  | The | cat | sat |
|---|---|---|---|
| **The** | −0.110 | 0.196 | −0.192 |
| **cat** | 0.022 | −0.087 | 0.035 |
| **sat** | 0.326 | −0.363 | 0.106 |

Row *i*, column *j* = how well word *i*'s Query matches word *j*'s Key.

### Step 3 — Scale

```python
scaled_scores = scores / np.sqrt(d_k)   # √4 = 2.00
```

### Step 4 — Softmax

```python
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

|  | The | cat | sat | Sum |
|---|---|---|---|---|
| **The** | 0.320 | 0.373 | 0.307 | 1.000 |
| **cat** | 0.339 | 0.321 | 0.341 | 1.000 |
| **sat** | 0.384 | 0.272 | 0.344 | 1.000 |

**Every row sums to 1** — each word distributes a fixed budget of attention.

> 🔍 **Read these numbers honestly:** they're all ≈0.33, i.e. nearly uniform. That's expected — the embeddings and projection matrices are **random**, not trained. A trained model produces sharply peaked rows. This demo shows the *mechanism*, not learned behaviour; don't mistake the flat distribution for a bug.

### Step 5 — Weighted sum of Values

```python
output = attention_weights @ V   # (3,3) @ (3,4) = (3,4)
```

The notebook verifies the matrix multiplication by hand for "cat":

```
'cat' attention weights: [0.339 0.321 0.341]
  → 0.339 × V_The  +  0.321 × V_cat  +  0.341 × V_sat

Manual:  [-0.017  0.007  0.223  0.073]
Matrix:  [-0.017  0.007  0.223  0.073]
Match: True
```

**This is the whole mechanism.** A word's output is literally a weighted average of every word's Value vector, with attention supplying the weights.

### The complete function

```python
def self_attention(X, W_Q, W_K, W_V):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ V

    return output, attention_weights
```

**Nine lines.** The mechanism behind every modern LLM fits in nine lines of NumPy.

---

## 5. The KV cache

This section is the most practically valuable in the week — it explains latency behaviour you observe daily.

### Generation is repeated attention over a growing sequence

```
Step 1: "The"          → Q,K,V → attention → next token "cat"
Step 2: "The cat"      → Q,K,V → attention → next token "sat"
Step 3: "The cat sat"  → Q,K,V → attention → next token "on"
```

Each step runs attention over the **entire** sequence.

### The key observation

When generating token 4, the new token needs to attend to all previous tokens:

- **New token's Q** — *"What am I looking for?"* → needs fresh computation
- **Previous tokens' K** — *"What do they contain?"* → **these don't change.** Same tokens, same keys.
- **Previous tokens' V** — *"What do they contribute?"* → **these don't change either.**

> **Only the NEW token's Q changes each step.**

### The waste

```
For generating "cat":
  Compute: Q_the, K_the, V_the      ← needed
           Q_cat, K_cat, V_cat      ← needed (new token)

For generating "sat":
  Compute: Q_the, K_the, V_the      ← SAME AS BEFORE!
           Q_cat, K_cat, V_cat      ← SAME AS BEFORE!
           Q_sat, K_sat, V_sat      ← needed (new token)
```

K and V for old tokens are recomputed every single step. Pure wasted work, and it grows quadratically.

### The solution

**Store K and V after first computation.**

```
Step 1: "The" → compute K_the, V_the → CACHE
Step 2: "cat" → compute K_cat, V_cat → CACHE
                use cached K_the, V_the
Step 3: "sat" → compute K_sat, V_sat → CACHE
                use cached K_the, K_cat, V_the, V_cat
```

Only compute K, V for the **new** token each step. Q is always fresh — and that's fine, since you only need Q for the current token anyway.

### Why the first token is slow

```
FIRST token (slow):
  → Process your ENTIRE prompt
  → Compute K, V for every token in the prompt
  → Fill the KV cache
  → Generate first response token

SUBSEQUENT tokens (fast):
  → K, V for prompt already cached
  → Only compute for the new token
```

> **"Time to first token" and "tokens per second" are different metrics. Now you know why.**

This is the *prefill* vs *decode* distinction. Prefill is compute-bound and scales with prompt length; decode is memory-bandwidth-bound and roughly constant per token. It's why a long prompt with a short answer feels slow to start but then streams quickly.

### Why long context is hard

```
100 tokens     → small cache, fast
10,000 tokens  → large cache, slower
100,000 tokens → HUGE cache, memory problems
```

**Each layer stores K and V for all tokens. GPT-4 has ~120 layers. A 128K context means a massive memory footprint.**

The KV cache — not the parameters — is often what limits how many concurrent users a GPU can serve. This one fact explains long-context pricing, context limits, and much of modern inference engineering. It also motivates architectural fixes like grouped-query attention (Week 6).

---

## 6. Multi-head attention

### One head isn't enough

A single attention head has **one set of Q, K, V weights** → **one "perspective"** on relationships.

But language has **many** types of relationship:
- **Syntactic** — subject → verb
- **Semantic** — pronoun → referent
- **Positional** — nearby words
- **Topical** — related concepts

One head can't capture all of these simultaneously — it has to compromise.

### The solution: parallel heads

```
Head 1: maybe learns syntax (subject–verb)
Head 2: maybe learns coreference (it → animal)
Head 3: maybe learns local context (nearby words)
Head 4: maybe learns topic relationships
```

**Each head has its OWN Q, K, V weights. Each head sees different patterns.**

Note the deck's careful hedging — *"maybe learns"*. Head specialisation is **not** assigned; it's an empirical tendency observed after training. Some heads are interpretable, many aren't.

### The dimension split

Instead of one attention over 768 dimensions:

```
8 heads, each with 96-dim Q, K, V     (768 ÷ 8 = 96)

Each head:
  Q_i = X @ W_Q_i     (96-dim queries)
  K_i = X @ W_K_i     (96-dim keys)
  V_i = X @ W_V_i     (96-dim values)
  output_i = Attention(Q_i, K_i, V_i)

Final: concatenate all heads → project back to 768
```

**Total dimension is preserved, not multiplied** — heads *split* the representation rather than duplicating it. Multi-head attention therefore costs roughly the same as single-head attention of the same width.

### Why 8? The trade-off

```
More heads  = more perspectives
But each head = smaller dimension

8 heads  × 96 dims = 768
16 heads × 48 dims = 768
```

- **Too many heads** → each is too small to capture anything meaningful
- **Too few heads** → not enough perspectives
- **Typical in practice: 8–32 heads**

A fixed budget being partitioned: you're trading *breadth of perspectives* against *depth of each*.

---

## Key takeaways

1. **Self-attention makes representations context-dependent** — solving the one-vector-per-word problem from Weeks 1 and 3.
2. **Q / K / V are three learned projections of the same token**: what I'm looking for, what I contain, what I offer.
3. **Attention is a soft database lookup** — partial matches against all keys, returning a weighted mixture of values.
4. `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` — similarity, scale, normalise, mix.
5. **√d_k scaling prevents softmax saturation**, which would kill gradients.
6. **Each attention row sums to 1** — every token distributes a fixed attention budget.
7. **The whole mechanism is ~9 lines of NumPy.**
8. **The KV cache** exploits the fact that old tokens' K and V never change — only the new token's Q is fresh.
9. **Prefill vs decode** explains time-to-first-token versus tokens-per-second.
10. **Long context is hard because of KV cache memory**: every layer × every token × ~120 layers.
11. **Multi-head attention splits dimensions** (8 × 96 = 768) to capture multiple relationship types at constant cost.

---

## Glossary

| Term | Meaning |
|---|---|
| **Self-attention** | Every token attending to every token in the same sequence. |
| **Query (Q)** | Projection representing what a token is searching for. |
| **Key (K)** | Projection representing what a token contains, used for matching. |
| **Value (V)** | Projection representing the content a token contributes when matched. |
| **W_Q / W_K / W_V** | Learned projection matrices producing Q, K, V from embeddings. |
| **Attention score** | Raw QKᵀ dot-product similarity between a query and a key. |
| **Scaling factor √d_k** | Divisor keeping scores small enough that softmax doesn't saturate. |
| **Softmax** | Turns scores into a probability distribution summing to 1. |
| **Attention weight** | The normalised amount one token attends to another. |
| **Attention matrix** | The seq_len × seq_len grid of all pairwise attention weights. |
| **Contextualized embedding** | A token representation updated using surrounding tokens. |
| **Soft lookup** | Retrieval returning a weighted mixture across all keys rather than one exact match. |
| **KV cache** | Stored K and V vectors for previously generated tokens, avoiding recomputation. |
| **Prefill** | Processing the whole prompt and filling the cache — determines time to first token. |
| **Decode** | Generating tokens one at a time using the cache — determines tokens per second. |
| **Multi-head attention** | Several attention mechanisms in parallel, each with its own projections. |
| **Head dimension** | `d_model ÷ n_heads` — e.g. 768 ÷ 8 = 96. |

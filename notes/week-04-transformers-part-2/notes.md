# Week 4 — Transformers Part 2: Self-Attention, KV Cache, Multi-Head

**Date:** 06/02/2026
**Sources:** `downloads/week-04-transformers-part-2.pdf` (42 slides) · `downloads/week-04-transformers-part-2.ipynb` (69 cells)
**Notion page:** https://100xschool.notion.site/300ffffa33e580448dcaf3f127b52b22

> **Deck note:** slides 1–23 are an exact repeat of Week 3 (tokenization → embeddings → positional encoding). **The new material starts at slide 24.** Likewise, notebook Parts 1–3 are identical to Week 3's; **Part 4 (Self-Attention, cells 47–67) is the new content** — this is the attention implementation Week 3's notebook promised but didn't contain. These notes cover only the new material; see `week-03` for the recap.

---

## 0. The idea in plain language

This is the single most important mechanism in modern AI. Take your time with it.

**The problem we're solving.** Week 3 ended with each word having a fixed vector looked up from a table. "Bank" gets one vector, whether it's a riverbank or a financial institution. That's obviously wrong, and fixing it is the whole job.

**What we want:** a word's representation should be **computed from its surroundings**, not looked up. "Bank" near "river" should produce a different vector from "bank" near "deposit."

**The idea.** Let every word *look at* every other word and pull in information from the ones that matter. "Bank" looks around, notices "river" is relevant, and absorbs some of its meaning. That's attention.

**But how does a word decide what's relevant?** This is where Q, K, V come from, and here's the intuition that makes them click.

Think about **searching on YouTube**. Three different things are involved:

- **Your search text** — "how to poach an egg." This is what you're *looking for*. → **Query**
- **Each video's title and tags** — how the video *advertises itself* so searches can find it. → **Key**
- **The video itself** — the actual content you get once you've matched. → **Value**

Notice these are genuinely three different things. A video's title (how it's found) is not the same as its content (what it gives you). And your search text is a fourth kind of thing entirely.

**Attention does exactly this, for every word simultaneously.** Every word:
- issues a **Query** — "what kind of information am I looking for?"
- advertises a **Key** — "here's what I'm about, in case anyone's looking"
- offers a **Value** — "and here's what I'll actually contribute if you pick me"

Every word's Query is compared against every word's Key. Strong matches get high weight. Then each word's new representation is a **weighted blend of everyone's Values** — mostly from the words it matched strongly.

**The one crucial difference from YouTube search:** YouTube returns the top result. Attention returns **a blend of everything**, weighted by match quality. Nothing is discarded — a word that's 5% relevant contributes 5%. That's what "soft" means in "soft lookup," and it's what makes the whole thing differentiable so Week 2's gradient descent can train it.

Everything below is this idea, made precise.

---

## 1. Self-attention — the engine of context

> **The defining feature that lets the model process information globally rather than sequentially.** It's the heart of the *"Attention is All You Need"* philosophy.

Three properties worth naming:

- **Mutual influence** — every token attends to every other token **simultaneously**, in one parallel operation rather than a sequential scan.
- **Contextualization** — words don't have a single fixed meaning; their representation is *refined and updated* based on what surrounds them.
- **Interconnectivity** — a dense web of pairwise connections lets the model resolve ambiguity and track dependencies at any distance.

### What it buys you

| Aspect | What it does |
|---|---|
| **Filtering noise** | Ignore irrelevant words, amplify signal from related ones |
| **Dynamic weighting** | Assign a unique importance score to **every word pair**, recomputed per sentence |
| **Reference resolution** | *"The bank was closed because it was a holiday"* — attention links "it" to "bank" |
| **Semantic context** | Meaning is defined by relationships; attention is the tool that maps them |

**"Self"-attention** means the sequence attends to *itself* — queries, keys, and values all come from the same input. (Cross-attention, where queries come from one sequence and keys/values from another, is used in encoder–decoder models like translation systems. Modern LLMs are decoder-only and use self-attention throughout.)

---

## 2. Q, K, V — the three roles of every word

This is the central abstraction. **Every token plays three roles at once**, and each role is a different transformation of the same underlying vector.

| Role | Question it answers | Meaning |
|---|---|---|
| **Query (Q)** | *"What am I looking for?"* | The search criteria this token uses to find relevant information |
| **Key (K)** | *"What do I contain?"* | The label other tokens use to judge whether this token is relevant to *their* search |
| **Value (V)** | *"What information do I offer?"* | The actual content passed along when a Query and Key match |

**The interaction:** the similarity between a Query and a Key determines the **attention weight** applied to the corresponding Value.

Three perspectives on the same word:
- **The Searcher (Q)** — the version used to find context in other words
- **The Label (K)** — the version other words use to find this one
- **The Content (V)** — the version that provides information once matched

> **Why three and not one?** This is the question that trips people up, so it's worth answering directly. If you used the raw embedding for all three roles, then "how well does A match B" would be symmetric — A matching B would equal B matching A. But relevance *isn't* symmetric. In "the animal didn't cross the street because **it** was tired," the word "it" desperately needs "animal," while "animal" doesn't particularly need "it." Separate Q and K projections let the model learn asymmetric relationships. And separating V lets the *information passed* differ from the *matching criterion* — exactly like a video's title differing from its content.

### The database analogy

| Hard lookup (traditional DB) | Soft lookup (attention) |
|---|---|
| Finds an **exact match** for a key | Finds **partial matches across all keys** |
| Returns a **single value** | Returns a **mixture of values**, weighted by match quality |
| Binary: match or no match | Continuous: everything contributes, some more than others |

- **Weighted sum** — a word's final representation is a weighted combination of *all* words' Values.
- **Softmax** — converts raw similarity scores into weights summing to 1, determining the mix.

This is genuinely the best mental model: **attention is a differentiable, fuzzy database query**, where every row returns something and the query decides how much of each. "Differentiable" is the load-bearing word — because nothing is a hard cutoff, gradients flow through it and Week 2's training loop works unchanged.

### Computing Q, K, V

A single embedding is multiplied by **three separate learned matrices** — `W_Q`, `W_K`, `W_V` — producing its Query, Key, and Value vectors.

```
x  (the token's embedding, 768 numbers)
   ├─ × W_Q →  q   (what I'm looking for)
   ├─ × W_K →  k   (how I advertise myself)
   └─ × W_V →  v   (what I contribute)
```

**These matrices are learned during training** by exactly the gradient descent from Week 2. Nothing about Q/K/V is hand-designed — only the *structure* is. The model works out for itself what projections make attention useful, and it does so because bad projections produce bad next-token predictions, which produce high loss, which produces gradients that fix them.

---

## 3. The attention formula

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V
```

Four operations. Taking them one at a time:

| Component | Purpose |
|---|---|
| **QKᵀ (dot product)** | Measures similarity between every Query and every Key |
| **÷ √d_k (scaling)** | Keeps the numbers small enough that softmax doesn't saturate |
| **softmax** | Converts raw scores into weights that sum to 1 |
| **× V (weighted sum)** | Applies those weights to Values, producing the context-aware output |

### Why the dot product measures similarity

If this is new: the dot product of two vectors multiplies them element-wise and sums the results. `[1,2,3]·[2,0,1] = 2 + 0 + 3 = 5`.

The useful property is that it's **large when two vectors point in the same direction** and small (or negative) when they point differently. So "how well does this Query match this Key" becomes a single multiply-and-sum. `QKᵀ` computes this for *every* query–key pair at once, producing a grid.

### Why the √d_k scaling matters

This is the least obvious term, and it's a direct callback to Week 2's vanishing gradient problem.

Dot products of higher-dimensional vectors tend to grow larger in magnitude — more terms are being summed. Feed large values into softmax and it **saturates**: one weight goes to ~1.0 and the rest collapse to ~0.

A saturated softmax has **near-zero gradient** — it's flat. Flat means no learning signal, so training stalls. Dividing by `√d_k` (the square root of the key dimension) keeps the distribution soft enough that gradients flow.

**Same underlying problem as sigmoid saturation in Week 2, different location.** Worth noticing the pattern: a lot of deep learning engineering is keeping numbers in a range where gradients survive.

### The attention matrix

The result of `softmax(QKᵀ/√d_k)` is a **square grid** — every word compared to every other word.

- **Heatmap logic** — each cell is an attention weight; brighter means stronger relationship
- **Rows sum to 1** — each word has a fixed budget of attention to distribute
- **Self-focus** — words often attend strongly to themselves, preserving their own identity while absorbing context
- **Context-focus** — the model learns to link pronouns to antecedents, verbs to subjects

> **Note the shape: `seq_len × seq_len`.** For 1,000 tokens that's a million cells; for 10,000 tokens, a hundred million. **Attention cost grows with the square of sequence length.** This one fact drives the entire long-context problem, and it's why S10's FlashAttention and Week 6's efficiency work exist.

### Context-aware meaning — the payoff

> *"I went to the bank..."* vs *"The bank was closed..."*

Static embeddings give a **blended average** meaning of "bank," useful for neither sense. Attention **updates the representation of "bank" using the surrounding tokens** — attending to "money," "loan," or "river" pulls in the features that resolve the ambiguity.

**This closes the loop opened in Week 1 §10 and Week 3 §6.** The one-vector-per-word limitation is solved: representations are now *computed from context* rather than looked up.

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

Three words, each an 8-number vector. Deliberately tiny so every number is inspectable.

### Step 1 — Project to Q, K, V

```python
W_Q = np.random.randn(d_model, d_k) * 0.1   # (8, 4)
W_K = np.random.randn(d_model, d_k) * 0.1   # (8, 4)
W_V = np.random.randn(d_model, d_v) * 0.1   # (8, 4)

Q = X @ W_Q    # (3,8) @ (8,4) = (3,4)
K = X @ W_K    # (3,4)
V = X @ W_V    # (3,4)
```

**Reading the shapes:** `(3,8) @ (8,4) = (3,4)` — 3 words in, each 8-dimensional; out come 3 words, each 4-dimensional. The inner dimensions (8 and 8) must match and cancel; the outer ones survive. Tracking shapes like this is the single most useful debugging habit in deep learning.

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

Row *i*, column *j* = how well word *i*'s Query matches word *j*'s Key. Note the matrix is **not symmetric** — that's the asymmetry Q and K exist to provide.

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

*(The `- np.max(...)` doesn't change the result mathematically — softmax is shift-invariant — but it prevents `exp()` overflowing on large inputs. A standard trick worth recognising.)*

|  | The | cat | sat | Sum |
|---|---|---|---|---|
| **The** | 0.320 | 0.373 | 0.307 | 1.000 |
| **cat** | 0.339 | 0.321 | 0.341 | 1.000 |
| **sat** | 0.384 | 0.272 | 0.344 | 1.000 |

**Every row sums to 1** — each word distributes a fixed budget of attention across all words.

> 🔍 **Read these numbers honestly:** they're all ≈0.33, i.e. nearly uniform. That's *expected* — the embeddings and projection matrices are **random**, not trained. A trained model produces sharply peaked rows where one or two words dominate. This demo shows the **mechanism**, not learned behaviour. Don't mistake the flat distribution for a bug, and don't conclude attention "doesn't do much."

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

**This is the whole mechanism.** A word's output is literally a weighted average of every word's Value vector, with attention supplying the weights. Nothing more mysterious than that.

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

**Nine lines.** The mechanism behind every modern LLM fits in nine lines of NumPy. Everything else — the scale, the data, the training infrastructure — is engineering around this core.

---

## 5. Causal masking — why generation works at all

*(Added — not covered in the deck, but essential, and everything in §6 depends on it.)*

There's a problem with what we've built. In the attention matrix above, **every word attends to every other word, including words that come after it.** "The" attends to "sat."

For understanding a complete sentence, that's fine and desirable. But LLMs are trained to **predict the next word** — and if the model can see the next word while predicting it, it learns nothing. It's an exam where the answer is printed on the question paper. The model would score perfectly in training and produce garbage at generation time, when future words genuinely don't exist yet.

**The fix: a causal mask** (also called a look-ahead mask). Before the softmax, set every score where *j > i* — attending to a future position — to **negative infinity**.

```
Before mask:                After mask:
     The   cat   sat             The   cat   sat
The  0.3   0.4   0.3        The  0.3   -inf  -inf
cat  0.3   0.3   0.4        cat  0.3   0.3   -inf
sat  0.4   0.3   0.3        sat  0.4   0.3   0.3
```

Why negative infinity? Because `exp(-inf) = 0`, so after softmax those positions get **exactly zero weight**, and the remaining weights renormalise to sum to 1. The future is not merely discouraged, it's structurally unreachable.

**The result is a triangular attention matrix.** Token 1 sees only itself. Token 2 sees tokens 1–2. Token 50 sees tokens 1–50. Nothing ever sees ahead.

**Two things this explains:**

**Why training is efficient.** With the mask in place, a single forward pass over a 1,000-token sequence trains 1,000 next-token predictions *simultaneously* — position 5 predicts token 6, position 6 predicts token 7, and so on, all in parallel, all valid because none of them could cheat. This is the property that makes transformer training practical at scale.

**Why the KV cache works** (§6). Because token 5's representation can never depend on token 6, adding a new token never changes any previous token's Keys or Values. Without causality, the whole cache would have to be recomputed every step.

> **Terminology:** models using this are **decoder-only** or **causal** models — GPT, Claude, Llama. Models *without* the mask, where every token sees everything, are **encoder** or **bidirectional** models — BERT, and most embedding models. That's why S6 notes embedding models are bidirectional: they aren't predicting a next token, so there's nothing to cheat at, and seeing the whole input is strictly better.

---

## 6. The KV cache

This section is the most practically valuable in the week — it explains latency behaviour you observe every day.

### Generation is repeated attention over a growing sequence

```
Step 1: "The"          → Q,K,V → attention → next token "cat"
Step 2: "The cat"      → Q,K,V → attention → next token "sat"
Step 3: "The cat sat"  → Q,K,V → attention → next token "on"
```

Each step runs attention over the **entire** sequence so far.

### The key observation

When generating token 4, the new token must attend to all previous tokens:

- **New token's Q** — *"What am I looking for?"* → needs fresh computation
- **Previous tokens' K** — *"What do they contain?"* → **these don't change.** Same tokens, same keys.
- **Previous tokens' V** — *"What do they contribute?"* → **these don't change either.**

> **Only the NEW token's Q changes each step.**

And this is only true **because of the causal mask** (§5). Old tokens can't see the new one, so nothing about them updates.

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

K and V for old tokens get recomputed at every single step. Pure waste, and it accumulates quadratically over a long generation.

### The solution

**Store K and V after computing them once.**

```
Step 1: "The" → compute K_the, V_the → CACHE
Step 2: "cat" → compute K_cat, V_cat → CACHE
                reuse cached K_the, V_the
Step 3: "sat" → compute K_sat, V_sat → CACHE
                reuse cached K_the, K_cat, V_the, V_cat
```

Only compute K and V for the **new** token each step. Q is always fresh — which is fine, because you only ever need the current token's Query.

### Why the first token is slow

```
FIRST token (slow) — "prefill":
  → Process your ENTIRE prompt
  → Compute K, V for every token in it
  → Fill the KV cache
  → Generate the first response token

SUBSEQUENT tokens (fast) — "decode":
  → K, V for the prompt are already cached
  → Only compute for the one new token
```

> **"Time to first token" and "tokens per second" are different metrics. Now you know why.**

This is the **prefill vs decode** distinction, and it's the foundation of S5's entire cost discussion:

- **Prefill** is compute-bound and scales with prompt length. Long prompt → slow start.
- **Decode** is memory-bandwidth-bound and roughly constant per token. Once started, it streams steadily.

It's exactly why a long prompt with a short answer feels sluggish to begin and then finishes quickly.

### Why long context is hard

```
100 tokens     → small cache, fast
10,000 tokens  → large cache, slower
100,000 tokens → HUGE cache, memory problems
```

**Every layer stores K and V for every token.** With ~120 layers in a large model, a 128K context means an enormous memory footprint — often *larger than the model's weights*.

**The KV cache, not the parameter count, is frequently what limits how many concurrent users a single GPU can serve.** This one fact explains long-context pricing, why context limits exist at all, and much of modern inference engineering. It's also the direct motivation for **grouped-query attention** (Week 6) and **PagedAttention** (S5), both of which exist to shrink or better manage this cache.

---

## 7. Multi-head attention

### One head isn't enough

A single attention head has **one set of Q, K, V weights** → **one "perspective"** on relationships.

But language has many *kinds* of relationship operating simultaneously:
- **Syntactic** — subject → verb agreement
- **Semantic** — pronoun → referent
- **Positional** — nearby words
- **Topical** — related concepts across a passage

One head has to compromise between all of these — a single set of projections can only encode one notion of "relevant."

### The solution: parallel heads

```
Head 1: maybe learns syntax (subject–verb)
Head 2: maybe learns coreference (it → animal)
Head 3: maybe learns local context (nearby words)
Head 4: maybe learns topic relationships
```

**Each head has its own Q, K, V weight matrices, so each head sees different patterns.** They run in parallel, and their outputs are concatenated and projected back.

> **Note the deck's careful hedging — *"maybe learns."*** Head specialisation is **not assigned**; it's an empirical tendency observed after training. Researchers have found interpretable heads (a genuine "previous token head," a genuine coreference head), but many heads do nothing legible at all, and pruning studies show a lot of redundancy. Treat "head 2 does coreference" as a useful story, not a specification.

### The dimension split

Instead of one attention over 768 dimensions:

```
8 heads, each with 96-dim Q, K, V     (768 ÷ 8 = 96)

Each head:
  Q_i = X @ W_Q_i     (96-dim queries)
  K_i = X @ W_K_i     (96-dim keys)
  V_i = X @ W_V_i     (96-dim values)
  output_i = Attention(Q_i, K_i, V_i)

Final: concatenate all 8 heads (8 × 96 = 768) → project back to 768
```

**Total dimension is preserved, not multiplied.** Heads *split* the representation rather than duplicating it, so multi-head attention costs roughly the same as single-head attention of the same width. You get multiple perspectives essentially for free — which is why it's universal.

### Why 8? The trade-off

```
More heads  = more perspectives
But each head = smaller dimension

8 heads  × 96 dims = 768
16 heads × 48 dims = 768
```

- **Too many heads** → each is too small to represent anything meaningful
- **Too few heads** → not enough distinct perspectives
- **Typical in practice: 8–32 heads**

It's a fixed budget being partitioned: you're trading *breadth of perspectives* against *depth of each*.

---

## Common confusions

**"Why do we need three matrices? Couldn't one vector do all three jobs?"** No — see the callout in §2. Using the raw embedding for both Q and K would force relevance to be symmetric, and relevance isn't. Separating V additionally lets "how I'm found" differ from "what I give you."

**"Is attention the same as the model 'paying attention' like a human?"** No. It's a weighted average with learned weights. The name is evocative and slightly misleading. Nothing is being consciously selected.

**"Why is the attention matrix nearly uniform in the notebook?"** Because the weights are random, not trained. See the callout in §4. A trained model's rows are sharply peaked.

**"Does attention understand grammar?"** It learns statistical relationships that often align with grammatical ones. Some heads behave in strikingly syntactic ways. But nothing encodes grammar explicitly — it's all learned from next-token prediction.

**"Is the KV cache the same as prompt caching?"** Related but different. The KV cache lives inside a single generation and disappears afterwards. **Prompt caching** (S5) is a provider feature that persists the prefill work across *separate API requests* with a shared prefix. Same underlying idea, different scope.

**"Why does long context cost so much more?"** Two compounding reasons: attention is quadratic in sequence length (§3), and the KV cache grows linearly with length *per layer* (§6). The memory pressure is usually the binding constraint in serving.

**"If every token attends to every token, how does the model generate anything?"** It doesn't attend to future tokens — that's the causal mask in §5, which the deck omits. Without it, training would be trivially cheatable and generation impossible.

---

## Key takeaways

1. **Self-attention makes representations context-dependent** — solving the one-vector-per-word problem from Weeks 1 and 3.
2. **Q / K / V are three learned projections of the same token**: what I'm looking for, how I advertise myself, what I contribute. Three are needed because relevance is asymmetric and matching differs from content.
3. **Attention is a soft, differentiable database lookup** — partial matches against all keys, returning a weighted mixture of values. "Differentiable" is why it trains.
4. `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` — similarity, scale, normalise, mix.
5. **√d_k scaling prevents softmax saturation**, which would flatten gradients — the same class of problem as Week 2's vanishing gradients.
6. **Each attention row sums to 1** — every token distributes a fixed attention budget.
7. **The attention matrix is seq_len × seq_len**, so cost grows with the *square* of length. This drives the entire long-context problem.
8. **The whole mechanism is ~9 lines of NumPy.**
9. **Causal masking sets future positions to −∞** so they get exactly zero weight. It's what makes training parallelisable *and* what makes the KV cache valid.
10. **The KV cache** exploits the fact that old tokens' K and V never change — only the new token's Q is fresh.
11. **Prefill vs decode** explains time-to-first-token versus tokens-per-second, and it's the basis of inference cost.
12. **Long context is hard because of KV cache memory**: every layer × every token, often exceeding the model's own weights.
13. **Multi-head attention splits dimensions** (8 × 96 = 768) to capture multiple relationship types at essentially constant cost.

---

## Glossary

| Term | Meaning |
|---|---|
| **Self-attention** | Every token attending to every token in the same sequence. |
| **Cross-attention** | Queries from one sequence, keys/values from another (encoder–decoder models). |
| **Query (Q)** | Projection representing what a token is searching for. |
| **Key (K)** | Projection representing what a token contains, used for matching. |
| **Value (V)** | Projection representing the content a token contributes when matched. |
| **W_Q / W_K / W_V** | Learned projection matrices producing Q, K, V from embeddings. |
| **Dot product** | Element-wise multiply-and-sum; large when vectors point the same way. |
| **Attention score** | Raw QKᵀ similarity between a query and a key. |
| **Scaling factor √d_k** | Divisor keeping scores small enough that softmax doesn't saturate. |
| **Softmax** | Turns scores into a distribution summing to 1. |
| **Attention weight** | The normalised amount one token attends to another. |
| **Attention matrix** | The seq_len × seq_len grid of all pairwise attention weights. |
| **Causal mask** | Setting future positions to −∞ so a token cannot attend ahead. |
| **Decoder-only / causal model** | A model using causal masking — GPT, Claude, Llama. |
| **Bidirectional / encoder model** | A model without the mask, seeing the whole input — BERT, embedding models. |
| **Contextualized embedding** | A token representation updated using surrounding tokens. |
| **Soft lookup** | Retrieval returning a weighted mixture across all keys rather than one exact match. |
| **KV cache** | Stored K and V vectors for previous tokens, avoiding recomputation. |
| **Prefill** | Processing the whole prompt and filling the cache — determines time to first token. |
| **Decode** | Generating tokens one at a time using the cache — determines tokens per second. |
| **Multi-head attention** | Several attention mechanisms in parallel, each with its own projections. |
| **Head dimension** | `d_model ÷ n_heads` — e.g. 768 ÷ 8 = 96. |

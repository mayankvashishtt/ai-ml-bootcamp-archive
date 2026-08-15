# Week 3 — Transformers Part 1: From Text to Attention

**Subtitle:** How Transformers See Language — connecting conceptual foundations to mechanical implementation
**Date:** 31/01/2026
**Sources:** `downloads/week-03-transformers-part-1.pdf` (24 slides) · `downloads/week-03-transformers-part-1.ipynb` (48 cells)
**Notion page:** https://100xschool.notion.site/2faffffa33e5805b9bb0f74a8754cf79

---

## What this week is for

Week 1 said *"attention focuses on relevant words."* Week 2 said *"models learn by adjusting weights."* Both were true and both were hand-wavy. This week fills three specific gaps:

| What we already know | The missing pieces |
|---|---|
| ✓ Neural networks process numbers | ? How does text become numbers? |
| ✓ Models learn by adjusting weights | ? What does the model actually "see"? |
| ✓ Attention focuses on relevant words | ? How does attention work mechanically? |

**Part 1 covers the input pipeline** — everything that happens *before* attention:

```
Raw text → [1] Tokenization → [2] Embedding → [3] Positional Encoding → ready for attention
```

> ⚠️ **Notebook scope note:** the notebook's intro promises "4. How attention actually works — coded from scratch," but the shipped notebook ends at the Positional Encoding checkpoint with *"→ Back to slides for Self-Attention."* Part 4 is not in this file — attention is coded in **Week 4**. Don't hunt for a missing cell; it isn't there.

---

## 1. The fundamental challenge

> Neural networks only understand **NUMBERS**. Human language is **TEXT**.

We need a translation layer that converts text into numbers **while preserving semantic relationships**. That last clause is the hard part — any mapping produces numbers, but we need one where the numbers *mean something*.

---

## 2. Tokenization — three attempts

### Attempt 1: Character-level

One integer per character. `"Hello"` → `[1, 2, 3, 3, 4]`.

Simple, tiny vocabulary — and it operates at the smallest possible unit, ignoring words and concepts entirely.

**Why it struggles:**

| Problem | Detail |
|---|---|
| **The step problem** | "understanding" needs 13 separate steps to process. The model works many times harder to see one concept. |
| **Manual pattern learning** | It must learn prefixes and roots from raw characters instead of receiving those units directly. |
| **High cognitive load** | Like teaching reading by showing only letters — grammar and meaning become much harder to build. |
| **Extreme inefficiency** | Enormous data is spent learning basic vocabulary rather than higher-level logic. |

### Attempt 2: Word-level

One integer per whole word. `"The cat sat"` → `[1, 2, 3]`. Intuitive — each word is a discrete unit of meaning.

**Why it struggles:**

| Problem | Detail |
|---|---|
| **The "form" problem** | Are "cat", "cats", and "cat's" three unrelated concepts? Word-level says yes, which is wrong. |
| **The unknown** | Typos like "teh", or rare technical terms — no entry exists. |
| **Out of vocabulary (OOV)** | Any unseen word becomes a **hole** in understanding. |

**The vocabulary explosion:**
- **170,000+** common English words
- **~3 billion** parameters just for the lookup table
- Each unique word needs its own embedding row → large memory overhead
- Fixed vocabularies fail on names, slang, new terms
- **Multilingual scaling** multiplies vocabulary, driving up compute and storage

### Attempt 3: Subword — the Goldilocks solution ✅

**Bigger than characters** (too granular), **smaller than whole words** (too many).

- Common words stay whole
- Rare words break into recognisable, meaningful pieces
- `"understanding"` → `["under", "stand", "ing"]`
- **A vocabulary of 30k–100k tokens can represent almost any text in any language**

This solves OOV completely: there is no such thing as an unknown word, because anything unseen decomposes into known pieces (in the worst case, individual characters).

---

## 3. Byte Pair Encoding (BPE)

The algorithm that builds modern tokenizers:

1. **Initialisation** — begin with each character as its own token
2. **Iterative merging** — merge the most frequent adjacent token pair into one
3. **Optimisation** — repeat until you hit the target vocabulary size
4. **Self-organising** — frequent patterns become single tokens; rare ones stay fragmented

The elegance: **the vocabulary is learned from data, not designed.** Frequency decides what deserves to be one token. Nobody sat down and chose these splits.

---

## 4. Tokenization in practice — real GPT-4 behaviour

The notebook uses **`tiktoken`** with GPT-4's `cl100k_base` encoding — **vocabulary size 100,277 tokens**.

```python
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode("Hello world")   # [9906, 1917]
```

### Real splits observed

| Text | Tokens | Pieces |
|---|---|---|
| `"Hello world"` | 2 | `['Hello', ' world']` |
| `"don't"` | 2 | `['don', "'t"]` |
| `"artificial intelligence"` | 3 | `['art', 'ificial', ' intelligence']` |
| `"I love AI"` | 3 | `['I', ' love', ' AI']` |
| `"supercalifragilisticexpialidocious"` | 11 | `['sup','erc','al','if','rag','il','istic','exp','ial','id','ocious']` |
| `"my name is rishabh"` | 6 | `['my',' name',' is',' r','ish','abh']` |
| `"🚀"` | 3 | (multi-byte emoji spans several tokens) |

**Three things to notice:**

1. **Spaces attach to the *following* word** — `' world'` includes its leading space. This is why `"hello"` and `" hello"` are different tokens, and a stray trailing space in a prompt can change tokenisation.
2. **Splits don't respect linguistics.** `"artificial"` → `['art', 'ificial']` is not a morpheme boundary. BPE optimises frequency, not meaning.
3. **Common names fragment.** `"rishabh"` → `[' r', 'ish', 'abh']`. Non-English and less common names cost more tokens.

---

## 5. Why tokenization matters — the practical consequences

### Operational impact

- **Context limits.** GPT-4's 128K limit is **tokens, not words.** English averages ~1.3 tokens per word; **code is much higher.**
- **Financial cost.** API billing is per token. Efficient prompting requires understanding token density.

**Measured token efficiency from the notebook:**

| Content type | Chars | Tokens | Chars/token |
|---|---|---|---|
| English prose | 44 | 10 | **4.4** |
| URL | 48 | 11 | **4.4** |
| Python code | 39 | 11 | **3.5** |
| JSON | 43 | 19 | **2.3** |
| Numbers | 32 | 14 | **2.3** |

**JSON costs nearly 2× English prose per character.** Structural punctuation (`{`, `"`, `:`, `,`) tokenises terribly. Practical upshot: if you're stuffing JSON into a prompt, you're paying roughly double — worth considering a more compact format when context is tight.

### Model behaviour — the token boundary problem

This explains the famous "dumb" LLM failures:

```
"strawberry" → ['str', 'aw', 'berry']
```

> *"The model sees these pieces, not individual letters. Counting 'r's requires looking INSIDE tokens — that's hard."*

```
"hello" → ['hello']     # a single atomic token
```

> *"If 'hello' is ONE token, the model can't easily reverse it. It would need to decompose something it sees as atomic."*

**This is the correct explanation for the strawberry-r-counting meme.** It isn't a reasoning failure — it's a *perception* failure. The model is being asked about structure it cannot see, like asking someone to count brushstrokes in a photo of a painting.

**Language equity:** different languages need different token counts for the same concept, which affects **both cost and performance** globally. English is the best-served language by these tokenizers; many others pay a systematic tax in both money and effective context.

### The summary principle

> **If the model can't "see" a pattern in the tokens, it can't learn the underlying meaning.**

The tokenizer determines the fundamental units the model uses for "thought."

---

## 6. From arbitrary IDs to meaning — embeddings

**The problem:** `"Hello world"` → `[15496, 995]`. These IDs are **arbitrary**. 15496 doesn't "know" it's a greeting; 995 doesn't mean anything about Earth. Nearby IDs are unrelated in meaning.

**The goal:** transform discrete arbitrary integers into **continuous dense vectors** where the numbers themselves encode semantics — "Hello" and "Hi" mathematically similar, "Hello" and "Banana" distant.

### How the lookup works

```
ID 15496 → [0.12, -0.45, 0.88, ...]     # a 768-dim vector for base models
```

Each token ID **indexes a row** in the embedding matrix. It is a simple, fast lookup — not a computation.

```python
vocab_size, embedding_dim = 50000, 768
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.02
# → 38,400,000 parameters just for embeddings
```

**38.4M parameters before any actual processing.** Embeddings are a substantial fraction of a small model's size.

### Properties

- **Spatial logic** — similar meanings sit near each other; proximity = semantic similarity
- **Multidimensionality** — hundreds or thousands of dimensions capture subtle nuance
- **Emergent structure** — dimensions like "gender" or "royalty" emerge **automatically during training, without manual labelling**

### Real embeddings in action (GloVe, 100-dim, 400k vocab)

Most similar to **"king"**: `prince 0.768 · queen 0.751 · son 0.702 · brother 0.699 · monarch 0.698 · throne 0.692 · kingdom 0.681`

Most similar to **"computer"**: `computers 0.875 · software 0.837 · technology 0.764 · pc 0.737 · hardware 0.729`

### Vector arithmetic, verified

```
king − man + woman  →  queen 0.770 · monarch 0.684 · throne 0.676 · princess 0.652
paris − france + germany → berlin 0.885 · frankfurt 0.799 · vienna 0.768
bigger − big + cold → cooler 0.688 · warmer 0.685 · colder 0.675
```

**Directional meaning:** the man→woman vector mirrors the king→queen vector. Relations are encoded as **consistent offsets** that recur across the vocabulary.

> 🔍 **Note the third analogy honestly:** `bigger − big + cold` returns *cooler* (0.688) and *warmer* (0.685) above *colder* (0.675) — and "warmer" is semantically the opposite of what we wanted. Embeddings capture the *comparative-form* direction well but are weak on **antonym polarity**, because opposites appear in near-identical contexts. This is a real limitation, not a fluke, and it's exactly why context-dependent representations are needed.

PCA projection of 24 words in four groups (royalty, family, animals, tech) shows clean clustering by meaning — **structure that emerges from training, not manual design.**

### The limitation that motivates everything next

> **Embeddings are "bag-of-words" by default.** They represent isolated meaning but **ignore the order of words.**

This vector is the **"raw" starting point** — general meaning *before* any context from the surrounding sentence is applied.

---

## 7. The problem of sequence

```
"The dog bit the man"   vs   "The man bit the dog"
```

Without extra information, attention treats these **identically**. The mathematical operations don't care about position.

This is **permutation invariance** — *the model sees a set of words, not a sequence.*

Verified in the notebook:
```
'dog bites man' → [18964, 49433, 893]
'man bites dog' → [1543, 49433, 5679]
# Same tokens, just reordered
```

**In human language, position is often the primary driver of meaning** — order determines who did what to whom. So: **we must inject "where" a word is into "what" a word is.**

---

## 8. Positional encoding

**The mechanism:** add a small positional vector to each embedding, so the result encodes **both meaning and position**.

### How the patterns work

- **Periodic functions** — sine and cosine waves of varying frequencies; each embedding dimension follows a different wave
- **Unique fingerprint** — every position (0, 1, 2, …) gets a unique combination of values
- **Relative distance** — the model can compute distance between words because the waves are mathematically related
- **Generalisation** — handles sequences longer than those seen during training

```python
def get_positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)   # even indices: sine
    pe[:, 1::2] = np.cos(position * div_term)   # odd indices: cosine
    return pe
```

**Reading the heatmap:** low dimensions = low-frequency waves that change slowly across positions; high dimensions = high-frequency waves that change quickly. Together they form a multi-scale code — like a binary counter where each bit flips at a different rate, giving both fine and coarse position information.

### Why sine waves specifically?

1. **Relative position** — for any fixed offset *k*, position *p+k* is a **linear function** of position *p*. This lets the model learn to attend by *relative* position easily.
2. **Generalisation** — the waves are continuous, so the model can extrapolate to longer sequences than it trained on.
3. **No learned parameters** — the patterns are fixed and pre-calculated; they cost nothing to train.
4. **Unique multi-scale patterns** — multiple frequencies give every position a unique fingerprint in high-dimensional space.

### Position gets *added*, not concatenated

```python
final_embeddings = embeddings + pe     # (3, 768) + (3, 768) → (3, 768)
```

**Addition, not concatenation** — dimensionality stays the same. This seems like it should corrupt the meaning, and in a sense it does perturb it; it works because the model has enough dimensions to disentangle the two signals during training.

Verified: the same word "the" at position 0 vs position 5 has a **Euclidean distance of 14.10** between representations. Same word, genuinely different vector.

---

## 9. The pipeline so far

| Step | Operation | Result |
|---|---|---|
| **1. Tokenization** | Break raw text into subword units | Token IDs |
| **2. Embedding** | Look up each ID in a matrix | Dense semantic vectors |
| **3. Positional encoding** | Add sinusoidal position patterns | Vectors encoding *what* **and** *where* |

> Now that each token knows **WHAT** it is and **WHERE** it is, we can move to the core of the transformer.

**Next: Self-Attention — The Engine of Context** (Week 4)

---

## Key takeaways

1. **Subwords are the Goldilocks unit** — characters are too granular, words too many; 30k–100k subword tokens cover any text in any language and eliminate OOV.
2. **BPE learns the vocabulary from data** by iteratively merging the most frequent adjacent pair.
3. **Tokens are the unit of cost, context, and perception.** 128K means tokens; billing means tokens; and what the model *cannot see* it cannot learn.
4. **JSON and numbers cost ~2× English prose per character** (2.3 vs 4.4 chars/token).
5. **The strawberry problem is a perception failure, not a reasoning failure** — the model sees `['str','aw','berry']`, never the letters.
6. **Embeddings turn arbitrary IDs into positioned meaning**, with semantic structure emerging during training rather than being designed.
7. **Vector arithmetic works** (king − man + woman ≈ queen) but has real weaknesses — notably antonym polarity.
8. **Raw embeddings are permutation-invariant**: "dog bites man" and "man bites dog" are indistinguishable without position.
9. **Sinusoidal positional encoding** is added to embeddings; it needs no parameters, encodes relative distance linearly, and extrapolates beyond training length.

---

## Glossary

| Term | Meaning |
|---|---|
| **Token** | The atomic unit a model reads — typically a subword. |
| **Tokenizer** | The component mapping text ↔ token IDs. Determines what the model can perceive. |
| **BPE** | Byte Pair Encoding — builds a vocabulary by repeatedly merging the most frequent adjacent pair. |
| **`cl100k_base`** | GPT-4's tiktoken encoding; 100,277 tokens. |
| **OOV** | Out of vocabulary — a word with no entry. Solved by subword tokenization. |
| **Vocabulary explosion** | Word-level tokenizers needing 170k+ entries and billions of lookup parameters. |
| **Token boundary problem** | Tasks requiring reasoning *within* a token (letter counting, reversal) are hard because sub-token structure is invisible. |
| **Embedding matrix** | The `vocab_size × d_model` lookup table mapping IDs to vectors. |
| **`d_model`** | Embedding dimensionality — 768 for base-size models. |
| **Dense vector** | A continuous vector where every dimension carries information (vs a sparse one-hot). |
| **Semantic similarity** | Closeness in embedding space, measured e.g. by cosine similarity. |
| **Bag-of-words** | A representation ignoring word order. |
| **Permutation invariance** | Producing identical output regardless of input order — attention's default state without positional encoding. |
| **Positional encoding** | Position information added to embeddings; sinusoidal in the original transformer. |
| **PCA / t-SNE** | Dimensionality-reduction techniques for visualising high-dimensional embeddings in 2D/3D. |

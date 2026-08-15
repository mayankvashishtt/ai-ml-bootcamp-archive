# Week 3 — Transformers Part 1: From Text to Attention

**Subtitle:** How Transformers See Language — connecting conceptual foundations to mechanical implementation
**Date:** 31/01/2026
**Sources:** `downloads/week-03-transformers-part-1.pdf` (24 slides) · `downloads/week-03-transformers-part-1.ipynb` (48 cells)
**Notion page:** https://100xschool.notion.site/2faffffa33e5805b9bb0f74a8754cf79

---

## 0. The idea in plain language

A neural network (Week 2) is arithmetic. It multiplies numbers by weights and adds them up. It has no idea what a letter is.

So before any of the clever transformer machinery can run, you have to solve a mundane-sounding problem: **turn text into numbers.** And not just any numbers — numbers where the *arithmetic means something*, so that "hello" and "hi" end up close together and "hello" and "banana" don't.

This week is that conversion, and it happens in three steps:

```
Raw text  →  [1] Tokenization  →  [2] Embedding  →  [3] Positional encoding  →  ready for attention
"the cat"     [1820, 8415]        two 768-number     same vectors, now also
              (chop into pieces)   vectors            encoding "I'm 1st, I'm 2nd"
              (look up meaning)
```

Each step exists because the previous one is insufficient:

1. **Tokenization** chops text into pieces the model has seen before. Needed because you can't have a number for every possible word.
2. **Embedding** replaces each piece's arbitrary ID number with a vector that actually encodes meaning. Needed because token ID 15496 tells you nothing — it's just a row number.
3. **Positional encoding** stamps each vector with where it sat in the sentence. Needed because attention, by default, has **no concept of word order at all** — "dog bites man" and "man bites dog" would be identical to it.

**The one idea to carry forward:** the tokenizer decides what the model can *perceive*. Anything the tokenizer hides, the model literally cannot see — which turns out to explain several famous LLM failures (§5).

> ⚠️ **Notebook scope note:** the notebook's intro promises "4. How attention actually works — coded from scratch," but the shipped file ends at the Positional Encoding checkpoint with *"→ Back to slides for Self-Attention."* Part 4 is not in this notebook — attention is coded in **Week 4**. Don't hunt for a missing cell; it isn't there.

---

## 1. What this week is for

Week 1 said *"attention focuses on relevant words."* Week 2 said *"models learn by adjusting weights."* Both true, both hand-wavy. This week closes three specific gaps:

| What we already know | The missing piece |
|---|---|
| ✓ Neural networks process numbers | ? How does text become numbers? |
| ✓ Models learn by adjusting weights | ? What does the model actually "see"? |
| ✓ Attention focuses on relevant words | ? How does attention work mechanically? (→ Week 4) |

> Neural networks only understand **NUMBERS**. Human language is **TEXT**.

We need a translation layer that converts text into numbers **while preserving semantic relationships**. That last clause is the entire difficulty. Any mapping produces numbers — assigning `a=1, b=2` produces numbers. The hard part is producing numbers whose *relationships* mirror meaning.

---

## 2. Tokenization — three attempts

The question: what's the right *unit*? Three candidates, and the failures of the first two motivate the third.

### Attempt 1: Character-level

One integer per character. `"Hello"` → `[1, 2, 3, 3, 4]`.

Simple, and the vocabulary is tiny — 26 letters plus punctuation and digits, maybe 100 entries total. No word is ever unknown, because every word is just characters.

**Why it struggles:**

| Problem | Detail |
|---|---|
| **The step problem** | "understanding" takes 13 separate steps to process. Sequences become extremely long, and since attention cost grows with the *square* of length (Week 4), this is expensive. |
| **Manual pattern learning** | The model must learn from scratch that "un-" reverses meaning and "-ing" marks a participle. With subwords, those units arrive pre-packaged. |
| **High cognitive load** | Like teaching someone to read by showing only individual letters — building up to grammar and meaning takes vastly longer. |
| **Extreme inefficiency** | Enormous training data is spent learning basic spelling rather than higher-level logic. |

### Attempt 2: Word-level

One integer per whole word. `"The cat sat"` → `[1, 2, 3]`. Intuitive — each word is a discrete unit of meaning, and sequences stay short.

**Why it struggles:**

| Problem | Detail |
|---|---|
| **The "form" problem** | Are "cat", "cats", and "cat's" three unrelated concepts? Word-level says yes, which is plainly wrong — the model must learn each independently, with no shared knowledge. |
| **The unknown** | A typo like "teh", or a rare technical term — there is simply no entry. |
| **Out of vocabulary (OOV)** | Any unseen word becomes a **hole** in understanding. Usually mapped to a generic `<UNK>` token, which discards the information entirely. |

**The vocabulary explosion, in numbers:**
- **170,000+** common English words
- Each needs its own row in the lookup table. At 768 numbers per row, that's ~130M parameters *just to store word meanings*
- Add plurals, tenses, proper nouns, and slang and it grows further
- **Multilingual** multiplies it again — one vocabulary per language is unaffordable

### Attempt 3: Subword — the Goldilocks solution ✅

**Bigger than characters** (too granular), **smaller than whole words** (too many).

- Common words stay whole: `"the"`, `"cat"`, `"running"` are each one token
- Rare words break into pieces: `"understanding"` → `["under", "stand", "ing"]`
- **A vocabulary of 30k–100k tokens can represent almost any text in any language**

**This solves OOV completely.** There is no such thing as an unknown word, because anything unseen decomposes into known pieces — and in the absolute worst case, into individual characters, which are always in the vocabulary. A word the model has never encountered still produces *something* meaningful rather than a hole.

---

## 3. Byte Pair Encoding (BPE)

The algorithm that builds modern tokenizers. It's simpler than it sounds:

1. **Initialisation** — start with every character as its own token
2. **Count** — find the most frequent adjacent pair in your training corpus
3. **Merge** — combine that pair into a single new token, and add it to the vocabulary
4. **Repeat** — until you reach the target vocabulary size

**A tiny worked example.** Suppose your corpus is mostly the word "low" and "lower":

```
Start:      l o w   l o w e r
Most frequent adjacent pair: "l o"  →  merge into "lo"
Now:        lo w    lo w e r
Most frequent pair: "lo w"  →  merge into "low"
Now:        low     low e r
Most frequent pair: "e r"  →  merge into "er"
Now:        low     low er
```

After three merges, "low" is one token and "lower" is two. That's the whole algorithm.

**The elegance: the vocabulary is learned from data, not designed.** Frequency alone decides what deserves to be a single token. Nobody sat down and chose these splits — which is also why they don't respect linguistics (§4).

**A consequence that matters later:** because merges are learned from a corpus, the tokenizer **inherits that corpus's distribution**. A tokenizer trained mostly on English will represent English efficiently and everything else badly. (S11 covers what this costs.)

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
| `"🚀"` | 3 | (a multi-byte emoji spans several tokens) |

**Three things to notice, all of which bite in practice:**

**1. Spaces attach to the *following* word.** `' world'` includes its leading space. So `"hello"` and `" hello"` are **different tokens with different IDs**. A stray trailing space at the end of your prompt genuinely changes how the next word tokenises, and this has been the cause of real, baffling prompt bugs.

**2. Splits don't respect linguistics.** `"artificial"` → `['art', 'ificial']`. That's not a morpheme boundary — "art" has nothing to do with "artificial" semantically. BPE optimises frequency, not meaning, and the model has to learn to reassemble the concept from arbitrary fragments.

**3. Common names fragment.** `"rishabh"` → `[' r', 'ish', 'abh']` — three tokens for one name. Non-English names and words cost systematically more tokens, which means more money and less content fitting in the context window.

---

## 5. Why tokenization matters — the practical consequences

### Operational impact

**Context limits are in tokens, not words.** GPT-4's "128K" is 128,000 *tokens*. English averages roughly 1.3 tokens per word, so 128K tokens is roughly 98,000 words — but **code is much denser**, so a 128K window holds far less code than that.

**Billing is per token.** Every API call is priced by tokens in and tokens out. Efficient prompting requires knowing what's expensive.

**Measured token efficiency from the notebook:**

| Content type | Chars | Tokens | Chars/token |
|---|---|---|---|
| English prose | 44 | 10 | **4.4** |
| URL | 48 | 11 | **4.4** |
| Python code | 39 | 11 | **3.5** |
| JSON | 43 | 19 | **2.3** |
| Numbers | 32 | 14 | **2.3** |

**JSON costs nearly 2× English prose per character.** Structural punctuation — `{`, `"`, `:`, `,` — tokenises terribly because those characters appear in endless combinations, so BPE never merges them into efficient units.

**Practical upshot:** if you're stuffing JSON into a prompt, you're paying roughly double what the same information would cost as prose. When context is tight, a more compact format (YAML, CSV, or plain prose) is a real saving. This comes back in S5 as a cost lever.

### Model behaviour — the token boundary problem

This is the section that explains several famous "LLMs are dumb" moments, and it's worth understanding properly because the usual explanation is wrong.

```
"strawberry" → ['str', 'aw', 'berry']
```

> *"The model sees these pieces, not individual letters. Counting 'r's requires looking INSIDE tokens — that's hard."*

```
"hello" → ['hello']     # a single atomic token
```

> *"If 'hello' is ONE token, the model can't easily reverse it. It would need to decompose something it sees as atomic."*

**This is the correct explanation for the strawberry-r-counting meme, and the important point is what kind of failure it is.** It is **not a reasoning failure** — it's a **perception failure**. The model is being asked about structure it cannot see.

The analogy: imagine someone shows you a photograph of a painting and asks how many brushstrokes it contains. You can reason perfectly well; you simply don't have access to the information. The model's "eyes" are the tokenizer, and the tokenizer threw the letters away.

**Why this matters beyond the meme:** it's a template for a whole class of failures. Whenever a model fails at something surprisingly basic, ask *"can it actually see the thing I'm asking about?"* before concluding it can't reason. The same question applies to small text in images (S9) and to exact identifiers in retrieval (S6).

**Language equity:** different languages need different token counts for the same meaning, which affects **cost, effective context length, and quality** simultaneously. English is the best-served language by these tokenizers; many others pay a systematic tax on all three. This isn't a minor technical footnote — it's a structural unfairness baked into pricing.

### The summary principle

> **If the model can't "see" a pattern in the tokens, it can't learn the underlying meaning.**

The tokenizer determines the fundamental units the model uses for thought.

---

## 6. From arbitrary IDs to meaning — embeddings

**The problem:** `"Hello world"` → `[15496, 995]`. These IDs are **arbitrary row numbers**. 15496 doesn't "know" it's a greeting. 995 has no relationship to Earth. Token 15497 is not related in meaning to 15496 — they're just adjacent rows in a table.

If you fed these integers straight into a network, it would try to do arithmetic on them, and "15496 is larger than 995" is a meaningless statement about "Hello" and "world."

**The goal:** replace each arbitrary integer with a **dense vector** whose numbers actually encode meaning — so that "Hello" and "Hi" come out mathematically similar, and "Hello" and "Banana" come out distant.

> **"Dense" vs "sparse," since the terms appear constantly:** a *sparse* representation is mostly zeros — e.g. one-hot encoding, where "Hello" is a 100,277-long vector of zeros with a single 1 at position 15496. It's huge and carries no similarity information (every pair of words is equally far apart). A *dense* vector is short (768 numbers) and every dimension carries information. Dense is what we want.

### How the lookup works

```
ID 15496 → [0.12, -0.45, 0.88, ...]     # a 768-dimension vector
```

Each token ID **indexes a row** in the embedding matrix. It's a simple, fast lookup — not a computation. Row 15496 of the table *is* the meaning of "Hello," as far as the model is concerned.

```python
vocab_size, embedding_dim = 50000, 768
embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.02
# → 38,400,000 parameters just for embeddings
```

**38.4M parameters before any actual processing happens.** For a small model, embeddings are a substantial fraction of total size. Note also that these values start **random** — the meanings are not designed, they're *learned during training* along with everything else, by the exact gradient descent from Week 2.

### Properties

- **Spatial logic** — similar meanings sit near each other; proximity means semantic similarity
- **Multidimensionality** — hundreds of dimensions capture many independent aspects of meaning at once
- **Emergent structure** — axes like "gender" or "royalty" emerge **automatically during training, without anyone labelling them**

> **How similarity is measured:** usually **cosine similarity** — the cosine of the angle between two vectors. It ranges from −1 (opposite) through 0 (unrelated) to 1 (identical direction). It measures *direction*, ignoring length, which is what you want: "cat" and "cats" should be similar regardless of how often each appears. The numbers in the tables below are cosine similarities.

### Real embeddings in action (GloVe, 100-dim, 400k vocab)

Most similar to **"king"**: `prince 0.768 · queen 0.751 · son 0.702 · brother 0.699 · monarch 0.698 · throne 0.692 · kingdom 0.681`

Most similar to **"computer"**: `computers 0.875 · software 0.837 · technology 0.764 · pc 0.737 · hardware 0.729`

Nobody wrote these relationships. They fell out of training on ordinary text.

### Vector arithmetic, verified

```
king − man + woman        →  queen 0.770 · monarch 0.684 · throne 0.676 · princess 0.652
paris − france + germany  →  berlin 0.885 · frankfurt 0.799 · vienna 0.768
bigger − big + cold       →  cooler 0.688 · warmer 0.685 · colder 0.675
```

**Directional meaning:** the man→woman offset is roughly the same vector as the king→queen offset. Relationships are encoded as **consistent directions** that recur across the whole vocabulary.

> 🔍 **Read the third analogy honestly.** `bigger − big + cold` returns *cooler* (0.688) and **warmer** (0.685) above *colder* (0.675) — and "warmer" is semantically the *opposite* of what was wanted. Embeddings capture the *comparative-form* direction well but are **weak on antonym polarity**, because opposites appear in nearly identical contexts ("the water was warmer/colder than expected"). Co-occurrence-based training cannot easily separate them. This is a real, well-known limitation rather than a fluke, and it's part of why context-dependent representations are needed.

A PCA projection of 24 words in four groups (royalty, family, animals, tech) shows clean clustering by meaning — **structure that emerges from training, not manual design.**

### The limitation that motivates everything next

> **Embeddings are "bag-of-words" by default.** They represent isolated meaning but **ignore word order entirely.**

This vector is the **"raw" starting point** — a word's general meaning *before* any context from the surrounding sentence has been applied. It's the exact "one vector per word" problem Week 1 §10 identified. Attention (Week 4) is what fixes it.

---

## 7. The problem of sequence

```
"The dog bit the man"   vs   "The man bit the dog"
```

Without extra information, attention treats these **identically**. The mathematical operations involved simply don't reference position — they'd produce the same output for any ordering of the same words.

This property is called **permutation invariance** — *the model sees a set of words, not a sequence.*

Verified in the notebook:
```
'dog bites man' → [18964, 49433, 893]
'man bites dog' → [1543, 49433, 5679]
# Same tokens, just reordered
```

**In human language, position is often the primary carrier of meaning** — word order determines who did what to whom. A model that can't distinguish these two sentences is useless.

So: **we must inject "where" a word is into "what" a word is.**

---

## 8. Positional encoding

**The mechanism:** add a small positional vector to each embedding, so the result encodes **both meaning and position**.

### The intuition first

Think of each position getting a unique "fingerprint" made of waves. Position 0 gets one pattern, position 1 a slightly different one, position 2 different again. Because the patterns come from waves of many different frequencies, every position ends up with a combination no other position has — and, crucially, positions that are *near* each other get *similar* patterns.

The closest everyday analogy is a **binary counter**: in `000, 001, 010, 011, 100...` the rightmost bit flips every step, the next every two steps, the next every four. Multiple "rates of change" combine to give every number a unique code. Sinusoidal encoding is the same idea with smooth waves instead of bits.

### The implementation

```python
def get_positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)   # even indices: sine
    pe[:, 1::2] = np.cos(position * div_term)   # odd indices: cosine
    return pe
```

`div_term` produces a range of frequencies — some waves oscillate quickly, others slowly. Even-numbered dimensions use sine, odd-numbered use cosine.

**Reading the heatmap:** low dimensions are low-frequency waves that change slowly across positions (coarse "roughly where in the sentence"); high dimensions are high-frequency waves that change rapidly (fine "exactly which position"). Together they form a multi-scale code.

### Why sine waves specifically?

1. **Relative position comes out linearly.** For any fixed offset *k*, the encoding at position *p+k* is a linear function of the encoding at position *p*. This means the model can learn "attend to the token 3 places back" as a simple operation, rather than memorising every absolute position pair.
2. **Generalisation.** The waves are continuous and defined for any position, so the encoding exists for sequences longer than anything seen in training.
3. **No learned parameters.** The patterns are fixed and computed once. They cost nothing to train and nothing to store.
4. **Unique multi-scale fingerprints.** Multiple frequencies guarantee every position is distinguishable.

### Position gets *added*, not concatenated

```python
final_embeddings = embeddings + pe     # (3, 768) + (3, 768) → (3, 768)
```

**Addition, not concatenation** — the dimensionality stays 768 rather than doubling.

This is genuinely counterintuitive: you're adding a position signal directly on top of the meaning signal, which should surely corrupt the meaning. And it does perturb it. **It works because 768 dimensions is a lot of room** — the model has enough capacity to learn to disentangle "how much of this vector is meaning" from "how much is position" during training. It's not elegant, but it's cheap and it works.

Verified: the same word "the" at position 0 versus position 5 has a **Euclidean distance of 14.10** between its two representations. Same word, genuinely different vector — which is exactly what we wanted.

> **Worth knowing for Week 6:** sinusoidal encoding is the *original* (2017) approach. Modern models mostly use **RoPE (Rotary Position Embedding)**, which encodes position by *rotating* the query and key vectors rather than adding to them. It handles long contexts better. Week 6 covers why the field moved.

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

## Common confusions

**"Is a token a word?"** No. Sometimes a token is a whole word, sometimes several tokens make one word, and sometimes a token is a fragment plus a leading space. Roughly 1.3 tokens per English word on average — but that average hides enormous variation.

**"Why does the space belong to the *next* word?"** It's just how the BPE merges shook out for these tokenizers, and it's consistent enough that models rely on it. The practical consequence is that `"hello"` ≠ `" hello"` as far as the model is concerned.

**"Are the 768 embedding dimensions interpretable?"** No. Nobody labelled dimension 4 "royalty." The model learns whatever axes minimise its loss, and those rarely correspond to human concepts. Week 1's royalty/gender table was a teaching device.

**"Where do embeddings come from — are they downloaded?"** They start as random numbers and are learned during training by the same gradient descent as every other weight (Week 2). GloVe, used in the notebook's examples, is a *pretrained* set you can download, but a model trained from scratch learns its own.

**"Is the strawberry failure a reasoning problem?"** No — it's a perception problem. The model never sees letters, only tokens. This distinction generalises: before concluding a model can't reason about something, check whether it can *see* it.

**"Why add positional info instead of concatenating?"** Concatenating would increase dimensionality and therefore cost. Addition works because high-dimensional space has enough room for the model to separate the two signals during training.

**"If attention is permutation-invariant, isn't that a design flaw?"** It's a design *property* that buys parallelisation (Week 1 §12) — nothing depends on order, so everything computes at once. Positional encoding adds order back in as data rather than as a structural constraint. That trade is exactly why transformers train faster than RNNs.

---

## Key takeaways

1. **Subwords are the Goldilocks unit** — characters are too granular, words too numerous. 30k–100k subword tokens cover any text in any language and eliminate OOV entirely.
2. **BPE learns the vocabulary from data** by repeatedly merging the most frequent adjacent pair. Nobody designs the splits — which is why they ignore linguistics.
3. **Tokens are the unit of cost, context, and perception.** 128K means tokens; billing means tokens; and what the model cannot see, it cannot learn.
4. **JSON and numbers cost ~2× English prose per character** (2.3 vs 4.4 chars/token) — a real prompt-cost lever.
5. **The strawberry problem is a perception failure, not a reasoning failure.** The model sees `['str','aw','berry']` and never the letters.
6. **Token IDs are arbitrary row numbers; embeddings give them meaning** — dense vectors where proximity encodes similarity, learned rather than designed.
7. **Vector arithmetic works** (king − man + woman ≈ queen) but has real weaknesses, notably **antonym polarity**, because opposites share contexts.
8. **Raw embeddings are permutation-invariant** — "dog bites man" and "man bites dog" are indistinguishable without positional information.
9. **Sinusoidal positional encoding is added (not concatenated)**, needs no parameters, encodes relative distance linearly, and extrapolates past training length.
10. **Embeddings are still context-free** — one vector per token regardless of surroundings. That's the exact limitation attention removes in Week 4.

---

## Glossary

| Term | Meaning |
|---|---|
| **Token** | The atomic unit a model reads — typically a subword. |
| **Tokenizer** | The component mapping text ↔ token IDs. Determines what the model can perceive. |
| **BPE** | Byte Pair Encoding — builds a vocabulary by repeatedly merging the most frequent adjacent pair. |
| **`cl100k_base`** | GPT-4's tiktoken encoding; 100,277 tokens. |
| **OOV** | Out of vocabulary — a word with no entry. Solved by subword tokenization. |
| **Vocabulary explosion** | Word-level tokenizers needing 170k+ entries and enormous lookup tables. |
| **Token boundary problem** | Tasks needing reasoning *within* a token (letter counting, reversal) are hard because sub-token structure is invisible. |
| **Embedding matrix** | The `vocab_size × d_model` lookup table mapping IDs to vectors. |
| **`d_model`** | Embedding dimensionality — 768 for base-size models. |
| **Dense vector** | A short vector where every dimension carries information. |
| **Sparse / one-hot** | A long, mostly-zero vector; carries no similarity information. |
| **Cosine similarity** | Cosine of the angle between two vectors; −1 to 1, measures direction not length. |
| **Bag-of-words** | A representation ignoring word order. |
| **Permutation invariance** | Producing identical output regardless of input order — attention's default state. |
| **Positional encoding** | Position information added to embeddings; sinusoidal in the original transformer. |
| **RoPE** | Rotary Position Embedding — the modern replacement, covered in Week 6. |
| **PCA / t-SNE** | Dimensionality-reduction techniques for visualising high-dimensional embeddings in 2D/3D. |

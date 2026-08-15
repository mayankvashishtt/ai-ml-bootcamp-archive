# Week 3 — Quiz (20 questions)

**Topic:** Transformers Part 1 — Tokenization, Embeddings, Positional Encoding
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** Character-level tokenization's core weakness is that:
- A) The vocabulary is too large
- B) Extreme granularity forces many steps per concept and wastes capacity on basic vocabulary
- C) It cannot represent punctuation
- D) It produces out-of-vocabulary errors

**2.** Word-level tokenization suffers from all of the following EXCEPT:
- A) The "form" problem — "cat", "cats", "cat's" treated as unrelated
- B) Out-of-vocabulary holes for typos and rare terms
- C) Vocabulary explosion across languages
- D) Inability to process text in parallel

**3.** Roughly how many tokens does a subword vocabulary need to cover almost any text in any language?
- A) 1k–5k
- B) 30k–100k
- C) 170,000+
- D) 3 billion

**4.** The correct order of BPE's algorithm is:
- A) Start with words, split the least frequent, repeat
- B) Start with characters, merge the most frequent adjacent pair, repeat to target vocab size
- C) Start with a fixed dictionary and prune it
- D) Randomly sample substrings until coverage is reached

**5.** In `cl100k_base`, `"Hello world"` becomes `['Hello', ' world']`. This shows that:
- A) Spaces are discarded
- B) Spaces attach to the beginning of the following word
- C) Every word is exactly one token
- D) Capitalisation is ignored

**6.** Based on the measured table, which content type is LEAST token-efficient?
- A) English prose (4.4 chars/token)
- B) URLs (4.4 chars/token)
- C) Python code (3.5 chars/token)
- D) JSON (2.3 chars/token)

**7.** LLMs struggle to count the r's in "strawberry" because:
- A) The model's reasoning ability is too weak
- B) It sees `['str', 'aw', 'berry']` and cannot perceive individual letters
- C) The word is out of vocabulary
- D) Counting is not in the training data

**8.** An embedding matrix of 50,000 tokens × 768 dimensions contains how many parameters?
- A) 50,768
- B) 384,000
- C) 38,400,000
- D) 3,840,000,000

**9.** Which best describes the default state of raw embeddings before positional encoding?
- A) Order-aware
- B) Bag-of-words / permutation-invariant
- C) Context-dependent
- D) Sparse one-hot

**10.** Sinusoidal positional encoding is combined with token embeddings by:
- A) Concatenation, doubling the dimension
- B) Element-wise addition, preserving the dimension
- C) Matrix multiplication
- D) Replacing the embedding entirely

**11.** Which is NOT a stated advantage of sinusoidal positional encoding?
- A) Requires no learned parameters
- B) Position p+k is a linear function of position p
- C) Extrapolates to sequences longer than seen in training
- D) Guarantees the model attends to nearby tokens more strongly

**12.** In the notebook, the same word "the" at position 0 versus position 5 had a Euclidean distance of:
- A) 0.0
- B) 1.41
- C) 14.10
- D) 141.0

---

## Short answer

**13.** Describe the three tokenization approaches and the specific failure that motivated moving from each to the next.

**14.** Explain how subword tokenization eliminates the out-of-vocabulary problem entirely.

**15.** Your prompt costs more than expected. Using the measured chars/token figures, explain why and give one concrete mitigation.

**16.** Explain the token boundary problem and why it is a *perception* failure rather than a *reasoning* failure. Give both notebook examples.

**17.** Why are raw token IDs insufficient, and what specific property do embeddings add?

**18.** The analogy `bigger − big + cold` returned *cooler* (0.688), *warmer* (0.685), *colder* (0.675). What does this reveal about a limitation of static embeddings, and why does it happen?

**19.** State the permutation-invariance problem with a concrete example and explain how positional encoding solves it.

**20.** Give the four reasons the original Transformer chose sine and cosine waves for positional encoding.

---
---

## Answer key

**1. B** — Granularity: "understanding" takes 13 steps, the model must learn prefixes and roots from raw characters, and large amounts of data go into learning basic vocabulary rather than higher-level logic.

**2. D** — Parallel processing is unaffected by tokenizer choice; that is a property of the architecture. The other three are all genuine word-level failures.

**3. B** — 30k–100k tokens. For comparison, GPT-4's `cl100k_base` has 100,277.

**4. B** — Initialise with characters, iteratively merge the most frequent adjacent pair, repeat until the target vocabulary size is reached. Frequent patterns become single tokens; rare ones stay fragmented.

**5. B** — The space attaches to the front of the following word, so `' world'` is one token including its leading space.

**6. D** — JSON at 2.3 chars/token (tied with raw numbers), roughly half the efficiency of English prose, because structural punctuation tokenises poorly.

**7. B** — A perception limitation: the model receives three subword tokens, never the individual characters, so counting letters requires resolving structure it cannot see.

**8. C** — 50,000 × 768 = 38,400,000 parameters, before any processing layers.

**9. B** — Bag-of-words: the mathematical operations are permutation-invariant, so word order is invisible without added position information.

**10. B** — Element-wise addition. `embeddings + pe` keeps the shape at `(seq_len, d_model)`.

**11. D** — Nothing about sinusoidal encoding guarantees stronger attention to nearby tokens; attention weights are learned. The other three are explicitly claimed advantages.

**12. C** — 14.10, demonstrating that identical tokens receive genuinely different representations at different positions.

**13.** *Character-level* — one integer per character. Failed on granularity: many steps per concept, roots and prefixes must be learned from scratch, and data is wasted on basic vocabulary. → *Word-level* — one integer per word. Failed on the form problem ("cat"/"cats"/"cat's" unrelated), out-of-vocabulary holes for typos and rare terms, and vocabulary explosion (170k+ words, ~3B lookup parameters, multiplied per language). → *Subword* — the Goldilocks middle ground: common words stay whole, rare words split into meaningful pieces, and 30k–100k tokens cover essentially any text in any language.

**14.** Because the vocabulary contains subword pieces down to individual characters, **any** input string can be decomposed into units the tokenizer already knows. A never-before-seen word such as a new product name or a typo does not produce an "unknown" symbol; it fragments into known pieces (e.g. `"rishabh"` → `[' r','ish','abh']`). In the worst case a string falls back to character-level tokens. There is therefore no input the tokenizer cannot represent, so the OOV category disappears by construction.

**15.** Cost is billed per **token**, and token density varies sharply by content type: English prose runs ~4.4 chars/token while JSON and raw numbers run ~2.3 — roughly **half the efficiency**, so the same character count costs nearly twice as much. If a prompt embeds structured data, its punctuation (`{`, `"`, `:`, `,`) is consuming the budget. *Mitigations:* switch to a more compact serialisation for prompt payloads (e.g. CSV or a compact YAML/plain-text layout instead of pretty-printed JSON), strip indentation and unnecessary whitespace, or send only the fields the model actually needs.

**16.** The **token boundary problem** is that tasks requiring reasoning *inside* a token are hard, because the model never receives sub-token structure. *Example 1:* `"strawberry"` → `['str', 'aw', 'berry']` — counting r's requires looking inside tokens. *Example 2:* `"hello"` is a **single** token — reversing it means decomposing something the model sees as atomic. It is a **perception** failure because the information is absent from the input representation, not because the model reasons badly: it is being asked about structure it cannot see, analogous to asking someone to count brushstrokes from a low-resolution photograph. This is why such failures persist even in models that handle far harder reasoning tasks correctly.

**17.** Token IDs are **arbitrary labels**: 15496 does not "know" it represents a greeting, and numerically adjacent IDs bear no relation in meaning, so no useful arithmetic can be performed on them. Embeddings replace each ID with a **dense continuous vector** positioned in a high-dimensional semantic space, where **proximity encodes similarity** and consistent directional offsets encode relations. This makes meaning mathematically manipulable — similarity becomes distance, and analogy becomes vector addition.

**18.** It reveals that static embeddings capture the **comparative-form direction** correctly (all three results are comparatives) but are unreliable on **antonym polarity** — "warmer" scores essentially as highly as "colder," despite being the opposite of the intended answer. This happens because embeddings are learned from **distributional context**, and antonyms appear in nearly identical contexts: "the weather is warmer/colder today" are equally plausible sentences. Since the training signal is which words appear in similar surroundings, opposites end up close together in the space. It is a structural limitation of context-free embeddings, and part of the motivation for context-dependent representations from attention.

**19.** **Permutation invariance** means the model's operations produce the same result regardless of input order, so it sees a *set* of words rather than a *sequence*. Concretely, `"dog bites man"` → `[18964, 49433, 893]` and `"man bites dog"` → `[1543, 49433, 5679]` contain the same tokens reordered; without position information the model cannot tell who bit whom, even though order is the primary determinant of meaning. **Positional encoding solves it** by adding a position-specific vector to each token embedding, so a token's final representation encodes both *what* it is and *where* it sits. The notebook verifies this: "the" at position 0 versus position 5 differs by a Euclidean distance of 14.10.

**20.** (i) **Relative position** — for any fixed offset *k*, position *p+k* can be expressed as a linear function of position *p*, letting the model easily learn to attend by relative distance. (ii) **Generalisation** — the waves are continuous, so the encoding extrapolates to sequences longer than any seen during training. (iii) **No learned parameters** — the patterns are fixed and pre-computed, costing nothing to train and nothing in model size. (iv) **Unique multi-scale patterns** — combining many frequencies gives every position a unique fingerprint, with low dimensions varying slowly and high dimensions varying quickly, so both coarse and fine positional information are available.

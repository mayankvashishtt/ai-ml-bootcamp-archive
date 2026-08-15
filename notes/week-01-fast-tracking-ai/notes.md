# Week 1 — Fast-Tracking the Course of AI

**Subtitle:** From "What is AI?" to "How ChatGPT works"
**Date:** 18/01/2026
**Source:** `downloads/week-01-fast-tracking-ai.pdf` (48 slides)
**Notion page:** https://100xschool.notion.site/2ebffffa33e5800b8afec5ea13110d3d

**Referenced links:**
- [bbycroft.net/llm](https://bbycroft.net/llm) — 3D interactive walkthrough of a GPT's internals. Genuinely worth an hour.
- [github.com/elder-plinius/L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) — jailbreak prompt collection, shown as evidence that "safety" is a behavioural layer, not a property of the weights.

---

## The shape of this lecture

This is a **compressed history of the whole field**, told as a chain of failures. Each approach solves the previous one's problem and reveals a new one. That chain is the thing to remember — it explains *why* transformers look the way they do, and it's the scaffold every later week hangs off.

```
Rules → can't scale to intuition
  └→ Machine Learning → not enough data/compute (AI winters)
       └→ Data + GPUs + Deep Learning (2012) → language still hard
            └→ Dictionaries → no context
                 └→ Statistics → pattern ≠ meaning
                      └→ Embeddings → one vector per word, context-blind
                           └→ RNNs → forgetting over long ranges
                                └→ Attention / Transformers → today
```

---

## 1. Groundwork: learning, knowledge, intelligence

The lecture opens philosophically, and these definitions get reused all course.

**Learning** = changing yourself based on experience. Concretely, a loop:

> **Try something → see what happens (feedback) → adjust → repeat**

Hold onto this. It is *exactly* the training loop in Week 2 (guess → measure error → adjust → repeat) and *exactly* the RL loop in Week 15. The course keeps returning to this one shape.

**Knowledge** comes in two kinds:
- **Explicit** — statable in words. "Paris is the capital of France." Writable as a rule.
- **Implicit** — knowable but not statable. Riding a bike, recognising a face, hearing sarcasm.

This distinction is the **entire reason rule-based AI failed**. Rules can only encode explicit knowledge, but most of what humans do fluently is implicit.

**Intelligence** = *the ability to achieve goals in a wide range of situations.* Note the emphasis on **range** — a chess engine achieves one goal superbly and is not, by this definition, intelligent. Generality is the criterion, which is why "foundation models" later matter so much.

**AI** = making machines do things that would require intelligence if a human did them. Examples: recognising faces, understanding language, playing chess, driving.

---

## 2. Attempt #1 — Write the rules (Expert Systems)

Encode human expertise as explicit `if/then` logic:

```
IF email contains "free money"     → spam
IF temperature > 100°F             → fever
IF piece can move to square AND square has enemy → capture
```

**Why it breaks:** it can only capture explicit knowledge. The lecture's three examples:

| Task | Why rules fail |
|---|---|
| **Recognising a face** | Define the exact eye distance and nose curve for *every* angle and lighting condition. You can't enumerate it. |
| **Understanding sarcasm** | "Oh, great." Praise or complaint? Depends on tone and context, which rules don't see. |
| **Decoding idioms** | "It's raining cats and dogs" — a literal rule engine starts looking for falling animals. |

Two failure modes worth naming: **rules don't scale** (you can't write one for every scenario) and **rigid logic** (anything outside the predefined set breaks the system).

---

## 3. Attempt #2 — Let machines learn

Invert the problem. Instead of writing rules, **show thousands of examples and let the machine find the pattern.**

The intuitive loop:
1. Show many examples
2. Machine guesses
3. Tell it right or wrong
4. Machine adjusts slightly
5. Repeat millions of times

Analogy: *a child learning by trial and error.*

**Terminology to keep straight** — these are constantly conflated in the wild:

- **AI** = the *goal* (smart machines)
- **ML** = one *method* for reaching it (learning from data)
- **Deep Learning** = one *kind* of ML (many-layered neural networks)

Strictly nested: Deep Learning ⊂ ML ⊂ AI.

---

## 4. Early ML and the AI winters

**What worked:** spam filters, recommendation systems (Netflix, Amazon), basic image recognition.

**What didn't:** couldn't hold a conversation, couldn't understand a paragraph, couldn't recognise objects as well as a toddler.

**Two bottlenecks:**

1. **Not enough data.** ML needs massive example sets. The internet was in its infancy — where would millions of labelled examples come from?
2. **Not enough compute.** Training is mathematically intense. Machines were too slow; training on millions of examples would have taken years.

Note that *neither bottleneck was algorithmic*. Many core ideas already existed — they were starved of inputs. This is the lecture's implicit argument for why scale later worked so dramatically.

**The two AI winters:**

| Period | Cause |
|---|---|
| **1970s** | Researchers promised human-level AI within 20 years. They didn't deliver; funding and interest collapsed. |
| **1980s–90s** | Expert systems were supposed to transform business but proved too brittle for the real world. Funding vanished again. |

> "The field became a joke. Saying you worked on 'AI' was career suicide."

Useful historical calibration: the field has twice been left for dead by people who were, at the time, being reasonable.

---

## 5. Why AI exploded after 2012

Three ingredients converged:

1. **Data — the internet.** Massive text, image, and video datasets available for the first time.
2. **Compute — GPUs.** Graphics cards built for gaming turned out to be near-perfect for the parallel matrix maths ML needs. (An accident of history worth appreciating: the hardware substrate of the AI boom was funded by video games.)
3. **Algorithms — deep learning.** Researchers finally worked out how to train much deeper networks.

**All three had to arrive together.** Any one alone would have gone nowhere — which is precisely why the earlier ideas failed despite being roughly correct.

---

## 6. Why language specifically is hard

The lecture narrows to language as its lens. The core problem is **ambiguity**:

> "I saw the man with the telescope."

Who has the telescope? Both readings are grammatically valid. Computers need precision; human language is messy, ambiguous, and context-dependent.

A second example, closer to home: *"Fast-tracking the AI course"* vs *"Fast-tracking the course of AI"* — near-identical words, entirely different meanings. (This is also the lecture title's own joke.)

---

## 7. NLP Attempt #1 — Dictionaries

Give the machine definitions. **Apple**: *the round fruit of a tree of the rose family…*

**Fails because** a definition isn't meaning-in-context. "Apple" is also a trillion-dollar company and a famous record label. **Context changes everything**, and a dictionary has no notion of context.

---

## 8. NLP Attempt #2 — Statistical patterns

Learn from frequency: *if "New" is followed by "York" → city.* This powered Google Translate for nearly a decade — so it genuinely worked, commercially, at scale.

**Fails because pattern matching ≠ understanding.** The machine predicts the next word by frequency with no concept of what words *mean*.

> Worth flagging: modern LLMs are also, mechanically, next-word predictors. The lecture's later claim (§13) is that at sufficient scale this stops being "mere" pattern matching. Whether that's a difference of kind or only of degree is the open question sitting under this whole slide deck — hold the tension rather than resolving it too early.

---

## 9. The breakthrough — words as numbers

Computers understand numbers, not words. So: turn each word into a **vector** — a list of numbers — where each dimension encodes some aspect of meaning.

```
"Apple" → [0.92, -0.14, 0.05, ...]
```

Toy illustration with interpretable dimensions:

| Word | Royalty | Gender (M) | Edibility |
|---|---|---|---|
| King | 0.98 | 0.95 | 0.01 |
| Queen | 0.97 | 0.05 | 0.02 |
| Apple | 0.02 | 0.00 | 0.94 |

King and Queen share royalty, differ sharply on gender. That's concept-as-data.

⚠️ **Caveat the lecture is honest about:** real embedding dimensions are *not* interpretable like this. Nobody labelled dimension 4 "royalty" — the model learns whatever axes are useful, and they rarely map to human concepts. The table is a teaching device.

### Words as positions in space

If a word is a list of numbers, it's a **point on a map**. Similar words cluster; different words sit far apart. `king`/`queen` near each other, `man`/`woman` near each other, `cat`/`dog` together, `car`/`bicycle` together, and "Apple (Corp)" somewhere quite different from the fruit.

### Word math

The famous demonstration that geometry captures relationships:

```
King   − Man    + Woman  = Queen
Paris  − France + Italy  = Rome
Walking − Walk  + Swim   = Swimming
```

Relationships (gender, capital-of, verb tense) turn out to be **consistent directions** in the space. Nobody designed this — it falls out of training.

---

## 10. Where embeddings break — one word, one vector

```
"I went to the bank to deposit cash"
"I sat on the bank of the river"
```

In classic word embeddings each word has **exactly one** position in space. So "bank" gets a single vector that can't be both financial and geographic. The model can't disambiguate because **it doesn't look at surrounding words.**

This is the specific gap the rest of the lecture closes. The fix — *context-dependent* representations — is precisely what attention delivers.

---

## 11. NLP Attempt #4 — Sequence models (RNNs)

Process the sentence one word at a time, accumulating understanding:

```
"I" → "I went" → "I went to" → "I went to the" → "I went to the bank..."
```

The model keeps a **memory** (hidden state) of what it has read. This finally brings in context — real progress.

**Fails because of forgetting:**

> "The cat, which was sitting on the mat that I bought from the store near the old church on the corner, was happy."

By the time the model reaches "was," it has usually lost the subject ("cat") from the start. RNNs process **linearly** and have **limited memory capacity** — the long-range dependency problem.

---

## 12. The Transformer idea (2017)

**What was in place by 2017:** word embeddings, sequence models, huge internet data, powerful GPUs.
**What blocked progress:** the *word-by-word bottleneck* — sequential processing is slow, and models still forget the start of long sentences.

**The change:**

| Old way — Sequential | New way — Simultaneous |
|---|---|
| Process words one by one, like reading a book front to back | Look at **every** word in the sentence at once |

**The mechanism: attention.** The model "attends" to the most relevant words regardless of distance.

### Attention in action

> "The animal didn't cross the street because **it** was too **tired**."

Processing "it," attention weights point most strongly at **"animal."** That's how the model resolves the pronoun.

Now change one word:

> "The animal didn't cross the street because **it** was too **wide**."

Attention shifts to **"street."** Same structure, one word different, and the representation of "it" changes accordingly. **This is the fix for the "bank" problem in §10** — a word's representation is now computed *from its context* rather than looked up in a fixed table.

### Why attention is powerful — three properties

1. **No forgetting.** Every word sees every other word simultaneously, at any distance. Kills the RNN long-range problem.
2. **Speed.** All words processed in parallel rather than one at a time — training becomes vastly faster, which is what makes scaling affordable.
3. **Context.** Builds a mathematical map of how every word relates to every other, *in this specific sentence*.

Property 2 deserves emphasis: the reason transformers won isn't only that they're better, it's that they're **parallelisable on GPUs**. Quality plus trainability-at-scale.

**The Transformer** is an architecture built *entirely* on attention: no recurrence, no convolution. It's the foundation of every modern LLM and it enables massive scaling of data and compute.

---

## 13. How ChatGPT works (simplified)

**Transformer + Internet Data + Prediction**

1. **Architecture** — a massive neural network built on attention.
2. **Training data** — trillions of words from books, articles, the public internet.
3. **Objective** — given a sequence of words, **predict the most likely next word.** That's it.

### Why "predict the next word" produces understanding

The argument is that the task is *incidentally* hard enough to force real capability:

- **Grammar & syntax** — you can't predict well without internalising language structure.
- **Factual knowledge** — predicting "The capital of France is ___" correctly requires knowing the fact.
- **Logic & reasoning** — following a chain of argument is required to anticipate its continuation.

The objective is simple; satisfying it well is not.

### Generation is a loop

```
Step 1: The
Step 2: The quick
Step 3: The quick brown
Step 4: The quick brown fox...
```

Each predicted word is **appended to the input**, and the model predicts again from the updated context. This is *autoregressive* generation — the same loop behind every chatbot you've used, and the reason latency scales with output length.

---

## 14. Scaling and emergence

The unexpected discovery: **bigger = smarter**, along three axes simultaneously.

1. **Data** — millions → trillions of words
2. **Parameters** — millions → hundreds of billions of connections
3. **Compute** — days → months on thousands of GPUs

**The result: emergent capabilities.** Reasoning, coding, and logic appear *spontaneously* at scale — they weren't designed in, and they weren't present in smaller models. Nobody wrote a reasoning module.

---

## 15. Foundation models — the paradigm shift

| Old paradigm | New paradigm |
|---|---|
| Task-specific AI: Model A translates, Model B summarises, Model C does sentiment | One massive general-purpose model, steered to any task by **prompting** |

The "foundation" is base knowledge you build on rather than rebuild. This is the economic hinge of the whole field: **you no longer build tools from scratch, you build on top of these giants** — which is exactly what "AI Engineering" (Week 0) means, and why the course spends Weeks 8–25 on building *with* models rather than training them.

---

## 16. Where we are (2024–2025)

1. **Multimodality** — not just text: sees images, hears voices, speaks back in real time.
2. **Reasoning** — models designed to "think" before answering, solving harder maths and logic. (Picked up in Week 15, RLVR.)
3. **Agents** — the shift from chatbots to systems that use tools, browse, and complete multi-step tasks. (Weeks 8, 21, 22, 25.)

These three trends map directly onto the back half of the syllabus.

---

## Key takeaways

- The field advances by **hitting a wall and reformulating**, not by steady accumulation. Learn the chain of failures; it predicts what breaks next.
- **Implicit vs explicit knowledge** is the root cause of rule-based AI's failure.
- **Data + compute + algorithms** all had to converge (~2012). Missing one ingredient is why earlier attempts stalled.
- **Embeddings** turned meaning into geometry; relationships became directions in space.
- Static embeddings' fatal flaw — **one vector per word** — is what attention fixes by making representations context-dependent.
- Attention won on **three** counts: no forgetting, parallel speed, and rich context. The speed is what made scale affordable.
- **Next-word prediction at scale** yields grammar, facts, and reasoning as side effects — and reasoning **emerges** rather than being designed.
- **Foundation models** shifted the industry from building models to building *on* models.

---

## Glossary

| Term | Meaning |
|---|---|
| **Expert System** | Rule-based AI encoding human expertise as if/then logic. |
| **Explicit knowledge** | Knowledge that can be stated in words and written as rules. |
| **Implicit knowledge** | Knowledge you have but can't articulate — faces, bikes, sarcasm. |
| **AI winter** | A period of collapsed funding and interest after overpromising. Two: 1970s, 1980s–90s. |
| **Embedding** | A vector of numbers representing a word, where geometric position encodes meaning. |
| **Word math** | That vector arithmetic on embeddings mirrors semantic relationships (King − Man + Woman = Queen). |
| **RNN** | Recurrent Neural Network — processes sequences one step at a time carrying a hidden memory. |
| **Long-range dependency** | Relating information across a long span; the failure mode of RNNs. |
| **Attention** | Mechanism letting every token weigh every other token directly, regardless of distance. |
| **Transformer** | Architecture built entirely on attention — no recurrence, no convolution. |
| **Autoregressive generation** | Generating one token at a time, feeding each output back as input. |
| **Emergent capability** | An ability that appears at scale without being explicitly trained for. |
| **Foundation model** | A large general-purpose pretrained model adapted to many tasks via prompting. |
| **Multimodality** | Handling multiple input/output types — text, image, audio. |

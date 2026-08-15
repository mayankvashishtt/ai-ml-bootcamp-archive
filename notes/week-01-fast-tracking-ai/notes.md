# Week 1 — Fast-Tracking the Course of AI

**Subtitle:** From "What is AI?" to "How ChatGPT works"
**Date:** 18/01/2026
**Source:** `downloads/week-01-fast-tracking-ai.pdf` (48 slides)
**Notion page:** https://100xschool.notion.site/2ebffffa33e5800b8afec5ea13110d3d

**Referenced links:**
- [bbycroft.net/llm](https://bbycroft.net/llm) — 3D interactive walkthrough of a GPT's internals. Genuinely worth an hour, and best revisited after Week 4.
- [github.com/elder-plinius/L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) — jailbreak prompt collection, shown as evidence that "safety" is a behavioural layer, not a property of the weights. (Picked up properly in S4.)

---

## 0. The idea in plain language

This lecture is a **compressed history of the entire field**, and it's told as a chain of failures. That framing is the point, so don't skim it: each approach solves the previous one's problem and, in doing so, exposes a new one. The chain is what explains *why transformers look the way they do* — otherwise the architecture in Weeks 3–4 looks like an arbitrary pile of components someone happened to bolt together.

Here's the whole lecture in one diagram. Everything below is an expansion of it:

```
Write the rules by hand
   └─ fails: most human skill can't be written as rules
      └─ Let machines learn from examples
           └─ fails: not enough data, not enough compute (the "AI winters")
              └─ Internet + GPUs + deep learning (~2012)
                   └─ but language specifically is still hard
                      └─ Dictionaries → no sense of context
                         └─ Word statistics → patterns without meaning
                            └─ Embeddings → meaning as geometry, but ONE vector per word
                               └─ RNNs → context at last, but they forget
                                  └─ Attention / Transformers → today
```

**If you remember one thing:** every step is a response to a specific, nameable failure. When you meet a design choice later that seems weird, the question to ask is "what failure was this fixing?"

---

## 1. Groundwork: learning, knowledge, intelligence

The lecture opens philosophically. These definitions get reused all course, so they're worth taking seriously rather than skipping to the technical part.

### Learning

**Learning = changing yourself based on experience.** Concretely, a loop:

> **Try something → see what happens (feedback) → adjust → repeat**

Hold onto this shape. It is *exactly* the training loop in Week 2 (guess → measure error → adjust → repeat), *exactly* the agent loop in Week 8 (act → observe → decide again), and *exactly* the reinforcement learning loop in Weeks 14–15. The course returns to this one pattern at every level of the stack. When something new is introduced later, it's worth asking "where's the loop?" — you'll almost always find it.

### Knowledge comes in two kinds

**Explicit knowledge** can be stated in words. "Paris is the capital of France." "A prime number has exactly two divisors." You can write it down, put it in a database, encode it as a rule.

**Implicit knowledge** you have but cannot articulate. Riding a bicycle — you can do it, but the instructions you'd write would not enable anyone to do it. Recognising a friend's face in a crowd. Hearing sarcasm. Knowing that "the big red barn" sounds right and "the red big barn" sounds wrong, without knowing the adjective-ordering rule that governs it.

**This distinction is the entire reason rule-based AI failed**, and it's worth sitting with. Rules can only ever encode explicit knowledge. But most of what humans do fluently — perception, language, judgement — is implicit. You are trying to write down something that, by definition, cannot be written down.

### Intelligence

**Intelligence = the ability to achieve goals in a wide range of situations.**

Note the emphasis on **range**. A chess engine achieves one goal superbly and is not, by this definition, intelligent — it cannot do anything else at all. Generality is the criterion. This is why "foundation models" (§15) matter so much later: the shift was from many narrow systems to one general one.

### AI

**AI = making machines do things that would require intelligence if a human did them.** Recognising faces, understanding language, playing chess, driving a car.

Notice this definition is about the *task*, not the *method*. A hand-written rule engine that diagnoses disease is AI. So is a neural network that does the same. This is why §5's vocabulary matters — AI is the goal, not a technique.

---

## 2. Attempt #1 — Write the rules (Expert Systems)

The first serious approach, dominant into the 1980s: encode human expertise as explicit `if/then` logic.

```
IF email contains "free money"                      → spam
IF temperature > 100°F                              → fever
IF piece can move to square AND square has enemy    → capture
```

This is not a silly idea. It works well when the domain genuinely *is* explicit — tax calculation, chess legality checking, basic medical triage. Expert systems made real money in the 1980s.

### Why it breaks

It can only capture explicit knowledge, and the interesting problems are implicit. The lecture's three examples:

| Task | Why rules fail |
|---|---|
| **Recognising a face** | You would need to define exact eye distance and nose curvature for *every* angle, lighting condition, expression, age, and occlusion. The list doesn't terminate. |
| **Understanding sarcasm** | "Oh, great." Praise or complaint? It depends on tone, context, and what just happened — none of which the rule can see. |
| **Decoding idioms** | "It's raining cats and dogs." A literal rule engine starts looking for falling animals. |

Two failure modes worth naming separately, because they're different problems:

**Rules don't scale.** You cannot write one for every scenario. Every rule you add interacts with every existing rule, so the system gets harder to maintain superlinearly. Large expert systems became unmaintainable — nobody could predict what adding rule 4,001 would break.

**Rigid logic.** Anything outside the predefined set doesn't degrade gracefully; it breaks. A human seeing a situation they've never encountered improvises. A rule engine returns nothing, or worse, matches the wrong rule confidently.

---

## 3. Attempt #2 — Let machines learn

Invert the problem entirely. **Instead of writing the rules, show thousands of examples and let the machine find the pattern itself.**

The loop, intuitively:

1. Show it many examples
2. The machine guesses
3. Tell it whether it was right or wrong
4. The machine adjusts itself slightly
5. Repeat, millions of times

The analogy is a child learning by trial and error. Nobody teaches a child the adjective-ordering rule; they hear enough English that the wrong order starts sounding wrong.

**Why this escapes the trap of §2:** you never have to *articulate* the pattern. The machine finds a pattern that works without anyone being able to state it. Implicit knowledge becomes learnable, because the requirement to write it down has been removed.

**What you give up:** you can no longer inspect the reasoning. An expert system can tell you "I said spam because rule 12 matched." A learned model has a pattern spread across millions of numbers with no human-readable explanation. This trade — capability for interpretability — is permanent, and it's why Week 24's observability lecture exists at all.

> **Terminology check** (constantly conflated in the wild): **AI** is the *goal*, **ML** is one *method* for reaching it, **Deep Learning** is one *kind* of ML using many-layered neural networks. Strictly nested: Deep Learning ⊂ ML ⊂ AI. See Week 0 §5 for the full breakdown.

---

## 4. Early ML and the AI winters

**What worked:** spam filters, recommendation systems (Netflix, Amazon), basic image recognition.

**What didn't:** couldn't hold a conversation, couldn't understand a paragraph, couldn't recognise objects as reliably as a toddler.

### Two bottlenecks

**Not enough data.** ML needs massive example sets. In the 1990s the internet was in its infancy — where would you obtain millions of labelled examples? There was no ImageNet, no Common Crawl, no Wikipedia.

**Not enough compute.** Training is arithmetically intense — billions of multiply-and-add operations, repeated millions of times. Machines were too slow. Training on millions of examples would have taken years of wall-clock time.

**The crucial observation: neither bottleneck was algorithmic.** Many of the core ideas — neural networks, backpropagation, even convolutional networks — already existed by the late 1980s. They were starved of inputs, not of insight. This is the lecture's implicit argument for why scale later worked so dramatically: the algorithms had been waiting.

### The two AI winters

| Period | Cause |
|---|---|
| **1970s** | Researchers promised human-level AI within 20 years. They didn't deliver; funding and public interest collapsed. |
| **1980s–90s** | Expert systems were supposed to transform business but proved too brittle for real-world messiness. Funding vanished again. |

> "The field became a joke. Saying you worked on 'AI' was career suicide."

**Why this history is worth knowing rather than just amusing:** the field has twice been left for dead by people who were, at the time, being entirely reasonable. Both winters followed overpromising, not technical failure — the technology underdelivered against claims, not against physics. It's a useful calibration in both directions when you read confident predictions today, in either direction.

---

## 5. Why AI exploded after 2012

Three ingredients converged, and the convergence is the whole story:

**1. Data — the internet.** Massive text, image, and video datasets became available for the first time. ImageNet (2009) gave the vision community millions of labelled images. The web gave the language community effectively unlimited text.

**2. Compute — GPUs.** Graphics cards, built to render video game polygons, turned out to be near-perfect for the parallel matrix arithmetic that neural networks need. A GPU does thousands of simple operations simultaneously, which is exactly the shape of the problem. *(An accident of history worth appreciating: the hardware substrate of the AI boom was funded by video games.)*

**3. Algorithms — deep learning.** Researchers finally worked out how to reliably train much deeper networks — better initialisation, better activation functions, better regularisation.

**All three had to arrive together.** Any one alone would have gone nowhere, which is precisely why the earlier attempts stalled despite the ideas being roughly correct. This is also the cleanest illustration of the lecture's thesis: progress came from removing a *constraint*, not from a new insight.

---

## 6. Why language specifically is hard

The lecture now narrows to language as its lens. The core difficulty is **ambiguity**.

> "I saw the man with the telescope."

Who has the telescope? Did you use a telescope to see him, or did you see a man who was carrying one? Both readings are perfectly grammatical. A human resolves it from context — or asks. Computers need precision; human language is messy, ambiguous, and context-dependent by design.

A second example, closer to home: *"Fast-tracking the AI course"* versus *"Fast-tracking the course of AI."* Near-identical words, entirely different meanings. (That's the lecture title's own joke.)

**Why this matters for everything that follows:** every NLP attempt from here on is judged by one question — *does it handle context?* Dictionaries don't. Statistics don't. Static embeddings don't. RNNs partially do. Attention does. That's the through-line.

---

## 7. NLP Attempt #1 — Dictionaries

Give the machine definitions. **Apple**: *the round fruit of a tree of the rose family…*

**Why it fails:** a definition is not meaning-in-context. "Apple" is also a trillion-dollar company and a famous record label. Which one is meant depends entirely on the surrounding words, and a dictionary has no notion of surroundings — it maps a word to a fixed definition regardless of where it appears.

This is the first appearance of the problem that the rest of the lecture chases: **the same word means different things in different places.**

---

## 8. NLP Attempt #2 — Statistical patterns

Learn from frequency instead of definitions. If "New" is very often followed by "York," the machine learns that association. Scale that to billions of word pairs and triples and you can do a surprising amount.

**This genuinely worked.** It powered Google Translate for nearly a decade — commercially, at scale. It's not a strawman.

**Why it fails: pattern matching is not understanding.** The machine predicts the next word by frequency, with no concept of what any word *means*. It can produce fluent-sounding text about a topic while having no representation of the topic at all. It also can't generalise to phrasings it hasn't seen, because it only knows co-occurrence counts.

> **Worth flagging honestly:** modern LLMs are also, mechanically, next-word predictors. The lecture's later claim (§13) is that at sufficient scale this stops being "mere" pattern matching. Whether that's a difference of *kind* or only of *degree* is the genuinely open question sitting under this whole deck. **Hold the tension rather than resolving it early** — S13 revisits it with actual experimental evidence, and the answer is more interesting than either slogan.

---

## 9. The breakthrough — words as numbers

Computers process numbers, not words. So the move is: **turn each word into a vector** — an ordered list of numbers — where the position in that numeric space encodes something about meaning.

```
"Apple" → [0.92, -0.14, 0.05, 0.71, ...]
```

### What a vector actually is, if that word is new

A **vector** here is just a list of numbers of fixed length. If the length is 2, you can plot it as a point on a graph — `[3, 4]` is 3 across and 4 up. If the length is 3, it's a point in 3D space. Real embeddings have lengths like 768 or 1536, which you can't visualise, but the mathematics is identical: **each word is a point in a very high-dimensional space**, and points that are close together mean similar things.

That's the entire idea. "Meaning" becomes "location."

### A toy illustration

Imagine three dimensions with human-readable labels:

| Word | Royalty | Gender (M) | Edibility |
|---|---|---|---|
| King | 0.98 | 0.95 | 0.01 |
| Queen | 0.97 | 0.05 | 0.02 |
| Apple | 0.02 | 0.00 | 0.94 |

King and Queen share royalty and differ sharply on gender. Apple is unrelated to both and scores high on edibility. That's a concept represented as data.

⚠️ **A caveat the lecture is honest about, and you should be too:** real embedding dimensions are **not** interpretable like this. Nobody labelled dimension 4 "royalty." The model learns whatever axes are useful for its objective, and they almost never correspond to human concepts. The table above is a teaching device, not a description of what's inside a model. (S6 covers what's actually in there.)

### Word math — the famous demonstration

If meaning is geometry, then *relationships* should be **directions**. They are:

```
King    − Man     + Woman  ≈ Queen
Paris   − France  + Italy  ≈ Rome
Walking − Walk    + Swim   ≈ Swimming
```

Read the first one as: "start at King, subtract the male-ness, add female-ness, and you land near Queen." The "gender direction" turns out to be roughly the same vector everywhere in the space. So does "capital-of." So does "verb tense."

**Nobody designed this.** It falls out of training on ordinary text. That's what made it startling in 2013 and it's the first genuine evidence in this lecture that a learned representation captures structure rather than just co-occurrence.

---

## 10. Where embeddings break — one word, one vector

```
"I went to the bank to deposit cash"
"I sat on the bank of the river"
```

In classic word embeddings, each word has **exactly one** position in space. So "bank" gets a single vector that must somehow be both financial and geographic — in practice it lands somewhere in between, useless for both.

The model cannot disambiguate, because **it never looks at the surrounding words.** The embedding is a lookup: word in, fixed vector out, context irrelevant.

**This is the specific gap the rest of the lecture closes**, and it's worth holding clearly in mind. The fix — representations computed *from context* rather than looked up — is precisely what attention delivers in §12, and it's what Week 4 implements in code.

---

## 11. NLP Attempt #4 — Sequence models (RNNs)

Process the sentence one word at a time, accumulating understanding as you go:

```
"I" → "I went" → "I went to" → "I went to the" → "I went to the bank..."
```

The model maintains a **hidden state** — a memory vector updated at each word, representing "everything I've read so far." This finally brings context into the picture, and it was real progress: RNNs powered translation, speech recognition, and text generation for years.

### Why it fails — forgetting

> "The cat, which was sitting on the mat that I bought from the store near the old church on the corner, was happy."

By the time the model reaches "was," it has usually lost "cat" — the subject it needs in order to conjugate correctly. The memory vector is fixed-size, so every new word overwrites some of what was there. Information from far back gets diluted away.

Two structural problems, and both matter:

**Limited memory capacity.** A fixed-size hidden state must compress an arbitrarily long history. Something has to be discarded, and the model has no way to know in advance what will matter later.

**Linear processing.** Words must be handled strictly in order, because word 5's hidden state depends on word 4's. This is the **long-range dependency problem**, and — critically — it also means **you cannot parallelise training.** Remember this second point; §12 shows it's arguably the more important of the two.

---

## 12. The Transformer idea (2017)

**What was in place by 2017:** word embeddings, sequence models, huge internet datasets, powerful GPUs.
**What was blocking progress:** the word-by-word bottleneck — sequential processing is slow, and models still forget the beginning of long inputs.

### The change

| Old way — Sequential | New way — Simultaneous |
|---|---|
| Process words one at a time, like reading a book front to back | Look at **every** word in the sentence at once |

**The mechanism is attention:** the model "attends" to the most relevant words regardless of how far away they are. Distance stops mattering.

### Attention in action — the example to remember

> "The animal didn't cross the street because **it** was too **tired**."

When processing "it," attention weights point most strongly at **"animal."** That's how the pronoun gets resolved.

Now change exactly one word:

> "The animal didn't cross the street because **it** was too **wide**."

Attention now shifts to **"street."** Same sentence structure, one word different, and the internal representation of "it" changes accordingly.

**This is the fix for the "bank" problem in §10.** A word's representation is now *computed from its context* rather than looked up in a fixed table. The same word in two different sentences gets two different vectors. That's the whole breakthrough, stated in one sentence.

### Why attention is powerful — three properties

**1. No forgetting.** Every word can see every other word directly, at any distance. Word 200 attends to word 1 as easily as to word 199. This kills the RNN long-range problem outright.

**2. Speed — and this is the underrated one.** Because there's no sequential dependency, all words are processed **in parallel**. Training becomes dramatically faster on GPUs, which is what made scaling to trillions of tokens economically possible.

**3. Context.** It builds a mathematical map of how every word relates to every other, *in this specific sentence*.

**Property 2 deserves emphasis because it's usually undersold.** Transformers didn't win only by being better at language — they won by being **trainable at scale on GPUs**. An architecture that's slightly better but can't be parallelised loses to one that's slightly worse and can, because the second one gets trained on a hundred times more data. Quality *plus* trainability. Keep this in mind for Week 6, where nearly every "improvement since 2017" turns out to be an efficiency improvement rather than a capability one.

**The Transformer** is an architecture built entirely on attention — no recurrence, no convolution. It's the foundation of every modern LLM, and Weeks 3–4 build one from scratch.

---

## 13. How ChatGPT works (simplified)

**Transformer + Internet Data + Next-Word Prediction.**

1. **Architecture** — a very large neural network built on attention.
2. **Training data** — trillions of words from books, articles, and the public internet.
3. **Objective** — given a sequence of words, **predict the most likely next word.** That is genuinely the whole training objective.

### Why "predict the next word" produces apparent understanding

The argument is that the task is *incidentally* hard enough to force real capability. To predict well, you are forced to learn:

- **Grammar and syntax** — you cannot predict the next word reliably without internalising sentence structure.
- **Factual knowledge** — predicting the last word of "The capital of France is ___" correctly requires knowing the fact.
- **Logic and reasoning** — to anticipate the continuation of an argument, you must be able to follow it.

**The objective is simple; satisfying it well is not.** That gap is the lecture's central claim, and it's why a task that sounds trivial produced systems that don't feel trivial.

### Generation is a loop

```
Step 1: The
Step 2: The quick
Step 3: The quick brown
Step 4: The quick brown fox ...
```

Each predicted word is **appended to the input**, and the model predicts again from the now-longer context. This is **autoregressive generation**.

Three consequences that come up repeatedly later, so note them now:

- **Latency scales with output length.** Every token requires a full forward pass through the network. This is why long responses are slow, and it's the foundation of S5's entire cost discussion.
- **The model cannot revise.** Once a token is emitted, it's in the context and it stays. There's no backspace, which is why a model that starts down a wrong path tends to commit to it.
- **Errors compound.** A wrong token becomes input for the next prediction, so mistakes can cascade.

---

## 14. Scaling and emergence

The unexpected discovery of the last decade: **bigger reliably means smarter**, along three axes at once.

1. **Data** — millions → trillions of words
2. **Parameters** — millions → hundreds of billions of weights
3. **Compute** — days → months on thousands of GPUs

**The result: emergent capabilities.** Reasoning, coding, and multi-step logic appear *spontaneously* at scale. They weren't designed in, and they weren't present in smaller models. Nobody wrote a reasoning module — it showed up when the model got big enough.

> **A note on "emergence," because the term is contested:** some researchers argue apparent emergence is partly an artifact of how capabilities are measured — a smooth underlying improvement can look like a sudden jump if your metric is all-or-nothing (e.g. exact-match accuracy on a multi-step problem). This doesn't make the observation useless, but "capabilities appear suddenly and unpredictably" is a stronger claim than the evidence strictly supports. S10 covers what scaling laws actually predict — and importantly, they predict *loss*, not capability.

---

## 15. Foundation models — the paradigm shift

| Old paradigm | New paradigm |
|---|---|
| Task-specific AI: Model A translates, Model B summarises, Model C does sentiment | One massive general-purpose model, steered to any task by **prompting** |

The "foundation" is base knowledge you build *on* rather than rebuild each time.

**This is the economic hinge of the entire field.** When one pretrained model can be adapted to thousands of tasks just by changing the text you send it, most of the value shifts from *making models* to *building with them*. You no longer build tools from scratch; you build on top of these giants.

That's exactly what "AI Engineering" (Week 0 §7) means, and it's why this course spends Weeks 8–25 building *with* models rather than training them.

---

## 16. Where we are (2024–2025)

1. **Multimodality** — not just text: models see images, hear voices, and speak back in real time. *(S9 covers how this actually works.)*
2. **Reasoning** — models designed to "think" before answering, solving harder maths and logic. *(Week 15, RLVR.)*
3. **Agents** — the shift from chatbots to systems that use tools, browse, and complete multi-step tasks. *(Weeks 8, 21, 22, 25.)*

These three trends map directly onto the back half of the syllabus.

---

## Common confusions

**"Is an LLM just a fancy autocomplete?"** Mechanically, next-token prediction is what it does — that part is accurate. The disputed part is whether doing it extremely well requires (and produces) something more. §8 and §13 deliberately hold this open; S13 examines the actual evidence, which shows models building real internal representations that are causally used, while remaining patchy and unreliable. Neither "just autocomplete" nor "it understands" survives contact with the evidence intact.

**"Do embedding dimensions mean things?"** No. The royalty/gender/edibility table in §9 is a teaching device. Real dimensions are whatever the optimiser found useful and are generally not human-interpretable.

**"Did attention replace RNNs because it's more accurate?"** Partly — but the bigger reason is **parallelisation** (§12). Attention could be trained on vastly more data in the same wall-clock time. Being trainable at scale beat being marginally better.

**"If bigger is better, why not just keep scaling?"** Because you eventually run out of both money and data, and because inference cost scales with model size forever while training cost is paid once. S10 covers why the field now deliberately trains *smaller* models on *more* data.

**"Is the 'AI winter' history relevant, or just trivia?"** Relevant. Both winters followed overpromising rather than technical failure, which is a useful calibration when reading confident forecasts today.

---

## Key takeaways

1. **The field advances by hitting a wall and reformulating**, not by steady accumulation. Learn the chain of failures — it's the best predictor of what breaks next.
2. **Implicit vs explicit knowledge** is the root cause of rule-based AI's failure: rules can only encode what you can state, and most human skill can't be stated.
3. **ML escapes that trap** by learning patterns nobody has to articulate — at the cost of interpretability, permanently.
4. **Data + compute + algorithms all had to converge (~2012).** Missing any one is why earlier attempts stalled despite roughly correct ideas.
5. **Embeddings turned meaning into geometry** — words became points, and relationships became consistent directions.
6. **Static embeddings' fatal flaw is one vector per word**, which attention fixes by computing representations from context.
7. **RNNs brought context but forgot**, and — just as importantly — couldn't be parallelised.
8. **Attention won on three counts:** no forgetting, parallel speed, and rich context. The speed is what made scale affordable, and it's the most underrated of the three.
9. **Next-word prediction at scale** yields grammar, facts, and reasoning as side effects. The objective is simple; satisfying it is not.
10. **Autoregressive generation** means latency scales with output length, the model can't revise, and errors compound.
11. **Foundation models shifted the industry** from building models to building *on* models — which is the reason this course exists in the shape it does.

---

## Glossary

| Term | Meaning |
|---|---|
| **Expert System** | Rule-based AI encoding human expertise as if/then logic. |
| **Explicit knowledge** | Knowledge that can be stated in words and written as rules. |
| **Implicit knowledge** | Knowledge you have but can't articulate — faces, bikes, sarcasm. |
| **AI winter** | A period of collapsed funding and interest following overpromising. Two: 1970s, 1980s–90s. |
| **Vector** | An ordered list of numbers; here, a point in high-dimensional space representing a word. |
| **Embedding** | A vector representing a word, where geometric position encodes meaning. |
| **Word math** | Vector arithmetic on embeddings mirroring semantic relationships (King − Man + Woman ≈ Queen). |
| **RNN** | Recurrent Neural Network — processes sequences one step at a time carrying a hidden memory. |
| **Hidden state** | The fixed-size memory vector an RNN updates at each step. |
| **Long-range dependency** | Relating information across a long span; the failure mode of RNNs. |
| **Attention** | Mechanism letting every token weigh every other token directly, regardless of distance. |
| **Transformer** | Architecture built entirely on attention — no recurrence, no convolution. |
| **Autoregressive generation** | Producing one token at a time, feeding each output back in as input. |
| **Emergent capability** | An ability appearing at scale without being explicitly trained for — though see §14's caveat. |
| **Foundation model** | A large general-purpose pretrained model adapted to many tasks via prompting. |
| **Multimodality** | Handling multiple input/output types — text, image, audio. |

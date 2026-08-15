# Week 1 — Quiz (20 questions)

**Topic:** Fast-Tracking the Course of AI — from "What is AI?" to "How ChatGPT works"
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** According to the lecture, learning is best described as:
- A) Memorising facts accurately
- B) Changing yourself based on experience — a try / feedback / adjust / repeat loop
- C) Copying the behaviour of an expert
- D) Storing rules in a database

**2.** Which pair correctly matches the type of knowledge?
- A) Explicit = riding a bike; Implicit = "Paris is the capital of France"
- B) Explicit = "Paris is the capital of France"; Implicit = recognising a face
- C) Both are statable in words
- D) Neither can be encoded in a computer

**3.** The lecture defines intelligence as:
- A) The ability to score highly on standardised tests
- B) The ability to store and retrieve large amounts of information
- C) The ability to achieve goals in a wide range of situations
- D) The ability to perform arithmetic faster than a human

**4.** Rule-based AI (expert systems) fundamentally failed because:
- A) Computers were not fast enough to evaluate rules
- B) Rules can only encode explicit knowledge, while most human competence is implicit
- C) Nobody knew how to write if/then statements
- D) There was no way to store the rules

**5.** Which is the correct containment relationship?
- A) AI ⊂ ML ⊂ Deep Learning
- B) ML ⊂ AI ⊂ Deep Learning
- C) Deep Learning ⊂ ML ⊂ AI
- D) They are three unrelated fields

**6.** The two bottlenecks that limited early machine learning were:
- A) Bad algorithms and bad programmers
- B) Not enough data and not enough compute
- C) Lack of funding and lack of interest
- D) Poor hardware design and no operating systems

**7.** Which three ingredients converged around 2012 to trigger the AI explosion?
- A) Cloud, mobile, and social media
- B) Data (internet), compute (GPUs), and algorithms (deep learning)
- C) Transformers, RLHF, and multimodality
- D) Python, PyTorch, and Colab

**8.** In the sentence "I saw the man with the telescope," the difficulty illustrated is:
- A) Spelling variation
- B) Structural ambiguity — multiple valid interpretations
- C) Sarcasm
- D) Idiom

**9.** The statistical-pattern approach to NLP ("New" → "York") failed principally because:
- A) It was too slow to run
- B) Pattern matching is not the same as understanding meaning
- C) It required too much memory
- D) It could not be parallelised

**10.** The central limitation of classic word embeddings is:
- A) They require too many dimensions
- B) Each word has exactly one vector, so context cannot change its representation
- C) They cannot represent nouns
- D) They only work for English

**11.** Why do RNNs struggle with "The cat, which was sitting on the mat that I bought…, was happy"?
- A) The sentence is grammatically invalid
- B) They process linearly with limited memory and forget the distant subject
- C) They cannot handle commas
- D) They run out of vocabulary

**12.** Beyond quality, the property of attention that made scaling economically feasible is:
- A) It uses less memory than an RNN
- B) It processes all words in parallel, making training far faster on GPUs
- C) It requires no training data
- D) It removes the need for embeddings

---

## Short answer

**13.** Write out the chain of approaches from rule-based AI to Transformers, naming the specific failure that motivated each successor.

**14.** Explain "word math" (King − Man + Woman = Queen). What does it reveal about the structure of embedding space, and why is it surprising?

**15.** Using the two "bank" sentences, explain precisely what static embeddings cannot do and how attention fixes it.

**16.** The lecture criticises statistical NLP as "pattern matching ≠ understanding," yet LLMs are also next-word predictors. How does the lecture reconcile this, and what remains genuinely open?

**17.** Give the three reasons "predict the next word" produces grammar, facts, and reasoning.

**18.** What is an *emergent capability*, and why does emergence make model behaviour hard to predict in advance?

**19.** Contrast the task-specific and foundation-model paradigms. Why does this shift define the job title "AI Engineer"?

**20.** The interpretable embedding table (Royalty / Gender / Edibility) is a teaching device. What is misleading about it, and why is the simplification still worth making?

---
---

## Answer key

**1. B** — "Changing yourself based on experiences, basically adjustments," expressed as a loop: try → feedback → adjust → repeat. This same loop reappears as the training loop (Week 2) and the RL loop (Week 15).

**2. B** — Explicit knowledge is statable and rule-writable; implicit knowledge (faces, bikes, sarcasm) is possessed but not articulable.

**3. C** — "The ability to achieve goals in a wide range of situations." The emphasis on *range* is what excludes narrow systems like chess engines.

**4. B** — Rules capture only explicit knowledge. Two named failure modes: rules don't scale, and rigid logic breaks on unseen data.

**5. C** — Deep Learning ⊂ ML ⊂ AI. AI is the goal, ML one method, deep learning one kind of ML.

**6. B** — Not enough data (the internet was young) and not enough compute (training would have taken years). Notably, neither was algorithmic.

**7. B** — Data from the internet, compute from GPUs (built for gaming), and deep-learning algorithms. All three were required together.

**8. B** — Structural ambiguity: who holds the telescope is grammatically undecidable without context.

**9. B** — It predicts by frequency with no concept of meaning. It nonetheless powered Google Translate for nearly a decade.

**10. B** — One vector per word. "Bank" gets a single position that cannot be both financial and geographic, because the model never looks at surrounding words.

**11. B** — Linear processing plus limited memory capacity means the subject at the start is lost by the time the verb arrives — the long-range dependency problem.

**12. B** — Parallel processing of all words at once. RNNs are inherently sequential; transformers saturate GPUs, making large-scale training affordable.

**13.** Rules/expert systems → *can't encode implicit knowledge, don't scale, too brittle* → Machine learning → *starved of data and compute; two AI winters* → Data + GPUs + deep learning from 2012 → *language still hard due to ambiguity* → Dictionary approach → *definitions ignore context* → Statistical patterns → *frequency isn't meaning* → Word embeddings → *one fixed vector per word, context-blind* → RNNs → *forget across long ranges, and are sequential/slow* → Attention & Transformers.

**14.** Vector arithmetic on embeddings mirrors semantic relationships: King − Man + Woman lands near Queen; Paris − France + Italy lands near Rome; Walking − Walk + Swim lands near Swimming. It reveals that relationships such as gender, capital-of, and verb tense are encoded as **consistent directions** in the space. It is surprising because nobody designed those axes — the geometry falls out of training on the objective alone.

**15.** "I went to the bank to deposit cash" and "I sat on the bank of the river" use the same token. A static embedding assigns "bank" one vector, so both sentences receive an identical representation and the senses cannot be distinguished — the model never consults surrounding words. Attention computes each word's representation *from its context*: processing "bank," the model attends to "deposit"/"cash" in one sentence and "river"/"sat on" in the other, producing two different context-dependent vectors for the same token.

**16.** The reconciliation is one of **scale and consequence**: predicting the next word *well enough*, across trillions of tokens, requires internalising grammar, factual relations, and chains of reasoning — so capabilities that look like understanding emerge as by-products of the objective. What remains open is whether that constitutes understanding in kind or is a far more competent version of the same pattern matching. The lecture asserts the practical result without settling the philosophical claim, and the honest position is to hold that tension.

**17.** (i) **Grammar & syntax** — accurate prediction forces the model to internalise language structure. (ii) **Factual knowledge** — correct continuations depend on real relations between concepts ("The capital of France is ___"). (iii) **Logic & reasoning** — anticipating the continuation of an argument requires following its chain.

**18.** An emergent capability is an ability — reasoning, coding, logic — that appears spontaneously as models scale, without being explicitly trained for and without being present in smaller models. It makes behaviour hard to predict because capability is not a smooth, extrapolable function of size: you cannot reliably infer from a small model what a larger one will be able to do, so new abilities (and new risks) are discovered after training rather than planned before it.

**19.** *Task-specific:* a separate model per task — one for translation, one for summarisation, one for sentiment — each built and trained from scratch. *Foundation:* one large general-purpose model trained on everything, steered to any task by prompting. The shift defines "AI Engineer" because the scarce work moves from *training models* to *building systems on top of existing models* — prompting, retrieval, agents, evaluation, observability — which is precisely the content of Weeks 8–25.

**20.** It is misleading because real embedding dimensions are **not interpretable**: no dimension is labelled "royalty," and learned axes rarely correspond to human-nameable concepts; the model uses whatever axes reduce loss, typically distributed and entangled across many dimensions. The simplification is still worth making because it conveys the essential and true idea — that meaning is encoded as position in a continuous space, and that similarity and relationships become geometric — which is what the rest of the course actually depends on.

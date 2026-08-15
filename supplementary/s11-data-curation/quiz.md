# S11 — Quiz (20 questions)

**Topic:** Data — Curation, Quality, and Tokenizers
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** "More data is better" is:
- A) True at every stage
- B) Roughly true for pretraining, actively false for instruction tuning
- C) False at every stage
- D) True only for preference data

**2.** Typical pretraining pipelines retain approximately:
- A) 80% of collected data
- B) A low single-digit percentage
- C) 50%
- D) 25%

**3.** The empirical finding about deduplication is that it:
- A) Trades quality for training speed
- B) Improves models at fixed compute — less data, better model
- C) Only matters for privacy
- D) Helps small models but not large ones

**4.** MinHash produces a signature whose collision probability equals:
- A) Cosine similarity
- B) Jaccard similarity of the documents' n-gram sets
- C) Edit distance
- D) Perplexity ratio

**5.** Perplexity filtering should drop:
- A) Only high-perplexity documents
- B) Both extremes — very low perplexity indicates boilerplate
- C) Only low-perplexity documents
- D) Documents at the median

**6.** Aggressive toxicity filtering measurably:
- A) Has no downside
- B) Reduces the model's ability to *recognise* toxicity
- C) Improves multilingual performance
- D) Increases memorisation

**7.** Including code in a general pretraining mix:
- A) Only improves coding ability
- B) Improves general reasoning as well
- C) Degrades prose quality
- D) Has no measured effect

**8.** A tokenizer trained mostly on English makes Hindi or Thai text:
- A) Cheaper per sentence
- B) Cost 3–5× the tokens, fitting less in context and performing worse
- C) Unrepresentable
- D) Identical in cost

**9.** "LLMs are bad at arithmetic" is, in significant part:
- A) A fundamental reasoning limit
- B) A tokenization artifact from inconsistent digit splitting
- C) An attention limitation
- D) Caused by RLHF

**10.** After pretraining, changing a tokenizer requires:
- A) Nothing — it's independent
- B) Retraining the model, since embedding IDs would all change meaning
- C) Only re-running SFT
- D) A vocabulary remap

**11.** The LIMA finding is that instruction tuning works well with:
- A) Millions of examples
- B) ~1,000 carefully curated, diverse examples
- C) Only synthetic data
- D) Preference pairs instead of demonstrations

**12.** The characteristic failure mode of synthetic data generation is:
- A) Factual drift
- B) Diversity collapse toward the generator's favourite phrasings
- C) Excessive length
- D) Tokenizer mismatch

---

## Short answer

**13.** Explain why the three data stages have opposite rules, and why conflating them causes mistakes.

**14.** Explain the four reasons duplicates hurt, and describe the three levels of deduplication.

**15.** Explain the trap in the word "quality" for data filtering, with two concrete examples, and state the practical rule.

**16.** Explain contamination, why it's nearly guaranteed at web scale, and what it means for how you read benchmark scores.

**17.** Explain how the tokenizer determines what a model is bad at, covering numbers, non-English text, and code.

**18.** Explain why quality beats quantity for instruction data, and define what "quality" concretely means there.

**19.** Explain what makes preference data hard, including the ceiling imposed by annotator agreement.

**20.** You're building an SFT dataset to make a 7B model good at answering questions about your company's internal engineering documentation. Design the pipeline end to end.

---
---

## Answer key

**1. B** — Pretraining is filtering at scale; instruction tuning is curation where mediocre examples actively teach mediocrity.

**2. B** — Most of building a dataset is deciding what to discard.

**3. B** — A rare strictly-better outcome rather than a trade-off.

**4. B** — That property is what makes MinHash useful for near-duplicate detection.

**5. B** — Boilerplate is extremely predictable; you want the middle.

**6. B** — You cannot classify what you've never seen; a real, repeatedly documented trade-off.

**7. B** — The leading hypothesis is that code supplies long chains of strict logical dependency.

**8. B** — Higher cost, less content per context window, and worse quality — three penalties from one decision.

**9. B** — Inconsistent digit splitting forces the model to learn digit relationships through an unstable representation; modern tokenizers often split digits individually.

**10. B** — The embedding table is indexed by token ID, so every ID would mean something different.

**11. B** — Pretraining installed the capability; SFT teaches format and behaviour, which needs examples rather than volume.

**12. B** — Left unchecked, generators produce variations on a theme.

**13.** **Pretraining is a filtering problem at absurd scale.** You start with petabytes of crawl and discard 90%+; the goal is broad coverage of language and knowledge, and volume genuinely matters because Chinchilla-scale training needs trillions of tokens (S10). Here "more data is better" is roughly true, provided it's deduplicated and filtered. **Instruction tuning is a curation problem at small scale.** The model already has the knowledge from pretraining; SFT teaches **format and behaviour** — how to structure an answer, what register to use, when to refuse. That is a much smaller thing to learn, and it's learned from examples rather than from volume. Critically, **SFT is imitation learning**, so mediocre examples don't dilute the signal harmlessly — they **actively teach mediocrity**, because the model learns to reproduce exactly what you showed it, flaws included. A thousand excellent examples beat a hundred thousand mediocre ones. **Preference tuning is an elicitation problem**: the difficulty isn't volume or filtering but producing *meaningful comparisons*, and it's bounded by whether humans can even agree on which response is better. **Conflating them is the most common mistake** — someone applies "more data is better" to instruction tuning, scrapes a hundred thousand mediocre pairs, trains, and gets a model that is measurably worse than one trained on a curated thousand. The opposite error also occurs: over-filtering a pretraining corpus into narrowness because "quality matters."

**14.** **Four reasons duplicates hurt.** **Memorisation** — a passage repeated a thousand times gets memorised verbatim rather than generalised from, which is simultaneously a privacy problem (training-data extraction attacks), a copyright problem, and a capability problem, since memorised text crowds out learned patterns. **Wasted compute** — every duplicate consumes training budget teaching nothing new, which at Chinchilla scale is real money. **Distorted distribution** — Common Crawl massively over-represents boilerplate, so without dedup the model spends disproportionate capacity modelling cookie banners and navigation menus. **Contamination risk** — duplicates make it far likelier that benchmark data slipped in somewhere. The striking empirical result is that **deduplication improves models at fixed compute**: you train on less data and get a better model, which is a strictly better outcome rather than a trade-off. **Three levels.** **Exact dedup** hashes each document and drops repeats — cheap, and catches more than you'd expect. **Fuzzy/near-dedup** is the important one, because the web is full of documents differing only by a timestamp or an ad; the standard technique is **MinHash + LSH**, where documents are represented as sets of n-gram shingles, MinHash produces compact signatures whose collision probability equals the **Jaccard similarity** of those sets, and **LSH** buckets similar signatures so you only compare plausible candidates instead of doing an impossible N² all-pairs comparison — the same "avoid comparing everything to everything" insight as S6's ANN indexes. **Semantic dedup** embeds documents and removes near-neighbours in vector space, catching paraphrases sharing no n-grams; it's more expensive and needs care, since aggressive settings remove legitimate diversity. Also dedup **within** documents, for repeated lines and boilerplate on a single page.

**15.** **"Quality" is not a property of text; it's a relationship between text and a purpose.** Every filter encodes an opinion about what good text looks like, and that opinion becomes the model's opinion — filtering is therefore a **model-behaviour decision wearing the costume of a data-cleaning step.** **Example one: toxicity filtering.** Aggressively removing toxic content measurably reduces the model's ability to **recognise** toxicity, because you cannot classify what you've never seen. A model trained on scrupulously clean data is a worse content moderator, and possibly worse at handling adversarial input generally — a real, repeatedly documented trade-off rather than a hypothetical concern. **Example two: quality classifiers trained on Wikipedia-likeness.** Filtering toward encyclopedic prose produces a model that writes like an encyclopedia and handles conversational registers poorly. Worse, English-centric notions of quality systematically discard dialects, non-Western content, and informal registers, which later surfaces as "model bias" when it was in fact a specific, traceable decision in a filtering script. **The practical rule is: make filtering decisions explicitly, document them, and ablate them.** Follow FineWeb's example — train small models with and without a given filter and **measure downstream performance** rather than reasoning about whether the filter sounds sensible. Almost every filter sounds sensible; that's precisely why intuition is a poor guide here, and why the corpora worth learning from are the ones that publish their ablations.

**16.** **Contamination** is test data appearing in training data. A contaminated model produces evaluation scores measuring **memorisation rather than capability**, and the scores look great, which is what makes it dangerous rather than merely annoying. **It is nearly guaranteed at web scale** because benchmarks are *published on the web*: they appear in papers, blog posts, tutorials, GitHub repos, leaderboard discussions, and Stack Overflow answers. Common Crawl contains them many times over, and §3's point compounds this — duplication multiplies the chance that at least one copy survives filtering. **Decontamination** searches the corpus for n-gram overlaps with known benchmark sets and removes matches, but it is inherently imperfect: paraphrases and translations survive, and **you can only decontaminate against benchmarks you know about**, which excludes any benchmark created after your pipeline ran. **What this means for reading scores:** treat public benchmark numbers **sceptically**, especially from models whose data pipeline isn't documented — this is S3's data-leakage lesson at corpus scale, and Week 20's scepticism about reported numbers now with a concrete mechanism attached. **Held-out evaluation on your own private data is the only measurement you can fully trust** — not because published numbers are dishonest, but because you cannot verify they're clean, and the model's authors may not be able to either. **Newer benchmarks built from recent data are more informative** than older ones purely because there's been less time for leakage, which is why benchmark freshness is itself a quality signal.

**17.** **BPE tokenizers learn their merges from a corpus**, so the tokenizer inherits that corpus's distribution — and since the embedding table is indexed by token ID, the choice is effectively permanent after pretraining. **Numbers.** If "12345" is split inconsistently — sometimes "123"+"45", sometimes "1"+"2345" depending on which sequences were frequent in the tokenizer's training corpus — the model must learn digit relationships through an unstable representation, and the same numeric value has different token decompositions in different contexts. Arithmetic becomes far harder than it needs to be, which is why modern tokenizers frequently **split digits individually**. **A meaningful part of "LLMs are bad at maths" is therefore a tokenization artifact rather than a reasoning limitation** — a useful thing to know before concluding a model can't reason. **Non-English languages.** A tokenizer trained mostly on English fragments other scripts into many more tokens; the same sentence in Hindi or Thai may cost **3–5× the tokens** of its English equivalent. That produces three penalties from one decision: **higher API cost, less actual content fitting in the context window, and worse quality** because each concept is spread thinly across many poorly-trained tokens. It is also a genuine equity issue, since speakers of under-represented languages pay more for worse service. **Code.** Whitespace handling determines whether Python indentation is represented efficiently; tokenizers trained without code handle it badly, wasting tokens on runs of spaces. **Domain terms** — medical, legal, chemical vocabulary — fragment heavily if absent from tokenizer training, making the model least efficient in exactly the specialised domains people most want to fine-tune for (Week 12), and the fine-tune inherits the tokenizer along with the weights.

**18.** **Pretraining already installed the knowledge and capability.** Instruction tuning isn't teaching the model facts or reasoning — it's teaching **format and behaviour**: how to respond, how to structure an answer, when to refuse, what register to adopt, how long to be. That is a comparatively small thing to learn, and it's learned from **examples of the target behaviour** rather than from sheer volume. The decisive mechanism is that **SFT is imitation learning**: the model learns to produce what you showed it. So a mediocre example isn't neutral filler that gets averaged away — it is a **demonstration of mediocrity that the model faithfully learns to reproduce**. This is why the **LIMA** result ("Less Is More for Alignment") held up: 1,000 carefully curated examples were competitive with models tuned on vastly larger sets. **Concretely, quality means five things.** **Correctness** — a wrong answer in SFT data becomes a wrong answer the model produces confidently. **Response quality** — the response must be one you'd happily ship, because its style, structure, and length will be imitated. **Diversity** — coverage across task types matters far more than count within a type; a thousand examples spanning many tasks beats ten thousand of one. **Consistency** — contradictory examples, where one similar request is refused and another complied with, teach inconsistency directly, and this is the most common defect in crowd-sourced instruction data. **Format consistency** — inconsistent formatting in equals inconsistent formatting out. The practical corollary is that **reading 100 examples by hand before training on 10,000** is the highest-value hour in the process, because systematic defects are obvious to a human and invisible to every automatic filter.

**19.** Preference pairs — a prompt with a chosen and a rejected response — are harder to produce well than SFT demonstrations, for several compounding reasons. **Hard pairs teach most, and they're the hard ones to make.** Week 14's own point, echoed in S6's hard-negative discussion: when the rejected response is genuinely good and loses on something subtle, the model learns a real decision boundary; when it's obviously terrible, the model learns nothing it didn't already know. But generating a *nearly as good* response deliberately is much harder than generating a bad one, so datasets drift toward easy pairs that look impressive by volume and teach little. **Annotator agreement is a hard ceiling.** If humans agree only 65% of the time on which response is better, that disagreement is irreducible noise, and no volume of additional data pushes the model past it — you are training on a label that isn't well-defined. **Measure inter-annotator agreement before scaling annotation**, because low agreement means the task specification is ambiguous, and the fix is a better rubric rather than more labellers. **Length bias is pervasive and reliably observed.** Annotators prefer longer responses largely independent of content, so models learn "longer is better" and become verbose — one of the most consistently documented pathologies in preference tuning, and it requires explicit control (length-normalised comparisons, length-matched pairs, or explicit penalties) rather than hoping it averages out. It's also a clean instance of Week 14's Goodhart problem: the proxy rewards a correlate of quality rather than quality. **AI feedback (RLAIF)** scales far better than human labelling and works reasonably well, but it **inherits the judge model's biases** — including its own length and style preferences — so S3's caveats about model-as-judge apply in full, and calibrating the judge against human labels on a sample is not optional.

**20.** **Start by defining the task and the eval, before collecting anything.** "Answering questions about internal engineering docs" is underspecified — decide whether answers should cite sources, how they handle missing information, what register they use, and how long they should be, because SFT is imitation learning and every one of those becomes a property the model copies. Then **build a held-out eval set first** (S3) from real questions with known-good answers, and never let it touch training. Without this you cannot tell whether the fine-tune helped. **Second, ask whether SFT is the right tool at all** — Week 12's decision table matters here. Internal documentation is **knowledge**, and knowledge changes; **RAG adds knowledge, fine-tuning changes behaviour**. A model fine-tuned on today's docs is stale the moment they change, and it will confidently answer from memorised outdated content. The strong default is **RAG over the docs (Week 9, S6), with SFT used to teach the *behaviour*** — answering in your house format, citing chunk IDs, saying "not documented" when retrieval comes back empty. Frame the dataset accordingly: examples are `(question + retrieved context) → grounded answer`, not `question → answer`. **Third, source questions from real usage.** Support tickets, internal chat questions, search logs, and onboarding questions beat imagined ones every time, because they carry the real distribution including the badly-phrased and ambiguous ones. Supplement with **synthetic questions generated per document chunk** (Week 13) to get coverage of areas nobody has asked about yet, using **high temperature** for diversity (S7). **Fourth, generate and filter responses hard.** Use a strong model to draft answers given the retrieved context, then filter with: rule-based checks (does it cite a chunk that was actually retrieved? does it stay within the context?), model-as-judge scoring with the S3 caveats, and **execution-based verification wherever the domain allows** — if an answer contains a command, a config snippet, or a code sample, *run it*. Verifiable filtering is the strongest signal available and it's the same insight as Week 15's RLVR. **Fifth, deduplicate, including semantically.** Synthetic generation collapses toward the generator's favourite phrasings, so semantic dedup (S6) is the main defence, and it also prevents heavily-documented subsystems from dominating the set. **Sixth, balance and audit coverage** — deliberately include refusal/abstention examples ("this isn't documented"), because a model never shown how to decline will fabricate, and that's the single most damaging failure mode for an internal docs assistant. Check that examples span services, seniority levels of question, and both happy-path and troubleshooting queries. **Seventh, read 100 examples end to end by hand.** Non-negotiable, and the highest-value hour in the project — it catches systematic problems (all answers too long, all citing the same chunk, inconsistent refusal behaviour) that no automatic filter surfaces. **Then train small and measure:** LoRA rather than full fine-tuning (Week 12) to avoid catastrophic forgetting and keep adapters versionable; start with roughly 1,000–2,000 curated examples per LIMA's finding rather than scaling first; and evaluate on the held-out set with confidence intervals over n samples, comparing against the RAG-only baseline. **If RAG-only wins, ship that** — the honest outcome, and a common one. **Finally, plan for staleness:** version the dataset against the documentation snapshot it came from, track provenance so any removed or superseded doc can be traced to the examples derived from it, and expect to refresh rather than treating the fine-tune as finished.

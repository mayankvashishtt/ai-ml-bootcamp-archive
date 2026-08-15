# S11 — Data: Curation, Quality, and Tokenizers

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. Week 13 covers generating synthetic data for SFT, and that's the only place the course looks at datasets directly. Nothing covers where pretraining corpora come from, how they're cleaned, why deduplication matters enormously, what makes instruction data good, or why the tokenizer — a component you never think about — quietly determines what your model is bad at.

**Fills the gap after:** Week 13 (Fine-tuning Pt 2 — SFT)
**Prerequisites:** Weeks 3 (tokenization), 12–14 (fine-tuning), S3 (evaluation) for §5

---

## 0. The idea in plain language

**A model is a compression of its training data.** Everything it knows, every bias it has, every failure mode it exhibits — all of it came from the data. Architecture determines how efficiently the model can absorb data; the data determines what there is to absorb.

This is the least glamorous part of machine learning and, by a wide margin, the highest-leverage. The uncomfortable industry truth is that most of the difference between a mediocre model and a good one at the same size is **data work**, not clever architecture. Labs guard their data pipelines more closely than their model designs, which tells you where they think the value is.

Three claims this lecture defends:

1. **Deduplication is worth more than almost any architectural tweak** (§3).
2. **Data quality beats data quantity past a surprisingly low threshold** — especially for fine-tuning (§8).
3. **The tokenizer is a data decision** that permanently shapes what your model finds easy (§7).

---

## 1. The three data stages

Map these to what you already know:

| Stage | Data | Scale | Course reference |
|---|---|---|---|
| **Pretraining** | Raw text from the web, books, code | Trillions of tokens | Week 7 (in miniature) |
| **Instruction tuning (SFT)** | Prompt → good response pairs | Thousands to millions | Week 13 |
| **Preference tuning** | Prompt → chosen/rejected pairs | Thousands to hundreds of thousands | Week 14 |

They have almost nothing in common operationally. Pretraining is a **filtering** problem at absurd scale — you're throwing away 90%+ of what you collect. Instruction tuning is a **curation** problem at small scale, where a few thousand excellent examples beat a hundred thousand mediocre ones. Preference tuning is an **elicitation** problem, where the difficulty is producing meaningful comparisons at all.

Conflating them is the most common mistake. "More data is better" is roughly true for pretraining and actively false for instruction tuning.

---

## 2. Where pretraining data comes from

**Common Crawl** is the backbone — a nonprofit that has crawled the public web since 2008, releasing petabytes of raw HTML. It is free, enormous, and mostly garbage: boilerplate, navigation menus, SEO spam, adult content, machine-translated filler, and near-infinite duplicates.

The pipeline every lab runs looks roughly like:

```
raw HTML
  → text extraction (strip boilerplate, nav, ads)
  → language identification (keep target languages)
  → quality filtering (heuristics + classifiers)
  → deduplication (exact, then fuzzy)
  → decontamination (remove benchmark data)
  → PII / toxicity filtering
  → final corpus
```

The attrition is brutal: **filtering typically retains a low single-digit percentage** of what came in. Most of the work of "building a dataset" is deciding what to throw away.

**Open reference corpora worth knowing** — because they document their pipelines, which is how you learn this:
- **C4** — Colossal Clean Crawled Corpus, the T5 dataset. Heuristic filtering, and one of the first to be documented in detail.
- **The Pile** (EleutherAI) — 22 curated sub-corpora (papers, code, books, Q&A) rather than crawl-only, an early demonstration that **mixture matters**.
- **RedPajama** — an open reproduction of LLaMA's described data mix.
- **FineWeb / FineWeb-Edu** (HuggingFace) — the most useful modern reference, because it publishes **ablations** showing which filtering choices actually improved downstream performance. FineWeb-Edu in particular filtered for educational content using a classifier and showed sizeable gains — direct evidence for the quality-over-quantity claim.

**Also non-web sources:** code (GitHub, with licence filtering), books, academic papers (arXiv, PubMed), Wikipedia, Q&A sites, and increasingly **synthetic data** — which is Week 13's technique applied at pretraining scale.

---

## 3. Deduplication — the highest-value step

If you do one thing to a dataset, do this.

**Why duplicates hurt:**

- **Memorisation.** A passage repeated a thousand times gets memorised verbatim rather than generalised from. That's a privacy problem (training data extraction), a copyright problem, and a capability problem — memorised text crowds out learned patterns.
- **Wasted compute.** Every duplicate consumes training budget teaching the model nothing new.
- **Distorted distribution.** Common Crawl massively over-represents boilerplate. Without dedup, the model spends disproportionate capacity on cookie banners.
- **Benchmark contamination.** Duplicates make it far more likely that test data slipped in somewhere (§5).

The consistent empirical finding is that **deduplication improves models at fixed compute** — you train on *less* data and get a *better* model. That's rare and worth internalising: it's not a trade-off, it's a strictly better outcome.

**Three levels:**

**Exact dedup** — hash each document, drop repeats. Cheap, and catches a surprising amount.

**Fuzzy / near-dedup** — the important one, because the web is full of documents that differ by a timestamp or an ad. The standard technique is **MinHash + LSH**:
- Represent each document as a set of n-grams (shingles).
- **MinHash** produces a compact signature such that the probability two signatures match equals the **Jaccard similarity** of the underlying sets.
- **LSH (Locality-Sensitive Hashing)** buckets similar signatures together so you only compare plausible candidates, avoiding the impossible N² all-pairs comparison.

This is the same "avoid comparing everything to everything" idea as S6's ANN indexes, solving a different problem with the same insight.

**Semantic dedup** — embed documents and remove near-neighbours in vector space (S6). Catches paraphrases that share no n-grams. More expensive, and requires care: aggressive semantic dedup removes legitimate diversity.

**Also dedup within documents** — repeated lines, repeated paragraphs, and repeated boilerplate inside a single page.

---

## 4. Quality filtering, and the trap in the word "quality"

Three families of approach:

**Heuristic filters** — cheap rules: minimum length, punctuation ratio, fraction of lines ending in "…", stopword presence, symbol-to-word ratio, repeated-line fraction, blocklists. Crude, fast, effective at removing obvious junk. C4's ruleset is the canonical example.

**Classifier-based filters** — train a small classifier to score "is this like my reference set?" Typically trained with high-quality text (Wikipedia, books, curated web) as positives and random crawl as negatives. This is how FineWeb-Edu was built, and it's the most effective general approach.

**Perplexity filtering** — score documents with a small language model and drop the extremes. High perplexity means gibberish; **very low perplexity also means junk** — boilerplate is extremely predictable. It's the middle you want, which is a genuinely counterintuitive and useful fact.

### The trap

**"Quality" is not a property of text; it's a relationship between text and a purpose.** Every filter encodes an opinion about what good text looks like, and that opinion becomes the model's opinion.

Concretely:
- Filtering for Wikipedia-likeness produces a model that writes like an encyclopedia and handles conversational registers poorly.
- Aggressive toxicity filtering measurably reduces the model's ability to *recognise* toxicity — you cannot classify what you've never seen. This is a real and repeatedly documented trade-off, not a hypothetical.
- Filters trained on English-centric notions of quality systematically discard dialects and non-Western content, which then shows up as bias that looks like a model problem but was a data-pipeline decision.

**The practical rule:** filtering decisions are model-behaviour decisions. Make them explicitly, document them, and — following FineWeb's example — **ablate them**. Train small models with and without a filter and measure. Never adopt a filter because it sounds sensible.

---

## 5. Decontamination — and why S3 cares

**Contamination** is test data appearing in training data. A model that has seen the test set produces evaluation scores that measure memorisation rather than capability.

At web scale this is nearly guaranteed by default: benchmarks are published on the web, discussed in papers, copied into blog posts and GitHub repos. Common Crawl contains them many times over.

**Decontamination** searches the training corpus for n-gram overlaps with known benchmark sets and removes matches. It's imperfect — paraphrases and translations survive, and you can only decontaminate against benchmarks you know about.

**The consequences you should carry from this:**
- **Treat public benchmark scores sceptically**, particularly for models whose data pipeline isn't documented. This is S3's data-leakage lesson at the corpus level, and Week 20's scepticism about reported numbers with a mechanism attached.
- **Held-out evaluation on your own private data is the only trustworthy measurement.** Not because public benchmarks are dishonest, but because you cannot verify they're clean.
- **Newer benchmarks with recent data are more informative** than older ones, purely because there's been less time for leakage.

---

## 6. Mixtures and curriculum

A corpus isn't one pile — it's a weighted blend, and the weights matter as much as the contents.

**Domain mixing.** How much web vs code vs books vs papers? These are deliberate choices with predictable effects. The most-cited example: **including code in the pretraining mix improves general reasoning**, not just coding ability — the leading hypothesis being that code provides long chains of strict logical dependency that prose rarely does. Whatever the mechanism, the effect is real enough that essentially every general model now includes substantial code.

**Upsampling.** High-quality sources are often repeated more than once per epoch while crawl data is seen once. Repeating a small high-quality set is a deliberate bet that its quality outweighs the memorisation cost of repetition.

**Curriculum / annealing.** Data order matters. A common modern pattern is to train mostly on the broad mix, then **anneal** on a much higher-quality subset (curated text, textbooks, synthetic instruction-like data) for the final phase. Late data has outsized influence on final behaviour — the same reason "last examples matter" appears in fine-tuning.

**Multilingual mixing.** Language proportions determine capability directly, and there's transfer between related languages, so a little of a low-resource language goes further than you'd expect.

---

## 7. Tokenizers — the invisible data decision

Week 3 taught what tokenization is. This is why the choice is load-bearing and effectively permanent.

**BPE (Byte-Pair Encoding)** is the standard: start from bytes, repeatedly merge the most frequent adjacent pair, and stop at the target vocabulary size. Crucially, **the merges are learned from a corpus** — so the tokenizer inherits that corpus's distribution.

### The trade-offs

**Vocabulary size:**
- **Larger vocab** → fewer tokens per document → shorter sequences → cheaper attention and more text per context window. But the embedding and output layers grow, and rare tokens get few training examples.
- **Smaller vocab** → longer sequences → more compute per document, but each token is better trained.

Modern models trend toward larger vocabularies (100k–200k+), largely because context efficiency is worth more than parameter savings.

### Why this determines what your model is bad at

**Numbers.** If the tokenizer splits "12345" inconsistently — sometimes "123"+"45", sometimes "1"+"2345" — arithmetic becomes harder than it needs to be, because the model must learn digit relationships through an inconsistent representation. Modern tokenizers frequently **split digits individually** specifically to fix this. **A meaningful chunk of "LLMs are bad at maths" is a tokenization artifact**, not a reasoning limitation.

**Non-English languages.** If the tokenizer was trained mostly on English, other languages fragment into many more tokens. The same sentence in Hindi or Thai might cost 3–5× the tokens of its English equivalent. That means: **higher cost, less content fitting in the context window, and worse quality** — three penalties from one decision, and it's a genuine equity issue in how these systems are priced.

**Code.** Whitespace handling determines whether Python indentation is represented efficiently. Tokenizers trained without code handle it terribly.

**Rare words and domain terms.** Medical, legal, and chemical terminology fragments into many pieces if absent from tokenizer training, making the model less efficient in exactly the domains people want to specialise in.

### The permanence

**You cannot change a tokenizer after pretraining without retraining the model.** The embedding table is indexed by token ID; change the tokenizer and every ID means something different. This is why tokenizer choice is one of the highest-stakes early decisions, and why fine-tuning a model for a new domain (Week 12) inherits that model's tokenizer along with its weaknesses.

---

## 8. Instruction data: quality over quantity

This is where the rules invert most sharply, and where your work actually lands (Week 13).

**The core finding**, popularised by the **LIMA** paper ("Less Is More for Alignment") and repeatedly confirmed since: **a small number of high-quality, diverse instruction examples outperforms a large number of mediocre ones.** LIMA used 1,000 carefully curated examples and was competitive with models tuned on vastly more.

**Why this works:** pretraining already installed the knowledge and capability. Instruction tuning is teaching **format and behaviour** — how to respond, how to structure an answer, when to refuse, what register to use. That's a much smaller thing to learn, and it's learned from examples rather than from volume. **Mediocre examples actively teach mediocrity**, because SFT is imitation learning: the model learns to produce what you showed it, including the flaws.

**What "quality" means concretely here:**

- **Correctness** — a wrong answer in SFT data is a wrong answer the model learns to produce confidently.
- **Response quality** — the response should be one you'd be happy shipping, because the model will imitate its style, structure, and length.
- **Diversity** — coverage of task types matters more than count within a type. A thousand examples spanning many tasks beats ten thousand of one.
- **Consistency** — contradictory examples (one refuses, another complies, for similar requests) teach inconsistency directly. This is the most common defect in crowd-sourced instruction data.
- **Format consistency** — inconsistent formatting produces inconsistent output formatting.

**Practical pipeline for building your own:**

1. **Start from real usage** — actual user queries beat imagined ones, always. Logs are the best source you have.
2. **Generate candidate responses** with a strong model (Week 13's synthetic approach), using high temperature for diversity (S7).
3. **Filter hard.** Rule-based checks, model-as-judge scoring (S3, with its caveats), and execution-based verification where possible — for code, *run it*; for maths, *check it*. Verifiable filtering is the strongest signal available and connects directly to Week 15's RLVR logic.
4. **Deduplicate**, including semantically (§3). Synthetic data collapses toward the generator's favourite phrasings, and dedup is your main defence.
5. **Human-review a sample.** You cannot skip this. Read 100 examples end to end before training on 10,000 — it is the single highest-value hour in the whole process, and it catches systematic problems no automatic filter will.
6. **Hold out a clean eval set before you start** (S3), and never let it touch training.

**On synthetic data specifically:** it's genuinely effective and now standard, but it has a characteristic failure — **diversity collapse**. Left unchecked, a generator produces variations on a theme. Counter it with varied seed prompts, high temperature, explicit diversity instructions, multiple generator models, and aggressive semantic dedup. Also be aware of the **model-collapse** concern — training repeatedly on your own outputs degrades quality over generations — which argues for keeping real human data in the mix rather than going fully synthetic.

---

## 9. Preference data (Week 14, revisited)

Preference pairs — prompt with a chosen and a rejected response — are harder to produce well than SFT data.

- **Hard pairs teach most.** Week 14's own point, and S6's: when the rejected response is genuinely good and loses on something subtle, the model learns a real boundary. When it's obviously terrible, the model learns nothing it didn't know.
- **Annotator agreement is the ceiling.** If humans agree only 65% of the time on which response is better, no amount of data pushes the model past that noise floor. **Measure inter-annotator agreement before scaling annotation** — it tells you whether your task is well-defined at all.
- **Length bias is pervasive.** Annotators prefer longer responses; models learn "longer is better" and become verbose. This is one of the most reliably observed pathologies in preference tuning, and it needs explicit control.
- **AI feedback (RLAIF)** scales far better than human labelling and works reasonably, but it inherits the judge model's biases — including its own length bias (S3's judge caveats apply in full).

---

## 10. Licensing, provenance, and PII

The part that gets skipped and then becomes a problem.

**Licensing.** Web data is not automatically usable. Code licences (GPL, AGPL) carry obligations. Book corpora have been the subject of active litigation. The legal position on training data is genuinely unsettled and varies by jurisdiction — this lecture is not legal advice, and "everyone does it" is not a defence.

**Provenance.** Track where every part of your dataset came from. If a source must be removed later — licence dispute, takedown request, discovered contamination — you need to know what to remove and what to retrain. Datasets without provenance cannot be remediated.

**PII.** Web-scraped data contains names, emails, phone numbers, addresses, and credentials. Standard practice is detection and redaction before training. This matters more than it sounds because of §3: **memorised data can be extracted from a trained model**, and duplication makes memorisation likelier. Deduplication is therefore also a privacy control.

**Consent and opt-out.** Increasingly a legal requirement in some jurisdictions rather than a courtesy, and `robots.txt` and similar signals are now widely respected by responsible crawlers.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 3** (tokenization) | Tokenizers are trained on a corpus and inherit its distribution — which is why they determine what the model is bad at. |
| **Week 7** (training a model) | The MiniLLM's corpus was handed to you. This is what producing one actually involves. |
| **Week 12** (LoRA/CPT) | Continued pretraining is a data problem: which domain corpus, how much, mixed with how much general data to avoid forgetting. |
| **Week 13** (SFT, synthetic data) | This lecture is the quality framework around that technique — filtering, dedup, diversity collapse, and human review. |
| **Week 14** (RLHF/DPO) | Preference data quality, annotator agreement as a ceiling, and length bias. |
| **Week 15** (RLVR) | Execution-based verification is the strongest data filter available — the same insight as verifiable rewards. |
| **Week 20** (reading papers) | Documented data pipelines (FineWeb, The Pile) are where you learn this; undocumented ones are where scepticism belongs. |
| **S3** (evaluation) | Contamination is data leakage at corpus scale. Held-out private evals are the only trustworthy measurement. |
| **S6** (embeddings) | MinHash+LSH and ANN indexes share the "avoid N² comparison" insight; semantic dedup uses embeddings directly. |
| **S10** (scaling laws) | Chinchilla demands trillions of tokens — this is where they come from and why filtering matters at that scale. |

---

## Key takeaways

1. **A model is a compression of its data.** Most of the quality difference between models of the same size is data work, not architecture.
2. **The three stages have opposite rules.** Pretraining is filtering at scale; instruction tuning is curation at small scale; preference tuning is elicitation. "More is better" is true for the first and false for the second.
3. **Deduplication improves models at fixed compute** — less data, better model. Not a trade-off, a strictly better outcome.
4. **MinHash + LSH** is the standard fuzzy-dedup technique, and it's the same "don't compare everything to everything" insight as ANN search.
5. **"Quality" encodes an opinion.** Every filter shapes model behaviour — including toxicity filtering reducing toxicity *recognition*. Ablate filters; don't adopt them because they sound sensible.
6. **Perplexity filtering wants the middle**, not the low end — boilerplate is extremely predictable.
7. **Contamination is nearly guaranteed at web scale.** Public benchmark scores from undocumented pipelines deserve scepticism; private held-out evals are the only measurement you can trust.
8. **Mixture and order matter** — code improves general reasoning, and annealing on high-quality data at the end has outsized influence.
9. **The tokenizer determines what your model is bad at** — arithmetic, non-English languages, code — and cannot be changed after pretraining.
10. **For instruction data, quality beats quantity decisively.** A thousand excellent, diverse, consistent examples beat a hundred thousand mediocre ones, because SFT is imitation learning.
11. **Synthetic data collapses toward the generator's habits.** Vary seeds, raise temperature, use multiple generators, and dedup semantically.
12. **Read 100 examples by hand** before training on 10,000. Nothing automated substitutes for it.
13. **Track provenance.** A dataset you can't audit is a dataset you can't remediate.

---

## Glossary

| Term | Definition |
|---|---|
| **Common Crawl** | Nonprofit web crawl, the backbone of most pretraining corpora |
| **C4 / The Pile / RedPajama / FineWeb** | Documented open corpora; FineWeb notably publishes filtering ablations |
| **Text extraction** | Stripping HTML boilerplate, navigation, and ads to recover content |
| **Exact dedup** | Hash-and-drop identical documents |
| **MinHash** | Compact document signature whose collision probability equals Jaccard similarity |
| **LSH** | Locality-Sensitive Hashing; buckets similar signatures to avoid N² comparison |
| **Semantic dedup** | Removing near-duplicates in embedding space |
| **Heuristic filter** | Rule-based quality filter (length, punctuation, symbol ratios) |
| **Classifier filter** | Small model scoring documents against a reference "good text" set |
| **Perplexity filter** | Dropping documents a small LM finds too surprising *or* too predictable |
| **Contamination** | Benchmark/test data present in training data |
| **Decontamination** | Removing n-gram overlaps with known benchmarks |
| **Data mixture** | Weighted blend of domains (web, code, books, papers) |
| **Upsampling** | Repeating high-quality sources more than once per epoch |
| **Annealing** | Final training phase on a much higher-quality subset |
| **BPE** | Byte-Pair Encoding; iteratively merges frequent adjacent pairs |
| **Token fragmentation** | One concept splitting into many tokens; penalises non-English text and rare terms |
| **LIMA** | "Less Is More for Alignment" — 1,000 curated examples competitive with far larger sets |
| **Diversity collapse** | Synthetic generators converging on a narrow range of phrasings |
| **Model collapse** | Degradation from repeatedly training on model-generated data |
| **Inter-annotator agreement** | How often labellers agree; a hard ceiling on achievable model quality |
| **Provenance** | Records of where each part of a dataset came from; required for remediation |

# 100xSchool AI & ML — Course Archive

Complete offline archive of the [100xSchool AI & ML bootcamp](https://100xschool.notion.site/AI-ML-2e3ffffa33e5806ba537f3c3965fe0c0), with detailed notes and a 20-question quiz for every lecture — plus 13 supplementary lectures written to fill gaps the course leaves open, a cumulative final exam, a master glossary, and a decision cheat sheet.

**Instructor:** Rishabh — Research Engineer (ex-AI Lab London, LLM & LLM-RL) · [@rishabh10x](https://twitter.com/rishabh10x)
**Span:** 09/01/2026 – 08/07/2026 · 26 weeks (Week 19 cancelled) · 25 lectures with content

```
AI-ML/
├── README.md           this file — index and reading order
├── CHEATSHEET.md       decision tables: prompt vs RAG vs fine-tune, cost, safety, debugging
├── GLOSSARY.md         every term across all 38 lectures, A–Z, with pointers
├── FINAL-EXAM.md       50 cumulative questions + 6 design problems + full key
├── downloads/          42 files — every slide PDF + Colab notebook (41 MB)
│   └── MANIFEST.md     what's here, mapped to weeks, with known source bugs
├── notes/
│   └── week-NN-slug/
│       ├── notes.md    detailed lecture notes
│       └── quiz.md     20 questions + answer key
└── supplementary/      13 additional lectures (not part of the course)
    └── sNN-slug/
        ├── notes.md
        └── quiz.md
```

**38 lectures total** — 25 from the course, 13 supplementary — each with notes and a 20-question quiz. **760 quiz questions**, plus a 50-question cumulative exam.

**Every lecture follows the same structure:** a plain-language "the idea in plain language" opening, numbered sections building depth, a table mapping the material back to the course, numbered key takeaways, and a glossary. Quizzes are 12 multiple choice + 8 short-answer/design, with a full answer key.

**New here?** Start with [`CHEATSHEET.md`](CHEATSHEET.md) for the practical decisions, then work through `notes/` in week order using the [reading order](#suggested-reading-order) below.

---

## The four arcs

| Weeks | Arc | What it builds |
|---|---|---|
| **0–7** | **Fundamentals** | How models actually work — from neurons to a working LLM |
| **8–11** | **Building with models** | Agents and retrieval |
| **12–16** | **Adapting models** | Fine-tuning and reinforcement learning |
| **17–25** | **Engineering discipline** | Making it reliable, observable, and shippable |

---

## Lectures

### Fundamentals

| # | Lecture | Date | Materials |
|---|---|---|---|
| 00 | [Orientation](notes/week-00-orientation/) | 09/01 | pdf |
| 01 | [Fast-Tracking the Course of AI](notes/week-01-fast-tracking-ai/) | 18/01 | pdf |
| 02 | [Neural Networks from Scratch](notes/week-02-neural-networks-from-scratch/) | 24/01 | pdf + nb |
| 03 | [Transformers Pt 1 — Tokens, Embeddings, Position](notes/week-03-transformers-part-1/) | 31/01 | pdf + nb |
| 04 | [Transformers Pt 2 — Attention, KV Cache, Multi-Head](notes/week-04-transformers-part-2/) | 06/02 | pdf + nb |
| 05 | [Tensors and PyTorch](notes/week-05-tensors-and-pytorch/) | 14/02 | pdf + nb |
| 06 | [What Changed Since 2017 — RMSNorm, SwiGLU, RoPE, GQA](notes/week-06-what-changed-since-2017/) | 21/02 | pdf |
| 07 | [Training Your First Model — MiniLLM end to end](notes/week-07-training-your-first-model/) | 28/02 | pdf + nb |

### Building with models

| # | Lecture | Date | Materials |
|---|---|---|---|
| 08 | [From APIs to Agents](notes/week-08-from-apis-to-agents/) | 07/03 | pdf + nb |
| 09 | [RAG from the Ground Up, Pt 1](notes/week-09-rag-part-1/) | 13/03 | pdf + nb |
| 10 | [Why RAG Breaks — Context Rot & PageIndex](notes/week-10-rag-part-2/) | 23/03 | pdf + nb |
| 11 | [Recursive Language Models · Claude Code vs Cursor](notes/week-11-recursive-language-model/) | 28/03 | pdf + nb |

### Adapting models

| # | Lecture | Date | Materials |
|---|---|---|---|
| 12 | [Fine-tuning Pt 1 — LoRA, QLoRA, CPT](notes/week-12-fine-tuning-part-1/) | 04/04 | pdf + nb |
| 13 | [Fine-tuning Pt 2 — SFT](notes/week-13-fine-tuning-part-2/) | 11/04 | pdf + nb |
| 14 | [Fine-tuning Pt 3 — RLHF & DPO](notes/week-14-fine-tuning-part-3/) | 17/04 | pdf + nb |
| 15 | [RLVR — Reasoning Models (o1 → R1, GRPO)](notes/week-15-rlvr/) | 25/04 | pdf + nb |
| 16 | [RL Environments for LLMs](notes/week-16-rl-environments-for-llms/) | 08/05 | pdf + nb |

### Engineering discipline

| # | Lecture | Date | Materials |
|---|---|---|---|
| 17 | [Harness, Context, and Evals](notes/week-17-harness-context-evals/) | 09/05 | pdf + nb |
| 18 | [Memory](notes/week-18-memory/) | 16/05 | pdf |
| — | *Week 19 — cancelled* | — | — |
| 20 | [How to Read Research Papers](notes/week-20-how-to-read-research-papers/) | 31/05 | pdf |
| 21 | [LangGraph — Agents as State Machines](notes/week-21-langgraph/) | 06/06 | pdf + nb |
| 22 | [Coding an Agent (Assignment)](notes/week-22-coding-an-agent-assignment/) | 13/06\* | links only |
| 23 | [Hugging Face End-to-End](notes/week-23-hugging-face-end-to-end/) | 20/06 | pdf |
| 24 | [LLM Observability](notes/week-24-llm-observability/) | 01/07 | pdf + nb |
| 25 | [Computer Use Agents](notes/week-25-computer-use-agents/) | 08/07 | pdf + nb |

\* *The Notion page lists 13/12/2026, almost certainly a typo — it sits between Weeks 21 and 23, and the linked repo is named `13-june-assignment`.*

---

## Supplementary lectures

The course is strong on architecture, fine-tuning, and RL, and thinner on the engineering discipline around them. These 8 lectures fill gaps that a student would otherwise hit in a job or an interview. **They are not part of the 100xSchool curriculum** — every one is marked as supplementary at the top of its notes, and each states which course week it follows and what it assumes.

### Tier 1 — fills a gap that affects everything else

| # | Lecture | Fills the gap after | Why it matters |
|---|---|---|---|
| S1 | [Classical ML and Tabular Data](supplementary/s01-classical-ml-and-tabular/) | Week 2 | The course starts at neural networks. Most real-world ML problems are tabular, and gradient boosting still wins them. Also covers bias–variance, regularisation, and why "just use an LLM" is often the wrong answer. |
| S2 | [Prompt Engineering](supplementary/s02-prompt-engineering/) | Week 8 | The course uses prompts constantly and never teaches them. Structure, examples, decomposition, chain-of-thought and where it stopped helping, and prompts as versioned artefacts. |
| S3 | [Evaluation and Statistics](supplementary/s03-evaluation-and-statistics/) | Week 17 | Week 17 covers evals; this covers whether your eval result means anything. Data leakage, bootstrap CIs, McNemar's test, calibration, sample sizes, and why a 2-point gain on 100 examples is noise. |
| S4 | [Safety, Jailbreaks, and Guardrails](supplementary/s04-safety-jailbreaks-guardrails/) | Week 8 | Prompt injection, the lethal trifecta, indirect injection through tool results, guardrail architecture, and why "ignore previous instructions" is the least interesting attack. |

### Tier 2 — depth on things the course touches briefly

| # | Lecture | Fills the gap after | Why it matters |
|---|---|---|---|
| S5 | [Inference Optimisation and Cost](supplementary/s05-inference-optimization-and-cost/) | Week 17 | Prompt caching, continuous batching, PagedAttention, speculative decoding, quantisation, and the TTFT/TPOT/goodput vocabulary you need to reason about a production LLM bill. |
| S6 | [Embeddings, Properly](supplementary/s06-embeddings-deep-dive/) | Week 9 | Contrastive learning, hard negatives, why the bi-/cross-encoder split is forced, Matryoshka embeddings, ANN indexes, and mechanistic explanations for Week 9's observed RAG failures. |
| S7 | [Sampling and Decoding](supplementary/s07-sampling-and-decoding/) | Week 7 | What happens between logits and printed tokens: temperature, top-k/top-p/min-p, penalties, beam search, constrained decoding — and why frontier Claude models have removed sampling parameters entirely. |
| S8 | [Model Context Protocol](supplementary/s08-model-context-protocol/) | Week 8 | How tools get distributed between systems. The N×M problem, host/client/server, the three primitives, and why an MCP server is a trust boundary. |

### Tier 3 — the rest of the picture

| # | Lecture | Fills the gap after | Why it matters |
|---|---|---|---|
| S9 | [Multimodal Models: How a Model Sees](supplementary/s09-multimodal-and-vision/) | Week 25 | The course is text-only, yet Week 25's computer-use agents work by looking at screenshots. Patches as tokens, CLIP, why images cost what they cost, and the failure list — counting, small text, charts, coordinates. |
| S10 | [Scale: MoE, Distributed Training, Scaling Laws](supplementary/s10-scale-moe-distributed/) | Week 7 | How the models you call are actually built. Chinchilla, why nobody is compute-optimal any more, the memory arithmetic that explains QLoRA, ZeRO/FSDP, and **Mixture-of-Experts** — absent from Week 6's list of what changed since 2017. |
| S11 | [Data: Curation, Quality, and Tokenizers](supplementary/s11-data-curation/) | Week 13 | Where corpora come from, why deduplication improves models *at fixed compute*, why "quality" encodes an opinion, and how the tokenizer quietly decides what your model is bad at. |
| S12 | [Multi-Agent Systems](supplementary/s12-multi-agent-systems/) | Week 21 | The first thing people try after one agent works — and where a lot of projects quietly get worse. The three valid reasons, the context tax, and why shallow beats deep. |
| S13 | [World Models](supplementary/s13-world-models/) | Week 16 | Models that predict *what happens*, not what token comes next. Model-based RL, Dreamer, MuZero, compounding error, and a careful answer to "do LLMs have world models?" |

---

## Threads that run through the whole course

Several ideas recur across many weeks. Following one end to end is a good way to revise.

**The learning loop** — *try → feedback → adjust → repeat* appears as the definition of learning (W1), the training loop (W2, W7), the agent loop (W8), and the RL loop (W14, W15).

**Context is finite** — attention as a fixed budget (W4) → context rot measured (W10) → the attention-budget principle and JIT context (W17) → memory architectures (W18).

**Giving the model control of its own context** — passive RAG (W9) → smarter retrieval (W10) → reasoning-based navigation (W10) → active exploration via code (W11).

**Knowledge vs behaviour** — RAG adds knowledge, fine-tuning changes behaviour (W12); the full decision table appears in W12 and W14, extended by RLVR in W15.

**Goodhart's Law** — reward hacking in RLHF (W14), surviving even deterministic verifiers (W15), and LLM-judge rubrics (W16, W17).

**Build it badly first** — NumPy backprop before PyTorch (W2), a raw ReAct loop before LangGraph (W8 → W21), hand-rolled RAG before frameworks (W9), a deliberately failing GRPO run (W15), and a hand-rolled computer-use agent before the native tool (W25). W25's closing line names the principle: *"The API is shaped by the pain of Stage 3."*

**Scepticism about numbers** — vendor-reported benchmarks (W18), single-benchmark claims and missing baselines (W20), `pass@k` vs `pass^k` (W17), and reading the OSWorld footnote (W25).

---

## Notes on the source material

The notes flag issues found in the original materials rather than reproducing them silently. The main ones:

- **W2** — the shipped notebook's training output starts at loss 0.000265 instead of ~0.25; cells were re-run without re-running the reset cell. Restart and run all to see the real curve.
- **W3** — the notebook promises "attention coded from scratch" but ends before it; that content is in **W4's** notebook (Part 4).
- **W4** — the deck repeats W3's slides 1–23; new material starts at slide 24.
- **W5** — the gradient-accumulation demo prints 6.0/6.0/6.0, contradicting its own caption, because `zero_grad()` is called before the second backward.
- **W8** — `search_wikipedia` fails on valid queries, most likely a missing `User-Agent` header.
- **W9** — hybrid search ranked the RS256 chunk *lower* than naive semantic search did; the caption overstates the result.
- **W11** — the cited RLM arXiv ID doesn't resolve to a real paper; verify before citing.
- **W10** — the FinanceBench comparison is against *naive* RAG, not the advanced pipeline built in W9.

---

## How to use the quizzes

Each `quiz.md` has 12 multiple-choice questions, 8 short-answer/design questions, and a full answer key at the bottom. Cover the key and attempt the whole set first — the short-answer questions are where the actual understanding gets tested, and several ask you to critique claims made in the lectures rather than recall them.

## Suggested reading order

Work through `notes/` in week order. The supplementary lectures are written to be read **at the point where they fill the gap**, and each names its prerequisites:

| After… | Read | Why there |
|---|---|---|
| Week 2 | **S1** — Classical ML | Before you assume neural networks are the answer |
| Week 7 | **S7** — Sampling · **S10** — Scale | W7 mentions temperature in one line; W7 trains one model on one GPU |
| Week 8 | **S2** — Prompting · **S4** — Safety · **S8** — MCP | You now have an agent that calls tools and reads untrusted content |
| Week 9 | **S6** — Embeddings | Explains the RAG failures W9 demonstrated but didn't fully diagnose |
| Week 13 | **S11** — Data | W13 generates synthetic data; this is the quality framework around it |
| Week 16 | **S13** — World models | W16 builds RL environments — which *are* hand-built world models |
| Week 17 | **S3** — Evaluation · **S5** — Inference cost | W17 covers evals; S3 covers whether the result means anything |
| Week 21 | **S12** — Multi-agent | Right after you learn to orchestrate one agent |
| Week 25 | **S9** — Vision | The missing foundation under computer-use agents |

Reading them in `s01…s13` order works too; the placement above just means you meet each idea when the course has already given you a reason to care.

**Then:** [`FINAL-EXAM.md`](FINAL-EXAM.md) — 50 cumulative questions that require connecting lectures, including a "what's wrong with this plan?" section and six interview-style design problems.

---

## The reference files

| File | What it's for | When you'd reach for it |
|---|---|---|
| [`CHEATSHEET.md`](CHEATSHEET.md) | 14 decision tables — prompt vs RAG vs fine-tune, fixing RAG in priority order, agent vs workflow, decoding settings, cost levers, OOM ladder, safety checklist, universal debugging order | On the job, when you need the answer not the argument |
| [`GLOSSARY.md`](GLOSSARY.md) | Every term across all 38 lectures, A–Z, each pointing to the lecture that explains it | Revision, or when a term appears with no context |
| [`FINAL-EXAM.md`](FINAL-EXAM.md) | 50 cross-cutting questions + 6 design problems, full answer key | After finishing everything, to find out what didn't stick |
| [`downloads/MANIFEST.md`](downloads/MANIFEST.md) | The 42 source files mapped to weeks, with sizes and the known bugs in each | Finding the original slides or notebook for a lecture |

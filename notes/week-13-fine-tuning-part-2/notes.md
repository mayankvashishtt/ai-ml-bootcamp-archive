# Week 13 — Fine-tuning Part 2: Supervised Fine-Tuning (SFT)

**Subtitle:** From Autocompleter to Assistant — SmolLM-135M + Financial Q&A + Unsloth
**Date:** 11/04/2026
**Sources:** `downloads/week-13-fine-tuning-part-2.pdf` (27 slides) · `downloads/week-13-fine-tuning-part-2.ipynb` (45 cells)
**Notion page:** https://100xschool.notion.site/341ffffa33e58039a986c189d898e9e7
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=yMBhyaNkIWOQkPBly4EYA,aN9ukNsNwWCvy0cQeYLpTg)

---

## Where we are

```
Base Model (next-token predictor)
  ├── CPT       [DONE — Week 12]     Taught domain language
  ├── SFT       [TODAY]              Teaches instruction following
  ├── RLHF/DPO  [Week 14/15]         Teaches human preferences
  └── RLVR      [Week 15]            Teaches reasoning
```

---

## 1. The problem SFT solves

**Prompt:** *"What is EBITDA?"*

| After CPT | After SFT |
|---|---|
| *"What is EBITDA? EBITDA is a measure of a company's operating performance. EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. EBITDA is used by analysts to compare…"* | *"EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It measures a company's operating profitability by excluding non-operating expenses. Investors use it to compare companies across different tax and capital structures."* |
| → **CONTINUES.** Doesn't know when to stop.<br>→ Doesn't know this is a **QUESTION**. | → Concise. Answers the question. **Stops.** |

> **SFT teaches: when you see an instruction, produce a response. When you're done responding, stop.**

Note that the CPT model's *content* isn't wrong — it knows what EBITDA is. What it lacks is the **behavioural frame**: recognising a question as a question, and recognising completion.

---

## 2. What IS SFT?

> **SFT = train the model on (instruction, response) pairs.**

Same training loop as CPT — forward → loss → backward → update. **Only two things differ:**

**1. The data format**
```
CPT: "raw text raw text raw text..."
SFT: "### Instruction: Q ### Response: A"
```

**2. Where the loss is computed**
```
CPT: loss on ALL tokens
SFT: loss ONLY on response tokens
```

That second point is the technically distinctive part of SFT and is covered in §5.

---

## 3. Why SFT works with so few examples

### The Superficial Alignment Hypothesis (LIMA paper, 2023)

> *"A model's knowledge and capabilities are learned almost entirely during pre-training. Alignment teaches it the FORMAT of interaction."*

| Stage | Scale | What it produces |
|---|---|---|
| **Pre-training** | Trillions of tokens | Knowledge + capability |
| **SFT** | Thousands of examples | Behaviour pattern |

> The model **ALREADY knows** how to answer questions — it saw millions of Q&A pages during pre-training. SFT just teaches it: *"when you see THIS template, activate THAT existing capability."*
>
> **It's not learning new skills. It's learning the UI.**
>
> That's why **1,000–10,000 examples suffice.** You're teaching the interface, not the intelligence.

This explains the otherwise baffling economics: how can 1,000 examples meaningfully change an 8-billion-parameter model? Because they aren't teaching it anything — they're **selecting** among behaviours it already has.

---

## 4. Data formats

### The Alpaca format

```json
{
  "instruction": "What is EBITDA?",
  "input": "",
  "output": "EBITDA stands for Earnings Before Interest, Taxes..."
}
```

Rendered into a training prompt:

```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
What is EBITDA?

### Response:
EBITDA stands for Earnings...{EOS}
```

> **The EOS token at the end is critical — that's how the model learns when to stop.**

Three fields: `instruction`, `input`, `output`. Named after Stanford's Alpaca project.

### Chat template format (alternative)

```json
{"messages": [
  {"role": "system",    "content": "You are a financial analyst."},
  {"role": "user",      "content": "What is EBITDA?"},
  {"role": "assistant", "content": "EBITDA stands for..."}
]}
```

**Each model has its OWN special tokens:**

| Model | Tokens |
|---|---|
| SmolLM | `<\|im_start\|>user\n...<\|im_end\|>` |
| Llama 3 | `<\|start_header_id\|>user<\|end_header_id\|>...<\|eot_id\|>` |
| Qwen | `<\|im_start\|>user\n...<\|im_end\|>` |

> ⚠️ **NEVER format these manually.** Use `tokenizer.apply_chat_template(messages)`.

A single wrong special token silently degrades everything — the model sees an unfamiliar frame and falls back to autocomplete. The tokenizer ships the correct template; use it.

---

## 5. Loss masking — the key difference from CPT

Training text: `"### Instruction: What is EBITDA? ### Response: EBITDA stands for..."`

**CPT loss (all tokens):**
```
[###] [Inst] [:] [What] [is] [EB] [IT] [DA] [?] [###] [Resp] [:] [EB] [IT]...
LOSS  LOSS  LOSS LOSS  LOSS LOSS LOSS LOSS LOSS LOSS  LOSS  LOSS LOSS LOSS
```

**SFT loss (response only):**
```
[###] [Inst] [:] [What] [is] [EB] [IT] [DA] [?] [###] [Resp] [:] [EB] [IT]...
-100  -100  -100 -100  -100 -100 -100 -100 -100 -100  -100  -100 LOSS LOSS
```

> **-100 = "ignore this token in loss computation."** PyTorch's cross-entropy skips any label that's −100.

```python
text = "### Instruction:\nWhat is revenue?\n### Response:\nRevenue is...<EOS>"
input_ids = tokenizer(text).input_ids
labels = input_ids.clone()

response_start = find_response_start(input_ids)
labels[:response_start] = -100          # mask everything BEFORE the response

loss = cross_entropy(logits, labels)    # loss only where labels != -100
```

Unsloth's `SFTTrainer` with the Alpaca template does this automatically.

**Why mask?** You want the model to learn *how to respond given a question*, not *how to generate questions*. Computing loss on the prompt would spend capacity modelling the input distribution — which you don't control at inference, since the user supplies it. Masking focuses every gradient on the only thing the model actually produces.

---

## 6. Packing with multiple SFT examples

```
Without packing (wasteful):
  Seq 1: [Q&A pair 1] [PAD] [PAD] [PAD] [PAD] [PAD]
  Seq 2: [Q&A pair 2] [PAD] [PAD] [PAD] [PAD]

With packing (efficient):
  Seq 1: [Q&A pair 1] [EOS] [Q&A pair 2] [EOS] [Q&A pair 3]
```

**"But doesn't pair 2 attend to pair 1?"**

> **No — the trainer uses ATTENTION MASKING.** Each packed example only attends to itself; EOS tokens act as boundaries; no cross-contamination. Handled automatically by Unsloth and TRL.

This is the causal mask from Week 7, extended: instead of a single lower-triangular mask, it's **block-diagonal** — each example gets its own triangle and cannot see across the boundary.

---

## 7. The EOS token — teaching the model to stop

| Without EOS training | With EOS training |
|---|---|
| *"Revenue is the total income generated from business operations. Revenue is also known as sales or turnover. Revenue can be calculated…"* → **Never stops.** | *"Revenue is the total income a company generates from its business operations."* `[EOS]` → **Stops when done.** |

> **Every training example ends with EOS.** The model learns: complete response, then STOP.

Stopping is a *learned behaviour*, not a built-in one. A model that never saw EOS in a response position has no reason to ever emit it — generation only ends when you hit `max_tokens`.

---

## 8. LoRA config — CPT vs SFT

| | CPT | SFT |
|---|---|---|
| **rank (r)** | 32 | **16** |
| **target_modules** | all linear + `embed_tokens` + `lm_head` | all linear (**no** `embed_tokens`, **no** `lm_head`) |
| **lora_alpha** | 32 | **16** |
| **learning_rate** | 2e-4 | 2e-4 |
| **embedding_lr** | 2e-5 | not needed |

> **Why lower rank?** SFT = behaviour change → **fewer directions of change**.
> **Why no embed/lm_head?** Vocabulary is fine; **behaviour** needs to change.

This is the practical payoff of Week 12's `embed_tokens` discussion, stated as a config diff.

---

## 9. The dataset

**`virattt/financial-qa-10K`** — 7,000 financial Q&A pairs from SEC 10-K filings.

```json
{
  "question": "What was the total revenue for FY2023?",
  "context":  "The company reported total revenues of $53.8 billion for the fiscal year ended...",
  "answer":   "The total revenue for FY2023 was $53.8 billion."
}
```

Converted to Alpaca: `instruction=question`, `input=context`, `output=answer`.

### Data quality >> quantity

- GPT-3.5 was SFT'd on **~12,000 examples**
- LIMA paper: **1,000 good > 50,000 noisy**
- 7,000 high-quality pairs is plenty

---

## 10. Inference and deployment

**What happens at inference:**
1. User provides `"### Instruction:\nWhat is EBITDA?\n### Response:\n"`
2. Model sees the familiar template structure
3. Model generates tokens one by one
4. Model outputs **EOS**
5. Generation stops

> **The template is the INTERFACE between human and model.**

### Merging adapters

| Option | Pro | Con |
|---|---|---|
| **1. Keep separate** — load base + adapter at runtime | Swap adapters per request | Slightly more complex |
| **2. Merge** — `W_new = W + (α/r)·B·A` | Single model, zero overhead | Can't unmix |
| **3. Push to Hub** | Share with the world, one line to use | — |

```python
model.merge_and_unload()                            # merge
model.push_to_hub("your-name/financial-qa-smollm")  # share
```

### The full pipeline in practice

```
1. Base model (SmolLM, Llama, Qwen)
2. CPT on domain text (raw SEC filings)        → learns financial language
3. SFT on instruction-response pairs           → learns to answer
4. (Optional) DPO/RLHF on preference data      → learns human preferences
5. Merge all adapters
6. Deploy with RAG for real-time knowledge
```

> **CPT without SFT** = sounds smart, can't help.
> **SFT without CPT** = follows instructions, limited domain knowledge.
> **CPT + SFT** = follows instructions WITH domain expertise.

Note step 6: **fine-tuning and RAG are complementary, not alternatives.** CPT+SFT gives the model the language and behaviour; RAG supplies facts that change.

---

## 11. The real skill — creating instruction data from raw text

### Why public datasets don't work

Alpaca, ShareGPT, UltraChat are **generic, stale, wrong domain**, and variable in quality.

**What you actually need:** YOUR domain's questions · YOUR format requirements · YOUR accuracy standards · **updated when your domain changes.**

> **Solution: use an LLM to GENERATE instruction data from YOUR raw text.**

### The synthetic data pipeline

```
Raw domain text (SEC filings, docs, wikis)
        ↓
Chunk into passages (500 words)
        ↓
For each chunk, ask an LLM to generate:
  - Q&A pairs (factual, analytical, synthesis)
  - Summaries
  - Unanswerable questions (teaches refusal!)
        ↓
Quality filter (remove bad examples)
        ↓
Format to Alpaca template + EOS
        ↓
Train with SFT
```

> **A strong model runs once for data generation. A small model runs cheaply in production forever.**

That's the economic logic of distillation — pay for intelligence once at data-creation time, then serve something small.

### Prompt engineering for data generation

| Bad prompt | Good prompt |
|---|---|
| *"Generate questions from this text."* | *"Generate 4 diverse Q&A pairs:*<br>*1. Factual (who/what/how much)*<br>*2. Analytical (why/compare)*<br>*3. Synthesis (summarize)*<br>*4. **Unanswerable** (context lacks)*<br>*Answers cite specific numbers."* |
| → Shallow, repetitive, yes/no questions | → Diverse, deep, grounded |

> **Prompt quality → Data quality → Model quality.**

The **unanswerable** category is the clever part — you're deliberately manufacturing training examples for *refusal*, which is the fix for failure mode 2 below.

### Quality control

**GOOD:**
> Q: *"What was the year-over-year change in operating expenses?"*
> A: *"Operating expenses increased $1.1B, a 15.5% YoY increase."*
> `[specific] [accurate] [cites numbers] [concise]`

**BAD:** answer contradicts the context (hallucinated numbers) · answer is the context copy-pasted · question unanswerable from the context · three paragraphs for a simple fact · **missing EOS token** · inconsistent format.

> **Always sample 50–100 examples and read them manually. 1 hour auditing saves 10 hours debugging a broken model.**

---

## 12. Failure modes

### Failure Mode 1 — Sycophancy

**User:** *"Revenue was $100 billion, right?"* — **Context says:** revenue was $53 billion.

| Sycophantic | Correct |
|---|---|
| *"Yes, you're right! Revenue was ~$100 billion."* | *"Actually, revenue was $53 billion, not $100 billion."* |

**WHY:** training data has too many "agreeable" answers.
**FIX:** include examples where **the model corrects the user.**

### Failure Mode 2 — Hallucination amplification

**Q:** *"What was the employee count?"* — **Context:** only revenue/income data, **no employee info**.

| Bad model | Good model |
|---|---|
| *"The company had approximately 47,000 employees."* (completely made up) | *"The provided context does not contain information about employee count."* |

**WHY: SFT teaches confident answering. Not "I don't know."**
**FIX:** include **unanswerable** examples in training data.

> Critical for finance, where making up numbers has legal consequences.

This is a genuinely important insight: SFT can make hallucination *worse*. Every training example pairs a question with a confident answer, so the model learns "questions get answers" as an unconditional rule. Refusal must be explicitly taught.

### Failure Mode 3 — Format overfitting

| With template | Without template |
|---|---|
| `"### Instruction:\nWhat is EBITDA?\n### Response:\n"` → *"EBITDA stands for…"* ✅ | `"What is EBITDA?"` → random autocomplete ❌ |

**WHY:** the model learned **ONE exact template**, not "answering questions."
**FIX:** train on **MULTIPLE prompt formats** — mix Alpaca + raw questions + chat template.

---

## 13. The alignment tax

| Before SFT | After SFT on financial Q&A |
|---|---|
| Decent at text completion, translation, code, creative writing | **Great** at financial Q&A. **Worse** at code. **Worse** at creative writing. |

> **The model "forgets" behaviours not in the SFT dataset.**
>
> **Mitigation:** mix domain-specific + general instruction data. **80% domain, 20% general** — same principle as CPT mixing.

Catastrophic forgetting (Week 12) applies to *behaviours* as well as *knowledge*, and the remedy is identical: give the optimizer a reason to preserve what you care about.

---

## Key takeaways

1. **SFT turns an autocompleter into an assistant** — recognise an instruction, respond, stop.
2. **Same loop as CPT; only the data format and loss masking differ.**
3. **Loss is computed on response tokens only** (`-100` masks the prompt) — learn how to answer, not how to ask.
4. **The Superficial Alignment Hypothesis** explains why 1,000–10,000 examples suffice: you're teaching the interface, not the intelligence.
5. **EOS is how the model learns to stop** — every example must end with it.
6. **Never hand-format chat templates**; use `tokenizer.apply_chat_template`.
7. **Packing uses block-diagonal attention masking** so examples can't see each other.
8. **SFT uses lower rank (16) and no `embed_tokens`/`lm_head`** — behaviour change, not vocabulary change.
9. **Quality beats quantity** — 1,000 good > 50,000 noisy.
10. **Generate your own instruction data** from raw text with a strong model; a small model then serves cheaply forever.
11. **Deliberately include unanswerable examples** — SFT otherwise amplifies hallucination.
12. **Train on multiple formats** to avoid format overfitting.
13. **The alignment tax is real** — mix 20% general instruction data.

---

## Glossary

| Term | Meaning |
|---|---|
| **SFT** | Supervised Fine-Tuning on (instruction, response) pairs. |
| **Alpaca format** | `{instruction, input, output}` rendered into a `### Instruction / ### Response` prompt. |
| **Chat template** | Model-specific special-token format for system/user/assistant turns. |
| **`apply_chat_template`** | Tokenizer method producing the model's correct chat formatting. |
| **Loss masking** | Setting prompt-token labels to −100 so loss is computed only on the response. |
| **−100** | PyTorch's ignore-index for cross-entropy. |
| **EOS token** | End-of-sequence marker that teaches the model to stop. |
| **Packing** | Concatenating examples into one sequence to remove padding waste. |
| **Attention masking (packing)** | Block-diagonal mask keeping packed examples from attending to each other. |
| **Superficial Alignment Hypothesis** | LIMA's claim that alignment teaches interaction format, not capability. |
| **Synthetic data generation** | Using a strong LLM to create instruction data from raw domain text. |
| **Unanswerable example** | A training pair whose correct response is a refusal — teaches "I don't know." |
| **Sycophancy** | Agreeing with an incorrect user assertion. |
| **Hallucination amplification** | SFT increasing confident fabrication by only ever modelling confident answers. |
| **Format overfitting** | Working only with the exact training template. |
| **Alignment tax** | Loss of unrelated general capability after alignment training. |
| **`merge_and_unload()`** | Folds LoRA adapters into base weights for deployment. |

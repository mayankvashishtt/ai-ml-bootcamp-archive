# Week 12 — Fine-tuning Part 1: What, Why & Continued Pre-Training

**Date:** 04/04/2026
**Sources:** `downloads/week-12-fine-tuning-part-1.pdf` (41 slides) · `downloads/week-12-fine-tuning-part-1.ipynb` (43 cells)
**Notion page:** https://100xschool.notion.site/339ffffa33e58006aae4fb5d13ae178d

---

## The journey so far

| Stage | What it does |
|---|---|
| **Pre-training** | The model learns language |
| **Prompting** | We talk to the model |
| **RAG** | We give it external knowledge |
| **Agents** | We give it tools |
| **Fine-tuning** | **We change the model itself** |

Everything before this week left the weights untouched. This is the first technique that reaches inside.

---

## 1. What is fine-tuning?

| Pre-training | Fine-tuning |
|---|---|
| Learn language from the internet | Adapt that knowledge for your needs |
| Trillions of tokens, months, $$$$$ | Thousands of tokens, hours, $ |

> **Same mechanism. Different scale. Different data.**
>
> *Like a university graduate who already knows how to learn — you're not teaching them from scratch, you're training them for a specific job.*

The final slide's summary is worth holding onto throughout: **"Fine-tuning is not magic — it's gradient descent on different data."** Everything in Weeks 12–14 is the Week 2 training loop with a different dataset.

---

## 2. Why fine-tune?

**Prompting doesn't work when:**
- The model doesn't know your domain's language
- You need a specific style/format **consistently**
- You need the model **smaller/faster** for production
- You want behaviour the prompt **can't enforce**

**RAG doesn't work when:**
- The issue isn't missing knowledge, it's **wrong behaviour**
- **Latency** matters (no retrieval step)
- The model **can't understand what it retrieves**

> **Key insight: RAG adds knowledge. Fine-tuning changes behaviour and style. They solve different problems. Best production systems combine both.**

That third RAG bullet is subtle and important: if the model doesn't speak your domain's language, retrieval hands it text it cannot parse. **Sometimes you need CPT so that RAG works.**

### The decision framework

| Symptom | Solution |
|---|---|
| "Model doesn't **KNOW** something" | **RAG or CPT** |
| "Model doesn't **DO** something well" | **SFT** |
| "Model does it but **not how humans want**" | **RLHF / DPO** |
| "Model needs to **reason better**" | **RLVR** |
| Best systems | **Combine all of them** |

### The full post-training pipeline

```
Base Model (next-token predictor)
    │
    ├── CPT   (Continued Pre-Training)     ← raw text, same loss
    │
    ├── SFT   (Supervised Fine-Tuning)     ← instruction-response pairs
    │
    ├── RLHF  (RL from Human Feedback)     ← human preferences
    │   └── DPO (Direct Preference Opt.)   ← simpler alternative
    │
    └── RLVR  (RL with Verifiable Rewards) ← math/code/logic
```

> **Today: CPT. Next: SFT. Then: RLHF/DPO → RLVR.**
> **Today: what the model knows. Next: what it does. Then: what it values.**

---

## 3. What's inside a model?

A transformer is a stack of layers; each layer has:

| Attention block | FFN block |
|---|---|
| `q_proj` (W_q) · `k_proj` (W_k) · `v_proj` (W_v) · `o_proj` (W_o) | `gate_proj` (W_g) · `up_proj` (W_up) · `down_proj` (W_dn) |
| *"Which tokens to pay attention to"* | *"What to do with the information"* |

Plus `embed_tokens` (input) and `lm_head` (output).

> **These are all WEIGHT MATRICES. Giant grids of numbers.**

Every name here was built in Week 7 — Q/K/V/O from GQA, gate/up/down from SwiGLU, weight-tied embedding and head. Fine-tuning is just choosing which of these to update.

---

## 4. Full fine-tuning and why it breaks

**Full fine-tuning = update EVERY weight in EVERY matrix.** Forward → loss → backward → update, exactly the Week 2 loop.

### The memory maths

To **train** a model you need in GPU memory:

```
1. Model weights        1× model size
2. Gradients            1× model size   (one gradient per param)
3. Optimizer states     2× model size   (AdamW stores m and v)
──────────────────────────────────────
Total:                 ~4× model size   (minimum!)
+ activations, batch   → often 6–8×
```

| Model | fp16 weights | To train |
|---|---|---|
| SmolLM-135M | 270 MB | ~1–2 GB ✓ |
| LLaMA-3 8B | 16 GB | ~64–128 GB △ |
| LLaMA-3 70B | 140 GB | ~560+ GB ✗ |

> **A100 GPU: 80 GB. Your laptop: 8–16 GB.**

**The 4× multiplier is the fact to remember.** People routinely assume a model that *runs* in 16 GB can be *trained* in 16 GB. It can't — AdamW alone doubles the footprint by storing first and second moment estimates per parameter.

### Three more problems beyond memory

| Problem | Detail |
|---|---|
| **Overfitting** | 1000 SEC filings (~5M tokens) vs **8 billion parameters** = massive overfitting. Memorises instead of learning. |
| **Catastrophic forgetting** | Every weight updated → general knowledge overwritten. **No "safe zone" of preserved knowledge.** |
| **Storage & serving** | Each fine-tune = a full copy. **5 variants of LLaMA-70B = 700 GB.** |

### What we actually need

✓ Train on a single GPU (or a laptop) · ✓ Don't overfit on small datasets · ✓ Preserve general knowledge · ✓ Store just the changes · ✓ No inference slowdown

> **What if we could freeze the original model and only train a TINY set of new parameters?**

---

## 5. LoRA

### The key insight (Hu et al., 2021)

> *"When you fine-tune a large model, the weight CHANGES have very low intrinsic dimensionality."*

- The update matrix **ΔW is LOW-RANK**
- Most of ΔW is **redundant**
- The same update can be represented with **far fewer numbers**

> *Like a photo of the sky — mostly one colour. You don't need 10M unique pixels to represent it.*

### What "low-rank" means

| Rank | Meaning |
|---|---|
| **4096** | Full matrix. 4096 × 4096 = 16.7M unique numbers. |
| **1** | Just one pattern, repeated and scaled. |
| **32** | 32 independent patterns combined. |

> **Fine-tuning updates tend to be rank-32-ish, even for a 4096 × 4096 matrix.**

**The choir analogy:** 100 people singing, but only ~4 singing unique parts — the other 96 harmonise or double. You only need to record the 4 unique voices.

### The mechanism

```
Original weight matrix W: (4096 × 4096) = 16.7M params

Instead of learning the full update ΔW:
    ΔW = B × A
    A: (4096 × 32) = 131K params    "down-project to rank"
    B: (32 × 4096) = 131K params    "up-project back"
    Total: 262K params vs 16.7M     (64× fewer!)

W is FROZEN. Only A and B are trained.
Final output: W_frozen · x + (α/r) · B · A · x
```

```
Input x
 ├─ W_frozen × x          NO gradients        → original output
 └─ A × x → B × (Ax)      gradients flow here → LoRA output
                    +
Final = original + (α/r) · LoRA output
```

> **After training: merge!** `W_new = W + (α/r)·BA` → one matrix, **zero added inference cost.**

The mergeability is what makes LoRA production-viable: at serving time there is no adapter, no extra matmul, no latency penalty — just a modified weight matrix.

### The memory savings

| Full fine-tuning LLaMA-3 8B | LoRA (rank 32) on LLaMA-3 8B |
|---|---|
| Weights: 16 GB | Frozen weights: 16 GB → **4-bit → 4 GB** |
| Gradients: 16 GB | LoRA params: ~40 MB |
| Optimizer: 32 GB | Gradients: ~40 MB · Optimizer: ~80 MB |
| **Total: ~64+ GB** ← needs A100 | **Total: ~4.2 GB** ← fits on a laptop |

Saved adapter file: **~80 MB** vs 16 GB for the full model.

> **5 customers, 5 fine-tunes? 400 MB of adapters sharing one base model, vs 80 GB of full copies.**

This is the multi-tenancy argument, and it's why every fine-tuning API you've used is LoRA underneath.

### Hyperparameters

| Parameter | Meaning |
|---|---|
| **`r` (rank)** | Size of the bottleneck. Higher = more capacity, more memory. Common: **8, 16, 32, 64** |
| **`α` (alpha)** | Scaling factor controlling how much LoRA affects output. **Rule of thumb: α = r** (so scaling = 1) |
| **`target_modules`** | Which matrices get adapters. **Minimum:** `q_proj`, `v_proj`. **Recommended:** all attention + all FFN. **CPT special:** + `embed_tokens`, `lm_head` |
| **`dropout`** | Usually 0 (Unsloth optimises for this) |

---

## 6. Why CPT needs `embed_tokens` and `lm_head`

| | |
|---|---|
| **`embed_tokens`** | token ID → vector — *"What does this word mean?"* |
| **`lm_head`** | vector → token ID — *"What word should come next?"* |

| SFT | CPT |
|---|---|
| Model already knows the vocabulary | Model is learning **NEW domain language** |
| Teaching it to *use* words differently | "EBITDA", "10-K", "diluted EPS", "subordinated debentures" |
| → Don't need to touch embeddings | → **MUST update `embed_tokens` + `lm_head`** |

> **Most common CPT mistake: copying an SFT LoRA config and wondering why domain adaptation doesn't work.**

The logic: if a domain term tokenises into pieces whose embeddings encode general-English meanings, no amount of adapting the middle layers fixes the *input representation*. You have to change what those tokens mean.

---

## 7. Quantization and QLoRA

| Precision | Bits/param | SmolLM-135M |
|---|---|---|
| Float32 | 32 | 540 MB |
| Float16 | 16 | 270 MB |
| Int8 | 8 | 135 MB |
| **NF4** | **4** | **67 MB** |

> **QLoRA = quantized base (4-bit) + LoRA adapters (16-bit).** The frozen weights are compressed; only the tiny adapters stay in full precision.
>
> **NF4 is designed for neural network weight distributions — almost no quality loss.**

The division of labour is elegant: precision only matters where gradients flow. Frozen weights just need to be *approximately* right; the trainable adapters need to be *precisely* right.

---

## 8. What Unsloth actually does

> **Unsloth is NOT a different training algorithm. It's the SAME math, made faster.**

**What it does:**
- Custom CUDA kernels for attention + LoRA (**2× speed**)
- Fused operations (fewer memory round-trips)
- Smart gradient checkpointing (**50% less VRAM**)
- Automatic mixed precision (bf16/fp16)
- Optimized packing (no wasted padding tokens)

**What it doesn't do:** change the training objective, change the architecture, or do anything impossible with raw HF + PEFT + TRL.

> *Like NumPy vs hand-written Python matrix multiplication. Same math, wildly different speed. If Unsloth vanished, you'd use HF Transformers + PEFT + TRL.*

### Gradient checkpointing

```
Normal training: store ALL activations
  Layer 1 → [STORED]   Layer 2 → [STORED]  ...  Layer 24 → [STORED]
  Memory grows with depth!

Gradient checkpointing: store SOME
  Layer 1 → [STORED]   Layer 2 → [RECOMPUTE when needed]
  Layer 3 → [STORED]   Layer 4 → [RECOMPUTE when needed]  ...
```

> **Trade-off: ~30% slower training, ~50% less VRAM.** Unsloth's version is smarter about *which* layers to checkpoint. (`use_gradient_checkpointing='unsloth'`)

A pure compute-for-memory trade: recompute activations on the backward pass instead of storing them.

---

## 9. The training loop internals

```
1. TOKENIZE   text → token IDs
2. FORWARD    tokens through model → logits
3. LOSS       cross-entropy between predicted and actual next token
4. BACKWARD   compute gradients (only for LoRA params!)
5. UPDATE     optimizer step (AdamW) on LoRA params only
6. REPEAT
```

> **Same loop for pre-training, CPT, and SFT. The only difference is the DATA.**

### Key training concepts

| Concept | Meaning |
|---|---|
| **Packing** | Concatenate short texts into one sequence. Without: `[text1 + PAD PAD PAD]`. With: `[text1 \| text2 \| text3]` — no waste. |
| **Gradient accumulation** | Simulate larger batches. Batch 32 × accumulation 2 = **effective batch 64**. |
| **Warmup** | Start with a tiny LR and ramp up. Prevents early training from destroying weights. |
| **Early stopping** | Monitor eval loss; stop when it stops improving. Prevents overfitting on small datasets. |

---

## 10. The enemy — catastrophic forgetting

```
Before CPT:
  "The cat sat on the"   → "mat"                  ✓ general English
  "Revenue increased by" → random garbage         ✗ no finance

After CPT (no mixing):
  "Revenue increased by" → "12% year over year"   ✓ finance!
  "The cat sat on the"   → "balance sheet"        ✗ forgot English
```

### Why it happens

> **Gradient updates are GREEDY.** At each step the optimizer only cares about reducing loss on the **current batch**.

If every batch is SEC filings:
- Gradients always point toward "be better at finance"
- Gradients **never** point toward "stay good at general text"
- General knowledge slowly erodes

> **The model doesn't "choose" to forget. The optimizer just never gets a signal to remember.**

That's the cleanest available explanation: forgetting isn't a defect, it's the optimizer doing exactly what you asked with an incomplete objective.

### Five solutions

1. **Data mixing** — 80% domain + 20% general
2. **Low learning rate** — don't move far from base (1e-5 full, **2e-4 LoRA**)
3. **LoRA itself** — freezing base weights **IS** forgetting prevention
4. **Short training** — watch eval loss; stop at the plateau
5. **Embedding LR** — lower LR for `embed_tokens`/`lm_head` (**10% of main**), because these layers affect **ALL** tokens, not just domain ones

---

## 11. Hands-on — CPT with SEC 10-K filings

**The plan:** load SmolLM-135M → fail on financial text → get SEC 10-Ks (PleIAs/SEC on HuggingFace) → clean & chunk → CPT with Unsloth + LoRA → retest → measure perplexity → experiment with data mixing.

**What a 10-K is:** the annual report every public US company files with the SEC — business description, risk factors, financial statements, MD&A. Style sample:

> *"The Company recognized impairment charges of $142 million related to goodwill associated with the North America reporting unit during the fiscal year ended March 31, 2024."*

> **SmolLM has never seen this kind of text at scale.**

### Data preparation

```
Raw 10-K filing
    ├── Strip boilerplate (legal disclaimers, page numbers)
    ├── Remove reference/exhibit sections
    ├── Word-level chunking (256 words, 20% overlap)
    ↓
JSONL: {"text": "chunk of financial text..."}
Train/Val split: hold out ~50 filings for evaluation
```

### The config

```python
# LoRA
r = 32                              # higher rank for CPT
target_modules = [
    "q_proj", "k_proj", "v_proj",   # attention
    "o_proj",                        # output projection
    "gate_proj", "up_proj",          # FFN — knowledge lives here!
    "down_proj",
    "embed_tokens", "lm_head"        # CPT-specific: new vocabulary
]
lora_alpha = 32                      # = rank, so scaling = 1

# Training
learning_rate = 2e-4                 # standard for LoRA
embedding_learning_rate = 2e-5       # 10× lower for embeddings
packing = True                       # no wasted padding
warmup_steps = 100                   # gentle start
```

> **Target ALL linear layers including FFN — that's where factual knowledge is stored.**

### What to watch

**Good run:** train loss steadily decreasing · eval loss tracking train loss (not diverging) · eval loss plateaus → stop.

**Bad signs:** eval loss up while train loss drops → **overfitting** · train loss spikes → **LR too high** · loss stuck → **LR too low or data too small**.

> **Perplexity = e^(loss).** Lower = better; the model is less "surprised" by the text. **The eval loss is your north star.**

### Results

**Prompt:** *"The company reported total revenue of"*

| | Output |
|---|---|
| **Base SmolLM** | *"the most important thing is to be able to get a good understanding of the world"* |
| **After CPT** | *"approximately $4.2 billion for the fiscal year ended December 31, 2023, representing an increase of 12%"* |

**Perplexity on SEC test set:** base ~180 → CPT **~45**. *4× lower — from confused to fluent in financial language.*

### NEFTune

> **Add random noise to embeddings during training:** `input_embeds = input_embeds + α * random_noise`

- Acts as **regularization** (like dropout, but for embeddings)
- Prevents overfitting to exact token sequences
- Consistent improvements, **especially on small datasets**

Used in **full training mode**, not LoRA — since LoRA already regularizes by constraining the update space.

### The forgetting experiment

| Run A | Run B |
|---|---|
| 100% SEC filings | 80% SEC + 20% general (`scientific_papers`) |

Evaluated on the SEC test set (domain) and general text (wikitext).

**Expected:** Run A great on SEC, degraded on general. Run B slightly worse on SEC, **maintained general.** That's the trade you're consciously making.

---

## 12. The limit of CPT

**Prompt:** *"What was Apple's revenue in fiscal year 2023?"*

**After CPT:** *"The company's revenue was $383.3 billion for the fiscal year ended September 30, 2023, representing a decrease of 2.8% compared to the prior fiscal year revenue of $394.3..."*

> **It continued the text. It didn't ANSWER the question.**
>
> **CPT teaches the model to SOUND like a domain. SFT teaches the model to RESPOND to instructions.**

CPT uses the *same objective as pre-training* — next-token prediction on raw text — so it produces a better autocompleter, not an assistant. Turning an autocompleter into an assistant is Week 13's job.

---

## Key takeaways

1. **Fine-tuning modifies the model itself** (unlike prompting/RAG).
2. **LoRA: freeze base, train tiny adapter matrices** — 64× fewer params, mergeable at inference.
3. **CPT = same loss as pre-training, different (domain) data.**
4. **For CPT: include `embed_tokens` + `lm_head`** in LoRA targets — the most common CPT mistake.
5. **Catastrophic forgetting is real** — mix in general data.
6. **Unsloth = same math, faster execution.**
7. **CPT teaches domain language, NOT instruction following.**
8. **Training needs ~4× model size in memory** (weights + gradients + AdamW states), often 6–8×.
9. **QLoRA = 4-bit frozen base + 16-bit adapters** — precision only where gradients flow.
10. **RAG adds knowledge; fine-tuning changes behaviour.** Best systems combine both.

> **Fine-tuning is not magic — it's gradient descent on different data.**

---

## Glossary

| Term | Meaning |
|---|---|
| **CPT** | Continued Pre-Training — more next-token training on domain text. |
| **SFT** | Supervised Fine-Tuning — training on instruction-response pairs. |
| **RLHF / DPO / RLVR** | Preference and reasoning alignment stages (Weeks 13–15). |
| **Full fine-tuning** | Updating every parameter; needs ~4–8× model size in memory. |
| **LoRA** | Low-Rank Adaptation — freeze W, learn `ΔW = B×A`. |
| **Rank (r)** | LoRA bottleneck size; common values 8–64. |
| **Alpha (α)** | LoRA scaling factor; typically set equal to r. |
| **`target_modules`** | Which weight matrices receive adapters. |
| **Adapter merging** | Folding `(α/r)·BA` into W so inference cost is unchanged. |
| **Quantization** | Storing weights at lower precision (fp16 → int8 → NF4). |
| **NF4** | 4-bit format designed for neural weight distributions. |
| **QLoRA** | 4-bit quantized frozen base plus 16-bit LoRA adapters. |
| **Catastrophic forgetting** | Loss of general ability when training only on domain data. |
| **Data mixing** | Blending general data (~20%) into domain training to prevent forgetting. |
| **Gradient checkpointing** | Recomputing activations instead of storing them: ~30% slower, ~50% less VRAM. |
| **Packing** | Concatenating short texts to eliminate padding waste. |
| **Gradient accumulation** | Summing gradients over steps to simulate a larger batch. |
| **Warmup** | Ramping the learning rate from near zero at the start of training. |
| **Perplexity** | `e^loss` — how "surprised" the model is; lower is better. |
| **NEFTune** | Adding noise to embeddings during training as regularization. |
| **10-K** | Annual report filed with the SEC by US public companies. |
| **Unsloth** | Optimised training library — same math, ~2× faster, ~50% less VRAM. |

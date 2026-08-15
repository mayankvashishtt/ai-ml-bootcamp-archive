# Week 12 — Quiz (20 questions)

**Topic:** Fine-tuning Part 1 — LoRA, QLoRA, and Continued Pre-Training
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** In the decision framework, "Model doesn't DO something well" points to:
- A) RAG
- B) CPT
- C) SFT
- D) RLVR

**2.** Training a model requires roughly how much GPU memory relative to model size, at minimum?
- A) 1×
- B) 2×
- C) 4×
- D) 10×

**3.** The 4× multiplier comes from:
- A) Weights, activations, batch data, and logits
- B) Weights, gradients, and two AdamW optimizer states
- C) Four copies of the weights for parallelism
- D) Four-bit quantization overhead

**4.** LoRA's core insight from Hu et al. (2021) is that:
- A) Attention layers are redundant
- B) Weight *changes* during fine-tuning have low intrinsic dimensionality
- C) Only the FFN needs training
- D) Quantization improves accuracy

**5.** With W of 4096×4096 and rank 32, LoRA trains approximately:
- A) 16.7M parameters
- B) 262K parameters
- C) 131K parameters
- D) 32 parameters

**6.** The recommended rule of thumb for alpha is:
- A) α = 1
- B) α = r
- C) α = 2r
- D) α = r/2

**7.** After training, merging LoRA gives:
- A) A permanent 2× inference slowdown
- B) `W_new = W + (α/r)·BA` and zero added inference cost
- C) A model that can no longer be quantized
- D) Two separate models to serve

**8.** For CPT specifically, which modules must be added to the LoRA targets?
- A) `q_proj` and `v_proj`
- B) `gate_proj` and `up_proj`
- C) `embed_tokens` and `lm_head`
- D) `o_proj` only

**9.** QLoRA is best described as:
- A) A quantized LoRA adapter with a full-precision base
- B) A 4-bit quantized frozen base with 16-bit LoRA adapters
- C) 8-bit training throughout
- D) A different training objective

**10.** Gradient checkpointing trades:
- A) ~30% slower training for ~50% less VRAM
- B) ~50% slower training for ~30% less VRAM
- C) Accuracy for speed
- D) VRAM for disk space

**11.** Catastrophic forgetting happens because:
- A) The model deliberately discards old knowledge
- B) Gradients are greedy and never receive a signal to preserve general ability
- C) The tokenizer changes during training
- D) Weights decay toward zero

**12.** After CPT, asking "What was Apple's revenue in fiscal year 2023?" produced:
- A) A correct direct answer
- B) A fluent continuation of the text rather than an answer
- C) An error
- D) General English unrelated to finance

---

## Short answer

**13.** Distinguish CPT, SFT, RLHF/DPO, and RLVR by the symptom each addresses.

**14.** Work through the memory maths for LLaMA-3 8B under full fine-tuning, then under QLoRA rank 32. Explain each line.

**15.** Explain "low-rank" using the choir analogy, and why it makes LoRA possible.

**16.** Explain why LoRA can be merged after training, and why that matters for production.

**17.** Explain why CPT requires `embed_tokens` and `lm_head` but SFT typically doesn't.

**18.** Explain the mechanics of catastrophic forgetting and list five mitigations, saying why LoRA itself is one.

**19.** Explain why CPT produced a fluent continuation rather than an answer, and what that reveals about the training objective.

**20.** Your colleague says "we'll fine-tune the model on our docs so it knows our product." Critique this plan and propose a better one.

---
---

## Answer key

**1. C** — SFT. RAG/CPT address missing knowledge; RLHF/DPO address behaviour humans dislike; RLVR addresses reasoning.

**2. C** — About 4× at minimum, often 6–8× once activations and batch data are included.

**3. B** — 1× weights + 1× gradients + 2× AdamW optimizer states (the m and v moment estimates).

**4. B** — The update matrix ΔW is low-rank, so the same update can be represented with far fewer numbers.

**5. B** — 262K: A is 4096×32 = 131K and B is 32×4096 = 131K, roughly 64× fewer than 16.7M.

**6. B** — α = r, giving a scaling factor of 1.

**7. B** — The adapters fold into the base weights, so serving is a single matrix with no extra computation.

**8. C** — `embed_tokens` and `lm_head`, because CPT teaches genuinely new domain vocabulary.

**9. B** — A 4-bit (NF4) quantized frozen base with LoRA adapters kept in 16-bit.

**10. A** — Roughly 30% slower training in exchange for roughly 50% less VRAM.

**11. B** — The optimizer greedily minimises loss on the current batch; if every batch is domain data, no gradient ever points toward preserving general ability.

**12. B** — It continued the passage fluently in financial-report style without answering the question.

**13.** **CPT** — the model doesn't *know* a domain's language; retrain with the pre-training objective on raw domain text. **SFT** — the model doesn't *do* a task well; train on instruction-response pairs so it follows instructions instead of continuing text. **RLHF/DPO** — the model does the task but *not in the way humans want*; align it to human preferences over outputs. **RLVR** — the model needs to *reason better*; train against automatically verifiable rewards in domains like maths, code, and logic. In shorthand: CPT changes what it knows, SFT what it does, RLHF/DPO what it values, RLVR how well it thinks.

**14.** **Full fine-tuning, LLaMA-3 8B:** weights 16 GB (8B params × 2 bytes at fp16); gradients 16 GB, since every trainable parameter needs one gradient; optimizer states 32 GB, because AdamW stores two values per parameter (the first-moment m and second-moment v estimates). Total ~64 GB before activations and batch data — requiring an A100. **QLoRA rank 32:** the frozen base is quantized to 4-bit NF4, so 16 GB becomes ~4 GB; the LoRA parameters are only ~40 MB, since with r=32 each adapted matrix contributes 262K parameters instead of 16.7M; gradients are ~40 MB because gradients exist only for trainable parameters; optimizer states are ~80 MB, twice the trainable parameter count. Total ~4.2 GB, which fits on a laptop. The saved artefact is an ~80 MB adapter rather than a 16 GB model.

**15.** Rank measures how many genuinely independent patterns a matrix contains. A rank-4096 matrix has 16.7M unique values; a rank-1 matrix is a single pattern repeated and scaled; a rank-32 matrix is 32 independent patterns combined. **The choir analogy:** 100 singers, but only about 4 sing distinct parts while the remaining 96 harmonise or double them — so recording the 4 unique voices captures nearly everything. **Why this enables LoRA:** the empirical finding is that fine-tuning updates are themselves roughly rank-32 even for a 4096×4096 matrix, so ΔW can be factored as B×A with a small inner dimension. Instead of learning 16.7M numbers you learn 262K, capturing the same update because the extra degrees of freedom were redundant anyway.

**16.** During training the layer computes `W_frozen·x + (α/r)·B·A·x`, which is a sum of two linear maps applied to the same input. Because both branches are linear, the sum is itself a single linear map, so the adapters can be folded in algebraically: `W_new = W + (α/r)·BA`, and `W_new·x` is exactly equivalent. **Why it matters in production:** serving becomes a single matrix multiply with no extra branch, no adapter loading, and no additional latency — so a LoRA-tuned model costs precisely what the base model costs at inference. This is the property that makes LoRA deployable rather than merely cheap to train, and it distinguishes it from adapter methods that insert genuinely extra layers. Note the trade-off: once merged you lose the ability to swap adapters per request, so multi-tenant setups often keep them unmerged and accept a small overhead.

**17.** `embed_tokens` maps token IDs to vectors ("what does this word mean?") and `lm_head` maps vectors back to token IDs ("what word comes next?"). **SFT** works with vocabulary the model already understands — it is teaching the model to *use* known words differently, to follow instructions rather than continue text — so the input and output representations are already adequate and only the intermediate behaviour needs changing. **CPT** teaches genuinely new domain language: terms like "EBITDA", "diluted EPS", and "subordinated debentures" tokenise into pieces whose embeddings currently encode general-English meanings. If the input representation of those tokens stays wrong, no amount of adapting the middle layers can compensate, because the model never receives a correct signal about what the token means. The same applies at the output: predicting domain vocabulary well requires updating `lm_head`. The lecture names copying an SFT config as **the most common CPT mistake** for exactly this reason.

**18.** **Mechanics:** gradient descent is greedy — at each step the optimizer minimises loss on the *current batch only*. If every batch is SEC filings, every gradient points toward "be better at finance" and none points toward "stay good at general English," so the weights encoding general language are gradually overwritten. The model does not choose to forget; the optimizer simply never receives a signal to remember. **Mitigations:** (i) **data mixing** — roughly 80% domain plus 20% general data, so some batches supply the missing gradient signal; (ii) **low learning rate** — 1e-5 for full fine-tuning, 2e-4 for LoRA, keeping weights near the base; (iii) **LoRA itself**; (iv) **short training** with early stopping when eval loss plateaus; (v) **lower embedding learning rate**, around 10% of the main rate, because `embed_tokens` and `lm_head` affect *all* tokens rather than only domain ones. **Why LoRA is itself a mitigation:** the base weights are frozen and never receive updates, so the general knowledge stored in them cannot be overwritten. All adaptation is confined to a small low-rank subspace added on top, which structurally limits how far behaviour can drift — the "safe zone" that full fine-tuning lacks.

**19.** CPT uses **exactly the same objective as pre-training** — cross-entropy on next-token prediction over raw text. The training data is a corpus of 10-K filings containing no questions and no answers, only continuous financial prose. The model therefore learned the *distribution of financial writing* extremely well, and when given "What was Apple's revenue in fiscal year 2023?" it did the only thing the objective ever rewarded: it produced the most plausible continuation, which in 10-K style is a fluent revenue disclosure sentence. **What this reveals:** the objective determines the behaviour, not the content. Training on domain text makes a better *autocompleter for that domain*, and instruction-following is a separate capability that must be trained separately with instruction-response pairs — which is precisely SFT's job. As the slide puts it, CPT teaches the model to **sound** like a domain; SFT teaches it to **respond**.

**20.** **The critique:** "so it knows our product" is a *knowledge* problem, and the decision framework routes knowledge problems to **RAG**, not fine-tuning. Fine-tuning on documents has three specific defects here. It is **unreliable for facts** — the model absorbs style and distribution far more readily than specific figures, and will confidently produce plausible-sounding but wrong details with no way to distinguish memorised fact from confabulation. It **cannot cite**, so answers are unverifiable, whereas RAG returns the source chunk. And it **goes stale immediately**: every documentation update requires retraining, whereas a vector index is updated by re-embedding the changed files. There is also a real risk of **catastrophic forgetting** if the corpus is small and unmixed. **A better plan:** start with **RAG** over the product docs, with recursive chunking, hybrid search, and citations, and measure where it fails. If retrieval succeeds but the model mangles the domain's vocabulary — struggling with internal jargon and product names — add **CPT** on the docs so it can actually parse what is retrieved. If it retrieves and understands correctly but answers in the wrong format or tone, add **SFT** on a few hundred real support question-answer pairs. As the lecture notes, the best production systems combine these — but the ordering matters, and RAG is where you start.

# Week 23 — Quiz (20 questions)

**Topic:** Hugging Face End-to-End — Hub, libraries, deployment, extras
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The three-layer mental model is:
- A) Train, evaluate, deploy
- B) Store, build, ship
- C) Data, model, inference
- D) Free, pro, enterprise

**2.** Safetensors replaced pickle primarily because pickle:
- A) Was slower to parse
- B) Allows arbitrary code execution on load
- C) Could not store float16
- D) Was proprietary

**3.** Xet improves on Git-LFS by providing:
- A) Encryption at rest
- B) Chunk-level deduplication
- C) Faster HTTP
- D) Automatic quantization

**4.** Transformers v5 (Dec 2025) notably:
- A) Added TensorFlow support
- B) Went PyTorch-first, dropping TF and Flax
- C) Removed quantization
- D) Merged with Diffusers

**5.** `load_dataset(..., streaming=True)` allows you to:
- A) Train faster on small datasets
- B) Iterate TB-scale data without downloading it
- C) Automatically shard across GPUs
- D) Convert datasets to JSON

**6.** The Tokenizers library uses a Rust backend because:
- A) Rust has better ML libraries
- B) Tokenization is a hot per-token loop where Python is too slow
- C) It is required by safetensors
- D) PyTorch mandates it

**7.** Which TRL trainer corresponds to Week 15's reasoning work?
- A) `SFTTrainer`
- B) `DPOTrainer`
- C) `GRPOTrainer`
- D) `PPOTrainer`

**8.** Inference Providers is best described as:
- A) A dedicated GPU per model
- B) An OpenAI-compatible gateway routing to Groq, Together, Fireworks, Cerebras
- C) A free rate-limited endpoint
- D) A browser inference runtime

**9.** Inference Endpoints cost approximately:
- A) Free
- B) $9/month flat
- C) ~$0.50/hr+, with scale-to-zero
- D) $50/hr minimum

**10.** ZeroGPU on Spaces gives PRO users:
- A) Unlimited H200 access
- B) 25 min/day on an H200
- C) A dedicated A100
- D) CPU-only inference

**11.** smolagents differs from LangGraph by being:
- A) Graph-based with explicit nodes and edges
- B) A code-first agent loop, minimal and opinionated
- C) Focused on durability and human-in-the-loop
- D) Written in Rust

**12.** Which library actually implements the NF4 quantization behind QLoRA?
- A) `peft`
- B) `accelerate`
- C) `bitsandbytes`
- D) `optimum`

---

## Short answer

**13.** Explain the three-layer model and give two components of each layer.

**14.** Explain the pickle vulnerability and why safetensors is both safer *and* faster.

**15.** Explain Xet's advantage over Git-LFS with a concrete scenario.

**16.** Map five course weeks onto the libraries that implement them.

**17.** Compare the three deployment surfaces and say when you'd choose each.

**18.** Why does the deck say "read the model card before you download"? What specifically should you check?

**19.** Explain what "quantization is first-class" in Transformers v5 signals about the field.

**20.** You've fine-tuned a LoRA adapter for a support bot and need to ship it. Walk through the Hugging Face path end to end.

---
---

## Answer key

**1. B** — Store (the Hub), build (core libraries), ship (deployment), plus a long tail of extras.

**2. B** — Python pickle executes arbitrary code on load, so downloading a model meant running a stranger's code.

**3. B** — Chunk-level deduplication, so only changed chunks transfer.

**4. B** — PyTorch-first, with TensorFlow and Flax dropped; it also unified `AttentionInterface` and made quantization first-class.

**5. B** — Iterate datasets larger than local storage, fetching lazily via Parquet and memory-mapping.

**6. B** — Tokenization is a per-token loop over billions of tokens, one of the few places Python's overhead genuinely dominates.

**7. C** — `GRPOTrainer`, matching Week 15's RLVR/GRPO work.

**8. B** — An OpenAI-compatible gateway routing to multiple inference providers, swappable from OpenAI in about two lines.

**9. C** — Roughly $0.50/hr and up, with scale-to-zero so idle time is not billed.

**10. B** — 25 minutes per day on an H200.

**11. B** — A minimal, code-first agent loop, in contrast to LangGraph's low-level graph orchestration with durability and HITL.

**12. C** — `bitsandbytes`. PEFT provides the LoRA layer; bitsandbytes provides the 4-bit NF4 quantization.

**13.** **Layer 1 — The Hub (store):** git for weights, holding **models** and **datasets** (also Spaces), with **model cards** documenting licence and intended use, and `huggingface_hub` as the download/auth plumbing. **Layer 2 — Core libraries (build):** **`transformers`** for loading and running models and **`trl`** for post-training trainers (also `datasets`, `tokenizers`, `accelerate`, `peft`, `diffusers`). **Layer 3 — Deployment (ship):** **Inference Providers** (an OpenAI-compatible multi-provider gateway) and **Inference Endpoints** (dedicated GPUs, scale-to-zero), plus free serverless inference and **Spaces + Gradio** for shipping a model as a web app. There is also a fourth informal layer of **extras** — smolagents, Trackio, OpenEnv, LeRobot, Transformers.js.

**14.** PyTorch's original `.bin` checkpoints were serialised with **Python pickle**, a format that reconstructs objects by **executing code** during deserialisation. Loading a downloaded checkpoint therefore meant running arbitrary code authored by whoever uploaded it — a genuine remote-code-execution vector on a public hub with millions of models, not a theoretical concern. **Safetensors is safer** because it is a pure data format: a header describing tensor shapes, dtypes, and byte offsets, followed by raw bytes, with nothing executable to interpret. **It is also faster** because that layout supports **zero-copy memory-mapping** — the file is mapped directly into memory and tensors point into it, so there is no deserialisation pass and no duplicate copy in RAM. Security and speed came from the same design decision: describing data rather than reconstructing objects.

**15.** **Git-LFS versions whole files.** A model repository holding a 14 GB `.safetensors` checkpoint treats any modification as a new object, so changing a single tensor — or even just re-saving after a small config tweak — uploads and stores another full 14 GB, and every collaborator pulls 14 GB again. Across many revisions the storage and bandwidth cost scales linearly with commits. **Xet deduplicates at chunk level:** the file is split into content-addressed chunks, and only chunks whose contents actually changed are transferred or stored. **Concrete scenario:** you fine-tune a model, then push three successive checkpoints that differ in a minority of weights. Under Git-LFS that is roughly 42 GB moved; under Xet the second and third pushes transfer only the differing chunks, which may be a small fraction. The benefit is stated precisely as "faster pulls on large, slowly-changing repos" — exactly the checkpoint-iteration workflow.

**16.** **Week 3 (tokenization)** → **`tokenizers`**, the Rust-backed BPE/WordPiece/Unigram implementation, described in the deck as "the real thing" behind that lecture. **Week 9–10 (RAG)** → **`sentence-transformers`** for the embeddings driving semantic search. **Week 12 (LoRA/QLoRA)** → **`peft`** for the adapters, with **`bitsandbytes`** supplying the 4-bit NF4 quantization. **Week 13 (SFT)** → **`trl.SFTTrainer`**. **Week 15 (RLVR/GRPO)** → **`trl.GRPOTrainer`**. (Equally valid: Week 14 → `trl.DPOTrainer`; Weeks 5/7 → `transformers` + `accelerate`; Week 16 → `OpenEnv`; Week 17 → `lighteval`/`evaluate`.) Notably, Unsloth from Weeks 12–15 was itself described as "same math, faster" layered on `transformers` + `peft` + `trl` — this is the stack beneath it.

**17.** **Serverless (`hf-inference`)** is free but rate-limited to a few hundred requests per hour, explicitly **prototyping only**, with a $9/month PRO tier raising the limits — use it while exploring, never for production traffic. **Inference Providers** is an **OpenAI-compatible gateway** routing to Groq, Together, Fireworks, and Cerebras, reachable via `InferenceClient` as roughly a two-line swap from the OpenAI SDK — choose it for production when you want managed capacity, provider redundancy, and no infrastructure to run, and it is the same pattern as OpenRouter from Week 8. **Inference Endpoints** provisions a **dedicated GPU per model** at roughly $0.50/hr and up with **scale-to-zero**, defaulting to vLLM (SGLang for RAG workloads, TGI in maintenance) — choose it when you need a custom or fine-tuned model that no provider hosts, guaranteed capacity and predictable latency, or data isolation for compliance. Scale-to-zero is what makes it viable for bursty traffic, since idle time is not billed.

**18.** Because the card carries the information that determines whether you can legally and safely use the model — and it is cheaper to read than a download is to undo. **Check specifically:** **the licence**, including whether the model is **gated** (requiring accepted terms or approval) and whether it permits **commercial** use, since "open weights" is emphatically not the same as "commercially usable" and several widely-used models carry restrictions that only appear here; **intended use and limitations**, which state what the model was built for and the failure modes the authors already know about; and **how to evaluate it**, meaning the benchmarks and caveats the authors report. The Week 17 caution compounds this: reported benchmark numbers are **foundation evals** and tell you little about whether the model works for your product, so treat card metrics as a screening filter and plan your own product eval regardless.

**19.** It signals that **quantization has moved from research technique to default infrastructure** in roughly two years. QLoRA — a 4-bit quantized frozen base with 16-bit LoRA adapters — was presented in Week 12 as the enabling trick that made fine-tuning an 8B model fit in about 4.2 GB instead of 64+ GB. Being "first-class" in a major framework release means it is no longer an add-on bolted on via a third-party integration but a supported path through the core loading and serving code. The broader signal is about **where the field's binding constraint sits**: the same v5 release dropped TensorFlow and Flax to go PyTorch-first, indicating the framework question is settled, while elevating quantization indicates that **memory, not capability, is the practical bottleneck** for most practitioners. Making 4-bit loading routine is what lets people run and adapt large models on hardware they actually own — the same democratisation argument LoRA made, now baked into the default stack.

**20.** **Store.** Push the adapter to the Hub with `push_to_hub()` — this uploads only the LoRA weights, a few megabytes rather than gigabytes, since the base model is referenced rather than copied. Write a **model card** stating the base model, the training data's provenance, intended use, known limitations, and the licence, which is doubly important because your adapter inherits the base model's licence terms. Ensure weights are saved as **`.safetensors`**, and note that **Xet** makes subsequent checkpoint pushes cheap when you iterate. **Decide merged versus unmerged.** Per Week 12, `merge_and_unload()` folds the adapter into the base for zero inference overhead and a single artifact, but you lose the ability to hot-swap adapters — keep it unmerged if you plan to serve several tenant-specific variants from one base. **Ship.** Start on **Serverless** for a smoke test, accepting its few-hundred-per-hour cap. For production, the decisive fact is that this is a **custom fine-tune no third-party provider hosts**, which rules out Inference Providers for the model itself and points to an **Inference Endpoint** — a dedicated GPU at roughly $0.50/hr and up, running vLLM by default, with **scale-to-zero** keeping costs low for the bursty traffic typical of support workloads. **Demo and iterate.** Put a **Gradio Space** in front of it so non-engineers can try it, using **ZeroGPU** (25 min/day on an H200 with PRO) if you would rather not keep an endpoint warm for demos. Track training runs with **Trackio**, whose dashboard can itself live on a Space. **Evaluate before and after.** Use `lighteval`/`evaluate` for a baseline, but heed Week 17: build a **product eval** with binary criteria on real support queries, since leaderboard metrics will not tell you whether this bot answers your customers correctly.

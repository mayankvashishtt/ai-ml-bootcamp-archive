# Week 23 — Hugging Face End-to-End

**Date:** 20/06/2026
**Source:** `downloads/week-23-hugging-face-end-to-end.pdf` (31 slides, 39-slide deck) — *no Colab notebook this week*
**Notion page:** https://100xschool.notion.site/397ffffa33e5802fb1d8ff15b6e7bfcd

> A map lecture. Almost every library named here was used earlier in the course — this session puts them in one frame and shows how they connect.

---

## 0. The idea in plain language

This is a **map lecture**. You've already used most of these tools — `transformers` in Week 7, `peft` for LoRA in Week 12, `trl` for SFT and DPO in Weeks 13–14, `datasets` throughout. This session puts them in one frame so the ecosystem stops feeling like a pile of unrelated imports.

**The simplest way to think about Hugging Face is as three layers:**

**1. The Hub — where things are stored.** GitHub, but for models, datasets, and demos. Every model you've downloaded came from here. It's genuinely the reason open-source ML moves as fast as it does: someone fine-tunes a model, pushes it, and everyone else can use it that afternoon.

**2. The libraries — how you build.** Each does one job, and they're designed to compose:

| Library | What it does | You used it in |
|---|---|---|
| `transformers` | Load and run models | Week 7 |
| `datasets` | Load and process training data | Weeks 12–14 |
| `tokenizers` | Text ↔ tokens, fast | Week 3 |
| `peft` | LoRA / QLoRA | Week 12 |
| `trl` | SFT, DPO, GRPO trainers | Weeks 13–15 |
| `accelerate` | Multi-GPU without rewriting code | (S10's territory) |

**3. Deployment — how you ship.** Inference endpoints, serverless APIs, and **Spaces** (a hosted demo, usually Gradio, that turns a model into a shareable web page in about twenty lines).

**The one genuinely important thing in this lecture is the model card**, and it's easy to skim past. A model card is the README on a model's Hub page, and reading it properly is a real skill:

- **What was it trained on?** Determines what it's good at, and what biases it inherited.
- **What's the licence?** Not every "open" model is commercially usable — some are research-only, some restrict use above a revenue threshold. This catches teams out.
- **Base or instruct?** (Week 13) A base model continues text; an instruct model answers. Download the wrong one and it rambles at you.
- **What are the reported numbers, and against what baseline?** Week 20's scepticism applies to model cards exactly as it does to papers.
- **Total vs active parameters** if it's an MoE (S10) — the first tells you what hardware can host it, the second what it costs to run.

**The practical framing to leave with:** Hugging Face is the reason you didn't have to implement attention, LoRA, or a DPO trainer yourself. It's infrastructure, and the skill isn't memorising the API — it's knowing **which layer solves which problem** so you can find the right tool quickly.

---

## The mental model

> **Three layers — store, build, ship — plus a long tail of extras.**

| Layer | What |
|---|---|
| **1. The Hub** | Store — models, datasets, Spaces |
| **2. Core libraries** | Build — the training and inference stack |
| **3. Deployment** | Ship — three inference surfaces + Spaces |
| **4. Extras** | Everything else, and where it's going |

**Scale (Jan 2026):** **2.4M+ models · 730K+ datasets · ~1M Spaces**

---

# Layer 1 — The Hub

> **Git for weights.**

### Model & dataset cards

- **License** — gated? commercial?
- **Intended use, and limitations**
- **How to evaluate before you download**

The licence check is the one people skip and regret. "Open weights" is not the same as "commercially usable" — several popular models carry restrictions that only surface in the card.

### `huggingface_hub` — the plumbing

```python
snapshot_download(...)
hf_hub_download(...)
push_to_hub(...)
login()
```

> **The plumbing every other library sits on.**

Worth knowing because when `from_pretrained` fails, it's usually this layer — auth, gated access, cache paths.

### Safetensors — the security beat

- **Replaced pickle**
- **Pickle = arbitrary code execution**
- **`.safetensors` = safe, fast, zero-copy**

A real vulnerability, not a theoretical one: PyTorch's original `.bin` format used Python pickle, which executes arbitrary code on load. Downloading a model from a public hub meant running a stranger's code. Safetensors is a pure data format — nothing to execute — and it memory-maps, so loading is also faster. **Prefer `.safetensors` always.**

### Xet — the new storage backend

- **Chunk-level deduplication**
- **Replacing Git-LFS**
- **Faster pulls on large, slowly-changing repos**

Git-LFS versions whole files, so a one-line change to a 14 GB checkpoint re-uploads 14 GB. Xet dedupes at chunk level, transferring only what changed.

---

# Layer 2 — Core libraries

### Transformers

- **Load any model in about three lines**
- `pipeline()` · `AutoModel` · `AutoTokenizer`
- **Text · vision · audio · multimodal**

**Transformers v5 (Dec 2025)** — the first major release in five years:
- **PyTorch-first — TF and Flax dropped**
- **Unified `AttentionInterface`**
- **Quantization is first-class**

Note what this signals: the framework war is over. And "quantization is first-class" reflects Week 12 — QLoRA went from a research technique to core infrastructure in about two years.

### Datasets

- **Stream TB-scale without downloading**
- **Parquet + memory-mapping**
- `load_dataset(..., streaming=True)`

Streaming is the feature that matters: you can iterate a dataset larger than your disk, because batches are fetched lazily.

### Tokenizers

- **Rust backend**
- **BPE · WordPiece · Unigram**
- → **Week 3 — the real thing**

BPE is exactly what Week 3 built up from characters. Rust because tokenization is a per-token loop over billions of tokens — the one place Python is genuinely too slow.

### Accelerate

- **Same code: one GPU → many**
- **FSDP · DeepSpeed · fp8**
- **Device-agnostic**

### PEFT

- **LoRA / QLoRA**
- **An adapter is a few MB, not GB**
- → **Weeks 12 / 13 / 14**

Exactly Week 12's arithmetic: an ~80 MB adapter against a 16 GB model.

### TRL

- **`SFTTrainer` · `DPOTrainer` · `GRPOTrainer`**
- → **Weeks 13 / 14 / 15**
- **v1 shipped March 2026**

One trainer per post-training stage. The library structure *is* the pipeline from Weeks 13–15.

### Diffusers

- **Image · video · audio generation**
- **The branch we haven't touched**
- **Same Hub, same patterns**

Honest signposting: the course is language-only, and this is the door to the other half of the field.

### More

| Library | Purpose |
|---|---|
| **Optimum** | ONNX · TPU · Trainium |
| **Sentence-Transformers** | Embeddings (**RAG** — Weeks 9–10) |
| **timm** | Vision backbones |

---

# Layer 3 — Deployment

> **Three confusingly-named products.** (The deck says this out loud, which is fair — the names are genuinely hard to keep straight.)

| # | Surface | What it is |
|---|---|---|
| **1** | **Serverless** (`hf-inference`) | **Free, rate-limited** — a few hundred/hour. **Prototyping only.** PRO ($9/mo) raises the limits. |
| **2** | **Inference Providers** | An **OpenAI-compatible gateway** routing to **Groq · Together · Fireworks · Cerebras**. `InferenceClient` — **a two-line swap from OpenAI.** |
| **3** | **Inference Endpoints** | **A dedicated GPU per model.** ~**$0.50/hr+**, **scale-to-zero**. **vLLM default · SGLang for RAG · TGI in maintenance.** |

**How to choose:** free/rate-limited for prototyping → shared multi-provider gateway for production without infrastructure → dedicated GPU when you need guaranteed capacity, custom models, or data isolation. Inference Providers is the same pattern as OpenRouter from Week 8 — one API, many backends.

*Scale-to-zero* is what makes option 3 viable for bursty traffic: you don't pay while idle.

### Spaces + Gradio

- **Ship a model as a web app**
- **Gradio · Streamlit · static**
- **ZeroGPU — 25 min/day on an H200 with PRO**

---

# Layer 4 — The extras

### smolagents

- **Agents in a few lines**
- **A code-first agent loop**
- **vs LangGraph (Week 21)**

"Code-first" means the model **writes Python** to act rather than emitting JSON tool calls — closer to the Week 11 RLM pattern than to the Week 8 ReAct loop. The contrast with LangGraph is scope: smolagents is minimal and opinionated; LangGraph is orchestration infrastructure with durability and HITL.

### AutoTrain · Lighteval · Evaluate

| Tool | Purpose |
|---|---|
| **AutoTrain** | Low-code fine-tuning |
| **Lighteval / Evaluate** | The eval harness |
| **Leaderboards** | → Week 17 |

The Week 17 caution applies: leaderboard evals are **foundation evals**, not product evals.

### Trackio

- **wandb drop-in: `import trackio as wandb`**
- **Local-first, free**
- **Dashboard on a Space**

### OpenEnv

- **Agentic RL execution environments**
- → **Week 16 — RL envs**
- **Step / reset / close, Docker-isolated**

The Meta + Hugging Face standard from Week 16, with the Gymnasium-shaped API.

### LeRobot

- **Real-world robotics in PyTorch**
- **Imitation + RL**
- **SmolVLA foundation model**

### More

| | |
|---|---|
| **Transformers.js** | Inference **in the browser** |
| **bitsandbytes** | Quantization |

`bitsandbytes` is what actually implements the NF4 4-bit quantization behind QLoRA in Week 12.

---

## How this maps to the course

Almost the entire course can be rebuilt from this stack:

| Week | Library |
|---|---|
| **3** — Tokenization | `tokenizers` |
| **5, 7** — PyTorch, training | `transformers`, `accelerate` |
| **9–10** — RAG embeddings | `sentence-transformers` |
| **12** — CPT, LoRA/QLoRA | `peft`, `bitsandbytes` |
| **13** — SFT | `trl.SFTTrainer` |
| **14** — DPO | `trl.DPOTrainer` |
| **15** — GRPO / RLVR | `trl.GRPOTrainer` |
| **16** — RL environments | `OpenEnv` |
| **17** — Evals | `lighteval`, `evaluate` |
| **21** — Agents | `smolagents` (vs LangGraph) |

And Unsloth (Weeks 12–15) was explicitly described as "same math, faster" on top of `transformers` + `peft` + `trl` — this is the layer underneath it.

---

## Common confusions

**"`transformers` vs `trl` vs `peft` — which do I use?"** They stack rather than compete. `transformers` loads and runs the model. `peft` attaches LoRA adapters to it. `trl` provides the training loop (SFT, DPO, GRPO) that trains those adapters. A typical fine-tune imports all three.

**"Base or instruct — which model should I download?"** **Instruct** if you want it to answer questions out of the box. **Base** if you're going to fine-tune it yourself, since you don't want to fight someone else's instruction tuning. If a model rambles and never stops, you probably grabbed the base version (Week 13 §0).

**"Is 'open weights' the same as 'open source'?"** No, and this matters commercially. Open weights means you can download and run the model. The **licence** determines whether you can use it commercially, fine-tune it, or serve it to customers — and some popular models restrict all three. Read the licence on the model card *before* you build on it, not after.

**"Why is the model card worth reading carefully?"** Because it's where the constraints hide: training data (what it's good at, what it's biased toward), licence, base-vs-instruct, context length, and reported benchmarks. Apply Week 20's baseline scepticism — a model card is marketing as well as documentation.

**"What's a Space?"** A hosted demo — usually a Gradio or Streamlit app — that turns a model into a shareable web page. Genuinely useful for showing work to non-technical people, and free at small scale.

**"Serverless API vs Inference Endpoints?"** Serverless is shared, cheap, and cold-starts. Endpoints are dedicated hardware you pay for continuously with predictable latency. Prototype on serverless; move to endpoints when latency matters or volume justifies it.

**"Do I need `accelerate`?"** Only when one GPU stops being enough. It abstracts multi-GPU and mixed-precision setup so the same training script runs on one GPU or eight without changes. S10 covers what it's abstracting.

**"How do I know a Hub model is any good?"** Downloads and likes are weak signals — popularity tracks recency and hype. Better: check whether the card documents its training data and evaluation, whether the licence is usable, and then **run it on your own held-out data** (S3). No leaderboard substitutes for that.

**"Is Hugging Face required?"** No — it's convenience infrastructure, and you could implement all of it yourself. That's the point: it's the reason you didn't have to write attention, LoRA, or a DPO trainer from scratch.

---

## Key takeaways

1. **Three layers: Hub (store) → libraries (build) → deployment (ship)**, plus extras.
2. **2.4M+ models, 730K+ datasets, ~1M Spaces** as of Jan 2026.
3. **Read the model card** — licence, intended use, limitations — *before* downloading.
4. **`.safetensors` replaced pickle** because pickle allows arbitrary code execution on load.
5. **Xet dedupes at chunk level**, replacing Git-LFS for large slowly-changing repos.
6. **Transformers v5 is PyTorch-only** and treats quantization as first-class.
7. **`datasets` streams TB-scale data** without downloading it.
8. **One TRL trainer per post-training stage:** SFT → DPO → GRPO, matching Weeks 13–15.
9. **Three deployment surfaces:** free serverless for prototyping, an OpenAI-compatible multi-provider gateway, and dedicated scale-to-zero GPUs.
10. **`smolagents` is code-first**, in contrast to LangGraph's graph orchestration.
11. **`bitsandbytes` is the quantization underneath QLoRA**; `peft` is the LoRA underneath everything in Weeks 12–15.

---

## Glossary

| Term | Meaning |
|---|---|
| **The Hub** | Hugging Face's git-based store for models, datasets, and Spaces. |
| **Model card** | Documentation covering licence, intended use, and limitations. |
| **`huggingface_hub`** | The download/upload/auth plumbing under every other library. |
| **Safetensors** | Safe, fast, zero-copy weight format replacing pickle-based `.bin`. |
| **Pickle vulnerability** | Python pickle executes arbitrary code on load. |
| **Xet** | Chunk-deduplicating storage backend replacing Git-LFS. |
| **Transformers** | Model loading and inference — `pipeline()`, `AutoModel`, `AutoTokenizer`. |
| **`AttentionInterface`** | v5's unified attention abstraction. |
| **Datasets** | Parquet + memory-mapped dataset library with TB-scale streaming. |
| **Tokenizers** | Rust-backed BPE / WordPiece / Unigram implementation. |
| **Accelerate** | Device-agnostic scaling: FSDP, DeepSpeed, fp8. |
| **PEFT** | Parameter-efficient fine-tuning — LoRA and QLoRA adapters. |
| **TRL** | Post-training trainers: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`. |
| **Diffusers** | Image, video, and audio generation. |
| **Optimum / Sentence-Transformers / timm** | Hardware export / embeddings / vision backbones. |
| **Serverless (`hf-inference`)** | Free rate-limited inference for prototyping. |
| **Inference Providers** | OpenAI-compatible gateway routing to Groq, Together, Fireworks, Cerebras. |
| **Inference Endpoints** | Dedicated GPU per model, ~$0.50/hr+, scale-to-zero. |
| **vLLM / SGLang / TGI** | Serving engines — default / RAG-oriented / in maintenance. |
| **Spaces + Gradio** | Hosted web apps for models; ZeroGPU gives 25 min/day on an H200 with PRO. |
| **smolagents** | Minimal code-first agent framework. |
| **AutoTrain / Lighteval / Evaluate** | Low-code fine-tuning / eval harnesses. |
| **Trackio** | Local-first wandb drop-in (`import trackio as wandb`). |
| **OpenEnv** | Docker-isolated RL environments with step/reset/close. |
| **LeRobot / SmolVLA** | Robotics in PyTorch and its foundation model. |
| **Transformers.js / bitsandbytes** | Browser inference / quantization. |

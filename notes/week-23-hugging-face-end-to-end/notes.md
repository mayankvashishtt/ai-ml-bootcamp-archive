# Week 23 — Hugging Face End-to-End

**Date:** 20/06/2026
**Source:** `downloads/week-23-hugging-face-end-to-end.pdf` (31 slides, 39-slide deck) — *no Colab notebook this week*
**Notion page:** https://100xschool.notion.site/397ffffa33e5802fb1d8ff15b6e7bfcd

> A map lecture. Almost every library named here was used earlier in the course — this session puts them in one frame and shows how they connect.

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

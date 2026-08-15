# Master Glossary

Every term defined across the 33 lectures, merged and alphabetised, with a pointer to where it's explained in depth.

**Refs:** `W##` = course week in `notes/` · `S##` = supplementary lecture in `supplementary/`

---

## A

| Term | Definition | Ref |
|---|---|---|
| **Accessibility tree** | Structured representation of UI elements; more reliable than pixels for grounding | S9 |
| **Activation** | Intermediate value from the forward pass, retained for the backward pass | W2, S10 |
| **Active parameters** | What actually runs per token in an MoE; determines compute cost | S10 |
| **Adapter (LoRA)** | Small trainable matrices added to a frozen base model | W12 |
| **Adapter (multimodal)** | Projection mapping vision features into an LLM's embedding space | S9 |
| **Adversarial verification** | An agent tasked with *refuting* a finding, defaulting to "unproven" | S12 |
| **Agent** | System where the model decides its own path via tools in a loop | W8 |
| **Annealing** | Final training phase on a much higher-quality data subset | S11 |
| **ANN (Approximate Nearest Neighbour)** | Index trading exact recall for search speed (HNSW, IVF, PQ) | S6 |
| **Attention** | Mechanism weighting how much each position attends to every other | W4 |
| **Autoregressive** | Generating one token at a time, each conditioned on all previous | W3 |

## B

| Term | Definition | Ref |
|---|---|---|
| **Barrier** | Synchronisation point where all parallel work must complete before proceeding | S12 |
| **Batch API** | Asynchronous bulk processing at a large discount | S5 |
| **Beam search** | Maintaining B partial sequences by total log-probability; unused for open-ended text | S7 |
| **bf16** | Brain float 16 — fp32's exponent range, reduced precision; no loss scaling needed | S10 |
| **Bias–variance tradeoff** | Underfitting vs overfitting; the core generalisation framing | S1 |
| **Bi-encoder** | Encodes query and document separately, enabling precomputation | S6 |
| **Blackboard** | Shared mutable state that multiple agents read and write | S12 |
| **BM25** | Term-frequency ranking function; matches exact strings, no notion of meaning | W9 |
| **Bootstrap** | Resampling to estimate confidence intervals | S3 |
| **BPE** | Byte-Pair Encoding; iteratively merges frequent adjacent token pairs | W3, S11 |

## C

| Term | Definition | Ref |
|---|---|---|
| **Calibration** | Whether stated confidence matches actual accuracy | S3 |
| **Catastrophic forgetting** | Losing prior capabilities while fine-tuning on new data | W12 |
| **Chain-of-thought** | Prompting the model to reason step by step before answering | S2 |
| **Chinchilla-optimal** | Compute-optimal ratio of ~20 training tokens per parameter | S10 |
| **CLIP** | Contrastive Language–Image Pretraining; aligned image and text encoders | S9 |
| **Compounding error** | Per-step prediction errors accumulating over a rollout until it's fiction | S13 |
| **Confused deputy** | Untrusted input directing a privileged component's legitimate authority | S8, S4 |
| **Constrained decoding** | Masking illegal tokens to −∞ via a grammar/schema state machine | S7 |
| **Contamination** | Benchmark or test data present in training data | S3, S11 |
| **Context firewall** | Using a subagent so its exploration never enters the parent's context | S12 |
| **Context rot** | Measured degradation in quality as context fills with noise | W10 |
| **Contextual retrieval** | Prepending LLM-generated situating context to a chunk before embedding | W9 |
| **Continuous batching** | Adding/removing sequences from a serving batch as they start and finish | S5 |
| **Contrastive learning** | Training a geometry: pull related items together, push unrelated apart | S6 |
| **CPT (Continued Pretraining)** | Further pretraining on domain data before instruction tuning | W12 |
| **Cross-encoder** | Encodes query and document jointly; accurate, cannot precompute | S6 |
| **Curriculum** | Deliberate ordering of training data | S11 |

## D

| Term | Definition | Ref |
|---|---|---|
| **Data parallel (DDP)** | Full model on every GPU, different batch slices, all-reduce gradients | S10 |
| **Debate** | Agents arguing toward consensus; prone to agreeable convergence | S12 |
| **Decoding strategy** | The algorithm converting a probability distribution into a chosen token | S7 |
| **Decontamination** | Removing n-gram overlaps with known benchmarks from training data | S11 |
| **Deduplication** | Removing duplicate documents; improves models *at fixed compute* | S11 |
| **Diffusion of responsibility** | Each agent assuming another handled an edge case; none did | S12 |
| **Diversity collapse** | Synthetic generators converging on a narrow range of phrasings | S11 |
| **DPO** | Direct Preference Optimization; preference tuning without a reward model | W14 |
| **Dynamic discovery** | Learning a server's tools at runtime via `tools/list` | S8 |
| **Dynamic resolution / tiling** | Splitting a large image into fixed-size tiles, often plus a thumbnail | S9 |

## E

| Term | Definition | Ref |
|---|---|---|
| **Early fusion** | Training a single backbone on interleaved modalities from the start | S9 |
| **Elicitation** | An MCP server asking the host to collect input from the user | S8 |
| **Embedding** | Dense vector representation placing similar items near each other | W3, S6 |
| **Error compounding (agents)** | Reliability multiplying down a sequential chain (0.9^N) | S12 |
| **Expert parallelism** | Distributing MoE experts across GPUs, routing tokens over the network | S10 |

## F

| Term | Definition | Ref |
|---|---|---|
| **Fan-out / fan-in** | Dispatching N parallel subagents, then aggregating their results | S12 |
| **Few-shot prompting** | Including worked examples in the prompt | S2 |
| **FlashAttention** | *Exact* attention algorithm avoiding materialisation of the N×N matrix | S10 |
| **Frame subsampling** | Taking a fraction of video frames to control token cost | S9 |
| **Frequency penalty** | Logit penalty proportional to how often a token has been used | S7 |
| **FSDP** | PyTorch's native fully-sharded data parallel; essentially ZeRO-3 | S10 |
| **Function calling** | Model emits a structured tool invocation; your code executes it | W8 |

## G

| Term | Definition | Ref |
|---|---|---|
| **Genie** | Model learning *interactive* environments from unlabelled video | S13 |
| **Goodhart's Law** | When a measure becomes a target, it ceases to be a good measure | W14 |
| **Goodput** | Useful throughput under a latency SLO — the metric that matters | S5 |
| **GQA (Grouped-Query Attention)** | Sharing key/value heads across query heads to shrink KV cache | W6 |
| **Gradient accumulation** | Summing gradients over small batches to simulate a large one | W5, S10 |
| **Gradient boosting** | Sequential ensemble of trees; still state of the art on tabular data | S1 |
| **Gradient checkpointing** | Storing a subset of activations, recomputing the rest; memory-for-compute | S10 |
| **Greedy decoding** | Always selecting the argmax token | S7 |
| **GRPO** | Group Relative Policy Optimization; advantage from a group of rollouts | W15 |
| **Grounding (vision)** | Mapping a language reference onto a location in an image | S9 |
| **Guardrail** | Input/output filter around a model, independent of its own judgement | S4 |

## H

| Term | Definition | Ref |
|---|---|---|
| **Handoff** | Transfer of task and context between agents; inherently lossy | S12 |
| **Hard negative** | A superficially similar but genuinely irrelevant item; teaches the boundary | S6 |
| **Harness** | Everything around the model: tools, context assembly, loop, feedback | W17 |
| **HNSW** | Hierarchical Navigable Small World; graph-based ANN index | S6 |
| **Host (MCP)** | Application owning the LLM conversation and enforcing permissions | S8 |
| **Hybrid search** | Combining dense (semantic) and sparse (BM25) retrieval, fused with RRF | W9 |
| **HyDE** | Embedding a hypothetical *answer* instead of the query | W9, S6 |

## I–J

| Term | Definition | Ref |
|---|---|---|
| **Imagination / latent rollout** | Simulating trajectories inside a world model, no environment interaction | S13 |
| **In-batch negatives** | Using other pairs' positives in a batch as negatives, for free | S6 |
| **InfoNCE** | Contrastive loss; softmax over similarities, anchor vs negatives | S6 |
| **Instruction prefix** | Model-specific prefix distinguishing query from document embeddings | S6 |
| **Inter-annotator agreement** | How often labellers agree; a hard ceiling on achievable model quality | S11 |
| **JEPA** | Joint Embedding Predictive Architecture; predicting in representation space | S13 |
| **JSON-RPC 2.0** | The wire protocol MCP messages are encoded in | S8 |

## K–L

| Term | Definition | Ref |
|---|---|---|
| **KV cache** | Stored keys/values so past tokens aren't recomputed each step | W4 |
| **Latent state** | Compact learned representation replacing raw observations for prediction | S13 |
| **Late interaction** | Storing a vector per token (ColBERT); between bi- and cross-encoder | S6 |
| **Lethal trifecta** | Private data + untrusted content + external communication | S4 |
| **LIMA** | "Less Is More for Alignment" — 1,000 curated examples competitive with far more | S11 |
| **Linear probe** | Simple classifier testing whether info is linearly recoverable from activations | S13 |
| **Load-balancing loss** | Auxiliary MoE loss preventing router collapse onto a few experts | S10 |
| **Logit** | Unnormalised score per vocabulary token, before softmax | S7 |
| **Logprob** | Log-probability of a token; used for confidence, perplexity, classification | S7 |
| **LoRA** | Low-Rank Adaptation; train small matrices, freeze the base | W12 |
| **LSH** | Locality-Sensitive Hashing; buckets similar items to avoid N² comparison | S11 |

## M

| Term | Definition | Ref |
|---|---|---|
| **Matryoshka (MRL)** | Embeddings front-loaded so a truncated prefix remains usable | S6 |
| **McNemar's test** | Significance test for paired model comparisons on the same examples | S3 |
| **MCP** | Model Context Protocol; open standard for exposing capabilities to LLMs | S8 |
| **MCP connector** | Messages API feature attaching remote MCP servers to a request | S8 |
| **Memory (agent)** | Persistent state across sessions; working, episodic, semantic | W18 |
| **Microbatch** | Batch subdivision keeping pipeline-parallel stages busy | S10 |
| **MinHash** | Signature whose collision probability equals Jaccard similarity | S11 |
| **Min-p** | Keep tokens with probability ≥ min_p × (top token's probability) | S7 |
| **Mixed precision** | Computing in bf16/fp16 with an fp32 master copy for updates | S10 |
| **MLA** | Multi-head Latent Attention; compressing KV cache into a latent space | S10 |
| **Model collapse** | Degradation from repeatedly training on model-generated data | S11 |
| **Model exploitation** | A policy discovering and exploiting flaws in a learned world model | S13 |
| **Model-based RL** | Learning dynamics, then planning or training a policy against that model | S13 |
| **Model-free RL** | Learning a policy directly from experience, no dynamics model | S13, W15 |
| **MoE** | Mixture-of-Experts; multiple FFNs, a router selecting top-k per token | S10 |
| **MuZero** | Model-based agent learning dynamics sufficient only for reward/value/policy | S13 |

## N–O

| Term | Definition | Ref |
|---|---|---|
| **N×M problem** | Every application needing a bespoke integration with every system | S8 |
| **Orchestrator** | Agent decomposing a task, dispatching subagents, synthesising results | S12 |
| **Othello-GPT** | Model shown to build a *causally used* internal board representation | S13 |
| **Overfitting** | Fitting training noise; low training error, high test error | S1 |

## P

| Term | Definition | Ref |
|---|---|---|
| **PagedAttention** | Managing KV cache in fixed pages, like OS virtual memory | S5 |
| **Partial observability** | Observations not fully determining state; motivates memory | S13, W18 |
| **Patch** | Fixed-size square of pixels (commonly 16×16) treated as one token | S9 |
| **Perplexity filter** | Dropping documents an LM finds too surprising *or* too predictable | S11 |
| **Pipeline parallel** | Assigning contiguous layer blocks to different GPUs; pipeline bubbles | S10 |
| **Presence penalty** | Flat one-time logit penalty on any token already used | S7 |
| **Prompt (MCP)** | User-controlled reusable template, surfaced as a slash command | S8 |
| **Prompt caching** | Reusing computation for a stable prompt prefix | S5 |
| **Prompt injection** | Untrusted content carrying instructions the model follows | S4 |
| **Provenance** | Records of where each part of a dataset came from | S11 |

## Q–R

| Term | Definition | Ref |
|---|---|---|
| **QLoRA** | LoRA over a 4-bit quantised frozen base | W12 |
| **Quantisation** | Reducing weight precision to save memory and increase throughput | S5, W12 |
| **RAG** | Retrieval-Augmented Generation; retrieve context, then generate | W9 |
| **ReAct** | Reason–Act–Observe loop; the basic agent pattern | W8 |
| **Recall@k** | Fraction of relevant items appearing in the top k results | S6, S3 |
| **Reranking** | Cross-encoder rescoring of a retrieved shortlist | S6 |
| **Repetition penalty** | Blunt logit penalty on any already-used token | S7 |
| **Resource (MCP)** | Application-controlled data identified by URI; read-only context | S8 |
| **Reward hacking** | Optimising the reward proxy rather than the intended goal | W14 |
| **RLAIF** | RL from AI Feedback; scales better, inherits the judge's biases | W14 |
| **RLHF** | RL from Human Feedback; reward model trained on human preferences | W14 |
| **RLM** | Recursive Language Model; exploring a large space via code, not context | W11 |
| **RLVR** | RL with Verifiable Rewards; reward from execution, not a learned model | W15 |
| **RMSNorm** | Simplified normalisation without mean subtraction | W6 |
| **RoPE** | Rotary Position Embedding; encoding position by rotating query/key vectors | W6 |
| **Router (MoE)** | Small network choosing which experts process each token | S10 |
| **RRF** | Reciprocal Rank Fusion; combining ranked lists from multiple retrievers | W9 |
| **RSSM** | Recurrent State-Space Model; Dreamer's deterministic + stochastic latent | S13 |
| **Rug pull** | An MCP server changing tools or behaviour after being approved | S8 |

## S

| Term | Definition | Ref |
|---|---|---|
| **Sample efficiency** | How much real experience is needed to reach a given performance | S13 |
| **Sampling (MCP)** | A server asking the host to run an LLM completion on its behalf | S8 |
| **Scaling law** | Power-law relation between loss and model size, data, or compute | S10 |
| **Self-consistency** | Sampling multiple reasoning paths, taking the majority; needs T > 0 | S7 |
| **Semantic dedup** | Removing near-duplicates in embedding space | S11 |
| **Server (MCP)** | Program exposing capabilities over MCP, usually wrapping one system | S8 |
| **Set-of-mark prompting** | Overlaying numbered markers so the model selects a number, not a coordinate | S9 |
| **SFT** | Supervised Fine-Tuning on prompt→response demonstrations | W13 |
| **Sim-to-real gap** | Discrepancy between simulated dynamics and reality | S13 |
| **Softmax** | Converts scores into a probability distribution summing to 1 | W4, S7 |
| **Speculative decoding** | A small model drafts tokens; the large model verifies in parallel | S5 |
| **Spectrogram** | 2D frequency-over-time audio representation, processed like an image | S9 |
| **stdio transport** | MCP server as a local subprocess over stdin/stdout | S8 |
| **Stop reason** | Why generation ended — `end_turn` (natural) vs `max_tokens` (truncated) | S7 |
| **Stop sequence** | String that terminates generation when produced | S7 |
| **Streamable HTTP** | Remote MCP transport using HTTP with SSE | S8 |
| **Structured output** | API feature guaranteeing responses conform to a JSON Schema | S7 |
| **SwiGLU** | Gated activation function used in modern transformer FFNs | W6 |
| **Synthetic data** | Model-generated training data | W13, S11 |

## T

| Term | Definition | Ref |
|---|---|---|
| **Telephone game** | Cumulative fidelity loss across a chain of agent handoffs | S12 |
| **Temperature** | Divisor applied to logits before softmax; controls sharpness | S7 |
| **Tensor parallel** | Splitting weight matrices across GPUs; needs very high bandwidth | S10 |
| **Token fragmentation** | One concept splitting into many tokens; penalises non-English and rare terms | S11 |
| **Tool (MCP)** | Model-controlled function with a JSON Schema; may have side effects | S8 |
| **Tool poisoning** | Hiding instructions in a tool description the user never reads | S8 |
| **Top-k** | Keep only the k highest-probability tokens before sampling | S7 |
| **Top-p / nucleus** | Keep the smallest token set with cumulative probability ≥ p | S7 |
| **Total parameters** | Everything stored in an MoE; determines memory to host it | S10 |
| **TPOT** | Time per output token — streaming smoothness | S5 |
| **Transition function** | Dynamics mapping current state and action to the next state | S13 |
| **TTFT** | Time to first token — perceived responsiveness | S5 |

## U–Z

| Term | Definition | Ref |
|---|---|---|
| **Upsampling (data)** | Repeating high-quality sources more than once per epoch | S11 |
| **ViT** | Vision Transformer; standard transformer over image patches | S9 |
| **World model** | Learned predictive model of dynamics: state + action → next state | S13 |
| **Worktree isolation** | Giving each agent its own repo copy to prevent conflicting writes | S12 |
| **Zero-shot classification** | Classifying by comparing an image embedding to label-text embeddings | S9 |
| **ZeRO** | Sharding optimizer states (1), gradients (2), parameters (3) across GPUs | S10 |

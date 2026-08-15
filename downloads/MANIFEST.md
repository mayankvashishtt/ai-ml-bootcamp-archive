# Downloads Manifest

Every source file from the 100xSchool AI & ML bootcamp, downloaded from the Notion course page and its linked Google Drive / Colab materials.

**42 files · 41 MB · 24 PDFs + 18 notebooks · 25 lectures**

Filenames map directly to weeks (`week-NN-slug.ext`), so this folder sorts into course order. Notes for each lecture are in `../notes/week-NN-slug/`.

---

## Contents

| Week | Lecture | PDF | Notebook | Notes |
|---|---|---|---|---|
| 00 | Orientation | 117 KB | — | [notes](../notes/week-00-orientation/) |
| 01 | Fast-Tracking the Course of AI | 2.4 MB | — | [notes](../notes/week-01-fast-tracking-ai/) |
| 02 | Neural Networks from Scratch | 3.0 MB | 370 KB | [notes](../notes/week-02-neural-networks-from-scratch/) |
| 03 | Transformers Pt 1 | 7.3 MB | 347 KB | [notes](../notes/week-03-transformers-part-1/) |
| 04 | Transformers Pt 2 | 8.9 MB | 218 KB | [notes](../notes/week-04-transformers-part-2/) |
| 05 | Tensors and PyTorch | 700 KB | 593 KB | [notes](../notes/week-05-tensors-and-pytorch/) |
| 06 | What Changed Since 2017 | 3.7 MB | — | [notes](../notes/week-06-what-changed-since-2017/) |
| 07 | Training Your First Model | 388 KB | 658 KB | [notes](../notes/week-07-training-your-first-model/) |
| 08 | From APIs to Agents | 396 KB | 64 KB | [notes](../notes/week-08-from-apis-to-agents/) |
| 09 | RAG from the Ground Up, Pt 1 | 2.2 MB | 101 KB | [notes](../notes/week-09-rag-part-1/) |
| 10 | Why RAG Breaks — Context Rot | 256 KB | 153 KB | [notes](../notes/week-10-rag-part-2/) |
| 11 | Recursive Language Models | 471 KB | 51 KB | [notes](../notes/week-11-recursive-language-model/) |
| 12 | Fine-tuning Pt 1 — LoRA, QLoRA, CPT | 411 KB | 201 KB | [notes](../notes/week-12-fine-tuning-part-1/) |
| 13 | Fine-tuning Pt 2 — SFT | 300 KB | 256 KB | [notes](../notes/week-13-fine-tuning-part-2/) |
| 14 | Fine-tuning Pt 3 — RLHF & DPO | 551 KB | 113 KB | [notes](../notes/week-14-fine-tuning-part-3/) |
| 15 | RLVR — Reasoning Models | 551 KB | 214 KB | [notes](../notes/week-15-rlvr/) |
| 16 | RL Environments for LLMs | 770 KB | 75 KB | [notes](../notes/week-16-rl-environments-for-llms/) |
| 17 | Harness, Context, and Evals | 567 KB | 80 KB | [notes](../notes/week-17-harness-context-evals/) |
| 18 | Memory | 2.1 MB | — | [notes](../notes/week-18-memory/) |
| 19 | *cancelled* | — | — | — |
| 20 | How to Read Research Papers | 383 KB | — | [notes](../notes/week-20-how-to-read-research-papers/) |
| 21 | LangGraph — Agents as State Machines | 606 KB | 66 KB | [notes](../notes/week-21-langgraph/) |
| 22 | Coding an Agent (assignment) | — | — | [notes](../notes/week-22-coding-an-agent-assignment/) |
| 23 | Hugging Face End-to-End | 368 KB | — | [notes](../notes/week-23-hugging-face-end-to-end/) |
| 24 | LLM Observability | 410 KB | 27 KB | [notes](../notes/week-24-llm-observability/) |
| 25 | Computer Use Agents | 610 KB | 467 KB | [notes](../notes/week-25-computer-use-agents/) |

---

## Notes on gaps

- **Week 19** was cancelled — no materials exist.
- **Week 22** is an assignment with no slides. The Notion page links a spec and a repo; the assignment content is captured in the notes folder.
- **Weeks 00, 01, 06, 18, 20, 23** have slides only, no accompanying notebook.

## Known issues in the source materials

These are reproduced as-downloaded and documented rather than silently corrected. Details in each lecture's notes and in the main [README](../README.md).

| File | Issue |
|---|---|
| `week-02-*.ipynb` | Training output starts at loss 0.000265 instead of ~0.25 — cells re-run without the reset cell. Restart and run all. |
| `week-03-*.ipynb` | Promises "attention from scratch" but ends before it; that content is in Week 4's notebook. |
| `week-04-*.pdf` | Repeats Week 3's slides 1–23; new material starts at slide 24. |
| `week-05-*.ipynb` | Gradient-accumulation demo prints 6.0/6.0/6.0, contradicting its caption — `zero_grad()` called before the second backward. |
| `week-08-*.ipynb` | `search_wikipedia` fails on valid queries, most likely a missing `User-Agent` header. |
| `week-09-*.ipynb` | Hybrid search ranks the RS256 chunk *lower* than naive semantic search; the caption overstates the result. |
| `week-10-*.pdf` | FinanceBench comparison is against *naive* RAG, not the advanced pipeline built in Week 9. |
| `week-11-*.pdf` | The cited RLM arXiv ID doesn't resolve to a real paper; verify before citing. |

## Provenance

Scraped from the [public Notion course page](https://100xschool.notion.site/AI-ML-2e3ffffa33e5806ba537f3c3965fe0c0) via the `loadPageChunk` API, with files pulled from the public Google Drive and Colab links it contains. All files were publicly accessible; nothing here required credentials.

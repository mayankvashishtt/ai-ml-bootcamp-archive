# Week 0 — Quiz (20 questions)

**Topic:** AI & ML Orientation
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom** — cover it and try the whole set first.

> Since Week 0 is a short orientation session, several questions ask you to reason about *why* the course is structured the way it is, using the syllabus as evidence. Those are the ones worth thinking hardest about.

---

## Multiple choice

**1.** Which of the following is NOT one of the four stated goals of the course?
- A) Understand how LLMs actually work
- B) Learn new trends in AI Engineering
- C) Derive backpropagation from first principles
- D) Learn how to stay up to date as the field changes

**2.** What is the instructor's stated approach to mathematics?
- A) Full rigorous derivations for every concept
- B) Minimal maths; whatever appears will be handled with intuitions
- C) No maths whatsoever at any point
- D) Maths taught in optional weekend sessions

**3.** Which platform are students told to have ready before the hands-on sessions begin?
- A) A local CUDA-enabled GPU workstation
- B) A Google Colab account
- C) An AWS SageMaker instance
- D) A Hugging Face Spaces account

**4.** What was the instructor's most recent role before the course?
- A) Research Engineer at an AI lab based in London
- B) Professor of Computer Science
- C) Product Manager at a large tech firm
- D) Founder of a hardware startup

**5.** Which two free tools are recommended for students to try?
- A) Copilot and Cursor
- B) Opencode and Gemini CLI
- C) LangChain and LlamaIndex
- D) PyTorch and TensorFlow

**6.** "Basic Python" is best described in this course as:
- A) Taught from scratch in Week 1
- B) An assumed prerequisite that will not be taught
- C) Optional — everything can be done without code
- D) Only needed after Week 15

**7.** The instructor's specialisation in LLM-RL best explains the presence of which weeks in the syllabus?
- A) Weeks 3–4 on Transformers
- B) Weeks 9–10 on RAG
- C) Weeks 15–16 on RLVR and RL environments
- D) Week 24 on LLM Observability

**8.** Students are advised to explore different models on the market primarily in order to:
- A) Benchmark them formally against each other
- B) Develop first-hand intuition for how they differ
- C) Pick a single vendor and commit to it
- D) Reduce their API spending

**9.** The goal "how to be up to date, with new things coming everyday" is most directly served by which later lecture?
- A) Week 20 — How To Read Research Papers
- B) Week 12 — Fine-tuning
- C) Week 8 — From APIs to Agents
- D) Week 5 — Tensors and PyTorch

**10.** "AI Engineering," as the course uses the term, is best described as:
- A) Training foundation models from scratch
- B) Designing GPU hardware
- C) Building applications and systems on top of existing models
- D) Academic publication of novel architectures

**11.** Which statement about the course's ordering is most accurate?
- A) Production tooling first, theory last
- B) Mechanism and fundamentals first, applied engineering later
- C) Purely random ordering by topic
- D) Reinforcement learning first, then everything else

**12.** Trying a mature coding agent early is useful before Weeks 22 and 25 because:
- A) It's required to submit assignments
- B) It gives you a working reference for how agents plan, call tools, and recover from errors
- C) It automatically writes your notes
- D) It replaces the need to learn Python

---

## Short answer

**13.** State all four goals of the course.

**14.** What does "intuition-first" mean in practice, and what is the one maths-related skill you still genuinely need?

**15.** Why does an intuition-first course still devote a full lecture to reading research papers? What tension does that resolve?

**16.** List the three concrete setup items a student should complete before Week 2.

**17.** The course spends Weeks 2–6 on neural networks and transformers before touching a production API in Week 8. Argue for this ordering.

**18.** Give one risk of the intuition-first approach and how you would personally mitigate it.

**19.** Looking at the syllabus arc (fundamentals → building → adapting → engineering discipline), explain in one sentence what each of the four phases is for.

**20.** The course goal mentions "how to adapt" rather than "what to learn." What does that word choice tell you about the instructor's view of the field's shelf life?

---
---

## Answer key

**1. C** — Deriving backpropagation is explicitly *not* a goal; the instructor states maths will be kept light and intuition-led.

**2. B** — "Will try to not keep as much math, if any will be done with intuitions."

**3. B** — A Colab account. 18 of 26 weeks ship a Colab notebook.

**4. A** — Research Engineer at an AI lab based out of London, working on LLM and LLM-RL.

**5. B** — Opencode and Gemini CLI.

**6. B** — An assumed prerequisite, listed under "Things to do," not taught in the course.

**7. C** — Weeks 15–16 (RLVR, RL environments for LLMs). LLM-RL is the instructor's specialism, which is why this block is unusually deep for a general course.

**8. B** — To build first-hand intuition for how the available models differ.

**9. A** — Week 20, How To Read Research Papers. Reading primary sources is the durable skill for keeping current.

**10. C** — Building on top of existing foundation models rather than training them.

**11. B** — Mechanism first (W1–7), then building (W8–11), adapting (W12–16), and engineering discipline (W17–25).

**12. B** — Using a mature agent gives you a reference implementation to compare your own against.

**13.** (i) Understand what's really going on in the field and how to adapt; (ii) understand how LLMs actually work; (iii) new trends in AI Engineering; (iv) how to stay up to date as new things ship daily.

**14.** Concepts are introduced in plain language first, with equations appearing only as compact restatements of an already-understood mechanism. You are not asked to derive results. The skill you still need is **reading** mathematical notation fluently — Week 20's papers present maths without simplification.

**15.** Intuition gets you to *current* understanding; papers are how understanding stays current. Since the field moves faster than any curriculum, the course teaches the retrieval skill rather than trying to pre-load every future result. It resolves the tension between "keep the maths light" and "you must read primary sources" by treating paper-reading as a learnable, separate skill.

**16.** (i) Refresh basic Python; (ii) get a working Colab account; (iii) explore the models currently available on the market.

**17.** Model APIs are a thin, fast-changing surface over a stable mechanism. Learning the mechanism first means API-level knowledge attaches to something durable — you can reason about why a model fails, why context length matters, or why fine-tuning helps, instead of pattern-matching on documentation. It also directly serves the "how to adapt" goal: mechanism transfers across vendors, API syntax doesn't.

**18.** *Risk:* without derivations you may be able to describe a mechanism but not debug it quantitatively — e.g. recognising a vanishing-gradient problem without being able to reason about gradient magnitude. *Mitigation (example):* re-derive the core results yourself alongside the course — the chain rule through a two-layer network, the softmax gradient, the attention scaling factor — or pair the course with a rigorous text. Any credible, specific plan is acceptable.

**19.** *Fundamentals* — build the mental model of how models actually work. *Building* — use existing models to make things (APIs, agents, RAG). *Adapting* — change a model's behaviour for your purpose (fine-tuning, RL). *Engineering discipline* — make it reliable and observable in production (harnesses, evals, memory, observability).

**20.** It signals that the instructor regards specific facts, tools, and APIs as short-lived, and treats the durable deliverable as a *process* for absorbing new developments. Content is perishable; the ability to re-learn is not.

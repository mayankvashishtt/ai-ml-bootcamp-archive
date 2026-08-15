# Week 0 — Orientation: What You're Signing Up For

**Date:** 09/01/2026
**Instructor:** Rishabh (Research Engineer, ex-AI Lab London — LLM & LLM-RL; Twitter [@rishabh10x](https://twitter.com/rishabh10x))
**Source:** `downloads/week-00-orientation.pdf` (6 slides)
**Notion page:** https://100xschool.notion.site/2e3ffffa33e5807da538db58eeb3aa52

> **Scope note:** the original deck is 6 slides of housekeeping. Sections 1–4 below cover it completely. **Sections 5–8 are added** — they're the orientation a genuine beginner needs before Week 1 opens with "what is intelligence," and they exist so you can start from zero. If you already know the field's basic vocabulary, skim to §9.

---

## 1. What the course is trying to do

Four stated goals, and they read as a statement of philosophy rather than a syllabus:

1. **Understand what's really going on in this field, and how to adapt.** The emphasis is on adaptation, not memorisation. Techniques that dominate today are frequently obsolete within eighteen months; the underlying mechanisms change far more slowly. The course bets on mechanisms.
2. **Understand how LLMs actually work.** Mechanism, not API surface. This is why Weeks 2–6 grind through neural networks and transformers from scratch before you ever call a production API.
3. **New trends in AI Engineering.** The applied layer — agents, retrieval, fine-tuning, evaluation, observability.
4. **How to stay up to date when new things ship every day.** This goal is why Week 20 is an entire lecture on *how to read research papers*. The course treats "can read a paper" as a deliverable, not a bonus.

**The arc this implies:** fundamentals (W1–7) → building with models (W8–11) → adapting models (W12–16) → engineering discipline around them (W17–25).

---

## 2. Prerequisites and setup

| Requirement | Detail |
|---|---|
| **Basic Python** | Assumed, not taught. Functions, loops, classes, list comprehensions, `pip`. |
| **A Colab account** | Have it working *before* Week 2. 18 of the 26 weeks ship a Colab notebook. |
| **Model exploration** | Actually go and use the models on the market. Form your own opinions about how they differ. |

**On "basic Python":** you need to be able to read and modify code, not write a framework. If you can follow a `for` loop that builds a list, understand what `def f(x): return x * 2` does, and install a package with `pip install`, you're fine. NumPy and PyTorch are taught as you go.

**On Colab:** Google Colaboratory is a free, browser-based notebook environment with GPU access. You don't install anything. This matters because from Week 2 onward you'll be running code that needs a GPU, and buying one is not a prerequisite for this course.

---

## 3. The stance on mathematics

> "Will try to not keep as much math, if any will be done with intuitions."

This is the single most important line on the slides for setting expectations. The course is **intuition-first**. You will meet the equations that matter — `output = activation(Σ wᵢxᵢ + b)`, mean squared error, the attention formula — but they arrive as descriptions of a mechanism you've already understood in words, never as derivations.

**What this means practically:**

- **Don't stall waiting to feel mathematically "ready."** You are not expected to derive backpropagation. You are expected to understand what it does and why it's needed.
- **Do get comfortable *reading* notation**, because the papers in Week 20 won't soften it for you. Specifically: summation (Σ), matrix multiplication, and the idea of a gradient. Each is introduced when first needed.
- **If you want rigour, add it alongside.** The course won't block you, but it won't supply it either.

**The honest trade-off:** intuition-first gets you productive quickly and leaves gaps. You'll be able to build things and reason about them long before you could prove anything about them. For engineering, that's usually the right trade. Know that you're making it.

---

## 4. Tools mentioned

Two free coding agents flagged as worth trying:

- **Opencode** — open-source terminal coding agent.
- **Gemini CLI** — Google's terminal agent, free tier available.

These aren't incidental. Weeks 22 and 25 have you *building* agents; using a mature one first gives you a reference for what good looks like — how it plans, when it decides to call a tool, how it recovers when something fails. It's much easier to build a thing you've used.

---

## 5. The vocabulary, sorted out properly

*(Added — not on the slides. These five terms get used interchangeably everywhere and they don't mean the same thing. Getting them straight now prevents a lot of confusion later.)*

**Artificial Intelligence (AI)** is the **goal**: making machines do things that would require intelligence if a human did them. It's an aspiration, not a technique. Chess engines, spam filters, and ChatGPT are all AI.

**Machine Learning (ML)** is one **method** for achieving that goal: instead of writing rules by hand, you show the machine many examples and let it work out the pattern itself. Most AI you encounter today is ML.

**Deep Learning (DL)** is one **kind** of ML: it uses **neural networks with many layers** ("deep" = many layers). This is what Week 2 builds from scratch.

**Large Language Model (LLM)** is one **application** of deep learning: a very large neural network trained to predict the next word in text. GPT, Claude, Llama, and Gemini are LLMs.

**Generative AI** is a **category of use**: any model that produces new content — text, images, audio, video — rather than just classifying or scoring existing content.

They nest like this:

```
AI  ─── the goal
 └── ML  ─── learning from data instead of hand-written rules
      └── Deep Learning  ─── ML using many-layered neural networks
           └── LLMs  ─── deep learning applied to language
```

**The distinction that matters most for this course** is between *learning from data* and *being programmed*. A spam filter written as `if email contains "free money" then spam` is AI but not ML — a human wrote the rule. A spam filter that was shown a million labelled emails and worked out the pattern itself is ML. Week 1 is largely the story of why the second approach won.

---

## 6. What a "model" actually is

*(Added.)*

The word "model" is used constantly and rarely defined. Here is the concrete version.

A model is **a big pile of numbers, plus instructions for how to combine them with your input.**

That's genuinely it. When people say a model has "7 billion parameters," they mean it contains 7 billion individual numbers. Those numbers are called **weights**. Running the model means doing arithmetic — mostly multiplication and addition — between your input and those weights, producing an output.

**Training** is the process of finding good values for those numbers. You start with random values (the model produces garbage), show it an example, measure how wrong the output was, and nudge every number very slightly in the direction that would have made it less wrong. Repeat billions of times. That's the whole of Week 2.

**Inference** is using the trained model: the numbers are now fixed, and you just do the arithmetic to get an output.

Two consequences worth internalising early:

- **The model doesn't "look anything up."** There's no database inside it. When it tells you the capital of France, that fact is distributed across those billions of numbers as a pattern, not stored in a row somewhere. This is why models can be confidently wrong — a slightly-off pattern produces a fluent, false answer with no internal alarm.
- **Training is expensive; inference is cheap.** Training a frontier model costs millions of dollars and months of compute. Running it costs fractions of a cent. This asymmetry drives essentially the entire economics of the field, and it's why "AI Engineering" (§7) exists as a job.

---

## 7. What "AI Engineering" means, and why it's a distinct job

*(Added.)*

There are broadly three roles in this space, and the course is training you for the third.

**ML Researcher** — invents new architectures and training methods. Publishes papers. Needs deep mathematics.

**ML Engineer** — trains models. Builds data pipelines, runs training jobs, tunes hyperparameters. Needs solid mathematics and serious infrastructure skills.

**AI Engineer** — builds products on top of models that already exist. Chooses models, designs prompts, builds retrieval systems, wires up tools and agents, evaluates quality, controls cost, and ships. Needs strong software engineering and a working understanding of mechanism.

**The third role exists because of the asymmetry in §6.** When a single pretrained model can be adapted to thousands of tasks by prompting it differently, most of the value moves from *making models* to *building with them*. That shift — from task-specific models to general "foundation models" — is the subject of Week 1 §15, and it's the economic reason this course spends Weeks 8–25 building *with* models rather than training them.

**What this course does and doesn't prepare you for:** you will finish able to build and evaluate real systems on top of models, fine-tune them for behaviour, and reason about cost and failure. You will not finish able to train a frontier model from scratch — nobody trains those outside a handful of labs, and the course is honest about that.

---

## 8. How to actually study this

*(Added.)*

**Run the notebooks. Don't just read them.** 18 weeks ship code. Reading code produces a feeling of understanding that evaporates the moment you try to change something. Change something — break it deliberately, then work out why it broke.

**Expect the fundamentals block (W2–7) to be the hardest part.** It's the most abstract and the least immediately useful, and it's where people drop out. It's also what makes everything after it make sense rather than feel like memorised recipes. If you push through one thing, push through Week 4 (attention).

**Don't skip ahead to the agent weeks.** They're the fun ones and they're much less useful without the foundations. An agent that misbehaves is debugged by understanding context, tokens, and tool feedback — all of which come earlier.

**Use the quizzes properly.** Attempt the short-answer questions in writing *before* looking at the key. The multiple-choice questions test recall; the short-answer ones test whether you can explain a mechanism, which is the actual skill.

**Note where the source material is wrong.** Several notebooks and decks in this course contain genuine bugs and overstated claims — they're documented in the [README](../../README.md) and flagged inline in each week's notes. Finding them yourself is good practice for Week 20.

---

## 9. Common confusions

**"Is AI the same as machine learning?"** No. AI is the goal, ML is one method of reaching it. Rule-based systems are AI without ML. See §5.

**"Do I need to be good at maths?"** Not to complete this course. You need to tolerate notation and understand ideas like "gradient" conceptually. You'd need real mathematics to do research.

**"Does the model understand what it's saying?"** This is a genuinely open question and the course is careful not to resolve it too early — Week 1 §8 and §13 deliberately hold the tension. Mechanically, it predicts the next token. Whether doing that extremely well constitutes understanding is contested. (S13 revisits this with actual evidence.)

**"Is a bigger model always better?"** No, and this becomes a running theme. Bigger models cost more to run, and most production systems deliberately use smaller ones where they suffice (S5, S10).

**"If I learn a specific tool, is that enough?"** Tools in this field turn over fast. The instructor's first stated goal is adaptability precisely because the specific libraries taught in Week 21 may be superseded before you use them professionally. Mechanisms last longer than APIs.

---

## Key takeaways

1. The course optimises for **mechanism and adaptability** over breadth of tools — which is why it starts with neural networks rather than with APIs.
2. **Intuition over derivation.** You won't derive backprop, but you must be able to *read* notation by Week 20.
3. **AI ⊃ ML ⊃ Deep Learning ⊃ LLMs.** The key distinction is learning from data versus being programmed by hand.
4. **A model is a large pile of numbers plus instructions for combining them with your input.** Training finds the numbers; inference uses them.
5. **Models don't store facts in rows** — knowledge is a distributed pattern, which is exactly why they can be fluently and confidently wrong.
6. **Training is expensive, inference is cheap.** That asymmetry created the AI Engineer role and this course's whole back half.
7. **Get Python and Colab working now.** Week 2 assumes both.
8. **Start using agents early** so you have a reference for what you'll later build.
9. The instructor's background is **LLM + reinforcement learning**, which explains the unusually heavy RL block (Weeks 15–16) — specialist territory most courses skip.

---

## Glossary

| Term | Meaning |
|---|---|
| **AI** | The goal: machines doing things that would require intelligence in a human. |
| **Machine Learning (ML)** | Learning patterns from examples rather than following hand-written rules. |
| **Deep Learning** | ML using neural networks with many layers. |
| **LLM** | Large Language Model — a large network trained to predict the next token. |
| **Generative AI** | Models that produce new content rather than only classifying existing content. |
| **Model** | A collection of learned numbers (weights) plus instructions for combining them with input. |
| **Parameter / weight** | One of the individual numbers inside a model, adjusted during training. |
| **Training** | Finding good values for the weights by repeatedly measuring error and adjusting. |
| **Inference** | Running a trained model to get an output; weights are fixed. |
| **Foundation model** | A large general-purpose pretrained model adapted to many tasks via prompting. |
| **AI Engineering** | Building products on top of existing models, as opposed to training them. |
| **Colab** | Google Colaboratory — hosted Jupyter notebooks with free GPU access. |
| **LLM-RL** | Applying reinforcement learning to language models to shape behaviour after pretraining. |

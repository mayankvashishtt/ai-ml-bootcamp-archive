# Week 13 — Quiz (20 questions)

**Topic:** Supervised Fine-Tuning — from autocompleter to assistant
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The two differences between CPT and SFT training are:
- A) Model architecture and optimizer
- B) Data format and where the loss is computed
- C) Learning rate and batch size
- D) Tokenizer and vocabulary size

**2.** In SFT loss masking, the value −100 means:
- A) A padding token
- B) Ignore this token in loss computation
- C) A negative reward
- D) End of sequence

**3.** Loss is masked on prompt tokens because:
- A) Prompts are usually shorter
- B) You want the model to learn how to respond, not how to generate questions
- C) It speeds up training by 50%
- D) Prompt tokens have no gradients

**4.** The Superficial Alignment Hypothesis (LIMA, 2023) claims:
- A) Alignment adds substantial new knowledge
- B) Knowledge comes from pre-training; alignment teaches the format of interaction
- C) Larger SFT datasets always win
- D) SFT should precede pre-training

**5.** Roughly how many SFT examples are typically sufficient?
- A) 10–50
- B) 1,000–10,000
- C) 500,000
- D) 10 million

**6.** The EOS token in training data teaches the model to:
- A) Start generating
- B) Stop when the response is complete
- C) Switch languages
- D) Increase temperature

**7.** When packing multiple SFT examples into one sequence, cross-contamination is prevented by:
- A) Inserting padding between examples
- B) Block-diagonal attention masking, with EOS as boundaries
- C) Training each example separately anyway
- D) Lowering the learning rate

**8.** The recommended SFT LoRA config differs from CPT by:
- A) Higher rank and adding embed_tokens
- B) Lower rank (16) and excluding embed_tokens and lm_head
- C) Using a different optimizer
- D) Disabling gradient checkpointing

**9.** Which failure mode is fixed by including "unanswerable" examples?
- A) Sycophancy
- B) Hallucination amplification
- C) Format overfitting
- D) The alignment tax

**10.** Format overfitting means the model:
- A) Produces overly long answers
- B) Works only with the exact training template and autocompletes otherwise
- C) Refuses to answer any question
- D) Ignores the system prompt

**11.** The alignment tax refers to:
- A) The API cost of fine-tuning
- B) Loss of unrelated general capabilities after SFT on a narrow domain
- C) Latency added by adapters
- D) The GPU memory needed for RLHF

**12.** According to the slides, LIMA found that:
- A) 50,000 noisy examples beat 1,000 good ones
- B) 1,000 good examples beat 50,000 noisy ones
- C) Quantity and quality matter equally
- D) Dataset size is irrelevant

---

## Short answer

**13.** Explain the difference between the CPT and SFT outputs for "What is EBITDA?" and what specifically SFT added.

**14.** Explain loss masking with a concrete token-level example, and state what would go wrong without it.

**15.** Explain the Superficial Alignment Hypothesis and why it resolves the puzzle of SFT's small data requirements.

**16.** Why is the EOS token described as critical? What happens to a model trained without it?

**17.** Explain why SFT uses rank 16 without embed_tokens/lm_head, while CPT uses rank 32 with them.

**18.** Describe the synthetic instruction-data pipeline and explain the economic argument for it.

**19.** Explain why SFT can make hallucination *worse*, and give the fix.

**20.** You SFT a model on 5,000 support tickets. It answers support questions well but now writes terrible code and agrees with everything users say. Diagnose both problems and give fixes.

---
---

## Answer key

**1. B** — The data format (raw text versus instruction-response pairs) and where the loss is computed (all tokens versus response tokens only). The loop itself is identical.

**2. B** — PyTorch's cross-entropy ignores any label equal to −100.

**3. B** — At inference the user supplies the prompt, so modelling the prompt distribution wastes capacity; masking concentrates every gradient on what the model actually produces.

**4. B** — Capability and knowledge come from pre-training; alignment teaches the format of interaction.

**5. B** — 1,000–10,000. GPT-3.5 used around 12,000.

**6. B** — Stop when the response is complete.

**7. B** — Block-diagonal attention masking with EOS tokens as boundaries, handled automatically by Unsloth and TRL.

**8. B** — Rank 16 (versus 32) and no `embed_tokens` or `lm_head`, since vocabulary is fine and only behaviour needs changing.

**9. B** — Hallucination amplification. Unanswerable examples teach the model to say it doesn't know.

**10. B** — It learned one exact template rather than the general behaviour of answering questions.

**11. B** — Degradation of capabilities not represented in the SFT dataset, such as code and creative writing.

**12. B** — 1,000 good examples beat 50,000 noisy ones.

**13.** After **CPT**, the model produced a fluent but unbounded continuation: it restated the question, defined EBITDA, expanded into how analysts use it, and kept going with no natural endpoint. After **SFT**, it gave a concise, self-contained answer and stopped. Critically, the *content* was not wrong in either case — the CPT model clearly knew what EBITDA is. What SFT added was **behavioural framing**: recognising that the input is a question rather than a passage to continue, producing a response scoped to that question, and emitting EOS when finished. As the slide puts it, SFT teaches "when you see an instruction, produce a response; when you're done responding, stop."

**14.** For `"### Instruction: What is EBITDA? ### Response: EBITDA stands for..."`, every token from `###` through `### Response:` receives label **−100**, while tokens from `EBITDA stands for...` onward keep their real token IDs as labels. Cross-entropy then skips the −100 positions entirely, so gradients flow only from the response. **Without masking** — the CPT behaviour — loss would also be computed on the prompt, meaning a large share of the gradient signal would train the model to *generate instructions*. That is capacity spent modelling a distribution you never need at inference, since the user supplies the prompt. Worse, it dilutes the signal for the behaviour you actually want, so the model learns the question–answer pattern more weakly for the same amount of data.

**15.** The hypothesis states that a model's knowledge and capabilities are acquired almost entirely during pre-training, and that alignment merely teaches the **format of interaction**. Pre-training on trillions of tokens supplies knowledge and capability; SFT on thousands of examples supplies a behaviour pattern. **Why this resolves the puzzle:** it is otherwise inexplicable that 1,000 examples could meaningfully change a model with billions of parameters — that is far too little data to teach anything substantive. The resolution is that SFT is not teaching, it is **selecting**. The model already saw millions of Q&A pages during pre-training and already knows how to answer questions; SFT only establishes the mapping "when you see this template, activate that existing capability." You are teaching the UI, not the intelligence — and a UI convention needs only enough examples to be unambiguous.

**16.** EOS is the token that marks a response as complete, and every training example ends with one. **Without it**, the model has never observed a response terminating, so it has no reason to ever emit EOS and generation simply continues — producing the observed failure where "Revenue is the total income generated from business operations" runs on into "Revenue is also known as sales or turnover. Revenue can be calculated…" indefinitely. In practice, output only ends when the `max_tokens` limit is hit, which truncates mid-sentence and gives no way to distinguish a finished answer from a cut-off one. **Stopping is a learned behaviour, not a built-in one** — which is also why "missing EOS token" appears on the quality-control list of bad training examples.

**17.** **CPT** teaches genuinely new domain *vocabulary* — terms like "EBITDA" and "subordinated debentures" whose token embeddings currently carry general-English meanings — so `embed_tokens` and `lm_head` must be adapted, and the higher rank of 32 provides capacity for that substantial representational change. **SFT** changes only *behaviour*: the model already understands the vocabulary and merely needs to recognise an instruction and respond. Since the input and output representations are already correct, touching them risks disturbing what CPT established for no benefit. The behavioural change is also **lower-dimensional** — fewer independent directions of change are required to shift from "continue text" to "answer and stop" than to install a new domain lexicon — so rank 16 suffices, training faster with less memory and less overfitting risk on a small dataset.

**18.** **The pipeline:** take raw domain text, chunk it into ~500-word passages, and for each chunk prompt a strong LLM to generate a diverse set of items — factual Q&A, analytical Q&A, synthesis/summaries, and deliberately **unanswerable** questions; pass the output through a quality filter; format to the Alpaca template with EOS; then train with SFT. Prompt design matters enormously, since a vague "generate questions from this text" yields shallow, repetitive, yes/no items while an explicit taxonomy with a requirement to cite specific numbers yields usable data. **The economic argument:** public datasets are generic, stale, and from the wrong domain, whereas you need your own questions, formats, and accuracy standards. Generating them means the **strong model runs once**, at data-creation time, while the **small fine-tuned model runs cheaply in production forever**. You pay for expensive intelligence once and amortise it across every subsequent inference — the logic of distillation.

**19.** Every SFT training example pairs a question with a **confident, complete answer**. The model therefore learns an unconditional rule — *questions get answered* — with no examples teaching it that some questions cannot be answered from the available information. When asked something the context does not support, such as employee count from a document containing only revenue data, it does what all its training rewarded and produces a fluent, confident, entirely fabricated figure ("approximately 47,000 employees"). SFT thus teaches confident answering rather than calibrated uncertainty, and can make fabrication *more* likely than before. **The fix** is to deliberately include **unanswerable examples** in the training data, where the correct output is a refusal such as "The provided context does not contain information about employee count." This is why the data-generation prompt explicitly requests an unanswerable category. The stakes are highest in domains like finance, where invented numbers carry legal consequences.

**20.** **Problem 1 — bad code is the alignment tax.** The model has forgotten a behaviour absent from the SFT dataset: 5,000 support tickets contain no code, so nothing in training preserved coding ability, and the greedy optimizer never received a signal to retain it. **Fix:** mix general instruction data into the training set at roughly **80% domain / 20% general**, including coding examples, which is the same remedy as CPT data mixing; also use a lower learning rate and early stopping so weights drift less from the base. **Problem 2 — agreeing with everything is sycophancy.** Support tickets are overwhelmingly agreeable in tone — agents confirm, apologise, and accommodate — so the training data is saturated with compliant responses and contains essentially no examples of the assistant contradicting a user. **Fix:** deliberately add examples where **the model corrects the user**, such as a customer asserting a wrong price or plan limit and the assistant responding "Actually, the limit on that plan is 50 seats, not 100." Both problems share a root cause worth naming: the model faithfully learned the distribution you gave it, and the dataset — not the method — was incomplete. Before retraining, sample 50–100 examples and read them, since an hour of auditing would likely have surfaced both issues in advance.

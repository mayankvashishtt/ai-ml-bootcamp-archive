# S9 — Quiz (20 questions)

**Topic:** Multimodal Models and Vision
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** A transformer processes an image by:
- A) Using a separate convolutional vision network
- B) Converting it into a sequence of vectors, then processing it like any sequence
- C) Feeding one vector per pixel
- D) Converting it to a text description first

**2.** A 224×224 image with 16×16 patches produces:
- A) 224 tokens
- B) 196 tokens
- C) 50,176 tokens
- D) 768 tokens

**3.** Patch embeddings come from a matrix multiply rather than a lookup table because:
- A) It's faster
- B) There are infinitely many possible patches, so you compute rather than store
- C) Images have three colour channels
- D) Position must be added first

**4.** Position embeddings matter especially for images because:
- A) Images are larger than text
- B) Attention is permutation-invariant, and 2D spatial arrangement carries most of the meaning
- C) Patches have no natural order
- D) RoPE requires them

**5.** CLIP cannot:
- A) Do zero-shot classification
- B) Generate text — it has no decoder
- C) Embed images
- D) Support text-to-image retrieval

**6.** CLIP's training objective is essentially:
- A) Next-token prediction
- B) Contrastive learning across modalities, with in-batch negatives — S6's machinery
- C) Masked patch reconstruction
- D) Reinforcement learning

**7.** The LLaVA-style adapter pattern works by:
- A) Training a multimodal model from scratch
- B) Gluing a pretrained vision encoder to a pretrained LLM with a small trainable projector
- C) Converting images to captions, then prompting
- D) Fine-tuning CLIP on instructions

**8.** Doubling both dimensions of an image changes its token cost by roughly:
- A) 2×
- B) 4×
- C) 1× (unchanged)
- D) 8×

**9.** A VLM reading a value off a dense chart should be:
- A) Trusted — charts are easy for vision models
- B) Verified — it combines small text with precise spatial reasoning, two known weaknesses
- C) Trusted only at high temperature
- D) Avoided entirely; use OCR only

**10.** Set-of-mark prompting helps agents because it:
- A) Reduces image resolution
- B) Converts a hard spatial-regression problem into an easy selection problem
- C) Removes untrusted content
- D) Compresses the screenshot

**11.** Video models miss fast events primarily because:
- A) Temporal attention is weak
- B) Frames are aggressively subsampled, so the frame containing the event was never sampled
- C) Motion blur confuses the encoder
- D) Audio is processed separately

**12.** A screenshot in a computer-use agent should be treated as:
- A) Trusted system input
- B) Untrusted content that can carry prompt injection
- C) A resource, in the MCP sense
- D) Cached context

---

## Short answer

**13.** Explain, in plain language, how an image becomes something a transformer can process, and why that means "image tokens" is literal rather than metaphorical.

**14.** Compare text tokenisation with image patching — what's the same, what differs, and why.

**15.** Explain the three multimodal architecture families and when you'd choose each.

**16.** Explain why VLM failures are systematic rather than random, using counting and small text as examples.

**17.** Explain how image cost scales and give four practical rules for controlling it in an agent loop.

**18.** Explain the trade-off between a dedicated OCR pipeline and a native VLM for document understanding, and why hybrid is usually right.

**19.** Explain why grounding is the hard part of computer use, and what actually helps.

**20.** You're building an agent that processes scanned invoices — extracting vendor, line items, totals, and dates — at ~5,000 documents per day. Design it end to end, including accuracy, cost, and safety.

---
---

## Answer key

**1. B** — Everything reduces to a sequence of vectors; the transformer is unchanged.

**2. B** — 224/16 = 14, and 14×14 = 196.

**3. B** — A finite vocabulary can be stored in a table; the space of possible patches cannot, so a learned projection computes the embedding.

**4. B** — Without position, shuffled patches are indistinguishable, and "sky above grass" would equal "grass above sky."

**5. B** — CLIP scores and retrieves; it has no generative decoder.

**6. B** — Matching pairs are positives, other captions in the batch are negatives, same InfoNCE loss.

**7. B** — Two expensive pretrained models reused, connected by a small trainable projection.

**8. B** — Cost scales with area, so 2× per side is 4× the patches.

**9. B** — Small text plus precise spatial reasoning is the worst-case combination.

**10. B** — Picking "element 7" is far easier than regressing pixel coordinates.

**11. B** — Token cost forces subsampling, often to 1 fps or less.

**12. B** — Rendered text can carry instructions the model reads as readily as yours.

**13.** A transformer has no concept of an image — it only knows how to process a **sequence of vectors**. Text works because a tokenizer splits a string into tokens and an embedding table maps each token to a vector. To make a model see, you do the same thing to a picture: **cut it into a grid of small squares (patches), turn each patch into a vector, and feed the resulting sequence into the same transformer.** A 16×16 patch is a block of 16×16×3 numbers (three colour channels); flatten it to a vector of 768 values and multiply by a learned projection matrix to get an embedding of the model's dimension. Add a position embedding so the model knows where the patch sat in the grid, and the image is now indistinguishable, mechanically, from a short passage of text. **There is no separate vision brain** — attention doesn't care whether a vector came from a word or a patch. This is why "image tokens" is literal: those patch embeddings occupy real positions in the context window, count toward the context limit, and are what you are billed for. A large screenshot genuinely consumes thousands of tokens in the same budget your prompt is drawing from.

**14.** **The same:** both convert raw input into a sequence of fixed-size vectors, both add position information, and both then hand off to an identical transformer. The unit of both is a chunk of the input chosen to be small enough to be meaningful and large enough to be tractable. **What differs:** text embeddings come from a **lookup table**, because the vocabulary is finite — roughly 128,000 possible tokens means you can simply store a vector for each. Patch embeddings come from a **learned linear projection**, because the space of possible 16×16 pixel blocks is effectively infinite and cannot be enumerated, so the embedding must be computed rather than retrieved. Position is **1D for text and 2D for images**, since a patch has a row and a column and "above/below" is as meaningful as "left/right". And the motivation for chunking differs: subword tokenisation exists to balance vocabulary size against sequence length, while patching exists because **pixels are enormously redundant and attention is quadratic** — one vector per pixel on a 1000×1000 image would mean a million-token sequence and a trillion attention operations per layer, which is simply impossible. Patching is a compression step that text doesn't need in the same way.

**15.** **Dual encoders (CLIP)** use separate image and text encoders, each producing one vector, trained contrastively so matching pairs land close and mismatched pairs land far apart. They give you **zero-shot classification** (embed the image, embed candidate label texts, take the nearest) and a **shared image–text space** for cross-modal retrieval. They **cannot generate text** — there is no decoder. Choose this when you need multimodal *search* or classification, which is most retrieval work. **Adapter / projector models (LLaVA pattern)** take a pretrained vision encoder — often CLIP's — and a pretrained LLM, and connect them with a small trainable projection that maps vision features into the LLM's embedding space, so images arrive as ordinary tokens in the sequence. This is cheap and pragmatic, reusing two expensive pretrained models, and it is how most open multimodal models are built. Its limitation is that vision and language were learned separately and meet only at a thin seam, so detail the encoder discarded is gone before the LLM sees anything. Choose it when running open models or fine-tuning your own. **Native / early-fusion multimodal** trains a single unified backbone on interleaved text and images from the start, allowing cross-modal reasoning at every layer rather than one junction. It is the frontier approach and the one you experience through commercial APIs — and it is not something you will train yourself.

**16.** VLM failures follow from two structural properties of how these models are built, which is why they recur predictably and don't respond to better prompting. **First, training captions describe the gist, not the detail.** Web image–text pairs say "a dog on a beach," not "three dogs, the leftmost partly occluded, forty pixels from the left edge." So the objective rewards capturing overall content and provides almost no gradient for precise counting, exact position, or absence. **Counting** fails for exactly this reason: enumerating discrete objects requires tracking each one individually, while the representation encodes an overall impression — models handle 1–4 reasonably and degrade beyond that. **Negation** fails for a related reason: captions describe what *is* present, so "is there no exit sign?" has almost no training signal behind it. **Hallucinating expected objects** — reporting a refrigerator in a kitchen that has none — is the same prior-driven completion that causes text hallucination, since kitchens usually do have one. **Second, patching destroys sub-patch detail.** Anything smaller than a patch has been averaged into a single vector before the transformer sees it. **Small text** is therefore not misread — it is *absent*, and the model fills the gap by guessing from context, which is worse than failing because it fails confidently. **Dense charts** combine both weaknesses at once, requiring small-text reading and precise position-to-value mapping, which is why chart-derived numbers must always be verified. None of these is a bug awaiting a fix; they are properties of the representation, and the practical response is S3's: evaluate on your own images rather than trusting benchmark scores.

**17.** Images are tiled into patches, so **token cost scales with area, not with width** — doubling both dimensions quadruples the count, meaning 512×512 → 1024×1024 is roughly 4× and → 2048×2048 is roughly 16×. Production VLMs handle large images with **tiling / dynamic resolution**, splitting them into fixed-size tiles each encoded separately, often plus a downscaled thumbnail for global context, which is why a high-resolution screenshot can consume a startling number of tokens. **Four practical rules:** **(1) Send the smallest image containing the information** — resizing a 4K screenshot to 1024px is frequently free in accuracy and a 4× saving, though there is a task-dependent floor below which small text is destroyed, so establish it empirically rather than guessing. **(2) Crop rather than resize** when only part of the image matters: a cropped region at full resolution beats the whole image downscaled, winning on both cost and accuracy simultaneously. **(3) Budget screenshots across the loop** — an agent capturing a screenshot every step accumulates image tokens linearly, and twenty full-resolution screenshots is a context-limit problem, a cost problem, and a context-rot problem (Week 10) at once; keep the most recent at good resolution and downscale or drop older ones, since the agent rarely needs step 3's screenshot at step 19. **(4) Order the context for caching** — prompt caching (S5) only helps on a stable prefix, and screenshots change every turn, so put stable instructions and tool definitions first and let the volatile image sit at the end.

**18.** A **dedicated OCR pipeline** runs a purpose-built engine to extract text and layout, then hands *text* to an LLM. Its advantages are real: OCR engines are trained for precisely this task, they emit **character-level confidence scores**, they are cheap, and their output is verifiable against the source. Its weakness is that layout, tables, and figures collapse into linear text, and anything genuinely visual is lost. A **native VLM** takes the page image directly, preserving layout and visual structure, interpreting tables and figures in place, and handling everything with one model — but it is expensive at the resolution documents require, offers **no confidence signal**, and its errors are silent, which is the dangerous property. **Hybrid is usually right for anything that matters** because the two approaches fail differently: run OCR to get a verifiable text layer with confidences, and use the VLM for layout, structure, figures, and the cases OCR fumbles. The genuine payoff is that **disagreement between them is a free error detector** — where OCR and the VLM produce different values for the same field, you have automatic identification of exactly the records needing human review, without having to build a separate confidence model. **Resolution must be settled empirically in either case**, because document tasks show a sharp accuracy cliff as resolution drops and the cliff's position depends on the font sizes in your particular documents.

**19.** Grounding is mapping a language reference — "click the Submit button" — onto an actual location in the image, and it is hard because **precise coordinates are among the weakest things a VLM does**. The representation is trained on gist-level captions that never specify pixel offsets, so asking for coordinates is asking for spatial regression the model was never rewarded for. The failure is also unforgiving: a click 30 pixels off doesn't degrade gracefully, it hits the wrong element or nothing, and the agent then reasons from a corrupted world state. **What actually helps is changing the problem rather than the prompt.** **Set-of-mark prompting** overlays numbered boxes on the interactive elements before sending the screenshot and asks the model to pick a *number* — converting hard spatial regression into easy selection, and it is one of the highest-leverage techniques available. **Accessibility trees or the DOM**, where available in browsers and a11y-aware native apps, provide the structured element list directly; this is far more reliable than pixels, and the right architecture treats the tree as the source of truth with the screenshot as a supplement. **Coordinate-trained models** explicitly trained for UI grounding are genuinely better at raw coordinates — a capability difference, not a prompting one. **Also verify rather than assume**: the model clicks and the next screenshot is the only evidence anything happened, so the loop must check the result, which is Week 17's feedback-quality lesson expressed in pixels. **And treat the screenshot as untrusted** (S4) — a computer-use agent has private access, ingests untrusted rendered content, and can act, which is the lethal trifecta by construction, so sandbox it.

**20.** **Start by narrowing the vision problem, because invoices are a structured-extraction task, not an open-ended one.** Define the target schema up front — vendor, invoice number, date, line items with description/quantity/unit price, subtotal, tax, total — and enforce it with **structured outputs / a strict tool schema** (S7), so the shape is valid by construction and you never write a parser or a retry loop. **Architecture: hybrid OCR + VLM, not either alone.** Run OCR first for a verifiable text layer with **character-level confidences**, and run the VLM on the page image for layout, table structure, and the fields OCR mangles. Where the two disagree on a field, flag that record — this disagreement is the **free error detector** and it should drive the human-review queue, not a guess about model confidence. **Exploit the arithmetic that makes invoices unusually checkable:** line items must sum to the subtotal, subtotal plus tax must equal the total. Verify this **in code**, not in the prompt. A document that fails arithmetic is a document that was extracted wrong, and this catches a large fraction of errors with zero model involvement — the single highest-value component of the design. **Cost and resolution:** cost scales with area, and 5,000 documents/day makes this the dominant line item, so establish the **resolution floor empirically** on a sample of your worst scans rather than defaulting to full resolution. Crop to the regions that matter where invoice layouts are consistent per vendor. Put the stable instructions and schema at the front of the context so **prompt caching** (S5) applies across the whole batch, and run the workload through the **Batch API** if latency permits, since this is a nightly-processing shape rather than an interactive one. Consider routing by difficulty: clean digital PDFs may not need the VLM at all, and reserving it for scans is a large saving. **Accuracy engineering:** build a **labelled eval set** of real invoices including the ugly ones — skewed scans, handwriting, multi-page, unusual vendors, foreign currencies — and measure **per-field accuracy**, since an aggregate number hides that dates are fine and line items are broken (S3). Report confidence intervals and use enough examples that a 2-point difference means something. Expect and specifically test the known failure modes: **small text** in fine print, **dense tables** of line items, and **counting** the number of rows. Ask the model to **quote the source text** for each extracted field so errors become visible, and explicitly permit "unreadable" so an illegible scan produces a flag rather than a confident fabrication. **Safety and operations:** invoices are **untrusted content** (S4) — a supplier can put text on a document, and a PDF can contain rendered instructions the model reads as readily as yours, which matters enormously if extraction feeds an automated payment system. Never let extraction trigger a payment directly; require approval on any new vendor, any new bank details, and any amount over a threshold, since **vendor bank-detail modification is the actual fraud vector here**. Log every document, extraction, confidence, and disagreement (Week 23) — you will need it for both debugging and audit. **Route to humans deliberately:** arithmetic failures, OCR/VLM disagreements, low OCR confidence, new vendors, and out-of-distribution totals. The goal is not full automation but a well-calibrated review queue that shrinks as you learn which vendors are reliable.

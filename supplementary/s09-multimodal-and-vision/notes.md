# S9 — Multimodal Models: How a Model Sees

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. The entire course is text-only — and yet Week 25 teaches computer-use agents, which work by **looking at screenshots**. Nothing in the curriculum explains how an image becomes something a transformer can process, why vision models fail in the specific ways they do, or what an image costs you. This lecture fills that in.

**Fills the gap after:** Week 25 (Computer Use Agents) — but readable any time after Week 4
**Prerequisites:** Weeks 3–4 (tokens, embeddings, attention), S6 (contrastive learning) helps for §4

---

## 0. The idea in plain language

Start with the thing that trips everyone up: **a transformer has no idea what an image is.** It only knows how to process a *sequence of vectors*. Text works because a tokenizer chops a string into tokens and an embedding table turns each token into a vector — sequence of vectors, done.

So to make a model "see," you need to do the same thing to a picture: **turn it into a sequence of vectors.**

That's the whole trick. There is no separate "vision brain." You cut the image into little squares, turn each square into a vector, and feed that sequence into the same machinery you already understand. An image becomes, quite literally, **a bunch of tokens** — and once it's tokens, attention doesn't care whether they came from a sentence or a screenshot.

Everything else in this lecture is a detail on top of that one idea:
- How you cut it up (§2)
- How the model knows *where* each piece was (§3)
- How the vision part and the language part get connected (§4)
- What it costs (§6)
- What breaks (§7)

---

## 1. Why "just add pixels" doesn't work

The naive idea is to feed the model one vector per pixel. A 1000×1000 image is a million pixels. Attention is quadratic in sequence length, so a million-token sequence is a trillion attention operations per layer. Completely impossible.

You need to compress. The insight — from **Vision Transformers (ViT, Dosovitskiy et al., 2020)** — is that pixels are enormously redundant. Neighbouring pixels are almost always similar. So don't treat pixels as the unit; treat **patches** as the unit.

---

## 2. Patches: chopping an image into tokens

Take a 224×224 image and cut it into a grid of 16×16 pixel squares:

```
224 / 16 = 14   →   a 14 × 14 grid   →   196 patches
```

196 tokens. That's a short paragraph. Now it's tractable.

Each patch is a 16×16×3 block of numbers (three colour channels) = 768 raw values. Flatten it into a vector of length 768 and multiply by a learned **projection matrix** to get an embedding of whatever size the model wants:

```
patch (16×16×3)  →  flatten [768]  →  × W [768, d_model]  →  patch embedding [d_model]
```

Compare this to text:

| | Text | Image |
|---|---|---|
| Raw input | A string | A pixel grid |
| Unit | Token (subword) | Patch (16×16 pixels) |
| Lookup | Embedding table | Learned linear projection |
| Result | Vector per token | Vector per patch |
| Then | Transformer | Transformer (the same one) |

The one real difference: text embeddings come from a **lookup table** (there are only ~128,000 possible tokens, so you can store a vector for each), while patch embeddings come from a **matrix multiply** (there are infinitely many possible patches, so you compute rather than look up). Conceptually they do the same job.

**This is why "image tokens" is not a metaphor.** When an API bills you for an image, it is billing you for real tokens occupying real space in the context window.

---

## 3. Position: how the model knows where a patch was

Attention is permutation-invariant (Week 3's point about why position embeddings exist at all). Shuffle the patches and, without positional information, the model sees the same thing. That's fatal for images — "sky above grass" and "grass above sky" would be identical.

So each patch embedding gets a **position embedding** added, exactly as in Week 3. The only wrinkle is that image position is **two-dimensional**: a patch is at row 4, column 7, not just "position 53."

Approaches you'll meet:
- **Learned 1D position embeddings** over the flattened patch order — simple, and works surprisingly well because the model learns the grid structure implicitly.
- **2D position embeddings**, encoding row and column separately.
- **2D RoPE**, extending Week 6's rotary embeddings to two axes — increasingly common, and it inherits RoPE's better handling of resolutions not seen in training.

The practical consequence: **models are sensitive to resolution changes** because the position scheme was trained at a particular grid size. Feed a much larger image and you're extrapolating positions, which is exactly the situation Week 6 discussed for long text contexts.

---

## 4. Connecting vision to language — the three architectures

You now have image patches as vectors. How do they meet the language model? Three families, and they answer different questions.

### (a) Dual encoder — CLIP

Two separate encoders: one for images, one for text. Both produce a single vector. Train them **contrastively** so that matching image–text pairs land close together and mismatched pairs land far apart.

**This is precisely S6's machinery**, applied across modalities instead of within one. Given a batch of N image–caption pairs, each image's correct caption is the positive and the other N−1 captions are in-batch negatives. Same InfoNCE loss, same reason batch size matters for quality.

**CLIP (Radford et al., 2021)** trained this on ~400M image–text pairs from the web and produced two useful things:
- **Zero-shot classification** — embed the image, embed the candidate labels as text ("a photo of a dog", "a photo of a cat"), take the nearest. No classifier head, no fine-tuning.
- **A shared image–text embedding space**, which makes cross-modal search work: text query → image results.

**What CLIP cannot do is generate text.** It has no decoder. It scores and retrieves. Its real legacy is that its image encoder became the vision backbone for most of what came next.

### (b) Adapter / projector — the LLaVA pattern

Take a pretrained vision encoder (often CLIP's) and a pretrained LLM, and **glue them together with a small trainable projection**:

```
image → vision encoder → patch features → projector (MLP) → "image tokens"
                                                                   ↓
                    text tokens ────────────────────────→  [ LLM ]  → text out
```

The projector's only job is to map vision features into the LLM's embedding space, so that from the LLM's point of view the image arrives as a handful of ordinary tokens sitting in the sequence alongside the text.

This is **cheap and pragmatic** — you reuse two expensive pretrained models and train a small connector, sometimes with light fine-tuning of the LLM afterwards. It's how most open multimodal models are built (LLaVA, and the broad family it inspired).

Its limitation is that vision and language were learned separately and only ever meet at this thin seam. Fine visual detail that the encoder discarded is gone before the LLM sees anything.

### (c) Native / early-fusion multimodal

Train the model on interleaved text and images **from the start**, with a single unified backbone. There's no bolted-on adapter — the model has been multimodal for its whole life. This is broadly the direction the frontier models have gone.

The advantage is deeper integration: the model can reason across modalities at every layer rather than at one junction. The cost is that you must train the whole thing, which is not a thing you will do.

**As a user of an API, you mostly experience (c).** As someone running open models, you mostly encounter (b). As someone building retrieval, you probably want (a).

---

## 5. What "understanding" actually looks like inside

Worth being precise, because the marketing language is loose.

A vision-language model does **not** parse an image into a scene graph of objects and relations. It converts patches into vectors and lets attention relate them to each other and to the text tokens. What emerges is a statistical association between visual patterns and language, learned from hundreds of millions of image–caption pairs.

That has two consequences that explain nearly every failure in §7:

1. **Captions describe the gist, not the detail.** Web captions say "a dog on a beach," not "three dogs, the leftmost partly occluded, 40 pixels from the left edge." So the training signal rewards gist and largely ignores precise counting, position, and small text.
2. **Patching destroys sub-patch detail.** Anything smaller than a patch has been averaged into a single vector. If 8-point text falls inside a 16×16 patch that got downsampled, that text does not exist as far as the model is concerned.

Neither is a bug to be fixed with a better prompt. They're properties of the representation.

---

## 6. What an image costs

Images are tokens, tokens cost money and context, and the arithmetic surprises people.

A rough model: an image is tiled into patches, and **cost scales with area, not with width**. Doubling both dimensions of an image quadruples its token count.

```
 512 ×  512  →  ~1×  cost
1024 × 1024  →  ~4×  cost
2048 × 2048  →  ~16× cost
```

Most production VLMs handle large images with **tiling / dynamic resolution**: the image is split into several fixed-size tiles, each encoded separately, often alongside a downscaled "thumbnail" of the whole image so the model retains global context. This is why a high-resolution screenshot can consume a startling number of tokens.

**Practical rules that follow directly:**

- **Send the smallest image that contains the information.** Resizing a 4K screenshot to 1024px before sending is often free in accuracy and a 4× saving in cost.
- **But don't over-shrink text.** If the task is reading small labels, downscaling destroys the very signal you need. There's a floor, and it's task-dependent — test it.
- **Crop instead of resizing** when you only care about part of the image. A cropped region at full resolution beats the whole image at low resolution, for both cost and accuracy.
- **Images blow up multi-turn agent loops.** A computer-use agent (Week 25) that takes a screenshot every step accumulates image tokens linearly. Twenty steps of full-resolution screenshots is a context-window problem, a cost problem, and a context-rot problem (Week 10) simultaneously. Drop or downscale old screenshots — the agent rarely needs step 3's screenshot at step 19.
- **Prompt caching applies** (S5), but only to a stable prefix. Screenshots change every turn, so they sit at the *end* of the context and cache poorly by nature. Put stable instructions first.

---

## 7. What vision models reliably get wrong

Knowing the failure list is more useful than knowing the architecture, because these failures are *systematic* — they're not random flakiness, and they don't go away with a better prompt.

**Counting.** Ask "how many chairs?" and expect confident errors past small numbers. Counting requires enumerating discrete objects; the representation encodes gist. Models are decent at 1–4 and unreliable beyond.

**Precise spatial relations.** "Is the cup to the left of the laptop?" is often right; "is it 200 pixels or 400 pixels from the edge?" is not. Coordinates are especially weak — which matters enormously for agents (§9).

**Small text.** Sub-patch text is gone. If it's small in the image, the model isn't reading it — it's guessing from context, which is worse than failing, because it fails *confidently*.

**Dense charts and tables.** A chart requires reading axes, mapping position to value, and tracking series — precise spatial reasoning on small text, i.e. two weaknesses at once. Models will produce plausible numbers that are wrong. **Never trust a number a VLM read off a chart without verification.**

**Negation and absence.** "Is there no exit sign?" is much harder than "is there an exit sign?" The training signal is captions describing what *is* present; absence is almost never captioned.

**Fine-grained discrimination.** Distinguishing similar species, near-identical UI states, or two versions of a document is unreliable unless the model was specifically trained on it.

**Hallucinating expected objects.** Shown a kitchen, models will sometimes report a refrigerator that isn't there, because kitchens usually have one. This is the same prior-driven completion that causes text hallucination, expressed visually.

**The meta-lesson**, and it's S3's: **evaluate on your images.** Benchmark scores tell you very little about how a model handles your specific screenshots, documents, or products.

---

## 8. Documents and OCR

Document understanding is the most common real multimodal use case, and it deserves its own treatment because "just send the PDF to a VLM" is a decision, not a default.

**Two approaches:**

**Dedicated OCR pipeline** — a purpose-built OCR engine extracts text and layout, then you hand *text* to an LLM. Advantages: OCR engines are trained for exactly this, they output character-level confidence, they're cheap, and the output is verifiable. Disadvantages: layout, tables, and figures degrade into linear text, and you lose anything visual.

**Native VLM** — send the page image directly. Advantages: layout and visual structure are preserved, tables and figures are interpretable in place, and one model handles everything. Disadvantages: expensive at the resolution documents need, no character-level confidence, and errors are silent.

**The honest recommendation** is a hybrid for anything that matters: run OCR for the verifiable text layer, and use the VLM for layout, structure, figures, and anything OCR fumbles. Where the two disagree, that disagreement is a useful signal to flag for review — and it's the only free error detector you get.

**Always resolve the resolution question empirically.** Document tasks have a sharp accuracy cliff as resolution drops, and where the cliff sits depends on the font size in *your* documents.

---

## 9. Vision in agents — the Week 25 connection

A computer-use agent runs a loop: **screenshot → model decides → action → screenshot**. Everything above now bites at once.

**Grounding is the hard part.** The model must map "click the Submit button" onto actual pixel coordinates, and §7 established that precise coordinates are a weak spot. Approaches that help:

- **Set-of-mark prompting** — overlay numbered boxes on the interactive elements before sending the screenshot, then ask the model to pick a *number* rather than a coordinate. This converts a hard spatial-regression problem into an easy selection problem, and it is one of the highest-leverage tricks in the area.
- **Accessibility trees / DOM** — when available (browsers, native apps with a11y APIs), the structured element tree is far more reliable than pixels. Use it and treat the screenshot as a supplement, not the source of truth.
- **Coordinate-trained models** — some models are explicitly trained for UI grounding and are meaningfully better at raw coordinates. This is a real capability difference, not a prompting difference.

**Verify actions rather than assuming them.** The model clicks, and the next screenshot is the only evidence anything happened. Build the loop so it checks the result — Week 17's feedback-quality lesson, in pixels.

**Manage the screenshot budget.** As in §6: keep the most recent screenshot at good resolution, downscale or drop older ones, and don't let twenty turns of 4K images crowd out the actual task.

**Security note (S4):** a screenshot is untrusted content. Text rendered in a webpage, a document, or a UI can carry a prompt injection, and the model reads it just as readily as it reads your instructions. A computer-use agent is, by construction, an agent with private access, exposure to untrusted content, and the ability to act — **the lethal trifecta by default**. Sandbox it.

---

## 10. Audio and video, briefly

**Audio** follows the same pattern: convert the waveform to a spectrogram (a 2D image of frequency over time), patch it, and feed it in. Speech models like Whisper are encoder-decoder transformers doing exactly this. Native audio models skip the transcription step and process audio tokens directly, which preserves tone, emphasis, and speaker identity that a transcript throws away.

**Video** is the hard one, because it's images times time and the token cost is brutal. One second of 30fps video at 200 tokens per frame is 6,000 tokens; a minute is 360,000. So every practical video model **subsamples frames aggressively** (often 1 fps or less) and may compress across time. The consequence to remember: **models miss fast events**, because the frame containing the event was very likely never sampled.

---

## 11. Prompting for vision

S2's principles apply, with additions specific to images:

- **Put the image before the question** where the API allows. The model reads sequentially; the question lands better with the visual context already in place.
- **Ask for structured extraction**, not free description. "List every form field with its label and current value as JSON" beats "describe this form" — and it composes with structured outputs (S7).
- **Ask it to quote what it reads.** If the model must reproduce the exact text it's basing an answer on, its errors become visible instead of silent.
- **Tell it to say when it can't tell.** VLMs are strongly biased toward answering. Explicitly permitting "the text is too small to read reliably" converts a confident hallucination into a useful signal.
- **Decompose.** "Read the axis labels" then "read the values" outperforms "interpret this chart" for the same reason chain-of-thought helps in text.
- **Multiple images need labels.** Say "Image 1: before. Image 2: after." Without labels, the model will conflate them.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 3** (tokens, embeddings, position) | Patches *are* tokens; the projection replaces the embedding table; position embeddings become 2D. |
| **Week 4** (attention) | Unchanged. Attention over image patches is the same operation, which is precisely the point. |
| **Week 6** (RoPE, GQA) | 2D RoPE extends Week 6's rotary embeddings to images; resolution extrapolation is the same problem as context-length extrapolation. |
| **Week 9–10** (RAG) | CLIP-style shared embedding spaces enable multimodal retrieval — text queries over images, or retrieval over document pages. |
| **Week 10** (context rot) | Screenshot accumulation in agent loops is context rot with a large constant factor. |
| **Week 17** (harness) | Grounding aids (set-of-mark, a11y trees) are harness design, not model capability. |
| **Week 25** (computer use) | This lecture is the missing foundation: how the agent sees, why grounding is hard, what screenshots cost. |
| **S2** (prompt engineering) | Structured extraction, decomposition, and permission-to-abstain all transfer directly. |
| **S3** (evaluation) | Vision failures are systematic, so benchmark scores generalise poorly. Evaluate on your images. |
| **S4** (safety) | Screenshots are untrusted content; computer-use agents assemble the lethal trifecta by default. |
| **S5** (inference cost) | Images are tokens. Resolution decisions are cost decisions. Screenshots cache poorly. |
| **S6** (embeddings) | CLIP is contrastive learning across modalities — same InfoNCE, same batch-size effect, same hard-negative logic. |

---

## Key takeaways

1. **An image becomes a sequence of vectors, and then it's just a transformer.** Patches are tokens. There is no separate vision brain.
2. **Patches exist because pixels are redundant and attention is quadratic.** 224×224 at 16×16 patches is 196 tokens instead of 50,176 pixels.
3. **Position embeddings matter more for images than text**, because 2D spatial arrangement is most of the meaning, and resolution changes are position extrapolation.
4. **Three architectures:** CLIP-style dual encoders (retrieval, zero-shot, no generation), adapter/projector models (cheap, most open models), and native multimodal (deepest integration, frontier).
5. **CLIP is S6's contrastive learning across modalities** — the same loss, the same reason batch size is a quality knob.
6. **Cost scales with area.** Double the dimensions, quadruple the tokens. Crop before you resize; resize before you send.
7. **The failures are systematic, not random:** counting, precise coordinates, small text, dense charts, negation, and hallucinating expected objects. They follow from gist-level captions and sub-patch information loss.
8. **Never trust a number a VLM read off a chart** without verification.
9. **For documents, hybrid OCR + VLM beats either alone**, and their disagreements are a free error detector.
10. **For agents, convert spatial regression into selection** — set-of-mark prompting and accessibility trees beat asking for raw coordinates.
11. **Video models subsample frames**, so they miss fast events by construction.
12. **A screenshot is untrusted input.** Computer-use agents are the lethal trifecta assembled by default.

---

## Glossary

| Term | Definition |
|---|---|
| **Patch** | A fixed-size square of pixels (commonly 16×16) treated as one token |
| **ViT (Vision Transformer)** | Architecture applying a standard transformer to sequences of image patches |
| **Patch embedding** | Vector produced by flattening a patch and applying a learned linear projection |
| **CLIP** | Contrastive Language–Image Pretraining; dual encoders trained to align image and text vectors |
| **Zero-shot classification** | Classifying by comparing an image embedding to embeddings of candidate label texts |
| **Projector / adapter** | Small trainable network mapping vision features into an LLM's embedding space |
| **Early fusion / native multimodal** | Training a single backbone on interleaved modalities from the start |
| **Dynamic resolution / tiling** | Splitting a large image into fixed-size tiles, often with a global thumbnail |
| **Image tokens** | The literal tokens an image occupies in the context window |
| **Grounding** | Mapping a language reference ("the Submit button") onto a location in an image |
| **Set-of-mark prompting** | Overlaying numbered markers on elements so the model selects a number, not a coordinate |
| **Accessibility tree** | Structured representation of UI elements, more reliable than pixels for grounding |
| **Spectrogram** | 2D frequency-over-time representation of audio, processed like an image |
| **Frame subsampling** | Taking a fraction of video frames to control token cost, at the price of missing fast events |

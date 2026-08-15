# Week 11 — Quiz (20 questions)

**Topic:** Recursive Language Models · Claude Code vs Cursor
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The core RLM reframe is that the prompt becomes:
- A) A longer context window
- B) A variable the model interacts with via code
- C) A set of pre-computed embeddings
- D) A fine-tuning dataset

**2.** In RLM, the Root LLM receives:
- A) The entire input document
- B) Only the query — not the massive input directly
- C) The top-K retrieved chunks
- D) A vector index

**3.** Which is NOT one of the RLM tools listed?
- A) `context[:1000]`
- B) `re.findall(...)`
- C) `rlm_agent(q, chunk)`
- D) `embed(context)`

**4.** RLM avoids context rot because:
- A) It uses a larger attention window
- B) No single LLM call ever sees a large context
- C) It compresses tokens before attention
- D) It disables softmax normalisation

**5.** The cost advantage of RLM comes from:
- A) Using cheaper models throughout
- B) Filtering with free string operations instead of expensive transformer passes
- C) Caching all responses
- D) Batching requests

**6.** RLM(GPT-5-mini) versus vanilla GPT-5 reportedly gained:
- A) +5.1 points
- B) +19.5 points
- C) +28.3%
- D) +34.2 points

**7.** Claude Code's code-search toolkit consists of:
- A) Embeddings and a vector DB
- B) `ls`, `glob`, `grep`, `cat`
- C) Tree-sitter and Merkle trees
- D) A pre-built symbol index

**8.** Agentic search is precise for code primarily because:
- A) Code files are small
- B) Code has exact symbols, and grep matches exactly with zero ambiguity
- C) Codebases are always well documented
- D) Grep understands syntax

**9.** Cursor uses Tree-sitter in order to:
- A) Detect which files changed
- B) Chunk code at syntactic boundaries so logic isn't split mid-function
- C) Compress embeddings
- D) Rank search results

**10.** Cursor uses Merkle trees to:
- A) Store embeddings efficiently
- B) Detect exactly which files changed for incremental re-embedding
- C) Parse code into ASTs
- D) Deduplicate search results

**11.** Which is listed as a struggle of Claude Code's approach?
- A) Index staleness
- B) Hours-long initial setup
- C) Token burn from grep results loading into context
- D) Semantic noise

**12.** The stated 2026 consensus is:
- A) Agentic search has won outright
- B) Indexed RAG has won outright
- C) The best teams use both, for different workflows
- D) Both approaches are being replaced by fine-tuning

---

## Short answer

**13.** Explain how RLM differs from RAG in *who* decides what the model sees, and why that's a categorical rather than incremental difference.

**14.** Walk through the four steps of the ACME Q3 APAC revenue example, naming the operation at each step.

**15.** Explain the recursion tree and why it makes 100M-token inputs tractable. Does total work decrease?

**16.** Explain the claim "only gold tokens hit the GPU" and why it produces a cost advantage.

**17.** "A mini model using RLM beats a full model with raw context." Explain why this is the lecture's central claim.

**18.** Give the three reasons agentic search works well for code, and connect the first to Week 9's findings.

**19.** Describe Cursor's four engineering techniques and what problem each solves.

**20.** Give one thing grep-based search genuinely cannot do that semantic search can, and explain how you'd design a system capturing both strengths.

---
---

## Answer key

**1. B** — The prompt is stored as a variable the model manipulates via code, rather than being fed through attention.

**2. B** — Only the query. The root model acts as a controller and never processes the full input.

**3. D** — `embed(context)` is not an RLM tool; RLM is explicitly vectorless. The others are peek, search, and recurse.

**4. B** — Each call sees only a small, focused slice, so the problem is avoided by construction rather than mitigated.

**5. B** — Irrelevant text is filtered by string operations, which are effectively free, instead of by the transformer, which is expensive.

**6. D** — +34.2 points.

**7. B** — `ls`, `glob`, `grep`, `cat`. No embeddings, no index.

**8. B** — Code consists of exact symbols, so grep gives exact matches with zero ambiguity, whereas embeddings return fuzzy positives.

**9. B** — Parsing to ASTs so chunks land on function and class boundaries rather than splitting logic in half.

**10. B** — Detecting exactly which files changed, so only modified files are re-embedded.

**11. C** — Token burn. Index staleness, setup time, and semantic noise are Cursor's struggles.

**12. C** — The best teams use both: Claude Code for deep autonomous multi-file work, Cursor for quick semantic discovery and interactive iteration.

**13.** In **RAG**, a pipeline decides what the model sees: chunking, embedding, similarity search, and reranking all run *before* the model is invoked, and it then receives a fixed set of chunks with no ability to ask for more or reject what arrived — the "passive model" problem from Week 10. In **RLM**, the model itself decides, *at inference time*, by writing code against its context: it can peek at the start, regex for a pattern, slice a range, inspect what came back, and issue a different query based on what it just learned. **Why this is categorical rather than incremental:** every RAG improvement — hybrid search, contextual retrieval, reranking — optimises a *one-shot guess* made before the model has seen anything. RLM replaces the one-shot guess with an *iterative feedback loop*, so the model can recover from a bad first query, which no amount of retrieval tuning enables. It moves the decision from the pipeline to the model.

**14.** **Step 1 — Initial peek:** read `context[:2000]` to find the table of contents and understand the document's structure. **Step 2 — Targeted search:** run `re.findall(r'APAC|Asia Pacific', context)` to locate the offsets of relevant passages. **Step 3 — Context slicing:** slice `context[45000:52000]` and pass that region to a sub-LLM via `rlm_agent` to extract the specific Q3 figure. **Step 4 — Final synthesis:** the root LLM receives the sub-agent's extracted answer and composes the final response. Throughout, the root model never processes the full 500 pages.

**15.** The tree has the root at depth 0 issuing `rlm_agent(query, chunk_n)` calls at depth 1, each of which may spawn further calls at depth 2, and so on; results propagate back up and are synthesised at each level. It makes enormous inputs tractable because **no individual LLM call ever holds a large context** — each sees a small, focused slice where accuracy is high — so the failure mode that kills long-context prompts never arises, and depth can simply increase as input grows. **No, total work does not decrease.** The same volume of text must still be examined, and there are now many LLM calls instead of one, so aggregate token consumption may even rise. What changes is the *distribution*: work is split across many small, reliable calls rather than concentrated in one large unreliable one, and cheap string operations eliminate most text before any model call happens. The win is reliability and scalability, not less computation.

**16.** In a traditional long-context prompt, **every** token passes through the transformer, so you pay full attention cost on 500 pages even though perhaps two paragraphs matter. In RLM, the bulk of the document is filtered by **Python string operations** — slicing, regex, indexing — which run on CPU at negligible cost and never touch the model. Only the small surviving region, the "gold," is sent through an LLM call. **The cost advantage** follows from the asymmetry: a regex over ten million characters costs effectively nothing, while pushing ten million characters through attention is enormously expensive and, per Week 4, grows quadratically with sequence length. Moving the filtering step from GPU to CPU is precisely where the savings come from — and it is also why RLM can be *cheaper* than vanilla long context while being more accurate.

**17.** It claims that **how a model relates to its context matters more than how capable the model is** — a smaller, weaker model with a good context strategy outperformed a larger, stronger model given the same information as raw text. This is the lecture's thesis because it inverts the default assumption that quality problems are solved by using a better model or a bigger window. Week 10 established that more context can actively *hurt* through context rot and distractors; this result is the constructive consequence: since the frontier model is bottlenecked by its relationship to context rather than by its reasoning, fixing that relationship yields more than upgrading the model. It also has a practical implication — architecture is something an engineer controls, whereas model capability is something they buy — which is why the closing slide says understanding this "separates someone who uses AI tools from someone who builds them."

**18.** **(i) Precision** — code consists of exact symbols, so `grep` returns exact matches with zero ambiguity, whereas embeddings produce "fuzzy positives" that are similar but technically wrong. **(ii) Freshness** — code changes constantly, and agentic search reads the actual current file, with no waiting for re-indexing, re-chunking, or re-embedding after every save. **(iii) Security and simplicity** — there is no index to build or maintain and no sensitive code shipped to a third-party vector database; Boris Cherny notes Anthropic did not want to upload its own codebase anywhere. **Connection to Week 9:** that lecture showed semantic search failing specifically on exact identifiers — dates, IDs, acronyms, commands like `RS256` and `2024-09-15` — because embeddings encode topic rather than identity. Code is almost *entirely* such identifiers: a function name is not a concept to be approximated, it either matches or it does not. Code is therefore close to the worst possible domain for dense retrieval and the best possible one for keyword search.

**19.** **Smart chunking (Tree-sitter):** parses source into ASTs and chunks at syntactic boundaries such as functions and classes — solving the Week 9 chunking problem, since a mid-function split leaves both halves meaningless. **Incremental sync (Merkle trees):** hashes the file tree so exactly which files changed can be identified, re-embedding only those — solving index staleness cheaply, since a full re-index on every save would be impossible. **Team reuse:** clones of a repository share about 92% similarity, so a new team member can reuse an existing teammate's index — solving the initial setup cost, which can run to hours on large repositories. **Semantic search:** enables conceptual queries like "Where do we handle rate limiting?" without knowing any function name — solving the discovery problem that exact-match search cannot address, and reported to improve accuracy by 12.5%.

**20.** **What grep cannot do: find code whose name you don't know.** Searching `grep "rate limit"` misses a function called `throttleRequests`, a class named `Bucket`, or a middleware called `guard` — all of which implement rate limiting without containing the phrase. Exact matching requires you to already know the vocabulary the codebase uses, which is precisely what a newcomer, or anyone exploring unfamiliar code, lacks. **A combined design:** use **semantic search for discovery and grep for verification**. Expose both as tools to the agent and let it choose — semantic search over AST-aligned chunks to surface *candidate* locations for conceptual queries, then `grep` and `cat` on the real files to confirm, expand, and trace callers, so the final context always comes from current source rather than a possibly-stale index. This inherits semantic search's conceptual reach while retaining agentic search's precision and freshness, and mirrors the lecture's own conclusion that the best teams use both — with the refinement that a single agent can arbitrate per query rather than the human switching tools.

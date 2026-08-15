# Week 8 — Quiz (20 questions)

**Topic:** From APIs to Agents — function calling and the ReAct loop
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** The course defines an agent as:
- A) A fine-tuned model with a long context window
- B) LLM + Tools + Loop
- C) A model that has been given internet access
- D) A multi-model ensemble with a router

**2.** The three steps of the ReAct loop are:
- A) Plan, Execute, Verify
- B) Thought, Action, Observation
- C) Retrieve, Augment, Generate
- D) Prompt, Sample, Decode

**3.** When a model uses function calling, it:
- A) Directly executes the function in its own runtime
- B) Generates structured JSON requesting a call, which your code executes
- C) Compiles the function into its weights
- D) Sends an HTTP request itself

**4.** In the tool-call response, `finish_reason` was `tool_calls` and `content` was:
- A) The full answer
- B) `None`
- C) An error message
- D) The tool's output

**5.** In the multi-turn demo, prompt tokens grew from 23 to 184 because:
- A) The model stored the conversation internally
- B) The entire conversation history is re-sent on every call
- C) The second question was longer
- D) Temperature was increased

**6.** Tool `description` fields matter primarily because:
- A) They are shown in API documentation
- B) They are the only information the model uses to decide whether and how to call the tool
- C) They are required by JSON Schema validation
- D) They determine execution order

**7.** When the agent requests an unknown tool, the loop:
- A) Raises an exception and terminates
- B) Returns an error string as an OBSERVATION so the model can recover
- C) Silently skips the iteration
- D) Restarts the conversation

**8.** `max_iterations=10` exists to:
- A) Improve answer quality
- B) Prevent infinite loops
- C) Limit the temperature
- D) Cap the number of tools

**9.** In the `123*123` test, iteration 1 produced:
- A) The correct answer immediately
- B) A format issue that triggered a nudge
- C) An unknown-tool error
- D) A timeout

**10.** Which capability does the code agent's system prompt explicitly require after writing a file?
- A) Commit the file to git
- B) Run `run_python` to test it
- C) Read the file back
- D) Ask the user for confirmation

**11.** The three stages of the 2021 → 2025 evolution are:
- A) Rules → Statistics → Neural
- B) Autocomplete → Chat assistant → Agent
- C) Small → Medium → Large models
- D) Local → Cloud → Edge

**12.** Which is NOT one of Claude Code's four listed tools?
- A) `read_file(path)`
- B) `run_command(cmd)`
- C) `search_code(query)`
- D) `deploy_service(config)`

---

## Short answer

**13.** Explain the claim "the model never calls anything." What actually happens, and why does this matter for security?

**14.** Explain why "the model has NO memory" and list three practical consequences.

**15.** Give the three reasons the agent loop outperforms one-shot prompting. Which is most fundamental, and why?

**16.** Explain why the ReAct system prompt "IS the architecture," and what follows about model compatibility.

**17.** Why are tool errors returned as OBSERVATIONs rather than raised as exceptions? Give a concrete example.

**18.** Explain the context engineering problem in agent loops: why it happens, what it costs, and two mitigations.

**19.** What distinguishes a "code agent" from a code generator? Reference the specific system prompt rules.

**20.** Audit the notebook's code-agent tools for safety. Identify three risks and how you would mitigate each.

---
---

## Answer key

**1. B** — LLM (reasoning engine) + Tools (external functions) + Loop (iteration until the goal is met).

**2. B** — THOUGHT (natural language reasoning), ACTION (structured JSON tool request), OBSERVATION (tool result fed back).

**3. B** — The model only generates text/JSON describing what it wants; your code parses and executes it.

**4. B** — `None`. The model produced no prose, only a structured tool request.

**5. B** — The model is stateless, so the full history must be resent each call.

**6. B** — Descriptions are all the model sees when choosing a tool, so they are prompt engineering rather than documentation.

**7. B** — It returns `{"error": "Unknown tool ..."}` as an observation, letting the model reason about the failure and try something else.

**8. B** — Prevent infinite loops (and unbounded token spend) when the agent fails to converge.

**9. B** — A format issue; the loop nudged the model and iteration 2 produced a correct ACTION.

**10. B** — "After writing code to a file, always `run_python` to TEST it."

**11. B** — Autocomplete (2021, Copilot) → Chat assistant (2023, ChatGPT/Claude) → Agent (2025, Claude Code/Codex).

**12. D** — `deploy_service` is not listed. The four are `read_file`, `write_file`, `run_command`, `search_code`.

**13.** The model's only output is **text**. When "calling a function," it emits structured JSON such as `{"city": "Tokyo"}` alongside a tool name — a *request*, not an invocation. **Your code** parses that JSON, decides whether to honour it, executes the corresponding Python function, and inserts the result back into the conversation as a tool message. **Security implication:** the tools you register form a hard capability boundary — an agent can do exactly what its tools permit and nothing more, so the security question is never "what might the model do?" but "what did I give it the ability to do, and with what arguments?" Since the model chooses the *arguments*, every tool must validate its inputs as if they came from an untrusted source, because effectively they do.

**14.** Each API call is independent: the model retains nothing between requests, so any appearance of continuity comes from **your application resending the full transcript**. Demonstrated by prompt tokens growing 23 → 184 across two turns. **Consequences:** (i) **cost grows superlinearly** in a long conversation, since every turn pays for all prior text; (ii) **the context window is a hard ceiling** — long conversations and long agent runs must summarise, prune, or drop history; (iii) **"memory" must be engineered** as an explicit system — storage, retrieval, and selection of what to resend — which is precisely why Week 18 exists.

**15.** (i) **Real-world feedback** — tools supply data absent from the static weights. (ii) **Iterative correction** — the agent can try, observe failure, and adjust, like a developer debugging. (iii) **Persistence** — it can work autonomously for hours on multi-file tasks that would overwhelm one prompt. **Most fundamental: real-world feedback.** The weights are frozen at training time, so tools are the *only* channel through which genuinely new information — current data, the actual contents of a file, whether the test passed — enters the model's context. Correction and persistence are both downstream of it: you cannot correct what you cannot observe, and there is no point persisting without new information each round. This is also why retrieval (Weeks 9–11) is a natural extension.

**16.** ReAct is a purely **textual protocol** — THOUGHT/ACTION/ACTION_INPUT/FINAL_ANSWER are just strings the model is instructed to emit and the loop parses with regex. Nothing in the model architecture knows about tools; the prompt alone defines the contract, the available tools, the response format, and the rules ("one action per turn," "wait for the OBSERVATION," "if a tool errors, try a different approach"). **Consequence for compatibility:** the pattern needs **no special model support** — no fine-tuning, no native function-calling API — so it works with any instruction-following model. That portability is why ReAct spread so quickly, and it also means improving an agent often means improving its prompt rather than its code.

**17.** Because an exception **terminates** the loop, while an observation **informs** it. The model's entire view of the world arrives through OBSERVATION text, so an error placed there becomes something it can reason about and route around — which is exactly the "iterative correction" advantage the loop exists to provide. **Concrete example:** in the notebook, `search_wikipedia("Albert Einstein")` returned `{"error": "Page not found for 'Albert Einstein'. Try a different term."}`. Because this came back as an observation, the agent read it, reasoned that the search term needed changing, and retried with a different query. Had the tool raised, the run would have died on a recoverable problem. Note the error message itself is written to be *useful to a reader* — including the available tool list, or a suggestion to try another term — which is deliberate: error strings are part of the prompt.

**18.** **Why it happens:** the model is stateless, and the loop appends every assistant message and every observation to `messages`, resending the whole transcript on each iteration. **What it costs:** input tokens grow monotonically, so a 10-step run costs far more than 10× a single step; latency rises with prompt length (prefill, Week 4); and a long run eventually exhausts the context window and fails. **Mitigations:** (i) **truncate observations at the source** — the notebook caps Wikipedia summaries at 800 characters, file reads at 3000, and directory listings at 50 entries, so no single tool result can flood the context; (ii) **summarise or prune older history** once it passes a threshold, compressing early thought/observation pairs into a short digest while keeping the system prompt and recent turns intact. (Other valid answers: externalise results to files and re-read on demand, or use prompt caching for the stable prefix.)

**19.** A code **generator** produces code and stops; correctness is the human's problem. A code **agent** closes the loop on its own output — it writes code, executes it, reads the result, and iterates until it works. The system prompt is where this is specified: *"After writing code to a file, always `run_python` to TEST it"*, *"If a test fails, read the error, fix the code, try again"*, *"Verify your work before giving FINAL_ANSWER"*, and *"Include `print()` statements so you can see output."* Supporting this, `run_python` returns **stderr and exit_code**, not just stdout — an agent that cannot see the error message cannot debug. The value is not better code on the first attempt; it is that a wrong first attempt gets caught and repaired without a human.

**20.** **(i) Arbitrary code execution.** `run_python` passes model-generated code to `subprocess.run(["python3", "-c", code])`, so anything the model emits runs with the process's full privileges — acceptable only because this is a disposable Colab VM. *Mitigation:* execute in a real sandbox (container or gVisor/Firecracker) with no network, a read-only filesystem apart from a scratch directory, and enforced CPU/memory limits; the existing `timeout=15` is necessary but nowhere near sufficient. **(ii) `eval()` in `calculate`.** The character allowlist blocks the obvious payloads, but `eval` is the wrong primitive and allowlists are easy to get subtly wrong — an expression like `9**9**9` passes the filter and hangs the process. *Mitigation:* use `ast.literal_eval` or a dedicated expression parser that supports only arithmetic, and bound the result size. **(iii) Unrestricted file writes.** `write_file` accepts any path and calls `os.makedirs`, so the agent can write outside the intended working directory, including to paths like `~/.bashrc`. *Mitigation:* resolve the path with `os.path.realpath` and reject anything outside an allowed root, refuse symlinks, and deny overwrites of existing files unless explicitly permitted. Broadly: this code is written to teach the loop, not to be safe, and none of it should be lifted into a real system unchanged.

# Week 8 — From APIs to Agents

**Subtitle:** First Principles
**Date:** 07/03/2026
**Sources:** `downloads/week-08-from-apis-to-agents.pdf` (7 slides) · `downloads/week-08-from-apis-to-agents.ipynb` (48 cells)
**Notion page:** https://100xschool.notion.site/31cffffa33e58088a0f8df02481dad9d
**Extra link:** [Excalidraw board](https://excalidraw.com/#json=eesSkvg9N8hlYekpHbzmd,XV1VhLMWEGRyPssJ1Nx78A)

> **The pivot week.** Weeks 1–7 built a model from scratch. From here on the course *uses* models. The slides are thin; **the notebook is the lecture** — it builds a working ReAct agent in ~60 lines with no frameworks.

---

## 1. The evolution: autocomplete → chat → agent

| Year | Stage | Example | What it does |
|---|---|---|---|
| **2021** | **Autocomplete** | GitHub Copilot | Suggests the next line inside your editor. **The human drives logic and structure.** |
| **2023** | **Chat assistant** | ChatGPT / Claude | You describe a problem; the AI returns code blocks that **you manually copy-paste.** |
| **2025** | **Agent** | Claude Code / Codex | **Goal-driven autonomy.** The agent plans, executes, uses tools, and iterates until the objective is met. |

The axis is **who closes the loop.** In 2021 the human integrates every suggestion; in 2023 the human still transfers code by hand; in 2025 the agent observes results and corrects itself.

Claude Code runs in a **local terminal**: reads the codebase, edits files, runs tests, commits code.

---

## 2. The definition

> ### Agent = **LLM + Tools + Loop**

- **LLM** — the reasoning engine that understands intent and plans steps
- **Tools** — external functions the LLM can call to interact with the world
- **Loop** — the iterative process that continues until the goal is met

That's it. There is no fourth ingredient, and no framework is required.

---

## 3. Why the loop outperforms one-shot

**Iterative feedback** — the agent loop is a powerful extension of chain-of-thought reasoning. **Every iteration is a new forward pass that incorporates real-world data from tools**, letting the model observe results and adjust.

Three specific advantages:

1. **Real-world feedback** — tools provide data that **isn't present in the static model weights.** The agent gains fresh context by interacting with the environment.
2. **Iterative correction** — the agent can try a solution, **see it fail, and adjust** — like a developer debugging in real time.
3. **Persistence** — the architecture lets tools work autonomously **for hours** on complex, multi-file features that would overwhelm a single prompt.

Point 1 is the deep one. A model's weights are frozen at training time; tools are the only channel through which genuinely new information enters. Everything in Weeks 9–11 (RAG) is an elaboration of this idea.

---

## 4. The ReAct loop (Yao et al., 2022)

> A structured sequence of **text-based operations**. There is no hidden "magic" — it is a simple **"Text In → Text Out"** cycle that produces sophisticated autonomous behaviour.

| Step | Nature | Function |
|---|---|---|
| **THOUGHT** | Natural language | The model generates tokens reasoning about the goal and planning the next step |
| **ACTION** | Structured JSON | The model outputs a request for a tool call, e.g. `search_database` |
| **OBSERVATION** | Tool result | **The system** executes the tool and feeds real-world data back into context |

> **Repeat until: "I have all the information. Here is the final answer."**

**The critical asymmetry:** the model *requests*; **your code executes.** The LLM never touches the world directly — it only emits text describing what it wants. Everything an agent can do is bounded by the tools you hand it, which is also the whole story of agent security.

### Claude Code's minimalist toolset

| Tool | Purpose |
|---|---|
| `read_file(path)` | Accesses file contents for analysis |
| `write_file(path, content)` | Modifies or creates files — the primary mechanism for implementing changes |
| `run_command(cmd)` | Executes bash — run tests, check diffs, compile |
| `search_code(query)` | Navigates large codebases via grep/ripgrep |

**Four tools.** The capability comes from the loop, not from tool count.

---

## 5. Notebook Part 1 — API foundations

Uses **OpenRouter** (OpenAI-compatible: same SDK, different `base_url`), free tier, no credit card.

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

### The first call

```python
response = client.chat.completions.create(
    model=FREE_MODEL,
    messages=[{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
    temperature=0.0,
    max_tokens=50,
)
# Response: Four
# Tokens — in: 21, out: 4
```

> **This is inference.** You send tokens, get tokens back. The model you trained in Class 5 does the same thing — the API just wraps it with HTTP.

### Temperature, revisited

Same `softmax(logits / T)` from Week 7. At **T=0.0** the three sampled stories were near-identical; at **T=1.0** they diverged. Confirms in practice what Week 7 showed in theory.

### ⚠️ The model has NO memory

> **Multi-turn: the entire conversation is re-sent every time. "Chat memory" is your application re-sending history.**

Demonstrated by watching prompt tokens grow:

```
Turn 1 → prompt tokens:  23
Turn 2 → prompt tokens: 184
```

**This is one of the most important practical facts in the course.** The model is stateless. Every appearance of memory — chat history, agent context, conversation continuity — is your code resending text. It's why cost grows superlinearly in long conversations, why context windows are a hard constraint, and why Week 18 (Memory) needs to exist at all.

---

## 6. Notebook Part 2 — Function calling

> **Function calling = the model can REQUEST tools.** It outputs structured JSON saying "I want to call X with args Y." **YOUR code executes the function** and feeds the result back.
>
> **The model never "calls" anything. It generates text that says what it wants.**

### Declaring tools

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city. Returns temperature and conditions.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
            },
            "required": ["city"]
        }
    }
}, ...]
```

The **description fields are prompt engineering**, not documentation. They are the only thing the model sees when deciding whether and how to call a tool. Vague descriptions produce wrong tool choices.

### The model requests a tool

```
Finish reason: tool_calls
Content: None
Tool calls: [... function=Function(arguments='{"city": "Tokyo"}', name='get_weather') ...]

→ Model wants to call: get_weather({"city": "Tokyo"})
```

Note `content: None` and `finish_reason: tool_calls` — the model produced *no prose*, only a structured request.

### Completing the cycle

```
User → Model requests tool → OUR code runs tool → Result back to model → Final answer
```

```python
result = TOOL_FNS[tc.function.name](**json.loads(tc.function.arguments))

messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    msg,                                                    # the assistant's tool request
    {"role": "tool", "tool_call_id": tc.id, "content": result}
]
r2 = client.chat.completions.create(model=TOOL_MODEL, messages=messages, tools=tools)
# Final answer: The weather in Tokyo is currently 22°C with partly cloudy conditions.
```

The `tool_call_id` links the result back to the specific request — necessary when several tools are called in one turn.

---

## 7. Notebook Part 3 — A ReAct agent from scratch

> **No LangChain. No CrewAI. No frameworks. Just Python + an API + a loop.**

### The system prompt IS the architecture

```
When you need to use a tool, respond in EXACTLY this format:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <arguments as valid JSON>

When you have enough information for the final answer:

THOUGHT: <your final reasoning>
FINAL_ANSWER: <your complete answer to the user>

## Rules
- Always start with THOUGHT
- Use only ONE action per turn
- Wait for the OBSERVATION before continuing
- If a tool returns an error, reason about it and try a different approach
- Be concise in your thoughts
```

Since ReAct is purely textual, **the prompt defines the protocol.** No special model support is needed — this works with any instruction-following model, which is exactly why ReAct spread so fast.

### The loop — the heart of the class

```python
def run_agent(user_query, max_iterations=10, verbose=True):
    tool_desc = "\n".join(f"- {t['desc']}" for t in TOOLS.values())
    system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]

    for i in range(max_iterations):
        # 1. Ask the LLM what to do
        response = client.chat.completions.create(
            model=FREE_MODEL, messages=messages, temperature=0, max_tokens=800)
        text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # 2. Parse
        thought = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)

        if "FINAL_ANSWER:" in text:
            return {"answer": text.split("FINAL_ANSWER:")[-1].strip(), "iterations": i + 1}

        action_match = re.search(r"ACTION:\s*(\w+)", text)
        input_match  = re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:                       # format recovery
            messages.append({"role": "user",
                "content": "Please respond with either ACTION + ACTION_INPUT or FINAL_ANSWER."})
            continue

        # 3. Execute the tool
        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        if tool_name not in TOOLS:
            observation = json.dumps({"error": f"Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"})
        else:
            try:
                args = json.loads(raw_input) if raw_input.startswith("{") else {"query": raw_input.strip("\"'")}
                observation = TOOLS[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": f"Failed to call {tool_name}: {e}"})

        # 4. Feed observation back
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
```

**Four design decisions worth noticing:**

1. **`max_iterations` is a safety rail.** Without it a confused agent loops forever, burning tokens.
2. **Errors are returned as observations, not raised.** `{"error": "Unknown tool ..."}` goes back into context so the model can *reason about the failure and recover*. This is the single most important pattern in agent design — an exception kills the agent, an observation teaches it.
3. **Format recovery** — if the model doesn't produce a parseable ACTION, the loop nudges rather than crashing.
4. **Everything accumulates in `messages`** — which is exactly why the context grows (see Part 4).

### Observed behaviour

Simple arithmetic (`123*123`):
```
Iteration 1 → ⚠️ Format issue — nudging...
Iteration 2 → THOUGHT: I need to calculate 123 * 123...
              ACTION: calculate({"expression": "123 * 123"})
              OBSERVATION: {"result": 15129}
Iteration 3 → FINAL_ANSWER
```

Note **iteration 1 failed the format check** and the nudge recovered it. Real agents misformat constantly; the recovery path is not optional.

> 🔍 **Bug in the shipped notebook:** `search_wikipedia` returns `"Page not found"` for obviously valid queries — "Albert Einstein", "Romeo and Juliet", "Wakanda". The Wikipedia REST summary endpoint normally resolves these fine; the likely cause is a missing `User-Agent` header (Wikimedia rejects unidentified clients) rather than bad URL construction. **Fix:** pass headers, e.g. `requests.get(url, headers={"User-Agent": "100x-course/1.0 (contact@example.com)"}, timeout=10)`.
>
> Silver lining: the failures accidentally demonstrate the point above — the agent *read the error and retried with different search terms* instead of collapsing. That's the resilience the loop buys you.

---

## 8. Notebook Part 4 — Tracing: the context engineering problem

The traced version logs `prompt_tokens` and `completion_tokens` per iteration:

```
Iter 1:   xxx in +  xxx out
Iter 2:  xxxx in +  xxx out     ← grows
Iter 3:  xxxx in +  xxx out     ← grows more
→ prompt_tokens GROWS each iteration — the context engineering problem.
```

**Why:** every iteration re-sends the *entire* history — system prompt, every thought, every action, every observation. The model is stateless (§5), so the whole transcript must be replayed each time.

**Consequences:**
- **Cost is quadratic-ish in steps** — a 10-step agent doesn't cost 10× a 1-step agent, it costs considerably more
- **Long-running agents hit the context window** and must summarise or prune
- **Observations should be truncated** — note the notebook caps Wikipedia summaries at 800 chars and file reads at 3000

This is the problem Week 17 (Harness, Context, Evals) and Week 18 (Memory) exist to solve.

---

## 9. Notebook Part 5 — A code agent

Same loop, different tools. *"These are essentially the same tools Claude Code kinda has."*

| Tool | Implementation note |
|---|---|
| `read_file(path)` | Truncates at 3000 chars, reports `truncated: true` |
| `write_file(path, content)` | `os.makedirs(..., exist_ok=True)` first |
| `run_python(code)` | `subprocess.run` with **`timeout=15`**, returns stdout/stderr/exit_code |
| `list_files(path)` | Caps at 50 entries |
| `calculate(expression)` | Character-allowlist before `eval` |

**The system prompt is where the agentic behaviour is specified:**

```
- ONE action per turn. Wait for OBSERVATION.
- After writing code to a file, always run_python to TEST it.
- If a test fails, read the error, fix the code, try again.
- Verify your work before giving FINAL_ANSWER.
- Include print() statements so you can see output.
```

**"After writing code, always test it"** and **"if a test fails, read the error and fix it"** are what turn a code *generator* into a code *agent*. The write→run→read-error→fix cycle is the entire value proposition.

Note also `run_python` returns **stderr and exit_code**, not just stdout. An agent that can't see the error message can't debug.

### Safety observations

The sandboxing here is thin and worth naming:

- `run_python` executes **arbitrary code** via subprocess — only acceptable because it runs in a disposable Colab VM
- `calculate` uses `eval()` behind a character allowlist. The allowlist blocks the obvious attacks, but `eval` remains a poor primitive — `ast.literal_eval` or a proper expression parser is the safer choice
- `write_file` can write **anywhere** the process can reach — no path confinement

Fine for a lesson; **do not lift this into anything real** without a sandbox, path restrictions, and an execution timeout you actually trust.

---

## Key takeaways

1. **Agent = LLM + Tools + Loop.** Nothing more.
2. **The evolution is about who closes the loop** — human (autocomplete) → human (chat) → agent (2025).
3. **ReAct is Text In → Text Out**: THOUGHT (prose) → ACTION (JSON) → OBSERVATION (tool result) → repeat.
4. **The model requests; your code executes.** Tools bound what an agent can do — and are the security boundary.
5. **The loop beats one-shot** because tools bring in information absent from the weights, and failures can be observed and corrected.
6. **The model has no memory.** Every turn re-sends the whole conversation; "chat memory" is your application resending history.
7. **Tool descriptions are prompt engineering** — they're all the model has to decide with.
8. **Return errors as observations, never exceptions** — that's what lets an agent recover.
9. **Always cap iterations** to prevent infinite loops.
10. **Context grows every iteration** — the central cost and scaling problem, addressed in Weeks 17–18.
11. **A code agent is the same loop plus file I/O and execution**, with "write → test → read error → fix" in the prompt.
12. **~60 lines reproduces the pattern behind Claude Code.**

---

## Glossary

| Term | Meaning |
|---|---|
| **Agent** | LLM + tools + loop, running until a goal is met. |
| **ReAct** | Reasoning + Acting (Yao et al., 2022) — interleaved thought, action, observation. |
| **THOUGHT / ACTION / OBSERVATION** | Reasoning prose / structured tool request / tool result fed back. |
| **Function calling** | A model emitting structured JSON requesting a tool invocation. |
| **Tool** | An external function the model can request; the only channel to the outside world. |
| **Tool registry** | The dict mapping tool names to implementations and descriptions. |
| **`tool_call_id`** | Identifier linking a tool result back to its originating request. |
| **`finish_reason: tool_calls`** | Signals the model stopped in order to request a tool. |
| **System prompt** | The instruction block defining the agent's protocol and rules. |
| **Max iterations** | Hard cap preventing infinite agent loops. |
| **Format recovery** | Nudging the model when its output can't be parsed, rather than failing. |
| **Context engineering** | Managing the growing prompt across agent iterations. |
| **Stateless model** | A model retaining nothing between calls; history must be resent. |
| **OpenRouter** | OpenAI-compatible gateway to many models via one API key. |
| **Multi-hop reasoning** | Chaining several tool calls to answer one question. |

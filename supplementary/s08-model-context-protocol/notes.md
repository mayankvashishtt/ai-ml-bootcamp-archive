# S8 — Model Context Protocol: The Integration Layer

> ⚠️ **Supplementary lecture.** Not part of the 100xSchool course. The course taught function calling (Week 8), agent harnesses (Week 11), and LangGraph orchestration (Week 21) — but never addressed how tools get *distributed* between systems. Every tool in the course was defined inside the same codebase as the agent. That works for a demo and breaks at organisational scale. MCP is the standard that emerged to fix it, and it is now the dominant way real agents acquire capabilities.

**Fills the gap after:** Week 8 (Function Calling), Week 11 (Harness), Week 21 (LangGraph)
**Prerequisites:** Week 8 (tool use), Week 11 (agent loop), S4 (safety — required for §8)

---

## 0. The problem MCP solves

Week 8 taught you to define a tool: write a JSON Schema, hand it to the model, execute the function when the model calls it, feed the result back. Everything lived in one repository — your agent, your schemas, your execution code.

Now scale that to an organisation. You have four agent applications: a coding assistant, a support bot, an internal research agent, and a data-analysis tool. You have six systems they need to reach: GitHub, Postgres, Slack, Google Drive, Jira, and an internal metrics service.

That is **4 × 6 = 24 integrations**, each written separately, each with its own auth handling, error handling, schema definitions, and pagination logic. Add a fifth agent and you write six more. Add a seventh system and you write four more. The GitHub integration exists in four places, drifts in four directions, and gets fixed in one of them.

**This is the N×M problem**, and it is not new. HTTP solved it for documents, ODBC solved it for databases, and the Language Server Protocol solved it for editors — before LSP, every editor needed a plugin for every language (N×M); after LSP, every language ships one server and every editor speaks one protocol (N+M). MCP is explicitly modelled on LSP, for the same reason.

**The Model Context Protocol** is an open standard — introduced by Anthropic in November 2024 and now adopted well beyond it — that defines a uniform way for applications to expose context and capabilities to LLMs. Write **one** MCP server per system; **any** MCP-compatible client can use it. 4 + 6 = 10 pieces instead of 24, and each system's integration exists exactly once.

The frequently-quoted framing is that **MCP is a USB-C port for AI applications**: one connector shape, many devices on either side.

---

## 1. Architecture: host, client, server

Three roles, and the distinction between the first two is the part people get wrong.

**Host** — the application the user interacts with. Claude Code, the Claude desktop app, an IDE extension, your own agent. The host owns the LLM conversation, decides which servers to connect to, and enforces user permissions.

**Client** — a connector *inside* the host that maintains a **1:1 stateful session with one server**. A host running five MCP servers instantiates five clients. Clients exist so that servers are isolated from one another: one server cannot see another's traffic, and the host mediates everything.

**Server** — a program exposing capabilities over the protocol. It wraps a system (a database, an API, a filesystem) and presents its functionality in MCP's vocabulary. Servers are usually small, single-purpose, and independently deployable.

```
┌─────────────────── Host (Claude Code / your agent) ───────────────────┐
│                                                                       │
│   LLM conversation  ·  permission enforcement  ·  context assembly     │
│                                                                       │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                    │
│   │ Client A │      │ Client B │      │ Client C │                    │
│   └────┬─────┘      └────┬─────┘      └────┬─────┘                    │
└────────┼─────────────────┼─────────────────┼──────────────────────────┘
         │ stdio           │ HTTP            │ HTTP
    ┌────▼─────┐      ┌────▼─────┐      ┌────▼─────┐
    │ Postgres │      │  GitHub  │      │  Slack   │
    │  server  │      │  server  │      │  server  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
     local DB          GitHub API        Slack API
```

**The isolation property is the security story.** The host is the only component that sees everything, so it is the only place that can enforce policy. A server cannot reach into the conversation, cannot see other servers' tools, and cannot act without the host relaying a request. Whether that boundary holds in practice is §8's subject.

Underneath, MCP speaks **JSON-RPC 2.0** with a capability-negotiation handshake at session start: client and server each declare what they support, and neither assumes features the other lacks.

---

## 2. The three primitives

MCP defines three things a server can expose. The distinction between them is about **who decides to use it**, and that is the whole design.

### Tools — model-controlled

Functions the model can invoke. Each has a name, a description, and a JSON Schema for its input — **exactly the Week 8 shape**, which is deliberate: MCP tools become tool definitions in the model's request with no conceptual translation.

```json
{
  "name": "query_database",
  "description": "Run a read-only SQL query against the analytics warehouse",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql":   { "type": "string", "description": "A SELECT statement" },
      "limit": { "type": "integer", "default": 100 }
    },
    "required": ["sql"]
  }
}
```

Tools have side effects, so they are the primitive that requires approval gates. The protocol supports annotations that hint whether a tool is read-only or destructive, but **annotations are hints from the server, not enforcement** — the host must not trust them for security decisions (§8).

### Resources — application-controlled

Data the server can provide, identified by URI: `file:///logs/app.log`, `postgres://db/schema`, `git://repo/HEAD`. Resources are for **context, not action** — they are read, not executed, and they should be side-effect free.

The critical distinction: **the model does not decide to fetch a resource; the host application does.** Resources can be listed, subscribed to for change notifications, and templated with parameters. Think "files the host can attach to the conversation," not "functions the model can call."

Note that many real servers expose read-only *tools* instead of resources, precisely because tools are model-driven and therefore work without host-side UI. This is a legitimate design choice, not a mistake — but it does mean you will meet fewer resources in the wild than the spec might suggest.

### Prompts — user-controlled

Reusable templated prompts the server offers, typically surfaced in the host as slash commands or menu items. A Git server might offer `/commit-message`; a database server might offer `/explain-query`.

**The user selects these deliberately.** They are not injected automatically and are not chosen by the model.

### The control axis, summarised

| Primitive | Controlled by | Analogy | Side effects |
|---|---|---|---|
| **Tools** | The model | Function call | Expected |
| **Resources** | The host application | File attachment / GET | Should be none |
| **Prompts** | The user | Slash command / template | None |

If you remember one thing about MCP's design, remember this table. Newer revisions of the spec add further capabilities — **sampling** (a server asking the host to run an LLM completion on its behalf) and **elicitation** (a server asking the host to collect input from the user) — but the three primitives above are the core, and support for the newer ones varies by client.

---

## 3. Transports

**stdio** — the server runs as a **local subprocess**; the client writes JSON-RPC to its stdin and reads from its stdout. No network, no ports, no auth layer, minimal latency. This is the default for local tools: filesystem access, a local database, a git repository, a CLI wrapper. It is also the simplest thing to debug, because you can run the server by hand and type at it.

> ⚠️ A server using stdio transport **must never write anything but protocol messages to stdout.** A stray `print()` corrupts the JSON-RPC stream and breaks the session in a way that produces confusing errors. Log to **stderr** or to a file. This is the single most common bug when writing a first MCP server.

**Streamable HTTP** — the server runs remotely and is reached over HTTP, with Server-Sent Events used for streaming responses and server-initiated messages. This is the transport for hosted, shared, multi-user servers, and it is where authentication and authorisation actually matter. (An earlier HTTP+SSE transport has been superseded by Streamable HTTP; you may still encounter the older form in existing servers.)

The choice is straightforward: **local and single-user → stdio; hosted and shared → HTTP.** The protocol semantics are identical either way, which is the point — a server can change transport without changing its tools.

---

## 4. MCP vs. function calling — the distinction that confuses everyone

This is the most common point of confusion, so state it precisely:

**MCP does not replace function calling. MCP is a delivery mechanism for function calling.**

Week 8's mechanism is unchanged. The model still receives tool definitions, still emits a `tool_use` block, still receives a `tool_result`. The stop-reason loop is identical. What MCP changes is **where the tool definitions came from and who executes them**.

| | Week 8 function calling | MCP |
|---|---|---|
| Tool definition | Written in your agent's code | Fetched from a server at connect time |
| Execution | Your code | The server's code, in the server's process |
| Reuse across apps | Copy-paste | Connect the same server |
| Auth to the target system | Your agent handles it | The server handles it |
| Discovery | Static, compiled in | Dynamic — `tools/list` at runtime |
| Versioning | Your release cycle | The server's release cycle |

**Dynamic discovery is the genuinely new capability.** The client calls `tools/list` and learns what exists *at runtime*. A server can add a tool and every connected client sees it on the next connection without redeploying. This is also a security consideration (§8) — a tool set that can change after you approved it is a tool set you did not fully approve.

**MCP's other real contribution is operational**, and it is easy to undersell: auth, retries, pagination, rate limiting, and error mapping for a given system get written **once**, inside the server, by whoever understands that system. Every agent that connects inherits correct behaviour instead of reimplementing it badly.

---

## 5. MCP vs. LangGraph — different layers entirely

Week 21 taught LangGraph: nodes, edges, state, conditional routing. Students routinely ask which to use. The question is malformed, because they operate at different layers.

- **LangGraph orchestrates**: it defines control flow — which step runs next, what the state looks like, when to loop, when to stop, where the human approval gate sits.
- **MCP integrates**: it defines how a capability is exposed and consumed across a process boundary.

A LangGraph node can call an MCP tool. An MCP server can be consumed by a LangGraph agent, a Claude Code session, and a custom loop simultaneously. They compose; they do not compete.

The analogy that lands: **LangGraph is your application's control flow; MCP is your application's driver interface.** Asking "LangGraph or MCP?" is like asking "state machine or USB?"

---

## 6. Building a server

Minimal Python server using the official SDK's `FastMCP`:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("analytics")

@mcp.tool()
def query_database(sql: str, limit: int = 100) -> str:
    """Run a read-only SQL query against the analytics warehouse.

    Args:
        sql: A SELECT statement. Non-SELECT statements are rejected.
        limit: Maximum rows to return.
    """
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT statements are permitted."
    rows = run_query(sql, limit)
    return format_as_markdown_table(rows)

@mcp.resource("schema://tables")
def table_schema() -> str:
    """The warehouse schema, for reference."""
    return load_schema_description()

if __name__ == "__main__":
    mcp.run()   # stdio by default
```

Three things to notice, because they carry more weight than the code does:

**The docstring is the tool description, and the tool description is prompt engineering.** The model chooses tools based entirely on names, descriptions, and parameter documentation. A vague description produces a tool that is never called or called wrongly. This is S2's territory: say what the tool does, when to use it, when *not* to use it, and what it returns. Write it for a competent colleague who has never seen your system.

**The return value enters the context window.** Returning 10,000 rows is a context-management failure, not a completeness win — this is Week 10's context rot arriving through a new door. Truncate, summarise, paginate, and say explicitly when you have truncated. A tool that silently returns the first 100 of 40,000 rows will produce confidently wrong analysis.

**Errors should be returned as content the model can read, not raised as protocol errors.** `"Error: table 'user' not found. Did you mean 'users'?"` lets the model recover on the next turn. An opaque stack trace ends the run. Week 11's harness lesson applies exactly: the agent's ability to recover depends on the quality of the feedback it receives.

Servers can be written in any language with an SDK — Python, TypeScript, Java, Kotlin, C#, Go, Rust, and others — and configured in a host by declaring the command to run (stdio) or the URL to reach (HTTP).

---

## 7. Using MCP from the Claude API

Beyond desktop and CLI hosts, MCP servers can be attached directly to Messages API requests via the **MCP connector**, which lets Claude call remote MCP servers without you writing any client code.

**The shape requires two things together**, and omitting the second is a validation error rather than a silent no-op:

1. `mcp_servers` — an array of server declarations: `{ "type": "url", "url": "...", "name": "..." }`
2. `tools` — an entry `{ "type": "mcp_toolset", "mcp_server_name": "<the same name>" }`

Plus the beta header `mcp-client-2025-11-20`.

The `name` in `mcp_servers` and the `mcp_server_name` in the toolset must match — that pairing is how a declared server is actually made available to the model. Declaring the server alone does nothing.

This matters architecturally: it means **a remote MCP server can be attached to a stateless API call**, not just to a long-running desktop session. Your production agent gets the same integrations as your local tooling.

---

## 8. Security: MCP is a trust boundary

This section is not optional. An MCP server is code you did not write, running with your credentials, injecting text into your model's context, and — via dynamic discovery — able to change its own capabilities after you approved it. Treat it accordingly.

**Recall S4's lethal trifecta:** the dangerous combination is (1) access to private data, (2) exposure to untrusted content, and (3) the ability to communicate externally. **A typical MCP setup assembles all three by default.** A database server provides private data. A web-fetch or issue-tracker server ingests untrusted content. A Slack or email server provides exfiltration. Each is individually reasonable; together they are exploitable, and nothing in the protocol notices you have assembled them.

**Specific attack classes to know:**

**Prompt injection through tool results.** A tool returns text, and that text enters the context as authoritative-looking content. A GitHub issue body, a web page, a database row, a filename — any of these can contain instructions. This is Week 8's and S4's problem, delivered at scale, because MCP makes it trivial to connect many untrusted content sources at once.

**Tool poisoning.** The tool *description* is part of the prompt. A malicious server can embed instructions in a description that the user never reads — hosts typically display tool names, not full descriptions. The model reads it; the human does not. Trust the server, or read its descriptions.

**Rug pulls.** Because tools are discovered dynamically, a server that was benign when you approved it can change its tool descriptions or behaviour later. Approval at install time is not approval of all future versions. Pin versions where you can.

**Confused deputy.** The host holds the user's credentials and acts on the model's instructions. If untrusted content can influence those instructions, it can direct the host's authority — the host becomes a deputy acting on an attacker's behalf with a legitimate user's permissions. This is the underlying pattern behind most real MCP exploits.

**Cross-server interference.** Servers are isolated from each other by design, but their *outputs share one context window*. Data from a compromised server can influence how the model uses an entirely different, trustworthy server. Isolation at the connection layer is not isolation at the reasoning layer.

**Token and credential handling.** Servers hold credentials to the systems they wrap. A server compromised or malicious is a credential compromise. Prefer scoped, least-privilege tokens; treat "read-only" as something you enforce with credentials, not something you request in a description.

**Defensive practice:**

- **Install servers as you would install dependencies** — audit the source, prefer official/reference implementations, pin versions, and understand that "it's just an MCP server" is the same reasoning that produces supply-chain incidents.
- **Least privilege on credentials.** A read-only database role makes a whole category of prompt-injection outcomes impossible rather than merely unlikely. Enforcement in the credential beats enforcement in the prompt every time.
- **Human approval on consequential actions**, and design the approval so it is meaningful: show the actual arguments, not just the tool name. An approval dialog nobody reads is an audit trail, not a control.
- **Deliberately break the trifecta.** If a session reads untrusted content, do not also give it exfiltration channels. Separate agents, separate sessions, separate credentials.
- **Validate server-supplied schemas and outputs**; do not assume a server is well-behaved just because it speaks the protocol correctly.
- **Log every tool call with arguments and results.** This is Week 23's observability requirement, and MCP raises the stakes because the tools are external.

---

## 9. When to use MCP — and when not to

**Use MCP when:**
- Multiple applications need the same integration.
- The integration will be maintained by a different team than the agent.
- You want capability without redeploying the agent.
- An official server already exists for the system — connecting beats writing.
- You are building something to be consumed by hosts you do not control.

**Do not use MCP when:**
- You have one agent and a handful of tools that are genuinely specific to it. Week 8's approach is less machinery and easier to debug. **A protocol boundary you do not need is pure overhead.**
- The tool is trivially coupled to your agent's internal state — MCP's process boundary makes shared in-memory state awkward by design.
- Latency is critical and you cannot afford a process hop.
- The security review is not worth it for a one-off script.

**The honest summary:** MCP earns its keep at the point where the same integration would otherwise be written twice. Before that point, it is an abstraction tax. This is the same judgement Week 21 asked for about LangGraph — reach for the framework when the complexity is real, not in anticipation of complexity that may never arrive.

---

## How this connects to the course

| Course topic | Connection |
|---|---|
| **Week 8** (function calling) | MCP tools *are* Week 8 tools. The protocol changes distribution and execution, not the model-side mechanism. |
| **Week 10** (context rot) | Tool results enter the context window. Verbose servers cause context rot through the integration layer. |
| **Week 11** (harness) | An MCP server is part of the harness. Description quality and error message quality determine agent success. |
| **Week 12** (memory) | Memory can be an MCP server, making it shareable across agents rather than trapped in one application. |
| **Week 21** (LangGraph) | Orthogonal layers: LangGraph is control flow, MCP is the driver interface. A node calls an MCP tool. |
| **Week 23** (observability) | External tool calls must be traced. Latency, errors, and arguments all cross a process boundary now. |
| **Week 24** (computer use) | Computer-use environments are increasingly exposed through MCP servers rather than bespoke harness code. |
| **S2** (prompt engineering) | Tool descriptions are prompts. Writing them well is the single largest lever on whether a server is usable. |
| **S4** (safety) | MCP assembles the lethal trifecta by default. Tool poisoning, rug pulls, and confused-deputy attacks all live here. |
| **S5** (inference cost) | Every tool definition occupies context on every request. Twenty connected servers is a standing token cost. |

---

## Key takeaways

1. **MCP solves the N×M integration problem** — N agents × M systems becomes N + M, exactly as LSP did for editors and languages.
2. **Three roles:** host (owns the conversation and enforces policy), client (one stateful session per server), server (wraps one system). Isolation between servers is what makes host-side policy enforcement possible.
3. **Three primitives, distinguished by who controls them:** tools (model), resources (application), prompts (user).
4. **stdio for local, Streamable HTTP for remote.** Never write non-protocol output to stdout on stdio — it is the classic first bug.
5. **MCP does not replace function calling; it distributes it.** The model-side mechanism from Week 8 is unchanged.
6. **Dynamic discovery is the new capability** — and simultaneously a security consideration, because the tool set can change after approval.
7. **The API connector needs both `mcp_servers` and a matching `mcp_toolset` entry in `tools`**; declaring the server alone is a validation error.
8. **Tool descriptions are prompt engineering** and tool outputs are context. Both determine whether the agent succeeds.
9. **An MCP server is a trust boundary** — untrusted code, your credentials, your context window. Least-privilege credentials, pinned versions, meaningful approvals, and deliberately broken trifectas.
10. **Do not adopt MCP for a single agent with a few tools.** It pays off at the second consumer, not the first.

---

## Glossary

| Term | Definition |
|---|---|
| **MCP** | Model Context Protocol — open standard for exposing context and capabilities to LLM applications |
| **N×M problem** | Every application needing a bespoke integration with every system |
| **Host** | The application owning the LLM conversation and enforcing permissions |
| **Client** | A connector inside the host maintaining a 1:1 session with one server |
| **Server** | A program exposing capabilities over MCP, usually wrapping one system |
| **Tool** | Model-controlled function with a JSON Schema; may have side effects |
| **Resource** | Application-controlled data identified by URI; read-only context |
| **Prompt** | User-controlled reusable template, typically surfaced as a slash command |
| **stdio transport** | Server runs as a local subprocess communicating over stdin/stdout |
| **Streamable HTTP** | Remote transport using HTTP with SSE for streaming and server-initiated messages |
| **JSON-RPC 2.0** | The wire protocol MCP messages are encoded in |
| **Dynamic discovery** | Learning a server's tools at runtime via `tools/list` |
| **MCP connector** | Messages API feature attaching remote MCP servers to an API request |
| **Tool poisoning** | Hiding instructions in a tool description that the model reads and the user does not |
| **Rug pull** | A server changing its tools or behaviour after being approved |
| **Confused deputy** | Untrusted input directing a privileged component's legitimate authority |
| **Lethal trifecta** | Private data + untrusted content + external communication (S4) |

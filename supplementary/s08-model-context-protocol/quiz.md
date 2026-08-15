# S8 — Quiz (20 questions)

**Topic:** Model Context Protocol
**Format:** Q1–12 multiple choice · Q13–20 short answer
**Answer key at the bottom.**

---

## Multiple choice

**1.** With 4 agent applications and 6 backend systems, MCP reduces the integration count from:
- A) 24 to 10
- B) 24 to 6
- C) 10 to 4
- D) 24 to 1

**2.** MCP is explicitly modelled on:
- A) GraphQL
- B) The Language Server Protocol (LSP)
- C) gRPC
- D) OpenAPI

**3.** The relationship between host and client is:
- A) They are the same thing
- B) A host contains one client per connected server
- C) A client contains multiple hosts
- D) Clients run inside servers

**4.** Resources are controlled by:
- A) The model
- B) The host application
- C) The user
- D) The server

**5.** Prompts (the MCP primitive) are controlled by:
- A) The model
- B) The host application
- C) The user
- D) The LLM provider

**6.** The classic first bug when writing a stdio MCP server is:
- A) Forgetting the JSON Schema
- B) Writing log output to stdout, corrupting the JSON-RPC stream
- C) Using the wrong port
- D) Missing the beta header

**7.** MCP's relationship to function calling is best described as:
- A) A replacement for it
- B) A delivery mechanism for it — the model-side mechanism is unchanged
- C) An alternative that avoids JSON Schema
- D) A wrapper that removes the need for tool results

**8.** Compared with Week 21's LangGraph, MCP:
- A) Competes directly — pick one
- B) Operates at a different layer: LangGraph is control flow, MCP is the driver interface
- C) Replaces LangGraph's state management
- D) Is a LangGraph feature

**9.** The Messages API MCP connector requires:
- A) Only `mcp_servers`
- B) Both `mcp_servers` and a `mcp_toolset` entry in `tools` with a matching server name
- C) Only a `mcp_toolset` tool entry
- D) A local stdio subprocess

**10.** "Tool poisoning" refers to:
- A) Sending malformed JSON to a server
- B) Hiding instructions in a tool description that the model reads but the user never sees
- C) Overloading a server with requests
- D) Returning too many rows

**11.** A "rug pull" in the MCP context is:
- A) Uninstalling a server mid-session
- B) A server changing its tools or behaviour after you approved it, enabled by dynamic discovery
- C) A network timeout
- D) Revoking a credential

**12.** MCP is generally *not* worth adopting when:
- A) Multiple applications need the same integration
- B) You have one agent with a few tools specific to it
- C) An official server already exists
- D) A different team maintains the integration

---

## Short answer

**13.** Explain the N×M problem and how MCP's architecture reduces it, using the LSP analogy.

**14.** Describe the host/client/server roles and explain why the client-per-server design matters for security.

**15.** Explain the three primitives and the control axis that distinguishes them.

**16.** Explain the difference between stdio and Streamable HTTP transports and when each is right.

**17.** Explain precisely what MCP changes relative to Week 8 function calling, and what it does *not* change.

**18.** You're writing an MCP server wrapping a data warehouse. Name three design decisions that matter more than the code, and explain each.

**19.** Explain why an MCP setup typically assembles S4's lethal trifecta, and give concrete defences.

**20.** Your company has 5 agent applications and wants to give them all access to an internal customer-data service. Design the approach end to end, including whether MCP is even the right call.

---
---

## Answer key

**1. A** — 4×6 = 24 bespoke integrations becomes 4 clients + 6 servers = 10 pieces.

**2. B** — LSP solved the same shape of problem for editors and languages.

**3. B** — The host instantiates one client per server; five servers means five clients.

**4. B** — Resources are application-controlled; the model does not decide to fetch them.

**5. C** — Prompts are user-selected, typically surfaced as slash commands.

**6. B** — On stdio, stdout carries the protocol; log to stderr or a file.

**7. B** — Tool definitions, `tool_use` blocks, and `tool_result` blocks are all unchanged; MCP changes where definitions come from and who executes them.

**8. B** — They compose: a LangGraph node can call an MCP tool.

**9. B** — `mcp_servers` declares the server; a matching `mcp_toolset` entry in `tools` makes it available. Omitting the toolset is a validation error.

**10. B** — Hosts typically display tool names, not full descriptions, so the model reads text the human never sees.

**11. B** — Approval at install time is not approval of every future version.

**12. B** — A protocol boundary you don't need is pure overhead; Week 8's approach is less machinery and easier to debug.

**13.** With N agent applications and M backend systems, each application needs its own integration with each system: **N×M bespoke integrations**. Four agents and six systems is 24, each with its own auth handling, error handling, schema definitions, and pagination logic. The GitHub integration exists in four places, drifts in four directions, and gets fixed in one of them. Adding a fifth agent costs six more integrations; adding a seventh system costs four more — the cost grows multiplicatively in exactly the dimension you expect to grow. **The fix in every prior instance of this problem has been a protocol.** The **Language Server Protocol** is the closest analogy: before LSP, every editor needed a plugin for every language (N×M); after LSP, every language ships **one** language server and every editor implements **one** client, so the cost becomes **N+M**. MCP applies this to LLM integrations: write one MCP server per system, and any MCP-compatible client can use it. Four plus six equals ten pieces instead of 24, and each system's integration exists exactly once — maintained by whoever actually understands that system, with auth, retries, pagination, and rate limiting written correctly once instead of badly four times. The usual shorthand is that MCP is **a USB-C port for AI applications**: one connector shape, many devices on either side.

**14.** The **host** is the application the user interacts with — Claude Code, the desktop app, an IDE extension, your own agent. It owns the LLM conversation, decides which servers to connect to, assembles context, and enforces user permissions. The **client** is a connector *inside* the host maintaining a **1:1 stateful session with exactly one server**; a host running five servers instantiates five clients. The **server** is a program exposing capabilities over the protocol, usually wrapping a single system, and is typically small, single-purpose, and independently deployable. **The client-per-server design is the security architecture.** Because each server talks to its own dedicated client and never to another server, a server cannot see another server's traffic, cannot enumerate what other capabilities exist, and cannot act without the host relaying a request. That makes the **host the only component with a complete view**, which in turn makes it the only place policy *can* be enforced — permission prompts, approval gates, logging, and credential scoping all live there because only there is the whole picture available. The important caveat is that this isolation holds at the *connection* layer but not at the *reasoning* layer: all servers' outputs land in one shared context window, so a compromised server's data can still influence how the model uses a different, trustworthy server.

**15.** **Tools are model-controlled** — functions the model can invoke, each with a name, description, and JSON Schema for its input, exactly matching Week 8's shape so they translate into tool definitions with no conceptual gap. Tools have side effects and are therefore the primitive needing approval gates. The protocol supports annotations hinting whether a tool is read-only or destructive, but these are **hints from the server, not enforcement**, and must not be trusted for security decisions. **Resources are application-controlled** — data identified by URI (`file:///logs/app.log`, `postgres://db/schema`), meant for **context, not action**: read rather than executed, and side-effect free. The key point is that the *model* does not decide to fetch a resource; the *host* does. They can be listed, subscribed to for change notifications, and templated. **Prompts are user-controlled** — reusable templates the server offers, surfaced as slash commands or menu items (`/commit-message`, `/explain-query`), selected deliberately by a human rather than injected automatically or chosen by the model. **The control axis is the whole design**: model / application / user, mapping to function call / file attachment / slash command. Newer spec revisions add sampling (a server asking the host to run a completion) and elicitation (a server asking the host for user input), but the three primitives above are the core, and client support for the newer ones varies.

**16.** **stdio** runs the server as a **local subprocess**: the client writes JSON-RPC to its stdin and reads responses from its stdout. There is no network, no port, no auth layer, and minimal latency, and you can debug it by running the server by hand and typing at it. It is the default for local, single-user tools — filesystem access, a local database, a git repository, a CLI wrapper. **The one rule is absolute: nothing but protocol messages may go to stdout.** A stray `print()` corrupts the JSON-RPC stream and produces confusing session failures; logging belongs on stderr or in a file. **Streamable HTTP** runs the server remotely, reached over HTTP with Server-Sent Events for streaming responses and server-initiated messages. It is the transport for hosted, shared, multi-user servers, and it is where authentication and authorisation genuinely matter, since the server is now a network-exposed service holding credentials for the system it wraps. (An earlier HTTP+SSE transport has been superseded by Streamable HTTP, though you may still meet it in existing servers.) **The decision rule is simple — local and single-user → stdio; hosted and shared → HTTP** — and the important property is that **protocol semantics are identical either way**, so a server can change transport without changing a single tool definition.

**17.** **What is unchanged is the entire model-side mechanism.** The model still receives a list of tool definitions with JSON Schemas, still emits a `tool_use` block when it wants to call one, still receives a `tool_result` block back, and the harness still loops on stop reason until the model stops requesting tools. Nothing about Week 8's mechanics differs, and this is deliberate — MCP tool definitions map onto request tool definitions with no conceptual translation. **What changes is everything around it.** Tool definitions are **fetched from a server at connect time** rather than written in the agent's code. Execution happens in the **server's process** rather than yours. Reuse across applications is **connecting the same server** rather than copy-paste. **Auth to the target system** is the server's responsibility, not the agent's. Versioning follows the **server's** release cycle rather than the agent's. And discovery becomes **dynamic**: the client calls `tools/list` at runtime, so a server can add a tool and every connected client sees it on the next connection without any agent being redeployed. **Dynamic discovery is the genuinely new capability** — and it cuts both ways, since a tool set that can change after you approved it is a tool set you did not fully approve. **MCP's other real contribution is operational rather than conceptual**: retries, pagination, rate limiting, and error mapping for a given system get written once, by whoever understands that system, and every consumer inherits correct behaviour instead of reimplementing it badly.

**18.** **First, the tool descriptions, because they are prompt engineering, not documentation.** The model selects tools based entirely on names, descriptions, and parameter documentation — it cannot read the implementation. A vague description produces a tool that is either never called or called wrongly, and neither failure is visible in the server's logs. Following S2, state what the tool does, **when to use it, when not to use it**, and what it returns, written for a competent colleague who has never seen the warehouse. For a SQL tool that means naming the dialect, stating that only SELECT is permitted, and pointing at the schema resource. **Second, output size, because return values enter the context window.** Returning 10,000 rows is a context-management failure rather than a completeness win — Week 10's context rot arriving through the integration layer, and S5's cost problem arriving with it. Truncate, paginate, or summarise; return a markdown table rather than raw JSON where it is going to be read rather than parsed; and **state explicitly when output was truncated**, because a tool that silently returns the first 100 of 40,000 rows will produce confidently wrong analysis that nothing downstream can detect. **Third, error handling, because errors are the agent's feedback channel.** Return errors as **content the model can read** — `"Error: table 'user' not found. Did you mean 'users'?"` — rather than raising opaque protocol errors or stack traces. The first lets the model recover on the next turn; the second ends the run. This is Week 11's harness lesson exactly: an agent's ability to recover is bounded by the quality of the feedback it gets. (A close fourth: enforce read-only in the **database credential**, not just in the description — see Q19.)

**19.** **S4's lethal trifecta** is the combination of (1) access to private data, (2) exposure to untrusted content, and (3) the ability to communicate externally — each individually reasonable, jointly exploitable. **A typical MCP setup assembles all three by default and nothing in the protocol notices.** A Postgres or internal-API server supplies private data. A web-fetch, email, or issue-tracker server ingests untrusted content — an attacker-authored GitHub issue body or web page enters the context looking exactly like legitimate tool output. A Slack, email, or HTTP server supplies the exfiltration channel. The user connected three useful servers; the combination is a data-exfiltration pipeline triggerable by anyone who can write text the agent will read. **The concrete attack classes** are prompt injection through tool results, **tool poisoning** (instructions hidden in a description the model reads and the user never sees, since hosts display names rather than full descriptions), **rug pulls** (a server changing behaviour after approval, enabled by dynamic discovery), **confused deputy** (untrusted content directing the host's legitimate credentials — the pattern underlying most real exploits), and **cross-server interference**, since connection-layer isolation does not isolate the shared context window. **Defences:** install servers as you would dependencies — audit the source, prefer official implementations, **pin versions**. Apply **least privilege on credentials**, because a read-only database role makes an entire category of outcomes impossible rather than merely unlikely, and enforcement in the credential beats enforcement in the prompt every time. Require **human approval on consequential actions** and show the **actual arguments**, since an approval dialog nobody reads is an audit trail rather than a control. **Deliberately break the trifecta** by separating sessions, agents, and credentials so a session that reads untrusted content has no exfiltration path. Validate server-supplied schemas and outputs. And **log every tool call with arguments and results** (Week 23), because the tools are now external.

**20.** **First, confirm MCP is the right call — and here it is.** Five consumers of one system is squarely the N×M case: without a protocol you write five integrations, each with its own auth, pagination, and error handling, and each drifting independently. MCP makes it **one server plus five clients**, with the customer-data team owning the server. The rule of thumb is that MCP earns its keep at the point where the same integration would otherwise be written twice; five is well past that. It would be the wrong call for a single agent with two bespoke tools, where Week 8's approach is less machinery and easier to debug. **Transport: Streamable HTTP, not stdio.** Five applications, plausibly multiple environments and machines, means a hosted shared service — stdio would require the server binary and its credentials to be distributed to every consumer, which is the deployment problem you are trying to eliminate. **Design the surface deliberately.** Expose **narrow, purpose-built tools** — `lookup_customer_by_id`, `search_customers`, `get_subscription_status` — rather than a generic `run_sql`, because narrow tools are easier for the model to select correctly, easier to authorise, easier to rate-limit, and dramatically easier to audit. Expose the **schema and field definitions as a resource** so the model has reference context without spending a tool call. Write descriptions as prompts (S2): what each returns, when to use it, when not to. **Bound every output** and paginate, since customer records are exactly the shape of data that quietly floods a context window (Week 10, S5). Return errors as readable content so agents recover (Week 11). **Security is the substantial part of this design, because customer data is private data.** Give the server a **least-privilege, read-only** credential to the underlying store, and enforce it there rather than in a description. Authenticate and authorise **per calling application** — five consumers should not share one token, and a support bot should not have the research agent's reach; scope tools per client where the deployment supports it. Consider whether **PII should be redacted or tokenised at the server boundary** so it never enters a context window at all — the strongest control available here, and one only the server can apply. **Then check the trifecta explicitly:** this server is now a private-data source, so audit what *else* each of the five agents connects to. Any agent combining it with an untrusted-content source and an outbound channel is an exfiltration path, and the fix is to separate those capabilities across sessions or agents rather than to write a better system prompt. **Operationally:** version the server and pin versions in consumers so a change is a deliberate upgrade rather than a rug pull; require human approval with visible arguments for any write path; **log every call with caller identity, arguments, and results** (Week 23), which is also what makes the inevitable privacy audit answerable; and build an eval set of real queries (S3) so a server change can be measured rather than guessed at.
